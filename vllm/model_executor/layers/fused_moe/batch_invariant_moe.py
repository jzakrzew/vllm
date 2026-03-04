# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant NVFP4 fused MoE expert implementation."""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    _compute_pid,
    _unswizzle_scale,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers used by ``fused_moe_batch_invariant_nvfp4``
# ---------------------------------------------------------------------------


def _nvfp4_get_expert_scalar(
    tensor: torch.Tensor | None,
    expert_id: int,
    *,
    field_name: str,
) -> torch.Tensor:
    if tensor is None:
        raise RuntimeError(f"Missing required NVFP4 MoE tensor: {field_name}")

    if tensor.ndim == 0 or tensor.numel() == 1:
        value = tensor.reshape(())
    else:
        if expert_id >= tensor.shape[0]:
            raise RuntimeError(
                f"NVFP4 MoE tensor '{field_name}' is missing expert {expert_id}."
            )
        value = tensor[expert_id]
        if value.ndim != 0:
            value = value.reshape(())
    return value.to(dtype=torch.float32)


def _nvfp4_moe_map_experts(
    topk_ids: torch.Tensor, expert_map: torch.Tensor
) -> torch.Tensor:
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    if expert_map.numel() == 0:
        return torch.full_like(topk_ids, -1, dtype=torch.long)
    valid = (flat_ids >= 0) & (flat_ids < expert_map.numel())
    clamped = flat_ids.clamp(min=0, max=max(0, expert_map.numel() - 1))
    remapped = expert_map.to(torch.long).index_select(0, clamped)
    mapped = torch.where(valid, remapped, torch.full_like(remapped, -1))
    return mapped.view_as(topk_ids)


# ---------------------------------------------------------------------------
# Grouped NVFP4 GEMM kernel and wrapper
# ---------------------------------------------------------------------------


@triton.jit
def _grouped_matmul_nvfp4_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    alpha_ptr,
    M,
    N,
    K,
    stride_ea,
    stride_eas,
    stride_eb,
    stride_ebs,
    stride_ec,
    stride_am,
    stride_ak,
    stride_ask,
    stride_bm,
    stride_bk,
    stride_bsk,
    stride_cm,
    stride_cn,
    a_scale_cols_total,
    b_scale_cols_total,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
):
    """Grouped NVFP4 GEMM: E independent (M,K)x(K,N) multiplies in one launch.

    Grid: (E, tiles_per_expert) where tiles_per_expert = cdiv(M,BM)*cdiv(N,BN).
    Each CTA handles one (expert, tile_m, tile_n) combination.
    """
    expert_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    K_BYTES: tl.constexpr = BLOCK_SIZE_K // 2
    SCALE_K_TILE: tl.constexpr = BLOCK_SIZE_K // 16
    SCALE_K_TILES: tl.constexpr = SCALE_K_TILE // 4
    SCALE_M_TILES: tl.constexpr = BLOCK_SIZE_M // 128
    SCALE_N_TILES: tl.constexpr = BLOCK_SIZE_N // 128

    pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, 0)
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    k_bytes = K // 2

    a_expert_ptr = a_ptr + expert_id * stride_ea
    a_scale_expert_ptr = a_scale_ptr + expert_id * stride_eas
    b_expert_ptr = b_ptr + expert_id * stride_eb
    b_scale_expert_ptr = b_scale_ptr + expert_id * stride_ebs
    c_expert_ptr = c_ptr + expert_id * stride_ec

    a_desc = tl.make_tensor_descriptor(
        a_expert_ptr,
        shape=[M, k_bytes],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, K_BYTES],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_expert_ptr,
        shape=[N, k_bytes],
        strides=[stride_bm, stride_bk],
        block_shape=[BLOCK_SIZE_N, K_BYTES],
        padding_option="zero",
    )
    a_scale_desc = tl.make_tensor_descriptor(
        a_scale_expert_ptr,
        shape=[1, tl.cdiv(M, 128), a_scale_cols_total // 4, 2, 256],
        strides=[
            tl.cdiv(M, 128) * (a_scale_cols_total // 4) * 512 * stride_ask,
            (a_scale_cols_total // 4) * 512 * stride_ask,
            512 * stride_ask,
            256 * stride_ask,
            stride_ask,
        ],
        block_shape=[1, SCALE_M_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )
    b_scale_desc = tl.make_tensor_descriptor(
        b_scale_expert_ptr,
        shape=[1, tl.cdiv(N, 128), b_scale_cols_total // 4, 2, 256],
        strides=[
            tl.cdiv(N, 128) * (b_scale_cols_total // 4) * 512 * stride_bsk,
            (b_scale_cols_total // 4) * 512 * stride_bsk,
            512 * stride_bsk,
            256 * stride_bsk,
            stride_bsk,
        ],
        block_shape=[1, SCALE_N_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        c_expert_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        padding_option="zero",
    )

    alpha = tl.load(alpha_ptr + expert_id).to(tl.float32)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in range(k_tiles):
        k_start_bytes = ki * K_BYTES
        if A_LARGE or B_LARGE:
            k_start_bytes = k_start_bytes.to(tl.int64)

        a_m_start = start_m
        b_n_start = start_n
        if A_LARGE:
            a_m_start = a_m_start.to(tl.int64)
        if B_LARGE:
            b_n_start = b_n_start.to(tl.int64)

        a = a_desc.load([a_m_start, k_start_bytes])
        b = b_desc.load([b_n_start, k_start_bytes]).T

        scale_tile_k = ki * SCALE_K_TILES
        if A_LARGE or B_LARGE:
            scale_tile_k = scale_tile_k.to(tl.int64)
        scale_tile_m = start_m // 128
        scale_tile_n = start_n // 128
        if A_LARGE:
            scale_tile_m = scale_tile_m.to(tl.int64)
        if B_LARGE:
            scale_tile_n = scale_tile_n.to(tl.int64)

        a_scale_raw = a_scale_desc.load([0, scale_tile_m, scale_tile_k, 0, 0])
        b_scale_raw = b_scale_desc.load([0, scale_tile_n, scale_tile_k, 0, 0])

        a_scale = _unswizzle_scale(
            a_scale_raw.reshape(BLOCK_SIZE_M, SCALE_K_TILE),
            TILE_ROWS=BLOCK_SIZE_M,
            TILE_SCALE_COLS=SCALE_K_TILE,
        )
        b_scale = _unswizzle_scale(
            b_scale_raw.reshape(BLOCK_SIZE_N, SCALE_K_TILE),
            TILE_ROWS=BLOCK_SIZE_N,
            TILE_SCALE_COLS=SCALE_K_TILE,
        )

        accumulator = tl.dot_scaled(
            a,
            a_scale,
            "e2m1",
            b,
            b_scale,
            "e2m1",
            accumulator,
        )

    accumulator *= alpha
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_desc.store([start_m, start_n], c)


def _grouped_matmul_nvfp4(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Grouped NVFP4 GEMM: ``C[e] = A[e] @ B[e].T * alpha[e]`` for all e.

    Args:
        a_fp4:   [E, M, K_packed] uint8 packed FP4 inputs.
        b_fp4:   [E, N, K_packed] uint8 packed FP4 weights.
        a_scale: [E, M_pad, K_s]  float8_e4m3fn block scales (swizzled).
        b_scale: [E, N_pad, K_s]  float8_e4m3fn block scales (swizzled).
        alpha:   [E]              float32 per-expert alpha.
        output_dtype:             dtype for the output tensor.

    Returns:
        [E, M, N] tensor of the given output dtype.
    """
    assert a_fp4.ndim == 3 and b_fp4.ndim == 3, (
        "Expected 3-D [E, rows, K_packed] tensors."
    )
    assert a_fp4.dtype == torch.uint8 and b_fp4.dtype == torch.uint8
    assert a_scale.dtype == torch.float8_e4m3fn and b_scale.dtype == torch.float8_e4m3fn
    assert alpha.dtype == torch.float32

    E = a_fp4.shape[0]
    M = a_fp4.shape[1]
    N = b_fp4.shape[1]
    K = a_fp4.shape[2] * 2
    assert b_fp4.shape[0] == E and b_fp4.shape[2] == a_fp4.shape[2]

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 256

    c = torch.empty((E, M, N), device=a_fp4.device, dtype=output_dtype)

    tiles_per_expert = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    grid = (E, tiles_per_expert)

    _grouped_matmul_nvfp4_kernel[grid](
        a_fp4,
        a_scale,
        b_fp4,
        b_scale,
        c,
        alpha,
        M,
        N,
        K,
        a_fp4.stride(0),
        a_scale.stride(0),
        b_fp4.stride(0),
        b_scale.stride(0),
        c.stride(0),
        a_fp4.stride(1),
        a_fp4.stride(2),
        a_scale.stride(2),
        b_fp4.stride(1),
        b_fp4.stride(2),
        b_scale.stride(2),
        c.stride(1),
        c.stride(2),
        a_scale.shape[2],
        b_scale.shape[2],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        A_LARGE=a_fp4.numel() > 2**31,
        B_LARGE=b_fp4.numel() > 2**31,
        num_stages=2,
        num_warps=8,
    )
    return c


# ---------------------------------------------------------------------------
# Standalone deterministic NVFP4 MoE implementation
# ---------------------------------------------------------------------------


def fused_moe_batch_invariant_nvfp4(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    a1_gscale: torch.Tensor | None,
    g1_alphas: torch.Tensor | None,
    a2_gscale: torch.Tensor | None,
    g2_alphas: torch.Tensor | None,
    activation: Any,
    *,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """
    Deterministic NVFP4 MoE built from a grouped batch-invariant NVFP4 GEMM.

    Instead of launching 2*E separate Triton kernels (one per expert per GEMM),
    this implementation collects all expert inputs, quantises them, and
    dispatches a single ``grouped_matmul_nvfp4`` call for W1 and another for
    W2, reducing kernel-launch overhead from O(E) to O(1).
    """
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.fused_moe.activation import (
        apply_moe_activation,
    )

    if hidden_states.ndim != 2:
        raise RuntimeError(
            f"Expected 2D hidden_states for NVFP4 MoE fallback, "
            f"got {hidden_states.shape}."
        )
    if topk_ids.shape != topk_weights.shape:
        raise RuntimeError(
            "NVFP4 MoE fallback expects topk_ids and topk_weights "
            "to have identical shapes."
        )
    if topk_ids.ndim != 2:
        raise RuntimeError(
            f"Expected 2D top-k routing tensors, got shape {topk_ids.shape}."
        )
    if apply_router_weight_on_input and topk_ids.shape[1] != 1:
        raise RuntimeError(
            "apply_router_weight_on_input=True is only supported for top_k == 1."
        )

    activation_kind = (
        activation
        if isinstance(activation, MoEActivation)
        else MoEActivation.from_str(str(activation))
    )

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]

    routed_topk_ids = topk_ids
    if expert_map is not None:
        routed_topk_ids = _nvfp4_moe_map_experts(topk_ids, expert_map)
    routed_topk_ids = routed_topk_ids.to(torch.long)
    routed_topk_weights = topk_weights.to(torch.float32)

    expert_ids = routed_topk_ids.reshape(-1)
    assignment_weights = routed_topk_weights.reshape(-1)
    M_total = num_tokens * top_k
    repeated_hidden_states = (
        hidden_states.unsqueeze(1)
        .expand(num_tokens, top_k, hidden_dim)
        .reshape(M_total, hidden_dim)
        .contiguous()
    )
    if apply_router_weight_on_input:
        repeated_hidden_states = repeated_hidden_states * assignment_weights.unsqueeze(
            -1
        ).to(repeated_hidden_states.dtype)
    output_scale = (
        torch.ones_like(assignment_weights)
        if apply_router_weight_on_input
        else assignment_weights
    )

    device = hidden_states.device
    dtype = hidden_states.dtype
    w1_padding_cols = max(0, w13_weight.shape[-1] - hidden_dim // 2)
    w2_output_size = hidden_dim
    w1_output_size = w13_weight.shape[1]

    # ------------------------------------------------------------------
    # Phase 1: mask + quantise inputs for every expert  (GEMM-1)
    # ------------------------------------------------------------------
    all_a1_fp4: list[torch.Tensor] = []
    all_a1_scale: list[torch.Tensor] = []
    for expert_id in range(num_experts):
        expert_mask = (expert_ids == expert_id).unsqueeze(-1)
        expert_inputs = repeated_hidden_states * expert_mask.to(dtype)

        x_fp4, x_scale = scaled_fp4_quant(
            expert_inputs,
            _nvfp4_get_expert_scalar(a1_gscale, expert_id, field_name="a1_gscale"),
            is_sf_swizzled_layout=True,
            backend=quant_backend,
        )
        if w1_padding_cols > 0:
            x_fp4 = torch.nn.functional.pad(x_fp4, (0, w1_padding_cols)).contiguous()
        all_a1_fp4.append(x_fp4)
        all_a1_scale.append(x_scale)

    batched_a1_fp4 = torch.stack(all_a1_fp4)
    batched_a1_scale = torch.stack(all_a1_scale)
    del all_a1_fp4, all_a1_scale

    g1_alpha_vec = torch.stack(
        [
            _nvfp4_get_expert_scalar(g1_alphas, e, field_name="g1_alphas")
            for e in range(num_experts)
        ]
    )

    # ------------------------------------------------------------------
    # Phase 2: grouped GEMM-1   (all experts in one kernel launch)
    # ------------------------------------------------------------------
    gemm1_out = _grouped_matmul_nvfp4(
        a_fp4=batched_a1_fp4,
        b_fp4=w13_weight,
        a_scale=batched_a1_scale,
        b_scale=w13_weight_scale,
        alpha=g1_alpha_vec,
        output_dtype=dtype,
    )
    del batched_a1_fp4, batched_a1_scale

    if gemm1_out.shape[-1] != w1_output_size:
        gemm1_out = gemm1_out[..., :w1_output_size].contiguous()

    # ------------------------------------------------------------------
    # Phase 3: activation   (batched across all experts)
    # ------------------------------------------------------------------
    activation_out_dim = (
        w1_output_size // 2 if activation_kind.is_gated else w1_output_size
    )
    activation_out = torch.empty(
        (num_experts, M_total, activation_out_dim),
        device=device,
        dtype=dtype,
    )
    for expert_id in range(num_experts):
        apply_moe_activation(
            activation=activation_kind,
            output=activation_out[expert_id],
            input=gemm1_out[expert_id],
        )
    del gemm1_out

    # ------------------------------------------------------------------
    # Phase 4: quantise intermediates for every expert  (GEMM-2)
    # ------------------------------------------------------------------
    w2_padding_cols = max(0, w2_weight.shape[-1] - activation_out_dim // 2)
    all_a2_fp4: list[torch.Tensor] = []
    all_a2_scale: list[torch.Tensor] = []
    for expert_id in range(num_experts):
        int_fp4, int_scale = scaled_fp4_quant(
            activation_out[expert_id],
            _nvfp4_get_expert_scalar(a2_gscale, expert_id, field_name="a2_gscale"),
            is_sf_swizzled_layout=True,
            backend=quant_backend,
        )
        if w2_padding_cols > 0:
            int_fp4 = torch.nn.functional.pad(
                int_fp4, (0, w2_padding_cols)
            ).contiguous()
        all_a2_fp4.append(int_fp4)
        all_a2_scale.append(int_scale)
    del activation_out

    batched_a2_fp4 = torch.stack(all_a2_fp4)
    batched_a2_scale = torch.stack(all_a2_scale)
    del all_a2_fp4, all_a2_scale

    g2_alpha_vec = torch.stack(
        [
            _nvfp4_get_expert_scalar(g2_alphas, e, field_name="g2_alphas")
            for e in range(num_experts)
        ]
    )

    # ------------------------------------------------------------------
    # Phase 5: grouped GEMM-2   (all experts in one kernel launch)
    # ------------------------------------------------------------------
    gemm2_out = _grouped_matmul_nvfp4(
        a_fp4=batched_a2_fp4,
        b_fp4=w2_weight,
        a_scale=batched_a2_scale,
        b_scale=w2_weight_scale,
        alpha=g2_alpha_vec,
        output_dtype=dtype,
    )
    del batched_a2_fp4, batched_a2_scale

    if gemm2_out.shape[-1] != w2_output_size:
        gemm2_out = gemm2_out[..., :w2_output_size].contiguous()

    # ------------------------------------------------------------------
    # Phase 6: apply expert masks, topk weights, and reduce
    # ------------------------------------------------------------------
    contribution = torch.zeros(
        (M_total, hidden_dim), device=device, dtype=torch.float32
    )
    for expert_id in range(num_experts):
        expert_mask = (expert_ids == expert_id).unsqueeze(-1)
        weighted_output = (
            gemm2_out[expert_id].to(torch.float32)
            * output_scale.unsqueeze(-1)
            * expert_mask.to(torch.float32)
        )
        contribution += weighted_output
    del gemm2_out

    reduced = torch.zeros((num_tokens, hidden_dim), device=device, dtype=torch.float32)
    contribution = contribution.view(num_tokens, top_k, hidden_dim)
    for slot in range(top_k):
        reduced += contribution[:, slot, :]
    return reduced.to(dtype)


# ---------------------------------------------------------------------------
# Modular-kernel wrapper class
# ---------------------------------------------------------------------------


class BatchInvariantNvfp4Experts(mk.FusedMoEPermuteExpertsUnpermute):
    """Deterministic, batch-invariant NVFP4 MoE expert implementation.

    Wraps ``fused_moe_batch_invariant_nvfp4`` (per-expert Triton
    ``tl.dot_scaled`` GEMMs) into the modular-kernel interface so it
    can be used as a drop-in backend alongside FlashInfer/CUTLASS.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        assert quant_config.weight_quant_dtype == "nvfp4", (
            "BatchInvariantNvfp4Experts only supports nvfp4 weight quantization, "
            f"got {quant_config.weight_quant_dtype}"
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.has_device_capability(100)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return True

    def supports_chunking(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace13 = (M, K)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace13, workspace2, output_shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        result = fused_moe_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w1,
            w13_weight_scale=self.w1_scale,
            w2_weight=w2,
            w2_weight_scale=self.w2_scale,
            a1_gscale=self.a1_gscale,
            g1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            g2_alphas=self.g2_alphas,
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            expert_map=expert_map,
        )
        output.copy_(result)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for batch-invariant NVFP4 MoE")
