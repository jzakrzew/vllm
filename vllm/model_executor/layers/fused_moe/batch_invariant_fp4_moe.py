# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant FP4 fused MoE expert implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import _compute_pid
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_unpermute,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units

logger = init_logger(__name__)


def _map_extra_rows(m_total: int, cols: int) -> int:
    """Extra workspace13 rows needed to embed ``a_map`` and ``c_map``."""
    bytes_needed = 2 * m_total * 4  # 2 maps x M_total x sizeof(int32)
    row_bytes = cols * 2  # cols x sizeof(bfloat16)
    return (bytes_needed + row_bytes - 1) // row_bytes


@dataclass
class NvFP4MoEWorkspace:
    """Pre-allocated scratch buffers for batch-invariant FP4 MoE.

    Expert-count-dependent tensors are allocated once in the experts class and
    reused across forward calls to avoid hot-path allocations.

    The M-dependent ``a_map`` / ``c_map`` buffers are carved out of the extra
    rows appended to ``workspace13`` by ``workspace_shapes``.
    """

    expert_offsets: torch.Tensor  # [E+1] int32
    blockscale_offsets: torch.Tensor  # [E+1] int32
    problem_sizes1: torch.Tensor  # [E, 3] int32
    problem_sizes2: torch.Tensor  # [E, 3] int32
    tiles_per_expert: torch.Tensor  # [E] int32
    expert_tile_start: torch.Tensor  # [E+1] int32
    dummy_bias: torch.Tensor  # [0] float32


class _GroupedGemmAMode(Enum):
    NVFP4_PACKED = 0
    MXFP8 = 1


_A_NVFP4_PACKED = tl.constexpr(int(_GroupedGemmAMode.NVFP4_PACKED.value))
_A_MXFP8 = tl.constexpr(int(_GroupedGemmAMode.MXFP8.value))
_A_DOT_TYPE: dict[_GroupedGemmAMode, str] = {
    _GroupedGemmAMode.NVFP4_PACKED: "e2m1",
    _GroupedGemmAMode.MXFP8: "e4m3",
}


def _quantize_mxfp8_experts(
    input_tensor: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert-packed activations to MXFP8 and swizzle their scales."""
    assert input_tensor.shape[1] % MXFP8_BLOCK_SIZE == 0, (
        "MXFP8 expert quantization requires the hidden dimension to be divisible "
        f"by {MXFP8_BLOCK_SIZE}, got {input_tensor.shape[1]}."
    )
    num_experts = problem_sizes.shape[0]
    quant_output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scale_output = torch.zeros(
        (
            ((input_tensor.shape[0] + 127 * num_experts + 127) // 128) * 128,
            input_tensor.shape[1] // MXFP8_BLOCK_SIZE,
        ),
        device=input_tensor.device,
        dtype=torch.uint8,
    )
    ops.mxfp8_experts_quant(
        input_tensor=input_tensor.contiguous(),
        problem_sizes=problem_sizes.to(dtype=torch.int32).contiguous(),
        expert_offsets=expert_offsets[:-1].to(dtype=torch.int32).contiguous(),
        blockscale_offsets=blockscale_offsets[:-1].to(dtype=torch.int32).contiguous(),
        quant_output=quant_output,
        scale_factor=scale_output,
    )
    scale_output = swizzle_blockscale(scale_output.view(torch.float8_e4m3fn)).view(
        torch.uint8
    )
    return quant_output, scale_output


def _validate_fp4_moe_shared_user_tensors(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    output: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> None:
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
        f"hidden_states must be float16 or bfloat16, got {hidden_states.dtype}"
    )
    assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
    assert topk_ids.dtype == torch.int32, (
        f"topk_ids must be int32, got {topk_ids.dtype}"
    )
    assert topk_weights.is_contiguous(), "topk_weights must be contiguous"
    assert topk_weights.dtype == torch.float32, (
        f"topk_weights must be float32, got {topk_weights.dtype}"
    )
    assert output.is_contiguous(), "output must be contiguous"
    assert workspace13.is_contiguous(), "workspace13 must be contiguous"
    assert workspace2.is_contiguous(), "workspace2 must be contiguous"
    assert w13_weight.is_contiguous(), "w13_weight must be contiguous"
    assert w13_weight_scale.is_contiguous(), "w13_weight_scale must be contiguous"
    assert w2_weight.is_contiguous(), "w2_weight must be contiguous"
    assert w2_weight_scale.is_contiguous(), "w2_weight_scale must be contiguous"
    assert expert_map is None, (
        "Batch-invariant FP4 MoE does not support expert_map. "
        "This kernel requires local expert IDs and ep_size == 1."
    )


def _validate_fused_moe_batch_invariant_nvfp4_inputs(
    *,
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
    output: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_map: torch.Tensor | None,
    num_experts: int,
) -> None:
    _validate_fp4_moe_shared_user_tensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        output=output,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_map=expert_map,
    )

    for name, tensor in (
        ("a1_gscale", a1_gscale),
        ("a2_gscale", a2_gscale),
        ("g1_alphas", g1_alphas),
        ("g2_alphas", g2_alphas),
    ):
        if tensor is None:
            raise RuntimeError(f"Missing required NVFP4 MoE tensor: {name}")
        assert tensor.ndim == 1, (
            f"NVFP4 MoE tensor '{name}' must be 1-D, got shape {tuple(tensor.shape)}."
        )
        assert tensor.dtype == torch.float32, (
            f"NVFP4 MoE tensor '{name}' must be float32, got {tensor.dtype}."
        )
        assert tensor.is_contiguous(), f"NVFP4 MoE tensor '{name}' must be contiguous."
        assert tensor.numel() == num_experts, (
            f"NVFP4 MoE tensor '{name}' must have {num_experts} elements, "
            f"got {tensor.numel()}."
        )


def _validate_fused_moe_batch_invariant_mxfp4_inputs(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    output: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> None:
    _validate_fp4_moe_shared_user_tensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        output=output,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_map=expert_map,
    )


@triton.jit
def _unswizzle_scale(
    scale_raw,
    TILE_ROWS: tl.constexpr,
    TILE_SCALE_COLS: tl.constexpr,
):
    """Un-swizzle block scales from hardware-interleaved 128x4 layout."""
    return (
        scale_raw.reshape(TILE_ROWS // 128, TILE_SCALE_COLS // 4, 32, 4, 4)
        .trans(0, 3, 2, 1, 4)
        .reshape(TILE_ROWS, TILE_SCALE_COLS)
    )


# ---------------------------------------------------------------------------
# Packed grouped FP4 GEMM kernel and wrappers
# ---------------------------------------------------------------------------


@triton.jit
def _find_expert_bin_search(
    expert_tile_start_ptr,
    global_tile_id,
    num_experts,
):
    """Find the expert that owns ``global_tile_id``."""
    lo: tl.int32 = 0
    hi: tl.int32 = num_experts
    while lo < hi:
        mid = (lo + hi) // 2
        val = tl.load(expert_tile_start_ptr + mid + 1).to(tl.int32)
        if val <= global_tile_id:
            lo = mid + 1
        else:
            hi = mid
    return lo


@triton.jit
def _maybe_widen(val, LARGE: tl.constexpr):
    """Promote *val* to int64 when LARGE (tensor exceeds int32 indexing)."""
    if LARGE:
        return val.to(tl.int64)
    return val


@triton.jit
def _grouped_matmul_fp4_packed_persistent_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    alpha_ptr,
    bias_ptr,
    expert_offsets_ptr,
    a_scale_offsets_ptr,
    problem_sizes_ptr,
    expert_tile_start_ptr,
    num_experts,
    M_total,
    N_total,
    K_total,
    a_scale_rows_total,
    stride_ps0,
    stride_ps1,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bm,
    stride_bk,
    stride_bsk,
    stride_cm,
    stride_cn,
    stride_bias_e,
    a_scale_cols_total,
    b_scale_cols_total,
    b_scale_n_per_expert,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    A_MODE: tl.constexpr,
    A_DOT_TYPE: tl.constexpr,
    B_SCALE_GROUP: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Persistent packed grouped FP4 GEMM with selectable A-side precision."""
    start_pid = tl.program_id(axis=0)
    total_tiles = tl.load(expert_tile_start_ptr + num_experts).to(tl.int32)

    K_BYTES: tl.constexpr = BLOCK_SIZE_K // 2
    SCALE_K_TILE: tl.constexpr = BLOCK_SIZE_K // B_SCALE_GROUP
    SCALE_K_TILES: tl.constexpr = SCALE_K_TILE // 4
    SCALE_M_TILES: tl.constexpr = BLOCK_SIZE_M // 128
    SCALE_N_TILES: tl.constexpr = BLOCK_SIZE_N // 128

    k_bytes_total = K_total // 2
    k_tiles = tl.cdiv(K_total, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N_total, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    if A_MODE == _A_NVFP4_PACKED:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M_total, k_bytes_total],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, K_BYTES],
            padding_option="zero",
        )
    else:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M_total, K_total],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            padding_option="zero",
        )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[num_experts * N_total, k_bytes_total],
        strides=[stride_bm, stride_bk],
        block_shape=[BLOCK_SIZE_N, K_BYTES],
        padding_option="zero",
    )
    a_scale_desc = tl.make_tensor_descriptor(
        a_scale_ptr,
        shape=[
            1,
            tl.cdiv(a_scale_rows_total, 128),
            a_scale_cols_total // 4,
            2,
            256,
        ],
        strides=[
            tl.cdiv(a_scale_rows_total, 128)
            * (a_scale_cols_total // 4)
            * 512
            * stride_ask,
            (a_scale_cols_total // 4) * 512 * stride_ask,
            512 * stride_ask,
            256 * stride_ask,
            stride_ask,
        ],
        block_shape=[1, SCALE_M_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )
    b_scale_n_tiles = tl.cdiv(num_experts * b_scale_n_per_expert, 128)
    b_scale_desc = tl.make_tensor_descriptor(
        b_scale_ptr,
        shape=[1, b_scale_n_tiles, b_scale_cols_total // 4, 2, 256],
        strides=[
            b_scale_n_tiles * (b_scale_cols_total // 4) * 512 * stride_bsk,
            (b_scale_cols_total // 4) * 512 * stride_bsk,
            512 * stride_bsk,
            256 * stride_bsk,
            stride_bsk,
        ],
        block_shape=[1, SCALE_N_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )

    for global_tid in tl.range(start_pid, total_tiles, NUM_SMS, flatten=True):
        expert_id = _find_expert_bin_search(
            expert_tile_start_ptr, global_tid, num_experts
        )
        local_tile_start = tl.load(expert_tile_start_ptr + expert_id).to(tl.int32)
        local_tile_id = global_tid - local_tile_start

        M = tl.load(problem_sizes_ptr + expert_id * stride_ps0).to(tl.int32)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

        pid_m, pid_n = _compute_pid(
            local_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, 0
        )

        expert_row_offset = tl.load(expert_offsets_ptr + expert_id).to(tl.int32)
        expert_scale_offset = tl.load(a_scale_offsets_ptr + expert_id).to(tl.int32)
        expert_row_offset = _maybe_widen(expert_row_offset, A_LARGE)
        expert_scale_offset = _maybe_widen(expert_scale_offset, A_LARGE)

        if HAS_ALPHA:
            alpha = tl.load(alpha_ptr + expert_id).to(tl.float32)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N_total

        b_row_offset = _maybe_widen(expert_id * N_total, B_LARGE)
        bs_row_offset = _maybe_widen(expert_id * b_scale_n_per_expert, B_LARGE)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            k_start_bytes = _maybe_widen(ki * K_BYTES, A_LARGE or B_LARGE)
            k_byte_mask = (ki * K_BYTES + tl.arange(0, K_BYTES)) < k_bytes_total

            a_m_start = _maybe_widen(expert_row_offset + start_m, A_LARGE)
            b_n_start = _maybe_widen(b_row_offset + start_n, B_LARGE)

            if A_MODE == _A_NVFP4_PACKED:
                a = a_desc.load([a_m_start, k_start_bytes])
                a = tl.where(m_mask[:, None] & k_byte_mask[None, :], a, 0)
            else:
                k_start_elems = _maybe_widen(ki * BLOCK_SIZE_K, A_LARGE)
                a = a_desc.load([a_m_start, k_start_elems])
                k_elem_mask = (ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) < K_total
                a = tl.where(m_mask[:, None] & k_elem_mask[None, :], a, 0)
            b_raw = b_desc.load([b_n_start, k_start_bytes])
            b = tl.where(n_mask[:, None] & k_byte_mask[None, :], b_raw, 0).T

            scale_tile_k = _maybe_widen(ki * SCALE_K_TILES, A_LARGE or B_LARGE)
            scale_tile_n = _maybe_widen((bs_row_offset + start_n) // 128, B_LARGE)
            b_scale_raw = b_scale_desc.load([0, scale_tile_n, scale_tile_k, 0, 0])
            b_scale = _unswizzle_scale(
                b_scale_raw.reshape(BLOCK_SIZE_N, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_N,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )

            scale_tile_m = _maybe_widen((expert_scale_offset + start_m) // 128, A_LARGE)
            a_scale_raw = a_scale_desc.load([0, scale_tile_m, scale_tile_k, 0, 0])
            a_scale = _unswizzle_scale(
                a_scale_raw.reshape(BLOCK_SIZE_M, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_M,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )

            accumulator = tl.dot_scaled(
                a,
                a_scale,
                A_DOT_TYPE,
                b,
                b_scale,
                "e2m1",
                accumulator,
            )

        if HAS_ALPHA:
            accumulator *= alpha
        if HAS_BIAS:
            bias_ptrs = bias_ptr + expert_id * stride_bias_e + offs_n
            bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            accumulator += bias_vals[None, :]
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_expert_ptr = c_ptr + expert_row_offset * stride_cm
        offs_m_c = _maybe_widen(offs_m, A_LARGE)
        offs_n_c = _maybe_widen(offs_n, B_LARGE)
        c_ptrs = (
            c_expert_ptr + offs_m_c[:, None] * stride_cm + offs_n_c[None, :] * stride_cn
        )
        tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def _grouped_matmul_nvfp4_packed(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    *,
    output: torch.Tensor,
    tiles_per_expert: torch.Tensor,
    expert_tile_start: torch.Tensor,
    dummy_bias: torch.Tensor,
    a_scale_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed grouped NVFP4 GEMM with expert-local problem metadata."""
    assert a_fp4.dtype == torch.uint8 and b_fp4.dtype == torch.uint8
    assert a_scale.dtype == torch.float8_e4m3fn and b_scale.dtype == torch.float8_e4m3fn
    assert alpha.dtype == torch.float32

    E = b_fp4.shape[0]
    if E == 0:
        M_logical = a_fp4.shape[0]
        N_out = b_fp4.shape[1]
        return output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if a_scale_offsets is None:
        a_scale_offsets = expert_offsets

    expert_offsets = expert_offsets[:-1]
    a_scale_offsets = a_scale_offsets[:-1]

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 256

    M_logical = a_fp4.shape[0]
    max_tiles_m = triton.cdiv(M_logical, BLOCK_SIZE_M)
    max_tiles_n = triton.cdiv(b_fp4.shape[1], BLOCK_SIZE_N)
    max_tiles_per_expert = max_tiles_m * max_tiles_n

    N_out = b_fp4.shape[1]
    c_work = output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if max_tiles_per_expert == 0:
        return c_work

    NUM_SMS = num_compute_units(a_fp4.device.index)
    worst_case_tiles = E * max_tiles_per_expert
    A_LARGE = max(a_fp4.numel(), a_scale.numel(), c_work.numel()) > 2**31
    B_LARGE = max(b_fp4.numel(), b_scale.numel()) > 2**31

    M_per_expert = problem_sizes[:, 0].to(torch.int32)
    torch.div(
        M_per_expert + BLOCK_SIZE_M - 1,
        BLOCK_SIZE_M,
        rounding_mode="floor",
        out=tiles_per_expert,
    )
    tiles_per_expert.mul_(max_tiles_n)
    tiles_per_expert.mul_((M_per_expert > 0).to(torch.int32))
    torch.cumsum(tiles_per_expert, dim=0, out=expert_tile_start[1:])

    b_fp4_flat = b_fp4.reshape(-1, b_fp4.shape[2])
    b_scale_flat = b_scale.reshape(-1, b_scale.shape[2])

    grid = (min(NUM_SMS, worst_case_tiles),)
    _grouped_matmul_fp4_packed_persistent_kernel[grid](
        a_fp4,
        a_scale,
        b_fp4_flat,
        b_scale_flat,
        c_work,
        alpha,
        dummy_bias,
        expert_offsets,
        a_scale_offsets,
        problem_sizes,
        expert_tile_start,
        E,
        a_fp4.shape[0],
        b_fp4.shape[1],
        a_fp4.shape[1] * 2,
        a_scale.shape[0],
        problem_sizes.stride(0),
        problem_sizes.stride(1),
        a_fp4.stride(0),
        a_fp4.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_fp4_flat.stride(0),
        b_fp4_flat.stride(1),
        b_scale_flat.stride(1),
        c_work.stride(0),
        c_work.stride(1),
        0,
        a_scale.shape[1],
        b_scale.shape[2],
        b_scale.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
        A_LARGE=A_LARGE,
        B_LARGE=B_LARGE,
        A_MODE=_GroupedGemmAMode.NVFP4_PACKED.value,
        A_DOT_TYPE=_A_DOT_TYPE[_GroupedGemmAMode.NVFP4_PACKED],
        B_SCALE_GROUP=16,
        HAS_ALPHA=True,
        HAS_BIAS=False,
        num_stages=2,
        num_warps=8,
    )
    return c_work


def _grouped_matmul_mxfp4_packed(
    a_mxfp8: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    *,
    output: torch.Tensor,
    tiles_per_expert: torch.Tensor,
    expert_tile_start: torch.Tensor,
    dummy_bias: torch.Tensor,
    bias: torch.Tensor | None = None,
    a_scale_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed grouped MXFP4 GEMM with MXFP8 activations."""
    assert a_mxfp8.dtype == torch.float8_e4m3fn
    assert b_fp4.dtype == torch.uint8
    assert a_scale.dtype == torch.uint8
    assert b_scale.dtype == torch.uint8

    E = b_fp4.shape[0]
    if E == 0:
        M_logical = a_mxfp8.shape[0]
        N_out = b_fp4.shape[1]
        return output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if a_scale_offsets is None:
        a_scale_offsets = expert_offsets

    expert_offsets = expert_offsets[:-1]
    a_scale_offsets = a_scale_offsets[:-1]

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    M_logical = a_mxfp8.shape[0]
    max_tiles_m = triton.cdiv(M_logical, BLOCK_SIZE_M)
    max_tiles_n = triton.cdiv(b_fp4.shape[1], BLOCK_SIZE_N)
    max_tiles_per_expert = max_tiles_m * max_tiles_n

    N_out = b_fp4.shape[1]
    c_work = output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if max_tiles_per_expert == 0:
        return c_work

    NUM_SMS = num_compute_units(a_mxfp8.device.index)
    worst_case_tiles = E * max_tiles_per_expert
    A_LARGE = max(a_mxfp8.numel(), a_scale.numel(), c_work.numel()) > 2**31
    B_LARGE = max(b_fp4.numel(), b_scale.numel()) > 2**31

    M_per_expert = problem_sizes[:, 0].to(torch.int32)
    torch.div(
        M_per_expert + BLOCK_SIZE_M - 1,
        BLOCK_SIZE_M,
        rounding_mode="floor",
        out=tiles_per_expert,
    )
    tiles_per_expert.mul_(max_tiles_n)
    tiles_per_expert.mul_((M_per_expert > 0).to(torch.int32))
    torch.cumsum(tiles_per_expert, dim=0, out=expert_tile_start[1:])

    b_fp4_flat = b_fp4.reshape(-1, b_fp4.shape[2])
    b_scale_flat = b_scale.reshape(-1, b_scale.shape[2])
    bias_tensor = bias if bias is not None else dummy_bias

    grid = (min(NUM_SMS, worst_case_tiles),)
    _grouped_matmul_fp4_packed_persistent_kernel[grid](
        a_mxfp8,
        a_scale,
        b_fp4_flat,
        b_scale_flat,
        c_work,
        dummy_bias,
        bias_tensor,
        expert_offsets,
        a_scale_offsets,
        problem_sizes,
        expert_tile_start,
        E,
        a_mxfp8.shape[0],
        b_fp4.shape[1],
        a_mxfp8.shape[1],
        a_scale.shape[0],
        problem_sizes.stride(0),
        problem_sizes.stride(1),
        a_mxfp8.stride(0),
        a_mxfp8.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_fp4_flat.stride(0),
        b_fp4_flat.stride(1),
        b_scale_flat.stride(1),
        c_work.stride(0),
        c_work.stride(1),
        bias_tensor.stride(0) if bias is not None else 0,
        a_scale.shape[1],
        b_scale.shape[2],
        b_scale.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
        A_LARGE=A_LARGE,
        B_LARGE=B_LARGE,
        A_MODE=_GroupedGemmAMode.MXFP8.value,
        A_DOT_TYPE=_A_DOT_TYPE[_GroupedGemmAMode.MXFP8],
        B_SCALE_GROUP=32,
        HAS_ALPHA=False,
        HAS_BIAS=bias is not None,
        num_stages=2,
        num_warps=8,
    )
    return c_work


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
    activation: MoEActivation,
    *,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    output: torch.Tensor,
    workspace: NvFP4MoEWorkspace,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """Deterministic NVFP4 MoE using packed routing metadata and grouped GEMMs."""
    assert hidden_states.ndim == 2, (
        f"Expected 2D hidden_states for NVFP4 MoE fallback, got {hidden_states.shape}."
    )
    assert topk_ids.shape == topk_weights.shape, (
        "NVFP4 MoE fallback expects topk_ids and topk_weights to have identical shapes."
    )
    assert topk_ids.ndim == 2, (
        f"Expected 2D top-k routing tensors, got shape {topk_ids.shape}."
    )
    assert not apply_router_weight_on_input or topk_ids.shape[1] == 1, (
        "apply_router_weight_on_input=True is only supported for top_k == 1."
    )
    assert quant_backend == "cutlass", (
        "Packed batch-invariant NVFP4 MoE requires quant_backend='cutlass'."
    )
    activation_kind = activation

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]
    _validate_fused_moe_batch_invariant_nvfp4_inputs(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        output=output,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_map=expert_map,
        num_experts=num_experts,
    )
    M_total = num_tokens * top_k
    if M_total == 0:
        return output

    routed_topk_ids = topk_ids
    valid_routes = (routed_topk_ids >= 0) & (routed_topk_ids < num_experts)
    routed_topk_ids = torch.where(valid_routes, routed_topk_ids, -1)
    routed_topk_weights = topk_weights

    if apply_router_weight_on_input:
        packed_hidden_states = hidden_states * routed_topk_weights.view(-1, 1).to(
            hidden_states.dtype
        )
    else:
        packed_hidden_states = hidden_states

    w1_output_size = w13_weight.shape[1]
    activation_out_dim = (
        w1_output_size // 2 if activation_kind.is_gated else w1_output_size
    )
    w2_output_size = hidden_dim
    w1_padding_cols = max(0, w13_weight.shape[-1] - hidden_dim // 2)
    w2_padding_cols = max(0, w2_weight.shape[-1] - activation_out_dim // 2)
    min_w13_cols = max(w13_weight.shape[1], hidden_dim)
    if workspace13.shape[0] < M_total or workspace13.shape[1] < min_w13_cols:
        raise RuntimeError(
            "workspace13 is too small for GEMM1 staging. "
            f"Need at least ({M_total}, {min_w13_cols}), "
            f"got {tuple(workspace13.shape)}."
        )
    required_workspace2_cols = max(activation_out_dim, w2_output_size)
    if workspace2.shape[0] < M_total or workspace2.shape[1] < required_workspace2_cols:
        raise RuntimeError(
            "workspace2 is too small for activation/GEMM2 staging. "
            f"Need at least ({M_total}, {required_workspace2_cols}), "
            f"got {tuple(workspace2.shape)}."
        )

    expert_offsets = workspace.expert_offsets
    blockscale_offsets = workspace.blockscale_offsets
    problem_sizes1 = workspace.problem_sizes1
    problem_sizes2 = workspace.problem_sizes2
    map_i32 = workspace13[M_total:].flatten().view(torch.int32)[: 2 * M_total]
    a_map = map_i32[:M_total]
    c_map = map_i32[M_total:]
    a_map.zero_()
    ops.get_cutlass_moe_mm_data(
        routed_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        activation_out_dim,
        hidden_dim,
        blockscale_offsets,
    )

    if not activation_kind.is_gated:
        problem_sizes1[:, 1].fill_(w1_output_size)
        problem_sizes2[:, 2].fill_(activation_out_dim)

    packed_hidden_states = ops.shuffle_rows(packed_hidden_states, a_map)

    a1_fp4, a1_scale = ops.scaled_fp4_experts_quant(
        packed_hidden_states,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        top_k,
    )
    a1_fp4 = pad_nvfp4_activation_for_cutlass(a1_fp4, w1_padding_cols)
    gemm1_out = slice_nvfp4_output(
        _grouped_matmul_nvfp4_packed(
            a_fp4=a1_fp4,
            b_fp4=w13_weight,
            a_scale=a1_scale,
            b_scale=w13_weight_scale,
            alpha=g1_alphas,
            expert_offsets=expert_offsets,
            a_scale_offsets=blockscale_offsets,
            problem_sizes=problem_sizes1,
            output=workspace13,
            tiles_per_expert=workspace.tiles_per_expert,
            expert_tile_start=workspace.expert_tile_start,
            dummy_bias=workspace.dummy_bias,
        ),
        w1_output_size,
    )

    if activation_kind == MoEActivation.SILU:
        int_fp4, int_scale = ops.silu_and_mul_scaled_fp4_experts_quant(
            gemm1_out,
            a2_gscale,
            expert_offsets,
            blockscale_offsets,
            top_k,
        )
    else:
        act_out = _resize_cache(workspace2, (M_total, activation_out_dim))
        apply_moe_activation(
            activation=activation_kind,
            output=act_out,
            input=gemm1_out,
        )
        int_fp4, int_scale = ops.scaled_fp4_experts_quant(
            act_out,
            a2_gscale,
            expert_offsets,
            blockscale_offsets,
            top_k,
        )
    int_fp4 = pad_nvfp4_activation_for_cutlass(int_fp4, w2_padding_cols)
    gemm2_out = slice_nvfp4_output(
        _grouped_matmul_nvfp4_packed(
            a_fp4=int_fp4,
            b_fp4=w2_weight,
            a_scale=int_scale,
            b_scale=w2_weight_scale,
            alpha=g2_alphas,
            expert_offsets=expert_offsets,
            a_scale_offsets=blockscale_offsets,
            problem_sizes=problem_sizes2,
            output=workspace2,
            tiles_per_expert=workspace.tiles_per_expert,
            expert_tile_start=workspace.expert_tile_start,
            dummy_bias=workspace.dummy_bias,
        ),
        w2_output_size,
    )

    epilogue_topk_weights = (
        torch.ones_like(routed_topk_weights)
        if apply_router_weight_on_input
        else routed_topk_weights
    )
    inv_permuted_idx = c_map.view(num_tokens, top_k)
    reduced = output
    moe_unpermute(
        out=reduced,
        permuted_hidden_states=gemm2_out,
        topk_weights=epilogue_topk_weights,
        inv_permuted_idx=inv_permuted_idx,
        expert_first_token_offset=expert_offsets.to(torch.int64),
    )
    return reduced


# ---------------------------------------------------------------------------
# Standalone deterministic MXFP4 MoE implementation
# ---------------------------------------------------------------------------


def fused_moe_batch_invariant_mxfp4(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    activation: MoEActivation,
    *,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    output: torch.Tensor,
    workspace: NvFP4MoEWorkspace,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic MXFP4 MoE using MXFP8 activations and packed routing."""
    assert hidden_states.ndim == 2, (
        f"Expected 2D hidden_states, got {hidden_states.shape}."
    )
    assert topk_ids.shape == topk_weights.shape, (
        "topk_ids and topk_weights must have identical shapes."
    )
    assert topk_ids.ndim == 2, (
        f"Expected 2D top-k routing tensors, got shape {topk_ids.shape}."
    )
    assert not apply_router_weight_on_input or topk_ids.shape[1] == 1, (
        "apply_router_weight_on_input=True is only supported for top_k == 1."
    )
    activation_kind = activation

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]
    _validate_fused_moe_batch_invariant_mxfp4_inputs(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_weight,
        w13_weight_scale=w13_weight_scale,
        w2_weight=w2_weight,
        w2_weight_scale=w2_weight_scale,
        output=output,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_map=expert_map,
    )
    M_total = num_tokens * top_k
    if M_total == 0:
        return output

    routed_topk_ids = topk_ids
    valid_routes = (routed_topk_ids >= 0) & (routed_topk_ids < num_experts)
    routed_topk_ids = torch.where(valid_routes, routed_topk_ids, -1)
    routed_topk_weights = topk_weights

    if apply_router_weight_on_input:
        packed_hidden_states = hidden_states * routed_topk_weights.view(-1, 1).to(
            hidden_states.dtype
        )
    else:
        packed_hidden_states = hidden_states

    w1_output_size = w13_weight.shape[1]
    activation_out_dim = (
        w1_output_size // 2 if activation_kind.is_gated else w1_output_size
    )
    w2_output_size = hidden_dim

    min_w13_cols = max(w1_output_size, hidden_dim)
    if workspace13.shape[0] < M_total or workspace13.shape[1] < min_w13_cols:
        raise RuntimeError(
            "workspace13 is too small for GEMM1 staging. "
            f"Need at least ({M_total}, {min_w13_cols}), "
            f"got {tuple(workspace13.shape)}."
        )
    required_workspace2_cols = max(activation_out_dim, w2_output_size)
    if workspace2.shape[0] < M_total or workspace2.shape[1] < required_workspace2_cols:
        raise RuntimeError(
            "workspace2 is too small for activation/GEMM2 staging. "
            f"Need at least ({M_total}, {required_workspace2_cols}), "
            f"got {tuple(workspace2.shape)}."
        )

    expert_offsets = workspace.expert_offsets
    blockscale_offsets = workspace.blockscale_offsets
    problem_sizes1 = workspace.problem_sizes1
    problem_sizes2 = workspace.problem_sizes2
    map_i32 = workspace13[M_total:].flatten().view(torch.int32)[: 2 * M_total]
    a_map = map_i32[:M_total]
    c_map = map_i32[M_total:]
    a_map.zero_()
    ops.get_cutlass_moe_mm_data(
        routed_topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_experts,
        activation_out_dim,
        hidden_dim,
        blockscale_offsets,
    )

    if not activation_kind.is_gated:
        problem_sizes1[:, 1].fill_(w1_output_size)
        problem_sizes2[:, 2].fill_(activation_out_dim)

    packed_hidden_states = ops.shuffle_rows(packed_hidden_states, a_map)
    a1_mxfp8, a1_scale = _quantize_mxfp8_experts(
        packed_hidden_states,
        problem_sizes1,
        expert_offsets,
        blockscale_offsets,
    )
    gemm1_out = _grouped_matmul_mxfp4_packed(
        a_mxfp8=a1_mxfp8,
        b_fp4=w13_weight,
        a_scale=a1_scale,
        b_scale=w13_weight_scale,
        expert_offsets=expert_offsets,
        problem_sizes=problem_sizes1,
        output=workspace13,
        tiles_per_expert=workspace.tiles_per_expert,
        expert_tile_start=workspace.expert_tile_start,
        dummy_bias=workspace.dummy_bias,
        bias=w13_bias,
        a_scale_offsets=blockscale_offsets,
    )
    if gemm1_out.shape[-1] != w1_output_size:
        gemm1_out = gemm1_out[:, :w1_output_size].contiguous()

    act_out = _resize_cache(workspace2, (M_total, activation_out_dim))
    apply_moe_activation(
        activation=activation_kind,
        output=act_out,
        input=gemm1_out,
    )

    gemm2_a, gemm2_a_scale = _quantize_mxfp8_experts(
        act_out,
        problem_sizes2,
        expert_offsets,
        blockscale_offsets,
    )
    gemm2_out = _grouped_matmul_mxfp4_packed(
        a_mxfp8=gemm2_a,
        b_fp4=w2_weight,
        a_scale=gemm2_a_scale,
        b_scale=w2_weight_scale,
        expert_offsets=expert_offsets,
        problem_sizes=problem_sizes2,
        output=workspace2,
        tiles_per_expert=workspace.tiles_per_expert,
        expert_tile_start=workspace.expert_tile_start,
        dummy_bias=workspace.dummy_bias,
        bias=w2_bias,
        a_scale_offsets=blockscale_offsets,
    )
    if gemm2_out.shape[-1] != w2_output_size:
        gemm2_out = gemm2_out[:, :w2_output_size].contiguous()

    epilogue_topk_weights = (
        torch.ones_like(routed_topk_weights)
        if apply_router_weight_on_input
        else routed_topk_weights
    )
    inv_permuted_idx = c_map.view(num_tokens, top_k)
    reduced = output
    moe_unpermute(
        out=reduced,
        permuted_hidden_states=gemm2_out,
        topk_weights=epilogue_topk_weights,
        inv_permuted_idx=inv_permuted_idx,
        expert_first_token_offset=expert_offsets.to(torch.int64),
    )
    return reduced


# ---------------------------------------------------------------------------
# Modular-kernel wrapper classes
# ---------------------------------------------------------------------------


class _BatchInvariantFP4ExpertsBase(mk.FusedMoEExpertsModular, ABC):
    """Shared batch-invariant FP4 MoE expert logic."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._num_local_experts = moe_config.num_local_experts

        E = moe_config.num_local_experts
        device = moe_config.device
        self._workspace = NvFP4MoEWorkspace(
            expert_offsets=torch.empty(E + 1, dtype=torch.int32, device=device),
            blockscale_offsets=torch.empty(E + 1, dtype=torch.int32, device=device),
            problem_sizes1=torch.empty(E, 3, dtype=torch.int32, device=device),
            problem_sizes2=torch.empty(E, 3, dtype=torch.int32, device=device),
            tiles_per_expert=torch.empty(E, dtype=torch.int32, device=device),
            expert_tile_start=torch.zeros(E + 1, dtype=torch.int32, device=device),
            dummy_bias=torch.empty(0, device=device, dtype=torch.float32),
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
    def _supports_activation(activation: MoEActivation) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_chunking(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
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
        act_out_dim = self.adjust_N_for_activation(N, activation)
        m_total = M * topk
        cols13 = max(N, K)
        extra = _map_extra_rows(m_total, cols13)
        workspace13 = (m_total + extra, cols13)
        workspace2 = (m_total, max(act_out_dim, K))
        output_shape = (M, K)
        return (workspace13, workspace2, output_shape)

    @abstractmethod
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
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None: ...

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for batch-invariant FP4 MoE")


class BatchInvariantNvfp4Experts(_BatchInvariantFP4ExpertsBase):
    """Batch-invariant NVFP4 (W4A4) MoE experts."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._cached_scale_vecs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def is_supported_config(
        cls,
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if (
            not envs.VLLM_BATCH_INVARIANT
            and moe_config.moe_backend != "batch_invariant"
        ):
            return (
                False,
                "NvFP4 batch-invariant MoE is not available unless explicitly enabled "
                "using VLLM_BATCH_INVARIANT=1 or moe_backend='batch_invariant'",
            )
        if moe_config.moe_parallel_config.ep_size > 1:
            return (
                False,
                "kernel does not support expert parallel for NVFP4 batch-invariant "
                "MoE. Use ep_size==1.",
            )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    def _get_nvfp4_scale_vecs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cached_scale_vecs is not None:
            return self._cached_scale_vecs
        self._cached_scale_vecs = (
            self.a1_gscale,
            self.a2_gscale,
            self.g1_alphas,
            self.g2_alphas,
        )
        return self._cached_scale_vecs

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
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        a1_gscale_vec, a2_gscale_vec, g1_alpha_vec, g2_alpha_vec = (
            self._get_nvfp4_scale_vecs()
        )
        fused_moe_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w1,
            w13_weight_scale=self.w1_scale,
            w2_weight=w2,
            w2_weight_scale=self.w2_scale,
            a1_gscale=a1_gscale_vec,
            g1_alphas=g1_alpha_vec,
            a2_gscale=a2_gscale_vec,
            g2_alphas=g2_alpha_vec,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            output=output,
            workspace=self._workspace,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            expert_map=expert_map,
        )


class BatchInvariantMxfp4Experts(_BatchInvariantFP4ExpertsBase):
    """Batch-invariant MXFP4 experts using MXFP8 activations."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp8Dynamic)

    @staticmethod
    def is_supported_config(
        cls,
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if (
            not envs.VLLM_BATCH_INVARIANT
            and moe_config.moe_backend != "batch_invariant"
        ):
            return (
                False,
                "MxFP4 batch-invariant MoE is not available unless explicitly enabled "
                "using VLLM_BATCH_INVARIANT=1 or moe_backend='batch_invariant'",
            )
        if moe_config.moe_parallel_config.ep_size > 1:
            return (
                False,
                "kernel does not support expert parallel for MXFP4 batch-invariant "
                "MoE. Use ep_size==1.",
            )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None

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
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        if self.quant_dtype != "mxfp8":
            raise RuntimeError(
                "BatchInvariantMxfp4Experts expects activation quant dtype 'mxfp8', "
                f"got {self.quant_dtype!r}."
            )
        fused_moe_batch_invariant_mxfp4(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w1,
            w13_weight_scale=self.w1_scale,
            w2_weight=w2,
            w2_weight_scale=self.w2_scale,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            output=output,
            workspace=self._workspace,
            w13_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            expert_map=expert_map,
        )
