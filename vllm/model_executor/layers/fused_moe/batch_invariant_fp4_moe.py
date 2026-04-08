# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant FP4 fused MoE expert implementations (NVFP4 and MXFP4)."""

from typing import Any

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    _compute_pid,
    _unswizzle_scale,
)
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.platform_utils import num_compute_units

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers used by ``fused_moe_batch_invariant_nvfp4``
# ---------------------------------------------------------------------------


def _nvfp4_get_expert_vector(
    tensor: torch.Tensor | None,
    *,
    num_experts: int,
    field_name: str,
) -> torch.Tensor:
    """Returns a contiguous float32 per-expert vector of length ``num_experts``."""
    if tensor is None:
        raise RuntimeError(f"Missing required NVFP4 MoE tensor: {field_name}")

    if tensor.ndim == 0 or tensor.numel() == 1:
        return tensor.reshape(1).to(torch.float32).expand(num_experts).contiguous()

    if tensor.shape[0] < num_experts:
        raise RuntimeError(
            f"NVFP4 MoE tensor '{field_name}' has {tensor.shape[0]} entries, "
            f"but {num_experts} experts are required."
        )

    flattened = tensor[:num_experts].reshape(num_experts, -1)
    if flattened.shape[1] != 1:
        raise RuntimeError(
            f"NVFP4 MoE tensor '{field_name}' must provide one value per expert, "
            f"got shape {tuple(tensor.shape)}."
        )
    return flattened[:, 0].to(torch.float32).contiguous()


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
# Packed grouped NVFP4 GEMM kernel and wrapper
# ---------------------------------------------------------------------------


@triton.jit
def _find_expert_bin_search(
    expert_tile_start_ptr,
    global_tile_id,
    num_experts,
):
    """Binary search to find which expert owns ``global_tile_id``.

    ``expert_tile_start_ptr`` points to an ``[E+1]`` prefix-sum array where
    ``expert_tile_start[e]`` is the first global tile belonging to expert
    ``e``.  Returns the expert index ``e`` such that
    ``expert_tile_start[e] <= global_tile_id < expert_tile_start[e+1]``.
    """
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
    A_IS_FP4: tl.constexpr,
    B_SCALE_GROUP: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Persistent packed grouped FP4 GEMM (NVFP4 and MXFP4).

    Launches ``NUM_SMS`` programs that loop over a global tile index space
    built from a prefix-sum of per-expert tile counts.  Each iteration maps
    a global tile id to ``(expert_id, local_tile_id)`` via binary search,
    then executes the same GEMM tile logic as the non-persistent kernel.

    When ``A_IS_FP4=True`` (NVFP4), both A and B are packed FP4 with
    block scales.  When ``A_IS_FP4=False`` (MXFP4 W4A16), A is BF16
    and only B has FP4 weights with block scales.

    B and B-scale tensors are pre-flattened by the wrapper:
    ``b_fp4`` from ``[E, N, K_packed]`` to ``[E*N, K_packed]``, and
    ``b_scale`` from ``[E, N_pad, K_s]`` to ``[E*N_pad, K_s]``.
    ``b_scale_n_per_expert`` = N_pad (may differ from N_total).

    When ``HAS_BIAS=True``, a per-expert bias vector ``[E, N]`` is added
    to the float32 accumulator after the optional alpha multiply and before
    the output cast/store.
    """
    start_pid = tl.program_id(axis=0)
    total_tiles = tl.load(expert_tile_start_ptr + num_experts).to(tl.int32)

    K_BYTES: tl.constexpr = BLOCK_SIZE_K // 2
    SCALE_K_TILE: tl.constexpr = BLOCK_SIZE_K // B_SCALE_GROUP
    SCALE_K_TILES: tl.constexpr = SCALE_K_TILE // 4
    SCALE_M_TILES: tl.constexpr = BLOCK_SIZE_M // 128
    SCALE_N_TILES: tl.constexpr = BLOCK_SIZE_N // 128

    k_bytes_total = K_total // 2

    # Global descriptors created once before the loop.
    if A_IS_FP4:
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
    # B flattened to [E*N, K_packed] by the wrapper.
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[num_experts * N_total, k_bytes_total],
        strides=[stride_bm, stride_bk],
        block_shape=[BLOCK_SIZE_N, K_BYTES],
        padding_option="zero",
    )
    if A_IS_FP4:
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
    # B-scale flattened to [E*N_pad, K_s] by the wrapper.
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

        p_ptr = problem_sizes_ptr + expert_id * stride_ps0
        M = tl.load(p_ptr + 0 * stride_ps1).to(tl.int32)
        N = tl.load(p_ptr + 1 * stride_ps1).to(tl.int32)
        K = tl.load(p_ptr + 2 * stride_ps1).to(tl.int32)

        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        pid_m, pid_n = _compute_pid(
            local_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, 0
        )

        expert_row_offset = tl.load(expert_offsets_ptr + expert_id).to(tl.int32)
        if A_IS_FP4:
            expert_scale_offset = tl.load(a_scale_offsets_ptr + expert_id).to(tl.int32)
        if A_LARGE:
            expert_row_offset = expert_row_offset.to(tl.int64)
            if A_IS_FP4:
                expert_scale_offset = expert_scale_offset.to(tl.int64)

        k_bytes = K // 2
        if HAS_ALPHA:
            alpha = tl.load(alpha_ptr + expert_id).to(tl.float32)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        m_mask = offs_m < M
        n_mask = offs_n < N

        # B offset into the flattened [E*N, K_packed] view.
        b_row_offset = expert_id * N_total
        # B-scale offset into the flattened [E*N_pad, K_s] view.
        bs_row_offset = expert_id * b_scale_n_per_expert
        if B_LARGE:
            b_row_offset = b_row_offset.to(tl.int64)
            bs_row_offset = bs_row_offset.to(tl.int64)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            k_start_bytes = ki * K_BYTES
            if A_LARGE or B_LARGE:
                k_start_bytes = k_start_bytes.to(tl.int64)
            k_byte_mask = (ki * K_BYTES + tl.arange(0, K_BYTES)) < k_bytes

            a_m_start = expert_row_offset + start_m
            b_n_start = b_row_offset + start_n
            if A_LARGE:
                a_m_start = a_m_start.to(tl.int64)
            if B_LARGE:
                b_n_start = b_n_start.to(tl.int64)

            if A_IS_FP4:
                a = a_desc.load([a_m_start, k_start_bytes])
                a = tl.where(m_mask[:, None] & k_byte_mask[None, :], a, 0)
            else:
                k_start_elems = ki * BLOCK_SIZE_K
                if A_LARGE:
                    k_start_elems = k_start_elems.to(tl.int64)
                k_elem_mask = (ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) < K
                a = a_desc.load([a_m_start, k_start_elems])
                a = tl.where(m_mask[:, None] & k_elem_mask[None, :], a, 0)
            b_raw = b_desc.load([b_n_start, k_start_bytes])
            b = tl.where(n_mask[:, None] & k_byte_mask[None, :], b_raw, 0).T

            scale_tile_k = ki * SCALE_K_TILES
            if A_LARGE or B_LARGE:
                scale_tile_k = scale_tile_k.to(tl.int64)
            scale_tile_n = (bs_row_offset + start_n) // 128
            if B_LARGE:
                scale_tile_n = scale_tile_n.to(tl.int64)

            b_scale_raw = b_scale_desc.load([0, scale_tile_n, scale_tile_k, 0, 0])
            b_scale = _unswizzle_scale(
                b_scale_raw.reshape(BLOCK_SIZE_N, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_N,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )

            if A_IS_FP4:
                scale_tile_m = (expert_scale_offset + start_m) // 128
                if A_LARGE:
                    scale_tile_m = scale_tile_m.to(tl.int64)
                a_scale_raw = a_scale_desc.load([0, scale_tile_m, scale_tile_k, 0, 0])
                a_scale = _unswizzle_scale(
                    a_scale_raw.reshape(BLOCK_SIZE_M, SCALE_K_TILE),
                    TILE_ROWS=BLOCK_SIZE_M,
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
            else:
                accumulator = tl.dot_scaled(
                    a,
                    None,
                    "bf16",
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
        offs_m_c = offs_m
        offs_n_c = offs_n
        if A_LARGE:
            offs_m_c = offs_m_c.to(tl.int64)
        if B_LARGE:
            offs_n_c = offs_n_c.to(tl.int64)
        c_ptrs = (
            c_expert_ptr + offs_m_c[:, None] * stride_cm + offs_n_c[None, :] * stride_cn
        )
        tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def _canonicalize_grouped_offsets(
    offsets: torch.Tensor,
    *,
    num_experts: int,
    name: str,
) -> torch.Tensor:
    if offsets.ndim != 1:
        raise RuntimeError(f"{name} must be 1-D, got shape {tuple(offsets.shape)}.")
    if offsets.numel() == num_experts + 1:
        offsets = offsets[:-1]
    if offsets.numel() != num_experts:
        raise RuntimeError(
            f"{name} must have {num_experts} or {num_experts + 1} entries, "
            f"got {offsets.numel()}."
        )
    if offsets.dtype not in (torch.int32, torch.int64):
        offsets = offsets.to(dtype=torch.int32)
    return offsets.contiguous()


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
    a_scale_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed grouped NVFP4 GEMM with expert-local problem metadata.

    TMA descriptors use ``padding_option="zero"`` so boundary tiles that
    overshoot ``M_total`` are zero-filled by hardware -- no guard rows or
    ``F.pad`` copies are needed.

    Args:
        a_fp4: [M_total, K_packed] uint8 packed FP4 activations in
            expert-packed row order.
        b_fp4: [E, N_max, K_packed] uint8 packed FP4 expert weights.
        a_scale: [S_total, K_s] float8_e4m3fn swizzled activation block
            scales.
        b_scale: [E, N_pad, K_s] float8_e4m3fn swizzled weight block scales.
        alpha: [E] float32 per-expert alpha.
        expert_offsets: [E] or [E+1] start row offsets in ``a_fp4``/output.
        problem_sizes: [E, 3] int tensor containing per-expert (M, N, K).
        a_scale_offsets: Optional [E] or [E+1] start row offsets in ``a_scale``.
            If not provided, ``expert_offsets`` are reused.
        output: Pre-allocated [>= M_total, >= N_max] tensor for the GEMM output.

    Returns:
        [M_total, N_max] tensor with grouped expert GEMM outputs.
    """
    assert a_fp4.ndim == 2 and b_fp4.ndim == 3, (
        "Expected packed a_fp4=[M_total, K_packed] and b_fp4=[E, N, K_packed]."
    )
    assert a_fp4.dtype == torch.uint8 and b_fp4.dtype == torch.uint8
    assert a_scale.ndim == 2 and b_scale.ndim == 3
    assert a_scale.dtype == torch.float8_e4m3fn and b_scale.dtype == torch.float8_e4m3fn
    assert alpha.dtype == torch.float32
    assert problem_sizes.ndim == 2 and problem_sizes.shape[1] == 3, (
        f"Expected problem_sizes shape [E, 3], got {tuple(problem_sizes.shape)}."
    )

    E = b_fp4.shape[0]
    if E == 0:
        M_logical = a_fp4.shape[0]
        N_out = b_fp4.shape[1]
        return output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    assert problem_sizes.shape[0] == E, (
        f"Expected problem_sizes.shape[0] == num_experts ({E}), "
        f"got {problem_sizes.shape[0]}."
    )
    assert b_fp4.shape[2] == a_fp4.shape[1], "Packed K dimensions must match."
    assert alpha.numel() == E, f"alpha must have {E} elements, got {alpha.numel()}."

    if a_scale_offsets is None:
        a_scale_offsets = expert_offsets

    tensors = {
        "a_fp4": a_fp4,
        "b_fp4": b_fp4,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "alpha": alpha,
        "expert_offsets": expert_offsets,
        "problem_sizes": problem_sizes,
        "a_scale_offsets": a_scale_offsets,
    }
    for name, tensor in tensors.items():
        if tensor.device != a_fp4.device:
            raise RuntimeError(
                f"{name} must be on {a_fp4.device}, got {tensor.device}."
            )

    expert_offsets = _canonicalize_grouped_offsets(
        expert_offsets, num_experts=E, name="expert_offsets"
    )
    a_scale_offsets = _canonicalize_grouped_offsets(
        a_scale_offsets, num_experts=E, name="a_scale_offsets"
    )
    problem_sizes = problem_sizes.to(dtype=torch.int32).contiguous()
    if not a_fp4.is_contiguous():
        a_fp4 = a_fp4.contiguous()
    if not a_scale.is_contiguous():
        a_scale = a_scale.contiguous()
    if not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    M_logical = a_fp4.shape[0]

    # Upper-bound tile count from static tensor shapes.  The persistent
    # kernel grid is min(NUM_SMS, worst_case_tiles), both derived from
    # fixed shapes / device properties, so the grid is constant across
    # calls and safe for CUDA graph capture.
    max_tiles_m = triton.cdiv(M_logical, BLOCK_SIZE_M)
    max_tiles_n = triton.cdiv(b_fp4.shape[1], BLOCK_SIZE_N)
    max_tiles_per_expert = max_tiles_m * max_tiles_n

    N_out = b_fp4.shape[1]
    c_work = output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if max_tiles_per_expert == 0:
        return c_work

    A_LARGE = max(a_fp4.numel(), a_scale.numel(), c_work.numel()) > 2**31
    B_LARGE = max(b_fp4.numel(), b_scale.numel()) > 2**31
    NUM_SMS = num_compute_units(a_fp4.device.index)
    worst_case_tiles = E * max_tiles_per_expert

    # Pre-compute tile-to-expert mapping (all device-side, no host sync).
    M_per_expert = problem_sizes[:, 0].to(torch.int32).clamp(min=0)
    N_per_expert = problem_sizes[:, 1].to(torch.int32).clamp(min=0)
    K_per_expert = problem_sizes[:, 2].to(torch.int32).clamp(min=0)
    valid = (M_per_expert > 0) & (N_per_expert > 0) & (K_per_expert > 0)
    tiles_per_expert = (
        torch.div(
            M_per_expert + BLOCK_SIZE_M - 1,
            BLOCK_SIZE_M,
            rounding_mode="floor",
        )
        * torch.div(
            N_per_expert + BLOCK_SIZE_N - 1,
            BLOCK_SIZE_N,
            rounding_mode="floor",
        )
        * valid.to(torch.int32)
    )
    expert_tile_start = torch.zeros(E + 1, dtype=torch.int32, device=a_fp4.device)
    expert_tile_start[1:] = torch.cumsum(tiles_per_expert, dim=0)

    # Flatten B [E, N, K] -> [E*N, K] and B-scale [E, Np, Ks] -> [E*Np, Ks]
    # for a single global TMA descriptor.
    b_fp4_flat = b_fp4.reshape(-1, b_fp4.shape[2])
    b_scale_flat = b_scale.reshape(-1, b_scale.shape[2])

    # TMA tensor descriptors allocate host-side metadata; Triton requires an allocator.
    set_triton_allocator(a_fp4.device)

    grid = (min(NUM_SMS, worst_case_tiles),)
    dummy_bias = torch.empty(0, device=a_fp4.device, dtype=torch.float32)
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
        0,  # stride_bias_e (unused)
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
        A_IS_FP4=True,
        B_SCALE_GROUP=16,
        HAS_ALPHA=True,
        HAS_BIAS=False,
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
    activation: Any,
    *,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    output: torch.Tensor,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """
    Deterministic NVFP4 MoE using packed routing metadata and grouped GEMMs.

    The input ``[M, K]`` is expanded and permuted into expert-major packed
    order ``[M * topk, K]`` via ``shuffle_rows`` with CUTLASS metadata
    (``a_map``/``c_map``).  Both GEMMs then run in packed 2D form using
    per-expert offsets/problem sizes.  The epilogue uses ``moe_unpermute``
    to gather, apply router weights and reduce back to ``[M, K]`` while
    safely skipping invalid routes.  The final reduction writes into
    ``output``.

    ``workspace13`` and ``workspace2`` must match ``BatchInvariantFP4Experts.
    workspace_shapes`` (large enough for GEMM1 / GEMM2 staging and activations).
    """

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

    activation_kind = (
        activation
        if isinstance(activation, MoEActivation)
        else MoEActivation.from_str(str(activation))
    )

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]
    M_total = num_tokens * top_k
    if M_total == 0:
        return output

    routed_topk_ids = topk_ids
    if expert_map is not None:
        routed_topk_ids = _nvfp4_moe_map_experts(topk_ids, expert_map)
    routed_topk_ids = routed_topk_ids.to(torch.int32)
    # Out-of-range IDs are treated as invalid routes.
    valid_routes = (routed_topk_ids >= 0) & (routed_topk_ids < num_experts)
    routed_topk_ids = torch.where(
        valid_routes, routed_topk_ids, torch.full_like(routed_topk_ids, -1)
    )
    routed_topk_weights = topk_weights.to(torch.float32)

    if apply_router_weight_on_input:
        packed_hidden_states = hidden_states * routed_topk_weights.view(-1, 1).to(
            hidden_states.dtype
        )
    else:
        packed_hidden_states = hidden_states

    device = hidden_states.device
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
    # Per-expert metadata/permutations for packed grouped-GEMM.
    expert_offsets = torch.empty((num_experts + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(
        (num_experts + 1), dtype=torch.int32, device=device
    )
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    a_map = torch.zeros((M_total,), dtype=torch.int32, device=device)
    c_map = torch.empty((M_total,), dtype=torch.int32, device=device)
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

    # get_cutlass_moe_mm_data() assumes gated MLP (`w13` width == 2 * n).
    # For *_no_mul activations, overwrite the logical problem shapes.
    if not activation_kind.is_gated:
        problem_sizes1[:, 1].fill_(w1_output_size)
        problem_sizes2[:, 2].fill_(activation_out_dim)

    packed_hidden_states = ops.shuffle_rows(packed_hidden_states, a_map)

    def _ensure_expert_vec(t: torch.Tensor | None, name: str) -> torch.Tensor:
        if (
            t is not None
            and t.ndim == 1
            and t.dtype == torch.float32
            and t.numel() == num_experts
            and t.is_contiguous()
        ):
            return t
        return _nvfp4_get_expert_vector(t, num_experts=num_experts, field_name=name)

    a1_gscale_vec = _ensure_expert_vec(a1_gscale, "a1_gscale")
    a2_gscale_vec = _ensure_expert_vec(a2_gscale, "a2_gscale")
    g1_alpha_vec = _ensure_expert_vec(g1_alphas, "g1_alphas")
    g2_alpha_vec = _ensure_expert_vec(g2_alphas, "g2_alphas")

    a1_fp4, a1_scale = ops.scaled_fp4_experts_quant(
        packed_hidden_states,
        a1_gscale_vec,
        expert_offsets,
        blockscale_offsets,
        top_k,
    )
    if w1_padding_cols > 0:
        a1_fp4 = F.pad(a1_fp4, (0, w1_padding_cols))

    gemm1_out = _grouped_matmul_nvfp4_packed(
        a_fp4=a1_fp4,
        b_fp4=w13_weight,
        a_scale=a1_scale,
        b_scale=w13_weight_scale,
        alpha=g1_alpha_vec,
        expert_offsets=expert_offsets,
        a_scale_offsets=blockscale_offsets,
        problem_sizes=problem_sizes1,
        output=workspace13,
    )
    if gemm1_out.shape[-1] != w1_output_size:
        gemm1_out = gemm1_out[:, :w1_output_size].contiguous()

    if activation_kind == MoEActivation.SILU:
        int_fp4, int_scale = ops.silu_and_mul_scaled_fp4_experts_quant(
            gemm1_out,
            a2_gscale_vec,
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
            a2_gscale_vec,
            expert_offsets,
            blockscale_offsets,
            top_k,
        )
    if w2_padding_cols > 0:
        int_fp4 = F.pad(int_fp4, (0, w2_padding_cols))

    gemm2_out = _grouped_matmul_nvfp4_packed(
        a_fp4=int_fp4,
        b_fp4=w2_weight,
        a_scale=int_scale,
        b_scale=w2_weight_scale,
        alpha=g2_alpha_vec,
        expert_offsets=expert_offsets,
        a_scale_offsets=blockscale_offsets,
        problem_sizes=problem_sizes2,
        output=workspace2,
    )
    if gemm2_out.shape[-1] != w2_output_size:
        gemm2_out = gemm2_out[:, :w2_output_size].contiguous()

    epilogue_topk_weights = (
        torch.ones_like(routed_topk_weights)
        if apply_router_weight_on_input
        else routed_topk_weights
    )
    inv_permuted_idx = c_map.view(num_tokens, top_k)
    # moe_unpermute reads from gemm2_out and writes to reduced. Source/destination
    # must never alias (particularly in non-chunked mode where workspaces are reused).
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
# MXFP4 (W4A16) grouped GEMM wrapper
# ---------------------------------------------------------------------------


def _grouped_matmul_mxfp4_packed(
    a_bf16: torch.Tensor,
    b_fp4: torch.Tensor,
    b_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    *,
    output: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed grouped MXFP4 W4A16 GEMM (BF16 activations, MXFP4 weights).

    Args:
        a_bf16: [M_total, K] bfloat16 activations in expert-packed row order.
        b_fp4: [E, N_max, K_packed] uint8 packed FP4 expert weights.
        b_scale: [E, N_pad, K_s] swizzled weight block scales (group 32).
        expert_offsets: [E] or [E+1] start row offsets in ``a_bf16``/output.
        problem_sizes: [E, 3] int tensor containing per-expert (M, N, K).
        bias: Optional [E, N] per-expert bias added after the GEMM.
        output: Pre-allocated output tensor (shape [>= M_total, >= N_max]).

    Returns:
        [M_total, N_max] tensor with grouped expert GEMM outputs.
    """
    assert a_bf16.ndim == 2 and b_fp4.ndim == 3
    assert a_bf16.dtype in (torch.bfloat16, torch.float16)
    assert b_fp4.dtype == torch.uint8
    assert b_scale.ndim == 3
    assert problem_sizes.ndim == 2 and problem_sizes.shape[1] == 3

    E = b_fp4.shape[0]
    if E == 0:
        M_logical = a_bf16.shape[0]
        N_out = b_fp4.shape[1]
        return output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    assert problem_sizes.shape[0] == E

    expert_offsets = _canonicalize_grouped_offsets(
        expert_offsets, num_experts=E, name="expert_offsets"
    )
    problem_sizes = problem_sizes.to(dtype=torch.int32).contiguous()
    if not a_bf16.is_contiguous():
        a_bf16 = a_bf16.contiguous()
    if not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    M_logical = a_bf16.shape[0]
    K = a_bf16.shape[1]

    max_tiles_m = triton.cdiv(M_logical, BLOCK_SIZE_M)
    max_tiles_n = triton.cdiv(b_fp4.shape[1], BLOCK_SIZE_N)
    max_tiles_per_expert = max_tiles_m * max_tiles_n

    N_out = b_fp4.shape[1]
    c_work = output.flatten()[: M_logical * N_out].view(M_logical, N_out)

    if max_tiles_per_expert == 0:
        return c_work

    A_LARGE = max(a_bf16.numel(), c_work.numel()) > 2**31
    B_LARGE = max(b_fp4.numel(), b_scale.numel()) > 2**31
    NUM_SMS = num_compute_units(a_bf16.device.index)
    worst_case_tiles = E * max_tiles_per_expert

    M_per_expert = problem_sizes[:, 0].to(torch.int32).clamp(min=0)
    N_per_expert = problem_sizes[:, 1].to(torch.int32).clamp(min=0)
    K_per_expert = problem_sizes[:, 2].to(torch.int32).clamp(min=0)
    valid = (M_per_expert > 0) & (N_per_expert > 0) & (K_per_expert > 0)
    tiles_per_expert = (
        torch.div(
            M_per_expert + BLOCK_SIZE_M - 1,
            BLOCK_SIZE_M,
            rounding_mode="floor",
        )
        * torch.div(
            N_per_expert + BLOCK_SIZE_N - 1,
            BLOCK_SIZE_N,
            rounding_mode="floor",
        )
        * valid.to(torch.int32)
    )
    expert_tile_start = torch.zeros(E + 1, dtype=torch.int32, device=a_bf16.device)
    expert_tile_start[1:] = torch.cumsum(tiles_per_expert, dim=0)

    b_fp4_flat = b_fp4.reshape(-1, b_fp4.shape[2])
    b_scale_flat = b_scale.reshape(-1, b_scale.shape[2])

    # Dummy tensors for unused A-scale and alpha kernel arguments.
    dummy_a_scale = torch.empty(0, device=a_bf16.device, dtype=torch.uint8)
    dummy_alpha = torch.empty(0, device=a_bf16.device, dtype=torch.float32)
    dummy_a_scale_offsets = expert_offsets

    has_bias = bias is not None
    bias_tensor = (
        bias
        if bias is not None
        else torch.empty(0, device=a_bf16.device, dtype=torch.float32)
    )

    # TMA tensor descriptors allocate host-side metadata; Triton requires an allocator.
    set_triton_allocator(a_bf16.device)

    grid = (min(NUM_SMS, worst_case_tiles),)
    _grouped_matmul_fp4_packed_persistent_kernel[grid](
        a_bf16,
        dummy_a_scale,
        b_fp4_flat,
        b_scale_flat,
        c_work,
        dummy_alpha,
        bias_tensor,
        expert_offsets,
        dummy_a_scale_offsets,
        problem_sizes,
        expert_tile_start,
        E,
        a_bf16.shape[0],
        b_fp4.shape[1],
        K,
        0,  # a_scale_rows_total (unused)
        problem_sizes.stride(0),
        problem_sizes.stride(1),
        a_bf16.stride(0),
        a_bf16.stride(1),
        0,  # stride_asm (unused)
        0,  # stride_ask (unused)
        b_fp4_flat.stride(0),
        b_fp4_flat.stride(1),
        b_scale_flat.stride(1),
        c_work.stride(0),
        c_work.stride(1),
        bias_tensor.stride(0) if has_bias else 0,
        0,  # a_scale_cols_total (unused)
        b_scale.shape[2],
        b_scale.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        NUM_SMS=NUM_SMS,
        A_LARGE=A_LARGE,
        B_LARGE=B_LARGE,
        A_IS_FP4=False,
        B_SCALE_GROUP=32,
        HAS_ALPHA=False,
        HAS_BIAS=has_bias,
        num_stages=2,
        num_warps=8,
    )
    return c_work


# ---------------------------------------------------------------------------
# Standalone deterministic MXFP4 W4A16 MoE implementation
# ---------------------------------------------------------------------------


def fused_moe_batch_invariant_mxfp4(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    activation: Any,
    *,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    output: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Deterministic MXFP4 W4A16 MoE using packed routing and grouped GEMMs.

    Activations stay in BF16 throughout (no FP4 activation quantization).
    After the activation, activations are copied from ``workspace2`` into
    ``workspace13`` so GEMM2 can read A from ``workspace13`` and write C into
    ``workspace2`` (avoiding overlap with ``fused_out`` in the modular kernel
    and avoiding A/C aliasing in a single buffer).
    """
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

    activation_kind = (
        activation
        if isinstance(activation, MoEActivation)
        else MoEActivation.from_str(str(activation))
    )

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]
    M_total = num_tokens * top_k

    if M_total == 0:
        return output

    routed_topk_ids = topk_ids
    if expert_map is not None:
        routed_topk_ids = _nvfp4_moe_map_experts(topk_ids, expert_map)
    routed_topk_ids = routed_topk_ids.to(torch.int32)
    valid_routes = (routed_topk_ids >= 0) & (routed_topk_ids < num_experts)
    routed_topk_ids = torch.where(
        valid_routes, routed_topk_ids, torch.full_like(routed_topk_ids, -1)
    )
    routed_topk_weights = topk_weights.to(torch.float32)

    if apply_router_weight_on_input:
        packed_hidden_states = hidden_states * routed_topk_weights.view(-1, 1).to(
            hidden_states.dtype
        )
    else:
        packed_hidden_states = hidden_states

    device = hidden_states.device
    w1_output_size = w13_weight.shape[1]
    activation_out_dim = (
        w1_output_size // 2 if activation_kind.is_gated else w1_output_size
    )
    w2_output_size = hidden_dim

    min_w13_cols = max(w1_output_size, hidden_dim)
    if workspace13.shape[0] < M_total or workspace13.shape[1] < min_w13_cols:
        raise RuntimeError(
            "workspace13 is too small for GEMM1 staging / GEMM2 A staging. "
            f"Need at least ({M_total}, {min_w13_cols}), "
            f"got {tuple(workspace13.shape)}."
        )
    required_workspace2_cols = max(activation_out_dim, w2_output_size)
    if workspace2.shape[0] < M_total or workspace2.shape[1] < required_workspace2_cols:
        raise RuntimeError(
            "workspace2 is too small for activation / GEMM2 output. "
            f"Need at least ({M_total}, {required_workspace2_cols}), "
            f"got {tuple(workspace2.shape)}."
        )

    expert_offsets = torch.empty((num_experts + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(
        (num_experts + 1), dtype=torch.int32, device=device
    )
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
    a_map = torch.zeros((M_total,), dtype=torch.int32, device=device)
    c_map = torch.empty((M_total,), dtype=torch.int32, device=device)
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

    # GEMM1: BF16 activations x MXFP4 weights
    gemm1_out = _grouped_matmul_mxfp4_packed(
        a_bf16=packed_hidden_states,
        b_fp4=w13_weight,
        b_scale=w13_weight_scale,
        expert_offsets=expert_offsets,
        problem_sizes=problem_sizes1,
        output=workspace13,
        bias=w13_bias,
    )
    if gemm1_out.shape[-1] != w1_output_size:
        gemm1_out = gemm1_out[:, :w1_output_size].contiguous()

    # Activation into workspace2; copy A to workspace13 so GEMM2 can write C
    # to workspace2 without overlapping A or the modular fused_out slab.
    act_out = _resize_cache(workspace2, (M_total, activation_out_dim))
    apply_moe_activation(
        activation=activation_kind,
        output=act_out,
        input=gemm1_out,
    )
    gemm2_a = _resize_cache(workspace13, (M_total, activation_out_dim))
    gemm2_a.copy_(act_out)

    # GEMM2: BF16 activation output x MXFP4 weights
    gemm2_out = _grouped_matmul_mxfp4_packed(
        a_bf16=gemm2_a,
        b_fp4=w2_weight,
        b_scale=w2_weight_scale,
        expert_offsets=expert_offsets,
        problem_sizes=problem_sizes2,
        output=workspace2,
        bias=w2_bias,
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
# Modular-kernel wrapper class
# ---------------------------------------------------------------------------


class BatchInvariantFP4Experts(mk.FusedMoEExpertsModular):
    """Deterministic, batch-invariant FP4 MoE expert implementation.

    Supports both NVFP4 (W4A4) and MXFP4 (W4A16) via per-expert Triton
    ``tl.dot_scaled`` GEMMs.  Dispatches to the appropriate wrapper based
    on ``quant_config.weight_quant_dtype``.

    **NVFP4 and expert parallel:** EP is not supported for the NVFP4 path
    (``ep_size`` must be 1). Expert maps mark non-local experts with ``-1``;
    packed MoE keeps a fixed ``(M * topk, K)`` activation tensor for CUDA graph
    capture, while only ``expert_offsets[-1]`` rows correspond to real assignments.
    The libtorch/stable NVFP4 expert quant kernels
    (``torch.ops._C.scaled_fp4_experts_quant`` / ``nvfp4_experts_quant.cu``) are
    tightly coupled to that layout. MXFP4 uses BF16 activations and does not use
    those kernels, so EP remains available for MXFP4.
    """

    _SUPPORTED_DTYPES = {"nvfp4", "mxfp4"}

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._quant_dtype = quant_config.weight_quant_dtype
        assert self._quant_dtype in self._SUPPORTED_DTYPES, (
            f"BatchInvariantFP4Experts supports {self._SUPPORTED_DTYPES}, "
            f"got {self._quant_dtype!r}"
        )
        self._num_local_experts = moe_config.num_local_experts
        self._cached_scale_vecs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

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
        return (weight_key, activation_key) in {
            (kNvfp4Static, kNvfp4Dynamic),
            (kMxfp4Static, None),
        }

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

    @staticmethod
    def is_supported_config(
        cls,
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        # NVFP4 + EP: expert_map maps non-local experts to -1. MoE shuffle keeps a
        # fixed (M * topk, K) buffer so CUDA graphs see stable tensor shapes; valid
        # packed rows are only expert_offsets[-1]. Padding rows must not be quantized
        # incorrectly (see nvfp4_experts_quant.cu). Combining EP, expert_map, graphs,
        # and libtorch NVFP4 expert quant is unsupported—require ep_size == 1.
        if (weight_key, activation_key) == (
            kNvfp4Static,
            kNvfp4Dynamic,
        ) and moe_config.moe_parallel_config.ep_size > 1:
            return (
                False,
                "kernel does not support expert parallel for NVFP4 batch-invariant "
                "MoE (expert_map / -1 routes, fixed (M*topk,K) activations for CUDA "
                "graphs vs expert_offsets[-1] packed rows; libtorch "
                "scaled_fp4_experts_quant). Use ep_size==1.",
            )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    def supports_expert_map(self) -> bool:
        return self._quant_dtype != "nvfp4"

    def supports_chunking(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _get_nvfp4_scale_vecs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cached_scale_vecs is not None:
            return self._cached_scale_vecs
        n = self._num_local_experts
        self._cached_scale_vecs = (
            _nvfp4_get_expert_vector(
                self.a1_gscale, num_experts=n, field_name="a1_gscale"
            ),
            _nvfp4_get_expert_vector(
                self.a2_gscale, num_experts=n, field_name="a2_gscale"
            ),
            _nvfp4_get_expert_vector(
                self.g1_alphas, num_experts=n, field_name="g1_alphas"
            ),
            _nvfp4_get_expert_vector(
                self.g2_alphas, num_experts=n, field_name="g2_alphas"
            ),
        )
        return self._cached_scale_vecs

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
        workspace13 = (M * topk, max(N, K))
        workspace2 = (M * topk, max(act_out_dim, K))
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
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        if self._quant_dtype == "nvfp4":
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
                apply_router_weight_on_input=bool(apply_router_weight_on_input),
                expert_map=expert_map,
            )
        else:
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
                w13_bias=self.w1_bias,
                w2_bias=self.w2_bias,
                apply_router_weight_on_input=bool(apply_router_weight_on_input),
                expert_map=expert_map,
            )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for batch-invariant FP4 MoE")


# Backward-compatible alias for existing imports.
BatchInvariantNvfp4Experts = BatchInvariantFP4Experts
