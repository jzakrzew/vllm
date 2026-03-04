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


@triton.jit
def _grouped_matmul_nvfp4_packed_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    alpha_ptr,
    expert_offsets_ptr,
    a_scale_offsets_ptr,
    problem_sizes_ptr,
    stride_ps0,
    stride_ps1,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_eb,
    stride_ebs,
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
    """Packed grouped NVFP4 GEMM with per-expert offsets/problem sizes.

    Grid: ``(E, max_tiles_per_expert)``. Each program handles one potential
    (expert, tile) pair and exits early if the tile index is outside the
    expert's logical problem shape.
    """
    expert_id = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)

    p_ptr = problem_sizes_ptr + expert_id * stride_ps0
    M = tl.load(p_ptr + 0 * stride_ps1).to(tl.int32)
    N = tl.load(p_ptr + 1 * stride_ps1).to(tl.int32)
    K = tl.load(p_ptr + 2 * stride_ps1).to(tl.int32)
    if M <= 0 or N <= 0 or K <= 0:
        return

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

    expert_row_offset = tl.load(expert_offsets_ptr + expert_id).to(tl.int32)
    expert_scale_offset = tl.load(a_scale_offsets_ptr + expert_id).to(tl.int32)
    if A_LARGE:
        expert_row_offset = expert_row_offset.to(tl.int64)
        expert_scale_offset = expert_scale_offset.to(tl.int64)

    a_expert_ptr = a_ptr + expert_row_offset * stride_am
    a_scale_expert_ptr = a_scale_ptr + expert_scale_offset * stride_asm
    b_expert_ptr = b_ptr + expert_id * stride_eb
    b_scale_expert_ptr = b_scale_ptr + expert_id * stride_ebs
    c_expert_ptr = c_ptr + expert_row_offset * stride_cm

    k_bytes = K // 2
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
    output_dtype: torch.dtype,
    *,
    a_scale_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed grouped NVFP4 GEMM with expert-local problem metadata.

    Args:
        a_fp4: [M_total, K_packed] uint8 packed FP4 activations in expert-packed
            row order.
        b_fp4: [E, N_max, K_packed] uint8 packed FP4 expert weights.
        a_scale: [S_total, K_s] float8_e4m3fn swizzled activation block scales.
            For expert-wise quantization, S_total is typically the sum of
            per-expert padded scale-row regions.
        b_scale: [E, N_pad, K_s] float8_e4m3fn swizzled weight block scales.
        alpha: [E] float32 per-expert alpha.
        expert_offsets: [E] or [E+1] start row offsets in ``a_fp4``/output.
        problem_sizes: [E, 3] int tensor containing per-expert (M, N, K).
        output_dtype: output dtype.
        a_scale_offsets: Optional [E] or [E+1] start row offsets in ``a_scale``.
            If not provided, ``expert_offsets`` are reused.

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
        return torch.empty(
            (a_fp4.shape[0], b_fp4.shape[1]),
            device=a_fp4.device,
            dtype=output_dtype,
        )

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
    if not a_scale.is_contiguous():
        a_scale = a_scale.contiguous()
    if not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()

    if torch.any(expert_offsets < 0) or torch.any(a_scale_offsets < 0):
        raise RuntimeError("expert offsets must be non-negative.")
    max_row_end = (
        expert_offsets.to(torch.int64)
        + problem_sizes[:, 0].to(torch.int64).clamp_min(0)
    ).max()
    if int(max_row_end.item()) > a_fp4.shape[0]:
        raise RuntimeError(
            "expert_offsets + M from problem_sizes exceed packed activation rows."
        )
    max_scale_row_end = (
        a_scale_offsets.to(torch.int64)
        + problem_sizes[:, 0].to(torch.int64).clamp_min(0)
    ).max()
    if int(max_scale_row_end.item()) > a_scale.shape[0]:
        raise RuntimeError(
            "a_scale_offsets + M from problem_sizes exceed packed scale rows."
        )

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 256

    m_tiles = torch.div(
        problem_sizes[:, 0].clamp_min(0) + BLOCK_SIZE_M - 1,
        BLOCK_SIZE_M,
        rounding_mode="floor",
    )
    n_tiles = torch.div(
        problem_sizes[:, 1].clamp_min(0) + BLOCK_SIZE_N - 1,
        BLOCK_SIZE_N,
        rounding_mode="floor",
    )
    max_tiles_per_expert = int((m_tiles * n_tiles).max().item())

    c = torch.zeros(
        (a_fp4.shape[0], b_fp4.shape[1]),
        device=a_fp4.device,
        dtype=output_dtype,
    )
    if max_tiles_per_expert == 0:
        return c

    grid = (E, max_tiles_per_expert)
    _grouped_matmul_nvfp4_packed_kernel[grid](
        a_fp4,
        a_scale,
        b_fp4,
        b_scale,
        c,
        alpha,
        expert_offsets,
        a_scale_offsets,
        problem_sizes,
        problem_sizes.stride(0),
        problem_sizes.stride(1),
        a_fp4.stride(0),
        a_fp4.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_fp4.stride(0),
        b_scale.stride(0),
        b_fp4.stride(1),
        b_fp4.stride(2),
        b_scale.stride(2),
        c.stride(0),
        c.stride(1),
        a_scale.shape[1],
        b_scale.shape[2],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        A_LARGE=max(a_fp4.numel(), a_scale.numel(), c.numel()) > 2**31,
        B_LARGE=max(b_fp4.numel(), b_scale.numel()) > 2**31,
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
    Deterministic NVFP4 MoE using packed routing metadata and grouped GEMMs.

    The input is expanded to ``[M * topk, K]`` and permuted into expert-major
    packed order via CUTLASS metadata (`a_map`/`c_map`). Both GEMMs then run
    in packed 2D form using per-expert offsets/problem sizes. The epilogue
    uses ``moe_unpermute`` to gather, apply router weights and reduce back to
    ``[M, K]`` while safely skipping invalid routes.
    """
    import torch.nn.functional as F

    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import (
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        moe_unpermute,
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
    if quant_backend != "cutlass":
        raise RuntimeError(
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
        return hidden_states.new_empty((num_tokens, hidden_dim))

    routed_topk_ids = topk_ids
    if expert_map is not None:
        routed_topk_ids = _nvfp4_moe_map_experts(topk_ids, expert_map)
    routed_topk_ids = routed_topk_ids.to(torch.int32)
    # Out-of-range IDs are treated as invalid routes.
    valid_routes = (routed_topk_ids >= 0) & (routed_topk_ids < num_experts)
    routed_topk_ids = torch.where(
        valid_routes, routed_topk_ids, torch.full_like(routed_topk_ids, -1)
    ).contiguous()
    routed_topk_weights = topk_weights.to(torch.float32).contiguous()

    packed_hidden_states = (
        hidden_states.unsqueeze(1)
        .expand(num_tokens, top_k, hidden_dim)
        .reshape(M_total, hidden_dim)
        .contiguous()
    )
    if apply_router_weight_on_input:
        packed_hidden_states.mul_(
            routed_topk_weights.view(-1, 1).to(packed_hidden_states.dtype)
        )

    device = hidden_states.device
    dtype = hidden_states.dtype
    w1_output_size = w13_weight.shape[1]
    activation_out_dim = (
        w1_output_size // 2 if activation_kind.is_gated else w1_output_size
    )
    w2_output_size = hidden_dim
    w1_padding_cols = max(0, w13_weight.shape[-1] - hidden_dim // 2)
    w2_padding_cols = max(0, w2_weight.shape[-1] - activation_out_dim // 2)

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

    a1_gscale_vec = _nvfp4_get_expert_vector(
        a1_gscale, num_experts=num_experts, field_name="a1_gscale"
    )
    a2_gscale_vec = _nvfp4_get_expert_vector(
        a2_gscale, num_experts=num_experts, field_name="a2_gscale"
    )
    g1_alpha_vec = _nvfp4_get_expert_vector(
        g1_alphas, num_experts=num_experts, field_name="g1_alphas"
    )
    g2_alpha_vec = _nvfp4_get_expert_vector(
        g2_alphas, num_experts=num_experts, field_name="g2_alphas"
    )

    a1_fp4, a1_scale = ops.scaled_fp4_experts_quant(
        packed_hidden_states,
        a1_gscale_vec,
        expert_offsets,
        blockscale_offsets,
        top_k,
    )
    if w1_padding_cols > 0:
        a1_fp4 = F.pad(a1_fp4, (0, w1_padding_cols)).contiguous()

    gemm1_out = _grouped_matmul_nvfp4_packed(
        a_fp4=a1_fp4,
        b_fp4=w13_weight,
        a_scale=a1_scale,
        b_scale=w13_weight_scale,
        alpha=g1_alpha_vec,
        expert_offsets=expert_offsets,
        a_scale_offsets=blockscale_offsets,
        problem_sizes=problem_sizes1,
        output_dtype=dtype,
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
        act_out = torch.empty((M_total, activation_out_dim), device=device, dtype=dtype)
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
        int_fp4 = F.pad(int_fp4, (0, w2_padding_cols)).contiguous()

    gemm2_out = _grouped_matmul_nvfp4_packed(
        a_fp4=int_fp4,
        b_fp4=w2_weight,
        a_scale=int_scale,
        b_scale=w2_weight_scale,
        alpha=g2_alpha_vec,
        expert_offsets=expert_offsets,
        a_scale_offsets=blockscale_offsets,
        problem_sizes=problem_sizes2,
        output_dtype=dtype,
    )
    if gemm2_out.shape[-1] != w2_output_size:
        gemm2_out = gemm2_out[:, :w2_output_size].contiguous()

    epilogue_topk_weights = (
        torch.ones_like(routed_topk_weights)
        if apply_router_weight_on_input
        else routed_topk_weights
    )
    inv_permuted_idx = c_map.view(num_tokens, top_k)
    reduced = torch.empty((num_tokens, hidden_dim), device=device, dtype=dtype)
    moe_unpermute(
        out=reduced,
        permuted_hidden_states=gemm2_out,
        topk_weights=epilogue_topk_weights,
        inv_permuted_idx=inv_permuted_idx,
        # The last entry stores num_valid_routes. Invalid routes in c_map are
        # set to this sentinel and are skipped by moe_unpermute.
        expert_first_token_offset=expert_offsets.to(torch.int64),
    )
    return reduced


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
        return True

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
        workspace13 = (M * topk, max(N, K))
        workspace2 = (M * topk, self.adjust_N_for_activation(N, activation))
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
