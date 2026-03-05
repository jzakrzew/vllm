# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant NVFP4 fused MoE expert implementation."""

import os
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
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_NVFP4_MOE_DEBUG = os.environ.get("VLLM_NVFP4_MOE_DEBUG", "0") != "0"


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
# Packed grouped NVFP4 GEMM kernel and wrapper
# ---------------------------------------------------------------------------


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

    Descriptors are constructed with expert-local shapes so that TMA's
    ``padding_option="zero"`` handles boundary conditions, avoiding manual
    masking in the inner loop.
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
    k_bytes_total = K_total // 2

    # A/A-scale descriptors use expert-local M so TMA zero-pads boundaries.
    a_desc = tl.make_tensor_descriptor(
        a_expert_ptr,
        shape=[M, k_bytes],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, K_BYTES],
        padding_option="zero",
    )
    # B weights are full-sized per expert.
    b_desc = tl.make_tensor_descriptor(
        b_expert_ptr,
        shape=[N_total, k_bytes_total],
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
        shape=[1, tl.cdiv(N_total, 128), b_scale_cols_total // 4, 2, 256],
        strides=[
            tl.cdiv(N_total, 128) * (b_scale_cols_total // 4) * 512 * stride_bsk,
            (b_scale_cols_total // 4) * 512 * stride_bsk,
            512 * stride_bsk,
            256 * stride_bsk,
            stride_bsk,
        ],
        block_shape=[1, SCALE_N_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )

    alpha = tl.load(alpha_ptr + expert_id).to(tl.float32)
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

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


def _validate_packed_nvfp4_descriptor_constraints(
    *,
    a_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_fp4: torch.Tensor,
    b_scale: torch.Tensor,
    c: torch.Tensor,
    expert_offsets: torch.Tensor,
    a_scale_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    block_size_m: int,
    block_size_n: int,
    block_size_k: int,
) -> None:
    """Host-side descriptor math validation for packed grouped NVFP4 GEMM.

    This mirrors the descriptor index arithmetic used in
    ``_grouped_matmul_nvfp4_packed_kernel`` so invalid launch metadata fails
    fast with a clear error instead of a GPU-side unspecified launch failure.
    """
    # Kernel compile-time constraints.
    if block_size_m % 128 != 0:
        raise RuntimeError("BLOCK_SIZE_M must be a multiple of 128.")
    if block_size_n % 128 != 0:
        raise RuntimeError("BLOCK_SIZE_N must be a multiple of 128.")
    if (block_size_k // 16) % 4 != 0:
        raise RuntimeError("BLOCK_SIZE_K//16 must be a multiple of 4.")

    k_bytes_total = a_fp4.shape[1]
    k_total = k_bytes_total * 2
    k_bytes_tile = block_size_k // 2
    scale_k_tile = block_size_k // 16
    scale_k_tiles = scale_k_tile // 4

    if k_total % 16 != 0:
        raise RuntimeError("Packed K must be divisible by 16 for NVFP4 block scales.")
    if a_scale.shape[1] % 4 != 0:
        raise RuntimeError("a_scale cols must be divisible by 4 for swizzled layout.")
    if b_scale.shape[2] % 4 != 0:
        raise RuntimeError("b_scale cols must be divisible by 4 for swizzled layout.")
    if not a_scale.is_contiguous():
        raise RuntimeError("a_scale must be contiguous for descriptor swizzle strides.")
    if not b_scale.is_contiguous():
        raise RuntimeError("b_scale must be contiguous for descriptor swizzle strides.")
    if a_scale.stride(1) != 1:
        raise RuntimeError("a_scale must be contiguous in the last dimension.")
    if b_scale.stride(2) != 1:
        raise RuntimeError("b_scale must be contiguous in the last dimension.")
    if b_scale.stride(1) != b_scale.shape[2]:
        raise RuntimeError(
            "b_scale row stride must match contiguous [N, K_scale] layout."
        )
    if a_scale_offsets.numel() > 0 and torch.any(a_scale_offsets % 128 != 0):
        raise RuntimeError("a_scale_offsets must be 128-row aligned.")
    if expert_offsets.numel() > 1 and torch.any(
        expert_offsets[1:] < expert_offsets[:-1]
    ):
        raise RuntimeError("expert_offsets must be non-decreasing.")
    if a_scale_offsets.numel() > 1 and torch.any(
        a_scale_offsets[1:] < a_scale_offsets[:-1]
    ):
        raise RuntimeError("a_scale_offsets must be non-decreasing.")

    for name, tensor in (
        ("a_fp4", a_fp4),
        ("a_scale", a_scale),
        ("b_fp4", b_fp4),
        ("b_scale", b_scale),
        ("c", c),
    ):
        if tensor.data_ptr() % 16 != 0:
            raise RuntimeError(f"{name} base pointer must be 16-byte aligned.")

    m_total = a_fp4.shape[0]
    n_total = b_fp4.shape[1]
    a_scale_rows_total = a_scale.shape[0]
    a_scale_k_tiles_total = a_scale.shape[1] // 4
    b_scale_k_tiles_total = b_scale.shape[2] // 4
    a_scale_row_tiles_total = triton.cdiv(a_scale_rows_total, 128)
    b_scale_row_tiles_total = triton.cdiv(n_total, 128)

    m = problem_sizes[:, 0].to(torch.int64).clamp_min(0)
    n = problem_sizes[:, 1].to(torch.int64).clamp_min(0)
    k = problem_sizes[:, 2].to(torch.int64).clamp_min(0)
    if torch.any(problem_sizes[:, 0] < 0) or torch.any(problem_sizes[:, 1] < 0):
        raise RuntimeError("problem_sizes M/N must be non-negative.")
    if torch.any(problem_sizes[:, 2] < 0) or torch.any(problem_sizes[:, 2] % 2 != 0):
        raise RuntimeError("problem_sizes K must be non-negative and even.")

    m_tiles = torch.div(m + block_size_m - 1, block_size_m, rounding_mode="floor")
    n_tiles = torch.div(n + block_size_n - 1, block_size_n, rounding_mode="floor")
    k_tiles = torch.div(k + block_size_k - 1, block_size_k, rounding_mode="floor")

    start_m_max = (m_tiles - 1).clamp_min(0) * block_size_m
    start_n_max = (n_tiles - 1).clamp_min(0) * block_size_n
    k_start_bytes_max = (k_tiles - 1).clamp_min(0) * k_bytes_tile
    scale_tile_m_max = torch.div(start_m_max, 128, rounding_mode="floor")
    scale_tile_n_max = torch.div(start_n_max, 128, rounding_mode="floor")
    scale_tile_k_max = (k_tiles - 1).clamp_min(0) * scale_k_tiles

    # Descriptor-relative index checks.
    if torch.any(start_m_max >= m_total):
        raise RuntimeError("Packed A descriptor M dimension is too small.")
    if torch.any(start_n_max >= n_total):
        raise RuntimeError("Packed B descriptor N dimension is too small.")
    if torch.any(k_start_bytes_max >= k_bytes_total):
        raise RuntimeError("Packed A/B descriptor K-bytes dimension is too small.")
    if torch.any(scale_tile_m_max >= a_scale_row_tiles_total):
        raise RuntimeError("Packed A scale descriptor row-tile dimension is too small.")
    if torch.any(scale_tile_n_max >= b_scale_row_tiles_total):
        raise RuntimeError("Packed B scale descriptor row-tile dimension is too small.")
    if torch.any(scale_tile_k_max >= a_scale_k_tiles_total):
        raise RuntimeError("Packed A scale descriptor K-tile dimension is too small.")
    if torch.any(scale_tile_k_max >= b_scale_k_tiles_total):
        raise RuntimeError("Packed B scale descriptor K-tile dimension is too small.")

    # Base-pointer-offset + block-footprint checks.
    max_a_row = expert_offsets.to(torch.int64) + start_m_max + (block_size_m - 1)
    max_as_row = a_scale_offsets.to(torch.int64) + start_m_max + (block_size_m - 1)
    if torch.any(max_a_row >= m_total):
        raise RuntimeError(
            "A/C descriptor base offset + tile footprint exceeds packed rows."
        )
    if torch.any(max_as_row >= a_scale_rows_total):
        raise RuntimeError(
            "A scale descriptor base offset + tile footprint exceeds packed scale rows."
        )
    if c.shape[0] < m_total or c.shape[1] < n_total:
        raise RuntimeError("Output tensor does not match descriptor logical extents.")


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
    output: torch.Tensor | None = None,
    pre_padded: bool = False,
) -> torch.Tensor:
    """Packed grouped NVFP4 GEMM with expert-local problem metadata.

    Args:
        a_fp4: [M_total, K_packed] (or [M_total + PAD, K_packed] if
            ``pre_padded``) uint8 packed FP4 activations in expert-packed
            row order.
        b_fp4: [E, N_max, K_packed] uint8 packed FP4 expert weights.
        a_scale: [S_total, K_s] (or [S_total + PAD, K_s] if ``pre_padded``)
            float8_e4m3fn swizzled activation block scales.
        b_scale: [E, N_pad, K_s] float8_e4m3fn swizzled weight block scales.
        alpha: [E] float32 per-expert alpha.
        expert_offsets: [E] or [E+1] start row offsets in ``a_fp4``/output.
        problem_sizes: [E, 3] int tensor containing per-expert (M, N, K).
        output_dtype: output dtype.
        a_scale_offsets: Optional [E] or [E+1] start row offsets in ``a_scale``.
            If not provided, ``expert_offsets`` are reused.
        output: Optional pre-allocated [M_total + PAD_ROWS, N_max] tensor for
            the GEMM output including guard rows.  When provided the function
            writes into it (avoiding a dynamic allocation) and returns a view
            of the first ``M_total`` rows.
        pre_padded: If True, ``a_fp4`` and ``a_scale`` already include
            PAD_ROWS guard rows at the end, so the internal F.pad is skipped.

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
    if not a_fp4.is_contiguous():
        a_fp4 = a_fp4.contiguous()
    if not a_scale.is_contiguous():
        a_scale = a_scale.contiguous()
    if not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()

    _is_capturing = (
        torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
    )

    if _NVFP4_MOE_DEBUG and not _is_capturing:
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
        if int(problem_sizes[:, 1].max().item()) > b_fp4.shape[1]:
            raise RuntimeError(
                "problem_sizes N exceeds packed expert-weight row dimension."
            )
        if int(problem_sizes[:, 2].max().item()) > a_fp4.shape[1] * 2:
            raise RuntimeError("problem_sizes K exceeds packed activation K dimension.")

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 256
    PAD_ROWS = BLOCK_SIZE_M

    if pre_padded:
        a_fp4_work = a_fp4
        a_scale_work = a_scale
        M_logical = a_fp4.shape[0] - PAD_ROWS
    else:
        M_logical = a_fp4.shape[0]
        a_fp4_work = torch.nn.functional.pad(a_fp4, (0, 0, 0, PAD_ROWS)).contiguous()
        a_scale_work = torch.nn.functional.pad(
            a_scale, (0, 0, 0, PAD_ROWS)
        ).contiguous()

    # Worst-case grid from static tensor shapes so the grid is constant
    # across calls -- required for CUDA graph compatibility.  The kernel
    # early-exits for tiles beyond the actual per-expert problem size.
    max_tiles_m = triton.cdiv(a_fp4_work.shape[0], BLOCK_SIZE_M)
    max_tiles_n = triton.cdiv(b_fp4.shape[1], BLOCK_SIZE_N)
    max_tiles_per_expert = max_tiles_m * max_tiles_n

    M_padded = a_fp4_work.shape[0]
    N_out = b_fp4.shape[1]
    if output is not None:
        assert output.shape[0] >= M_padded and output.shape[1] >= N_out
        c_work = output.flatten()[: M_padded * N_out].view(M_padded, N_out)
    else:
        c_work = torch.empty(
            (M_padded, N_out),
            device=a_fp4.device,
            dtype=output_dtype,
        )
    c_work[M_logical:].zero_()

    if _NVFP4_MOE_DEBUG and not _is_capturing:
        _validate_packed_nvfp4_descriptor_constraints(
            a_fp4=a_fp4_work,
            a_scale=a_scale_work,
            b_fp4=b_fp4,
            b_scale=b_scale,
            c=c_work,
            expert_offsets=expert_offsets,
            a_scale_offsets=a_scale_offsets,
            problem_sizes=problem_sizes,
            block_size_m=BLOCK_SIZE_M,
            block_size_n=BLOCK_SIZE_N,
            block_size_k=BLOCK_SIZE_K,
        )

    if max_tiles_per_expert == 0:
        return c_work[:M_logical]

    grid = (E, max_tiles_per_expert)
    _grouped_matmul_nvfp4_packed_kernel[grid](
        a_fp4_work,
        a_scale_work,
        b_fp4,
        b_scale,
        c_work,
        alpha,
        expert_offsets,
        a_scale_offsets,
        problem_sizes,
        a_fp4_work.shape[0],
        b_fp4.shape[1],
        a_fp4_work.shape[1] * 2,
        a_scale_work.shape[0],
        problem_sizes.stride(0),
        problem_sizes.stride(1),
        a_fp4_work.stride(0),
        a_fp4_work.stride(1),
        a_scale_work.stride(0),
        a_scale_work.stride(1),
        b_fp4.stride(0),
        b_scale.stride(0),
        b_fp4.stride(1),
        b_fp4.stride(2),
        b_scale.stride(2),
        c_work.stride(0),
        c_work.stride(1),
        a_scale_work.shape[1],
        b_scale.shape[2],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        A_LARGE=max(a_fp4_work.numel(), a_scale_work.numel(), c_work.numel()) > 2**31,
        B_LARGE=max(b_fp4.numel(), b_scale.numel()) > 2**31,
        num_stages=2,
        num_warps=8,
    )
    return c_work[:M_logical]


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
    workspace13: torch.Tensor | None = None,
    workspace2: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Deterministic NVFP4 MoE using packed routing metadata and grouped GEMMs.

    The input ``[M, K]`` is expanded and permuted into expert-major packed
    order ``[M * topk, K]`` via ``shuffle_rows`` with CUTLASS metadata
    (``a_map``/``c_map``).  Both GEMMs then run in packed 2D form using
    per-expert offsets/problem sizes.  The epilogue uses ``moe_unpermute``
    to gather, apply router weights and reduce back to ``[M, K]`` while
    safely skipping invalid routes.
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
    )
    routed_topk_weights = topk_weights.to(torch.float32)

    if apply_router_weight_on_input:
        packed_hidden_states = hidden_states * routed_topk_weights.view(-1, 1).to(
            hidden_states.dtype
        )
    else:
        packed_hidden_states = hidden_states

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

    _is_capturing = (
        torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
    )

    if _is_capturing:
        # During CUDA graph capture we cannot transfer data to the host.
        # Use the worst-case row count; the kernels below already rely on
        # expert_offsets / problem_sizes for actual boundaries.
        valid_rows = M_total
    else:
        valid_rows = int(expert_offsets[-1].item())
        if valid_rows == 0:
            if workspace2 is not None:
                reduced = _resize_cache(workspace2, (num_tokens, hidden_dim))
                reduced.zero_()
            else:
                reduced = torch.zeros(
                    (num_tokens, hidden_dim), device=device, dtype=dtype
                )
            return reduced

    # `get_cutlass_moe_mm_data()` only defines valid source rows in the first
    # `expert_offsets[-1]` positions of `a_map`. Keep quant/GEMM on this packed
    # prefix so invalid routes never enter expert quantization.
    packed_hidden_states = ops.shuffle_rows(packed_hidden_states, a_map[:valid_rows])

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

    _PAD = 128

    def _pad_for_gemm(
        fp4: torch.Tensor, scale: torch.Tensor, col_pad: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if col_pad > 0:
            fp4 = F.pad(fp4, (0, col_pad))
        fp4 = F.pad(fp4, (0, 0, 0, _PAD)).contiguous()
        scale = F.pad(scale, (0, 0, 0, _PAD)).contiguous()
        return fp4, scale

    a1_fp4, a1_scale = ops.scaled_fp4_experts_quant(
        packed_hidden_states,
        a1_gscale_vec,
        expert_offsets,
        blockscale_offsets,
        top_k,
    )
    a1_fp4, a1_scale = _pad_for_gemm(a1_fp4, a1_scale, w1_padding_cols)

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
        output=workspace13,
        pre_padded=True,
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
        if workspace2 is not None:
            act_out = _resize_cache(workspace2, (valid_rows, activation_out_dim))
        else:
            act_out = torch.empty(
                (valid_rows, activation_out_dim), device=device, dtype=dtype
            )
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
    int_fp4, int_scale = _pad_for_gemm(int_fp4, int_scale, w2_padding_cols)

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
        output=workspace13,
        pre_padded=True,
    )
    if gemm2_out.shape[-1] != w2_output_size:
        gemm2_out = gemm2_out[:, :w2_output_size].contiguous()

    epilogue_topk_weights = (
        torch.ones_like(routed_topk_weights)
        if apply_router_weight_on_input
        else routed_topk_weights
    )
    inv_permuted_idx = c_map.view(num_tokens, top_k)
    if workspace2 is not None:
        reduced = _resize_cache(workspace2, (num_tokens, hidden_dim))
    else:
        reduced = torch.empty((num_tokens, hidden_dim), device=device, dtype=dtype)
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

    def _get_scale_vecs(
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

    _PAD_ROWS = 128

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
        workspace13 = (M * topk + self._PAD_ROWS, max(N, K))
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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        a1_gscale_vec, a2_gscale_vec, g1_alpha_vec, g2_alpha_vec = (
            self._get_scale_vecs()
        )
        result = fused_moe_batch_invariant_nvfp4(
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
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            expert_map=expert_map,
            workspace13=workspace13,
            workspace2=workspace2,
        )
        output.copy_(result)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for batch-invariant NVFP4 MoE")
