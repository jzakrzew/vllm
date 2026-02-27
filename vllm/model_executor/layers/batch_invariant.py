# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_torch_equal_or_newer
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, "
            f"tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, (
        "Currently assuming bias is 1D, let Horace know if you run into this"
    )
    NUM_SMS = num_compute_units(a.device.index)
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    # print(a.device, b.device, c.device)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_nvfp4_kernel_persistent(
    a_ptr,  # (M, K // 2), packed fp4
    a_scale_ptr,  # (M, K // 16), fp8 block scales
    b_ptr,  # (N, K // 2), packed fp4
    b_scale_ptr,  # (N, K // 16), fp8 block scales
    c_ptr,  # (M, N), output
    alpha_ptr,  # scalar fp32 global alpha
    M,
    N,
    K,  # K in fp4 elements
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bm,
    stride_bk,
    stride_bsm,
    stride_bsk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    k_byte_offsets_for_mask = tl.arange(0, BLOCK_SIZE_K // 2)
    k_scale_offsets_for_mask = tl.arange(0, BLOCK_SIZE_K // 16)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N

        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        mask_m = offs_am < M
        mask_n = offs_bn < N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k_bytes = (
                    ki * (BLOCK_SIZE_K // 2) + tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64)
                )
                offs_k_scales = (
                    ki * (BLOCK_SIZE_K // 16)
                    + tl.arange(0, BLOCK_SIZE_K // 16).to(tl.int64)
                )
            else:
                offs_k_bytes = ki * (BLOCK_SIZE_K // 2) + tl.arange(0, BLOCK_SIZE_K // 2)
                offs_k_scales = ki * (BLOCK_SIZE_K // 16) + tl.arange(0, BLOCK_SIZE_K // 16)

            a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k_bytes[None, :] * stride_ak
            b_ptrs = b_ptr + offs_bn[:, None] * stride_bm + offs_k_bytes[None, :] * stride_bk

            a_scale_ptrs = (
                a_scale_ptr
                + offs_am[:, None] * stride_asm
                + offs_k_scales[None, :] * stride_ask
            )
            b_scale_ptrs = (
                b_scale_ptr
                + offs_bn[:, None] * stride_bsm
                + offs_k_scales[None, :] * stride_bsk
            )

            remaining_k = K - ki * BLOCK_SIZE_K
            remaining_k_bytes = tl.cdiv(remaining_k, 2)
            remaining_k_scales = tl.cdiv(remaining_k, 16)

            k_mask_bytes = k_byte_offsets_for_mask < remaining_k_bytes
            k_mask_scales = k_scale_offsets_for_mask < remaining_k_scales

            a_mask = mask_m[:, None] & k_mask_bytes[None, :]
            b_mask = mask_n[:, None] & k_mask_bytes[None, :]
            a_scale_mask = mask_m[:, None] & k_mask_scales[None, :]
            b_scale_mask = mask_n[:, None] & k_mask_scales[None, :]

            # LHS fp4 is packed along K, RHS is loaded as (N, K//2) and transposed.
            a = tl.load(a_ptrs, mask=a_mask, other=0)
            b = tl.load(b_ptrs, mask=b_mask, other=0).T
            a_scale = tl.load(a_scale_ptrs, mask=a_scale_mask, other=0.0)
            b_scale = tl.load(b_scale_ptrs, mask=b_scale_mask, other=0.0)

            accumulator = tl.dot_scaled(
                a,
                a_scale,
                "e2m1",
                b,
                b_scale,
                "e2m1",
                accumulator,
            )

        alpha = tl.load(alpha_ptr).to(tl.float32)
        accumulator *= alpha

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_nvfp4_persistent(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    assert a_fp4.dtype == torch.uint8 and b_fp4.dtype == torch.uint8, (
        "Expected packed FP4 tensors in uint8 format."
    )
    assert a_scale.dtype == torch.float8_e4m3fn and b_scale.dtype == torch.float8_e4m3fn, (
        "Expected FP8 E4M3 block scales for NVFP4."
    )
    assert alpha.dtype == torch.float32, "Expected alpha to be float32."
    assert a_fp4.shape[1] == b_fp4.shape[1], "Incompatible packed K dimensions."

    M = a_fp4.shape[0]
    N = b_fp4.shape[0]
    K = a_fp4.shape[1] * 2
    NUM_SMS = num_compute_units(a_fp4.device.index)

    c = torch.empty((M, N), device=a_fp4.device, dtype=output_dtype)

    def grid(meta):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, meta["BLOCK_SIZE_M"])
                * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            ),
        )

    matmul_nvfp4_kernel_persistent[grid](
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
        a_fp4.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_fp4.stride(0),
        b_fp4.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a_fp4.numel() > 2**31,
        B_LARGE=b_fp4.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=256,
        GROUP_SIZE_M=8,
        num_stages=4,
        num_warps=8,
    )
    return c


@triton.jit
def bmm_kernel(
    a_ptr,  # (*, ) pointer to A, (B, M, K)
    b_ptr,  # (*, ) pointer to B, (B, K, N)
    c_ptr,  # (*, ) pointer to C, (B, M, N)
    B,  # int, batch size
    M,  # int, output rows
    N,  # int, output cols
    K,  # int, reduction dim
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N)

    Each program computes one (batch_idx, tile_m, tile_n) tile, accumulating
    along K in a fixed order to preserve batch invariance.
    """
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    if pid_b >= B:
        return

    # number of tiles along M / N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    # offs_m / offs_n: raw global row/col indices for this tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # masks for valid logical rows/cols within (M, N)
    mask_m = offs_m < M  # [BLOCK_SIZE_M]
    mask_n = offs_n < N  # [BLOCK_SIZE_N]

    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # hint for triton contiguous memory
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # base pointers for current batch, shape-wise:
    #   a_batch_ptr points to A[pid_b, 0, 0]
    #   b_batch_ptr points to B[pid_b, 0, 0]
    #   c_batch_ptr points to C[pid_b, 0, 0]
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # number of K-blocks this tile iterates over
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            # offs_k: [BLOCK_SIZE_K], global K indices
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        #   element (i, j) points to A[pid_b, offs_m[i], offs_k[j]]
        a_ptrs = a_batch_ptr + (
            offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        #   element (i, j) points to B[pid_b, offs_k[i], offs_n[j]]
        b_ptrs = b_batch_ptr + (
            offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        # valid K lanes for this block
        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        # A mask within (M, K): [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask = mask_m[:, None] & k_valid[None, :]
        # B mask within (K, N): [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask = k_valid[:, None] & mask_n[None, :]

        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] from A[offs_m, offs_k]
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] from B[offs_k, offs_n]
        b = tl.load(
            b_ptrs,
            mask=b_mask,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    # c_m / c_n: [BLOCK_SIZE_M] / [BLOCK_SIZE_N], row/col indices for C
    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)

    # c_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #   element (i, j) points to C[pid_b, c_m[i], c_n[j]]
    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    # mask out elements that fall outside logical (M, N) range
    c_mask = mask_m[:, None] & mask_n[None, :]
    # cast FP32 accumulator back to original dtype of C
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax
             (only -1 or last dim supported)
    >> Stashed changes
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype
               (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert -input.ndim <= dim < input.ndim, (
        f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    )

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    output_2d = output.reshape(M, 1, K).squeeze(1) if keepdim else output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def matmul_batch_invariant(a, b, *, out=None):
    # torch.matmul can handle various dimensions
    # For 2D x 2D, it's the same as mm
    if a.ndim == 2 and b.ndim == 2:
        result = matmul_persistent(a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 3 and b.ndim == 3:
        # Handle batched case like bmm
        return bmm_batch_invariant(a, b, out=out)
    elif a.ndim == 3 and b.ndim == 2:
        # Handle 3D x 2D: common for linear layers
        # (batch, seq, hidden) @ (hidden, out) -> (batch, seq, out)
        # Reshape to 2D, do mm, reshape back
        batch, seq, hidden = a.shape
        a_2d = a.reshape(-1, hidden)
        result_2d = matmul_persistent(a_2d, b)
        result = result_2d.reshape(batch, seq, -1)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim == 2 and b.ndim == 3:
        # Handle 2D x 3D: (M, K) @ (B, K, N) -> (B, M, N)
        # By broadcasting `a` to 3D, we can reuse the batched matrix
        # multiplication logic.
        a_expanded = a.unsqueeze(0).expand(b.shape[0], -1, -1)
        return bmm_batch_invariant(a_expanded, b, out=out)
    elif a.ndim == 4 and b.ndim == 4:
        # Handle 4D attention tensors: [batch, heads, seq, dim]
        # Reshape to 3D, process, reshape back
        batch, heads, seq_a, dim_a = a.shape
        _, _, dim_b, seq_b = b.shape

        # Reshape to [batch*heads, seq_a, dim_a]
        a_3d = a.reshape(batch * heads, seq_a, dim_a)
        b_3d = b.reshape(batch * heads, dim_b, seq_b)

        # Do batched matmul
        result_3d = bmm_batch_invariant(a_3d, b_3d)

        # Reshape back to [batch, heads, seq_a, seq_b]
        result = result_3d.reshape(batch, heads, seq_a, seq_b)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant currently only supports 2D x 2D, 3D x 3D, "
            f"3D x 2D, 2D x 3D, and 4D x 4D, "
            f"got shapes {a.shape} and {b.shape}"
        )


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    if not (a.ndim == 3 and b.ndim == 3):
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Batch dimensions of tensors must match, "
            f"but got {a.shape[0]} and {b.shape[0]}."
        )
    if a.shape[2] != b.shape[1]:
        raise ValueError(
            f"Incompatible inner dimensions for matmul: got {a.shape} and {b.shape}."
        )
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: got {a.dtype} and {b.dtype}.")

    B, M, K = a.shape
    _, _, N = b.shape
    dtype = a.dtype

    if out is None:
        c = torch.empty((B, M, N), device=a.device, dtype=dtype)
    else:
        assert out.shape == (B, M, N), "out tensor has incorrect shape"
        assert out.dtype == dtype and out.device == a.device, "out tensor mismatch"
        c = out

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    cfg = configs[dtype]
    # grid = (B, num_tiles_per_matrix)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )

    bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )

    return c


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def softmax_batch_invariant(input, dim, dtype=None):
    # Compute softmax in a deterministic way
    # First subtract max for numerical stability (standard practice)
    input_max = torch.amax(input, dim=dim, keepdim=True)
    input = input - input_max
    exp_x = torch.exp(input)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"

    result = input.to(torch.float32)

    if len(dim) == 0:
        dim = [i for i in range(len(input.shape))]

    # Sort dimensions to reduce from largest to smallest to handle shifting dims
    # during iterative reduction.
    sorted_dims = sorted([d % input.ndim for d in dim], reverse=True)

    # Iteratively apply a deterministic mean.
    for d in sorted_dims:
        result = mean_dim(result, dim=d, keepdim=True)

    if not keepdim:
        # Squeeze the reduced dimensions.
        for d in sorted_dims:
            result = result.squeeze(d)

    return result


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization along the last dimension of a 2D tensor.
    RMS Norm: y = x / sqrt(mean(x^2) + eps) * weight
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Compute sum of squares in float32 to avoid overflow
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        # Convert to float32 for accumulation to prevent overflow
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))

    # Step 2: Compute RMS (root mean square) in float32
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms

    # Step 3: Normalize and apply weight
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        # Compute in float32 then convert back to input dtype
        vals_f32 = vals.to(tl.float32)
        weight_f32 = weight.to(tl.float32)
        output_f32 = vals_f32 * inv_rms * weight_f32
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute RMS normalization using Triton kernel.

    RMS Norm normalizes the input by the root mean square and scales by weight:
    output = input / sqrt(mean(input^2) + eps) * weight

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Tensor with RMS normalization applied along the last dimension
    """
    assert weight.dim() == 1, "Weight must be 1-dimensional"
    assert input.shape[-1] == weight.shape[0], (
        f"Input last dimension ({input.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = input_2d.shape

    output = torch.empty_like(input_2d)
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        input_2d,
        weight,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)


def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Batch-invariant wrapper for RMS normalization.

    This function provides a deterministic, batch-invariant implementation
    of RMS normalization for use with the batch_invariant mode.

    Args:
        input: Input tensor of shape (..., hidden_size)
        weight: Weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        RMS normalized tensor
    """
    return rms_norm(input, weight, eps=eps)


def linear_batch_invariant(input, weight, bias=None):
    output = matmul_batch_invariant(input, weight.t())

    if bias is not None:
        output = output + bias
    return output


def linear_batch_invariant_nvfp4(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    output_size: int,
    bias: torch.Tensor | None = None,
    *,
    weights_padding_cols: int = 0,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """
    Deterministic Blackwell NVFP4 linear path using Triton tl.dot_scaled.
    """
    if not current_platform.has_device_capability(100):
        raise RuntimeError(
            "Batch-invariant NVFP4 path requires Blackwell (sm100+) GPUs."
        )
    if not hasattr(tl, "dot_scaled"):
        raise RuntimeError(
            "This Triton build does not expose tl.dot_scaled required for "
            "batch-invariant NVFP4."
        )
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "Batch-invariant NVFP4 path expects packed FP4 weights and FP8 "
            "E4M3 block scales."
        )
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"Batch-invariant NVFP4 path only supports fp16/bf16 inputs, got {input.dtype}."
        )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()

    # Lazily import to avoid circular import with vllm._custom_ops.
    from vllm._custom_ops import scaled_fp4_quant

    x_fp4, x_blockscale = scaled_fp4_quant(
        input_2d,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend=quant_backend,
    )
    if weights_padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()

    out_2d = matmul_nvfp4_persistent(
        a_fp4=x_fp4,
        b_fp4=weight,
        a_scale=x_blockscale,
        b_scale=weight_scale,
        alpha=alpha,
        output_dtype=input.dtype,
    )
    if out_2d.shape[-1] != output_size:
        out_2d = out_2d[:, :output_size].contiguous()

    output = out_2d.reshape(*original_shape[:-1], output_size)
    if bias is not None:
        output = output + bias
    return output


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
    Deterministic NVFP4 MoE fallback built from batch-invariant NVFP4 GEMMs.
    """
    if hidden_states.ndim != 2:
        raise RuntimeError(
            f"Expected 2D hidden_states for NVFP4 MoE fallback, got {hidden_states.shape}."
        )
    if topk_ids.shape != topk_weights.shape:
        raise RuntimeError(
            "NVFP4 MoE fallback expects topk_ids and topk_weights to have identical shapes."
        )
    if topk_ids.ndim != 2:
        raise RuntimeError(
            f"Expected 2D top-k routing tensors, got shape {topk_ids.shape}."
        )
    if apply_router_weight_on_input and topk_ids.shape[1] != 1:
        raise RuntimeError(
            "apply_router_weight_on_input=True is only supported for top_k == 1."
        )

    # Lazily import to avoid pulling MoE deps when the fallback is unused.
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
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
    repeated_hidden_states = (
        hidden_states.unsqueeze(1)
        .expand(num_tokens, top_k, hidden_dim)
        .reshape(-1, hidden_dim)
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
    contribution = torch.zeros(
        (repeated_hidden_states.shape[0], hidden_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    w1_padding_cols = max(0, w13_weight.shape[-1] - hidden_dim // 2)
    w2_output_size = hidden_dim
    w1_output_size = w13_weight.shape[1]

    for expert_id in range(num_experts):
        # CUDA graph capture-safe expert routing mask. This avoids dynamic
        # indexing ops such as nonzero/index_select that are capture-unsafe.
        expert_mask = (expert_ids == expert_id).unsqueeze(-1)
        expert_inputs = repeated_hidden_states * expert_mask.to(
            repeated_hidden_states.dtype
        )

        gemm1_out = linear_batch_invariant_nvfp4(
            input=expert_inputs,
            weight=w13_weight[expert_id].contiguous(),
            weight_scale=w13_weight_scale[expert_id].contiguous(),
            input_global_scale_inv=_nvfp4_get_expert_scalar(
                a1_gscale, expert_id, field_name="a1_gscale"
            ),
            alpha=_nvfp4_get_expert_scalar(
                g1_alphas, expert_id, field_name="g1_alphas"
            ),
            output_size=w1_output_size,
            weights_padding_cols=w1_padding_cols,
            quant_backend=quant_backend,
        )

        activation_out_dim = (
            gemm1_out.shape[-1] // 2
            if activation_kind.is_gated
            else gemm1_out.shape[-1]
        )
        activation_out = torch.empty(
            (gemm1_out.shape[0], activation_out_dim),
            device=gemm1_out.device,
            dtype=gemm1_out.dtype,
        )
        apply_moe_activation(
            activation=activation_kind,
            output=activation_out,
            input=gemm1_out,
        )

        w2_padding_cols = max(0, w2_weight.shape[-1] - activation_out_dim // 2)
        gemm2_out = linear_batch_invariant_nvfp4(
            input=activation_out,
            weight=w2_weight[expert_id].contiguous(),
            weight_scale=w2_weight_scale[expert_id].contiguous(),
            input_global_scale_inv=_nvfp4_get_expert_scalar(
                a2_gscale, expert_id, field_name="a2_gscale"
            ),
            alpha=_nvfp4_get_expert_scalar(
                g2_alphas, expert_id, field_name="g2_alphas"
            ),
            output_size=w2_output_size,
            weights_padding_cols=w2_padding_cols,
            quant_backend=quant_backend,
        )

        weighted_output = (
            gemm2_out.to(torch.float32)
            * output_scale.unsqueeze(-1)
            * expert_mask.to(torch.float32)
        )
        contribution += weighted_output

    reduced = torch.zeros(
        (num_tokens, hidden_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    contribution = contribution.view(num_tokens, top_k, hidden_dim)
    for slot in range(top_k):
        reduced += contribution[:, slot, :]
    return reduced.to(hidden_states.dtype)


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_original_torch_bmm = None
_original_fp16_reduction_precision = None
_original_bf16_reduction_precision = None
_original_cublas_workspace_cfg = None
_original_cublaslt_workspace_size = None


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    global _original_fp16_reduction_precision, _original_bf16_reduction_precision
    global _original_cublas_workspace_cfg, _original_cublaslt_workspace_size
    if _batch_invariant_MODE:
        return

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    if (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability(80)
        or current_platform.is_device_capability(89)
    ):
        # For PyTorch 2.9, B200 uses GEMV for bs=1
        # Requires https://github.com/pytorch/pytorch/pull/166735
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "CUDA")
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "CUDA")
    else:
        # Only source of batch invariance for Hopper is split-k, can disable through
        # cuBLAS workspace config
        _original_cublas_workspace_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)
        _original_cublaslt_workspace_size = os.environ.get(
            "CUBLASLT_WORKSPACE_SIZE", None
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"

    _batch_invariant_LIB.impl(
        "aten::_log_softmax", _log_softmax_batch_invariant, "CUDA"
    )
    _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")

    # Also monkeypatch torch.bmm directly as a fallback
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "CUDA")
    _original_torch_bmm = torch.bmm
    torch.bmm = bmm_batch_invariant

    _original_bf16_reduction_precision = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    _original_fp16_reduction_precision = (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )

    reduced_precision_val = (
        (False, False) if is_torch_equal_or_newer("2.10.0") else False
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        reduced_precision_val
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision_val
    )
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")


def _read_vllm_batch_invariant() -> bool:
    val = os.getenv("VLLM_BATCH_INVARIANT", "0")
    try:
        return int(val) != 0
    except ValueError:
        return False


VLLM_BATCH_INVARIANT: bool = _read_vllm_batch_invariant()


def vllm_is_batch_invariant() -> bool:
    return VLLM_BATCH_INVARIANT


def override_envs_for_invariance(
    attention_backend: AttentionBackendEnum | None,
):
    decode_invariant_backends = [
        AttentionBackendEnum.FLASH_ATTN,  # best supported backend
        AttentionBackendEnum.TRITON_ATTN,
    ]
    supported_backends = decode_invariant_backends + [
        # FlashInfer temporarily disabled due to invariant CTA sizes.
        # See FlashInfer issue #2424
        # AttentionBackendEnum.FLASHINFER,
        AttentionBackendEnum.FLASH_ATTN_MLA,
        AttentionBackendEnum.TRITON_MLA,
        # Not yet supported MLA backends
        # AttentionBackendEnum.FLASHMLA,
        # AttentionBackendEnum.FLEX_ATTENTION,  # IMA issue
        # AttentionBackendEnum.FLASHINFER_MLA,  # PR #28967
    ]
    if attention_backend not in supported_backends:
        supported_names = [b.name for b in supported_backends]
        backend_name = attention_backend.name if attention_backend else None
        error = (
            "VLLM batch_invariant mode requires an attention backend in "
            f"{supported_names}, but got '{backend_name}'. "
            "Please use --attention-backend or attention_config to set "
            "one of the supported backends before enabling batch_invariant."
        )
        raise RuntimeError(error)
    if attention_backend not in decode_invariant_backends:
        warning = (
            "You are using a non-decode-invariant form of batch invariance. "
            "This will not be invariant between prefill and decode."
        )
        logger.warning_once(warning, scope="local")
    os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # NCCL determinism settings
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["NCCL_P2P_NET_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    os.environ["NCCL_NTHREADS"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"

    # torch.compile settings
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"


def init_batch_invariance(
    attention_backend: AttentionBackendEnum | None,
):
    # this will hit all the csrc overrides as well
    if vllm_is_batch_invariant():
        override_envs_for_invariance(attention_backend)
        enable_batch_invariant_mode()

        # Disable TF32 for batch invariance - it causes non-deterministic rounding
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
