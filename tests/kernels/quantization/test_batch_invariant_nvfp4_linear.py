# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant_nvfp4
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    prepare_weights_for_nvfp4_cutlass,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Batch-invariant NVFP4 linear requires Blackwell (sm100+) support.",
        allow_module_level=True,
    )

if not hasattr(tl, "dot_scaled"):
    pytest.skip(
        reason="Installed Triton build does not expose tl.dot_scaled.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
SEEDS = [42]
DEVICES = ["cuda:0"]


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max()).to(torch.float32)


def _dequant_ref_output(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    output_size: int,
    weights_padding_cols: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_fp4, x_scale = ops.scaled_fp4_quant(x_2d, input_global_scale)
    if weights_padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()

    x_dq = dequantize_nvfp4_to_dtype(
        x_fp4,
        x_scale,
        input_global_scale,
        dtype=x.dtype,
        device=x.device,
        block_size=16,
    )
    w_dq = dequantize_nvfp4_to_dtype(
        weight_fp4,
        weight_scale,
        weight_global_scale,
        dtype=x.dtype,
        device=x.device,
        block_size=16,
    )

    out_2d = torch.matmul(x_dq, w_dq.t())
    out_2d = out_2d[:, :output_size].contiguous()
    out = out_2d.reshape(*x.shape[:-1], output_size)
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_matches_dequant_reference(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
) -> None:
    set_random_seed(seed)
    out_features = 96
    in_features = 128

    if input_ndim == 2:
        x = torch.randn((17, in_features), dtype=dtype, device=device)
    else:
        x = torch.randn((4, 5, in_features), dtype=dtype, device=device)

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    input_global_scale = _global_scale(x.reshape(-1, x.shape[-1]))
    weight_global_scale = _global_scale(w)
    alpha = (1.0 / (input_global_scale * weight_global_scale)).to(torch.float32)

    weight_fp4, weight_scale = ops.scaled_fp4_quant(w, weight_global_scale)
    weight_fp4, weight_scale, weights_padding_cols = prepare_weights_for_nvfp4_cutlass(
        weight_fp4, weight_scale
    )

    out = linear_batch_invariant_nvfp4(
        input=x,
        weight=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale_inv=input_global_scale,
        alpha=alpha,
        output_size=out_features,
        bias=bias,
        weights_padding_cols=weights_padding_cols,
        quant_backend="cutlass",
    )
    ref = _dequant_ref_output(
        x=x,
        weight_fp4=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale=input_global_scale,
        weight_global_scale=weight_global_scale,
        output_size=out_features,
        weights_padding_cols=weights_padding_cols,
        bias=bias,
    )
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_large_dims(
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test with dimensions that exercise multiple swizzle blocks (128-row,
    4-column) and multiple K-tiles (BLOCK_SIZE_K=256)."""
    set_random_seed(seed)
    out_features = 256
    in_features = 512
    m = 33

    x = torch.randn((m, in_features), dtype=dtype, device=device)
    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    input_global_scale = _global_scale(x)
    weight_global_scale = _global_scale(w)
    alpha = (1.0 / (input_global_scale * weight_global_scale)).to(torch.float32)

    weight_fp4, weight_scale = ops.scaled_fp4_quant(w, weight_global_scale)
    weight_fp4, weight_scale, weights_padding_cols = prepare_weights_for_nvfp4_cutlass(
        weight_fp4, weight_scale
    )

    out = linear_batch_invariant_nvfp4(
        input=x,
        weight=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale_inv=input_global_scale,
        alpha=alpha,
        output_size=out_features,
        bias=bias,
        weights_padding_cols=weights_padding_cols,
        quant_backend="cutlass",
    )
    ref = _dequant_ref_output(
        x=x,
        weight_fp4=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale=input_global_scale,
        weight_global_scale=weight_global_scale,
        output_size=out_features,
        weights_padding_cols=weights_padding_cols,
        bias=bias,
    )
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_batch_size_invariance(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
) -> None:
    set_random_seed(seed)
    out_features = 64
    in_features = 128
    output_size = out_features

    if input_ndim == 2:
        x_single = torch.randn((1, in_features), dtype=dtype, device=device)
        x_batch = torch.cat(
            [x_single, torch.randn((7, in_features), dtype=dtype, device=device)], dim=0
        )
    else:
        x_single = torch.randn((1, 4, in_features), dtype=dtype, device=device)
        x_batch = torch.cat(
            [x_single, torch.randn((3, 4, in_features), dtype=dtype, device=device)],
            dim=0,
        )

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)
    weight_global_scale = _global_scale(w)
    weight_fp4, weight_scale = ops.scaled_fp4_quant(w, weight_global_scale)
    weight_fp4, weight_scale, weights_padding_cols = prepare_weights_for_nvfp4_cutlass(
        weight_fp4, weight_scale
    )

    single_global_scale = _global_scale(x_single.reshape(-1, in_features))
    single_alpha = (1.0 / (single_global_scale * weight_global_scale)).to(torch.float32)
    out_single = linear_batch_invariant_nvfp4(
        input=x_single,
        weight=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale_inv=single_global_scale,
        alpha=single_alpha,
        output_size=output_size,
        bias=bias,
        weights_padding_cols=weights_padding_cols,
        quant_backend="cutlass",
    )

    batch_global_scale = _global_scale(x_batch.reshape(-1, in_features))
    batch_alpha = (1.0 / (batch_global_scale * weight_global_scale)).to(torch.float32)
    out_batch = linear_batch_invariant_nvfp4(
        input=x_batch,
        weight=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale_inv=batch_global_scale,
        alpha=batch_alpha,
        output_size=output_size,
        bias=bias,
        weights_padding_cols=weights_padding_cols,
        quant_backend="cutlass",
    )

    if input_ndim == 2:
        assert torch.equal(out_single[0], out_batch[0])
    else:
        assert torch.equal(out_single[0], out_batch[0])
