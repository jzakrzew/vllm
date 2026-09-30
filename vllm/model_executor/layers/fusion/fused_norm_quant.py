# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual RMSNorm and residual-add fusion with a consumer's NVFP4 input quant."""

from collections.abc import Callable
from typing import Any

import torch

from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

_rmsnorm_fp4quant: Callable[..., Any] | None = None
_add_rmsnorm_fp4quant: Callable[..., Any] | None = None
if (
    current_platform.is_cuda()
    and current_platform.has_device_capability(100)
    and hasattr(torch, "float4_e2m1fn_x2")
    and has_flashinfer()
):
    try:
        from flashinfer.norm import add_rmsnorm_fp4quant, rmsnorm_fp4quant
    except ImportError:
        pass
    else:
        _rmsnorm_fp4quant = rmsnorm_fp4quant
        _add_rmsnorm_fp4quant = add_rmsnorm_fp4quant


def _flashinfer_fused_add_rms_norm_nvfp4_quant(
    result: torch.Tensor,
    result_block_scale: torch.Tensor,
    residual: torch.Tensor | None,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    block_scale_unswizzled: torch.Tensor | None,
    is_sf_swizzled_layout: bool,
    epsilon: float,
) -> None:
    """Shared FlashInfer wrapper for standalone and residual-add NVFP4 norms."""
    if input.numel() == 0:
        return
    block_scale = result_block_scale.view(torch.float8_e4m3fn)
    if is_sf_swizzled_layout:
        block_scale = block_scale.flatten()
        if input.shape[-1] % 64 != 0:
            # Compiler callers may have a partial scale tile along K.
            block_scale.zero_()
    kwargs: dict[str, Any] = dict(
        y_fp4=result.view(torch.float4_e2m1fn_x2),
        block_scale=block_scale,
        global_scale=input_global_scale.reshape(1),
        eps=epsilon,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )
    if residual is None:
        assert _rmsnorm_fp4quant is not None
        _rmsnorm_fp4quant(input, weight, **kwargs)
    else:
        assert _add_rmsnorm_fp4quant is not None
        _add_rmsnorm_fp4quant(
            input,
            residual,
            weight,
            output_both_sf_layouts=False,
            block_scale_unswizzled=block_scale_unswizzled,
            **kwargs,
        )


def _flashinfer_fused_add_rms_norm_nvfp4_quant_fake(
    result: torch.Tensor,
    result_block_scale: torch.Tensor,
    residual: torch.Tensor | None,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    block_scale_unswizzled: torch.Tensor | None,
    is_sf_swizzled_layout: bool,
    epsilon: float,
) -> None:
    return None


direct_register_custom_op(
    op_name="flashinfer_fused_add_rms_norm_nvfp4_quant",
    op_func=_flashinfer_fused_add_rms_norm_nvfp4_quant,
    mutates_args=["result", "result_block_scale", "residual"],
    fake_impl=_flashinfer_fused_add_rms_norm_nvfp4_quant_fake,
)

_FLASHINFER_NVFP4_RMS_QUANT_OP = (
    torch.ops.vllm.flashinfer_fused_add_rms_norm_nvfp4_quant.default
    if _add_rmsnorm_fp4quant is not None
    else None
)


def _has_competing_collective_fusion(linear: torch.nn.Module) -> bool:
    if getattr(linear, "tp_size", 1) == 1:
        return False
    config = get_current_vllm_config_or_none()
    if config is None:
        return False
    pass_config = config.compilation_config.pass_config
    return bool(
        pass_config.fuse_allreduce_rms
        or pass_config.enable_sp
        or pass_config.fuse_gemm_comms
    )


def maybe_fused_norm_quant(
    norm: RMSNorm,
    x: torch.Tensor,
    linear: torch.nn.Module | None,
    residual: torch.Tensor | None = None,
    *,
    enabled: bool = True,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Normalize and optionally quantize, returning the unquantized residual.

    A supported NVFP4 consumer receives packed activations and swizzled block
    scales. Otherwise use the norm's ordinary forward. With residual addition,
    the fused kernel updates residual in place; without it, x is preserved.
    """
    kernel = _rmsnorm_fp4quant if residual is None else _add_rmsnorm_fp4quant
    supported = (
        enabled
        and linear is not None
        and type(norm) is RMSNorm
        and get_input_quant_key(linear) == kNvfp4Dynamic
        and kernel is not None
        and not envs.VLLM_BATCH_INVARIANT
        and x.is_cuda
        and x.ndim in (2, 3)
        and x.shape[-1] >= 64
        and x.shape[-1] % 64 == 0
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16)
        and norm.variance_size_override is None
        and (norm.pass_weight if residual is None else norm.pass_weight_add)
        and norm.weight.dtype == x.dtype
        and (
            residual is None or (residual.dtype == x.dtype and residual.is_contiguous())
        )
        and not _has_competing_collective_fusion(linear)
    )
    if not supported:
        if residual is None:
            return norm(x), x
        return norm(x, residual)

    from vllm._custom_ops import create_fp4_output_tensors

    assert linear is not None
    hidden_size = x.shape[-1]
    num_tokens = x.numel() // hidden_size
    data, block_scale = create_fp4_output_tensors(
        num_tokens, hidden_size, x.device, is_sf_swizzled_layout=True
    )
    torch.ops.vllm.flashinfer_fused_add_rms_norm_nvfp4_quant(
        data,
        block_scale,
        residual.view(num_tokens, hidden_size) if residual is not None else None,
        x.view(num_tokens, hidden_size),
        norm.weight,
        linear.input_global_scale_inv,
        None,
        True,
        norm.variance_epsilon,
    )
    return (
        QuantizedActivation(
            data=data,
            scale=block_scale,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kNvfp4Dynamic,
        ),
        x if residual is None else residual,
    )
