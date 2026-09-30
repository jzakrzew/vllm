# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the QuantizedActivation linear-kernel integration."""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_INT8_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
)
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
    expose_input_quant_key,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

# The only backends that consume a pre-quantized activation.
SUPPORTING = {
    CutlassFP8ScaledMMLinearKernel,
    FlashInferFP8ScaledMMLinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
}


def _all_kernel_classes() -> list[type]:
    seen: dict[type, None] = {}
    for registry in (
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_NVFP4_KERNELS,
        _POSSIBLE_MXFP8_KERNELS,
    ):
        for kernels in registry.values():
            for cls in kernels:
                seen.setdefault(cls, None)
    return list(seen)


def _probe(cls: type):
    """A bare kernel instance with a plausible config, so input_quant_key()
    can be queried without the hardware-gated constructor."""
    obj = cls.__new__(cls)  # type: ignore[call-overload]
    if issubclass(cls, NvFp4LinearKernel):
        obj.config = NvFp4LinearLayerConfig()
    elif issubclass(cls, Int8ScaledMMLinearKernel):
        obj.config = Int8ScaledMMLinearLayerConfig(
            is_static_input_scheme=True, is_channelwise=False, input_symmetric=True
        )
    else:
        obj.config = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticTensorSym,
            activation_quant_key=kFp8StaticTensorSym,
            weight_shape=(16, 16),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
    return obj


def _resolved_apply_weights(cls: type):
    for base in cls.__mro__:
        if "apply_weights" in base.__dict__:
            return base.__dict__["apply_weights"]
    raise AssertionError(f"{cls.__name__} has no apply_weights in its MRO")


def test_only_known_backends_support_prequantized_input():
    declarers = {c for c in _all_kernel_classes() if _probe(c).input_quant_key()}
    assert declarers == SUPPORTING


def test_supporting_backend_declares_consume_via_helper():
    for cls in SUPPORTING:
        fn = _resolved_apply_weights(cls)
        assert "as_quantized_activation" in fn.__code__.co_names, cls.__name__


def test_bridge_marks_supporting_and_skips_others():
    supported = _probe(FlashInferCutlassNvFp4LinearKernel)
    layer = torch.nn.Module()
    expose_input_quant_key(layer, supported)
    assert get_input_quant_key(layer) == kNvfp4Dynamic
    layer.requires_unquantized_input = True
    assert get_input_quant_key(layer) is None
    layer.requires_unquantized_input = False
    assert get_input_quant_key(layer) == kNvfp4Dynamic

    unsupported = _probe(FlashInferTrtllmNvFp4LinearKernel)
    assert unsupported.input_quant_key() is None
    layer = torch.nn.Module()
    expose_input_quant_key(layer, unsupported)
    assert get_input_quant_key(layer) is None


def test_as_quantized_activation_validates_key():
    qa = QuantizedActivation(
        data=torch.zeros(2, 4, dtype=current_platform.fp8_dtype()),
        scale=torch.tensor(1.0),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([2, 4]),
        quant_key=kFp8StaticTensorSym,
    )
    with pytest.raises(AssertionError):
        as_quantized_activation(qa, kNvfp4Dynamic)
    with pytest.raises(AssertionError):
        as_quantized_activation(qa, None)
    assert as_quantized_activation(torch.zeros(2, 4), kFp8StaticTensorSym) is None
    assert as_quantized_activation(qa, kFp8StaticTensorSym) is qa


@pytest.mark.parametrize("add_residual", [False, True])
def test_manual_norm_quant_fallback_preserves_residual(add_residual: bool):
    """An unsupported consumer/device must retain ordinary norm semantics."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fusion.fused_norm_quant import (
        maybe_fused_norm_quant,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm

    x = torch.randn(3, 64, device="cpu")
    residual = torch.randn_like(x) if add_residual else None
    original_x = x.clone()
    linear = torch.nn.Module()
    linear._input_quant_key = kNvfp4Dynamic
    with set_current_vllm_config(VllmConfig()):
        norm = RMSNorm(64, dtype=x.dtype)
        expected = norm(x.clone(), residual.clone() if residual is not None else None)
        actual, updated_residual = maybe_fused_norm_quant(norm, x, linear, residual)

    assert isinstance(actual, torch.Tensor)
    if residual is None:
        torch.testing.assert_close(actual, expected)
        assert updated_residual is x
        torch.testing.assert_close(x, original_x)
    else:
        torch.testing.assert_close(actual, expected[0])
        torch.testing.assert_close(updated_residual, expected[1])


@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize(
    "case", ["fused", "opt_out", "partial_scale_tile", "collective"]
)
def test_manual_norm_quant_with_unbacked_token_count(monkeypatch, add_residual, case):
    """Dispatch must not guard on token count; opt-outs still return plain tensors."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fusion import fused_norm_quant
    from vllm.model_executor.layers.layernorm import RMSNorm

    monkeypatch.setattr(fused_norm_quant, "_rmsnorm_fp4quant", lambda *a, **kw: None)
    monkeypatch.setattr(
        fused_norm_quant, "_add_rmsnorm_fp4quant", lambda *a, **kw: None
    )
    config = VllmConfig()
    config.compilation_config.pass_config.fuse_allreduce_rms = case == "collective"
    hidden_size = 80 if case == "partial_scale_tile" else 64
    shape_env = ShapeEnv()
    with (
        set_current_vllm_config(config),
        FakeTensorMode(shape_env=shape_env),
        torch.device("cuda"),
    ):
        x = torch.empty(
            (shape_env.create_unbacked_symint(), hidden_size), dtype=torch.bfloat16
        )
        norm = RMSNorm(hidden_size, dtype=x.dtype)
        linear = torch.nn.Module()
        linear._input_quant_key = kNvfp4Dynamic
        linear.input_global_scale_inv = torch.ones(1, dtype=torch.float32)
        linear.requires_unquantized_input = case == "opt_out"
        linear.tp_size = 2 if case == "collective" else 1
        result, _ = fused_norm_quant.maybe_fused_norm_quant(
            norm, x, linear, torch.empty_like(x) if add_residual else None
        )
    if case == "fused":
        assert isinstance(result, QuantizedActivation)
        assert result.data.shape[-1] == hidden_size // 2
    else:
        assert isinstance(result, torch.Tensor)
