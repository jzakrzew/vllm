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
    "case",
    [
        "fused",
        "allreduce",
        "opt_out",
        "partial_scale_tile",
        "fuse_allreduce_rms",
        "enable_sp",
        "fuse_gemm_comms",
        "tp1_collective",
        "compiler_pass_disabled",
        "eager",
    ],
)
def test_manual_norm_quant_with_unbacked_token_count(monkeypatch, add_residual, case):
    """Dispatch must not guard on token count; opt-outs still return plain tensors."""
    from types import SimpleNamespace

    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.fusion import fused_norm_quant
    from vllm.model_executor.layers.layernorm import RMSNorm

    monkeypatch.setattr(fused_norm_quant, "_rmsnorm_fp4quant", lambda *a, **kw: None)
    monkeypatch.setattr(
        fused_norm_quant, "_add_rmsnorm_fp4quant", lambda *a, **kw: None
    )
    if (
        getattr(torch.ops.vllm, "flashinfer_fused_add_rms_norm_nvfp4_quant", None)
        is None
    ):
        fused_norm_quant.direct_register_custom_op(
            op_name="flashinfer_fused_add_rms_norm_nvfp4_quant",
            op_func=fused_norm_quant._flashinfer_fused_add_rms_norm_nvfp4_quant,
            mutates_args=["result", "result_block_scale", "residual", "input"],
            fake_impl=fused_norm_quant._flashinfer_fused_add_rms_norm_nvfp4_quant_fake,
        )
    config = VllmConfig()
    pass_config = config.compilation_config.pass_config
    pass_config.fuse_norm_quant = case != "compiler_pass_disabled"
    collective = case in ("fuse_allreduce_rms", "enable_sp", "fuse_gemm_comms")
    if collective:
        setattr(pass_config, case, True)
    if case == "tp1_collective":
        pass_config.fuse_allreduce_rms = True
    config.model_config = SimpleNamespace(enforce_eager=case == "eager")
    hidden_size = 80 if case == "partial_scale_tile" else 64
    shape_env = ShapeEnv()
    with (
        FakeTensorMode(shape_env=shape_env),
        torch.device("cuda"),
    ):
        x = torch.empty(
            (shape_env.create_unbacked_symint(), hidden_size), dtype=torch.bfloat16
        )
        with set_current_vllm_config(config):
            norm = RMSNorm(hidden_size, dtype=x.dtype)
        linear = torch.nn.Module()
        linear._input_quant_key = kNvfp4Dynamic
        linear.input_global_scale_inv = torch.ones(1, dtype=torch.float32)
        linear.requires_unquantized_input = case == "opt_out"
        linear.tp_size = 2 if collective else 1
        assert get_current_vllm_config_or_none() is None
        result, _ = fused_norm_quant.maybe_fused_norm_quant(
            norm,
            x,
            linear,
            torch.empty_like(x) if add_residual else None,
            do_allreduce=case == "allreduce",
        )
    if case not in ("opt_out", "partial_scale_tile"):
        assert isinstance(result, QuantizedActivation)
        assert result.data.shape[-1] == hidden_size // 2
    else:
        assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("hidden_size", [80, 96, 112, 144])
def test_norm_quant_wrapper_zeros_only_padded_scale_columns(monkeypatch, hidden_size):
    """Compiler callers with partial K tiles must preserve valid scale entries."""
    from vllm.model_executor.layers.fusion import fused_norm_quant

    monkeypatch.setattr(fused_norm_quant, "_rmsnorm_fp4quant", lambda *a, **kw: None)
    scale_cols = hidden_size // 16
    padded_cols = (scale_cols + 3) // 4 * 4
    scales = torch.full((128, padded_cols // 4), 0x38383838, dtype=torch.int32)
    fused_norm_quant._flashinfer_fused_add_rms_norm_nvfp4_quant(
        torch.empty((1, hidden_size // 2), dtype=torch.uint8),
        scales,
        None,
        torch.zeros((1, hidden_size), dtype=torch.bfloat16),
        torch.ones(hidden_size, dtype=torch.bfloat16),
        torch.ones(1),
        None,
        True,
        1e-6,
    )
    unswizzled = (
        scales.view(torch.uint8)
        .reshape(1, padded_cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(128, padded_cols)
    )
    assert torch.all(unswizzled[:, :scale_cols] == 0x38)
    assert torch.all(unswizzled[:, scale_cols:] == 0)


@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("collective_support", ["fused", "capacity", "unavailable"])
def test_norm_quant_reduces_before_residual_add(
    monkeypatch, add_residual, collective_support
):
    """Collective fallback and fusion must add the replicated residual only once."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fusion import allreduce_norm, fused_norm_quant

    x = torch.full((2, 64), 2.0, dtype=torch.bfloat16)
    residual = torch.full_like(x, 5.0) if add_residual else None
    result = torch.empty((2, 32), dtype=torch.uint8)
    scales = torch.empty((128, 1), dtype=torch.int32)
    reductions = []
    kernels = []

    def reduce(input):
        reductions.append(input.clone())
        return input * 2 + 3

    def quantize(input, weight, **kwargs):
        kernels.append(input.clone())
        kwargs["y_fp4"].view(torch.uint8).fill_(42)

    def add_quantize(input, residual, weight, **kwargs):
        residual.add_(input)
        quantize(residual, weight, **kwargs)

    def collective(**kwargs):
        input, residual = kwargs["allreduce_in"], kwargs["residual"]
        reduced = input * 2 + 3
        if kwargs["norm_out"] is not None:
            input.copy_(reduced)
            kernels.append(reduced.clone())
        else:
            residual.add_(reduced)
            kernels.append(residual.clone())
        kwargs["quant_out"].fill_(42)

    monkeypatch.setattr(fused_norm_quant, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(
        allreduce_norm, "call_trtllm_fused_allreduce_norm", collective, raising=False
    )
    monkeypatch.setattr(
        allreduce_norm, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(allreduce_norm, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        allreduce_norm, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(
        allreduce_norm.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(to_int=lambda: 100),
    )
    monkeypatch.setattr(
        allreduce_norm,
        "flashinfer_comm",
        SimpleNamespace(
            AllReduceFusionPattern=SimpleNamespace(kARResidualRMSNormFP4Quant=3)
        ),
    )
    workspace = (
        None
        if collective_support == "unavailable"
        else SimpleNamespace(
            is_buffer_size_sufficient=lambda **kw: collective_support == "fused"
        )
    )
    monkeypatch.setattr(
        allreduce_norm, "get_fi_ar_quant_workspace", lambda **kw: workspace
    )
    monkeypatch.setattr(fused_norm_quant, "_rmsnorm_fp4quant", quantize)
    monkeypatch.setattr(fused_norm_quant, "_add_rmsnorm_fp4quant", add_quantize)
    fused_norm_quant._flashinfer_fused_add_rms_norm_nvfp4_quant(
        result,
        scales,
        residual,
        x,
        torch.ones(64),
        torch.ones(1),
        None,
        True,
        1e-6,
        True,
    )
    assert len(reductions) == (0 if collective_support == "fused" else 1)
    assert len(kernels) == 1
    torch.testing.assert_close(
        kernels[0], torch.full_like(x, 12 if add_residual else 7)
    )
    torch.testing.assert_close(
        x if residual is None else residual,
        torch.full_like(x, 12 if add_residual else 7),
    )
    assert torch.all(result == 42)


@pytest.mark.parametrize(
    "architecture", ["llama", "mistral", "aria", "telechat2", "glm4", "llama4", "mixed"]
)
@pytest.mark.parametrize(
    "boundary", ["final", "pipeline", "pipeline_input", "auxiliary", "conditioned"]
)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_decoder_norm_owns_projection_allreduces(
    monkeypatch, architecture, boundary, tp_size
):
    """Fixed norm reductions preserve TP, PP, and auxiliary output semantics."""
    from types import SimpleNamespace

    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.fusion import fused_norm_quant
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models import aria, glm4, llama, llama4, mistral, telechat2
    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm.sequence import IntermediateTensors

    model_cls, layer_cls = {
        "llama": (llama.LlamaModel, llama.LlamaDecoderLayer),
        "mistral": (mistral.MistralModel, mistral.MistralDecoderLayer),
        "aria": (aria.AriaTextModel, aria.AriaTextDecoderLayer),
        "telechat2": (telechat2.TeleChat2Model, llama.LlamaDecoderLayer),
        "glm4": (glm4.Glm4Model, glm4.Glm4DecoderLayer),
        "llama4": (llama4.Llama4Model, llama4.Llama4DecoderLayer),
        "mixed": (llama.LlamaModel, llama.LlamaDecoderLayer),
    }[architecture]
    reductions = []

    def reduce(x):
        reductions.append(x.clone())
        return x * 2 + 3

    monkeypatch.setattr(fused_norm_quant, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(llama, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(llama, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(
        llama,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=boundary != "pipeline_input",
            is_last_rank=boundary != "pipeline",
        ),
    )
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=4 if boundary == "pipeline_input" else 2,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            num_local_experts=2,
        )
    )
    with set_current_vllm_config(config):
        layers = []
        layer_types = (
            (llama.LlamaDecoderLayer, llama4.Llama4DecoderLayer)
            if architecture == "mixed"
            else (layer_cls, layer_cls)
        )
        for layer_cls in layer_types:
            layer = layer_cls.__new__(layer_cls)
            torch.nn.Module.__init__(layer)
            layer.input_layernorm = RMSNorm(64)
            layer.post_attention_layernorm = RMSNorm(64)
            layer._allreduce_input = False
            layer.self_attn = llama.LlamaAttention.__new__(llama.LlamaAttention)
            torch.nn.Module.__init__(layer.self_attn)
            layer.self_attn.qkv_proj = torch.nn.Identity()
            layer.self_attn.o_proj = SimpleNamespace(
                tp_size=tp_size, reduce_results=True
            )
            layer.self_attn.forward = (
                lambda positions, hidden_states, proj=layer.self_attn.o_proj: (
                    reduce(hidden_states * 2)
                    if proj.reduce_results and proj.tp_size > 1
                    else hidden_states * 2
                )
            )
            mlp_cls = (
                aria.AriaTextMoELayer if architecture == "aria" else llama.LlamaMLP
            )
            mlp = mlp_cls.__new__(mlp_cls)
            torch.nn.Module.__init__(mlp)
            mlp.gate_up_proj = torch.nn.Identity()
            if architecture != "aria":
                mlp.down_proj = SimpleNamespace(tp_size=tp_size, reduce_results=True)
            down_proj = getattr(mlp, "down_proj", None)
            mlp.forward = lambda hidden_states, proj=down_proj: (
                reduce(hidden_states * 3)
                if tp_size > 1 and (proj is None or proj.reduce_results)
                else hidden_states * 3
            )
            if isinstance(layer, llama4.Llama4DecoderLayer):
                layer.feed_forward = mlp
            else:
                layer.mlp = mlp
            if architecture == "glm4":
                layer.post_self_attn_layernorm = RMSNorm(64)
                layer.post_mlp_layernorm = RMSNorm(64)
            if architecture == "mistral":
                layer.ada_rms_norm_t_cond = (
                    torch.nn.Identity() if boundary == "conditioned" else None
                )
            layers.append(layer)
        start_layer = 2 if boundary == "pipeline_input" else 0
        monkeypatch.setattr(
            llama,
            "make_layers",
            lambda *a, **kw: (
                start_layer,
                start_layer + 2,
                torch.nn.ModuleList([PPMissingLayer()] * start_layer + layers),
            ),
        )
        monkeypatch.setattr(
            llama,
            "VocabParallelEmbedding",
            lambda vocab, hidden, **kw: torch.nn.Embedding(vocab, hidden),
        )
        model = model_cls(vllm_config=config)
        if boundary == "auxiliary":
            model._set_aux_hidden_state_layers((1,))
    x = torch.randn(3, 64)
    t_cond = torch.full_like(x, 0.5)
    input_residual = torch.randn_like(x) if boundary == "pipeline_input" else None
    hidden = x.clone()
    residual = input_residual.clone() if input_residual is not None else None
    expected_aux = []
    for i, layer in enumerate(layers):
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        hidden = hidden * 4 + 3 if tp_size > 1 else hidden * 2
        if architecture == "glm4":
            hidden = layer.post_self_attn_layernorm(hidden)
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        if architecture == "mistral" and boundary == "conditioned":
            hidden = hidden * (1 + t_cond)
        hidden = hidden * 6 + 3 if tp_size > 1 else hidden * 3
        if architecture == "glm4":
            hidden = layer.post_mlp_layernorm(hidden)
        if i + 1 in model.aux_hidden_state_layers:
            expected_aux.append(hidden + residual)
    kwargs = {"t_cond": t_cond} if architecture == "mistral" else {}
    intermediate = (
        IntermediateTensors({"hidden_states": x, "residual": input_residual})
        if input_residual is not None
        else None
    )
    actual = model(None, torch.arange(3), intermediate, inputs_embeds=x, **kwargs)
    assert len(reductions) == (4 if tp_size > 1 else 0)
    if boundary == "pipeline":
        assert isinstance(actual, IntermediateTensors)
        torch.testing.assert_close(actual["hidden_states"], hidden)
        torch.testing.assert_close(actual["residual"], residual)
    else:
        expected, _ = model.norm(hidden, residual)
        if boundary == "auxiliary":
            actual, aux = actual
            torch.testing.assert_close(aux[0], expected_aux[0])
        torch.testing.assert_close(actual, expected)
