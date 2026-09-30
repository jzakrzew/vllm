# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AriaTextConfig
from transformers.models.aria.modeling_aria import AriaTextMoELayer as HFMoE

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.models.aria import (
    AriaTextDecoderLayer,
    AriaTextModel,
    AriaTextMoELayer,
)
from vllm.model_executor.models.utils import AutoWeightsLoader


@pytest.mark.cpu_test
def test_aria_decoder_forward_with_moe():
    """Inherited decoder forward must support an MoE without gate_up_proj."""
    hidden_size = 64
    x = torch.randn(3, hidden_size, device="cpu")
    with set_current_vllm_config(VllmConfig()):
        layer = AriaTextDecoderLayer.__new__(AriaTextDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.input_layernorm = RMSNorm(hidden_size, dtype=x.dtype)
        layer.post_attention_layernorm = RMSNorm(hidden_size, dtype=x.dtype)
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = torch.nn.Identity()
        layer.self_attn.forward = lambda positions, hidden_states: hidden_states
        layer.mlp = AriaTextMoELayer.__new__(AriaTextMoELayer)
        torch.nn.Module.__init__(layer.mlp)
        layer.mlp.router_weight = torch.nn.Parameter(torch.zeros(2, hidden_size))
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.forward = lambda hidden_states, router_output: hidden_states

        expected, expected_residual = layer.post_attention_layernorm(
            layer.input_layernorm(x), x
        )
        actual, residual = layer(torch.arange(x.shape[0]), x, None)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(residual, expected_residual)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize(
    "checkpoint_name,loaded_name,param_name",
    [
        (
            "experts.fc1.weight",
            "experts.w13_weight",
            "experts.routed_experts.w13_weight",
        ),
        (
            "experts.fc2.weight",
            "experts.w2_weight",
            "experts.routed_experts.w2_weight",
        ),
        (
            "shared_experts.down_proj.weight",
            "shared_experts.down_proj.weight",
            "shared_experts.down_proj.weight",
        ),
    ],
)
def test_aria_expert_weights_load_with_checkpoint_layout(
    checkpoint_name: str, loaded_name: str, param_name: str
):
    """Real Aria checkpoint weights must reach the right parameter and layout."""
    config = AriaTextConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_num_experts=2,
        moe_topk=1,
        moe_num_shared_experts=1,
    )
    checkpoint_weight = HFMoE(config).state_dict()[checkpoint_name]
    checkpoint_weight.copy_(
        torch.arange(
            checkpoint_weight.numel(), dtype=checkpoint_weight.dtype
        ).reshape_as(checkpoint_weight)
    )
    expected = checkpoint_weight.clone()
    if checkpoint_name.startswith("experts."):
        expected = expected.transpose(-1, -2)

    layer = AriaTextMoELayer(config, quant_config=None, prefix="mlp")
    loaded = AutoWeightsLoader(layer).load_weights(
        [(checkpoint_name, checkpoint_weight)], mapper=AriaTextModel.hf_to_vllm_mapper
    )

    assert loaded == {loaded_name}
    torch.testing.assert_close(
        dict(layer.named_parameters())[param_name], expected, rtol=0, atol=0
    )


@pytest.mark.cpu_test
def test_aria_quant_config_renames_expert_modules():
    quant_config = CompressedTensorsConfig(
        target_scheme_map={},
        ignore=[
            "model.layers.0.mlp.experts.fc1",
            "model.layers.0.mlp.experts.fc2",
        ],
        quant_format="pack-quantized",
    )
    AriaTextModel.__new__(
        AriaTextModel, vllm_config=VllmConfig(quant_config=quant_config)
    )

    assert quant_config.ignore == [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "name,expected", [("fc1", "gate_up_proj"), ("fc2", "down_proj")]
)
def test_aria_expert_scale_names(name: str, expected: str):
    suffixes = ["weight_scale", "input_scale"]
    names = [f"model.layers.0.mlp.experts.{name}.{suffix}" for suffix in suffixes]
    assert AriaTextModel.hf_to_vllm_mapper.apply_list(names) == [
        f"model.layers.0.mlp.experts.{expected}.{suffix}" for suffix in suffixes
    ]
