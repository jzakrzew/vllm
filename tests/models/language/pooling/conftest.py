# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import gc

import pytest
import torch

from tests.utils import wait_for_memory_to_settle
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def pytest_sessionstart(session):
    """Configure ROCm-specific settings before test session starts."""
    if current_platform.is_rocm():
        torch.set_float32_matmul_precision("high")


@pytest.fixture(autouse=True)
def release_gpu_memory_between_tests():
    """Reclaim cached GPU memory between tests in this shard.

    These tests start many vLLM/HF engines sequentially in one pytest
    process; without explicit reclamation between tests, VRAM fragmentation
    accumulates and later tests fail at engine startup ("Engine core
    initialization failed").
    """
    yield
    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
    if current_platform.is_rocm():
        return
    try:
        wait_for_memory_to_settle()
    except ValueError as e:
        # Longer-lived (class/module-scoped) engine fixtures may legitimately
        # still hold VRAM; nothing further to reclaim in that case.
        logger.info("Failed to clean GPU memory: %s", e)


@pytest.fixture
def quantized_embedding_model(tmp_path, monkeypatch):
    """A tied INT4 embedding checkpoint and its dequantized HF reference."""
    from compressed_tensors.compressors.pack_quantized import (
        PackedQuantizationCompressor,
    )
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaModel, TokenizersBackend

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaModel(config).to(torch.bfloat16).eval()
    state_dict = model.state_dict()
    weight = state_dict.pop("embed_tokens.weight")
    scheme = QuantizationScheme(
        targets=["Embedding"],
        weights=QuantizationArgs(
            num_bits=4, type="int", strategy="group", group_size=32
        ),
        format="pack-quantized",
    )
    scale = weight.reshape(128, 4, 32).float().abs().amax(-1) / 7.5
    compressed = PackedQuantizationCompressor.compress(
        {"weight": weight, "weight_scale": scale.to(torch.bfloat16)}, scheme
    )
    with torch.no_grad():
        weight.copy_(
            PackedQuantizationCompressor.decompress(compressed, scheme)["weight"]
        )
    config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {"embeddings": scheme.model_dump(mode="json")},
    }
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(tmp_path)
    save_file(
        {f"model.{name}": value for name, value in state_dict.items()}
        | {f"model.embed_tokens.{name}": value for name, value in compressed.items()},
        tmp_path / "model.safetensors",
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel({f"token{i}": i for i in range(128)}, unk_token="token0")
        ),
        unk_token="token0",
    )
    tokenizer.save_pretrained(tmp_path)
    return tmp_path, model, torch.tensor([[1, 2, 3], [5, 17, 127]])
