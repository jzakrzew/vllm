# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.model_states import encoder_only


def test_encoder_only_attention_spec_uses_fp8_attention_dtype(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        encoder_only.current_platform,
        "fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )
    layer = SimpleNamespace(
        encoder_attention_dtype="fp8",
        kv_cache_dtype="auto",
        num_kv_heads=2,
        head_size=16,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(dtype=torch.float16),
    )

    spec = encoder_only._make_encoder_only_attention_spec(layer, vllm_config)

    assert spec.dtype == torch.float8_e4m3fn
