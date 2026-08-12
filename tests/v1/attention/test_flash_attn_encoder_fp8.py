# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.v1.attention.backend import AttentionType


class MockAttentionLayer:
    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor([0.5, 0.75], device=device)
        self._k_scale = torch.tensor([0.25, 0.5], device=device)
        self._v_scale = torch.tensor([0.125, 0.25], device=device)


def test_encoder_only_attention_uses_encoder_attention_dtype(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    import vllm.model_executor.layers.attention.encoder_only_attention as enc_mod
    from vllm.model_executor.layers.attention.encoder_only_attention import (
        EncoderOnlyAttention,
    )

    captured = {}

    class FakeImpl:
        supports_quant_query_input = False

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int | None,
            _alibi_slopes,
            _sliding_window,
            kv_cache_dtype: str,
            _logits_soft_cap=None,
            _attn_type=AttentionType.DECODER,
            _kv_sharing_target_layer_name=None,
            **_kwargs,
        ) -> None:
            captured["impl_kv_cache_dtype"] = kv_cache_dtype
            self.num_heads = num_heads
            self.head_size = head_size
            self.num_kv_heads = num_kv_heads
            self.scale = scale

    class FakeBackend:
        forward_includes_kv_cache_update = False

        @staticmethod
        def get_name() -> str:
            return "FLASH_ATTN"

        @staticmethod
        def get_impl_cls():
            return FakeImpl

        @classmethod
        def supports_alibi_sqrt(cls) -> bool:
            return False

    def fake_get_attn_backend(
        _head_size: int,
        _dtype: torch.dtype,
        kv_cache_dtype: str,
        **_kwargs,
    ):
        captured["selector_kv_cache_dtype"] = kv_cache_dtype
        return FakeBackend

    default_vllm_config.model_config = SimpleNamespace(
        encoder_attention_dtype="fp8",
        dtype=torch.float16,
        is_mm_prefix_lm=False,
    )
    default_vllm_config.cache_config.cache_dtype = "auto"
    monkeypatch.setattr(enc_mod, "get_attn_backend", fake_get_attn_backend)
    monkeypatch.setattr(
        enc_mod, "create_encoder_only_attention_backend", lambda backend: backend
    )

    layer = EncoderOnlyAttention(
        num_heads=2,
        head_size=16,
        scale=0.25,
        num_kv_heads=2,
        cache_config=default_vllm_config.cache_config,
        prefix="test.encoder_only_attn",
    )

    assert layer.kv_cache_dtype == "auto"
    assert layer.encoder_attention_dtype == "fp8"
    assert captured["selector_kv_cache_dtype"] == "auto"
    assert captured["impl_kv_cache_dtype"] == "fp8"


def _patch_flash_attn_fp8_environment(
    monkeypatch: pytest.MonkeyPatch,
    fa_mod,
    *,
    fa_version: int,
    supported: bool = True,
):
    monkeypatch.setattr(
        fa_mod,
        "get_flash_attn_version",
        lambda *args, **kwargs: fa_version,
    )
    monkeypatch.setattr(
        fa_mod,
        "flash_attn_supports_quant_query_input",
        lambda: True,
    )
    monkeypatch.setattr(
        fa_mod,
        "flash_attn_supports_kv_cache_dtype",
        lambda *args, **kwargs: supported,
    )
    monkeypatch.setattr(
        fa_mod.current_platform,
        "fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )


def test_flash_attn_encoder_fp8_initializes_quantizers_before_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.v1.attention.backends.flash_attn as fa_mod
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    _patch_flash_attn_fp8_environment(
        monkeypatch,
        fa_mod,
        fa_version=4,
    )

    config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(config):
        impl = FlashAttentionImpl(
            num_heads=4,
            head_size=16,
            scale=0.25,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            attn_type=AttentionType.ENCODER_ONLY,
        )

    assert impl._get_encoder_fp8_quant(None) is not None
    assert impl._get_encoder_fp8_quant(16) is not None
    assert impl._get_encoder_fp8_quant(32) is not None


@pytest.mark.parametrize("fa_version", [3, 4])
def test_flash_attn_encoder_only_fp8_quantizes_direct_qkv(
    monkeypatch: pytest.MonkeyPatch,
    fa_version: int,
):
    import vllm.v1.attention.backends.flash_attn as fa_mod
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    _patch_flash_attn_fp8_environment(
        monkeypatch,
        fa_mod,
        fa_version=fa_version,
    )

    quant_calls = []

    class FakeQuantFP8:
        def __init__(self, static: bool, group_shape: GroupShape):
            self.group_shape = group_shape

        def __call__(
            self,
            x: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            quant_calls.append((self.group_shape, tuple(x.shape), tuple(scale.shape)))
            return x.to(torch.float8_e4m3fn), scale

    flash_call = {}

    def fake_flash_attn_varlen_func(**kwargs):
        flash_call.update(kwargs)
        kwargs["out"].zero_()
        return kwargs["out"]

    monkeypatch.setattr(fa_mod, "QuantFP8", FakeQuantFP8)
    monkeypatch.setattr(
        fa_mod,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )

    impl = FlashAttentionImpl(
        num_heads=4,
        head_size=16,
        scale=0.25,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
        attn_type=AttentionType.ENCODER_ONLY,
    )
    layer = MockAttentionLayer(torch.device("cpu"))
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_query_len=3,
    )
    model_dtype = torch.bfloat16 if fa_version == 4 else torch.float16
    query = torch.randn(5, 4, 16, dtype=model_dtype)
    key = torch.randn(5, 2, 16, dtype=model_dtype)
    value = torch.randn(5, 2, 16, dtype=model_dtype)
    output = torch.empty_like(query)

    impl._forward_encoder_attention(query, key, value, output, metadata, layer)

    assert quant_calls == [
        (GroupShape(-1, 32), (5, 64), (2,)),
        (GroupShape(-1, 16), (5, 32), (2,)),
        (GroupShape(-1, 16), (5, 32), (2,)),
    ]
    assert flash_call["q"].dtype == torch.float8_e4m3fn
    assert flash_call["k"].dtype == torch.float8_e4m3fn
    assert flash_call["v"].dtype == torch.float8_e4m3fn
    assert flash_call["q_descale"].shape == (2, 2)
    assert flash_call["k_descale"].shape == (2, 2)
    assert flash_call["v_descale"].shape == (2, 2)
    assert flash_call["causal"] is False
    assert flash_call["fa_version"] == fa_version


def test_flash_attn_encoder_only_fp8_rejects_unsupported_version(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.v1.attention.backends.flash_attn as fa_mod
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    _patch_flash_attn_fp8_environment(
        monkeypatch,
        fa_mod,
        fa_version=2,
        supported=False,
    )

    with pytest.raises(NotImplementedError, match="does not support fp8"):
        FlashAttentionImpl(
            num_heads=4,
            head_size=16,
            scale=0.25,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
            attn_type=AttentionType.ENCODER_ONLY,
        )
