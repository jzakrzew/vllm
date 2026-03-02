# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.batch_invariant_moe import (
    fused_moe_batch_invariant_nvfp4,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Batch-invariant NVFP4 MoE requires Blackwell (sm100+) support.",
        allow_module_level=True,
    )

if not hasattr(tl, "dot_scaled"):
    pytest.skip(
        reason="Installed Triton build does not expose tl.dot_scaled.",
        allow_module_level=True,
    )

DTYPE = torch.bfloat16
DEVICE = "cuda:0"


def _make_nvfp4_moe_tensors(
    *,
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=DTYPE) / 10
    (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
        make_test_weights(
            e,
            n,
            k,
            in_dtype=DTYPE,
            quant_dtype="nvfp4",
            block_shape=None,
            per_out_ch_quant=False,
        )
    )
    assert w1_blockscale is not None and w2_blockscale is not None
    assert w1_gs is not None and w2_gs is not None

    score = torch.randn((m, e), device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    a1_gscale = torch.ones((e,), device=DEVICE, dtype=torch.float32)
    a2_gscale = torch.ones((e,), device=DEVICE, dtype=torch.float32)
    g1_alphas = (1.0 / w1_gs).to(torch.float32)
    g2_alphas = (1.0 / w2_gs).to(torch.float32)
    return (
        hidden_states,
        topk_weights,
        topk_ids,
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    )


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    [
        (1, False),
        (2, False),
        (1, True),
    ],
)
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_matches_cutlass(
    topk: int,
    apply_router_weight_on_input: bool,
    workspace_init,
) -> None:
    set_random_seed(11)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        (
            hidden_states,
            topk_weights,
            topk_ids,
            w1_q,
            w1_blockscale,
            w2_q,
            w2_blockscale,
            a1_gscale,
            a2_gscale,
            g1_alphas,
            g2_alphas,
        ) = _make_nvfp4_moe_tensors(m=32, n=128, k=128, e=8, topk=topk)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            CutlassExpertsFp4(
                moe_config=make_dummy_moe_config(),
                quant_config=quant_config,
            ),
            inplace=False,
        )

        cutlass_out = kernel(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        fallback_out = fused_moe_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w1_q,
            w13_weight_scale=w1_blockscale,
            w2_weight=w2_q,
            w2_weight_scale=w2_blockscale,
            a1_gscale=a1_gscale,
            g1_alphas=g1_alphas,
            a2_gscale=a2_gscale,
            g2_alphas=g2_alphas,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        torch.testing.assert_close(fallback_out, cutlass_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    [
        (1, False),
        (2, False),
        (1, True),
    ],
)
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_batch_size_invariance(
    topk: int,
    apply_router_weight_on_input: bool,
) -> None:
    set_random_seed(13)
    e, n, k = 8, 128, 128
    x_single = torch.randn((1, k), device=DEVICE, dtype=DTYPE) / 10
    x_batch = torch.cat(
        [x_single, torch.randn((7, k), device=DEVICE, dtype=DTYPE) / 10], dim=0
    )

    (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
        make_test_weights(
            e,
            n,
            k,
            in_dtype=DTYPE,
            quant_dtype="nvfp4",
            block_shape=None,
            per_out_ch_quant=False,
        )
    )
    assert w1_blockscale is not None and w2_blockscale is not None
    assert w1_gs is not None and w2_gs is not None

    a1_gscale = torch.ones((e,), device=DEVICE, dtype=torch.float32)
    a2_gscale = torch.ones((e,), device=DEVICE, dtype=torch.float32)
    g1_alphas = (1.0 / w1_gs).to(torch.float32)
    g2_alphas = (1.0 / w2_gs).to(torch.float32)

    topk_ids_single = torch.randint(0, e, (1, topk), device=DEVICE, dtype=torch.int64)
    topk_ids_batch = torch.cat(
        [topk_ids_single, torch.randint(0, e, (7, topk), device=DEVICE)], dim=0
    ).to(torch.int64)

    topk_weights_single = torch.rand((1, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_single /= topk_weights_single.sum(dim=-1, keepdim=True)
    topk_weights_batch = torch.rand((8, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_batch /= topk_weights_batch.sum(dim=-1, keepdim=True)
    topk_weights_batch[0] = topk_weights_single[0]

    out_single = fused_moe_batch_invariant_nvfp4(
        hidden_states=x_single,
        topk_ids=topk_ids_single,
        topk_weights=topk_weights_single,
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    out_batch = fused_moe_batch_invariant_nvfp4(
        hidden_states=x_batch,
        topk_ids=topk_ids_batch,
        topk_weights=topk_weights_batch,
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert torch.equal(out_single[0], out_batch[0])
