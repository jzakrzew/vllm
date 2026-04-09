# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.batch_invariant_fp4_moe import (
    _grouped_matmul_mxfp4_packed,
    _grouped_matmul_nvfp4_packed,
    _quantize_mxfp8_experts,
    fused_moe_batch_invariant_mxfp4,
    fused_moe_batch_invariant_nvfp4,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    prepare_weights_for_nvfp4_cutlass,
    slice_nvfp4_output,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.torch_utils import set_random_seed

HAS_SM90 = current_platform.has_device_capability(90)
HAS_SM100 = current_platform.has_device_capability(100)
REQUIRES_SM100 = pytest.mark.skipif(
    not HAS_SM100,
    reason="Batch-invariant NVFP4 and MXFP8-activation "
    "coverage requires Blackwell (sm100+).",
)
USES_MXFP8_ACTIVATIONS = HAS_SM100

if not HAS_SM90:
    pytest.skip(
        reason="Batch-invariant FP4 MoE requires Hopper or newer (sm90+).",
        allow_module_level=True,
    )

if not hasattr(tl, "dot_scaled"):
    pytest.skip(
        reason="Installed Triton build does not expose tl.dot_scaled.",
        allow_module_level=True,
    )

DTYPE = torch.bfloat16
DEVICE = "cuda:0"
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0


def _maybe_quantize_mxfp8_reference(x: torch.Tensor) -> torch.Tensor:
    if not USES_MXFP8_ACTIVATIONS:
        return x
    x_q, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    return dequant_mxfp8_to_bf16(x_q, x_scale).to(x.dtype)


def _batch_invariant_fp4_workspaces(
    m: int,
    topk: int,
    w1_row_dim: int,
    hidden_dim: int,
    activation: MoEActivation,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scratch buffers matching ``BatchInvariantFP4Experts.workspace_shapes``."""
    act_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        w1_row_dim, activation
    )
    m_total = m * topk
    workspace13 = torch.empty(
        (m_total, max(w1_row_dim, hidden_dim)), device=device, dtype=dtype
    )
    workspace2 = torch.empty(
        (m_total, max(act_out_dim, hidden_dim)), device=device, dtype=dtype
    )
    return workspace13, workspace2


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max()).to(torch.float32)


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


@REQUIRES_SM100
@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    [
        (1, False),
        (2, False),
        (4, False),
        (1, True),
    ],
)
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_matches_cutlass(
    topk: int,
    apply_router_weight_on_input: bool,
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

        w13, w2 = _batch_invariant_fp4_workspaces(
            m=32,
            topk=topk,
            w1_row_dim=w1_q.shape[1],
            hidden_dim=128,
            activation=MoEActivation.SILU,
            device=DEVICE,
            dtype=DTYPE,
        )

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )
        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            CutlassExpertsFp4(
                moe_config=make_dummy_moe_config(),
                quant_config=quant_config,
            ),
            inplace=False,
        )

        cutlass_out = kernel.apply(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        fallback_out = torch.empty_like(hidden_states)
        fused_moe_batch_invariant_nvfp4(
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
            workspace13=w13,
            workspace2=w2,
            output=fallback_out,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        torch.testing.assert_close(fallback_out, cutlass_out, atol=1e-1, rtol=1e-1)


@REQUIRES_SM100
@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    [
        (1, False),
        (2, False),
        (4, False),
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

    topk_ids_single = torch.randint(0, e, (1, topk), device=DEVICE, dtype=torch.int32)
    topk_ids_batch = torch.cat(
        [topk_ids_single, torch.randint(0, e, (7, topk), device=DEVICE)], dim=0
    ).to(torch.int64)

    topk_weights_single = torch.rand((1, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_single /= topk_weights_single.sum(dim=-1, keepdim=True)
    topk_weights_batch = torch.rand((8, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_batch /= topk_weights_batch.sum(dim=-1, keepdim=True)
    topk_weights_batch[0] = topk_weights_single[0]

    w13_1, w2_1 = _batch_invariant_fp4_workspaces(
        m=1,
        topk=topk,
        w1_row_dim=w1_q.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )
    w13_8, w2_8 = _batch_invariant_fp4_workspaces(
        m=8,
        topk=topk,
        w1_row_dim=w1_q.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_single = torch.empty_like(x_single)
    fused_moe_batch_invariant_nvfp4(
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
        workspace13=w13_1,
        workspace2=w2_1,
        output=out_single,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    out_batch = torch.empty_like(x_batch)
    fused_moe_batch_invariant_nvfp4(
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
        workspace13=w13_8,
        workspace2=w2_8,
        output=out_batch,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert torch.equal(out_single[0], out_batch[0])


# NVFP4 MoE does not support expert parallelism yet; this test fails until it does.
@pytest.mark.skip(
    reason="NVFP4 MoE expert parallelism not supported; test fails until then.",
)
@REQUIRES_SM100
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_ignores_invalid_sentinel_routes() -> None:
    set_random_seed(23)
    m, e, n, k = 32, 8, 128, 128
    (
        hidden_states,
        _,
        _,
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_moe_tensors(m=m, n=n, k=k, e=e, topk=1)

    valid_topk_ids = torch.randint(0, e, (m, 1), device=DEVICE, dtype=torch.int32)
    valid_topk_weights = torch.rand((m, 1), device=DEVICE, dtype=torch.float32)
    invalid_topk_ids = torch.full((m, 1), -1, device=DEVICE, dtype=torch.int32)
    invalid_topk_weights = torch.rand((m, 1), device=DEVICE, dtype=torch.float32)

    topk_ids_with_invalid = torch.cat([valid_topk_ids, invalid_topk_ids], dim=1)
    topk_weights_with_invalid = torch.cat(
        [valid_topk_weights, invalid_topk_weights], dim=1
    )

    w13, w2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=2,
        w1_row_dim=w1_q.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_with_invalid = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids_with_invalid,
        topk_weights=topk_weights_with_invalid,
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=MoEActivation.SILU,
        workspace13=w13,
        workspace2=w2,
        output=out_with_invalid,
    )
    out_valid_only = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_ids=valid_topk_ids,
        topk_weights=valid_topk_weights,
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=MoEActivation.SILU,
        workspace13=w13,
        workspace2=w2,
        output=out_valid_only,
    )

    torch.testing.assert_close(out_with_invalid, out_valid_only, atol=1e-1, rtol=1e-1)


# NVFP4 MoE does not support expert parallelism yet; this test fails until it does.
@pytest.mark.skip(
    reason="NVFP4 MoE expert parallelism not supported; test fails until then.",
)
@REQUIRES_SM100
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_expert_map_invalidation_matches_local_routes() -> (
    None
):
    set_random_seed(29)
    m, e_local, n, k = 32, 4, 128, 128
    (
        hidden_states,
        _,
        _,
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_moe_tensors(m=m, n=n, k=k, e=e_local, topk=1)

    global_num_experts = 8
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device=DEVICE)
    mapped_global_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device=DEVICE)
    expert_map[mapped_global_ids] = torch.arange(
        e_local, dtype=torch.int32, device=DEVICE
    )

    valid_global_ids = mapped_global_ids[
        torch.randint(0, mapped_global_ids.numel(), (m, 1), device=DEVICE)
    ]
    invalid_global_candidates = torch.tensor(
        [0, 2, 4, 6], dtype=torch.int64, device=DEVICE
    )
    invalid_global_ids = invalid_global_candidates[
        torch.randint(0, invalid_global_candidates.numel(), (m, 1), device=DEVICE)
    ]
    topk_ids_global = torch.cat([valid_global_ids, invalid_global_ids], dim=1).to(
        torch.int32
    )
    topk_weights_global = torch.rand((m, 2), device=DEVICE, dtype=torch.float32)

    w13_g, w2_g = _batch_invariant_fp4_workspaces(
        m=m,
        topk=2,
        w1_row_dim=w1_q.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )
    w13_l, w2_l = _batch_invariant_fp4_workspaces(
        m=m,
        topk=1,
        w1_row_dim=w1_q.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_with_expert_map = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids_global,
        topk_weights=topk_weights_global,
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=MoEActivation.SILU,
        workspace13=w13_g,
        workspace2=w2_g,
        output=out_with_expert_map,
        expert_map=expert_map,
    )

    topk_ids_local = expert_map[topk_ids_global[:, :1]].to(torch.int32)
    assert torch.all(topk_ids_local >= 0)
    out_local_only = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids_local,
        topk_weights=topk_weights_global[:, :1].contiguous(),
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=MoEActivation.SILU,
        workspace13=w13_l,
        workspace2=w2_l,
        output=out_local_only,
        expert_map=None,
    )

    torch.testing.assert_close(
        out_with_expert_map, out_local_only, atol=1e-1, rtol=1e-1
    )


@REQUIRES_SM100
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_all_invalid_routes_return_zero() -> None:
    set_random_seed(31)
    m, e, n, k = 16, 8, 128, 128
    (
        hidden_states,
        _,
        _,
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_moe_tensors(m=m, n=n, k=k, e=e, topk=1)

    topk_ids = torch.full((m, 2), -1, dtype=torch.int32, device=DEVICE)
    topk_weights = torch.rand((m, 2), dtype=torch.float32, device=DEVICE)
    w13_z, w2_z = _batch_invariant_fp4_workspaces(
        m=m,
        topk=2,
        w1_row_dim=w1_q.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )
    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_nvfp4(
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
        workspace13=w13_z,
        workspace2=w2_z,
        output=out,
    )

    torch.testing.assert_close(out, torch.zeros_like(out), atol=0.0, rtol=0.0)


@REQUIRES_SM100
@torch.inference_mode()
def test_grouped_matmul_nvfp4_packed_matches_cutlass_reference() -> None:
    set_random_seed(17)

    # Non-uniform M across experts exercises offset/problem-size dispatch.
    per_expert_rows = [17, 131, 64, 5]
    E = len(per_expert_rows)
    N = 128
    K = 256

    xs = [
        torch.randn((rows, K), dtype=DTYPE, device=DEVICE) for rows in per_expert_rows
    ]
    ws = [torch.randn((N, K), dtype=DTYPE, device=DEVICE) for _ in range(E)]

    input_gs = [_global_scale(x) for x in xs]
    weight_gs = [_global_scale(w) for w in ws]
    alphas = [
        (1.0 / (ig * wg)).to(torch.float32) for ig, wg in zip(input_gs, weight_gs)
    ]

    packed_a_fp4 = []
    packed_a_scale = []
    packed_b_fp4 = []
    packed_b_scale = []
    ref_outputs = []
    expert_offsets = []
    a_scale_offsets = []

    row_offset = 0
    scale_row_offset = 0
    for expert_id, rows in enumerate(per_expert_rows):
        a_fp4, a_scale = scaled_fp4_quant(
            xs[expert_id], input_gs[expert_id], is_sf_swizzled_layout=True
        )
        b_fp4, b_scale_raw = scaled_fp4_quant(ws[expert_id], weight_gs[expert_id])
        b_fp4, b_scale, weights_padding_cols = prepare_weights_for_nvfp4_cutlass(
            b_fp4, b_scale_raw
        )
        a_fp4 = pad_nvfp4_activation_for_cutlass(a_fp4, weights_padding_cols)

        # Reference: one CUTLASS launch per expert.
        ref = cutlass_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            a_scale,
            b_scale,
            alphas[expert_id],
            DTYPE,
        )
        ref_outputs.append(slice_nvfp4_output(ref, N))

        packed_a_fp4.append(a_fp4)
        packed_a_scale.append(a_scale)
        packed_b_fp4.append(b_fp4)
        packed_b_scale.append(b_scale)
        expert_offsets.append(row_offset)
        a_scale_offsets.append(scale_row_offset)
        row_offset += rows
        scale_row_offset += a_scale.shape[0]

    packed_a_fp4_t = torch.cat(packed_a_fp4, dim=0)
    packed_a_scale_t = torch.cat(packed_a_scale, dim=0)
    packed_b_fp4_t = torch.stack(packed_b_fp4, dim=0)
    packed_b_scale_t = torch.stack(packed_b_scale, dim=0)
    alpha_t = torch.stack(alphas)
    expert_offsets_t = torch.tensor(expert_offsets, dtype=torch.int32, device=DEVICE)
    a_scale_offsets_t = torch.tensor(a_scale_offsets, dtype=torch.int32, device=DEVICE)
    problem_sizes_t = torch.tensor(
        [[rows, N, K] for rows in per_expert_rows],
        dtype=torch.int32,
        device=DEVICE,
    )

    packed_out = torch.empty((row_offset, N), device=DEVICE, dtype=DTYPE)
    _grouped_matmul_nvfp4_packed(
        a_fp4=packed_a_fp4_t,
        b_fp4=packed_b_fp4_t,
        a_scale=packed_a_scale_t,
        b_scale=packed_b_scale_t,
        alpha=alpha_t,
        expert_offsets=expert_offsets_t,
        a_scale_offsets=a_scale_offsets_t,
        problem_sizes=problem_sizes_t,
        output=packed_out,
    )
    if packed_out.shape[1] != N:
        packed_out = packed_out[:, :N].contiguous()

    ref_cat = torch.cat(ref_outputs, dim=0)
    torch.testing.assert_close(packed_out, ref_cat, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# MXFP4 (W4A16) tests
# ---------------------------------------------------------------------------

MXFP4_BLOCK = 32
FP4_LOOKUP = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def _quantize_mxfp4_block(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise FP4 e2m1 quantize with e8m0fnu block scales (group=32).

    Args:
        w_bf16: [N, K] bfloat16 weight matrix (K must be divisible by 32).

    Returns:
        w_fp4: [N, K // 2] uint8 packed (two e2m1 per byte, low nibble first).
        w_scale: [N, K // 32] uint8 representing e8m0fnu block scales.
    """
    N, K = w_bf16.shape
    assert K % MXFP4_BLOCK == 0
    w = w_bf16.float().reshape(N, K // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)

    # e8m0fnu: power-of-two scale = 2^exponent, stored as raw uint8 exponent.
    log2_amax = torch.log2(amax / 6.0).clamp(min=-127)
    exponent = torch.floor(log2_amax).to(torch.int32) + 127
    exponent = exponent.clamp(0, 254)
    scale_f32 = torch.pow(2.0, (exponent - 127).float())

    w_scaled = w / scale_f32
    lookup = FP4_LOOKUP.to(w.device)
    w_flat = w_scaled.reshape(-1, 1)
    dists = (w_flat - lookup.unsqueeze(0)).abs()
    indices = dists.argmin(dim=-1).reshape(N, K // MXFP4_BLOCK, MXFP4_BLOCK)
    indices = indices.to(torch.uint8).reshape(N, K // 2, 2)
    packed = (indices[..., 0] & 0x0F) | ((indices[..., 1] & 0x0F) << 4)
    w_scale = exponent.squeeze(-1).to(torch.uint8).reshape(N, K // MXFP4_BLOCK)
    return packed, w_scale


def _dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed weights back to float32 for reference."""
    N = packed.shape[0]
    K_half = packed.shape[1]
    unpacked = torch.zeros(N, K_half * 2, dtype=torch.int32, device=packed.device)
    p = packed.to(torch.int32)
    unpacked[:, 0::2] = p & 0x0F
    unpacked[:, 1::2] = (p >> 4) & 0x0F
    lookup = FP4_LOOKUP.to(packed.device)
    w_float = lookup[unpacked.long()]
    scale_f32 = (
        torch.pow(2.0, (scale.to(torch.int32) - 127).float())
        .unsqueeze(-1)
        .expand(N, -1, MXFP4_BLOCK)
        .reshape(N, K_half * 2)
    )
    return w_float * scale_f32


def _swizzle_mxfp4_scale(scale_uint8: torch.Tensor) -> torch.Tensor:
    """Swizzle MXFP4 uint8 scales into the 128x4 block-interleaved TMA layout.

    Uses the same permutation as NVFP4's ``swizzle_blockscale`` but operates
    on raw uint8 instead of float8_e4m3fn.
    """
    from vllm.utils.math_utils import round_up

    if scale_uint8.ndim == 2:
        scale_uint8 = scale_uint8.unsqueeze(0)
    B, M, K = scale_uint8.shape
    M_pad = round_up(M, 128)
    K_pad = round_up(K, 4)
    padded = torch.zeros(B, M_pad, K_pad, dtype=torch.uint8, device=scale_uint8.device)
    padded[:, :M, :K] = scale_uint8
    padded = padded.reshape(B, M_pad // 128, 4, 32, K_pad // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous()
    return swizzled.reshape(B, M_pad, K_pad)


def _make_mxfp4_moe_weights(
    *, e: int, n: int, k: int, interleaved: bool = False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create MXFP4 MoE weight tensors (w13 and w2) with swizzled scales.

    Args:
        e: number of experts.
        n: intermediate size per projection (gate or up).
        k: hidden size.
        interleaved: if True, return w13 weights and scales with interleaved
            gate/up rows ``[g0, u0, g1, u1, ...]`` (checkpoint format) and
            raw (un-swizzled) scales.  The caller is responsible for
            de-interleaving and swizzling, e.g. via
            ``convert_to_mxfp4_moe_kernel_format``.

    Returns:
        w13_fp4: [E, 2*N, K//2] uint8
        w13_scale: [E, 2*N_pad, K_s_pad] uint8 swizzled (or raw if
            interleaved)
        w2_fp4: [E, K, N//2] uint8
        w2_scale: [E, K_pad, N_s_pad] uint8 swizzled
        w13_scale_raw: [E, 2*N, K//32] uint8 (row-major, for dequant
            reference; always in concatenated gate-then-up order)
        w2_scale_raw: [E, K, N//32] uint8
    """

    w13_list, w13_scale_list = [], []
    w2_list, w2_scale_list = [], []
    for _ in range(e):
        w1_bf16 = torch.randn(n, k, device=DEVICE, dtype=DTYPE) / 10
        w3_bf16 = torch.randn(n, k, device=DEVICE, dtype=DTYPE) / 10
        w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=0)  # [2N, K]
        w13_q, w13_s = _quantize_mxfp4_block(w13_bf16)
        w13_list.append(w13_q)
        w13_scale_list.append(w13_s)

        w2_bf16 = torch.randn(k, n, device=DEVICE, dtype=DTYPE) / 10
        w2_q, w2_s = _quantize_mxfp4_block(w2_bf16)
        w2_list.append(w2_q)
        w2_scale_list.append(w2_s)

    w13_fp4 = torch.stack(w13_list)
    w13_scale_raw = torch.stack(w13_scale_list)
    w2_fp4 = torch.stack(w2_list)
    w2_scale_raw = torch.stack(w2_scale_list)

    if interleaved:
        # Simulate checkpoint format: interleave gate/up rows and return
        # raw (un-swizzled) scales so the caller can run the full
        # convert_to_mxfp4_moe_kernel_format pipeline.
        w13_fp4 = _interleave_gate_up(w13_fp4)
        w13_scale_interleaved = _interleave_gate_up(w13_scale_raw)
        return (
            w13_fp4,
            w13_scale_interleaved,
            w2_fp4,
            w2_scale_raw,
            w13_scale_raw,
            w2_scale_raw,
        )

    w13_scale = _swizzle_mxfp4_scale(w13_scale_raw)
    w2_scale = _swizzle_mxfp4_scale(w2_scale_raw)

    return w13_fp4, w13_scale, w2_fp4, w2_scale, w13_scale_raw, w2_scale_raw


def _interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Interleave the gate and up halves along dim 1.

    [E, 2*N, ...] with layout [g0,g1,...,u0,u1,...] ->
    [E, 2*N, ...] with layout [g0,u0,g1,u1,...].
    """
    half = t.shape[1] // 2
    gate, up = t[:, :half], t[:, half:]
    paired = torch.stack([gate, up], dim=2)  # [E, N, 2, ...]
    return paired.reshape(t.shape[0], t.shape[1], *t.shape[2:]).contiguous()


def _map_topk_with_expert_map(
    topk_ids: torch.Tensor, expert_map: torch.Tensor
) -> torch.Tensor:
    """Same global→local mapping as ``_nvfp4_moe_map_experts`` (for reference MoE)."""
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    if expert_map.numel() == 0:
        return torch.full_like(topk_ids, -1, dtype=torch.long)
    valid = (flat_ids >= 0) & (flat_ids < expert_map.numel())
    clamped = flat_ids.clamp(min=0, max=max(0, expert_map.numel() - 1))
    remapped = expert_map.to(torch.long).index_select(0, clamped)
    mapped = torch.where(valid, remapped, torch.full_like(remapped, -1))
    return mapped.view_as(topk_ids)


def _reference_fused_moe_mxfp4_dequant(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_fp4: torch.Tensor,
    w13_scale_raw: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_scale_raw: torch.Tensor,
    activation: MoEActivation,
    *,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference MXFP4 MoE with BF16 or MXFP8-quantized activations."""
    device = hidden_states.device
    dtype = hidden_states.dtype
    m, hidden_dim = hidden_states.shape
    num_experts = w13_fp4.shape[0]
    top_k = topk_ids.shape[1]

    act_kind = (
        activation
        if isinstance(activation, MoEActivation)
        else MoEActivation.from_str(str(activation))
    )

    w13_deq = [
        _dequant_mxfp4(w13_fp4[e], w13_scale_raw[e]).to(dtype)
        for e in range(num_experts)
    ]
    w2_deq = [
        _dequant_mxfp4(w2_fp4[e], w2_scale_raw[e]).to(dtype) for e in range(num_experts)
    ]

    routed = topk_ids.to(torch.long)
    if expert_map is not None:
        routed = _map_topk_with_expert_map(topk_ids, expert_map)
    valid_routes = (routed >= 0) & (routed < num_experts)
    routed = torch.where(
        valid_routes, routed, torch.full_like(routed, -1, dtype=torch.long)
    )
    rw = topk_weights.to(torch.float32)

    out = torch.zeros(m, hidden_dim, device=device, dtype=dtype)
    for i in range(m):
        acc = torch.zeros(1, hidden_dim, device=device, dtype=dtype)
        for j in range(top_k):
            eid = int(routed[i, j].item())
            if eid < 0:
                continue
            w = rw[i, j].to(dtype)
            x = hidden_states[i : i + 1]
            if apply_router_weight_on_input:
                x = x * w
            x = _maybe_quantize_mxfp8_reference(x)
            gemm1_out = x @ w13_deq[eid].T
            if w13_bias is not None:
                gemm1_out = gemm1_out + w13_bias[eid : eid + 1].to(dtype)
            if act_kind.is_gated:
                act_dim = gemm1_out.shape[-1] // 2
            else:
                act_dim = gemm1_out.shape[-1]
            act_buf = torch.empty(1, act_dim, device=device, dtype=dtype)
            apply_moe_activation(act_kind, act_buf, gemm1_out)
            act_buf = _maybe_quantize_mxfp8_reference(act_buf)
            y = act_buf @ w2_deq[eid].T
            if w2_bias is not None:
                y = y + w2_bias[eid : eid + 1].to(dtype)
            acc = acc + y if apply_router_weight_on_input else acc + y * w
        out[i] = acc[0]
    return out


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    [
        (1, False),
        (2, False),
        (4, False),
        (1, True),
    ],
)
@pytest.mark.parametrize(
    "m,e,n,k",
    [
        (32, 8, 128, 256),
        (64, 4, 256, 128),
        (128, 8, 256, 384),
    ],
)
@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_matches_dequant_reference(
    m: int,
    e: int,
    n: int,
    k: int,
    topk: int,
    apply_router_weight_on_input: bool,
) -> None:
    """``fused_moe_batch_invariant_mxfp4`` matches the device-aware reference."""
    set_random_seed(42)

    hidden_states = torch.randn(m, k, device=DEVICE, dtype=DTYPE) / 10
    score = torch.randn(m, e, device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    (
        w13_fp4,
        w13_scale,
        w2_fp4,
        w2_scale,
        w13_scale_raw,
        w2_scale_raw,
    ) = _make_mxfp4_moe_weights(e=e, n=n, k=k)

    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=topk,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=workspace13,
        workspace2=workspace2,
        output=out,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    ref = _reference_fused_moe_mxfp4_dequant(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_fp4=w13_fp4,
        w13_scale_raw=w13_scale_raw,
        w2_fp4=w2_fp4,
        w2_scale_raw=w2_scale_raw,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert out.shape == (m, k)
    assert out.dtype == DTYPE
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_interleaved_checkpoint_format() -> None:
    """End-to-end: interleaved w13 -> convert_to_mxfp4_moe_kernel_format
    -> fused_moe_batch_invariant_mxfp4 must match the dequant reference.

    Real MXFP4 checkpoints store w13 with interleaved gate/up rows
    [g0, u0, g1, u1, ...].  ``convert_to_mxfp4_moe_kernel_format`` must
    de-interleave before the kernel sees the weights.
    """
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_to_mxfp4_moe_kernel_format,
    )

    set_random_seed(99)
    m, e, n, k = 32, 8, 128, 256
    topk = 2

    hidden_states = torch.randn(m, k, device=DEVICE, dtype=DTYPE) / 10
    score = torch.randn(m, e, device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    (
        w13_fp4_interleaved,
        w13_scale_interleaved,
        w2_fp4,
        w2_scale,
        w13_scale_raw,
        w2_scale_raw,
    ) = _make_mxfp4_moe_weights(e=e, n=n, k=k, interleaved=True)

    # Run the production convert path (de-interleave + swizzle).
    (w13_fp4_conv, w2_conv, w13_scale_conv, w2_scale_conv, _, _) = (
        convert_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.BATCH_INVARIANT,
            layer=None,  # type: ignore[arg-type]
            w13_weight=w13_fp4_interleaved,
            w2_weight=w2_fp4,
            w13_weight_scale=w13_scale_interleaved,
            w2_weight_scale=w2_scale,
        )
    )

    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=topk,
        w1_row_dim=w13_fp4_conv.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_fp4_conv,
        w13_weight_scale=w13_scale_conv,
        w2_weight=w2_conv,
        w2_weight_scale=w2_scale_conv,
        activation=MoEActivation.SILU,
        workspace13=workspace13,
        workspace2=workspace2,
        output=out,
    )

    # The reference uses concatenated (non-interleaved) raw scales.
    ref = _reference_fused_moe_mxfp4_dequant(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_fp4=w13_fp4_conv,
        w13_scale_raw=w13_scale_raw,
        w2_fp4=w2_fp4,
        w2_scale_raw=w2_scale_raw,
        activation=MoEActivation.SILU,
    )

    assert out.shape == (m, k)
    assert out.dtype == DTYPE
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_batch_size_invariance() -> None:
    """The first token's output must be identical regardless of batch size."""
    set_random_seed(50)
    e, n, k = 8, 128, 256
    topk = 2

    x_single = torch.randn(1, k, device=DEVICE, dtype=DTYPE) / 10
    x_batch = torch.cat(
        [x_single, torch.randn(7, k, device=DEVICE, dtype=DTYPE) / 10], dim=0
    )

    w13_fp4, w13_scale, w2_fp4, w2_scale, _, _ = _make_mxfp4_moe_weights(e=e, n=n, k=k)

    topk_ids_single = torch.randint(0, e, (1, topk), device=DEVICE, dtype=torch.int32)
    topk_ids_batch = torch.cat(
        [topk_ids_single, torch.randint(0, e, (7, topk), device=DEVICE)], dim=0
    ).to(torch.int64)
    topk_weights_single = torch.rand(1, topk, device=DEVICE, dtype=torch.float32)
    topk_weights_single /= topk_weights_single.sum(dim=-1, keepdim=True)
    topk_weights_batch = torch.rand(8, topk, device=DEVICE, dtype=torch.float32)
    topk_weights_batch /= topk_weights_batch.sum(dim=-1, keepdim=True)
    topk_weights_batch[0] = topk_weights_single[0]

    w13_1, w2_1 = _batch_invariant_fp4_workspaces(
        m=1,
        topk=topk,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )
    w13_8, w2_8 = _batch_invariant_fp4_workspaces(
        m=8,
        topk=topk,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_single = torch.empty_like(x_single)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=x_single,
        topk_ids=topk_ids_single,
        topk_weights=topk_weights_single,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13_1,
        workspace2=w2_1,
        output=out_single,
    )
    out_batch = torch.empty_like(x_batch)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=x_batch,
        topk_ids=topk_ids_batch,
        topk_weights=topk_weights_batch,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13_8,
        workspace2=w2_8,
        output=out_batch,
    )
    assert torch.equal(out_single[0], out_batch[0])


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_ignores_invalid_sentinel_routes() -> None:
    """Invalid -1 expert slots should be ignored; output matches valid-only top-1."""
    set_random_seed(23)
    m, e, n, k = 32, 8, 128, 256
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=DTYPE) / 10
    w13_fp4, w13_scale, w2_fp4, w2_scale, _, _ = _make_mxfp4_moe_weights(e=e, n=n, k=k)

    valid_topk_ids = torch.randint(0, e, (m, 1), device=DEVICE, dtype=torch.int32)
    valid_topk_weights = torch.rand((m, 1), device=DEVICE, dtype=torch.float32)
    invalid_topk_ids = torch.full((m, 1), -1, device=DEVICE, dtype=torch.int32)
    invalid_topk_weights = torch.rand((m, 1), device=DEVICE, dtype=torch.float32)

    topk_ids_with_invalid = torch.cat([valid_topk_ids, invalid_topk_ids], dim=1)
    topk_weights_with_invalid = torch.cat(
        [valid_topk_weights, invalid_topk_weights], dim=1
    )

    w13, w2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=2,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_with_invalid = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids_with_invalid,
        topk_weights=topk_weights_with_invalid,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13,
        workspace2=w2,
        output=out_with_invalid,
    )
    out_valid_only = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=valid_topk_ids,
        topk_weights=valid_topk_weights,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13,
        workspace2=w2,
        output=out_valid_only,
    )

    torch.testing.assert_close(out_with_invalid, out_valid_only, atol=1e-1, rtol=1e-1)


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_expert_map_invalidation_matches_local_routes() -> (
    None
):
    """Global IDs + expert_map must match local indices
    (no separate expert_map path)."""
    set_random_seed(29)
    m, e_local, n, k = 32, 4, 128, 256
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=DTYPE) / 10
    w13_fp4, w13_scale, w2_fp4, w2_scale, _, _ = _make_mxfp4_moe_weights(
        e=e_local, n=n, k=k
    )

    global_num_experts = 8
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device=DEVICE)
    mapped_global_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64, device=DEVICE)
    expert_map[mapped_global_ids] = torch.arange(
        e_local, dtype=torch.int32, device=DEVICE
    )

    valid_global_ids = mapped_global_ids[
        torch.randint(0, mapped_global_ids.numel(), (m, 1), device=DEVICE)
    ]
    invalid_global_candidates = torch.tensor(
        [0, 2, 4, 6], dtype=torch.int64, device=DEVICE
    )
    invalid_global_ids = invalid_global_candidates[
        torch.randint(0, invalid_global_candidates.numel(), (m, 1), device=DEVICE)
    ]
    topk_ids_global = torch.cat([valid_global_ids, invalid_global_ids], dim=1).to(
        torch.int32
    )
    topk_weights_global = torch.rand((m, 2), device=DEVICE, dtype=torch.float32)

    w13_g, w2_g = _batch_invariant_fp4_workspaces(
        m=m,
        topk=2,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )
    w13_l, w2_l = _batch_invariant_fp4_workspaces(
        m=m,
        topk=1,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out_with_expert_map = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids_global,
        topk_weights=topk_weights_global,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13_g,
        workspace2=w2_g,
        output=out_with_expert_map,
        expert_map=expert_map,
    )

    topk_ids_local = expert_map[topk_ids_global[:, :1]].to(torch.int32)
    assert torch.all(topk_ids_local >= 0)
    out_local_only = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids_local,
        topk_weights=topk_weights_global[:, :1].contiguous(),
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13_l,
        workspace2=w2_l,
        output=out_local_only,
        expert_map=None,
    )

    torch.testing.assert_close(
        out_with_expert_map, out_local_only, atol=1e-1, rtol=1e-1
    )


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_all_invalid_routes_return_zero() -> None:
    """All-invalid routing must produce a zero output."""
    set_random_seed(55)
    m, e, n, k = 16, 8, 128, 256

    hidden_states = torch.randn(m, k, device=DEVICE, dtype=DTYPE) / 10
    w13_fp4, w13_scale, w2_fp4, w2_scale, _, _ = _make_mxfp4_moe_weights(e=e, n=n, k=k)

    topk_ids = torch.full((m, 2), -1, dtype=torch.int32, device=DEVICE)
    topk_weights = torch.rand(m, 2, dtype=torch.float32, device=DEVICE)

    w13_z, w2_z = _batch_invariant_fp4_workspaces(
        m=m,
        topk=2,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=w13_z,
        workspace2=w2_z,
        output=out,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0.0, rtol=0.0)


def _make_mxfp4_moe_weights_non_gated(
    *,
    e: int,
    n: int,
    k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create MXFP4 MoE weights for non-gated activations.

    w1 is ``[E, N, K//2]`` (no gate/up split) and w2 is ``[E, K, N//2]``.
    """
    w1_list, w1_scale_list = [], []
    w2_list, w2_scale_list = [], []
    for _ in range(e):
        w1_bf16 = torch.randn(n, k, device=DEVICE, dtype=DTYPE) / 10
        w1_q, w1_s = _quantize_mxfp4_block(w1_bf16)
        w1_list.append(w1_q)
        w1_scale_list.append(w1_s)

        w2_bf16 = torch.randn(k, n, device=DEVICE, dtype=DTYPE) / 10
        w2_q, w2_s = _quantize_mxfp4_block(w2_bf16)
        w2_list.append(w2_q)
        w2_scale_list.append(w2_s)

    w1_fp4 = torch.stack(w1_list)
    w1_scale_raw = torch.stack(w1_scale_list)
    w2_fp4 = torch.stack(w2_list)
    w2_scale_raw = torch.stack(w2_scale_list)

    w1_scale = _swizzle_mxfp4_scale(w1_scale_raw)
    w2_scale = _swizzle_mxfp4_scale(w2_scale_raw)

    return w1_fp4, w1_scale, w2_fp4, w2_scale, w1_scale_raw, w2_scale_raw


@pytest.mark.parametrize(
    "activation",
    [
        MoEActivation.SILU_NO_MUL,
        MoEActivation.GELU_NO_MUL,
    ],
)
@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_non_gated_activation(
    activation: MoEActivation,
) -> None:
    """Non-gated activations exercise the ``is_gated=False`` problem-sizes path."""
    set_random_seed(70)
    m, e, n, k = 32, 4, 128, 256
    topk = 2

    hidden_states = torch.randn(m, k, device=DEVICE, dtype=DTYPE) / 10
    score = torch.randn(m, e, device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    w1_fp4, w1_scale, w2_fp4, w2_scale, w1_scale_raw, w2_scale_raw = (
        _make_mxfp4_moe_weights_non_gated(e=e, n=n, k=k)
    )

    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=topk,
        w1_row_dim=w1_fp4.shape[1],
        hidden_dim=k,
        activation=activation,
        device=DEVICE,
        dtype=DTYPE,
    )

    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w1_fp4,
        w13_weight_scale=w1_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=activation,
        workspace13=workspace13,
        workspace2=workspace2,
        output=out,
    )
    ref = _reference_fused_moe_mxfp4_dequant(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_fp4=w1_fp4,
        w13_scale_raw=w1_scale_raw,
        w2_fp4=w2_fp4,
        w2_scale_raw=w2_scale_raw,
        activation=activation,
    )
    assert out.shape == (m, k)
    assert out.dtype == DTYPE
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_large_multi_tile() -> None:
    """Larger shapes exercise multi-tile computation in M, N, and K."""
    set_random_seed(77)
    m, e, n, k = 256, 8, 512, 512
    topk = 2

    hidden_states = torch.randn(m, k, device=DEVICE, dtype=DTYPE) / 10
    score = torch.randn(m, e, device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    w13_fp4, w13_scale, w2_fp4, w2_scale, w13_scale_raw, w2_scale_raw = (
        _make_mxfp4_moe_weights(e=e, n=n, k=k)
    )

    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=topk,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=workspace13,
        workspace2=workspace2,
        output=out,
    )
    ref = _reference_fused_moe_mxfp4_dequant(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_fp4=w13_fp4,
        w13_scale_raw=w13_scale_raw,
        w2_fp4=w2_fp4,
        w2_scale_raw=w2_scale_raw,
        activation=MoEActivation.SILU,
    )
    assert out.shape == (m, k)
    assert out.dtype == DTYPE
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "per_expert_rows,N,K",
    [
        ([17, 131, 64, 5], 128, 256),
        ([32, 64], 256, 128),
        ([1, 200, 50], 256, 384),
        ([0, 64, 0, 32], 128, 256),
        ([140], 128, 256),
        ([17, 131, 64, 5], 384, 256),
    ],
)
@torch.inference_mode()
def test_grouped_matmul_mxfp4_packed_matches_reference(
    per_expert_rows: list[int],
    N: int,
    K: int,
) -> None:
    """_grouped_matmul_mxfp4_packed must match the device-aware reference."""
    set_random_seed(60)

    # Filter out zero-M experts for weight generation but keep them in
    # problem_sizes so the kernel must handle them.
    E = len(per_expert_rows)

    bf16_weights = [
        torch.randn(N, K, device=DEVICE, dtype=DTYPE) / 10 for _ in range(E)
    ]
    packed_b_fp4, packed_b_scale = [], []
    for w in bf16_weights:
        wq, ws = _quantize_mxfp4_block(w)
        packed_b_fp4.append(wq)
        packed_b_scale.append(ws)

    b_fp4_t = torch.stack(packed_b_fp4)  # [E, N, K//2]
    b_scale_raw = torch.stack(packed_b_scale)  # [E, N, K//32]
    b_scale_t = _swizzle_mxfp4_scale(b_scale_raw)  # [E, N_pad, K_s_pad]

    xs = [
        torch.randn(rows, K, device=DEVICE, dtype=DTYPE) / 10
        for rows in per_expert_rows
    ]
    a_bf16 = (
        torch.cat(xs, dim=0)
        if any(r > 0 for r in per_expert_rows)
        else (torch.empty(0, K, device=DEVICE, dtype=DTYPE))
    )

    expert_offsets = []
    row_offset = 0
    for rows in per_expert_rows:
        expert_offsets.append(row_offset)
        row_offset += rows
    expert_offsets_t = torch.tensor(expert_offsets, dtype=torch.int32, device=DEVICE)
    blockscale_offsets = []
    scale_row_offset = 0
    for rows in per_expert_rows:
        blockscale_offsets.append(scale_row_offset)
        scale_row_offset += ((rows + 127) // 128) * 128
    blockscale_offsets_t = torch.tensor(
        blockscale_offsets, dtype=torch.int32, device=DEVICE
    )
    problem_sizes_t = torch.tensor(
        [[rows, N, K] for rows in per_expert_rows],
        dtype=torch.int32,
        device=DEVICE,
    )

    packed_out = torch.empty((row_offset, b_fp4_t.shape[1]), device=DEVICE, dtype=DTYPE)
    if USES_MXFP8_ACTIVATIONS:
        a_fp8, a_scale = _quantize_mxfp8_experts(
            a_bf16,
            problem_sizes_t,
            expert_offsets_t,
            blockscale_offsets_t,
        )
        _grouped_matmul_mxfp4_packed(
            a_fp8,
            b_fp4_t,
            b_scale_t,
            expert_offsets_t,
            problem_sizes_t,
            output=packed_out,
            a_scale=a_scale,
            a_scale_offsets=blockscale_offsets_t,
        )
    else:
        _grouped_matmul_mxfp4_packed(
            a_bf16,
            b_fp4_t,
            b_scale_t,
            expert_offsets_t,
            problem_sizes_t,
            output=packed_out,
        )

    ref_outputs = []
    for i, rows in enumerate(per_expert_rows):
        if rows == 0:
            continue
        w_deq = _dequant_mxfp4(packed_b_fp4[i], packed_b_scale[i]).to(DTYPE)
        x_ref = _maybe_quantize_mxfp8_reference(xs[i])
        ref = x_ref @ w_deq.T
        ref_outputs.append(ref)

    if packed_out.shape[1] != N:
        packed_out = packed_out[:, :N].contiguous()
    if ref_outputs:
        ref_cat = torch.cat(ref_outputs, dim=0)
        torch.testing.assert_close(packed_out, ref_cat, atol=5e-2, rtol=5e-2)
    else:
        assert packed_out.numel() == 0


# ---------------------------------------------------------------------------
# MXFP4 expert bias tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    [
        (2, False),
        (1, True),
    ],
)
@pytest.mark.parametrize(
    "m,e,n,k",
    [
        (32, 8, 128, 256),
        (64, 4, 256, 128),
    ],
)
@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_with_bias(
    m: int,
    e: int,
    n: int,
    k: int,
    topk: int,
    apply_router_weight_on_input: bool,
) -> None:
    """``fused_moe_batch_invariant_mxfp4`` with non-zero expert biases
    must match the device-aware reference that also applies the biases."""
    set_random_seed(42)

    hidden_states = torch.randn(m, k, device=DEVICE, dtype=DTYPE) / 10
    score = torch.randn(m, e, device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    (
        w13_fp4,
        w13_scale,
        w2_fp4,
        w2_scale,
        w13_scale_raw,
        w2_scale_raw,
    ) = _make_mxfp4_moe_weights(e=e, n=n, k=k)

    w13_bias = torch.randn(e, 2 * n, device=DEVICE, dtype=torch.float32) / 10
    w2_bias = torch.randn(e, k, device=DEVICE, dtype=torch.float32) / 10

    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=m,
        topk=topk,
        w1_row_dim=w13_fp4.shape[1],
        hidden_dim=k,
        activation=MoEActivation.SILU,
        device=DEVICE,
        dtype=DTYPE,
    )

    out = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_mxfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w13_fp4,
        w13_weight_scale=w13_scale,
        w2_weight=w2_fp4,
        w2_weight_scale=w2_scale,
        activation=MoEActivation.SILU,
        workspace13=workspace13,
        workspace2=workspace2,
        output=out,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    ref = _reference_fused_moe_mxfp4_dequant(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_fp4=w13_fp4,
        w13_scale_raw=w13_scale_raw,
        w2_fp4=w2_fp4,
        w2_scale_raw=w2_scale_raw,
        activation=MoEActivation.SILU,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert out.shape == (m, k)
    assert out.dtype == DTYPE
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
