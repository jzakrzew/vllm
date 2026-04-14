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
    NvFP4MoEWorkspace,
    _grouped_matmul_nvfp4_packed,
    _map_extra_rows,
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
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Batch-invariant FP4 MoE requires Blackwell or newer (sm100+).",
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
BATCH_INVARIANT_CASES = [
    (1, False),
    (2, False),
    (4, False),
    (1, True),
]


@pytest.fixture(autouse=True)
def _triton_allocator_for_tma_kernels():
    """tl.make_tensor_descriptor in grouped NVFP4 GEMM needs triton.set_allocator."""
    set_triton_allocator(torch.device(DEVICE))
    yield


def _batch_invariant_fp4_workspaces(
    m: int,
    topk: int,
    w1_row_dim: int,
    hidden_dim: int,
    activation: MoEActivation,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scratch buffers matching ``BatchInvariantNvfp4Experts.workspace_shapes``."""
    act_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        w1_row_dim, activation
    )
    m_total = m * topk
    cols13 = max(w1_row_dim, hidden_dim)
    extra = _map_extra_rows(m_total, cols13)
    workspace13 = torch.empty((m_total + extra, cols13), device=device, dtype=dtype)
    workspace2 = torch.empty(
        (m_total, max(act_out_dim, hidden_dim)), device=device, dtype=dtype
    )
    return workspace13, workspace2


def _make_nvfp4_workspace(
    num_experts: int,
    device: torch.device | str = DEVICE,
) -> NvFP4MoEWorkspace:
    """Create an ``NvFP4MoEWorkspace`` for test helpers."""
    return NvFP4MoEWorkspace(
        expert_offsets=torch.empty(num_experts + 1, dtype=torch.int32, device=device),
        blockscale_offsets=torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        ),
        problem_sizes1=torch.empty(num_experts, 3, dtype=torch.int32, device=device),
        problem_sizes2=torch.empty(num_experts, 3, dtype=torch.int32, device=device),
        tiles_per_expert=torch.empty(num_experts, dtype=torch.int32, device=device),
        expert_tile_start=torch.zeros(
            num_experts + 1, dtype=torch.int32, device=device
        ),
        dummy_bias=torch.empty(0, device=device, dtype=torch.float32),
    )


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max()).to(torch.float32)


def _make_nvfp4_weights_and_scales(
    *,
    e: int,
    n: int,
    k: int,
    scale_source: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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

    a1_gscale = _global_scale(scale_source).expand(e).contiguous()
    a2_gscale = _global_scale(scale_source).expand(e).contiguous()
    g1_alphas = ((1.0 / w1_gs) / a1_gscale).to(torch.float32)
    g2_alphas = ((1.0 / w2_gs) / a2_gscale).to(torch.float32)
    return (
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    )


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
    (
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_weights_and_scales(
        e=e,
        n=n,
        k=k,
        scale_source=hidden_states,
    )

    score = torch.randn((m, e), device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

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


def _run_batch_invariant_nvfp4(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_q: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_q: torch.Tensor,
    w2_blockscale: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=hidden_states.shape[0],
        topk=topk_ids.shape[1],
        w1_row_dim=w1_q.shape[1],
        hidden_dim=hidden_states.shape[1],
        activation=activation,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = torch.empty_like(hidden_states)
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
        activation=activation,
        workspace13=workspace13,
        workspace2=workspace2,
        output=output,
        workspace=_make_nvfp4_workspace(w1_q.shape[0]),
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=expert_map,
    )
    return output


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    BATCH_INVARIANT_CASES,
)
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_matches_cutlass(
    workspace_init,
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

        fallback_out = _run_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_q=w1_q,
            w1_blockscale=w1_blockscale,
            w2_q=w2_q,
            w2_blockscale=w2_blockscale,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        torch.testing.assert_close(fallback_out, cutlass_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    BATCH_INVARIANT_CASES,
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

    (
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_weights_and_scales(
        e=e,
        n=n,
        k=k,
        scale_source=x_batch,
    )

    topk_ids_single = torch.randint(0, e, (1, topk), device=DEVICE, dtype=torch.int32)
    topk_ids_batch = torch.cat(
        [
            topk_ids_single,
            torch.randint(0, e, (7, topk), device=DEVICE, dtype=torch.int32),
        ],
        dim=0,
    )

    topk_weights_single = torch.rand((1, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_single /= topk_weights_single.sum(dim=-1, keepdim=True)
    topk_weights_batch = torch.rand((8, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_batch /= topk_weights_batch.sum(dim=-1, keepdim=True)
    topk_weights_batch[0] = topk_weights_single[0]

    out_single = _run_batch_invariant_nvfp4(
        hidden_states=x_single,
        topk_weights=topk_weights_single,
        topk_ids=topk_ids_single,
        w1_q=w1_q,
        w1_blockscale=w1_blockscale,
        w2_q=w2_q,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    out_batch = _run_batch_invariant_nvfp4(
        hidden_states=x_batch,
        topk_weights=topk_weights_batch,
        topk_ids=topk_ids_batch,
        w1_q=w1_q,
        w1_blockscale=w1_blockscale,
        w2_q=w2_q,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert torch.equal(out_single[0], out_batch[0])


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
    out = _run_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_q=w1_q,
        w1_blockscale=w1_blockscale,
        w2_q=w2_q,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
    )

    torch.testing.assert_close(out, torch.zeros_like(out), atol=0.0, rtol=0.0)


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
        b_scale = swizzle_blockscale(b_scale_raw)
        b_fp4, weights_padding_cols = pad_nvfp4_weight_for_cutlass(b_fp4)
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

    expert_offsets.append(row_offset)
    a_scale_offsets.append(scale_row_offset)

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
        tiles_per_expert=torch.empty(E, dtype=torch.int32, device=DEVICE),
        expert_tile_start=torch.zeros(E + 1, dtype=torch.int32, device=DEVICE),
        dummy_bias=torch.empty(0, device=DEVICE, dtype=torch.float32),
    )
    if packed_out.shape[1] != N:
        packed_out = packed_out[:, :N].contiguous()

    ref_cat = torch.cat(ref_outputs, dim=0)
    torch.testing.assert_close(packed_out, ref_cat, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# MXFP4 (W4A8) tests
# ---------------------------------------------------------------------------

MXFP4_BLOCK = 32
FP4_LOOKUP = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def _maybe_quantize_mxfp8_reference(x: torch.Tensor) -> torch.Tensor:
    x_q, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=False)
    return dequant_mxfp8_to_bf16(x_q, x_scale).to(x.dtype)


def _quantize_mxfp4_block(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise FP4 e2m1 quantize with e8m0fnu block scales (group=32)."""
    n, k = w_bf16.shape
    assert k % MXFP4_BLOCK == 0
    w = w_bf16.float().reshape(n, k // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)

    log2_amax = torch.log2(amax / 6.0).clamp(min=-127)
    exponent = torch.floor(log2_amax).to(torch.int32) + 127
    exponent = exponent.clamp(0, 254)
    scale_f32 = torch.pow(2.0, (exponent - 127).float())

    w_scaled = w / scale_f32
    lookup = FP4_LOOKUP.to(w.device)
    w_flat = w_scaled.reshape(-1, 1)
    dists = (w_flat - lookup.unsqueeze(0)).abs()
    indices = dists.argmin(dim=-1).reshape(n, k // MXFP4_BLOCK, MXFP4_BLOCK)
    indices = indices.to(torch.uint8).reshape(n, k // 2, 2)
    packed = (indices[..., 0] & 0x0F) | ((indices[..., 1] & 0x0F) << 4)
    w_scale = exponent.squeeze(-1).to(torch.uint8).reshape(n, k // MXFP4_BLOCK)
    return packed, w_scale


def _dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed weights back to float32 for reference."""
    n = packed.shape[0]
    k_half = packed.shape[1]
    unpacked = torch.zeros(n, k_half * 2, dtype=torch.int32, device=packed.device)
    p = packed.to(torch.int32)
    unpacked[:, 0::2] = p & 0x0F
    unpacked[:, 1::2] = (p >> 4) & 0x0F
    lookup = FP4_LOOKUP.to(packed.device)
    w_float = lookup[unpacked.long()]
    scale_f32 = (
        torch.pow(2.0, (scale.to(torch.int32) - 127).float())
        .unsqueeze(-1)
        .expand(n, -1, MXFP4_BLOCK)
        .reshape(n, k_half * 2)
    )
    return w_float * scale_f32


def _interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Interleave the gate and up halves along dim 1."""
    half = t.shape[1] // 2
    gate, up = t[:, :half], t[:, half:]
    paired = torch.stack([gate, up], dim=2)
    return paired.reshape(t.shape[0], t.shape[1], *t.shape[2:]).contiguous()


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
    """Create MXFP4 MoE weight tensors with swizzled scales."""
    w13_list, w13_scale_list = [], []
    w2_list, w2_scale_list = [], []
    for _ in range(e):
        w1_bf16 = torch.randn(n, k, device=DEVICE, dtype=DTYPE) / 10
        w3_bf16 = torch.randn(n, k, device=DEVICE, dtype=DTYPE) / 10
        w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=0)
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
        return (
            _interleave_gate_up(w13_fp4),
            _interleave_gate_up(w13_scale_raw),
            w2_fp4,
            w2_scale_raw,
            w13_scale_raw,
            w2_scale_raw,
        )

    w13_scale = swizzle_blockscale(w13_scale_raw.view(torch.float8_e4m3fn)).view(
        torch.uint8
    )
    w2_scale = swizzle_blockscale(w2_scale_raw.view(torch.float8_e4m3fn)).view(
        torch.uint8
    )
    return w13_fp4, w13_scale, w2_fp4, w2_scale, w13_scale_raw, w2_scale_raw


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
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference MXFP4 MoE with dequantized weights and MXFP8-quantized acts."""
    device = hidden_states.device
    dtype = hidden_states.dtype
    m, hidden_dim = hidden_states.shape
    num_experts = w13_fp4.shape[0]
    top_k = topk_ids.shape[1]

    w13_deq = [
        _dequant_mxfp4(w13_fp4[eid], w13_scale_raw[eid]).to(dtype)
        for eid in range(num_experts)
    ]
    w2_deq = [
        _dequant_mxfp4(w2_fp4[eid], w2_scale_raw[eid]).to(dtype)
        for eid in range(num_experts)
    ]

    routed = topk_ids.to(torch.long)
    valid_routes = (routed >= 0) & (routed < num_experts)
    routed = torch.where(valid_routes, routed, torch.full_like(routed, -1))
    router_weights = topk_weights.to(torch.float32)

    out = torch.zeros(m, hidden_dim, device=device, dtype=dtype)
    for i in range(m):
        acc = torch.zeros(1, hidden_dim, device=device, dtype=dtype)
        for j in range(top_k):
            expert_id = int(routed[i, j].item())
            if expert_id < 0:
                continue
            weight = router_weights[i, j].to(dtype)
            x = hidden_states[i : i + 1]
            if apply_router_weight_on_input:
                x = x * weight
            x = _maybe_quantize_mxfp8_reference(x)
            gemm1_out = x @ w13_deq[expert_id].T
            if w13_bias is not None:
                gemm1_out = gemm1_out + w13_bias[expert_id : expert_id + 1].to(dtype)
            act_dim = (
                gemm1_out.shape[-1] // 2 if activation.is_gated else gemm1_out.shape[-1]
            )
            act_buf = torch.empty(1, act_dim, device=device, dtype=dtype)
            apply_moe_activation(activation, act_buf, gemm1_out)
            act_buf = _maybe_quantize_mxfp8_reference(act_buf)
            y = act_buf @ w2_deq[expert_id].T
            if w2_bias is not None:
                y = y + w2_bias[expert_id : expert_id + 1].to(dtype)
            acc = acc + y if apply_router_weight_on_input else acc + y * weight
        out[i] = acc[0]
    return out


@pytest.mark.parametrize("topk,apply_router_weight_on_input", BATCH_INVARIANT_CASES)
@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_matches_dequant_reference(
    topk: int,
    apply_router_weight_on_input: bool,
) -> None:
    """`fused_moe_batch_invariant_mxfp4` matches the dequantized reference."""
    set_random_seed(42)
    m, e, n, k = 32, 8, 128, 256

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
        workspace=_make_nvfp4_workspace(e),
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
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@torch.inference_mode()
def test_batch_invariant_mxfp4_moe_interleaved_checkpoint_format() -> None:
    """The batch-invariant MXFP4 convert path de-interleaves and swizzles w13."""
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
        workspace=_make_nvfp4_workspace(e),
    )
    ref = _reference_fused_moe_mxfp4_dequant(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_fp4=torch.cat(
            [w13_fp4_interleaved[:, ::2, :], w13_fp4_interleaved[:, 1::2, :]], dim=1
        ),
        w13_scale_raw=w13_scale_raw,
        w2_fp4=w2_fp4,
        w2_scale_raw=w2_scale_raw,
        activation=MoEActivation.SILU,
    )
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
