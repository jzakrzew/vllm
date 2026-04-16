# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        "Nvfp4 Requires compute capability of 10 or above.", allow_module_level=True
    )


@dataclass
class _BatchInvariantCaseConfig:
    name: str
    m: int
    n: int
    k: int
    e: int
    topk: int
    routing_pattern: str
    probe_indices: tuple[int, ...]
    subset_indices: tuple[tuple[int, ...], ...]


@dataclass
class _CutlassFp4MoECase:
    config: _BatchInvariantCaseConfig
    hidden_states: torch.Tensor
    score: torch.Tensor
    kernel: mk.FusedMoEKernel
    w1_q: torch.Tensor
    w2_q: torch.Tensor
    w1_blockscale: torch.Tensor
    w2_blockscale: torch.Tensor
    w1_gs: torch.Tensor
    w2_gs: torch.Tensor
    activation: MoEActivation
    dtype: torch.dtype


_ROUTING_OFFSETS = (3, 13, 29, 47, 71, 89, 113, 137)
_ROUTING_WEIGHTS = (12.0, 8.0, 4.0, 1.0, 0.5, 0.25, 0.125, 0.0625)

# These cases are chosen to stress the grouped-GEMM implementation in a few
# different ways:
# - a small irregular batch with many tiny per-expert problems
# - a concentrated batch with M just above the 128-tile boundary
# - a non-square GEMM shape with partial 128x128 N/K tiles while keeping
#   CUTLASS FP4 alignment constraints satisfied
_REFERENCE_CASE = _BatchInvariantCaseConfig(
    name="reference_small",
    m=12,
    n=1024,
    k=1024,
    e=64,
    topk=4,
    routing_pattern="staggered",
    probe_indices=tuple(range(12)),
    subset_indices=((0, 3, 7), (1, 4, 8, 10), (11, 5, 2, 9, 6)),
)
_BATCH_INVARIANT_CASES = (
    _REFERENCE_CASE,
    _BatchInvariantCaseConfig(
        name="single_expert_two_m_tiles",
        m=129,
        n=1024,
        k=1024,
        e=40,
        topk=1,
        routing_pattern="concentrated",
        probe_indices=(0, 1, 31, 63, 64, 95, 127, 128),
        subset_indices=(
            (0, 64, 128),
            (1, 63, 64, 127),
            (32, 33, 95, 96, 128),
        ),
    ),
    _BatchInvariantCaseConfig(
        name="non_square_staggered",
        m=73,
        n=1472,
        k=1504,
        e=40,
        topk=4,
        routing_pattern="staggered",
        probe_indices=(0, 7, 18, 31, 47, 72),
        subset_indices=((0, 17, 35), (8, 36, 72), (1, 9, 33, 48, 64)),
    ),
)


def _make_router_logits(
    config: _BatchInvariantCaseConfig,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert config.topk <= len(_ROUTING_OFFSETS)
    score = torch.full((config.m, config.e), -32.0, device="cuda", dtype=dtype)
    preferred_weights = torch.tensor(
        _ROUTING_WEIGHTS[: config.topk], device="cuda", dtype=dtype
    )
    for idx in range(config.m):
        if config.routing_pattern == "staggered":
            preferred_experts = torch.tensor(
                [
                    (idx * 7 + _ROUTING_OFFSETS[slot]) % config.e
                    for slot in range(config.topk)
                ],
                device="cuda",
            )
        elif config.routing_pattern == "concentrated":
            preferred_experts = torch.tensor(
                [_ROUTING_OFFSETS[slot] % config.e for slot in range(config.topk)],
                device="cuda",
            )
        else:
            raise ValueError(f"Unsupported routing pattern: {config.routing_pattern}")
        score[idx, preferred_experts] = preferred_weights
    score += 0.01 * torch.randn_like(score)
    return score


def _make_cutlass_fp4_moe_batch_invariant_case(
    case_config: _BatchInvariantCaseConfig,
    activation: MoEActivation,
) -> _CutlassFp4MoECase:
    set_random_seed(7)
    dtype = torch.bfloat16

    hidden_states = (
        torch.randn((case_config.m, case_config.k), device="cuda", dtype=dtype) / 10
    )

    (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
        make_test_weights(
            case_config.e,
            case_config.n,
            case_config.k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_out_ch_quant=False,
        )
    )

    score = _make_router_logits(case_config, dtype)

    a1_gs = torch.ones((case_config.e,), device="cuda", dtype=torch.float32)
    a2_gs = torch.ones((case_config.e,), device="cuda", dtype=torch.float32)

    assert w1_gs is not None
    assert w2_gs is not None
    assert w1_blockscale is not None
    assert w2_blockscale is not None

    quant_config = nvfp4_moe_quant_config(
        g1_alphas=(1 / w1_gs),
        g2_alphas=(1 / w2_gs),
        a1_gscale=a1_gs,
        a2_gscale=a2_gs,
        w1_scale=w1_blockscale,
        w2_scale=w2_blockscale,
    )

    moe_config = make_dummy_moe_config(
        num_experts=case_config.e,
        experts_per_token=case_config.topk,
        hidden_dim=case_config.k,
        intermediate_size_per_partition=case_config.n,
        in_dtype=dtype,
    )
    kernel = mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        CutlassExpertsFp4(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
        inplace=False,
    )

    return _CutlassFp4MoECase(
        config=case_config,
        hidden_states=hidden_states,
        score=score,
        kernel=kernel,
        w1_q=w1_q,
        w2_q=w2_q,
        w1_blockscale=w1_blockscale,
        w2_blockscale=w2_blockscale,
        w1_gs=w1_gs,
        w2_gs=w2_gs,
        activation=activation,
        dtype=dtype,
    )


def _run_cutlass_fp4_moe(
    case: _CutlassFp4MoECase,
    hidden_states: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, case.config.topk, renormalize=False
    )
    return case.kernel.apply(
        hidden_states=hidden_states,
        w1=case.w1_q,
        w2=case.w2_q,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=case.config.e,
        activation=case.activation,
        apply_router_weight_on_input=False,
        expert_map=None,
    )


def _reference_moe(case: _CutlassFp4MoECase) -> torch.Tensor:
    quant_blocksize = 16
    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
        / torch.amax(case.hidden_states.flatten(), dim=-1)
    ).to(torch.float32)
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(
        case.hidden_states, a_global_scale
    )
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a_global_scale,
        dtype=case.hidden_states.dtype,
        device=case.hidden_states.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty(
        (case.config.e, 2 * case.config.n, case.config.k),
        device="cuda",
        dtype=case.dtype,
    )
    w2_d = torch.empty(
        (case.config.e, case.config.k, case.config.n),
        device="cuda",
        dtype=case.dtype,
    )
    for idx in range(case.config.e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            case.w1_q[idx],
            case.w1_blockscale[idx],
            case.w1_gs[idx],
            dtype=case.dtype,
            device=case.w1_q.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            case.w2_q[idx],
            case.w2_blockscale[idx],
            case.w2_gs[idx],
            dtype=case.dtype,
            device=case.w2_q.device,
            block_size=quant_blocksize,
        )

    return torch_moe(
        a_in_dtype,
        w1_d,
        w2_d,
        case.score,
        case.config.topk,
        activation=case.activation,
    )


def _batch_permutations(
    case: _CutlassFp4MoECase,
) -> tuple[tuple[str, torch.Tensor], ...]:
    indices = torch.arange(case.hidden_states.size(0), device=case.hidden_states.device)
    return (
        ("reversed", torch.flip(indices, dims=(0,))),
        ("evens_then_odds", torch.cat((indices[::2], indices[1::2]))),
    )


@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SWIGLUSTEP])
@torch.inference_mode()
def test_cutlass_fp4_moe_matches_reference(
    activation: MoEActivation,
    default_vllm_config,
    workspace_init,
):
    case = _make_cutlass_fp4_moe_batch_invariant_case(_REFERENCE_CASE, activation)
    # Run the kernel on the same mixed batch used by the invariance test, then
    # compare against a dequantized eager reference path.
    batch_output = _run_cutlass_fp4_moe(case, case.hidden_states, case.score)
    torch_ref = _reference_moe(case)

    torch.testing.assert_close(batch_output, torch_ref, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "case_config",
    _BATCH_INVARIANT_CASES,
    ids=[case.name for case in _BATCH_INVARIANT_CASES],
)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SWIGLUSTEP])
@torch.inference_mode()
def test_cutlass_fp4_moe_batch_invariant(
    case_config: _BatchInvariantCaseConfig,
    activation: MoEActivation,
    default_vllm_config,
    workspace_init,
):
    case = _make_cutlass_fp4_moe_batch_invariant_case(case_config, activation)
    # Establish the baseline output for the full mixed batch once, then
    # compare every smaller replay against the corresponding slice here.
    batch_output = _run_cutlass_fp4_moe(case, case.hidden_states, case.score)

    assert CutlassExpertsFp4._supports_batch_invariance()

    # Re-run the whole batch with a different token order to make sure the
    # grouped GEMM result is invariant to how work is packed for scheduling.
    for perm_name, perm in _batch_permutations(case):
        permuted_output = _run_cutlass_fp4_moe(
            case,
            case.hidden_states[perm],
            case.score[perm],
        )
        torch.testing.assert_close(
            batch_output[perm],
            permuted_output,
            atol=0,
            rtol=0,
            msg=(
                f"{case.config.name}: permutation '{perm_name}' changed outputs "
                "relative to the baseline batch order."
            ),
        )

    # Probe singleton replays at specific indices. Large cases use hand-picked
    # positions around tile boundaries instead of replaying every token.
    for idx in case.config.probe_indices:
        single_idx = torch.tensor([idx], device=case.hidden_states.device)
        single_output = _run_cutlass_fp4_moe(
            case,
            case.hidden_states[single_idx],
            case.score[single_idx],
        )
        torch.testing.assert_close(
            batch_output[single_idx],
            single_output,
            atol=0,
            rtol=0,
            msg=(
                f"{case.config.name}: token {idx} changed between full-batch "
                "and singleton execution."
            ),
        )

    # Re-run a few nontrivial sub-batches to catch interactions that only
    # appear when multiple tokens are grouped together.
    for subset_ids in case.config.subset_indices:
        subset = torch.tensor(subset_ids, device=case.hidden_states.device)
        subset_output = _run_cutlass_fp4_moe(
            case,
            case.hidden_states[subset],
            case.score[subset],
        )
        torch.testing.assert_close(
            batch_output[subset],
            subset_output,
            atol=0,
            rtol=0,
            msg=(
                f"{case.config.name}: sub-batch {list(subset_ids)} changed "
                "relative to full-batch execution."
            ),
        )
