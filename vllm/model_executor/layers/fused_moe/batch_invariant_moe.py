# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant NVFP4 fused MoE expert implementation."""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers used by ``fused_moe_batch_invariant_nvfp4``
# ---------------------------------------------------------------------------


def _nvfp4_get_expert_scalar(
    tensor: torch.Tensor | None,
    expert_id: int,
    *,
    field_name: str,
) -> torch.Tensor:
    if tensor is None:
        raise RuntimeError(f"Missing required NVFP4 MoE tensor: {field_name}")

    if tensor.ndim == 0 or tensor.numel() == 1:
        value = tensor.reshape(())
    else:
        if expert_id >= tensor.shape[0]:
            raise RuntimeError(
                f"NVFP4 MoE tensor '{field_name}' is missing expert {expert_id}."
            )
        value = tensor[expert_id]
        if value.ndim != 0:
            value = value.reshape(())
    return value.to(dtype=torch.float32)


def _nvfp4_moe_map_experts(
    topk_ids: torch.Tensor, expert_map: torch.Tensor
) -> torch.Tensor:
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    if expert_map.numel() == 0:
        return torch.full_like(topk_ids, -1, dtype=torch.long)
    valid = (flat_ids >= 0) & (flat_ids < expert_map.numel())
    clamped = flat_ids.clamp(min=0, max=max(0, expert_map.numel() - 1))
    remapped = expert_map.to(torch.long).index_select(0, clamped)
    mapped = torch.where(valid, remapped, torch.full_like(remapped, -1))
    return mapped.view_as(topk_ids)


# ---------------------------------------------------------------------------
# Standalone deterministic NVFP4 MoE implementation
# ---------------------------------------------------------------------------


def fused_moe_batch_invariant_nvfp4(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    a1_gscale: torch.Tensor | None,
    g1_alphas: torch.Tensor | None,
    a2_gscale: torch.Tensor | None,
    g2_alphas: torch.Tensor | None,
    activation: Any,
    *,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """
    Deterministic NVFP4 MoE fallback built from batch-invariant NVFP4 GEMMs.
    """
    from vllm.model_executor.layers.batch_invariant import (
        linear_batch_invariant_nvfp4,
    )
    from vllm.model_executor.layers.fused_moe.activation import (
        apply_moe_activation,
    )

    if hidden_states.ndim != 2:
        raise RuntimeError(
            f"Expected 2D hidden_states for NVFP4 MoE fallback, got {hidden_states.shape}."
        )
    if topk_ids.shape != topk_weights.shape:
        raise RuntimeError(
            "NVFP4 MoE fallback expects topk_ids and topk_weights to have identical shapes."
        )
    if topk_ids.ndim != 2:
        raise RuntimeError(
            f"Expected 2D top-k routing tensors, got shape {topk_ids.shape}."
        )
    if apply_router_weight_on_input and topk_ids.shape[1] != 1:
        raise RuntimeError(
            "apply_router_weight_on_input=True is only supported for top_k == 1."
        )

    activation_kind = (
        activation
        if isinstance(activation, MoEActivation)
        else MoEActivation.from_str(str(activation))
    )

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w13_weight.shape[0]
    top_k = topk_ids.shape[1]

    routed_topk_ids = topk_ids
    if expert_map is not None:
        routed_topk_ids = _nvfp4_moe_map_experts(topk_ids, expert_map)
    routed_topk_ids = routed_topk_ids.to(torch.long)
    routed_topk_weights = topk_weights.to(torch.float32)

    expert_ids = routed_topk_ids.reshape(-1)
    assignment_weights = routed_topk_weights.reshape(-1)
    repeated_hidden_states = (
        hidden_states.unsqueeze(1)
        .expand(num_tokens, top_k, hidden_dim)
        .reshape(-1, hidden_dim)
        .contiguous()
    )
    if apply_router_weight_on_input:
        repeated_hidden_states = repeated_hidden_states * assignment_weights.unsqueeze(
            -1
        ).to(repeated_hidden_states.dtype)
    output_scale = (
        torch.ones_like(assignment_weights)
        if apply_router_weight_on_input
        else assignment_weights
    )
    contribution = torch.zeros(
        (repeated_hidden_states.shape[0], hidden_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    w1_padding_cols = max(0, w13_weight.shape[-1] - hidden_dim // 2)
    w2_output_size = hidden_dim
    w1_output_size = w13_weight.shape[1]

    for expert_id in range(num_experts):
        # CUDA graph capture-safe expert routing mask. This avoids dynamic
        # indexing ops such as nonzero/index_select that are capture-unsafe.
        expert_mask = (expert_ids == expert_id).unsqueeze(-1)
        expert_inputs = repeated_hidden_states * expert_mask.to(
            repeated_hidden_states.dtype
        )

        gemm1_out = linear_batch_invariant_nvfp4(
            input=expert_inputs,
            weight=w13_weight[expert_id].contiguous(),
            weight_scale=w13_weight_scale[expert_id].contiguous(),
            input_global_scale_inv=_nvfp4_get_expert_scalar(
                a1_gscale, expert_id, field_name="a1_gscale"
            ),
            alpha=_nvfp4_get_expert_scalar(
                g1_alphas, expert_id, field_name="g1_alphas"
            ),
            output_size=w1_output_size,
            weights_padding_cols=w1_padding_cols,
            quant_backend=quant_backend,
        )

        activation_out_dim = (
            gemm1_out.shape[-1] // 2
            if activation_kind.is_gated
            else gemm1_out.shape[-1]
        )
        activation_out = torch.empty(
            (gemm1_out.shape[0], activation_out_dim),
            device=gemm1_out.device,
            dtype=gemm1_out.dtype,
        )
        apply_moe_activation(
            activation=activation_kind,
            output=activation_out,
            input=gemm1_out,
        )

        w2_padding_cols = max(0, w2_weight.shape[-1] - activation_out_dim // 2)
        gemm2_out = linear_batch_invariant_nvfp4(
            input=activation_out,
            weight=w2_weight[expert_id].contiguous(),
            weight_scale=w2_weight_scale[expert_id].contiguous(),
            input_global_scale_inv=_nvfp4_get_expert_scalar(
                a2_gscale, expert_id, field_name="a2_gscale"
            ),
            alpha=_nvfp4_get_expert_scalar(
                g2_alphas, expert_id, field_name="g2_alphas"
            ),
            output_size=w2_output_size,
            weights_padding_cols=w2_padding_cols,
            quant_backend=quant_backend,
        )

        weighted_output = (
            gemm2_out.to(torch.float32)
            * output_scale.unsqueeze(-1)
            * expert_mask.to(torch.float32)
        )
        contribution += weighted_output

    reduced = torch.zeros(
        (num_tokens, hidden_dim),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    contribution = contribution.view(num_tokens, top_k, hidden_dim)
    for slot in range(top_k):
        reduced += contribution[:, slot, :]
    return reduced.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Modular-kernel wrapper class
# ---------------------------------------------------------------------------


class BatchInvariantNvfp4Experts(mk.FusedMoEPermuteExpertsUnpermute):
    """Deterministic, batch-invariant NVFP4 MoE expert implementation.

    Wraps ``fused_moe_batch_invariant_nvfp4`` (per-expert Triton
    ``tl.dot_scaled`` GEMMs) into the modular-kernel interface so it
    can be used as a drop-in backend alongside FlashInfer/CUTLASS.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        assert quant_config.weight_quant_dtype == "nvfp4", (
            "BatchInvariantNvfp4Experts only supports nvfp4 weight quantization, "
            f"got {quant_config.weight_quant_dtype}"
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.has_device_capability(100)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return True

    def supports_chunking(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace13 = (M, K)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace13, workspace2, output_shape)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        result = fused_moe_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weight=w1,
            w13_weight_scale=self.w1_scale,
            w2_weight=w2,
            w2_weight_scale=self.w2_scale,
            a1_gscale=self.a1_gscale,
            g1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            g2_alphas=self.g2_alphas,
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            expert_map=expert_map,
        )
        output.copy_(result)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError(
            "LoRA is not supported for batch-invariant NVFP4 MoE"
        )
