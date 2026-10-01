# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer collective norm kernels shared by manual and compiler fusions."""

from importlib.util import find_spec
from types import ModuleType

import torch

from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    get_fi_ar_quant_workspace,
    get_fi_ar_workspace,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# The empirical value for small batch
PDL_ADVANCE_LAUNCH_TOKENS = 16

logger = init_logger(__name__)

flashinfer_comm: ModuleType | None = None
if find_spec("flashinfer"):
    try:
        import flashinfer.comm as _flashinfer_comm

        if hasattr(_flashinfer_comm, "allreduce_fusion") and hasattr(
            _flashinfer_comm, "create_allreduce_fusion_workspace"
        ):
            flashinfer_comm = _flashinfer_comm
    except Exception as e:
        logger.debug_once("flashinfer.comm import failed: %s", e)

# Max size of the input tensor per world size per device capability
# to use flashinfer fused allreduce
FI_ALLREDUCE_FUSION_MAX_SIZE_MB: dict[int, dict[int, float]] = {
    90: {
        2: 64,  # 64MB
        4: 2,  # 2MB
        8: 0.5,  # 0.5MB
    },
    100: {
        2: 64,  # 64MB
        4: 32,  # 32MB
        8: 1,  # 1MB
        16: 64,  # 64MB (mnnvl multi-node)
    },
    103: {
        2: 64,  # 64MB
        4: 64,  # 64MB
        8: 4,  # 4MB
        16: 64,  # 64MB (mnnvl multi-node)
    },
    107: {
        2: 64,  # 64MB
        4: 64,  # 64MB
        8: 2,  # 2MB
    },
}

# Max size of the input tensor per world size per device capability
# to use flashinfer one shot fused allreduce
# OneShot max size is at most 64MB / world size (FlashInfer restriction)
_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB: dict[int, dict[int, float]] = {
    90: {
        2: 32,  # 32MB
        4: 2,  # 2MB
        8: 0.5,  # 0.5MB
    },
    100: {
        2: 32,  # 32MB
        4: 4,  # 4MB
        8: 1,  # 1MB
    },
    103: {
        2: 32,  # 32MB
        4: 4,  # 4MB
        8: 2,  # 2MB
    },
    107: {
        2: 32,  # 32MB
        4: 4,  # 4MB
        8: 2,  # 2MB
    },
}

MiB = 1024 * 1024


def _select_flashinfer_allreduce_use_oneshot(
    workspace_backend: str,
    device_capability: int | None,
    world_size: int,
    current_tensor_size: int,
) -> bool | None:
    if workspace_backend == "mnnvl":
        # FlashInfer sizes MNNVL workspaces around its own AUTO strategy.
        # Forcing vLLM's per-rank threshold can request one-shot for tensors
        # larger than the MNNVL one-shot workspace.
        return None

    if device_capability is None:
        max_one_shot_size = None
    else:
        max_one_shot_size = _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB.get(
            device_capability, {}
        ).get(world_size)
    return max_one_shot_size is None or current_tensor_size <= max_one_shot_size * MiB


flashinfer_trtllm_fused_allreduce_norm = None
if flashinfer_comm is not None:
    ar_fusion_patterns = flashinfer_comm.AllReduceFusionPattern

    def call_trtllm_fused_allreduce_norm(
        allreduce_in: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_size: int,
        launch_with_pdl: bool,
        fp32_acc: bool,
        max_token_num: int,
        pattern_code: int,
        norm_out: torch.Tensor | None = None,
        quant_out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
        weight_bias: float = 0.0,
    ) -> None:
        # handle transformers backend passing outer batch dim.
        if allreduce_in.dim() != 2:
            hidden = allreduce_in.shape[-1]
            allreduce_in = allreduce_in.view(-1, hidden)
            residual = residual.view(-1, hidden)
            if norm_out is not None:
                norm_out = norm_out.view(-1, hidden)
        num_tokens, hidden_size = allreduce_in.shape
        element_size = allreduce_in.element_size()
        current_tensor_size = num_tokens * hidden_size * element_size
        max_tensor_size = max_token_num * hidden_size * element_size
        assert current_tensor_size <= max_tensor_size, (
            f"Current tensor size {current_tensor_size} is larger than "
            f"max token num {max_token_num} * hidden size {hidden_size} * "
            f"element size {element_size}"
        )
        curr_device = current_platform.get_device_capability()
        device_capability = curr_device.to_int() if curr_device is not None else None

        # Select workspace based on pattern: quant patterns use the quant
        # workspace, non-quant patterns use the primary workspace.
        is_quant_pattern = pattern_code in (
            ar_fusion_patterns.kARResidualRMSNormFP8Quant,
            ar_fusion_patterns.kARResidualRMSNormFP4Quant,
        )
        get_workspace_fn = (
            get_fi_ar_quant_workspace if is_quant_pattern else get_fi_ar_workspace
        )
        workspace = get_workspace_fn(
            world_size=world_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=max_token_num,
            hidden_dim=hidden_size,
            dtype=allreduce_in.dtype,
            group=get_tp_group().cpu_group,
        )
        assert workspace is not None, (
            "Flashinfer allreduce workspace must be initialized when using flashinfer"
        )
        use_oneshot = _select_flashinfer_allreduce_use_oneshot(
            workspace.backend,
            device_capability,
            world_size,
            current_tensor_size,
        )
        assert flashinfer_comm is not None
        if norm_out is None:
            norm_out = allreduce_in
            residual_out = residual
        else:
            # return residual_out as allreduce_out with zeroed residual_in
            # as flashinfer does not support rms_norm
            # and allreduce_out together
            residual_out = allreduce_in

        layout_code = None
        # vLLM quant patterns use swizzled scale-factor layout. Non-quant
        # patterns ignore layout_code.
        if workspace.backend in ("trtllm", "mnnvl"):
            layout_code = flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4

        flashinfer_comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=pattern_code,
            launch_with_pdl=launch_with_pdl,
            output=None,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            residual_in=residual,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            scale_factor=scale_factor,
            layout_code=layout_code,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
            weight_bias=weight_bias,
            # The one-shot Lamport all-reduce signals PDL completion before its
            # output buffer is committed when trigger_completion_at_end is
            # False, so the next PDL-launched kernel can read the uninitialized
            # Lamport buffer and produce NaN. This only fires for
            # num_tokens <= PDL_ADVANCE_LAUNCH_TOKENS (the batch=1 / spec-decode
            # shapes, where the one-shot path is always selected). Complete at
            # the end for the one-shot path; the two-shot path is synchronized
            # and keeps the early completion. Related one-shot instability in
            # the same kernel: flashinfer-ai/flashinfer#1223.
            trigger_completion_at_end=(use_oneshot is True)
            or num_tokens > PDL_ADVANCE_LAUNCH_TOKENS,
        )

    direct_register_custom_op(
        op_name="flashinfer_trtllm_fused_allreduce_norm",
        op_func=call_trtllm_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
    )
    flashinfer_trtllm_fused_allreduce_norm = (
        torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default
    )


def try_fused_allreduce_norm_nvfp4_quant(
    input: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    epsilon: float,
    result: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> bool:
    """Run the collective kernel when its workspace can hold this runtime shape."""
    if flashinfer_comm is None:
        return False
    world_size = get_tensor_model_parallel_world_size()
    capability = current_platform.get_device_capability()
    max_size_mb = FI_ALLREDUCE_FUSION_MAX_SIZE_MB.get(
        capability.to_int() if capability is not None else 0, {}
    ).get(world_size)
    if max_size_mb is None or input.nbytes > max_size_mb * MiB:
        return False
    num_tokens, hidden = input.shape
    max_tokens = int(max_size_mb * MiB) // (hidden * input.element_size())
    workspace = get_fi_ar_quant_workspace(
        world_size=world_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_tokens,
        hidden_dim=hidden,
        dtype=input.dtype,
        group=get_tp_group().cpu_group,
    )
    if workspace is None or not workspace.is_buffer_size_sufficient(
        tp_size=world_size, num_tokens=num_tokens, hidden_dim=hidden, dtype=input.dtype
    ):
        return False
    # Without an incoming residual, preserve the reduced input as the residual.
    norm_out = torch.empty_like(input) if residual is None else None
    call_trtllm_fused_allreduce_norm(
        allreduce_in=input,
        residual=torch.zeros_like(input) if residual is None else residual,
        rms_gamma=weight,
        rms_eps=epsilon,
        world_size=world_size,
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=max_tokens,
        pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
        norm_out=norm_out,
        quant_out=result,
        scale_out=block_scale.view(torch.float8_e4m3fn),
        scale_factor=global_scale,
    )
    return True
