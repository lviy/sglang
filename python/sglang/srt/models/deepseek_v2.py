# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""
from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.configs.model_config import (
    get_nsa_index_head_dim,
    get_nsa_index_n_heads,
    get_nsa_index_topk,
    is_deepseek_nsa,
)
from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.debug_utils.nan_diagnosis import (
    maybe_cuda_synchronize,
    maybe_log_event,
    maybe_log_tensor_stats,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_nsa_cp import NSACPLayerCommunicator
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
    DeepseekMHAForwardMixin,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _device_sm,
    _get_llama_4_scaling,
    _is_cpu,
    _is_cpu_amx_available,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _is_npu,
    _use_aiter,
    _use_aiter_gfx95,
    yarn_get_mscale,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    get_bool_env_var,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    make_layers,
    use_intel_amx_backend,
)

if _use_aiter_gfx95:

    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import (
        aiter_dsv3_router_gemm,
        fused_qk_rope_cat_and_cache_mla,
        get_dsv3_gemm_output_zero_allocator_size,
    )

if _is_cuda:
    from sgl_kernel import bmm_fp8, dsv3_fused_a_gemm, dsv3_router_gemm
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )
elif _is_npu:
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
        forward_mla_core_npu,
        forward_mla_prepare_npu,
    )
else:
    pass

logger = logging.getLogger(__name__)


def _parse_int_set_env(name: str) -> Optional[set[int]]:
    value = os.getenv(name)
    if not value:
        return None
    value = value.strip().lower()
    if value in ("*", "all"):
        return None
    layers = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layers.add(int(item))
        except ValueError:
            continue
    return layers if layers else None


def _parse_nan_diag_layer_watch() -> Optional[set[int]]:
    return _parse_int_set_env("SGLANG_NAN_DIAG_DEEPSEEK_LAYER_WATCH")


_NAN_DIAG_DEEPSEEK_LAYER_INTERVAL = max(
    0, int(os.getenv("SGLANG_NAN_DIAG_DEEPSEEK_LAYER_INTERVAL", "0"))
)
_NAN_DIAG_DEEPSEEK_LAYER_WATCH = _parse_nan_diag_layer_watch()
_NAN_DIAG_DEEPSEEK_BLOCK_ENABLE = get_bool_env_var(
    "SGLANG_NAN_DIAG_DEEPSEEK_BLOCK_ENABLE", "false"
)
_NAN_DIAG_DEEPSEEK_BLOCK_LAYER_WATCH = _parse_int_set_env(
    "SGLANG_NAN_DIAG_DEEPSEEK_BLOCK_LAYER_WATCH"
)
_NAN_DIAG_MLA_GUARD_PADDING = get_bool_env_var(
    "SGLANG_NAN_DIAG_MLA_GUARD_PADDING", "false"
)
_NAN_DIAG_MLP_GUARD_PADDING = get_bool_env_var(
    "SGLANG_NAN_DIAG_MLP_GUARD_PADDING", "false"
)
_NAN_DIAG_SYNC_AFTER_SELF_ATTN_OUTPUT = get_bool_env_var(
    "SGLANG_NAN_DIAG_SYNC_AFTER_SELF_ATTN_OUTPUT", "false"
)
_NAN_DIAG_BOUNDARY_ROW_CHECK_ENABLE = get_bool_env_var(
    "SGLANG_NAN_DIAG_BOUNDARY_ROW_CHECK_ENABLE", "false"
)
_NAN_DIAG_BOUNDARY_ROW_CHECK_ASSERT = get_bool_env_var(
    "SGLANG_NAN_DIAG_BOUNDARY_ROW_CHECK_ASSERT", "false"
)
_NAN_DIAG_BOUNDARY_ROW_CHECK_MAX_ROWS = max(
    1, int(os.getenv("SGLANG_NAN_DIAG_BOUNDARY_ROW_CHECK_MAX_ROWS", "8"))
)
_NAN_DIAG_DEEPSEEK_META_LOG_ONCE = get_bool_env_var(
    "SGLANG_NAN_DIAG_DEEPSEEK_META_LOG_ONCE", "true"
)
_NAN_DIAG_DEEPSEEK_META_MAX_EVENTS = max(
    1, int(os.getenv("SGLANG_NAN_DIAG_DEEPSEEK_META_MAX_EVENTS", "64"))
)
_NAN_DIAG_DEEPSEEK_META_EVENT_COUNT = 0
_NAN_DIAG_DEEPSEEK_META_SEEN_KEYS: set[tuple[str, int]] = set()


def _should_log_deepseek_layer(layer_id: int, start_layer: int, end_layer: int) -> bool:
    if _NAN_DIAG_DEEPSEEK_LAYER_WATCH is not None:
        return layer_id in _NAN_DIAG_DEEPSEEK_LAYER_WATCH
    if _NAN_DIAG_DEEPSEEK_LAYER_INTERVAL > 0:
        return (
            (layer_id - start_layer) % _NAN_DIAG_DEEPSEEK_LAYER_INTERVAL == 0
            or layer_id == end_layer - 1
        )
    return False


def _should_log_deepseek_block_layer(layer_id: int) -> bool:
    if not _NAN_DIAG_DEEPSEEK_BLOCK_ENABLE:
        return False
    if _NAN_DIAG_DEEPSEEK_BLOCK_LAYER_WATCH is None:
        return True
    return layer_id in _NAN_DIAG_DEEPSEEK_BLOCK_LAYER_WATCH


def _should_emit_deepseek_meta(stage: str, layer_id: int) -> bool:
    global _NAN_DIAG_DEEPSEEK_META_EVENT_COUNT
    if not _NAN_DIAG_DEEPSEEK_META_LOG_ONCE:
        return True
    key = (stage, layer_id)
    if key in _NAN_DIAG_DEEPSEEK_META_SEEN_KEYS:
        return False
    if _NAN_DIAG_DEEPSEEK_META_EVENT_COUNT >= _NAN_DIAG_DEEPSEEK_META_MAX_EVENTS:
        return False
    _NAN_DIAG_DEEPSEEK_META_SEEN_KEYS.add(key)
    _NAN_DIAG_DEEPSEEK_META_EVENT_COUNT += 1
    return True


FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if not hasattr(self.gate_up_proj, "weight"):
            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
        if not hasattr(self.down_proj, "weight"):
            self.down_proj.weight = getattr(self.down_proj, "weight_packed")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and self.gate_up_proj.weight.dtype == torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        quant_config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            correction_bias_dtype = (
                torch.bfloat16
                if quant_config is not None
                and quant_config.get_name() == "modelopt_fp4"
                and get_moe_runner_backend().is_flashinfer_trtllm()
                else torch.float32
            )
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=correction_bias_dtype)
            )
        else:
            self.e_score_correction_bias = None
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def forward(
        self,
        hidden_states,
        gemm_output_zero_allocator: BumpAllocator = None,
        forward_batch: ForwardBatch = None,
    ):
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states,
                self.weight,
                None,  # bias
                True,  # is_vnni
            )

        if get_global_server_args().enable_deterministic_inference:
            return F.linear(hidden_states, self.weight, None)

        if forward_batch is not None and nsa_use_prefill_cp(forward_batch):
            logits = F.linear(hidden_states, self.weight, None)
        else:
            # NOTE: For some unknown reason, router_gemm seems degrade accept length.
            if (
                _is_cuda
                and hidden_states.shape[0] <= 16
                and hidden_states.shape[1] == 7168
                and (self.weight.shape[0] == 256 or self.weight.shape[0] == 384)
                and _device_sm >= 90
            ):

                # router gemm output float32
                logits = dsv3_router_gemm(
                    hidden_states, self.weight, out_dtype=torch.float32
                )
            elif _use_aiter_gfx95 and hidden_states.shape[0] <= 256:
                logits = aiter_dsv3_router_gemm(
                    hidden_states, self.weight, gemm_output_zero_allocator
                )
            else:
                logits = F.linear(hidden_states, self.weight, None)

        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )
        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
        )

        # scaling factor for fused shared experts on AMD-platform.
        fused_shared_experts_scaling_factor = None
        if self.moe_ep_size > 1 and self.num_fused_shared_experts > 0:
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts
            + self.num_fused_shared_experts
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            prefix=add_prefix("experts", prefix),
        )

        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
            # and requires the output format to be standard (except trtllm). We use quant_config to determine the output format.
            output_format=(
                TopKOutputFormat.STANDARD
                if (quant_config is None)
                and (not get_moe_runner_backend().is_flashinfer_trtllm())
                else None
            ),
        )

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe, or with fp4 allgather
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    or get_moe_a2a_backend().is_mooncake()
                    or get_moe_a2a_backend().is_ascend_fuseep()
                    or get_moe_a2a_backend().is_flashinfer()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else {}
                ),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            ) and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() in {
                "awq",
                "awq_marlin",
                "moe_wna16",
            }
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            if self.shared_experts_is_fp8:
                if (
                    _use_aiter
                    and config.quantization_config.get("quant_method")
                    == "compressed-tensors"
                ):
                    # For compressed-tensors ptpc model, don't need to check the weight_block_size
                    pass
                else:
                    assert (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                        == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                    )
                    self.shared_experts_weight_block_size = (
                        self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                    )

        self.top_k = config.num_experts_per_tok

        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_ascend_fuseep()
            or get_moe_a2a_backend().is_flashinfer()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        if not self._enable_a2a_moe:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
            ):
                return self.forward_normal_dual_stream(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                )
            else:
                return self.forward_normal(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                )
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:

        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(
            hidden_states, gemm_output_zero_allocator
        )

        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_output = self.topk(hidden_states, router_logits)
            final_hidden_states = self.experts(hidden_states, topk_output)
            if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
                final_hidden_states *= self.routed_scaling_factor

        current_stream.wait_stream(self.alt_stream)
        final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
    ) -> torch.Tensor:
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states, should_allreduce_fusion)

        if hidden_states.shape[0] > 0:
            if (
                not self._fuse_shared_experts_inside_sbo
            ):  # TODO: check if it supports mtp
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator
                )
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if self._fuse_shared_experts_inside_sbo:
            shared_output = None

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output
                self.alt_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.alt_stream):
                    shared_output = self._forward_shared_experts(
                        hidden_states, gemm_output_zero_allocator
                    )

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                nonlocal shared_output
                torch.cuda.current_stream().wait_stream(self.alt_stream)
                post_combine_hook_handle.remove()

            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states,
            topk_output,
        )
        if (
            not _is_cuda
            and not _use_aiter
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)
        ):
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor
        if shared_output is not None:
            final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        fused_experts_out = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        assert use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ) == use_intel_amx_backend(self.shared_experts.down_proj)
        # [Note] inplace should be False in fused_experts.
        # If inplace is True in fused_experts (self.experts), hidden_states will be changed after fused_experts
        # While hidden_states is still needed in shared_expert.
        final_hidden_states = torch.ops.sgl_kernel.shared_expert_cpu(
            hidden_states,
            self.shared_experts.gate_up_proj.weight,
            self.shared_experts.down_proj.weight,
            fused_experts_out,
            self.routed_scaling_factor,
            True,  # inplace
            self.shared_experts_is_int8,  # use_int8_w8a8
            self.shared_experts_is_fp8,  # use_fp8_w8a16
            (
                self.shared_experts.gate_up_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.gate_up_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w1_scale
            (
                self.shared_experts.down_proj.weight_scale
                if self.shared_experts_is_int8
                else (
                    self.shared_experts.down_proj.weight_scale_inv
                    if self.shared_experts_is_fp8
                    else None
                )
            ),  # w2_scale
            (
                self.shared_experts_weight_block_size
                if self.shared_experts_is_fp8
                else None
            ),  # block_size
            None,  # a1_scale
            None,  # a2_scale
            True,  # is_vnni
        )
        if self.tp_size > 1 and not should_allreduce_fusion:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        shared_output = None
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            if not sbo_enabled_flag:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(hidden_states)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                shared_output = self._forward_shared_experts(hidden_states)
                for handle in deepep_dispatch_hook_handle:
                    handle.remove()

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):
                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                post_dispatch_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            assert isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            deepep_dispatch_hook_handle = (
                self.experts.dispatcher.register_deepep_dispatch_hook(
                    _deepep_dispatch_hook
                )
            )
            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        elif sbo_overlap_combine_flag:
            shared_output = None

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output

                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()

                # TODO reduce sm for non-deepgemm
                with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                    dispatcher.meta_overlap_args["compute_num_sms"]
                ):
                    shared_output = self._forward_shared_experts(hidden_states)

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )
        elif envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get():
            # On GB200: Shared experts overlapped on alt_stream, down gemm overlapped with DeepEP Combine

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):
                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()
                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)
        if shared_output is not None:
            x = shared_output
            if self.experts.should_fuse_routed_scaling_factor_in_topk:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not self.experts.should_fuse_routed_scaling_factor_in_topk:
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

    def _forward_shared_experts(
        self, hidden_states, gemm_output_zero_allocator: BumpAllocator = None
    ):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(
                hidden_states, gemm_output_zero_allocator=gemm_output_zero_allocator
            )
        else:
            return None

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_shared_experts(self, state):
        hidden_states_mlp_input = state.pop("hidden_states_mlp_input")
        if (self.num_fused_shared_experts == 0) and is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, hidden_states_mlp_input
        ):
            state.shared_output = self.shared_experts(hidden_states_mlp_input)
        else:
            state.shared_output = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input

        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        final_hidden_states = state.pop("hidden_states_after_combine")

        if (shared_output := state.pop("shared_output")) is not None:
            x = shared_output
            x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            final_hidden_states *= self.routed_scaling_factor

        state.hidden_states_mlp_output = final_hidden_states


class DeepseekV2AttentionMLA(nn.Module, DeepseekMHAForwardMixin):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.quant_config = quant_config
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.use_nsa = is_deepseek_nsa(config)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            assert self.use_nsa, "CP currently only supports deepseek v3.2 model"
        # cp reuse the attn_tp comm group but need to duplicate the weights
        if self.nsa_enable_prefill_cp and self.use_nsa:
            attn_tp_rank = 0
            attn_tp_size = 1
            self.cp_size = get_attention_tp_size()
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype

        # NOTE modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=self._get_q_b_proj_quant_config(quant_config),
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        if self.use_nsa:
            self.indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_nsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if not skip_rope:
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=False,
                device=get_global_server_args().device,
            )

            if rope_scaling:
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                scaling_factor = rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
            else:
                self.rotary_emb.forward = self.rotary_emb.forward_native
        else:
            self.rotary_emb = None
        self.use_deepseek_yarn_rope = rope_scaling is not None

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

        self.current_attention_backend = (
            None  # Attention backend used by current forward batch
        )
        self.rocm_fused_decode_mla = get_bool_env_var(
            "SGLANG_ROCM_FUSED_DECODE_MLA", "false"
        )

        # If we have self.fused_qkv_a_proj_with_mqa and we're running on CPU, we will choose the torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight kernel
        # which requires self.w_kc and self.w_vc to be packed.
        # If not, we will use torch.bmm and weight shouldn't be packed in this case
        has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
        if has_fused_proj and _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(
                weight_names=["w_kc", "w_vc"], transpose_dims=[[1, 2], [1, 2]]
            )

        is_packed_weight = (
            has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )
        self.use_min_latency_fused_a_gemm = (
            has_fused_proj
            and not is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.bfloat16
            and self.fused_qkv_a_proj_with_mqa.weight.shape[0] == 2112
            and self.fused_qkv_a_proj_with_mqa.weight.shape[1] == 7168
            and _is_cuda
            and 90 <= _device_sm < 120
        )

        self.qkv_proj_with_rope_is_int8 = (
            has_fused_proj
            and not is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.int8
        )
        self.qkv_proj_with_rope_is_fp8 = (
            has_fused_proj
            and not is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.float8_e4m3fn
        )

        self.weight_block_size = None
        if self.qkv_proj_with_rope_is_fp8 and _is_cpu and _is_cpu_amx_available:
            assert getattr(
                self.fused_qkv_a_proj_with_mqa.quant_method, "block_quant", False
            ) == getattr(self.q_b_proj.quant_method, "block_quant", False)
            use_block_quant = getattr(
                self.fused_qkv_a_proj_with_mqa.quant_method, "block_quant", False
            )

            if use_block_quant:
                assert (
                    self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.weight_block_size
                    == self.q_b_proj.quant_method.quant_config.weight_block_size
                )
                self.weight_block_size = (
                    self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.weight_block_size
                )

        self.init_mha_forward()

    def dispatch_attn_forward_method(
        self, forward_batch: ForwardBatch
    ) -> AttnForwardMethod:
        # Determine attention backend used by current forward batch
        if forward_batch.forward_mode.is_decode_or_idle():
            attention_backend = get_global_server_args().decode_attention_backend
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            # Use the specified backend for speculative operations (both verify and draft extend)
            if get_global_server_args().speculative_attention_mode == "decode":
                attention_backend = get_global_server_args().decode_attention_backend
            else:  # default to prefill
                attention_backend = get_global_server_args().prefill_attention_backend
        else:
            attention_backend = get_global_server_args().prefill_attention_backend
        self.current_attention_backend = attention_backend

        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)

    def _build_attn_diag_extra(self, forward_batch: ForwardBatch) -> Dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "forward_mode": int(forward_batch.forward_mode),
            "attention_backend": self.current_attention_backend,
            "attn_tp_rank": get_attention_tp_rank(),
            "attn_tp_size": get_attention_tp_size(),
            "attn_dp_rank": get_attention_dp_rank(),
            "attn_dp_size": get_attention_dp_size(),
        }

    @staticmethod
    def _first_non_finite_row(tensor: Optional[torch.Tensor]) -> Optional[int]:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        data = tensor.detach()
        non_finite_mask = ~torch.isfinite(data)
        if not bool(non_finite_mask.any().item()):
            return None
        if data.ndim >= 2:
            row_mask = non_finite_mask.reshape(data.shape[0], -1).any(dim=1)
            return int(row_mask.nonzero()[0].item())
        return int(non_finite_mask.reshape(-1).nonzero()[0].item())

    @staticmethod
    def _tensor_tail_ints(
        tensor: Optional[torch.Tensor], tail_size: int = 8
    ) -> Optional[list[int]]:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return []
        count = min(int(flat.numel()), tail_size)
        tail = flat[-count:].to(device="cpu")
        return [int(x) for x in tail.tolist()]

    @staticmethod
    def _tensor_numel(tensor: Optional[torch.Tensor]) -> Optional[int]:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        return int(tensor.numel())

    @staticmethod
    def _get_tensor_attr(obj: object, names: list[str]) -> Optional[torch.Tensor]:
        for name in names:
            value = getattr(obj, name, None)
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _get_q_total_rows_from_metadata(
        self, forward_batch: ForwardBatch
    ) -> Optional[int]:
        metadata = getattr(getattr(forward_batch, "attn_backend", None), "forward_metadata", None)
        qo_indptr = None
        if isinstance(metadata, (tuple, list)):
            return None
        if metadata is not None:
            qo_indptr = self._get_tensor_attr(
                metadata, ["qo_indptr", "query_start_loc", "cu_seqlens_q"]
            )
        if qo_indptr is None or qo_indptr.numel() == 0:
            return None
        return int(qo_indptr[-1].item())

    @staticmethod
    def _masked_zero_rows(
        tensor: Optional[torch.Tensor], row_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if (
            tensor is None
            or not isinstance(tensor, torch.Tensor)
            or row_mask is None
            or not isinstance(row_mask, torch.Tensor)
            or tensor.ndim == 0
            or tensor.shape[0] == 0
            or row_mask.numel() == 0
        ):
            return tensor
        rows = min(int(tensor.shape[0]), int(row_mask.shape[0]))
        if rows <= 0:
            return tensor
        row_mask = row_mask[:rows]
        if not bool(row_mask.any().item()):
            return tensor
        out = tensor.clone()
        out[:rows][row_mask] = 0
        return out

    def _sanitize_mla_padding_rows(
        self,
        forward_batch: ForwardBatch,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not _NAN_DIAG_MLA_GUARD_PADDING:
            return q, k, v, q_rope, k_rope, None
        source = q if isinstance(q, torch.Tensor) else k
        if source is None or source.ndim == 0 or source.shape[0] == 0:
            return q, k, v, q_rope, k_rope, None

        row_count = int(source.shape[0])
        row_mask = torch.zeros(row_count, dtype=torch.bool, device=source.device)
        mask_from_q_total = torch.zeros_like(row_mask)
        mask_from_out_cache_zero = torch.zeros_like(row_mask)

        q_total_rows = self._get_q_total_rows_from_metadata(forward_batch)
        if q_total_rows is not None and q_total_rows < row_count:
            mask_from_q_total[q_total_rows:row_count] = True
            row_mask |= mask_from_q_total

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if isinstance(out_cache_loc, torch.Tensor) and out_cache_loc.numel() > 0:
            n = min(row_count, int(out_cache_loc.shape[0]))
            mask_from_out_cache_zero[:n] = out_cache_loc[:n] == 0
            row_mask[:n] |= mask_from_out_cache_zero[:n]

        if not bool(row_mask.any().item()):
            return q, k, v, q_rope, k_rope, None

        masked_row_count = int(row_mask.sum().item())
        first_masked_row = int(row_mask.nonzero()[0].item())
        # One trailing masked row is common with dummy/padding slots; skip noisy logs.
        expected_tail_only = (
            masked_row_count == 1
            and first_masked_row == row_count - 1
            and (q_total_rows is None or q_total_rows >= row_count - 1)
        )

        if not expected_tail_only and _should_emit_deepseek_meta(
            "padding_guard", self.layer_id
        ):
            out_cache_zero_first_row = None
            if bool(mask_from_out_cache_zero.any().item()):
                out_cache_zero_first_row = int(mask_from_out_cache_zero.nonzero()[0].item())
            maybe_log_event(
                "deepseek_attn_mla_padding_guard",
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "masked_row_count": masked_row_count,
                    "first_masked_row": first_masked_row,
                    "masked_from_q_total_count": int(mask_from_q_total.sum().item()),
                    "masked_from_out_cache_zero_count": int(mask_from_out_cache_zero.sum().item()),
                    "out_cache_zero_first_row": out_cache_zero_first_row,
                    "q_total_rows_from_metadata": q_total_rows,
                    "source_rows": row_count,
                },
                force=True,
            )

        return (
            self._masked_zero_rows(q, row_mask),
            self._masked_zero_rows(k, row_mask),
            self._masked_zero_rows(v, row_mask),
            self._masked_zero_rows(q_rope, row_mask),
            self._masked_zero_rows(k_rope, row_mask),
            row_mask,
        )

    def _log_mla_metadata_on_anomaly(
        self,
        *,
        stage: str,
        forward_batch: ForwardBatch,
        extra: Dict[str, object],
        first_non_finite_row: Optional[int],
        positions: Optional[torch.Tensor] = None,
        source_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        if not _should_emit_deepseek_meta(f"meta_{stage}", self.layer_id):
            return
        metadata = getattr(getattr(forward_batch, "attn_backend", None), "forward_metadata", None)
        qo_indptr = None
        kv_indptr = None
        kv_indices = None
        kv_last_page_len = None
        page_table = None
        cache_seqlens = None

        info: Dict[str, object] = {
            **extra,
            "forward_batch_size": int(getattr(forward_batch, "batch_size", 0)),
            "first_non_finite_row": first_non_finite_row,
            "metadata_type": type(metadata).__name__ if metadata is not None else None,
            "source_tensor_rows": (
                int(source_tensor.shape[0])
                if isinstance(source_tensor, torch.Tensor)
                and source_tensor.ndim >= 1
                else None
            ),
            "source_tensor_shape": (
                tuple(source_tensor.shape)
                if isinstance(source_tensor, torch.Tensor)
                else None
            ),
        }

        if isinstance(metadata, (tuple, list)):
            info["metadata_is_tuple"] = True
            info["metadata_tuple_len"] = len(metadata)
            if len(metadata) > 2 and isinstance(metadata[2], torch.Tensor):
                kv_indptr = metadata[2]
            if len(metadata) > 3 and isinstance(metadata[3], torch.Tensor):
                kv_indices = metadata[3]
            if len(metadata) > 4 and isinstance(metadata[4], torch.Tensor):
                kv_last_page_len = metadata[4]
        elif metadata is not None:
            info["metadata_is_tuple"] = False
            qo_indptr = self._get_tensor_attr(
                metadata, ["qo_indptr", "query_start_loc", "cu_seqlens_q"]
            )
            kv_indptr = self._get_tensor_attr(
                metadata, ["kv_indptr", "cu_seqlens_k", "key_start_loc"]
            )
            kv_indices = self._get_tensor_attr(
                metadata, ["kv_indices", "page_table_1_flattened"]
            )
            kv_last_page_len = self._get_tensor_attr(metadata, ["kv_last_page_len"])
            page_table = self._get_tensor_attr(metadata, ["page_table", "swa_page_table"])
            cache_seqlens = self._get_tensor_attr(
                metadata, ["cache_seqlens_int32", "encoder_lens_int32"]
            )

        positions_tensor = positions if isinstance(positions, torch.Tensor) else None
        if positions_tensor is None:
            positions_tensor = getattr(forward_batch, "positions", None)
            if not isinstance(positions_tensor, torch.Tensor):
                positions_tensor = None

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        extend_start_loc = getattr(forward_batch, "extend_start_loc", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        seq_lens = getattr(forward_batch, "seq_lens", None)
        q_total_rows = None
        if qo_indptr is not None and qo_indptr.numel() > 0:
            q_total_rows = int(qo_indptr[-1].item())

        out_cache_loc_zero_count = None
        out_cache_loc_zero_indices = None
        if isinstance(out_cache_loc, torch.Tensor) and out_cache_loc.numel() > 0:
            out_zero = out_cache_loc == 0
            out_cache_loc_zero_count = int(out_zero.sum().item())
            if out_cache_loc_zero_count > 0:
                out_cache_loc_zero_indices = [
                    int(x)
                    for x in out_zero.nonzero(as_tuple=False).reshape(-1)[:8]
                    .detach()
                    .cpu()
                    .tolist()
                ]

        info.update(
            {
                "positions_numel": self._tensor_numel(positions_tensor),
                "positions_tail": self._tensor_tail_ints(positions_tensor),
                "out_cache_loc_numel": self._tensor_numel(out_cache_loc),
                "out_cache_loc_tail": self._tensor_tail_ints(out_cache_loc),
                "qo_indptr_numel": self._tensor_numel(qo_indptr),
                "qo_indptr_tail": self._tensor_tail_ints(qo_indptr),
                "kv_indptr_numel": self._tensor_numel(kv_indptr),
                "kv_indptr_tail": self._tensor_tail_ints(kv_indptr),
                "kv_indices_numel": self._tensor_numel(kv_indices),
                "kv_indices_tail": self._tensor_tail_ints(kv_indices),
                "kv_last_page_len_numel": self._tensor_numel(kv_last_page_len),
                "kv_last_page_len_tail": self._tensor_tail_ints(kv_last_page_len),
                "page_table_numel": self._tensor_numel(page_table),
                "page_table_tail": self._tensor_tail_ints(page_table),
                "cache_seqlens_numel": self._tensor_numel(cache_seqlens),
                "cache_seqlens_tail": self._tensor_tail_ints(cache_seqlens),
                "extend_start_loc_numel": self._tensor_numel(extend_start_loc),
                "extend_start_loc_tail": self._tensor_tail_ints(extend_start_loc),
                "extend_seq_lens_numel": self._tensor_numel(extend_seq_lens),
                "extend_seq_lens_tail": self._tensor_tail_ints(extend_seq_lens),
                "seq_lens_numel": self._tensor_numel(seq_lens),
                "seq_lens_tail": self._tensor_tail_ints(seq_lens),
                "q_total_rows_from_qo_indptr": q_total_rows,
                "q_rows_mismatch_with_source_tensor": (
                    bool(
                        isinstance(source_tensor, torch.Tensor)
                        and source_tensor.ndim >= 1
                        and q_total_rows is not None
                        and int(source_tensor.shape[0]) != q_total_rows
                    )
                    if isinstance(source_tensor, torch.Tensor) and source_tensor.ndim >= 1
                    else None
                ),
                "first_non_finite_row_outside_q_range": (
                    bool(first_non_finite_row is not None and q_total_rows is not None and first_non_finite_row >= q_total_rows)
                    if first_non_finite_row is not None
                    else None
                ),
                "out_cache_loc_zero_count": out_cache_loc_zero_count,
                "out_cache_loc_zero_indices": out_cache_loc_zero_indices,
            }
        )

        if (
            first_non_finite_row is not None
            and qo_indptr is not None
            and qo_indptr.numel() >= 2
        ):
            row_tensor = torch.tensor(
                [first_non_finite_row], device=qo_indptr.device, dtype=qo_indptr.dtype
            )
            req_idx = int(torch.searchsorted(qo_indptr, row_tensor, right=True).item()) - 1
            if 0 <= req_idx < qo_indptr.numel() - 1:
                q_begin = int(qo_indptr[req_idx].item())
                q_end = int(qo_indptr[req_idx + 1].item())
                info["first_non_finite_req_idx"] = req_idx
                info["first_non_finite_row_in_req"] = first_non_finite_row - q_begin
                info["first_non_finite_req_q_begin"] = q_begin
                info["first_non_finite_req_q_end_exclusive"] = q_end
                if kv_indptr is not None and kv_indptr.numel() >= req_idx + 2:
                    info["first_non_finite_req_kv_begin"] = int(kv_indptr[req_idx].item())
                    info["first_non_finite_req_kv_end_exclusive"] = int(
                        kv_indptr[req_idx + 1].item()
                    )
                if kv_last_page_len is not None and kv_last_page_len.numel() > req_idx:
                    info["first_non_finite_req_kv_last_page_len"] = int(
                        kv_last_page_len[req_idx].item()
                    )

        kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        if kv_pool is not None:
            try:
                key_buf = kv_pool.get_key_buffer(self.attn_mqa.layer_id)
                if isinstance(key_buf, torch.Tensor) and key_buf.numel() > 0:
                    slot0 = key_buf[0]
                    slot0_non_finite = ~torch.isfinite(slot0)
                    info["kv_slot0_non_finite_count"] = int(slot0_non_finite.sum().item())
                    info["kv_slot0_has_non_finite"] = bool(slot0_non_finite.any().item())

                    if isinstance(out_cache_loc, torch.Tensor) and out_cache_loc.numel() > 0:
                        max_idx = int(key_buf.shape[0]) - 1
                        last_out_idx = int(out_cache_loc[-1].item())
                        info["last_out_cache_loc_idx"] = last_out_idx
                        if 0 <= last_out_idx <= max_idx:
                            last_slot = key_buf[last_out_idx]
                            last_slot_non_finite = ~torch.isfinite(last_slot)
                            info["kv_last_out_slot_non_finite_count"] = int(
                                last_slot_non_finite.sum().item()
                            )
                            info["kv_last_out_slot_has_non_finite"] = bool(
                                last_slot_non_finite.any().item()
                            )
            except Exception as e:
                info["kv_slot_probe_error"] = str(e)

        maybe_log_event(
            f"deepseek_attn_mla_meta_{stage}",
            logger,
            extra=info,
            force=True,
        )

    def _log_mla_pre_kernel_inputs(
        self,
        *,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        extra: Dict[str, object],
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ) -> Optional[int]:
        row_hint = None
        has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_pre_kernel_q",
            q,
            logger,
            extra=extra,
        )
        if has_non_finite:
            row_hint = self._first_non_finite_row(q)
        has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_pre_kernel_k",
            k,
            logger,
            extra=extra,
        )
        if row_hint is None and has_non_finite:
            row_hint = self._first_non_finite_row(k)
        has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_pre_kernel_v",
            v,
            logger,
            extra=extra,
        )
        if row_hint is None and has_non_finite:
            row_hint = self._first_non_finite_row(v)
        has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_pre_kernel_q_rope",
            q_rope,
            logger,
            extra=extra,
        )
        if row_hint is None and has_non_finite:
            row_hint = self._first_non_finite_row(q_rope)
        has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_pre_kernel_k_rope",
            k_rope,
            logger,
            extra=extra,
        )
        if row_hint is None and has_non_finite:
            row_hint = self._first_non_finite_row(k_rope)
        return row_hint

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
        )
        return self.forward_core(s)

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        if self.attn_mha.kv_b_proj is None:
            self.attn_mha.kv_b_proj = self.kv_b_proj

        # when hidden_states is a tuple of tensors, the tuple will include quantized weight and scale tensor
        if isinstance(hidden_states, tuple):
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states[0].shape[0] == 0
            ):
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states[0]
        else:
            if (
                not get_attn_tp_context().input_scattered
                and hidden_states.shape[0] == 0
            ):
                assert (
                    not self.o_proj.reduce_results
                ), "short-circuiting allreduce will lead to hangs"
                return hidden_states, None, forward_batch, None

        attn_forward_method = self.dispatch_attn_forward_method(forward_batch)
        if attn_forward_method == AttnForwardMethod.MHA:
            inner_state = self.forward_normal_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            inner_state = self.forward_normal_chunked_kv_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            inner_state = self.forward_normal_one_shot_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA:
            inner_state = self.forward_absorb_prepare(
                positions, hidden_states, forward_batch, zero_allocator, llama_4_scaling
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
            inner_state = self.forward_absorb_fused_mla_rope_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            inner_state = self.forward_absorb_fused_mla_rope_cpu_prepare(
                positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            inner_state = forward_mha_prepare_npu(
                self, positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.MLA_NPU:
            inner_state = forward_mla_prepare_npu(
                self, positions, hidden_states, forward_batch, zero_allocator
            )
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            inner_state = forward_dsa_prepare_npu(
                self, positions, hidden_states, forward_batch, zero_allocator
            )
        else:
            raise NotImplementedError
        return None, attn_forward_method, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, attn_forward_method, forward_batch, inner_state = (
            intermediate_state
        )
        if inner_state is None:
            return hidden_states

        if attn_forward_method == AttnForwardMethod.MHA:
            return self.forward_normal_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_CHUNKED_KV:
            return self.forward_normal_chunked_kv_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_ONE_SHOT:
            return self.forward_normal_one_shot_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA:
            return self.forward_absorb_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE:
            return self.forward_absorb_fused_mla_rope_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_FUSED_ROPE_CPU:
            return self.forward_absorb_fused_mla_rope_cpu_core(*inner_state)
        elif attn_forward_method == AttnForwardMethod.MHA_NPU:
            return forward_mha_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.MLA_NPU:
            return forward_mla_core_npu(self, *inner_state)
        elif attn_forward_method == AttnForwardMethod.DSA_NPU:
            return forward_dsa_core_npu(self, *inner_state)
        else:
            raise NotImplementedError

    def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        assert self.q_lora_rank is not None
        if (
            (not isinstance(hidden_states, tuple))
            and hidden_states.shape[0] >= 1
            and hidden_states.shape[0] <= 16
            and self.use_min_latency_fused_a_gemm
        ):
            qkv_latent = dsv3_fused_a_gemm(
                hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T
            )
        else:
            qkv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        return qkv_latent

    def _fuse_rope_for_trtllm_mla(self, forward_batch: ForwardBatch) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        return (
            self.current_attention_backend == "trtllm_mla"
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        )

    def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):
        # support allgather+rerrange
        latent_cache[..., : self.kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., self.kv_lora_rank :] = k_pe.squeeze(1)
        latent_cache_output = cp_all_gather_rerange_output(
            latent_cache.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        k_nope = latent_cache_output[..., : self.kv_lora_rank].unsqueeze(1)
        k_pe = latent_cache_output[..., self.kv_lora_rank :].unsqueeze(1)
        return k_nope, k_pe

    def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                topk_indices = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:

                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True
        attn_diag_extra = self._build_attn_diag_extra(forward_batch)
        kernel_row_mask = None

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }

            pre_kernel_row = self._log_mla_pre_kernel_inputs(
                q=q_nope_out,
                k=k_nope,
                v=k_nope,
                q_rope=q_pe,
                k_rope=k_pe,
                extra=attn_diag_extra,
            )
            if pre_kernel_row is not None:
                self._log_mla_metadata_on_anomaly(
                    stage="pre_kernel",
                    forward_batch=forward_batch,
                    extra=attn_diag_extra,
                    first_non_finite_row=pre_kernel_row,
                    positions=positions,
                    source_tensor=q_nope_out,
                )
            (
                q_nope_kernel,
                k_nope_kernel,
                v_nope_kernel,
                q_pe_kernel,
                k_pe_kernel,
                kernel_row_mask,
            ) = self._sanitize_mla_padding_rows(
                forward_batch,
                q_nope_out,
                k_nope,
                k_nope,
                q_pe,
                k_pe,
            )
            attn_output = self.attn_mqa(
                q_nope_kernel,
                k_nope_kernel,
                v_nope_kernel,
                forward_batch,
                q_rope=q_pe_kernel,
                k_rope=k_pe_kernel,
                **extra_args,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            pre_kernel_row = self._log_mla_pre_kernel_inputs(
                q=q,
                k=k,
                v=k_nope,
                extra=attn_diag_extra,
            )
            if pre_kernel_row is not None:
                self._log_mla_metadata_on_anomaly(
                    stage="pre_kernel",
                    forward_batch=forward_batch,
                    extra=attn_diag_extra,
                    first_non_finite_row=pre_kernel_row,
                    positions=positions,
                    source_tensor=q,
                )
            (
                q_kernel,
                k_kernel,
                v_kernel,
                _,
                _,
                kernel_row_mask,
            ) = self._sanitize_mla_padding_rows(
                forward_batch,
                q,
                k,
                k_nope,
            )
            attn_output = self.attn_mqa(
                q_kernel,
                k_kernel,
                v_kernel,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        if kernel_row_mask is not None:
            attn_output = self._masked_zero_rows(attn_output, kernel_row_mask)
        post_kernel_has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_post_kernel",
            attn_output,
            logger,
            extra=attn_diag_extra,
        )
        if post_kernel_has_non_finite:
            self._log_mla_metadata_on_anomaly(
                stage="post_kernel",
                forward_batch=forward_batch,
                extra=attn_diag_extra,
                first_non_finite_row=self._first_non_finite_row(attn_output),
                positions=positions,
                source_tensor=attn_output,
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=attn_output.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        maybe_log_tensor_stats(
            "deepseek_attn_mla_post_bmm",
            attn_bmm_output,
            logger,
            extra=attn_diag_extra,
        )
        output, _ = self.o_proj(attn_bmm_output)
        maybe_log_tensor_stats(
            "deepseek_attn_mla_post_o_proj",
            output,
            logger,
            extra=attn_diag_extra,
        )

        return output

    def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        # NOTE: hidden_states can be a tuple for some quantization paths.
        # For shape/device/dtype, use the first tensor; still pass the original
        # hidden_states through linear ops which may accept tuple inputs.
        hidden_states_tensor = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )

        q_len = hidden_states_tensor.shape[0]
        q_input = hidden_states_tensor.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
        )

    def forward_absorb_fused_mla_rope_cpu_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"

        q_input, k_input, v_input = (
            torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self.q_b_proj.weight,
                self.w_kc,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.kv_a_layernorm.variance_epsilon,
                self.qkv_proj_with_rope_is_int8,
                self.qkv_proj_with_rope_is_fp8,
                (
                    self.fused_qkv_a_proj_with_mqa.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.fused_qkv_a_proj_with_mqa.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                (
                    self.q_b_proj.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.q_b_proj.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                True,  # is_vnni
                self.weight_block_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
        )
        return (q_input, k_input, v_input, forward_batch, zero_allocator)

    def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    ):
        attn_diag_extra = self._build_attn_diag_extra(forward_batch)
        pre_kernel_row = self._log_mla_pre_kernel_inputs(
            q=q_input,
            k=k_input,
            v=None,
            extra=attn_diag_extra,
        )
        if pre_kernel_row is not None:
            self._log_mla_metadata_on_anomaly(
                stage="pre_kernel",
                forward_batch=forward_batch,
                extra=attn_diag_extra,
                first_non_finite_row=pre_kernel_row,
                positions=positions,
                source_tensor=q_input,
            )
        q_input, k_input, _, _, _, kernel_row_mask = self._sanitize_mla_padding_rows(
            forward_batch,
            q_input,
            k_input,
            None,
        )
        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if kernel_row_mask is not None:
            attn_output = self._masked_zero_rows(attn_output, kernel_row_mask)

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        post_kernel_has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_post_kernel",
            attn_output,
            logger,
            extra=attn_diag_extra,
        )
        if post_kernel_has_non_finite:
            self._log_mla_metadata_on_anomaly(
                stage="post_kernel",
                forward_batch=forward_batch,
                extra=attn_diag_extra,
                first_non_finite_row=self._first_non_finite_row(attn_output),
                positions=positions,
                source_tensor=attn_output,
            )

        if _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        maybe_log_tensor_stats(
            "deepseek_attn_mla_post_bmm",
            attn_output,
            logger,
            extra=attn_diag_extra,
        )
        output, _ = self.o_proj(attn_output)
        maybe_log_tensor_stats(
            "deepseek_attn_mla_post_o_proj",
            output,
            logger,
            extra=attn_diag_extra,
        )

        return output

    def forward_absorb_fused_mla_rope_cpu_core(
        self, q_input, k_input, v_input, forward_batch, zero_allocator
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"

        attn_diag_extra = self._build_attn_diag_extra(forward_batch)
        pre_kernel_row = self._log_mla_pre_kernel_inputs(
            q=q_input,
            k=k_input,
            v=v_input,
            extra=attn_diag_extra,
        )
        if pre_kernel_row is not None:
            self._log_mla_metadata_on_anomaly(
                stage="pre_kernel",
                forward_batch=forward_batch,
                extra=attn_diag_extra,
                first_non_finite_row=pre_kernel_row,
                source_tensor=q_input,
            )
        q_input, k_input, v_input, _, _, kernel_row_mask = self._sanitize_mla_padding_rows(
            forward_batch,
            q_input,
            k_input,
            v_input,
        )
        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        if kernel_row_mask is not None:
            attn_output = self._masked_zero_rows(attn_output, kernel_row_mask)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        post_kernel_has_non_finite = maybe_log_tensor_stats(
            "deepseek_attn_mla_post_kernel",
            attn_output,
            logger,
            extra=attn_diag_extra,
        )
        if post_kernel_has_non_finite:
            self._log_mla_metadata_on_anomaly(
                stage="post_kernel",
                forward_batch=forward_batch,
                extra=attn_diag_extra,
                first_non_finite_row=self._first_non_finite_row(attn_output),
                source_tensor=attn_output,
            )

        # [Note] Align shapes of bmm inputs.
        # Shapes of inputs:
        #   q_nope: [M, B, K]
        #   original self.w_kc: [B, K, N]
        #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

        # Shapes of inputs to sgl_kernel.cpu.bmm:
        #   out: [B, M, N]
        #   mat1: [B, M, K]
        #   mat2: [B, N, K]
        B = self.w_vc.size(0)
        N = self.w_vc.size(1)
        M = attn_output.size(0)
        output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
        attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
        torch.ops.sgl_kernel.bmm_cpu(
            attn_bmm_output,
            attn_output.transpose(0, 1),
            self.w_vc,
            True,  # is_vnni
            None,  # scale
        )
        attn_output = output
        maybe_log_tensor_stats(
            "deepseek_attn_mla_post_bmm",
            attn_output,
            logger,
            extra=attn_diag_extra,
        )
        output, _ = self.o_proj(attn_output)
        maybe_log_tensor_stats(
            "deepseek_attn_mla_post_o_proj",
            output,
            logger,
            extra=attn_diag_extra,
        )

        return output

    @staticmethod
    def _get_q_b_proj_quant_config(quant_config):
        if envs.SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN.get():
            # refer to real DeepSeek V3 quant config
            return Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            return quant_config


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.nsa_enable_prefill_cp:
            self.layer_communicator = NSACPLayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )
        else:
            self.layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def _build_boundary_row_trace(
        self,
        *,
        row_idx: int,
        reasons: list[str],
        qo_indptr_cpu: Optional[torch.Tensor],
        req_pool_indices_cpu: Optional[torch.Tensor],
        out_cache_loc_cpu: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
    ) -> Dict[str, object]:
        trace: Dict[str, object] = {
            "row_idx": row_idx,
            "reasons": reasons,
        }

        if qo_indptr_cpu is not None and qo_indptr_cpu.numel() >= 2:
            row_tensor = torch.tensor([row_idx], dtype=qo_indptr_cpu.dtype)
            req_idx = int(torch.searchsorted(qo_indptr_cpu, row_tensor, right=True).item()) - 1
            trace["req_idx_from_qo_indptr"] = req_idx
            if 0 <= req_idx < qo_indptr_cpu.numel() - 1:
                q_begin = int(qo_indptr_cpu[req_idx].item())
                q_end = int(qo_indptr_cpu[req_idx + 1].item())
                trace["req_q_begin"] = q_begin
                trace["req_q_end_exclusive"] = q_end
                trace["row_in_req"] = row_idx - q_begin
                if (
                    req_pool_indices_cpu is not None
                    and req_pool_indices_cpu.numel() > req_idx
                ):
                    trace["req_pool_idx"] = int(req_pool_indices_cpu[req_idx].item())

        if (
            out_cache_loc_cpu is not None
            and out_cache_loc_cpu.ndim > 0
            and 0 <= row_idx < int(out_cache_loc_cpu.shape[0])
        ):
            trace["out_cache_loc_value"] = int(out_cache_loc_cpu[row_idx].item())

        if (
            isinstance(positions, torch.Tensor)
            and positions.ndim > 0
            and 0 <= row_idx < int(positions.shape[0])
        ):
            trace["position"] = int(positions[row_idx].item())

        return trace

    def _nan_diag_check_prepare_mlp_boundary(
        self,
        *,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        if not _NAN_DIAG_BOUNDARY_ROW_CHECK_ENABLE:
            return
        if hidden_states.ndim == 0:
            return
        if hidden_states.is_cuda:
            try:
                if torch.cuda.is_current_stream_capturing():
                    return
            except Exception:
                return

        row_count = int(hidden_states.shape[0])
        metadata = getattr(getattr(forward_batch, "attn_backend", None), "forward_metadata", None)
        qo_indptr = None
        if metadata is not None and not isinstance(metadata, (tuple, list)):
            qo_indptr = self.self_attn._get_tensor_attr(
                metadata, ["qo_indptr", "query_start_loc", "cu_seqlens_q"]
            )

        q_total_rows = None
        qo_indptr_cpu = None
        if isinstance(qo_indptr, torch.Tensor) and qo_indptr.numel() > 0:
            q_total_rows = int(qo_indptr[-1].item())
            qo_indptr_cpu = qo_indptr.detach().to("cpu")

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        out_cache_loc_rows = None
        out_cache_loc_cpu = None
        out_cache_zero_count = 0
        out_cache_zero_non_tail_indices: list[int] = []
        out_cache_zero_tail_begin = None
        if isinstance(out_cache_loc, torch.Tensor) and out_cache_loc.ndim > 0:
            out_cache_loc_rows = int(out_cache_loc.shape[0])
            rows_for_mask = min(row_count, out_cache_loc_rows)
            if rows_for_mask > 0:
                out_cache_loc_cpu = out_cache_loc[:rows_for_mask].detach().to("cpu")
                out_zero_cpu = out_cache_loc_cpu == 0
                zero_indices = [
                    int(x) for x in out_zero_cpu.nonzero(as_tuple=False).reshape(-1).tolist()
                ]
                out_cache_zero_count = len(zero_indices)
                if out_cache_zero_count > 0 and bool(out_zero_cpu[-1].item()):
                    idx = rows_for_mask - 1
                    while idx >= 0 and bool(out_zero_cpu[idx].item()):
                        idx -= 1
                    out_cache_zero_tail_begin = idx + 1
                if out_cache_zero_count > 0:
                    if out_cache_zero_tail_begin is None:
                        out_cache_zero_non_tail_indices = zero_indices[
                            :_NAN_DIAG_BOUNDARY_ROW_CHECK_MAX_ROWS
                        ]
                    else:
                        out_cache_zero_non_tail_indices = [
                            row
                            for row in zero_indices
                            if row < out_cache_zero_tail_begin
                        ][:_NAN_DIAG_BOUNDARY_ROW_CHECK_MAX_ROWS]

        mismatch_reasons: list[str] = []
        invalid_tail_candidates: list[tuple[int, str]] = []
        if q_total_rows is not None and q_total_rows != row_count:
            mismatch_reasons.append("row_count_vs_qo_indptr_mismatch")
            if q_total_rows < row_count:
                invalid_tail_candidates.append((q_total_rows, "tail_beyond_qo_indptr"))
        if out_cache_loc_rows is not None and out_cache_loc_rows != row_count:
            mismatch_reasons.append("row_count_vs_out_cache_loc_mismatch")
            if out_cache_loc_rows < row_count:
                invalid_tail_candidates.append(
                    (out_cache_loc_rows, "tail_beyond_out_cache_loc")
                )
        if out_cache_zero_tail_begin is not None:
            mismatch_reasons.append("tail_rows_marked_zero_in_out_cache_loc")
            invalid_tail_candidates.append(
                (out_cache_zero_tail_begin, "tail_zero_out_cache_loc")
            )
        if out_cache_zero_non_tail_indices:
            mismatch_reasons.append("non_tail_zero_in_out_cache_loc")

        if not mismatch_reasons:
            return

        req_pool_indices_cpu = None
        if isinstance(forward_batch.req_pool_indices, torch.Tensor):
            req_pool_indices_cpu = forward_batch.req_pool_indices.detach().to("cpu")

        invalid_tail_row_begin = None
        invalid_tail_reasons: list[str] = []
        if invalid_tail_candidates:
            invalid_tail_row_begin = min(start for start, _ in invalid_tail_candidates)
            invalid_tail_reasons = sorted(
                {reason for _, reason in invalid_tail_candidates}
            )

        row_traces = []
        if invalid_tail_row_begin is not None and invalid_tail_row_begin < row_count:
            row_end = min(
                row_count, invalid_tail_row_begin + _NAN_DIAG_BOUNDARY_ROW_CHECK_MAX_ROWS
            )
            for row_idx in range(invalid_tail_row_begin, row_end):
                row_reasons = []
                if q_total_rows is not None and row_idx >= q_total_rows:
                    row_reasons.append("tail_beyond_qo_indptr")
                if out_cache_loc_rows is not None and row_idx >= out_cache_loc_rows:
                    row_reasons.append("tail_beyond_out_cache_loc")
                if (
                    out_cache_zero_tail_begin is not None
                    and row_idx >= out_cache_zero_tail_begin
                ):
                    row_reasons.append("tail_zero_out_cache_loc")
                row_traces.append(
                    self._build_boundary_row_trace(
                        row_idx=row_idx,
                        reasons=row_reasons,
                        qo_indptr_cpu=qo_indptr_cpu,
                        req_pool_indices_cpu=req_pool_indices_cpu,
                        out_cache_loc_cpu=out_cache_loc_cpu,
                        positions=positions,
                    )
                )
        elif out_cache_zero_non_tail_indices:
            for row_idx in out_cache_zero_non_tail_indices:
                row_traces.append(
                    self._build_boundary_row_trace(
                        row_idx=row_idx,
                        reasons=["non_tail_zero_out_cache_loc"],
                        qo_indptr_cpu=qo_indptr_cpu,
                        req_pool_indices_cpu=req_pool_indices_cpu,
                        out_cache_loc_cpu=out_cache_loc_cpu,
                        positions=positions,
                    )
                )

        maybe_log_event(
            "deepseek_layer_hidden_pre_prepare_mlp_boundary_mismatch",
            logger,
            extra={
                "layer_id": self.layer_id,
                "forward_mode": int(forward_batch.forward_mode),
                "row_count": row_count,
                "q_total_rows_from_qo_indptr": q_total_rows,
                "out_cache_loc_rows": out_cache_loc_rows,
                "mismatch_reasons": mismatch_reasons,
                "invalid_tail_row_begin": invalid_tail_row_begin,
                "invalid_tail_row_count": (
                    row_count - invalid_tail_row_begin
                    if invalid_tail_row_begin is not None
                    else 0
                ),
                "invalid_tail_reasons": invalid_tail_reasons,
                "out_cache_zero_count": out_cache_zero_count,
                "out_cache_zero_non_tail_indices": out_cache_zero_non_tail_indices,
                "row_trace_sample": row_traces,
            },
            force=True,
        )
        if _NAN_DIAG_BOUNDARY_ROW_CHECK_ASSERT:
            raise RuntimeError(
                f"NaNDiag boundary mismatch in layer {self.layer_id}: {mismatch_reasons}"
            )

    def _sanitize_pre_mlp_padding_rows(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not _NAN_DIAG_MLP_GUARD_PADDING:
            return hidden_states, residual
        hidden_states, _, residual, _, _, _ = self.self_attn._sanitize_mla_padding_rows(
            forward_batch,
            hidden_states,
            None,
            residual,
        )
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        quant_format = (
            "mxfp4"
            if (
                _is_gfx95_supported
                and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
                is not None
                and getattr(self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None)
                is not None
                and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype == torch.uint8
            )
            else (
                "fp8"
                if (
                    _is_gfx95_supported
                    and getattr(self.self_attn, "fused_qkv_a_proj_with_mqa", None)
                    is not None
                    and getattr(
                        self.self_attn.fused_qkv_a_proj_with_mqa, "weight", None
                    )
                    is not None
                    and self.self_attn.fused_qkv_a_proj_with_mqa.weight.dtype
                    == getattr(torch, "float8_e4m3fn", None)
                )
                else ""
            )
        )

        if _should_log_deepseek_block_layer(self.layer_id):
            maybe_log_tensor_stats(
                "deepseek_layer_hidden_pre_prepare_attn",
                hidden_states,
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "is_layer_sparse": bool(self.is_layer_sparse),
                },
            )
            if residual is not None:
                maybe_log_tensor_stats(
                    "deepseek_layer_residual_pre_prepare_attn",
                    residual,
                    logger,
                    extra={
                        "layer_id": self.layer_id,
                        "forward_mode": int(forward_batch.forward_mode),
                        "is_layer_sparse": bool(self.is_layer_sparse),
                    },
                )

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format,
        )
        if _should_log_deepseek_block_layer(self.layer_id):
            maybe_log_tensor_stats(
                "deepseek_layer_hidden_post_prepare_attn",
                hidden_states,
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "is_layer_sparse": bool(self.is_layer_sparse),
                },
            )
            if residual is not None:
                maybe_log_tensor_stats(
                    "deepseek_layer_residual_post_prepare_attn",
                    residual,
                    logger,
                    extra={
                        "layer_id": self.layer_id,
                        "forward_mode": int(forward_batch.forward_mode),
                        "is_layer_sparse": bool(self.is_layer_sparse),
                    },
                )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
            llama_4_scaling=llama_4_scaling,
        )
        maybe_cuda_synchronize(
            _NAN_DIAG_SYNC_AFTER_SELF_ATTN_OUTPUT,
            "deepseek_layer_hidden_post_self_attn_sync",
            logger,
            extra={
                "layer_id": self.layer_id,
                "forward_mode": int(forward_batch.forward_mode),
                "is_layer_sparse": bool(self.is_layer_sparse),
            },
        )
        self._nan_diag_check_prepare_mlp_boundary(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )
        if _should_log_deepseek_block_layer(self.layer_id):
            post_self_attn_has_non_finite = maybe_log_tensor_stats(
                "deepseek_layer_hidden_post_self_attn",
                hidden_states,
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "is_layer_sparse": bool(self.is_layer_sparse),
                },
            )
            if post_self_attn_has_non_finite:
                self.self_attn._log_mla_metadata_on_anomaly(
                    stage="post_self_attn_output",
                    forward_batch=forward_batch,
                    extra=self.self_attn._build_attn_diag_extra(forward_batch),
                    first_non_finite_row=self.self_attn._first_non_finite_row(
                        hidden_states
                    ),
                    positions=positions,
                    source_tensor=hidden_states,
                )

        if _should_log_deepseek_block_layer(self.layer_id):
            pre_prepare_mlp_has_non_finite = maybe_log_tensor_stats(
                "deepseek_layer_hidden_pre_prepare_mlp",
                hidden_states,
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "is_layer_sparse": bool(self.is_layer_sparse),
                },
            )
            if residual is not None:
                maybe_log_tensor_stats(
                    "deepseek_layer_residual_pre_prepare_mlp",
                    residual,
                    logger,
                    extra={
                        "layer_id": self.layer_id,
                        "forward_mode": int(forward_batch.forward_mode),
                        "is_layer_sparse": bool(self.is_layer_sparse),
                    },
                )
            if pre_prepare_mlp_has_non_finite:
                self.self_attn._log_mla_metadata_on_anomaly(
                    stage="pre_prepare_mlp_input",
                    forward_batch=forward_batch,
                    extra=self.self_attn._build_attn_diag_extra(forward_batch),
                    first_non_finite_row=self.self_attn._first_non_finite_row(
                        hidden_states
                    ),
                    positions=positions,
                    source_tensor=hidden_states,
                )

        hidden_states, residual = self._sanitize_pre_mlp_padding_rows(
            hidden_states, residual, forward_batch
        )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        if _should_log_deepseek_block_layer(self.layer_id):
            maybe_log_tensor_stats(
                "deepseek_layer_hidden_post_prepare_mlp",
                hidden_states,
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "is_layer_sparse": bool(self.is_layer_sparse),
                },
            )
            if residual is not None:
                maybe_log_tensor_stats(
                    "deepseek_layer_residual_post_prepare_mlp",
                    residual,
                    logger,
                    extra={
                        "layer_id": self.layer_id,
                        "forward_mode": int(forward_batch.forward_mode),
                        "is_layer_sparse": bool(self.is_layer_sparse),
                    },
                )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if isinstance(self.mlp, DeepseekV2MLP):
            gemm_output_zero_allocator = None

        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            should_allreduce_fusion,
            use_reduce_scatter,
            gemm_output_zero_allocator,
        )
        if _should_log_deepseek_block_layer(self.layer_id):
            maybe_log_tensor_stats(
                "deepseek_layer_hidden_post_mlp",
                hidden_states,
                logger,
                extra={
                    "layer_id": self.layer_id,
                    "forward_mode": int(forward_batch.forward_mode),
                    "is_layer_sparse": bool(self.is_layer_sparse),
                    "should_allreduce_fusion": bool(should_allreduce_fusion),
                    "use_reduce_scatter": bool(use_reduce_scatter),
                },
            )

        if not self.nsa_enable_prefill_cp and should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True

        if not should_allreduce_fusion:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
            if _should_log_deepseek_block_layer(self.layer_id):
                maybe_log_tensor_stats(
                    "deepseek_layer_hidden_post_postprocess",
                    hidden_states,
                    logger,
                    extra={
                        "layer_id": self.layer_id,
                        "forward_mode": int(forward_batch.forward_mode),
                        "is_layer_sparse": bool(self.is_layer_sparse),
                    },
                )
                if residual is not None:
                    maybe_log_tensor_stats(
                        "deepseek_layer_residual_post_postprocess",
                        residual,
                        logger,
                        extra={
                            "layer_id": self.layer_id,
                            "forward_mode": int(forward_batch.forward_mode),
                            "is_layer_sparse": bool(self.is_layer_sparse),
                        },
                    )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                zero_allocator=zero_allocator,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        if not (
            enable_moe_dense_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            state.hidden_states_mlp_output = self.mlp(
                hidden_states, state.forward_batch
            )
        else:
            state.hidden_states_mlp_output = hidden_states

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            zero_allocator=state.zero_allocator,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "zero_allocator",
                "tbo_subbatch_index",
            }
        )
        return output


class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.first_k_dense_replace = config.first_k_dense_replace
        self.pp_group = get_pp_group()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_size = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
            offloader_kwargs=dict(
                submodule_accessor=lambda layer: (
                    layer.mlp.experts
                    if isinstance(layer.mlp, DeepseekV2MoE)
                    else layer.mlp
                ),
                whitelist_param_names_creator=lambda module: (
                    [
                        "w13_weight",
                        "w2_weight",
                        # only for nvfp4
                        *(
                            [
                                "w13_blockscale_swizzled",
                                "w2_blockscale_swizzled",
                            ]
                            if hasattr(module, "w13_blockscale_swizzled")
                            else []
                        ),
                    ]
                    if isinstance(module, FusedMoE)
                    else []
                ),
            ),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        self.gemm_output_zero_allocator_size = 0
        if (
            _use_aiter_gfx95
            and config.n_routed_experts == 256
            and self.embed_tokens.embedding_dim == 7168
        ):
            num_moe_layers = sum(
                [
                    1
                    for i in range(len(self.layers))
                    if isinstance(self.layers[i].mlp, DeepseekV2MoE)
                ]
            )

            allocate_size = 0
            for i in range(len(self.layers)):
                if isinstance(self.layers[i].mlp, DeepseekV2MoE):
                    tp_size = get_tensor_model_parallel_world_size()
                    intermediate_size = (
                        config.moe_intermediate_size * config.n_shared_experts
                    )
                    share_expert_output_size_per_partition = divide(
                        intermediate_size * 2, tp_size
                    )
                    allocate_size = share_expert_output_size_per_partition
                    break

            self.gemm_output_zero_allocator_size = (
                get_dsv3_gemm_output_zero_allocator_size(
                    config.n_routed_experts,
                    num_moe_layers,
                    allocate_size,
                    self.embed_tokens.embedding_dim,
                )
            )
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        # llama_4_scaling: for supporting Mistral-Large-3 model
        self.llama_4_scaling_config = getattr(config, "llama_4_scaling", None)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )

        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        maybe_log_tensor_stats(
            "deepseek_hidden_states_post_embed_or_pp_recv",
            hidden_states,
            logger,
            extra={
                "pp_is_first_rank": self.pp_group.is_first_rank,
                "pp_is_last_rank": self.pp_group.is_last_rank,
                "forward_mode": int(forward_batch.forward_mode),
            },
        )
        if residual is not None:
            maybe_log_tensor_stats(
                "deepseek_residual_post_pp_recv",
                residual,
                logger,
                extra={
                    "pp_is_first_rank": self.pp_group.is_first_rank,
                    "pp_is_last_rank": self.pp_group.is_last_rank,
                    "forward_mode": int(forward_batch.forward_mode),
                },
            )

        if nsa_use_prefill_cp(forward_batch):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        # llama_4_scaling: for supporting Mistral-Large-3 model
        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling: Optional[torch.Tensor] = None
        if self.llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=self.llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=self.llama_4_scaling_config["beta"],
                positions=positions,
            )

        normal_start_layer = self.start_layer
        normal_end_layer = self.end_layer
        if forward_batch.can_run_tbo:
            if (
                self.first_k_dense_replace > normal_start_layer
                and self.first_k_dense_replace < normal_end_layer
            ):
                normal_end_layer = self.first_k_dense_replace
            elif self.first_k_dense_replace < normal_start_layer:
                normal_end_layer = normal_start_layer = 0
        aux_hidden_states = []
        for i in range(normal_start_layer, normal_end_layer):
            # NOTE: torch dynamo does not support graph break in context manager
            ctx = (
                nullcontext()
                if get_global_server_args().enable_piecewise_cuda_graph
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                if i in self.layers_to_capture:
                    if self.enable_a2a_moe and i > self.first_k_dense_replace:
                        aux_hidden_state = tensor_model_parallel_all_gather(
                            hidden_states + residual, dim=0
                        )
                        aux_hidden_states.append(aux_hidden_state)
                    else:
                        aux_hidden_states.append(hidden_states + residual)
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                )
                if _should_log_deepseek_layer(i, normal_start_layer, normal_end_layer):
                    maybe_log_tensor_stats(
                        "deepseek_hidden_states_post_layer",
                        hidden_states,
                        logger,
                        extra={
                            "layer_id": i,
                            "forward_mode": int(forward_batch.forward_mode),
                            "tbo_enabled": bool(forward_batch.can_run_tbo),
                        },
                    )
                    if residual is not None:
                        maybe_log_tensor_stats(
                            "deepseek_residual_post_layer",
                            residual,
                            logger,
                            extra={
                                "layer_id": i,
                                "forward_mode": int(forward_batch.forward_mode),
                                "tbo_enabled": bool(forward_batch.can_run_tbo),
                            },
                        )

        if normal_end_layer != self.end_layer:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers[normal_end_layer : self.end_layer],
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
                input_data_scatter_mode=self.layers[
                    normal_end_layer - 1
                ].layer_scatter_modes.layer_output_mode,
                zero_allocator=zero_allocator,
            )
            maybe_log_tensor_stats(
                "deepseek_hidden_states_post_tbo",
                hidden_states,
                logger,
                extra={
                    "forward_mode": int(forward_batch.forward_mode),
                    "normal_end_layer": normal_end_layer,
                    "end_layer": self.end_layer,
                },
            )
            if residual is not None:
                maybe_log_tensor_stats(
                    "deepseek_residual_post_tbo",
                    residual,
                    logger,
                    extra={
                        "forward_mode": int(forward_batch.forward_mode),
                        "normal_end_layer": normal_end_layer,
                        "end_layer": self.end_layer,
                    },
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if not forward_batch.forward_mode.is_idle():
                maybe_log_tensor_stats(
                    "deepseek_hidden_states_pre_final_norm",
                    hidden_states,
                    logger,
                    extra={"forward_mode": int(forward_batch.forward_mode)},
                )
                if residual is not None:
                    maybe_log_tensor_stats(
                        "deepseek_residual_pre_final_norm",
                        residual,
                        logger,
                        extra={"forward_mode": int(forward_batch.forward_mode)},
                    )
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)
                maybe_log_tensor_stats(
                    "deepseek_hidden_states_post_final_norm",
                    hidden_states,
                    logger,
                    extra={"forward_mode": int(forward_batch.forward_mode)},
                )

        if self.pp_group.is_last_rank and nsa_use_prefill_cp(forward_batch):
            # allgather + rerrange
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            maybe_log_tensor_stats(
                "deepseek_hidden_states_post_cp_all_gather",
                hidden_states,
                logger,
                extra={
                    "forward_mode": int(forward_batch.forward_mode),
                    "cp_size": self.cp_size,
                },
            )
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class DeepseekV2ForCausalLM(nn.Module, DeepseekV2WeightLoaderMixin):
    # for quark model load
    packed_modules_mapping = {}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # for quark model load
        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.pp_group = get_pp_group()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.use_nsa = is_deepseek_nsa(config)
        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, DeepseekV2MoE)
            }
        )
        self.capture_aux_hidden_states = False

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_rank = self.cp_size = None

        q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
        get_attn_tp_context().init_context(q_lora_rank, is_deepseek_nsa(config))

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(
        self, architecture: str = "DeepseekV3ForCausalLM"
    ):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if (
            self.config.architectures[0] != architecture
            or self.config.n_routed_experts != 256
            or self.config.n_shared_experts != 1
        ):
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, self.use_nsa, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        hidden_states_for_diag = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )
        if torch.is_tensor(hidden_states_for_diag):
            maybe_log_tensor_stats(
                "deepseek_model_output_hidden_states_entry",
                hidden_states_for_diag,
                logger,
                extra={
                    "forward_mode": int(forward_batch.forward_mode),
                    "capture_aux_hidden_states": bool(self.capture_aux_hidden_states),
                },
            )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            maybe_log_tensor_stats(
                "deepseek_hidden_states_pre_logits_processor",
                hidden_states,
                logger,
                extra={
                    "forward_mode": int(forward_batch.forward_mode),
                    "capture_aux_hidden_states": bool(self.capture_aux_hidden_states),
                },
            )
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)
        self.post_load_weights(is_nextn=is_nextn)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM, DeepseekV32ForCausalLM]
