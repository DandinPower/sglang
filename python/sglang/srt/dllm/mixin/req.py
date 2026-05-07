from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@dataclass
class RollBackState:
    kv_committed_len: int
    kv_allocated_len: int
    kv_committed_freed: bool
    kv_overallocated_freed: bool
    extend_batch_idx: int
    is_chunked: int
    already_computed: int
    dllm_block_offset: int


def get_rollback_state(req: Req) -> RollBackState:
    return RollBackState(
        kv_committed_len=req.kv_committed_len,
        kv_allocated_len=req.kv_allocated_len,
        kv_committed_freed=req.kv_committed_freed,
        kv_overallocated_freed=req.kv_overallocated_freed,
        extend_batch_idx=req.extend_batch_idx,
        is_chunked=req.is_chunked,
        already_computed=req.already_computed,
        dllm_block_offset=req.dllm_block_offset,
    )


def restore_rollback_state(req: Req, state: RollBackState):
    req.kv_committed_len = state.kv_committed_len
    req.kv_allocated_len = state.kv_allocated_len
    req.kv_committed_freed = state.kv_committed_freed
    req.kv_overallocated_freed = state.kv_overallocated_freed
    req.extend_batch_idx = state.extend_batch_idx
    req.is_chunked = state.is_chunked
    req.already_computed = state.already_computed
    req.dllm_block_offset = state.dllm_block_offset


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_block_offset = 0
        self.dllm_config = dllm_config
        self.snapshot_state: Optional[RollBackState] = None
        self.update_ids: Optional[torch.Tensor] = None
        self.dllm_algorithm_state: dict[str, Any] = {}

        if self.dllm_config is not None:
            if len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

            self._init_dllm_algorithm_state()

    def _init_dllm_algorithm_state(self: Req):
        assert (
            self.dllm_config is not None
        ), "dllm_config should not be None when initializing dllm algorithm state"
        if self.dllm_config.algorithm == "LowConfidence":
            self.dllm_algorithm_state = {
                "current_block_finished": False,  # used for decide prefill or refresh in the next round
                "fwd_counts": 0,  # used for tracking the forward steps in current block
            }
        elif self.dllm_config.algorithm == "JointThreshold":
            self.dllm_algorithm_state = {
                "prompt_masks": None,  # list of bool tensor indicating the prompt tokens in each block
                "current_block_finished": False,  # used for decide prefill or refresh in the next round
                "post_edit_steps": 0,  # used for tracking the post edit steps
                "fwd_counts": 0,  # used for tracking the forward steps in current block
            }
        else:
            raise ValueError(f"Unsupported DLLM algorithm {self.dllm_config.algorithm}")

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def determine_dllm_phase(self: Req):
        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if not self.fill_ids:
            self.dllm_block_offset = 0
        elif self.update_ids is None:
            self.dllm_block_offset += self.dllm_config.block_size

        base_fill_ids = self.origin_input_ids + self.output_ids
        if self.update_ids is not None:
            # this means we need to rebuild the fill ids with latest updated input ids
            assert (
                len(self.update_ids) == self.dllm_config.block_size
            ), f"Unexpected update ids length: {len(self.update_ids)}"
            aligned_base_len = (
                len(base_fill_ids) // self.dllm_config.block_size
            ) * self.dllm_config.block_size
            self.fill_ids = base_fill_ids[:aligned_base_len] + self.update_ids.tolist()
        else:
            # this is the normal path where we initialize the request, or the last round is refresh or prefill
            self.fill_ids = (
                base_fill_ids + [self.dllm_config.mask_id] * self.dllm_config.block_size
            )

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
