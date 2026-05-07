from enum import Enum, auto

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class DllmPath(Enum):
    REFRESH = auto()
    PREFILL = auto()
    DECODE = auto()


def decide_dllm_path(
    current_block_finished: bool, prompt_masks: torch.Tensor | None, num_masks: int
) -> DllmPath:
    """
    Dispatch path logic:
    - If current_block_finished == True, use refresh.
    - If current_block_finished == False, prompt_masks == None, and num_masks == 0, use prefill.
    - Otherwise, use decode.
    """
    if current_block_finished:
        return DllmPath.REFRESH

    if prompt_masks is None and num_masks == 0:
        return DllmPath.PREFILL

    return DllmPath.DECODE


class JointThreshold(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        # Since the current model does not support edit behavior well, we tuned the threshold to match the LowConfidence behavior, which can still evaluate whether the implementation is correct. Complete testing can be done after the new model that supports edit behavior well becomes available.
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 1)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        dllm_algorithm_states = forward_batch.dllm_algorithm_states
        batch_size = forward_batch.batch_size

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids_list = []
        update_ids_list = []
        fwd_counts_list = []

        for batch_id in range(batch_size):
            block_start = batch_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end,]
            block_mask_index = block_input_ids == self.mask_id

            state = dllm_algorithm_states[batch_id]
            state["fwd_counts"] += 1
            num_masks = block_mask_index.sum().item()
            path = decide_dllm_path(
                current_block_finished=state["current_block_finished"],
                prompt_masks=state.get("prompt_masks"),
                num_masks=num_masks,
            )

            if path == DllmPath.REFRESH:
                next_token_ids_list.append(block_input_ids)
                update_ids_list.append(
                    torch.empty(
                        (0,),
                        dtype=torch.int64,
                        device=forward_batch.input_ids.device,
                    )
                )
                fwd_counts_list.append(state["fwd_counts"])
                # reset state for the next block processing
                state["prompt_masks"] = None
                state["current_block_finished"] = False
                state["post_edit_steps"] = 0
                state["fwd_counts"] = 0
            elif path == DllmPath.PREFILL:
                next_token_ids_list.append(
                    torch.empty(
                        (0,),
                        dtype=torch.int64,
                        device=forward_batch.input_ids.device,
                    )
                )
                update_ids_list.append(
                    torch.empty(
                        (0,),
                        dtype=torch.int64,
                        device=forward_batch.input_ids.device,
                    )
                )
                # Only monitor once per denoised block, so the timing occurs during the refresh stage, the other stage append None to indicate no monitoring.
                fwd_counts_list.append(None)
                state["fwd_counts"] = 0
            else:
                # decode
                if state["prompt_masks"] is None:
                    # this block has never been processed, record the prompt mask
                    prompt_mask = block_input_ids != self.mask_id
                    state["prompt_masks"] = prompt_mask

                logits = logits_output.full_logits[block_start:block_end]

                if self.penalty_lambda > 0:
                    prev_ids = block_input_ids[:-1]
                    logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )

                # Mask to token (M2T)
                mask_transfer_index = torch.zeros_like(block_mask_index)
                if block_mask_index.any():
                    confidence = torch.where(block_mask_index, p, -np.inf)
                    mask_transfer_index = confidence > self.threshold

                    if not mask_transfer_index.any():
                        _, select_index = torch.topk(confidence, k=1)
                        mask_transfer_index[select_index] = True
                else:
                    state["post_edit_steps"] += 1

                # Token to token (T2T)
                edit_mask = ~block_mask_index & ~state["prompt_masks"]
                edit_transfer_index = (
                    (p > self.edit_threshold) & (block_input_ids != x) & edit_mask
                )

                transfer_index = mask_transfer_index | edit_transfer_index

                if not transfer_index.any() or (
                    state["post_edit_steps"] > self.max_post_edit_steps
                    and self.max_post_edit_steps == 0
                ):
                    # 1. If no token is transferred, it means this forward pass is already stable and the KV cache has been refreshed. Therefore, we can skip the later refresh steps.
                    # 2. If post_edit_steps exceeds the maximum steps, this only happens when max_post_edit_steps is equal to zero, which means this round behaves like a refresh since we do not need post edit steps. Therefore, we can also skip the later refresh steps.
                    next_token_ids_list.append(block_input_ids)
                    update_ids_list.append(
                        torch.empty(
                            (0,),
                            dtype=torch.int64,
                            device=forward_batch.input_ids.device,
                        )
                    )
                    # Only monitor once per denoised block, so the timing occurs during the refresh stage, the other stage append None to indicate no monitoring.
                    fwd_counts_list.append(state["fwd_counts"])

                    state["prompt_masks"] = None
                    state["current_block_finished"] = False
                    state["post_edit_steps"] = 0
                    state["fwd_counts"] = 0
                else:
                    if (
                        state["post_edit_steps"] == self.max_post_edit_steps
                        and self.max_post_edit_steps > 0
                    ):
                        # This means the current round is the last post edit round. After this round, we should run the refresh logic in the next round.
                        # We should prevent into this path for the max_post_edit_steps == 0 scenario, otherwise it will refresh immediately even we still at mask transferred stage.
                        state["current_block_finished"] = True
                    block_input_ids[transfer_index] = x[transfer_index]
                    next_token_ids_list.append(
                        torch.empty(
                            (0,),
                            dtype=torch.int64,
                            device=forward_batch.input_ids.device,
                        )
                    )
                    update_ids_list.append(block_input_ids.clone())
                    # Only monitor once per denoised block, so the timing occurs during the refresh stage, the other stage append None to indicate no monitoring.
                    fwd_counts_list.append(None)

        if logits_output.customized_info is None:
            logits_output.customized_info = {}
        logits_output.customized_info["update_ids_list"] = update_ids_list
        logits_output.customized_info["fwd_counts_list"] = fwd_counts_list

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = JointThreshold
