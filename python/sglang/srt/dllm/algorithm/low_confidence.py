from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        # This can happen for two scenario:
        # 1) Normal prefill request or 2) Refresh request
        # For Refresh request, it should return the same next_token_ids as the input_ids to trigger the post-processing logic

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids_list = []
            denoise_req_rids = forward_batch.denoise_req_rids or set()
            for block_id in range(batch_size):
                rid = forward_batch.rids[block_id]
                if rid in denoise_req_rids:
                    block_start = block_id * self.block_size
                    block_end = block_start + self.block_size
                    block_input_ids = forward_batch.input_ids[block_start:block_end]
                    next_token_ids_list.append(block_input_ids)
                else:
                    next_token_ids_list.append(
                        torch.empty(
                            (0,),
                            dtype=torch.int64,
                            device=forward_batch.input_ids.device,
                        )
                    )

            return logits_output, next_token_ids_list, can_run_cuda_graph

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end,]
            block_mask_index = block_input_ids == self.mask_id
            curr_logits = logits_output.full_logits[curr_block_start:curr_block_end,]

            x = torch.argmax(curr_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(curr_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )
            x = torch.where(block_mask_index, x, block_input_ids)
            confidence = torch.where(block_mask_index, p, -np.inf)

            transfer_index = confidence > self.threshold

            if transfer_index.sum().item() == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True

            block_input_ids[transfer_index] = x[transfer_index]

        updated_input_ids_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            updated_input_ids_list.append(
                forward_batch.input_ids[block_start:block_end].clone()
            )
        if logits_output.customized_info is None:
            logits_output.customized_info = {}
        logits_output.customized_info["updated_input_ids"] = updated_input_ids_list

        # only advance the generated tokens as refresh request, so normal decode path should always return empty next_token_ids to avoid affecting the post-processing logic
        next_token_ids_list = []
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
