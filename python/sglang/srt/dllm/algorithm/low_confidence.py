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
        denoise_req_rids = forward_batch.denoise_req_rids or set()
        batch_size = forward_batch.batch_size
        assert (
            batch_size == forward_batch.input_ids.shape[0] // self.block_size
        ), "Batch size should be consistent with input_ids and block size"

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids_list = []
        update_ids_list = []

        for batch_id in range(batch_size):
            curr_block_start = batch_id * self.block_size
            curr_block_end = curr_block_start + self.block_size
            block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end,]
            block_mask_index = block_input_ids == self.mask_id

            if sum(block_mask_index).item() == 0:
                # no mask token in this block -> refresh path or prefill path
                if forward_batch.rids[batch_id] in denoise_req_rids:
                    # refresh path
                    block_start = batch_id * self.block_size
                    block_end = block_start + self.block_size
                    block_input_ids = forward_batch.input_ids[block_start:block_end]
                    next_token_ids_list.append(block_input_ids)
                    update_ids_list.append(
                        torch.empty(
                            (0,),
                            dtype=torch.int64,
                            device=forward_batch.input_ids.device,
                        )
                    )
                else:
                    # prefill path,
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
            else:
                # normal decode path
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]
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

                next_token_ids_list.append(
                    torch.empty(
                        (0,), dtype=torch.int64, device=forward_batch.input_ids.device
                    )
                )
                update_ids_list.append(
                    forward_batch.input_ids[curr_block_start:curr_block_end].clone()
                )

        if logits_output.customized_info is None:
            logits_output.customized_info = {}
        logits_output.customized_info["update_ids_list"] = update_ids_list

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
