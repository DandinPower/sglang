import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import _get_dllm_refresh_append_ids
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


def _dllm_config():
    return DllmConfig(
        algorithm="LowConfidence",
        algorithm_config={},
        block_size=4,
        mask_id=99,
        max_running_requests=1,
    )


def _make_req(origin_input_ids, output_ids=None):
    req = Req(
        rid="rid",
        origin_input_text="",
        origin_input_ids=origin_input_ids,
        sampling_params=SamplingParams(),
        dllm_config=_dllm_config(),
    )
    req.output_ids = list(output_ids or [])
    return req


class TestDllmBlockOverlap(CustomTestCase):
    def test_update_ids_align_non_aligned_prompt(self):
        req = _make_req([1, 2, 3, 4, 5, 6])
        req.update_ids = torch.tensor([50, 51, 52, 53], dtype=torch.int64)

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.fill_ids, [1, 2, 3, 4, 50, 51, 52, 53])

    def test_update_ids_align_short_prompt(self):
        req = _make_req([1])
        req.update_ids = torch.tensor([50, 51, 52, 53], dtype=torch.int64)

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.fill_ids, [50, 51, 52, 53])

    def test_update_ids_preserve_aligned_base(self):
        req = _make_req([1, 2, 3, 4], output_ids=[5, 6, 7, 8])
        req.update_ids = torch.tensor([50, 51, 52, 53], dtype=torch.int64)

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.fill_ids, [1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 52, 53])

    def test_no_update_ids_keeps_prompt_tail_before_masks(self):
        req = _make_req([1, 2, 3, 4, 5, 6])

        req._init_fill_ids_for_dllm()

        self.assertEqual(req.fill_ids, [1, 2, 3, 4, 5, 6, 99, 99, 99, 99])

    def test_refresh_append_skips_prompt_overlap_for_denoise_req(self):
        req = SimpleNamespace(
            rid="rid", origin_input_ids=[1, 2, 3, 4, 5, 6], output_ids=[]
        )

        append_ids = _get_dllm_refresh_append_ids(
            req=req,
            next_token_ids=[10, 11, 12, 13],
            block_size=4,
            denoise_req_rids={"rid"},
        )

        self.assertEqual(append_ids, [12, 13])

    def test_refresh_append_keeps_aligned_denoise_block(self):
        req = SimpleNamespace(
            rid="rid", origin_input_ids=[1, 2, 3, 4], output_ids=[5, 6, 7, 8]
        )

        append_ids = _get_dllm_refresh_append_ids(
            req=req,
            next_token_ids=[10, 11, 12, 13],
            block_size=4,
            denoise_req_rids={"rid"},
        )

        self.assertEqual(append_ids, [10, 11, 12, 13])

    def test_refresh_append_keeps_non_denoise_block(self):
        req = SimpleNamespace(
            rid="other", origin_input_ids=[1, 2, 3, 4, 5, 6], output_ids=[]
        )

        append_ids = _get_dllm_refresh_append_ids(
            req=req,
            next_token_ids=[10, 11, 12, 13],
            block_size=4,
            denoise_req_rids={"rid"},
        )

        self.assertEqual(append_ids, [10, 11, 12, 13])

    def test_refresh_append_keeps_suffix_shaped_ids(self):
        req = SimpleNamespace(
            rid="rid", origin_input_ids=[1, 2, 3, 4, 5, 6], output_ids=[]
        )

        append_ids = _get_dllm_refresh_append_ids(
            req=req,
            next_token_ids=[12, 13],
            block_size=4,
            denoise_req_rids={"rid"},
        )

        self.assertEqual(append_ids, [12, 13])


if __name__ == "__main__":
    unittest.main()
