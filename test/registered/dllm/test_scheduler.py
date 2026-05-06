import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1400, suite="stage-b-test-1-gpu-large")

BASE_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.9",
    "--max-running-requests",
    "4",
    "--attention-backend",
    "flashinfer",
    "--cuda-graph-bs",
    "1",
    "2",
    "3",
    "4",
]

BATCH_SIZE_COVERAGES = [1, 4]


class BaseTestCase(CustomTestCase):
    dllm_algorithm: str | None = None
    model: str | None = None
    diable_radix_cache: bool | None = None
    server_args = BASE_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        if cls.dllm_algorithm is None:
            raise unittest.SkipTest("Skip the base observability test class")

        assert cls.model is not None, "model should be set in subclass"

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._build_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @classmethod
    def _build_server_args(cls):
        args = cls.server_args.copy()
        assert (
            cls.dllm_algorithm is not None
        ), "dllm_algorithm should be set in subclass"
        assert (
            cls.diable_radix_cache is not None
        ), "diable_radix_cache should be set in subclass"

        args.extend(["--dllm-algorithm", cls.dllm_algorithm])

        if cls.diable_radix_cache:
            args.append("--disable-radix-cache")
        return args

    def test_gsm8k(self):
        for batch_size in BATCH_SIZE_COVERAGES:
            with self.subTest(batch_size=batch_size):
                args = SimpleNamespace(
                    base_url=self.base_url,
                    model=self.model,
                    eval_name="gsm8k",
                    api="completion",
                    max_tokens=512,
                    num_examples=100,
                    num_threads=batch_size,
                    repeat=1,
                )
                metrics = run_eval(args)
                print(f"{metrics=}")

                if self.model == "inclusionAI/LLaDA2.0-mini":
                    self.assertGreater(metrics["score"], 0.86)
                elif self.model == "JetLM/SDAR-8B-Chat":
                    self.assertGreater(metrics["score"], 0.9)
                else:
                    raise ValueError(f"Unexpected model {self.model}")

    def test_bs_1_speed(self):
        args = BenchArgs(
            port=int(self.base_url.split(":")[-1]), max_new_tokens=512, batch_size=1
        )
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

        # speed on H100
        if self.model == "inclusionAI/LLaDA2.0-mini":
            self.assertGreater(speed, 220)
        elif self.model == "JetLM/SDAR-8B-Chat":
            self.assertGreater(speed, 80)
        else:
            raise ValueError(f"Unexpected model {self.model}")


class TestSDAR8BLowConfidenceChunkCache(BaseTestCase):
    model = "JetLM/SDAR-8B-Chat"
    dllm_algorithm = "LowConfidence"
    diable_radix_cache = True


class TestSDAR8BLowConfidenceRadixCache(BaseTestCase):
    model = "JetLM/SDAR-8B-Chat"
    dllm_algorithm = "LowConfidence"
    diable_radix_cache = False


class TestLLaDA2MiniLowConfidenceChunkCache(BaseTestCase):
    model = "inclusionAI/LLaDA2.0-mini"
    dllm_algorithm = "LowConfidence"
    diable_radix_cache = True


class TestLLaDA2MiniLowConfidenceRadixCache(BaseTestCase):
    model = "inclusionAI/LLaDA2.0-mini"
    dllm_algorithm = "LowConfidence"
    diable_radix_cache = False


class TestLLaDA2MiniJointThresholdRadixCache(BaseTestCase):
    model = "inclusionAI/LLaDA2.0-mini"
    dllm_algorithm = "JointThreshold"
    diable_radix_cache = False


class TestSDAR8BJointThresholdRadixCache(BaseTestCase):
    model = "JetLM/SDAR-8B-Chat"
    dllm_algorithm = "JointThreshold"
    diable_radix_cache = False


if __name__ == "__main__":
    unittest.main()
