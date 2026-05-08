import json
from dataclasses import dataclass
from typing import Any

import requests

OBSERVABILITY_MAX_NEW_TOKENS = 100
OBSERVABILITY_SEED = 42
FORWARD_COUNTS_KEY = "dllm_forward_counts_per_block"
TIME_BETWEEN_BLOCK_KEY = "dllm_time_between_blocks"
PROMPT_1 = (
    "Human: What is the capital of France and how is that city like. "
    "Give me 3 trivial information about that city. "
    "Write in a format of json.\nAssistant:"
)
PROMPT_2 = (
    "Human: What is the capital of Germany and how is that city like. "
    "Give me 3 trivial information about that city. "
    "Write in a format of json.\nAssistant:"
)


@dataclass
class DllmObservabilityMetrics:
    forward_counts: list[int]
    time_between_block: list[int | float]


def build_single_prompt() -> list[str]:
    return [PROMPT_1]


def build_two_prompts() -> list[str]:
    return [PROMPT_1, PROMPT_2]


def _build_generate_payload(
    prompts: list[str],
    stream: bool,
    stream_interval: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": prompts[0] if len(prompts) == 1 else prompts,
        "sampling_params": {
            "sampling_seed": OBSERVABILITY_SEED,
            "temperature": 0.0,
            "max_new_tokens": OBSERVABILITY_MAX_NEW_TOKENS,
            "ignore_eos": True,
        },
        "stream": stream,
    }
    if stream_interval is not None:
        payload["sampling_params"]["stream_interval"] = stream_interval
    return payload


def _request_context(request_index: int | None) -> str:
    return "" if request_index is None else f" for request {request_index}"


def _extract_meta_info(
    output: dict[str, Any],
    request_index: int | None,
) -> dict[str, Any]:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        raise AssertionError(
            f"Missing meta_info{_request_context(request_index)}: {output}"
        )
    return meta_info


def _extract_metric_list(
    meta_info: dict[str, Any],
    metric_key: str,
    request_index: int | None,
) -> list[Any]:
    metric_values = meta_info.get(metric_key)
    if not isinstance(metric_values, list):
        raise AssertionError(
            f"Expected {metric_key}{_request_context(request_index)} to be a list, "
            f"got: {type(metric_values)}"
        )
    return metric_values


def _validate_forward_counts(
    counts: list[Any],
    request_index: int | None,
) -> list[int]:
    validated_counts: list[int] = []
    for position, count in enumerate(counts):
        if not isinstance(count, int):
            raise AssertionError(
                f"Every {FORWARD_COUNTS_KEY} entry{_request_context(request_index)} "
                f"should be an int, got {count!r} at position {position}"
            )
        if count < 1:
            raise AssertionError(
                f"Every {FORWARD_COUNTS_KEY} entry{_request_context(request_index)} "
                f"should be >= 1, got {count} at position {position}"
            )
        validated_counts.append(count)
    return validated_counts


def _validate_time_between_block_types(
    time_between_block: list[Any],
    request_index: int | None,
) -> list[int | float]:
    validated_time_between_block: list[int | float] = []
    for position, value in enumerate(time_between_block):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssertionError(
                f"Every {TIME_BETWEEN_BLOCK_KEY} entry"
                f"{_request_context(request_index)} should be numeric, got {value!r} "
                f"at position {position}"
            )
        validated_time_between_block.append(value)
    return validated_time_between_block


def _validate_time_between_block(
    time_between_block: list[Any],
    expected_length: int,
    request_index: int | None,
) -> list[int | float]:
    if len(time_between_block) != expected_length:
        raise AssertionError(
            f"Expected {TIME_BETWEEN_BLOCK_KEY}{_request_context(request_index)} "
            f"to match {FORWARD_COUNTS_KEY} length: expected length "
            f"{expected_length}, got {len(time_between_block)}"
        )

    if not time_between_block:
        raise AssertionError(
            f"Expected {TIME_BETWEEN_BLOCK_KEY}{_request_context(request_index)} "
            "to be non-empty"
        )

    validated_time_between_block = _validate_time_between_block_types(
        time_between_block,
        request_index,
    )
    first_value = validated_time_between_block[0]
    if first_value != -1:
        raise AssertionError(
            f"The first {TIME_BETWEEN_BLOCK_KEY} entry"
            f"{_request_context(request_index)} should be -1, got {first_value!r}"
        )

    for position, value in enumerate(validated_time_between_block[1:], start=1):
        if value <= 0:
            raise AssertionError(
                f"Every {TIME_BETWEEN_BLOCK_KEY} entry after the first"
                f"{_request_context(request_index)} should be > 0, got {value!r} "
                f"at position {position}"
            )
    return validated_time_between_block


def _format_time_between_block(time_between_block: list[int | float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in time_between_block) + "]"


def _extract_observability_metrics(
    output: dict[str, Any],
    request_index: int | None = None,
) -> DllmObservabilityMetrics:
    meta_info = _extract_meta_info(output, request_index)
    forward_counts = _validate_forward_counts(
        _extract_metric_list(meta_info, FORWARD_COUNTS_KEY, request_index),
        request_index,
    )
    time_between_block = _validate_time_between_block(
        _extract_metric_list(meta_info, TIME_BETWEEN_BLOCK_KEY, request_index),
        expected_length=len(forward_counts),
        request_index=request_index,
    )
    return DllmObservabilityMetrics(
        forward_counts=forward_counts,
        time_between_block=time_between_block,
    )


def _extract_incremental_observability_metrics_chunk(
    output: dict[str, Any],
    request_index: int | None = None,
) -> DllmObservabilityMetrics:
    meta_info = _extract_meta_info(output, request_index)
    forward_counts = _validate_forward_counts(
        _extract_metric_list(meta_info, FORWARD_COUNTS_KEY, request_index),
        request_index,
    )
    time_between_block = _validate_time_between_block_types(
        _extract_metric_list(meta_info, TIME_BETWEEN_BLOCK_KEY, request_index),
        request_index,
    )
    if len(time_between_block) != len(forward_counts):
        raise AssertionError(
            f"Expected incremental {TIME_BETWEEN_BLOCK_KEY}"
            f"{_request_context(request_index)} to match incremental "
            f"{FORWARD_COUNTS_KEY} length: expected length {len(forward_counts)}, "
            f"got {len(time_between_block)}"
        )
    return DllmObservabilityMetrics(
        forward_counts=forward_counts,
        time_between_block=time_between_block,
    )


def get_observability_metrics_from_generate_non_stream(
    base_url: str,
    prompts: list[str],
) -> dict[int, DllmObservabilityMetrics]:
    response = requests.post(
        f"{base_url}/generate",
        json=_build_generate_payload(
            prompts,
            stream=False,
            stream_interval=None,
        ),
    )
    outputs = response.json()
    if not isinstance(outputs, list):
        outputs = [outputs]
    return {
        output.get("index", index): _extract_observability_metrics(
            output,
            request_index=output.get("index", index),
        )
        for index, output in enumerate(outputs)
    }


def get_observability_metrics_from_generate_stream(
    base_url: str,
    prompts: list[str],
    stream_interval: int | None = None,
    incremental_streaming_output: bool = False,
) -> dict[int, DllmObservabilityMetrics]:
    response = requests.post(
        f"{base_url}/generate",
        json=_build_generate_payload(
            prompts,
            stream=True,
            stream_interval=stream_interval,
        ),
        stream=True,
    )

    chunks_by_index: dict[int, list[dict[str, Any]]] = {}
    incremental_metrics_by_index: dict[int, DllmObservabilityMetrics] = {}
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            break

        chunk = json.loads(line[len("data:") :].strip())
        index = chunk.get("index", 0)
        if incremental_streaming_output:
            chunk_metrics = _extract_incremental_observability_metrics_chunk(
                chunk,
                request_index=index,
            )
            aggregate_metrics = incremental_metrics_by_index.setdefault(
                index,
                DllmObservabilityMetrics(
                    forward_counts=[],
                    time_between_block=[],
                ),
            )
            aggregate_metrics.forward_counts.extend(chunk_metrics.forward_counts)
            aggregate_metrics.time_between_block.extend(
                chunk_metrics.time_between_block
            )
        else:
            chunks_by_index.setdefault(index, []).append(chunk)

    if incremental_streaming_output:
        return {
            index: _extract_observability_metrics(
                {
                    "meta_info": {
                        FORWARD_COUNTS_KEY: metrics.forward_counts,
                        TIME_BETWEEN_BLOCK_KEY: metrics.time_between_block,
                    }
                },
                request_index=index,
            )
            for index, metrics in incremental_metrics_by_index.items()
        }

    return {
        index: _extract_observability_metrics(chunks[-1], request_index=index)
        for index, chunks in chunks_by_index.items()
        if chunks
    }


class DllmObservabilityMixin:
    def assert_generate_stream_cumulative_matches_non_stream(
        self,
        base_url: str,
        prompts: list[str],
        stream_interval: int | None = None,
        incremental_streaming_output: bool = False,
    ) -> None:
        stream_label = "default" if stream_interval is None else str(stream_interval)
        non_stream_metrics_by_index = (
            get_observability_metrics_from_generate_non_stream(base_url, prompts)
        )
        stream_metrics_by_index = get_observability_metrics_from_generate_stream(
            base_url,
            prompts,
            stream_interval,
            incremental_streaming_output=incremental_streaming_output,
        )

        self.assertEqual(
            len(non_stream_metrics_by_index.items()),
            len(stream_metrics_by_index.items()),
            "Number of non-stream and stream outputs should match.",
        )
        self.assertEqual(
            set(non_stream_metrics_by_index),
            set(stream_metrics_by_index),
            "Non-stream and stream request indexes should match.",
        )

        for index in sorted(non_stream_metrics_by_index):
            non_stream_metrics = non_stream_metrics_by_index[index]
            stream_metrics = stream_metrics_by_index[index]
            print(
                f"Comparing forward counts for request {index} (stream_interval={stream_label}, "
                f"incremental_streaming_output={incremental_streaming_output}): "
                f"non-stream={non_stream_metrics.forward_counts}, "
                f"stream={stream_metrics.forward_counts}"
            )
            print(
                f"Demonstrating time between block for request {index} (stream_interval={stream_label}, "
                f"incremental_streaming_output={incremental_streaming_output}): "
                f"non-stream={_format_time_between_block(non_stream_metrics.time_between_block)}, "
                f"stream={_format_time_between_block(stream_metrics.time_between_block)}"
            )

            self.assertEqual(
                non_stream_metrics.forward_counts,
                stream_metrics.forward_counts,
                f"Streamed {FORWARD_COUNTS_KEY} should match non-stream output for "
                f"request {index} (stream_interval={stream_label}, "
                f"incremental_streaming_output={incremental_streaming_output}).",
            )
