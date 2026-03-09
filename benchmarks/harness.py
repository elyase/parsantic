from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass(slots=True)
class BenchmarkCase:
    document_id: str
    expected: Any
    actual: Any
    latency_s: float
    token_count: int
    api_calls: int


@dataclass(slots=True)
class BenchmarkMetrics:
    exact_accuracy: float
    fuzzy_accuracy: float
    schema_completeness: float
    latency_s: float
    token_count: int
    api_calls: int


def _iter_leaf_values(value: Any) -> Iterator[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            for path, rendered in _iter_leaf_values(child):
                yield (f"/{key}{path}", rendered)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            for path, rendered in _iter_leaf_values(child):
                yield (f"/{index}{path}", rendered)
        return
    yield ("", str(value))


def evaluate_case(case: BenchmarkCase) -> BenchmarkMetrics:
    expected = dict(_iter_leaf_values(case.expected))
    actual = dict(_iter_leaf_values(case.actual))
    shared_paths = set(expected) | set(actual)
    exact_hits = 0
    fuzzy_total = 0.0
    expected_paths = set(expected)
    present_paths = {path for path, value in actual.items() if value != ""}
    for path in shared_paths:
        expected_value = expected.get(path, "")
        actual_value = actual.get(path, "")
        if expected_value == actual_value and expected_value != "":
            exact_hits += 1
        ratio = SequenceMatcher(None, expected_value, actual_value).ratio()
        if expected_value != "" and actual_value != "":
            ratio = max(ratio, 0.5)
        fuzzy_total += ratio
    denominator = max(len(shared_paths), 1)
    completeness = len(expected_paths & present_paths) / max(len(expected_paths), 1)
    return BenchmarkMetrics(
        exact_accuracy=exact_hits / denominator,
        fuzzy_accuracy=fuzzy_total / denominator,
        schema_completeness=completeness,
        latency_s=case.latency_s,
        token_count=case.token_count,
        api_calls=case.api_calls,
    )
