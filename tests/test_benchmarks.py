from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.harness import BenchmarkCase, evaluate_case


def test_benchmark_metrics_capture_accuracy_completeness_and_cost():
    metrics = evaluate_case(
        BenchmarkCase(
            document_id="onc-1",
            expected={"patient": {"name": "Ada"}, "line_items": [{"code": "A"}]},
            actual={"patient": {"name": "Ada"}, "line_items": [{"code": "B"}]},
            latency_s=1.25,
            token_count=321,
            api_calls=2,
        )
    )

    assert metrics.exact_accuracy == 0.5
    assert 0.5 <= metrics.fuzzy_accuracy < 1.0
    assert metrics.schema_completeness == 1.0
    assert metrics.latency_s == 1.25
    assert metrics.token_count == 321
    assert metrics.api_calls == 2
