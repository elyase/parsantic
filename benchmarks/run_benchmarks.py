from __future__ import annotations

import json
from pathlib import Path

from .harness import BenchmarkCase, evaluate_case


def main(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    for item in manifest["cases"]:
        metrics = evaluate_case(
            BenchmarkCase(
                document_id=item["document_id"],
                expected=item["expected"],
                actual=item["actual"],
                latency_s=item["latency_s"],
                token_count=item["token_count"],
                api_calls=item["api_calls"],
            )
        )
        print(
            json.dumps(
                {
                    "document_id": item["document_id"],
                    "exact_accuracy": metrics.exact_accuracy,
                    "fuzzy_accuracy": metrics.fuzzy_accuracy,
                    "schema_completeness": metrics.schema_completeness,
                    "latency_s": metrics.latency_s,
                    "token_count": metrics.token_count,
                    "api_calls": metrics.api_calls,
                }
            )
        )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1]))
