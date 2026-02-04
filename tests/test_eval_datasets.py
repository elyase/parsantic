from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evals.runner import load_dataset, run_coercion_case, run_parsing_case  # noqa: E402


def _case_id(case: dict) -> str:
    name = case.get("name")
    return str(name) if name else "<unnamed>"


_DATASETS = _ROOT / "evals" / "datasets"

_PARSING_DATASET = load_dataset(_DATASETS / "parsing_v1.yaml")
_COERCION_DATASET = load_dataset(_DATASETS / "coercion_v1.yaml")


@pytest.mark.parametrize("case", _PARSING_DATASET["cases"], ids=_case_id)
def test_eval_dataset_parsing(case: dict) -> None:
    result = run_parsing_case(case)
    assert result.passed, "\n".join(
        [f"case={result.name}"] + result.errors + [f"output={result.output!r}"]
    )


@pytest.mark.parametrize("case", _COERCION_DATASET["cases"], ids=_case_id)
def test_eval_dataset_coercion(case: dict) -> None:
    result = run_coercion_case(case)
    assert result.passed, "\n".join(
        [f"case={result.name}"] + result.errors + [f"output={result.output!r}"]
    )
