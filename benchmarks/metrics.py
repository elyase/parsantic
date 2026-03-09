from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


def _pointer_escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _flatten(value: Any, *, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {path or "/": {}}
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child_path = (
                f"{path}/{_pointer_escape(str(key))}" if path else f"/{_pointer_escape(str(key))}"
            )
            flattened.update(_flatten(value[key], path=child_path))
        return flattened
    if isinstance(value, list):
        if not value:
            return {path or "/": []}
        flattened: dict[str, Any] = {}
        for index, item in enumerate(value):
            child_path = f"{path}/{index}" if path else f"/{index}"
            flattened.update(_flatten(item, path=child_path))
        return flattened
    return {path or "/": value}


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value}"
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _fuzzy_ratio(expected: Any, actual: Any) -> float:
    return SequenceMatcher(
        None,
        _normalize_scalar(expected),
        _normalize_scalar(actual),
    ).ratio()


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    exact_accuracy: float
    fuzzy_accuracy: float
    completeness: float
    wrong_present_rate: float
    abstention_rate: float
    selective_accuracy: float
    exact_matches: int
    fuzzy_matches: int
    expected_fields: int
    predicted_fields: int
    provenance_accuracy: float = 0.0
    page_coverage: float = 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    exact_accuracy: float
    fuzzy_accuracy: float
    completeness: float
    wrong_present_rate: float
    abstention_rate: float
    selective_accuracy: float
    provenance_accuracy: float
    page_coverage: float
    latency_s: float
    api_calls: int
    token_count: int
    estimated_cost_usd: float
    cases: list[CaseMetrics] = field(default_factory=list)


def score_case(expected: Any, actual: Any, *, fuzzy_threshold: float = 0.9) -> CaseMetrics:
    expected_flat = _flatten(expected)
    actual_flat = _flatten(actual)
    expected_fields = len(expected_flat)
    if expected_fields == 0:
        return CaseMetrics(
            exact_accuracy=1.0,
            fuzzy_accuracy=1.0,
            completeness=1.0,
            wrong_present_rate=0.0,
            abstention_rate=0.0,
            selective_accuracy=1.0,
            provenance_accuracy=0.0,
            page_coverage=0.0,
            exact_matches=0,
            fuzzy_matches=0,
            expected_fields=0,
            predicted_fields=len(actual_flat),
        )

    exact_matches = 0
    fuzzy_matches = 0
    present_matches = 0
    wrong_present = 0
    abstentions = 0
    for path, expected_value in expected_flat.items():
        if path not in actual_flat:
            abstentions += 1
            continue
        present_matches += 1
        actual_value = actual_flat[path]
        if actual_value is None:
            abstentions += 1
            continue
        if expected_value == actual_value:
            exact_matches += 1
            fuzzy_matches += 1
            continue
        if _fuzzy_ratio(expected_value, actual_value) >= fuzzy_threshold:
            fuzzy_matches += 1
        else:
            wrong_present += 1
    for path, actual_value in actual_flat.items():
        if actual_value is None:
            continue
        if path not in expected_flat:
            wrong_present += 1

    selective_denominator = exact_matches + wrong_present

    return CaseMetrics(
        exact_accuracy=exact_matches / expected_fields,
        fuzzy_accuracy=fuzzy_matches / expected_fields,
        completeness=present_matches / expected_fields,
        wrong_present_rate=wrong_present / max(len(actual_flat), 1),
        abstention_rate=abstentions / expected_fields,
        selective_accuracy=(
            exact_matches / selective_denominator if selective_denominator else 1.0
        ),
        provenance_accuracy=0.0,
        page_coverage=0.0,
        exact_matches=exact_matches,
        fuzzy_matches=fuzzy_matches,
        expected_fields=expected_fields,
        predicted_fields=len(actual_flat),
    )


def score_provenance_case(
    expected_sources: dict[str, dict[str, Any]],
    actual_sources: dict[str, Any],
) -> tuple[float, float]:
    if not expected_sources:
        return 0.0, 0.0
    exact_matches = 0
    page_expected = 0
    page_present = 0
    for path, expected in expected_sources.items():
        actual = actual_sources.get(path)
        expected_scope = expected.get("scope", "document")
        expected_pages = tuple(expected.get("pages", ()))
        if actual is not None:
            actual_scope = getattr(actual, "scope", None)
            actual_pages = tuple(getattr(actual, "pages", ()))
            if actual_scope == expected_scope and actual_pages == expected_pages:
                exact_matches += 1
            if expected_scope == "page":
                page_expected += 1
                if actual_scope == "page":
                    page_present += 1
        elif expected_scope == "page":
            page_expected += 1
    total = len(expected_sources)
    provenance_accuracy = exact_matches / total if total else 0.0
    page_coverage = page_present / page_expected if page_expected else 0.0
    return provenance_accuracy, page_coverage


def summarize_cases(
    cases: list[CaseMetrics],
    *,
    latency_s: float,
    api_calls: int,
    token_count: int,
    estimated_cost_usd: float,
) -> BenchmarkMetrics:
    if not cases:
        return BenchmarkMetrics(
            exact_accuracy=0.0,
            fuzzy_accuracy=0.0,
            completeness=0.0,
            wrong_present_rate=0.0,
            abstention_rate=0.0,
            selective_accuracy=0.0,
            provenance_accuracy=0.0,
            page_coverage=0.0,
            latency_s=latency_s,
            api_calls=api_calls,
            token_count=token_count,
            estimated_cost_usd=estimated_cost_usd,
            cases=[],
        )
    count = len(cases)
    return BenchmarkMetrics(
        exact_accuracy=sum(case.exact_accuracy for case in cases) / count,
        fuzzy_accuracy=sum(case.fuzzy_accuracy for case in cases) / count,
        completeness=sum(case.completeness for case in cases) / count,
        wrong_present_rate=sum(case.wrong_present_rate for case in cases) / count,
        abstention_rate=sum(case.abstention_rate for case in cases) / count,
        selective_accuracy=sum(case.selective_accuracy for case in cases) / count,
        provenance_accuracy=sum(case.provenance_accuracy for case in cases) / count,
        page_coverage=sum(case.page_coverage for case in cases) / count,
        latency_s=latency_s,
        api_calls=api_calls,
        token_count=token_count,
        estimated_cost_usd=estimated_cost_usd,
        cases=cases,
    )
