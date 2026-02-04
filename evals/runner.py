from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import StrictBool, StrictFloat, StrictInt

from parsantic.api import parse
from parsantic.coerce import CoerceOptions
from parsantic.jsonish import ParseOptions, parse_jsonish
from parsantic.types import CompletionState


class EvalColor(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class EvalDessert(enum.Enum):
    ICECREAM = "icecream"
    CAKE = "cake"


class EvalAccented(enum.Enum):
    CAFE = "caf\u00e9"


_TYPE_REGISTRY: dict[str, Any] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "StrictInt": StrictInt,
    "StrictFloat": StrictFloat,
    "StrictBool": StrictBool,
    "EvalColor": EvalColor,
    "EvalDessert": EvalDessert,
    "EvalAccented": EvalAccented,
}

_LIST_RE = re.compile(r"^list\[(.+)\]$")
_DICT_RE = re.compile(r"^dict\[(.+),(.+)\]$")


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"Invalid dataset format: {dataset_path}")
    return data


def resolve_target(spec: str) -> Any:
    spec = spec.strip()
    if spec in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[spec]

    m_list = _LIST_RE.match(spec)
    if m_list:
        inner = resolve_target(m_list.group(1))
        return list[inner]

    m_dict = _DICT_RE.match(spec)
    if m_dict:
        k = resolve_target(m_dict.group(1).strip())
        v = resolve_target(m_dict.group(2).strip())
        return dict[k, v]

    raise KeyError(f"Unknown target type spec: {spec!r}")


def _parse_evaluator(item: Any) -> tuple[str, Any]:
    if isinstance(item, str):
        return item, True
    if isinstance(item, dict) and len(item) == 1:
        ((k, v),) = item.items()
        return str(k), v
    raise ValueError(f"Invalid evaluator: {item!r}")


def _safe_equals(actual: Any, expected: Any) -> bool:
    # Avoid common Python footgun where True == 1.
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int) and isinstance(actual, bool):
        return False
    return actual == expected


def _coercion_value_for_compare(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value if isinstance(value.value, str) else value.name
    return value


def _preview(value: Any, *, limit: int = 200) -> str:
    try:
        s = repr(value)
    except Exception:
        s = f"<unrepr {type(value).__name__}>"
    if len(s) > limit:
        return s[:limit] + "…"
    return s


@dataclass(slots=True)
class EvalResult:
    name: str
    passed: bool
    errors: list[str]
    output: Any = None


def run_parsing_case(case: dict[str, Any]) -> EvalResult:
    name = str(case.get("name") or "<unnamed>")
    inputs = case.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise ValueError(f"Case {name!r} has invalid inputs")

    text = inputs.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Case {name!r} inputs.text must be a string")
    is_done = bool(inputs.get("is_done", True))

    parse_options = ParseOptions(**(inputs.get("parse_options") or {}))
    actual = parse_jsonish(text, options=parse_options, is_done=is_done)
    expected = case.get("expected_output")

    errors: list[str] = []
    for ev in case.get("evaluators") or []:
        ev_name, ev_params = _parse_evaluator(ev)

        if ev_name == "HasCandidates":
            want = bool(ev_params)
            got = bool(actual.candidates)
            if got != want:
                errors.append(f"HasCandidates: expected {want}, got {got}")

        elif ev_name == "MatchesExpected":
            if not ev_params:
                continue
            if "expected_output" not in case:
                errors.append("MatchesExpected: missing expected_output")
                continue
            if actual.candidates:
                if not any(_safe_equals(c.value, expected) for c in actual.candidates):
                    errors.append(
                        "MatchesExpected: expected output not found in candidates "
                        f"(expected={_preview(expected)}, "
                        f"candidates={[_preview(c.value) for c in actual.candidates[:8]]})"
                    )
            else:
                if not _safe_equals(actual.value, expected):
                    errors.append(
                        "MatchesExpected: expected output did not match parsed value "
                        f"(expected={_preview(expected)}, actual={_preview(actual.value)})"
                    )

        elif ev_name == "CompletionIs":
            want = str(ev_params).upper()
            if want not in {"COMPLETE", "INCOMPLETE"}:
                errors.append(f"CompletionIs: invalid expected state {ev_params!r}")
                continue
            got = actual.completion
            expected_state = (
                CompletionState.COMPLETE if want == "COMPLETE" else CompletionState.INCOMPLETE
            )
            if got != expected_state:
                errors.append(f"CompletionIs: expected {want}, got {got.name}")

        elif ev_name == "CandidateCountAtLeast":
            try:
                want = int(ev_params)
            except Exception:
                errors.append(f"CandidateCountAtLeast: invalid value {ev_params!r}")
                continue
            got = len(actual.candidates)
            if got < want:
                errors.append(f"CandidateCountAtLeast: expected >= {want}, got {got}")

        else:
            errors.append(f"Unknown evaluator: {ev_name}")

    output = {
        "completion": actual.completion.name,
        "candidate_count": len(actual.candidates),
        "candidate_previews": [c.value for c in actual.candidates[:8]],
    }
    return EvalResult(name=name, passed=not errors, errors=errors, output=output)


def run_coercion_case(case: dict[str, Any]) -> EvalResult:
    name = str(case.get("name") or "<unnamed>")
    inputs = case.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise ValueError(f"Case {name!r} has invalid inputs")

    text = inputs.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Case {name!r} inputs.text must be a string")
    target_spec = inputs.get("target")
    if not isinstance(target_spec, str):
        raise ValueError(f"Case {name!r} inputs.target must be a string")
    is_done = bool(inputs.get("is_done", True))

    parse_options = ParseOptions(**(inputs.get("parse_options") or {}))
    coerce_options = CoerceOptions(**(inputs.get("coerce_options") or {}))

    target = resolve_target(target_spec)
    result = parse(
        text,
        target,
        is_done=is_done,
        parse_options=parse_options,
        coerce_options=coerce_options,
    )

    expected = case.get("expected_output")

    errors: list[str] = []
    for ev in case.get("evaluators") or []:
        ev_name, ev_params = _parse_evaluator(ev)

        if ev_name == "MatchesExpected":
            if not ev_params:
                continue
            if "expected_output" not in case:
                errors.append("MatchesExpected: missing expected_output")
                continue
            actual_value = _coercion_value_for_compare(result.value)
            if not _safe_equals(actual_value, expected):
                errors.append(
                    "MatchesExpected: output mismatch "
                    f"(expected={_preview(expected)}, actual={_preview(actual_value)})"
                )

        elif ev_name == "FlagsInclude":
            want = ev_params or []
            if isinstance(want, str):
                want = [want]
            if not isinstance(want, list) or not all(isinstance(x, str) for x in want):
                errors.append(f"FlagsInclude: invalid value {ev_params!r}")
                continue
            missing = [f for f in want if f not in result.flags]
            if missing:
                errors.append(f"FlagsInclude: missing {missing!r} (flags={result.flags!r})")

        elif ev_name == "FlagsExclude":
            want = ev_params or []
            if isinstance(want, str):
                want = [want]
            if not isinstance(want, list) or not all(isinstance(x, str) for x in want):
                errors.append(f"FlagsExclude: invalid value {ev_params!r}")
                continue
            present = [f for f in want if f in result.flags]
            if present:
                errors.append(
                    f"FlagsExclude: unexpectedly present {present!r} (flags={result.flags!r})"
                )

        elif ev_name == "ScoreMax":
            try:
                want = int(ev_params)
            except Exception:
                errors.append(f"ScoreMax: invalid value {ev_params!r}")
                continue
            if result.score > want:
                errors.append(f"ScoreMax: expected <= {want}, got {result.score}")

        else:
            errors.append(f"Unknown evaluator: {ev_name}")

    output = {"value": result.value, "flags": result.flags, "score": result.score}
    return EvalResult(name=name, passed=not errors, errors=errors, output=output)


def run_dataset(path: str | Path, *, task: str) -> list[EvalResult]:
    dataset = load_dataset(path)
    cases = dataset["cases"]
    out: list[EvalResult] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"Invalid case: {case!r}")
        if task == "parsing":
            out.append(run_parsing_case(case))
        elif task == "coercion":
            out.append(run_coercion_case(case))
        else:
            raise ValueError(f"Unknown task: {task!r}")
    return out
