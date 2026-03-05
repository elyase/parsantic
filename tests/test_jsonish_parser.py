from __future__ import annotations

import pytest

from parsantic.jsonish import ParseOptions, parse_jsonish
from parsantic.types import CompletionState


def test_basic_parsing():
    # partial int is incomplete
    assert (
        parse_jsonish("1", options=ParseOptions(), is_done=False).completion
        == CompletionState.INCOMPLETE
    )
    # complete list
    v = parse_jsonish("[1]", options=ParseOptions(), is_done=False)
    assert v.candidates and v.candidates[0].value == [1]
    # incomplete list closing
    v2 = parse_jsonish("[1, 2", options=ParseOptions(), is_done=False)
    assert v2.candidates and any(isinstance(c.value, list) for c in v2.candidates)


# ---- depth_limit enforcement ----


def test_depth_limit():
    with pytest.raises(RecursionError, match="depth limit"):
        parse_jsonish(
            '```json\n{"a": 1}\n```\n', options=ParseOptions(depth_limit=1), is_done=True, _depth=1
        )
    with pytest.raises(ValueError, match="depth_limit must be > 0"):
        parse_jsonish("{}", options=ParseOptions(depth_limit=0), is_done=True)
    # normal depth works
    v = parse_jsonish('{"a": 1}', options=ParseOptions(), is_done=True)
    assert v.candidates and v.candidates[0].value == {"a": 1}


# ---- _close_unclosed_json usage ----


def test_close_unclosed_json_fallback():
    # object
    v = parse_jsonish('{"a": 1, "b": 2', options=ParseOptions(), is_done=False)
    assert any(
        d.get("a") == 1 and d.get("b") == 2
        for d in [c.value for c in v.candidates if isinstance(c.value, dict)]
    )
    # array
    v2 = parse_jsonish("[1, 2, 3", options=ParseOptions(), is_done=False)
    assert any(c.value == [1, 2, 3] for c in v2.candidates)


# ---- trailing commas in nested structures ----


def test_trailing_comma_nested():
    # object
    v = parse_jsonish('{"outer": {"inner": 1,},}', options=ParseOptions(), is_done=True)
    assert any(
        d.get("outer") == {"inner": 1}
        for d in [c.value for c in v.candidates if isinstance(c.value, dict)]
    )
    # array
    v2 = parse_jsonish('{"items": [1, 2, 3,],}', options=ParseOptions(), is_done=True)
    assert any(
        d.get("items") == [1, 2, 3]
        for d in [c.value for c in v2.candidates if isinstance(c.value, dict)]
    )


# ---- multiple consecutive JSON objects ----


def test_multiple_json_in_text():
    # objects
    v = parse_jsonish(
        'First result: {"a": 1} and then {"b": 2} end.', options=ParseOptions(), is_done=True
    )
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("a") == 1 for d in dicts) and any(d.get("b") == 2 for d in dicts)
    # arrays
    v2 = parse_jsonish("Here: [1,2] and also [3,4].", options=ParseOptions(), is_done=True)
    lists = [c.value for c in v2.candidates if isinstance(c.value, list)]
    assert any(item == [1, 2] for item in lists) and any(item == [3, 4] for item in lists)


def test_candidate_truncation_prefers_structured_candidates():
    text = 'Prefix {"a": 1} suffix'
    value = parse_jsonish(text, options=ParseOptions(max_candidates=1), is_done=True)
    assert len(value.candidates) == 1
    assert isinstance(value.candidates[0].value, dict)
