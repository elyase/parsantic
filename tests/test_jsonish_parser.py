from __future__ import annotations

import pytest

from parsantic.jsonish import ParseOptions, parse_jsonish
from parsantic.types import CompletionState


def test_partial_int_is_incomplete_like_baml():
    # Mirrors Rust test `test_partial_int` in `entry.rs`.
    v = parse_jsonish("1", options=ParseOptions(), is_done=False)
    assert v.completion == CompletionState.INCOMPLETE


def test_complete_list():
    v = parse_jsonish("[1]", options=ParseOptions(), is_done=False)
    # strict JSON parse yields COMPLETE array
    assert v.candidates
    assert v.candidates[0].value == [1]


def test_incomplete_list_fixing_parser_closes():
    v = parse_jsonish("[1, 2", options=ParseOptions(), is_done=False)
    assert v.candidates
    # Expect a candidate that became a list
    assert any(isinstance(c.value, list) for c in v.candidates)


# ---- depth_limit enforcement ----


def test_depth_limit_raises_on_deep_nesting():
    """When depth_limit is very low, deeply nested markdown parsing should raise."""
    # A markdown block inside a markdown block should exceed depth_limit=1.
    text = """```json
{"a": 1}
```
"""
    with pytest.raises(RecursionError, match="depth limit"):
        parse_jsonish(text, options=ParseOptions(depth_limit=1), is_done=True, _depth=1)


def test_depth_limit_zero_raises_value_error():
    with pytest.raises(ValueError, match="depth_limit must be > 0"):
        parse_jsonish("{}", options=ParseOptions(depth_limit=0), is_done=True)


def test_depth_limit_normal_does_not_raise():
    """Default depth_limit=100 should be fine for normal input."""
    v = parse_jsonish('{"a": 1}', options=ParseOptions(), is_done=True)
    assert v.candidates
    assert v.candidates[0].value == {"a": 1}


# ---- _close_unclosed_json usage ----


def test_close_unclosed_json_fallback_object():
    """When all other strategies fail, _close_unclosed_json should close brackets."""
    # Use options that disable markdown and find-all so the fallback is needed.
    text = '{"a": 1, "b": 2'
    v = parse_jsonish(text, options=ParseOptions(), is_done=False)
    assert v.candidates
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("a") == 1 and d.get("b") == 2 for d in dicts)


def test_close_unclosed_json_fallback_array():
    """_close_unclosed_json should handle unclosed arrays too."""
    text = "[1, 2, 3"
    v = parse_jsonish(text, options=ParseOptions(), is_done=False)
    assert v.candidates
    assert any(c.value == [1, 2, 3] for c in v.candidates)


# ---- trailing commas in nested structures ----


def test_trailing_comma_nested_object():
    """Trailing commas inside nested objects should be handled."""
    text = '{"outer": {"inner": 1,},}'
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("outer") == {"inner": 1} for d in dicts)


def test_trailing_comma_nested_array():
    """Trailing commas inside nested arrays should be handled."""
    text = '{"items": [1, 2, 3,],}'
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("items") == [1, 2, 3] for d in dicts)


# ---- multiple consecutive JSON objects ----


def test_multiple_json_objects_in_text():
    """Multiple JSON objects separated by non-JSON text should each be found."""
    text = 'First result: {"a": 1} and then {"b": 2} end.'
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("a") == 1 for d in dicts)
    assert any(d.get("b") == 2 for d in dicts)


def test_multiple_json_arrays_in_text():
    """Multiple JSON arrays separated by text should be found."""
    text = "Here: [1,2] and also [3,4]."
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    lists = [c.value for c in v.candidates if isinstance(c.value, list)]
    assert any(item == [1, 2] for item in lists)
    assert any(item == [3, 4] for item in lists)
