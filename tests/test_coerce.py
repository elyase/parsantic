"""Consolidated coercion and parse-level behavior tests.

Covers: primitive coercion, float parsing, key normalization, anyof handling,
implied keys, multi-object to list, and streaming.
"""

from __future__ import annotations

from pydantic import BaseModel, TypeAdapter

from parsantic.api import parse, parse_stream
from parsantic.coerce import (
    _FLAG_WEIGHTS,
    CoerceOptions,
    _float_from_comma_separated,
    coerce_jsonish_to_python,
    normalize_key,
)
from parsantic.jsonish import JsonishValue
from parsantic.types import CompletionState

# ===========================================================================
# Primitive coercion (from test_coerce_primitives.py)
# ===========================================================================


def test_string_to_int():
    res = parse('"123"', int, is_done=True)
    assert res.value == 123


def test_fraction_to_float():
    res = parse('"1/2"', float, is_done=True)
    assert abs(res.value - 0.5) < 1e-9


class _Obj(BaseModel):
    c: float
    d: float


def test_fraction_and_currency_to_float_in_model_fields():
    res = parse('{c: 1/2, d: "$1,234.56"}', _Obj, is_done=True)
    assert abs(res.value.c - 0.5) < 1e-9
    assert abs(res.value.d - 1234.56) < 1e-9


def test_single_to_array():
    res = parse("1", list[int], is_done=True)
    assert res.value == [1]


def test_optional_accepts_null():
    res = parse("null", int | None, is_done=True)
    assert res.value is None


# ===========================================================================
# Float from comma-separated (from test_float_from_comma_separated.py)
# ===========================================================================


def test_float_from_comma_separated_ported_from_baml():
    test_cases: list[tuple[str, float | None]] = [
        ("3,14", 314.0),
        ("1.234,56", None),
        ("1.234.567,89", None),
        ("€1.234,56", None),
        ("-€1.234,56", None),
        ("€1.234", 1.234),
        ("1.234€", 1.234),
        ("€1.234,56", None),
        ("€1,234.56", 1234.56),
        ("3,000", 3000.0),
        ("3,100.00", 3100.00),
        ("1,234.56", 1234.56),
        ("1,234,567.89", 1234567.89),
        ("$1,234.56", 1234.56),
        ("-$1,234.56", -1234.56),
        ("$1,234", 1234.0),
        ("1,234$", 1234.0),
        ("$1,234.56", 1234.56),
        ("+$1,234.56", 1234.56),
        ("-$1,234.56", -1234.56),
        ("$9,999,999,999", 9999999999.0),
        ("$1.23.456", None),
        ("$1.234.567.890", None),
        ("$1,234", 1234.0),
        ("$314", 314.0),
        ("$1,23,456", 123456.0),
        ("50%", 50.0),
        ("3.15%", 3.15),
        (".009%", 0.009),
        ("1.234,56%", None),
        ("$1,234.56%", 1234.56),
        ("The answer is 10,000", 10000.0),
        ("The total is €1.234,56 today", None),
        ("You owe $3,000 for the service", 3000.0),
        ("Save up to 20% on your purchase", 20.0),
        ("Revenue grew by 1,234.56 this quarter", 1234.56),
        ("Profit is -€1.234,56 in the last month", None),
        ("The answer is 10,000 and $3,000", None),
        ("We earned €1.234,56 and $2,345.67 this year", None),
        ("Increase of 5% and a profit of $1,000", None),
        ("Loss of -€500 and a gain of 1,200.50", None),
        ("Targets: 2,000 units and €3.000,75 revenue", None),
        ("12,111,123.", 12111123.0),
        ("12,111,123,", 12111123.0),
    ]
    for inp, expected in test_cases:
        got = _float_from_comma_separated(inp)
        assert got == expected, f"failed to parse {inp!r}: expected {expected}, got {got}"


# ===========================================================================
# Key normalization (from test_normalize_key.py)
# ===========================================================================


def test_remove_accents_like_baml():
    assert normalize_key("étude") == "etude"
    assert normalize_key("français") == "francais"
    assert normalize_key("Español") == "espanol"
    assert normalize_key("português") == "portugues"
    assert normalize_key("médium") == "medium"
    assert normalize_key("Grün") == "grun"
    assert normalize_key("Über") == "uber"
    assert normalize_key("Straße") == "strasse"
    assert normalize_key("Stadt") == "stadt"


def test_ligatures_like_baml():
    assert normalize_key("æ") == "ae"
    assert normalize_key("Æ") == "ae"
    assert normalize_key("ø") == "o"
    assert normalize_key("Ø") == "o"
    assert normalize_key("œ") == "oe"
    assert normalize_key("Œ") == "oe"
    assert normalize_key("København") == "kobenhavn"
    assert normalize_key("cœur") == "coeur"
    assert normalize_key("œuvre") == "oeuvre"
    assert normalize_key("Straße ældre øl œuvre") == "strasseaeldreoloeuvre"


# ===========================================================================
# AnyOf string handling (from test_anyof_string.py)
# ===========================================================================


def test_anyof_prefers_string_variant_like_baml():
    raw = "[json\nAnyOf[{,AnyOf[{,{},],]]"
    anyof = JsonishValue(
        value=raw,
        raw=raw,
        completion=CompletionState.INCOMPLETE,
        candidates=(
            JsonishValue(value="[json\n", raw=raw, completion=CompletionState.INCOMPLETE),
            JsonishValue(value={}, raw=raw, completion=CompletionState.INCOMPLETE),
        ),
    )
    res = coerce_jsonish_to_python(anyof, TypeAdapter(str), options=CoerceOptions())
    assert res.value == "[json\n"


def test_anyof_without_string_variant_falls_back_to_raw():
    raw = "some raw input"
    anyof = JsonishValue(
        value=raw,
        raw=raw,
        completion=CompletionState.INCOMPLETE,
        candidates=(
            JsonishValue(value={}, raw=raw, completion=CompletionState.INCOMPLETE),
            JsonishValue(value=[], raw=raw, completion=CompletionState.INCOMPLETE),
        ),
    )
    res = coerce_jsonish_to_python(anyof, TypeAdapter(str), options=CoerceOptions())
    assert res.value == raw


# ===========================================================================
# Model key normalization (from test_model_key_normalization.py)
# ===========================================================================


class _PersonHair(BaseModel):
    hair_color: str


def test_model_key_normalization():
    res = parse('{"hair color": "Grey"}', _PersonHair, is_done=True)
    assert res.value.hair_color == "Grey"


def test_streaming_partial_key_normalization():
    sp = parse_stream(_PersonHair)
    sp.feed('{"hair color": "Gr')
    partial = sp.parse_partial()
    assert hasattr(partial.value, "hair_color")
    assert partial.value.hair_color == "Gr"


# ===========================================================================
# Implied key wrapping (from test_implied_key.py)
# ===========================================================================


class _WrapperDict(BaseModel):
    data: dict[str, int]


def test_implied_key_single_field_model_from_object():
    res = parse('{"a": 1}', _WrapperDict, is_done=True)
    assert res.value.data == {"a": 1}


class _WrapperScalar(BaseModel):
    n: int


def test_implied_key_single_field_model_from_scalar():
    res = parse("123", _WrapperScalar, is_done=True)
    assert res.value.n == 123


# ===========================================================================
# Multi-object to list (from test_multi_object_to_list.py)
# ===========================================================================


class _Item(BaseModel):
    a: int


def test_multiple_markdown_codeblocks_can_be_list():
    text = """```json
{"a": 1}
```

```json
{"a": 2}
```"""
    res = parse(text, list[_Item], is_done=True)
    assert [x.a for x in res.value] == [1, 2]


def test_multiple_objects_in_free_text_can_be_list():
    text = 'prefix {"a": 1} middle {"a": 2} suffix'
    res = parse(text, list[_Item], is_done=True)
    assert [x.a for x in res.value] == [1, 2]


# ===========================================================================
# Streaming (from test_streaming.py)
# ===========================================================================


class _StreamObj(BaseModel):
    a: int
    b: int


def test_streaming_partial_then_finish():
    sp = parse_stream(_StreamObj)
    sp.feed('{"a": 1,')
    partial = sp.parse_partial()
    assert hasattr(partial.value, "a")
    assert partial.value.a == 1
    assert hasattr(partial.value, "b")
    assert partial.value.b is None
    sp.feed('"b": 2}')
    final = sp.finish()
    assert final.value.a == 1
    assert final.value.b == 2


def test_stream_parser_max_buffer_chars_limits_growth():
    sp = parse_stream(_StreamObj, max_buffer_chars=5)
    sp.feed("0123456789")
    assert sp.buffer == "56789"


def test_all_emitted_flags_have_explicit_weights():
    emitted_flags = {
        "fixed_json",
        "inferred_array",
        "markdown",
        "markdown_array",
        "markdown_tail",
        "closed_unclosed",
        "grepped_json",
        "grepped_array",
        "fixed_array",
        "as_string",
        "single_to_array",
        "object_to_string",
        "string_to_int",
        "string_to_float",
        "string_to_bool",
        "float_to_int",
        "default_from_missing",
        "extra_key",
        "substring_match",
        "strip_punct",
        "case_insensitive",
        "accent_insensitive",
        "key_normalized",
        "implied_key",
        "partial_model",
        "partial_unvalidated",
        "ambiguous_key",
        "ambiguous_key_kept",
        "ambiguous_enum",
        "key_collision",
        "max_depth_exceeded",
    }
    missing = sorted(emitted_flags - set(_FLAG_WEIGHTS))
    assert not missing, f"Missing flag weights: {missing}"
