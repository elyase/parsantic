"""Advanced coercion tests covering B1-B6 improvements."""

from __future__ import annotations

import enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from parsantic.api import ParseResult, coerce, coerce_debug, parse, parse_debug
from parsantic.coerce import (
    CoerceOptions,
    _coerce_to_type,
    _match_enum_value,
    _try_int,
)
from parsantic.types import CandidateDebug, ParseDebug

# ---------------------------------------------------------------------------
# B1: Enum/Literal matching
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class AccentedEnum(enum.Enum):
    CAFE = "caf\u00e9"
    NAIVE = "na\u00efve"
    RESUME = "r\u00e9sum\u00e9"


@pytest.mark.parametrize(
    "input_val,candidates,opts_kwargs,expected_match,expected_flags",
    [
        # exact
        ("red", ["red", "green", "blue"], {}, "red", ()),
        # no match
        ("purple", ["red", "green", "blue"], {}, None, None),
        # case insensitive
        ("RED", ["red", "green", "blue"], {}, "red", ("case_insensitive",)),
        ("Green", ["red", "green", "blue"], {}, "green", ("case_insensitive",)),
        # strip punct
        ("ice-cream", ["icecream", "cake", "pie"], {}, "icecream", ("strip_punct",)),
        ("U.S.A.", ["USA", "UK", "Canada"], {}, "USA", ("strip_punct",)),
        # accent insensitive
        (
            "cafe",
            ["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9"],
            {},
            "caf\u00e9",
            ("accent_insensitive",),
        ),
        ("caf\u00e9", ["cafe", "naive", "resume"], {}, "cafe", ("accent_insensitive",)),
        # substring disabled
        ("re", ["red", "green", "blue"], {}, None, None),
    ],
    ids=[
        "exact",
        "no-match",
        "case-RED",
        "case-Green",
        "punct-dash",
        "punct-dots",
        "accent-cafe",
        "accent-reverse",
        "substring-disabled",
    ],
)
def test_enum_matching(input_val, candidates, opts_kwargs, expected_match, expected_flags):
    opts = CoerceOptions(**opts_kwargs)
    matched, flags = _match_enum_value(input_val, candidates, opts)
    assert matched == expected_match
    if expected_flags is not None:
        assert flags == expected_flags


def test_enum_substring_enabled():
    opts = CoerceOptions(allow_substring_enum_match=True)
    matched, flags = _match_enum_value("re", ["red", "green", "blue"], opts)
    assert matched is not None
    assert "substring_match" in flags


def test_enum_substring_single_match():
    opts = CoerceOptions(allow_substring_enum_match=True)
    matched, flags = _match_enum_value("blu", ["red", "green", "blue"], opts)
    assert matched == "blue"
    assert flags == ("substring_match",)


def test_enum_ambiguous_picks_alphabetically():
    matched, flags = _match_enum_value("AB", ["Ab", "aB"], CoerceOptions())
    assert matched == "Ab"
    assert "ambiguous_enum" in flags
    assert "case_insensitive" in flags


@pytest.mark.parametrize(
    "input_str,target,expected_val,expected_flag",
    [
        ('"RED"', Color, Color.RED, "case_insensitive"),
        ('"red"', Color, Color.RED, None),
        ('"cafe"', AccentedEnum, AccentedEnum.CAFE, "accent_insensitive"),
        ('"hello"', Literal["hello", "world"], "hello", None),
        ('"HELLO"', Literal["hello", "world"], "hello", "case_insensitive"),
        ('"ice-cream"', Literal["icecream", "cake"], "icecream", "strip_punct"),
        ('"cafe"', Literal["caf\u00e9", "tea"], "caf\u00e9", "accent_insensitive"),
    ],
    ids=[
        "enum-case",
        "enum-exact",
        "enum-accent",
        "literal-exact",
        "literal-case",
        "literal-punct",
        "literal-accent",
    ],
)
def test_enum_literal_via_parse(input_str, target, expected_val, expected_flag):
    result = parse(input_str, target, is_done=True)
    assert result.value == expected_val
    if expected_flag:
        assert expected_flag in result.flags


# ---------------------------------------------------------------------------
# B2: Recursive coercion (list, dict, union, tuple)
# ---------------------------------------------------------------------------


class TestRecursiveCoercion:
    def test_list_coercion(self):
        opts = CoerceOptions()
        assert _coerce_to_type(["1", "2", "3"], list[int], opts).value == [1, 2, 3]
        assert _coerce_to_type(["1.5", "2.5"], list[float], opts).value == [1.5, 2.5]
        assert coerce(["1", "2", "3"], list[int]).value == [1, 2, 3]

    def test_dict_coercion(self):
        opts = CoerceOptions()
        assert _coerce_to_type({"a": "1.5", "b": "2.5"}, dict[str, float], opts).value == {
            "a": 1.5,
            "b": 2.5,
        }
        assert _coerce_to_type({"1": "hello", "2": "world"}, dict[int, str], opts).value == {
            1: "hello",
            2: "world",
        }

    def test_tuple_coercion(self):
        opts = CoerceOptions()
        assert _coerce_to_type(["1", "2", "3"], tuple[int, ...], opts).value == (1, 2, 3)
        assert _coerce_to_type(["1", "hello"], tuple[int, str], opts).value == (1, "hello")


class TestUnionCoercion:
    def test_union_coercion(self):
        opts = CoerceOptions()
        # Basic union
        sv = _coerce_to_type("42", int | str, opts)
        assert sv.value == 42 or sv.value == "42"
        # None variant
        assert _coerce_to_type(None, int | None, opts).value is None
        # PEP 604 with recursive branch
        target = dict[str, Literal["red", "blue"]] | list[int]
        sv2 = _coerce_to_type({"favorite": "RED"}, target, opts)
        assert sv2.value == {"favorite": "red"}
        assert "case_insensitive" in sv2.flags


class TestNestedModelCoercion:
    def test_nested_model_in_list(self):
        class Inner(BaseModel):
            x: int

        opts = CoerceOptions()
        sv = _coerce_to_type([{"x": "1"}, {"x": "2"}], list[Inner], opts)
        assert len(sv.value) == 2
        assert sv.value[0].x == 1
        assert sv.value[1].x == 2


# ---------------------------------------------------------------------------
# B3: Safe int coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("123", 123),
        ("3.0", 3),
        ("3.00", 3),
        ("1.4", None),
        ("1.5", None),
        ("-3.0", -3),
        ("-1.7", None),
    ],
    ids=[
        "int-str",
        "float-exact",
        "float-trailing-zeros",
        "float-not-int",
        "float-half",
        "neg-exact",
        "neg-not-int",
    ],
)
def test_try_int(input_str, expected):
    assert _try_int(input_str) == expected


class TestSafeIntCoercionViaParseAPI:
    def test_safe_int_via_parse(self):
        result = parse('"3.0"', int, is_done=True)
        assert result.value == 3


# ---------------------------------------------------------------------------
# B4: Key mapping with aliases and validation_alias
# ---------------------------------------------------------------------------


class TestKeyMappingAliases:
    def test_field_and_validation_aliases(self):
        class M1(BaseModel):
            my_field: str = Field(alias="myField")

        assert coerce({"myField": "hello"}, M1).value.my_field == "hello"

        class M2(BaseModel):
            my_field: str = Field(validation_alias="my-field")

        assert coerce({"my-field": "hello"}, M2).value.my_field == "hello"

        class M3(BaseModel):
            my_field: str = Field(validation_alias="inputField")

        assert coerce({"inputField": "hello"}, M3).value.my_field == "hello"


class TestExtraAllowModel:
    def test_extra_handling(self):
        class MAllow(BaseModel):
            model_config = {"extra": "allow"}
            name: str

        result = coerce({"name": "Alice", "age": 30}, MAllow)
        assert result.value.name == "Alice"
        assert result.value.age == 30  # type: ignore[attr-defined]

        class MDefault(BaseModel):
            name: str

        result2 = coerce({"name": "Alice", "age": 30}, MDefault)
        assert result2.value.name == "Alice"


class TestKeyCollision:
    def test_key_collision_first_wins(self):
        class M(BaseModel):
            model_config = {"populate_by_name": True}
            my_name: str = Field(alias="MyName")

        # Both "My Name" and "MY_NAME" normalize to "myname" which maps
        # to the field "my_name". First match wins.
        result = coerce({"My Name": "Alice", "MY_NAME": "Bob"}, M)
        assert result.value.my_name == "Alice"
        assert "key_collision" in result.flags


# ---------------------------------------------------------------------------
# B5: Deterministic candidate selection (tested indirectly)
# ---------------------------------------------------------------------------


class TestDeterministicSelection:
    def test_pick_best_by_index(self):
        """Ensure _pick_best uses index not repr for tie-breaking."""
        from parsantic.scoring import pick_best
        from parsantic.types import ScoredValue

        # Two candidates with identical score and flag count
        sv1 = ScoredValue(value="zzz", flags=(), score=0)
        sv2 = ScoredValue(value="aaa", flags=(), score=0)
        # With repr, "aaa" < "zzz" so sv2 would win.
        # With index, sv1 should win (comes first).
        result = pick_best([sv1, sv2])
        assert result.value == "zzz"  # index 0 wins


# ---------------------------------------------------------------------------
# B6: coerce() API and debug APIs
# ---------------------------------------------------------------------------


class TestCoerceAPI:
    def test_coerce_dict_to_model(self):
        class M(BaseModel):
            name: str
            age: int

        result = coerce({"name": "Alice", "age": 30}, M)
        assert isinstance(result, ParseResult)
        assert result.value.name == "Alice"
        assert result.value.age == 30

    def test_coerce_dict_with_string_age(self):
        class M(BaseModel):
            name: str
            age: int

        result = coerce({"name": "Alice", "age": "30"}, M)
        assert result.value.age == 30

    @pytest.mark.parametrize(
        "data,target,expected",
        [(42, int, 42), ("42", int, 42)],
        ids=["already-valid", "string-to-int"],
    )
    def test_coerce_primitives(self, data, target, expected):
        result = coerce(data, target)
        assert result.value == expected


class TestDebugAPIs:
    def test_parse_debug(self):
        debug = parse_debug('{"name": "Alice"}', dict[str, str])
        assert isinstance(debug, ParseDebug)
        assert debug.value is not None
        assert isinstance(debug.chosen, CandidateDebug)
        assert debug.raw_text is not None
        assert len(debug.candidates) >= 1
        # with coercion
        debug2 = parse_debug('"123"', int)
        assert debug2.value == 123

    def test_coerce_debug(self):
        class M(BaseModel):
            name: str
            age: int

        debug = coerce_debug({"name": "Alice", "age": "30"}, M)
        assert isinstance(debug, ParseDebug)
        assert debug.value.age == 30
        assert debug.chosen is not None
        assert debug.raw_text is None
        assert len(debug.candidates) >= 1

    def test_coerce_debug_failure(self):
        class M(BaseModel):
            name: str
            age: int

        debug = coerce_debug("not a dict at all", M)
        assert isinstance(debug, ParseDebug)
        assert len(debug.candidates) >= 1


# ---------------------------------------------------------------------------
# Integration: full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    def test_enum_in_model_coerce(self):
        class M(BaseModel):
            color: Color

        result = coerce({"color": "RED"}, M)
        assert result.value.color == Color.RED

    def test_literal_in_model_coerce(self):
        class M(BaseModel):
            status: Literal["active", "inactive"]

        result = coerce({"status": "ACTIVE"}, M)
        assert result.value.status == "active"

    def test_nested_list_model_coerce(self):
        class Item(BaseModel):
            value: int

        class Container(BaseModel):
            items: list[Item]

        result = coerce({"items": [{"value": "1"}, {"value": "2"}]}, Container)
        assert len(result.value.items) == 2
        assert result.value.items[0].value == 1

    def test_safe_int_in_model_and_recursive_dict(self):
        class M(BaseModel):
            count: int

        result = coerce({"count": "3.0"}, M)
        assert result.value.count == 3

        result2 = coerce({"a": "1.5", "b": "2.5"}, dict[str, float])
        assert result2.value == {"a": 1.5, "b": 2.5}


# ---------------------------------------------------------------------------
# Recursion depth limit
# ---------------------------------------------------------------------------


class TestRecursionDepthLimit:
    def test_deeply_nested_structure_returns_max_depth_flag(self):
        """When _coerce_to_type is called at depth >100 it should return a
        ScoredValue with max_depth_exceeded in its flags instead of raising
        RecursionError."""
        from typing import Any

        from parsantic.coerce import CoerceOptions, _coerce_to_type

        # Build a list nested >100 levels deep where the innermost value
        # is a string that cannot validate as list[...], forcing recursion
        # through _coerce_list at every level.
        inner: Any = "leaf"
        target: Any = int
        for _ in range(110):
            inner = [inner]
            target = list[target]

        opts = CoerceOptions()
        # The fast-path validation will fail at each level because the
        # innermost value ("leaf") is not a valid int, forcing full
        # recursive coercion that exceeds the depth limit.
        sv = _coerce_to_type(inner, target, opts)
        assert "max_depth_exceeded" in sv.flags
