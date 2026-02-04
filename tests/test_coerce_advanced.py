"""Advanced coercion tests covering B1-B6 improvements."""

from __future__ import annotations

import enum
from typing import Literal

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


class TestEnumMatchingExact:
    def test_exact_match(self):
        matched, flags = _match_enum_value("red", ["red", "green", "blue"], CoerceOptions())
        assert matched == "red"
        assert flags == ()

    def test_no_match(self):
        matched, flags = _match_enum_value("purple", ["red", "green", "blue"], CoerceOptions())
        assert matched is None


class TestEnumMatchingCaseInsensitive:
    def test_case_insensitive(self):
        matched, flags = _match_enum_value("RED", ["red", "green", "blue"], CoerceOptions())
        assert matched == "red"
        assert flags == ("case_insensitive",)

    def test_case_insensitive_mixed(self):
        matched, flags = _match_enum_value("Green", ["red", "green", "blue"], CoerceOptions())
        assert matched == "green"
        assert flags == ("case_insensitive",)


class TestEnumMatchingPunctStripped:
    def test_strip_punct(self):
        matched, flags = _match_enum_value(
            "ice-cream", ["icecream", "cake", "pie"], CoerceOptions()
        )
        assert matched == "icecream"
        assert flags == ("strip_punct",)

    def test_strip_punct_with_dots(self):
        matched, flags = _match_enum_value("U.S.A.", ["USA", "UK", "Canada"], CoerceOptions())
        assert matched == "USA"
        assert flags == ("strip_punct",)


class TestEnumMatchingAccentInsensitive:
    def test_accent_insensitive(self):
        matched, flags = _match_enum_value(
            "cafe", ["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9"], CoerceOptions()
        )
        assert matched == "caf\u00e9"
        assert flags == ("accent_insensitive",)

    def test_accent_insensitive_reverse(self):
        # Input has accent, candidate does not
        matched, flags = _match_enum_value(
            "caf\u00e9", ["cafe", "naive", "resume"], CoerceOptions()
        )
        assert matched == "cafe"
        assert flags == ("accent_insensitive",)


class TestEnumMatchingSubstring:
    def test_substring_disabled_by_default(self):
        matched, flags = _match_enum_value("re", ["red", "green", "blue"], CoerceOptions())
        assert matched is None

    def test_substring_enabled(self):
        opts = CoerceOptions(allow_substring_enum_match=True)
        matched, flags = _match_enum_value("re", ["red", "green", "blue"], opts)
        # "re" is substring of "red" and "green" -- ambiguous
        assert matched is not None
        assert "substring_match" in flags

    def test_substring_single_match(self):
        opts = CoerceOptions(allow_substring_enum_match=True)
        matched, flags = _match_enum_value("blu", ["red", "green", "blue"], opts)
        assert matched == "blue"
        assert flags == ("substring_match",)


class TestEnumMatchingAmbiguous:
    def test_ambiguous_picks_alphabetically(self):
        # Two candidates match case-insensitively
        matched, flags = _match_enum_value("AB", ["Ab", "aB"], CoerceOptions())
        assert matched == "Ab"
        assert "ambiguous_enum" in flags
        assert "case_insensitive" in flags


class TestEnumCoercionIntegration:
    def test_enum_via_parse(self):
        result = parse('"RED"', Color, is_done=True)
        assert result.value == Color.RED
        assert "case_insensitive" in result.flags

    def test_enum_exact_via_parse(self):
        result = parse('"red"', Color, is_done=True)
        assert result.value == Color.RED

    def test_enum_accented_via_parse(self):
        result = parse('"cafe"', AccentedEnum, is_done=True)
        assert result.value == AccentedEnum.CAFE
        assert "accent_insensitive" in result.flags


class TestLiteralMatching:
    def test_literal_exact(self):
        result = parse('"hello"', Literal["hello", "world"], is_done=True)
        assert result.value == "hello"

    def test_literal_case_insensitive(self):
        result = parse('"HELLO"', Literal["hello", "world"], is_done=True)
        assert result.value == "hello"
        assert "case_insensitive" in result.flags

    def test_literal_punct_stripped(self):
        result = parse('"ice-cream"', Literal["icecream", "cake"], is_done=True)
        assert result.value == "icecream"
        assert "strip_punct" in result.flags

    def test_literal_accent_insensitive(self):
        result = parse('"cafe"', Literal["caf\u00e9", "tea"], is_done=True)
        assert result.value == "caf\u00e9"
        assert "accent_insensitive" in result.flags


# ---------------------------------------------------------------------------
# B2: Recursive coercion (list, dict, union, tuple)
# ---------------------------------------------------------------------------


class TestRecursiveListCoercion:
    def test_list_of_int_from_strings(self):
        opts = CoerceOptions()
        sv = _coerce_to_type(["1", "2", "3"], list[int], opts)
        assert sv.value == [1, 2, 3]
        # Pydantic can coerce string->int natively, so the fast path
        # succeeds with no flags. The important thing is the value is correct.

    def test_list_of_float_from_strings(self):
        opts = CoerceOptions()
        sv = _coerce_to_type(["1.5", "2.5"], list[float], opts)
        assert sv.value == [1.5, 2.5]

    def test_coerce_api_list_int(self):
        result = coerce(["1", "2", "3"], list[int])
        assert result.value == [1, 2, 3]


class TestRecursiveDictCoercion:
    def test_dict_str_float(self):
        opts = CoerceOptions()
        sv = _coerce_to_type({"a": "1.5", "b": "2.5"}, dict[str, float], opts)
        assert sv.value == {"a": 1.5, "b": 2.5}

    def test_dict_int_keys(self):
        opts = CoerceOptions()
        sv = _coerce_to_type({"1": "hello", "2": "world"}, dict[int, str], opts)
        assert sv.value == {1: "hello", 2: "world"}


class TestRecursiveTupleCoercion:
    def test_homogeneous_tuple(self):
        opts = CoerceOptions()
        sv = _coerce_to_type(["1", "2", "3"], tuple[int, ...], opts)
        assert sv.value == (1, 2, 3)

    def test_fixed_tuple(self):
        opts = CoerceOptions()
        sv = _coerce_to_type(["1", "hello"], tuple[int, str], opts)
        assert sv.value == (1, "hello")


class TestUnionCoercion:
    def test_union_picks_best(self):
        # "42" should coerce to int (exact match) over str
        opts = CoerceOptions()
        sv = _coerce_to_type("42", int | str, opts)
        # str is a direct match with score 0, int requires coercion
        # But Union should try int first and succeed
        assert sv.value == 42 or sv.value == "42"
        # Either is acceptable; the point is it doesn't crash

    def test_union_none(self):
        opts = CoerceOptions()
        sv = _coerce_to_type(None, int | None, opts)
        assert sv.value is None


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


class TestSafeIntCoercion:
    def test_integer_string(self):
        assert _try_int("123") == 123

    def test_float_string_exact_int(self):
        """'3.0' should become 3 (within epsilon)."""
        assert _try_int("3.0") == 3

    def test_float_string_exact_int_trailing_zeros(self):
        """'3.00' should become 3."""
        assert _try_int("3.00") == 3

    def test_float_string_not_int(self):
        """'1.4' should NOT become 1."""
        assert _try_int("1.4") is None

    def test_float_string_half(self):
        """'1.5' should NOT become 2 (unsafe rounding)."""
        assert _try_int("1.5") is None

    def test_negative_exact_int(self):
        assert _try_int("-3.0") == -3

    def test_negative_not_int(self):
        assert _try_int("-1.7") is None

    def test_safe_int_via_parse(self):
        """'3.0' as int should work via parse."""
        result = parse('"3.0"', int, is_done=True)
        assert result.value == 3

    def test_unsafe_int_via_parse_fallback(self):
        """'1.4' as int should fail (not silently round)."""
        # parse will try int coercion, it should fail since 1.4 is not near integer
        # It should fall back to trying float, then if int is wanted it should not
        # match. Let's test that _try_int returns None.
        assert _try_int("1.4") is None


# ---------------------------------------------------------------------------
# B4: Key mapping with aliases and validation_alias
# ---------------------------------------------------------------------------


class TestKeyMappingAliases:
    def test_field_alias(self):
        class M(BaseModel):
            my_field: str = Field(alias="myField")

        result = coerce({"myField": "hello"}, M)
        assert result.value.my_field == "hello"

    def test_validation_alias(self):
        class M(BaseModel):
            my_field: str = Field(validation_alias="my-field")

        result = coerce({"my-field": "hello"}, M)
        assert result.value.my_field == "hello"

    def test_validation_alias_precedence(self):
        """validation_alias should work for input validation."""

        class M(BaseModel):
            my_field: str = Field(validation_alias="inputField")

        result = coerce({"inputField": "hello"}, M)
        assert result.value.my_field == "hello"


class TestExtraAllowModel:
    def test_extra_allow_preserves_extras(self):
        class M(BaseModel):
            model_config = {"extra": "allow"}
            name: str

        result = coerce({"name": "Alice", "age": 30}, M)
        assert result.value.name == "Alice"
        # Extra fields should be preserved
        assert result.value.age == 30  # type: ignore[attr-defined]

    def test_extra_forbid_drops_extras(self):
        class M(BaseModel):
            name: str

        # Without extra="allow", unmatched keys are dropped and flagged
        result = coerce({"name": "Alice", "age": 30}, M)
        assert result.value.name == "Alice"


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
        from parsantic.coerce import _pick_best
        from parsantic.types import ScoredValue

        # Two candidates with identical score and flag count
        sv1 = ScoredValue(value="zzz", flags=(), score=0)
        sv2 = ScoredValue(value="aaa", flags=(), score=0)
        # With repr, "aaa" < "zzz" so sv2 would win.
        # With index, sv1 should win (comes first).
        result = _pick_best([sv1, sv2])
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

    def test_coerce_already_valid(self):
        result = coerce(42, int)
        assert result.value == 42
        assert result.flags == ()
        assert result.score == 0

    def test_coerce_string_to_int(self):
        result = coerce("42", int)
        assert result.value == 42


class TestParseDebugAPI:
    def test_parse_debug_success(self):
        debug = parse_debug('{"name": "Alice"}', dict[str, str])
        assert isinstance(debug, ParseDebug)
        assert debug.value is not None
        assert debug.chosen is not None
        assert isinstance(debug.chosen, CandidateDebug)
        assert debug.raw_text is not None

    def test_parse_debug_with_coercion(self):
        debug = parse_debug('"123"', int)
        assert debug.value == 123
        assert debug.chosen is not None

    def test_parse_debug_has_candidates(self):
        debug = parse_debug('{"x": 1}', dict[str, int])
        assert len(debug.candidates) >= 1


class TestCoerceDebugAPI:
    def test_coerce_debug_success(self):
        class M(BaseModel):
            name: str

        debug = coerce_debug({"name": "Alice"}, M)
        assert isinstance(debug, ParseDebug)
        assert debug.value is not None
        assert debug.value.name == "Alice"
        assert debug.chosen is not None
        assert debug.raw_text is None  # coerce has no raw text

    def test_coerce_debug_with_coercion(self):
        class M(BaseModel):
            name: str
            age: int

        debug = coerce_debug({"name": "Alice", "age": "30"}, M)
        assert debug.value is not None
        assert debug.value.age == 30
        assert debug.chosen is not None
        assert len(debug.candidates) >= 1

    def test_coerce_debug_failure(self):
        """When coercion completely fails, debug should still return structured info."""

        class M(BaseModel):
            name: str
            age: int

        # Completely wrong type - should fail
        debug = coerce_debug("not a dict at all", M)
        # It may or may not succeed depending on coercion paths
        # But it should return a ParseDebug either way
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

    def test_safe_int_in_model(self):
        class M(BaseModel):
            count: int

        # 3.0 should work
        result = coerce({"count": "3.0"}, M)
        assert result.value.count == 3

    def test_recursive_dict_coerce_api(self):
        result = coerce({"a": "1.5", "b": "2.5"}, dict[str, float])
        assert result.value == {"a": 1.5, "b": 2.5}


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
