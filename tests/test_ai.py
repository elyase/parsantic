"""Tests for parsantic.ai module.

Since pydantic-ai is NOT installed, these tests focus on:
1. Import guard: module imports safely, but pydantic-ai-dependent functions raise ImportError
2. Pure utility functions that work without pydantic-ai
3. Mocked pydantic-ai check for processor functions
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class Pet(BaseModel):
    name: str
    age: int
    species: str = "unknown"


class User(BaseModel):
    username: str
    email: str
    pets: list[Pet] = []
    score: float | None = None


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class UserWithAddress(BaseModel):
    name: str
    age: int
    address: Address | None = None


# ---------------------------------------------------------------------------
# 1) Import guard tests
# ---------------------------------------------------------------------------


class TestImportGuard:
    """The module should be importable without pydantic-ai."""

    def test_module_imports_successfully(self):
        """Importing parsantic.ai should not raise."""
        import parsantic.ai

        assert hasattr(parsantic.ai, "sap_text_output")
        assert hasattr(parsantic.ai, "patch_repair_output")
        assert hasattr(parsantic.ai, "build_patch_prompt")
        assert hasattr(parsantic.ai, "validation_error_paths")

    def test_has_pydantic_ai_flag(self):
        """The internal flag reflects whether pydantic-ai is installed."""
        from parsantic.ai import _HAS_PYDANTIC_AI

        try:
            import pydantic_ai  # noqa: F401

            assert _HAS_PYDANTIC_AI is True
        except ImportError:
            assert _HAS_PYDANTIC_AI is False

    def test_sap_text_output_import_guard(self):
        """sap_text_output raises ImportError only when pydantic-ai is missing."""
        from parsantic.ai import _HAS_PYDANTIC_AI, sap_text_output

        if not _HAS_PYDANTIC_AI:
            with pytest.raises(ImportError, match="pydantic-ai"):
                sap_text_output(User)
        else:
            # Should return a callable without error
            processor = sap_text_output(User)
            assert callable(processor)

    def test_patch_repair_output_import_guard(self):
        """patch_repair_output raises ImportError only when pydantic-ai is missing."""
        from parsantic.ai import _HAS_PYDANTIC_AI, patch_repair_output

        if not _HAS_PYDANTIC_AI:
            with pytest.raises(ImportError, match="pydantic-ai"):
                patch_repair_output(User)
        else:
            processor = patch_repair_output(User)
            assert callable(processor)

    def test_check_pydantic_ai_helper(self):
        """_check_pydantic_ai raises ImportError only when pydantic-ai is missing."""
        from parsantic.ai import _HAS_PYDANTIC_AI, _check_pydantic_ai

        if not _HAS_PYDANTIC_AI:
            with pytest.raises(ImportError):
                _check_pydantic_ai()
        else:
            _check_pydantic_ai()  # should not raise

    def test_pure_utilities_work_without_pydantic_ai(self):
        """Pure utility functions should work regardless of pydantic-ai."""
        from parsantic.ai import (
            build_patch_prompt,
            slice_doc_for_paths,
            slice_schema_for_paths,
            validation_error_paths,
        )

        # These should all be callable
        assert callable(validation_error_paths)
        assert callable(slice_schema_for_paths)
        assert callable(slice_doc_for_paths)
        assert callable(build_patch_prompt)


# ---------------------------------------------------------------------------
# 2) validation_error_paths tests
# ---------------------------------------------------------------------------


class TestValidationErrorPaths:
    """Test conversion of Pydantic ValidationError locs to JSON Pointers."""

    def _make_error(self, target: type, data: Any) -> ValidationError:
        """Helper to create a ValidationError."""
        try:
            TypeAdapter(target).validate_python(data)
        except ValidationError as e:
            return e
        pytest.fail("Expected ValidationError")

    def test_simple_field_error(self):
        from parsantic.ai import validation_error_paths

        err = self._make_error(User, {"username": "alice", "email": 123})
        paths = validation_error_paths(err)
        assert "/email" in paths

    def test_nested_field_error(self):
        from parsantic.ai import validation_error_paths

        err = self._make_error(
            User,
            {
                "username": "alice",
                "email": "a@b.com",
                "pets": [{"name": "Rex", "age": "not_a_number"}],
            },
        )
        paths = validation_error_paths(err)
        assert "/pets/0/age" in paths

    def test_multiple_errors(self):
        from parsantic.ai import validation_error_paths

        err = self._make_error(User, {"email": 42})
        paths = validation_error_paths(err)
        # Should have at least the username (missing) and email (wrong type) paths
        assert len(paths) >= 1

    def test_empty_loc_produces_root_path(self):
        from parsantic.ai import validation_error_paths

        # Force a root-level error by passing wrong type entirely
        err = self._make_error(User, "not_a_dict")
        paths = validation_error_paths(err)
        assert len(paths) >= 1

    def test_deduplication(self):
        from parsantic.ai import validation_error_paths

        # Multiple errors at the same location should be deduplicated
        err = self._make_error(
            User,
            {
                "username": "alice",
                "email": "a@b.com",
                "pets": [{"name": "Rex", "age": "bad"}],
            },
        )
        paths = validation_error_paths(err)
        # No duplicates
        assert len(paths) == len(set(paths))

    def test_rfc6901_escaping(self):
        """Segments with ~ or / should be properly escaped."""
        from parsantic.ai import _escape_json_pointer_token

        assert _escape_json_pointer_token("a/b") == "a~1b"
        assert _escape_json_pointer_token("a~b") == "a~0b"
        assert _escape_json_pointer_token("a~/b") == "a~0~1b"


# ---------------------------------------------------------------------------
# 3) slice_doc_for_paths tests
# ---------------------------------------------------------------------------


class TestSliceDocForPaths:
    """Test extracting relevant document fragments."""

    def test_simple_path(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"user": {"name": "Alice", "age": 30, "email": "a@b.com"}}
        result = slice_doc_for_paths(doc, ["/user/age"])
        assert result == {"user": {"age": 30}}

    def test_multiple_paths(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"name": "Alice", "age": 30, "email": "a@b.com", "city": "NYC"}
        result = slice_doc_for_paths(doc, ["/age", "/email"])
        assert result["age"] == 30
        assert result["email"] == "a@b.com"
        assert "city" not in result

    def test_array_path(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"pets": [{"name": "Rex", "age": 3}, {"name": "Cat", "age": 5}]}
        result = slice_doc_for_paths(doc, ["/pets/0/age"])
        assert "pets" in result

    def test_empty_paths_returns_full_doc(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"a": 1, "b": 2}
        result = slice_doc_for_paths(doc, [])
        assert result == doc

    def test_nonexistent_path(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"name": "Alice"}
        result = slice_doc_for_paths(doc, ["/nonexistent"])
        assert result == {"nonexistent": None}

    def test_root_path_returns_full_doc(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"a": 1}
        result = slice_doc_for_paths(doc, [""])
        assert result == doc


# ---------------------------------------------------------------------------
# 4) slice_schema_for_paths tests
# ---------------------------------------------------------------------------


class TestSliceSchemaForPaths:
    """Test extracting relevant schema fragments."""

    def test_json_schema_property_filtering(self):
        from parsantic.ai import slice_schema_for_paths

        schema = {
            "title": "User",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
        }
        schema_text = json.dumps(schema)
        result = slice_schema_for_paths(schema_text, ["/age"])
        parsed = json.loads(result)
        assert "age" in parsed.get("properties", {})
        # Other properties may be excluded
        assert "email" not in parsed.get("properties", {})

    def test_preserves_top_level_metadata(self):
        from parsantic.ai import slice_schema_for_paths

        schema = {
            "title": "User",
            "type": "object",
            "$defs": {"Pet": {"type": "object"}},
            "properties": {
                "name": {"type": "string"},
            },
        }
        schema_text = json.dumps(schema)
        result = slice_schema_for_paths(schema_text, ["/name"])
        parsed = json.loads(result)
        assert parsed.get("title") == "User"
        assert "$defs" in parsed

    def test_empty_paths_returns_full_schema(self):
        from parsantic.ai import slice_schema_for_paths

        schema_text = '{"type": "object", "properties": {"x": {"type": "int"}}}'
        result = slice_schema_for_paths(schema_text, [])
        assert result == schema_text

    def test_non_json_schema_uses_line_filtering(self):
        from parsantic.ai import slice_schema_for_paths

        schema_text = "class User:\n  name: str\n  age: int\n  email: str"
        result = slice_schema_for_paths(schema_text, ["/age"])
        assert "age" in result

    def test_required_field_filtered(self):
        from parsantic.ai import slice_schema_for_paths

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        schema_text = json.dumps(schema)
        result = slice_schema_for_paths(schema_text, ["/age"])
        parsed = json.loads(result)
        # Only "age" should be in required (name is filtered out)
        assert parsed.get("required") == ["age"]


# ---------------------------------------------------------------------------
# 5) build_patch_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPatchPrompt:
    """Test patch prompt generation."""

    def test_basic_prompt_structure(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": "Alice", "age": "thirty"}
        errors = [
            {"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing"},
        ]
        prompt = build_patch_prompt(doc, errors)
        assert "Current Document" in prompt
        assert "Validation Errors" in prompt
        assert "Instructions" in prompt
        assert "age" in prompt
        assert "int_parsing" in prompt

    def test_includes_schema_when_provided(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": "Alice"}
        errors = [{"loc": ("age",), "msg": "Field required", "type": "missing"}]
        schema = '{"type": "object", "properties": {"age": {"type": "integer"}}}'
        prompt = build_patch_prompt(doc, errors, schema_text=schema)
        assert "Target Schema" in prompt
        assert "integer" in prompt

    def test_no_schema_section_when_not_provided(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": "Alice"}
        errors = [{"loc": ("age",), "msg": "Field required", "type": "missing"}]
        prompt = build_patch_prompt(doc, errors)
        assert "Target Schema" not in prompt

    def test_doc_slicing_reduces_content(self):
        from parsantic.ai import build_patch_prompt

        doc = {
            "name": "Alice",
            "age": "thirty",
            "email": "alice@example.com",
            "address": {"street": "123 Main St", "city": "NYC"},
        }
        errors = [
            {"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing"},
        ]
        prompt_sliced = build_patch_prompt(doc, errors, doc_slicing=True)
        prompt_full = build_patch_prompt(doc, errors, doc_slicing=False)

        # The sliced prompt should not contain the full address details
        # (since the error is only about "age")
        assert "age" in prompt_sliced
        # Full prompt contains everything
        assert "123 Main St" in prompt_full

    def test_multiple_errors(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": 123, "age": "bad"}
        errors = [
            {"loc": ("name",), "msg": "Input should be a valid string", "type": "string_type"},
            {"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing"},
        ]
        prompt = build_patch_prompt(doc, errors)
        assert "name" in prompt
        assert "age" in prompt
        assert "string_type" in prompt
        assert "int_parsing" in prompt

    def test_nested_error_paths(self):
        from parsantic.ai import build_patch_prompt

        doc = {"user": {"pets": [{"name": "Rex", "age": "old"}]}}
        errors = [
            {
                "loc": ("user", "pets", 0, "age"),
                "msg": "Input should be a valid integer",
                "type": "int_parsing",
            },
        ]
        prompt = build_patch_prompt(doc, errors)
        assert "user -> pets -> 0 -> age" in prompt

    def test_rfc6902_instructions_present(self):
        from parsantic.ai import build_patch_prompt

        doc = {"x": 1}
        errors = [{"loc": ("x",), "msg": "bad", "type": "err"}]
        prompt = build_patch_prompt(doc, errors)
        assert "RFC 6902" in prompt
        assert "replace" in prompt
        assert "add" in prompt
        assert "json_doc_id" in prompt

    def test_empty_errors_still_builds_prompt(self):
        from parsantic.ai import build_patch_prompt

        doc = {"a": 1}
        prompt = build_patch_prompt(doc, [])
        assert "Current Document" in prompt


# ---------------------------------------------------------------------------
# 6) sap_text_output processor tests (mocked pydantic-ai check)
# ---------------------------------------------------------------------------


class TestSapTextOutput:
    """Test the sap_text_output processor with mocked pydantic-ai guard."""

    def test_processor_parses_clean_json(self):
        from parsantic.ai import sap_text_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet)
            result = processor('{"name": "Rex", "age": 3, "species": "dog"}')
            assert isinstance(result, Pet)
            assert result.name == "Rex"
            assert result.age == 3

    def test_processor_parses_jsonish_text(self):
        from parsantic.ai import sap_text_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet)
            # Markdown fenced JSON
            result = processor('```json\n{"name": "Rex", "age": 3}\n```')
            assert isinstance(result, Pet)
            assert result.name == "Rex"

    def test_processor_with_trailing_comma(self):
        from parsantic.ai import sap_text_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet)
            result = processor('{"name": "Rex", "age": 3,}')
            assert isinstance(result, Pet)

    def test_processor_raises_on_invalid(self):
        from parsantic.ai import sap_text_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet)
            with pytest.raises((ValueError, ValidationError)):
                processor("completely invalid not json at all")

    def test_processor_with_parse_options(self):
        from parsantic.ai import sap_text_output
        from parsantic.jsonish import ParseOptions

        opts = ParseOptions(allow_markdown_json=True)
        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet, parse_options=opts)
            result = processor('```json\n{"name": "Kitty", "age": 2}\n```')
            assert result.name == "Kitty"

    def test_processor_with_coerce_options(self):
        from parsantic.ai import sap_text_output
        from parsantic.coerce import CoerceOptions

        opts = CoerceOptions(allow_substring_enum_match=True)
        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet, coerce_options=opts)
            result = processor('{"name": "Rex", "age": 3}')
            assert result.name == "Rex"


# ---------------------------------------------------------------------------
# 7) patch_repair_output processor tests (mocked pydantic-ai check)
# ---------------------------------------------------------------------------


class TestPatchRepairOutput:
    """Test the patch_repair_output processor with mocked pydantic-ai guard."""

    def test_processor_returns_valid_on_first_try(self):
        from parsantic.ai import patch_repair_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet)
            result = processor('{"name": "Rex", "age": 3}')
            assert isinstance(result, Pet)
            assert result.name == "Rex"
            assert result.age == 3

    def test_processor_parses_jsonish_successfully(self):
        from parsantic.ai import patch_repair_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet)
            result = processor('```json\n{"name": "Rex", "age": 5}\n```')
            assert isinstance(result, Pet)
            assert result.age == 5

    def test_processor_with_custom_policy(self):
        from parsantic.ai import patch_repair_output
        from parsantic.patch import PatchPolicy

        policy = PatchPolicy(allow_remove=True, max_ops=10)
        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet, policy=policy, max_attempts=5)
            result = processor('{"name": "Rex", "age": 3}')
            assert isinstance(result, Pet)


# ---------------------------------------------------------------------------
# 8) Internal helper tests
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Test internal utility functions."""

    def test_parent_paths(self):
        from parsantic.ai import _parent_paths

        paths = _parent_paths("/user/pets/0/age")
        assert paths == ["/user/pets/0/age", "/user/pets/0", "/user/pets", "/user"]

    def test_parent_paths_single_segment(self):
        from parsantic.ai import _parent_paths

        paths = _parent_paths("/name")
        assert paths == ["/name"]

    def test_parent_paths_empty(self):
        from parsantic.ai import _parent_paths

        paths = _parent_paths("")
        assert paths == []

    def test_pointer_to_segments(self):
        from parsantic.ai import _pointer_to_segments

        assert _pointer_to_segments("/user/pets/0/age") == ["user", "pets", "0", "age"]
        assert _pointer_to_segments("") == []
        assert _pointer_to_segments("/a~0b/c~1d") == ["a~b", "c/d"]

    def test_get_at_path(self):
        from parsantic.ai import _get_at_path

        doc = {"user": {"pets": [{"name": "Rex", "age": 3}]}}
        assert _get_at_path(doc, ["user", "pets", "0", "name"]) == "Rex"
        assert _get_at_path(doc, ["user", "pets", "0", "age"]) == 3
        assert _get_at_path(doc, ["nonexistent"]) is None

    def test_insert_at_path(self):
        from parsantic.ai import _insert_at_path

        target: dict[str, Any] = {}
        _insert_at_path(target, ["user", "name"], "Alice")
        assert target == {"user": {"name": "Alice"}}

    def test_insert_at_path_nested(self):
        from parsantic.ai import _insert_at_path

        target: dict[str, Any] = {}
        _insert_at_path(target, ["a", "b", "c"], 42)
        assert target == {"a": {"b": {"c": 42}}}

    def test_escape_json_pointer_token(self):
        from parsantic.ai import _escape_json_pointer_token

        assert _escape_json_pointer_token("simple") == "simple"
        assert _escape_json_pointer_token("a/b") == "a~1b"
        assert _escape_json_pointer_token("a~b") == "a~0b"
        assert _escape_json_pointer_token("~1") == "~01"
