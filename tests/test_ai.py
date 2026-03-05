"""Tests for parsantic.ai module.

Since pydantic-ai is NOT installed, these tests focus on:
1. Import guard: module imports safely, but pydantic-ai-dependent functions raise ImportError
2. Pure utility functions that work without pydantic-ai
3. Mocked pydantic-ai check for processor functions
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
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

    def test_sap_text_output_import_guard(self):
        """sap_text_output raises ImportError only when pydantic-ai is missing."""
        from parsantic.ai import _HAS_PYDANTIC_AI, sap_text_output

        if not _HAS_PYDANTIC_AI:
            with pytest.raises(ImportError, match="pydantic-ai"):
                sap_text_output(User)
        else:
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


# ---------------------------------------------------------------------------
# 2) validation_error_paths tests
# ---------------------------------------------------------------------------


class TestValidationErrorPaths:
    """Test conversion of Pydantic ValidationError locs to JSON Pointers."""

    def _make_error(self, target: type, data: Any) -> ValidationError:
        try:
            TypeAdapter(target).validate_python(data)
        except ValidationError as e:
            return e
        pytest.fail("Expected ValidationError")

    def test_various_error_paths(self):
        from parsantic.ai import validation_error_paths

        # simple field
        assert "/email" in validation_error_paths(
            self._make_error(User, {"username": "alice", "email": 123})
        )
        # nested field
        assert "/pets/0/age" in validation_error_paths(
            self._make_error(
                User,
                {
                    "username": "alice",
                    "email": "a@b.com",
                    "pets": [{"name": "Rex", "age": "not_a_number"}],
                },
            )
        )
        # multiple errors
        assert len(validation_error_paths(self._make_error(User, {"email": 42}))) >= 1
        # root-level error
        assert len(validation_error_paths(self._make_error(User, "not_a_dict"))) >= 1
        # deduplication
        paths = validation_error_paths(
            self._make_error(
                User,
                {"username": "alice", "email": "a@b.com", "pets": [{"name": "Rex", "age": "bad"}]},
            )
        )
        assert len(paths) == len(set(paths))

    def test_rfc6901_escaping(self):
        from parsantic.json_pointer import escape_json_pointer_token

        assert escape_json_pointer_token("a/b") == "a~1b"
        assert escape_json_pointer_token("a~b") == "a~0b"
        assert escape_json_pointer_token("a~/b") == "a~0~1b"


# ---------------------------------------------------------------------------
# 3) slice_doc_for_paths tests
# ---------------------------------------------------------------------------


class TestSliceDocForPaths:
    """Test extracting relevant document fragments."""

    def test_slice_doc(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"user": {"name": "Alice", "age": 30, "email": "a@b.com"}, "city": "NYC"}
        # simple path
        assert slice_doc_for_paths(doc, ["/user/age"]) == {"user": {"age": 30}}
        # empty paths → full doc
        assert slice_doc_for_paths(doc, []) == doc
        # root path → full doc
        assert slice_doc_for_paths({"a": 1}, [""]) == {"a": 1}
        # nonexistent
        assert slice_doc_for_paths({"name": "Alice"}, ["/nonexistent"]) == {"nonexistent": None}

    def test_slice_doc_multiple_and_array(self):
        from parsantic.ai import slice_doc_for_paths

        doc = {"name": "Alice", "age": 30, "email": "a@b.com", "city": "NYC"}
        result = slice_doc_for_paths(doc, ["/age", "/email"])
        assert result["age"] == 30
        assert result["email"] == "a@b.com"
        assert "city" not in result

        doc2 = {"pets": [{"name": "Rex", "age": 3}]}
        assert "pets" in slice_doc_for_paths(doc2, ["/pets/0/age"])


# ---------------------------------------------------------------------------
# 4) slice_schema_for_paths tests
# ---------------------------------------------------------------------------


class TestSliceSchemaForPaths:
    """Test extracting relevant schema fragments."""

    def test_json_schema_filtering(self):
        from parsantic.ai import slice_schema_for_paths

        schema = {
            "title": "User",
            "type": "object",
            "$defs": {"Pet": {"type": "object"}},
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
        }
        # property filtering + required filtering
        parsed = json.loads(slice_schema_for_paths(json.dumps(schema), ["/age"]))
        assert "age" in parsed["properties"] and "email" not in parsed["properties"]
        assert parsed.get("required") == ["age"]
        # preserves top-level metadata
        assert parsed.get("title") == "User" and "$defs" in parsed

    def test_empty_paths_and_non_json(self):
        from parsantic.ai import slice_schema_for_paths

        schema_text = '{"type": "object", "properties": {"x": {"type": "int"}}}'
        assert slice_schema_for_paths(schema_text, []) == schema_text
        assert "age" in slice_schema_for_paths(
            "class User:\n  name: str\n  age: int\n  email: str", ["/age"]
        )


# ---------------------------------------------------------------------------
# 5) build_patch_prompt tests
# ---------------------------------------------------------------------------


class TestBuildPatchPrompt:
    """Test patch prompt generation."""

    def test_basic_prompt_structure(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": "Alice", "age": "thirty"}
        errors = [
            {"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing"}
        ]
        prompt = build_patch_prompt(doc, errors)
        for expected in (
            "Current Document",
            "Validation Errors",
            "Instructions",
            "age",
            "int_parsing",
        ):
            assert expected in prompt
        # RFC 6902 instructions
        for expected in ("RFC 6902", "replace", "add", "json_doc_id"):
            assert expected in prompt

    def test_schema_inclusion(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": "Alice"}
        errors = [{"loc": ("age",), "msg": "Field required", "type": "missing"}]
        schema = '{"type": "object", "properties": {"age": {"type": "integer"}}}'
        assert "Target Schema" in build_patch_prompt(doc, errors, schema_text=schema)
        assert "Target Schema" not in build_patch_prompt(doc, errors)

    def test_doc_slicing_reduces_content(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": "Alice", "age": "thirty", "address": {"street": "123 Main St"}}
        errors = [{"loc": ("age",), "msg": "bad", "type": "int_parsing"}]
        assert "age" in build_patch_prompt(doc, errors, doc_slicing=True)
        assert "123 Main St" in build_patch_prompt(doc, errors, doc_slicing=False)

    def test_multiple_and_nested_errors(self):
        from parsantic.ai import build_patch_prompt

        doc = {"name": 123, "age": "bad", "user": {"pets": [{"name": "Rex", "age": "old"}]}}
        errors = [
            {"loc": ("name",), "msg": "bad", "type": "string_type"},
            {"loc": ("age",), "msg": "bad", "type": "int_parsing"},
            {"loc": ("user", "pets", 0, "age"), "msg": "bad", "type": "int_parsing"},
        ]
        prompt = build_patch_prompt(doc, errors)
        assert "string_type" in prompt
        assert "user -> pets -> 0 -> age" in prompt

    def test_edge_cases(self):
        from parsantic.ai import build_patch_prompt

        assert "Current Document" in build_patch_prompt({"a": 1}, [])
        assert "2024-01-01" in build_patch_prompt({"when": datetime(2024, 1, 1, tzinfo=UTC)}, [])


# ---------------------------------------------------------------------------
# 6) sap_text_output processor tests (mocked pydantic-ai check)
# ---------------------------------------------------------------------------


class TestSapTextOutput:
    """Test the sap_text_output processor with mocked pydantic-ai guard."""

    def test_processor_parses_various_formats(self):
        from parsantic.ai import sap_text_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet)
            # clean JSON
            result = processor('{"name": "Rex", "age": 3, "species": "dog"}')
            assert isinstance(result, Pet) and result.name == "Rex"
            # markdown fenced
            result = processor('```json\n{"name": "Rex", "age": 3}\n```')
            assert isinstance(result, Pet)
            # trailing comma
            result = processor('{"name": "Rex", "age": 3,}')
            assert isinstance(result, Pet)

    def test_processor_raises_on_invalid(self):
        from parsantic.ai import sap_text_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet)
            with pytest.raises((ValueError, ValidationError)):
                processor("completely invalid not json at all")

    def test_processor_with_custom_options(self):
        from parsantic.ai import sap_text_output
        from parsantic.jsonish import ParseOptions

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = sap_text_output(Pet, parse_options=ParseOptions(allow_markdown_json=True))
            result = processor('```json\n{"name": "Kitty", "age": 2}\n```')
            assert result.name == "Kitty"


# ---------------------------------------------------------------------------
# 7) patch_repair_output processor tests (mocked pydantic-ai check)
# ---------------------------------------------------------------------------


class TestPatchRepairOutput:
    """Test the patch_repair_output processor with mocked pydantic-ai guard."""

    def test_processor_parses_valid_inputs(self):
        from parsantic.ai import patch_repair_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet)
            # clean JSON
            result = processor('{"name": "Rex", "age": 3}')
            assert isinstance(result, Pet) and result.name == "Rex" and result.age == 3
            # markdown fenced
            result2 = processor('```json\n{"name": "Rex", "age": 5}\n```')
            assert isinstance(result2, Pet) and result2.age == 5
            # custom policy
            from parsantic.patch import PatchPolicy

            processor2 = patch_repair_output(
                Pet, policy=PatchPolicy(allow_remove=True, max_ops=10), max_attempts=5
            )
            assert isinstance(processor2('{"name": "Rex", "age": 3}'), Pet)

    def test_processor_reentrant_after_failure(self):
        from parsantic.ai import patch_repair_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet, max_attempts=0)
            assert processor('{"name": "Rex", "age": 3}').name == "Rex"
            with pytest.raises(ValueError):
                processor("totally invalid garbage not json at all")
            result = processor('{"name": "Luna", "age": 2}')
            assert result.name == "Luna" and result.age == 2

    def test_processor_run_id_and_concurrent_isolation(self):
        import uuid
        from dataclasses import dataclass

        from parsantic.ai import _RunState, patch_repair_output

        @dataclass
        class FakeRunContext:
            run_id: object

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet, max_attempts=0)

            # run_id isolation
            ctx1 = FakeRunContext(run_id=uuid.uuid4())
            assert processor(ctx1, text='{"name": "Rex", "age": 3}').name == "Rex"
            ctx2 = FakeRunContext(run_id=uuid.uuid4())
            result2 = processor(ctx2, text='{"name": "Luna", "age": 2}')
            assert result2.name == "Luna" and result2.age == 2
            assert processor._attempts == 0 and processor._prev_doc is None

            # concurrent isolation
            processor._run_states["run-A"] = _RunState(
                attempts=1, prev_doc={"name": "Rex", "age": "bad"}
            )
            ctx_b = FakeRunContext(run_id="run-B")
            assert processor(ctx_b, text='{"name": "Luna", "age": 2}').name == "Luna"
            assert processor._run_states["run-A"].attempts == 1

    def test_processor_has_name_attribute(self):
        from parsantic.ai import patch_repair_output

        with patch("parsantic.ai._HAS_PYDANTIC_AI", True):
            processor = patch_repair_output(Pet, max_attempts=0)
            assert processor.__name__ == "patch_repair_processor"


# ---------------------------------------------------------------------------
# 8) Internal helper tests
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    """Test internal utility functions."""

    @pytest.mark.parametrize(
        "pointer,expected",
        [
            ("/user/pets/0/age", ["/user/pets/0/age", "/user/pets/0", "/user/pets", "/user"]),
            ("/name", ["/name"]),
            ("", []),
        ],
        ids=["deep", "single", "empty"],
    )
    def test_parent_paths(self, pointer, expected):
        from parsantic.ai import _parent_paths

        assert _parent_paths(pointer) == expected

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

    @pytest.mark.parametrize(
        "segments,value,expected",
        [
            (["user", "name"], "Alice", {"user": {"name": "Alice"}}),
            (["a", "b", "c"], 42, {"a": {"b": {"c": 42}}}),
        ],
        ids=["shallow", "deep"],
    )
    def test_insert_at_path(self, segments, value, expected):
        from parsantic.ai import _insert_at_path

        target: dict[str, Any] = {}
        _insert_at_path(target, segments, value)
        assert target == expected
