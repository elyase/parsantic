"""Comprehensive tests for parsantic.patch module."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from parsantic.patch import (
    JsonPatchOp,
    PatchDoc,
    PatchError,
    PatchPolicy,
    PolicyViolationError,
    apply_patch,
    apply_patch_and_validate,
    normalize_patches,
)

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class Address(BaseModel):
    city: str
    zip: str | None = None


class User(BaseModel):
    name: str
    age: int
    address: Address | None = None
    tags: list[str] = []
    bio: str = ""


# ===================================================================
# Basic add/replace operations
# ===================================================================


class TestBasicAddReplace:
    def test_add_top_level_key(self):
        doc = {"name": "Alice"}
        patches = [JsonPatchOp(op="add", path="/age", value=30)]
        result = apply_patch(doc, patches)
        assert result == {"name": "Alice", "age": 30}

    def test_replace_top_level_key(self):
        doc = {"name": "Alice", "age": 28}
        patches = [JsonPatchOp(op="replace", path="/age", value=29)]
        result = apply_patch(doc, patches)
        assert result == {"name": "Alice", "age": 29}

    def test_add_overwrites_existing(self):
        doc = {"name": "Alice", "age": 28}
        patches = [JsonPatchOp(op="add", path="/age", value=99)]
        result = apply_patch(doc, patches)
        assert result == {"name": "Alice", "age": 99}

    def test_replace_missing_key_raises(self):
        doc = {"name": "Alice"}
        patches = [JsonPatchOp(op="replace", path="/age", value=30)]
        with pytest.raises(PatchError, match="does not exist.*cannot replace"):
            apply_patch(doc, patches)

    def test_multiple_ops(self):
        doc = {"name": "Alice", "age": 28}
        patches = [
            JsonPatchOp(op="replace", path="/name", value="Bob"),
            JsonPatchOp(op="replace", path="/age", value=35),
        ]
        result = apply_patch(doc, patches)
        assert result == {"name": "Bob", "age": 35}


# ===================================================================
# Nested path operations
# ===================================================================


class TestNestedPaths:
    def test_nested_replace(self):
        doc = {"user": {"address": {"city": "NYC"}}}
        patches = [JsonPatchOp(op="replace", path="/user/address/city", value="LA")]
        result = apply_patch(doc, patches)
        assert result["user"]["address"]["city"] == "LA"

    def test_nested_add(self):
        doc = {"user": {"name": "Alice"}}
        patches = [JsonPatchOp(op="add", path="/user/email", value="a@b.com")]
        result = apply_patch(doc, patches)
        assert result["user"]["email"] == "a@b.com"

    def test_add_creates_intermediate_dicts(self):
        doc = {"user": {}}
        patches = [JsonPatchOp(op="add", path="/user/address/city", value="NYC")]
        result = apply_patch(doc, patches)
        assert result["user"]["address"]["city"] == "NYC"

    def test_replace_missing_nested_raises(self):
        doc = {"user": {"name": "Alice"}}
        patches = [JsonPatchOp(op="replace", path="/user/email", value="a@b.com")]
        with pytest.raises(PatchError, match="does not exist"):
            apply_patch(doc, patches)

    def test_deeply_nested(self):
        doc = {"a": {"b": {"c": {"d": "old"}}}}
        patches = [JsonPatchOp(op="replace", path="/a/b/c/d", value="new")]
        result = apply_patch(doc, patches)
        assert result["a"]["b"]["c"]["d"] == "new"


# ===================================================================
# Array operations
# ===================================================================


class TestArrayOps:
    def test_append_with_dash(self):
        doc = {"items": [1, 2, 3]}
        patches = [JsonPatchOp(op="add", path="/items/-", value=4)]
        result = apply_patch(doc, patches)
        assert result["items"] == [1, 2, 3, 4]

    def test_add_at_index(self):
        doc = {"items": ["a", "c"]}
        patches = [JsonPatchOp(op="add", path="/items/1", value="b")]
        result = apply_patch(doc, patches)
        assert result["items"] == ["a", "b", "c"]

    def test_add_at_index_zero(self):
        doc = {"items": ["b", "c"]}
        patches = [JsonPatchOp(op="add", path="/items/0", value="a")]
        result = apply_patch(doc, patches)
        assert result["items"] == ["a", "b", "c"]

    def test_replace_at_index(self):
        doc = {"items": ["a", "b", "c"]}
        patches = [JsonPatchOp(op="replace", path="/items/1", value="X")]
        result = apply_patch(doc, patches)
        assert result["items"] == ["a", "X", "c"]

    def test_remove_at_index(self):
        doc = {"items": ["a", "b", "c"]}
        policy = PatchPolicy(allow_remove=True)
        patches = [JsonPatchOp(op="remove", path="/items/1")]
        result = apply_patch(doc, patches, policy=policy)
        assert result["items"] == ["a", "c"]

    def test_out_of_bounds_raises(self):
        doc = {"items": [1, 2]}
        patches = [JsonPatchOp(op="replace", path="/items/5", value=99)]
        with pytest.raises(PatchError, match="out of bounds"):
            apply_patch(doc, patches)

    def test_negative_index_raises(self):
        doc = {"items": [1, 2]}
        patches = [JsonPatchOp(op="replace", path="/items/-1", value=99)]
        with pytest.raises(PatchError, match="Invalid array index.*negative"):
            apply_patch(doc, patches)

    def test_nested_array_operations(self):
        doc = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
        patches = [JsonPatchOp(op="replace", path="/users/0/name", value="Alicia")]
        result = apply_patch(doc, patches)
        assert result["users"][0]["name"] == "Alicia"
        assert result["users"][1]["name"] == "Bob"


# ===================================================================
# RFC 6901 escaping (~0 and ~1)
# ===================================================================


class TestRFC6901Escaping:
    def test_tilde_zero_escaping(self):
        """``~0`` should decode to ``~``."""
        doc = {"a~b": "old"}
        patches = [JsonPatchOp(op="replace", path="/a~0b", value="new")]
        result = apply_patch(doc, patches)
        assert result["a~b"] == "new"

    def test_tilde_one_escaping(self):
        """``~1`` should decode to ``/``."""
        doc = {"a/b": "old"}
        patches = [JsonPatchOp(op="replace", path="/a~1b", value="new")]
        result = apply_patch(doc, patches)
        assert result["a/b"] == "new"

    def test_combined_escaping(self):
        """``~01`` should decode to ``~1`` (tilde + ``1``), not ``/``."""
        doc = {"~1": "old"}
        patches = [JsonPatchOp(op="replace", path="/~01", value="new")]
        result = apply_patch(doc, patches)
        assert result["~1"] == "new"

    def test_slash_in_key_add(self):
        doc = {}
        patches = [JsonPatchOp(op="add", path="/config~1setting", value=True)]
        result = apply_patch(doc, patches)
        assert result["config/setting"] is True


# ===================================================================
# Policy enforcement
# ===================================================================


class TestPolicyEnforcement:
    def test_remove_blocked_by_default(self):
        doc = {"name": "Alice", "age": 28}
        patches = [JsonPatchOp(op="remove", path="/age")]
        with pytest.raises(PolicyViolationError, match="Remove operations are not allowed"):
            apply_patch(doc, patches)

    def test_remove_allowed_with_policy(self):
        doc = {"name": "Alice", "age": 28}
        policy = PatchPolicy(allow_remove=True)
        patches = [JsonPatchOp(op="remove", path="/age")]
        result = apply_patch(doc, patches, policy=policy)
        assert result == {"name": "Alice"}

    def test_max_ops_exceeded(self):
        doc = {"x": 0}
        policy = PatchPolicy(max_ops=3)
        patches = [JsonPatchOp(op="replace", path="/x", value=i) for i in range(5)]
        with pytest.raises(PolicyViolationError, match="5 operations.*at most 3"):
            apply_patch(doc, patches, policy=policy)

    def test_max_ops_within_limit(self):
        doc = {"x": 0}
        policy = PatchPolicy(max_ops=3)
        patches = [JsonPatchOp(op="replace", path="/x", value=i) for i in range(3)]
        result = apply_patch(doc, patches, policy=policy)
        assert result["x"] == 2

    def test_max_path_depth_exceeded(self):
        doc = {"a": {"b": {"c": "val"}}}
        policy = PatchPolicy(max_path_depth=2)
        patches = [JsonPatchOp(op="replace", path="/a/b/c", value="new")]
        with pytest.raises(PolicyViolationError, match="depth 3.*max_path_depth=2"):
            apply_patch(doc, patches, policy=policy)

    def test_max_path_depth_within_limit(self):
        doc = {"a": {"b": "val"}}
        policy = PatchPolicy(max_path_depth=2)
        patches = [JsonPatchOp(op="replace", path="/a/b", value="new")]
        result = apply_patch(doc, patches, policy=policy)
        assert result["a"]["b"] == "new"

    def test_append_blocked_by_policy(self):
        doc = {"items": [1, 2]}
        policy = PatchPolicy(allow_append=False)
        patches = [JsonPatchOp(op="add", path="/items/-", value=3)]
        with pytest.raises(PolicyViolationError, match="Append.*not allowed"):
            apply_patch(doc, patches, policy=policy)

    def test_append_allowed_by_default(self):
        doc = {"items": [1, 2]}
        patches = [JsonPatchOp(op="add", path="/items/-", value=3)]
        result = apply_patch(doc, patches)
        assert result["items"] == [1, 2, 3]


# ===================================================================
# String concat edge case (trustcall: /- on a string field)
# ===================================================================


class TestStringConcat:
    def test_add_dash_on_string_field(self):
        """When ``/-`` is applied to a string field, concatenate the value."""
        doc = {"bio": "Hello"}
        patches = [JsonPatchOp(op="add", path="/bio/-", value=" World")]
        result = apply_patch(doc, patches)
        assert result["bio"] == "Hello World"

    def test_replace_dash_on_string_field(self):
        """``replace`` with ``/-`` on a string should also concat."""
        doc = {"bio": "Hello"}
        patches = [JsonPatchOp(op="replace", path="/bio/-", value=" World")]
        result = apply_patch(doc, patches)
        assert result["bio"] == "Hello World"

    def test_nested_string_concat(self):
        doc = {"user": {"bio": "Base"}}
        patches = [JsonPatchOp(op="add", path="/user/bio/-", value=" Extra")]
        result = apply_patch(doc, patches)
        assert result["user"]["bio"] == "Base Extra"

    def test_string_concat_with_non_string_value(self):
        """Non-string value should be converted to string for concat."""
        doc = {"count": "Items: "}
        patches = [JsonPatchOp(op="add", path="/count/-", value=42)]
        result = apply_patch(doc, patches)
        assert result["count"] == "Items: 42"


# ===================================================================
# normalize_patches
# ===================================================================


class TestNormalizePatches:
    def test_list_of_dicts(self):
        raw = [{"op": "add", "path": "/x", "value": 1}]
        result = normalize_patches(raw)
        assert len(result) == 1
        assert result[0].op == "add"
        assert result[0].path == "/x"
        assert result[0].value == 1

    def test_list_of_json_patch_ops(self):
        ops = [JsonPatchOp(op="replace", path="/x", value=2)]
        result = normalize_patches(ops)
        assert result == ops

    def test_json_string_input(self):
        raw = json.dumps([{"op": "add", "path": "/x", "value": 1}])
        result = normalize_patches(raw)
        assert len(result) == 1
        assert result[0].op == "add"

    def test_nested_under_patches_key(self):
        raw = {"patches": [{"op": "add", "path": "/x", "value": 1}]}
        result = normalize_patches(raw)
        assert len(result) == 1
        assert result[0].op == "add"

    def test_json_string_nested(self):
        raw = json.dumps({"patches": [{"op": "replace", "path": "/x", "value": 5}]})
        result = normalize_patches(raw)
        assert len(result) == 1
        assert result[0].op == "replace"
        assert result[0].value == 5

    def test_single_dict(self):
        raw = {"op": "add", "path": "/x", "value": 1}
        result = normalize_patches(raw)
        assert len(result) == 1

    def test_malformed_string_raises(self):
        with pytest.raises(PatchError, match="Cannot (parse|normalize)"):
            normalize_patches("this is not json at all")

    def test_malformed_type_raises(self):
        with pytest.raises(PatchError, match="Cannot normalize"):
            normalize_patches(12345)

    def test_malformed_item_raises(self):
        with pytest.raises(PatchError, match="Cannot normalize patch item"):
            normalize_patches([42])

    def test_invalid_op_in_dict_raises(self):
        with pytest.raises(PatchError, match="Cannot parse patch dict"):
            normalize_patches([{"op": "invalid_op", "path": "/x"}])

    def test_empty_list(self):
        result = normalize_patches([])
        assert result == []


# ===================================================================
# apply_patch_and_validate
# ===================================================================


class TestApplyPatchAndValidate:
    def test_basic_validation(self):
        doc = {"name": "Alice", "age": 28}
        patches = [JsonPatchOp(op="replace", path="/age", value=29)]
        result = apply_patch_and_validate(doc, patches, User)
        assert isinstance(result.value, User)
        assert result.value.age == 29
        assert result.value.name == "Alice"

    def test_validation_failure(self):
        doc = {"name": "Alice", "age": 28}
        patches = [JsonPatchOp(op="replace", path="/age", value="not_a_number")]
        # Pydantic will coerce "not_a_number" and fail
        with pytest.raises(ValidationError):
            apply_patch_and_validate(doc, patches, User)

    def test_with_base_model_input(self):
        user = User(name="Alice", age=28)
        patches = [JsonPatchOp(op="replace", path="/age", value=29)]
        result = apply_patch_and_validate(user, patches, User)
        assert result.value.age == 29

    def test_with_type_adapter(self):
        doc = {"name": "Alice", "age": 28}
        adapter = TypeAdapter(User)
        patches = [JsonPatchOp(op="replace", path="/age", value=30)]
        result = apply_patch_and_validate(doc, patches, adapter)
        assert result.value.age == 30

    def test_nested_model_validation(self):
        doc = {"name": "Alice", "age": 28, "address": {"city": "NYC", "zip": "10001"}}
        patches = [JsonPatchOp(op="replace", path="/address/city", value="LA")]
        result = apply_patch_and_validate(doc, patches, User)
        assert result.value.address is not None
        assert result.value.address.city == "LA"
        assert result.value.address.zip == "10001"

    def test_result_has_parse_result_shape(self):
        doc = {"name": "Alice", "age": 28}
        patches = [JsonPatchOp(op="replace", path="/name", value="Bob")]
        result = apply_patch_and_validate(doc, patches, User)
        assert hasattr(result, "value")
        assert hasattr(result, "flags")
        assert hasattr(result, "score")
        assert result.flags == ()
        assert result.score == 0


# ===================================================================
# Deep copy verification
# ===================================================================


class TestDeepCopy:
    def test_original_doc_unchanged(self):
        doc = {"name": "Alice", "nested": {"x": 1}}
        original = copy.deepcopy(doc)
        patches = [
            JsonPatchOp(op="replace", path="/name", value="Bob"),
            JsonPatchOp(op="replace", path="/nested/x", value=99),
        ]
        result = apply_patch(doc, patches)
        # Result should be modified.
        assert result["name"] == "Bob"
        assert result["nested"]["x"] == 99
        # Original should be untouched.
        assert doc == original
        assert doc["name"] == "Alice"
        assert doc["nested"]["x"] == 1

    def test_original_list_unchanged(self):
        doc = {"items": [1, 2, 3]}
        original = copy.deepcopy(doc)
        patches = [JsonPatchOp(op="add", path="/items/-", value=4)]
        result = apply_patch(doc, patches)
        assert len(result["items"]) == 4
        assert doc == original

    def test_nested_mutation_isolation(self):
        doc = {"a": {"b": [1, 2]}}
        patches = [JsonPatchOp(op="add", path="/a/b/-", value=3)]
        result = apply_patch(doc, patches)
        assert result["a"]["b"] == [1, 2, 3]
        assert doc["a"]["b"] == [1, 2]


# ===================================================================
# Edge cases and error handling
# ===================================================================


class TestEdgeCases:
    def test_empty_patches(self):
        doc = {"x": 1}
        result = apply_patch(doc, [])
        assert result == {"x": 1}

    def test_invalid_pointer_no_leading_slash(self):
        doc = {"x": 1}
        patches = [JsonPatchOp(op="replace", path="x", value=2)]
        with pytest.raises(PatchError, match="must start with '/'"):
            apply_patch(doc, patches)

    def test_traverse_into_scalar_raises(self):
        doc = {"x": 42}
        patches = [JsonPatchOp(op="replace", path="/x/y", value=1)]
        with pytest.raises(PatchError, match="Cannot (traverse|replace)"):
            apply_patch(doc, patches)

    def test_remove_missing_key_raises(self):
        doc = {"name": "Alice"}
        policy = PatchPolicy(allow_remove=True)
        patches = [JsonPatchOp(op="remove", path="/nonexistent")]
        with pytest.raises(PatchError, match="does not exist.*cannot remove"):
            apply_patch(doc, patches, policy=policy)

    def test_patch_doc_model(self):
        """PatchDoc is a valid Pydantic model for LLM tool schemas."""
        pd = PatchDoc(
            json_doc_id="doc",
            planned_edits="Update name",
            patches=[JsonPatchOp(op="replace", path="/name", value="Bob")],
        )
        assert pd.json_doc_id == "doc"
        assert len(pd.patches) == 1

    def test_json_patch_op_model(self):
        op = JsonPatchOp(op="add", path="/x", value=42)
        data = op.model_dump()
        assert data["op"] == "add"
        assert data["path"] == "/x"
        assert data["value"] == 42

    def test_remove_op_no_value(self):
        op = JsonPatchOp(op="remove", path="/x")
        assert op.value is None

    def test_add_to_empty_list(self):
        doc = {"items": []}
        patches = [JsonPatchOp(op="add", path="/items/-", value="first")]
        result = apply_patch(doc, patches)
        assert result["items"] == ["first"]

    def test_multiple_appends(self):
        doc = {"tags": ["a"]}
        patches = [
            JsonPatchOp(op="add", path="/tags/-", value="b"),
            JsonPatchOp(op="add", path="/tags/-", value="c"),
        ]
        result = apply_patch(doc, patches)
        assert result["tags"] == ["a", "b", "c"]

    def test_add_nested_object(self):
        doc = {"user": {}}
        patches = [
            JsonPatchOp(
                op="add",
                path="/user/profile",
                value={"name": "Alice", "active": True},
            )
        ]
        result = apply_patch(doc, patches)
        assert result["user"]["profile"] == {"name": "Alice", "active": True}

    def test_replace_with_none_value(self):
        doc = {"name": "Alice"}
        patches = [JsonPatchOp(op="replace", path="/name", value=None)]
        result = apply_patch(doc, patches)
        assert result["name"] is None


# ===================================================================
# create_missing: list vs dict auto-detection
# ===================================================================


class TestCreateMissingListDetection:
    def test_create_missing_list_when_next_token_is_digit(self):
        """When path is /items/0 and items is missing, items should be
        auto-created as a list (not a dict) because the next token '0' is a
        digit.  This directly exercises the _resolve_parent look-ahead branch
        without relying on /-."""
        doc = {}
        patches = [
            JsonPatchOp(op="add", path="/items/0", value={"name": "Widget"}),
        ]
        result = apply_patch(doc, patches)
        assert isinstance(result["items"], list)
        assert result["items"][0] == {"name": "Widget"}

    def test_create_missing_dict_when_next_token_is_string(self):
        """When path is /metadata/key and metadata is missing, metadata should
        be auto-created as a dict (existing behaviour)."""
        doc = {}
        patches = [JsonPatchOp(op="add", path="/metadata/key", value="val")]
        result = apply_patch(doc, patches)
        assert isinstance(result["metadata"], dict)
        assert result["metadata"]["key"] == "val"

    def test_create_missing_list_when_next_token_is_dash(self):
        """When the next token after a missing key is '-', the container should
        be a list (for append semantics)."""
        doc = {}
        patches = [JsonPatchOp(op="add", path="/tags/-", value="first")]
        result = apply_patch(doc, patches)
        assert isinstance(result["tags"], list)
        assert result["tags"] == ["first"]
