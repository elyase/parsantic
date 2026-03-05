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
    @pytest.mark.parametrize(
        "doc,patches,expected",
        [
            (
                {"name": "Alice"},
                [JsonPatchOp(op="add", path="/age", value=30)],
                {"name": "Alice", "age": 30},
            ),
            (
                {"name": "Alice", "age": 28},
                [JsonPatchOp(op="replace", path="/age", value=29)],
                {"name": "Alice", "age": 29},
            ),
            (
                {"name": "Alice", "age": 28},
                [JsonPatchOp(op="add", path="/age", value=99)],
                {"name": "Alice", "age": 99},
            ),
            (
                {"name": "Alice", "age": 28},
                [
                    JsonPatchOp(op="replace", path="/name", value="Bob"),
                    JsonPatchOp(op="replace", path="/age", value=35),
                ],
                {"name": "Bob", "age": 35},
            ),
        ],
        ids=["add-top", "replace-top", "add-overwrites", "multiple-ops"],
    )
    def test_happy_paths(self, doc, patches, expected):
        assert apply_patch(doc, patches) == expected

    def test_replace_missing_key_raises(self):
        with pytest.raises(PatchError, match="does not exist.*cannot replace"):
            apply_patch({"name": "Alice"}, [JsonPatchOp(op="replace", path="/age", value=30)])


# ===================================================================
# Nested path operations
# ===================================================================


class TestNestedPaths:
    @pytest.mark.parametrize(
        "doc,patches,path_keys,expected_val",
        [
            (
                {"user": {"address": {"city": "NYC"}}},
                [JsonPatchOp(op="replace", path="/user/address/city", value="LA")],
                ["user", "address", "city"],
                "LA",
            ),
            (
                {"user": {"name": "Alice"}},
                [JsonPatchOp(op="add", path="/user/email", value="a@b.com")],
                ["user", "email"],
                "a@b.com",
            ),
            (
                {"user": {}},
                [JsonPatchOp(op="add", path="/user/address/city", value="NYC")],
                ["user", "address", "city"],
                "NYC",
            ),
            (
                {"a": {"b": {"c": {"d": "old"}}}},
                [JsonPatchOp(op="replace", path="/a/b/c/d", value="new")],
                ["a", "b", "c", "d"],
                "new",
            ),
        ],
        ids=["nested-replace", "nested-add", "creates-intermediate", "deeply-nested"],
    )
    def test_nested_operations(self, doc, patches, path_keys, expected_val):
        result = apply_patch(doc, patches)
        node = result
        for key in path_keys:
            node = node[key]
        assert node == expected_val

    def test_replace_missing_nested_raises(self):
        with pytest.raises(PatchError, match="does not exist"):
            apply_patch(
                {"user": {"name": "Alice"}},
                [JsonPatchOp(op="replace", path="/user/email", value="a@b.com")],
            )


# ===================================================================
# Array operations
# ===================================================================


class TestArrayOps:
    @pytest.mark.parametrize(
        "doc,patches,expected_items,policy",
        [
            (
                {"items": [1, 2, 3]},
                [JsonPatchOp(op="add", path="/items/-", value=4)],
                [1, 2, 3, 4],
                None,
            ),
            (
                {"items": ["a", "c"]},
                [JsonPatchOp(op="add", path="/items/1", value="b")],
                ["a", "b", "c"],
                None,
            ),
            (
                {"items": ["b", "c"]},
                [JsonPatchOp(op="add", path="/items/0", value="a")],
                ["a", "b", "c"],
                None,
            ),
            (
                {"items": ["a", "b", "c"]},
                [JsonPatchOp(op="replace", path="/items/1", value="X")],
                ["a", "X", "c"],
                None,
            ),
            (
                {"items": ["a", "b", "c"]},
                [JsonPatchOp(op="remove", path="/items/1")],
                ["a", "c"],
                PatchPolicy(allow_remove=True),
            ),
        ],
        ids=["append-dash", "add-at-1", "add-at-0", "replace-at-1", "remove-at-1"],
    )
    def test_array_happy_paths(self, doc, patches, expected_items, policy):
        result = apply_patch(doc, patches, policy=policy) if policy else apply_patch(doc, patches)
        assert result["items"] == expected_items

    @pytest.mark.parametrize(
        "doc,path,match",
        [
            ({"items": [1, 2]}, "/items/5", "out of bounds"),
            ({"items": [1, 2]}, "/items/-1", "Invalid array index.*negative"),
        ],
        ids=["out-of-bounds", "negative-index"],
    )
    def test_array_errors(self, doc, path, match):
        with pytest.raises(PatchError, match=match):
            apply_patch(doc, [JsonPatchOp(op="replace", path=path, value=99)])

    def test_nested_array_operations(self):
        result = apply_patch(
            {"users": [{"name": "Alice"}, {"name": "Bob"}]},
            [JsonPatchOp(op="replace", path="/users/0/name", value="Alicia")],
        )
        assert result["users"][0]["name"] == "Alicia"
        assert result["users"][1]["name"] == "Bob"


# ===================================================================
# RFC 6901 escaping (~0 and ~1)
# ===================================================================


class TestRFC6901Escaping:
    @pytest.mark.parametrize(
        "doc,path,op,value,expected_key,expected_val",
        [
            ({"a~b": "old"}, "/a~0b", "replace", "new", "a~b", "new"),
            ({"a/b": "old"}, "/a~1b", "replace", "new", "a/b", "new"),
            ({"~1": "old"}, "/~01", "replace", "new", "~1", "new"),
            ({}, "/config~1setting", "add", True, "config/setting", True),
        ],
        ids=["tilde-zero", "tilde-one", "combined", "slash-in-key-add"],
    )
    def test_rfc6901_escaping(self, doc, path, op, value, expected_key, expected_val):
        patches = [JsonPatchOp(op=op, path=path, value=value)]
        result = apply_patch(doc, patches)
        assert result[expected_key] == expected_val


# ===================================================================
# Policy enforcement
# ===================================================================


class TestPolicyEnforcement:
    @pytest.mark.parametrize(
        "doc,patches,policy,match",
        [
            (
                {"name": "Alice", "age": 28},
                [JsonPatchOp(op="remove", path="/age")],
                None,
                "Remove operations are not allowed",
            ),
            (
                {"x": 0},
                [JsonPatchOp(op="replace", path="/x", value=i) for i in range(5)],
                PatchPolicy(max_ops=3),
                "5 operations.*at most 3",
            ),
            (
                {"a": {"b": {"c": "val"}}},
                [JsonPatchOp(op="replace", path="/a/b/c", value="new")],
                PatchPolicy(max_path_depth=2),
                "depth 3.*max_path_depth=2",
            ),
            (
                {"items": [1, 2]},
                [JsonPatchOp(op="add", path="/items/-", value=3)],
                PatchPolicy(allow_append=False),
                "Append.*not allowed",
            ),
        ],
        ids=["remove-blocked", "max-ops-exceeded", "max-depth-exceeded", "append-blocked"],
    )
    def test_policy_violations(self, doc, patches, policy, match):
        with pytest.raises(PolicyViolationError, match=match):
            apply_patch(doc, patches, policy=policy) if policy else apply_patch(doc, patches)

    def test_policy_allows_within_limits(self):
        # remove allowed
        assert apply_patch(
            {"name": "Alice", "age": 28},
            [JsonPatchOp(op="remove", path="/age")],
            policy=PatchPolicy(allow_remove=True),
        ) == {"name": "Alice"}
        # max_ops within limit
        assert (
            apply_patch(
                {"x": 0},
                [JsonPatchOp(op="replace", path="/x", value=i) for i in range(3)],
                policy=PatchPolicy(max_ops=3),
            )["x"]
            == 2
        )
        # max_path_depth within limit
        assert (
            apply_patch(
                {"a": {"b": "val"}},
                [JsonPatchOp(op="replace", path="/a/b", value="new")],
                policy=PatchPolicy(max_path_depth=2),
            )["a"]["b"]
            == "new"
        )
        # append allowed by default
        assert apply_patch({"items": [1, 2]}, [JsonPatchOp(op="add", path="/items/-", value=3)])[
            "items"
        ] == [1, 2, 3]


# ===================================================================
# String concat edge case (trustcall: /- on a string field)
# ===================================================================


class TestStringConcat:
    @pytest.mark.parametrize(
        "doc,path,op,value,check_path,expected",
        [
            ({"bio": "Hello"}, "/bio/-", "add", " World", ["bio"], "Hello World"),
            ({"bio": "Hello"}, "/bio/-", "replace", " World", ["bio"], "Hello World"),
            (
                {"user": {"bio": "Base"}},
                "/user/bio/-",
                "add",
                " Extra",
                ["user", "bio"],
                "Base Extra",
            ),
            ({"count": "Items: "}, "/count/-", "add", 42, ["count"], "Items: 42"),
        ],
        ids=["add-dash", "replace-dash", "nested", "non-string-value"],
    )
    def test_string_concat_happy_paths(self, doc, path, op, value, check_path, expected):
        result = apply_patch(doc, [JsonPatchOp(op=op, path=path, value=value)])
        node = result
        for key in check_path:
            node = node[key]
        assert node == expected

    def test_string_concat_invalid_list_index_is_patch_error(self):
        with pytest.raises(PatchError):
            apply_patch(
                {"items": ["a", "b"]},
                [JsonPatchOp(op="add", path="/items/not-an-index/-", value="x")],
            )


# ===================================================================
# normalize_patches
# ===================================================================


class TestNormalizePatches:
    @pytest.mark.parametrize(
        "raw,expected_op,expected_count",
        [
            ([{"op": "add", "path": "/x", "value": 1}], "add", 1),
            ([JsonPatchOp(op="replace", path="/x", value=2)], "replace", 1),
            (json.dumps([{"op": "add", "path": "/x", "value": 1}]), "add", 1),
            ({"patches": [{"op": "add", "path": "/x", "value": 1}]}, "add", 1),
            (json.dumps({"patches": [{"op": "replace", "path": "/x", "value": 5}]}), "replace", 1),
            ({"op": "add", "path": "/x", "value": 1}, "add", 1),
            ([], None, 0),
        ],
        ids=[
            "list-dicts",
            "list-ops",
            "json-string",
            "nested-key",
            "json-nested",
            "single-dict",
            "empty",
        ],
    )
    def test_normalize_happy_path(self, raw, expected_op, expected_count):
        result = normalize_patches(raw)
        assert len(result) == expected_count
        if expected_count > 0:
            assert result[0].op == expected_op

    @pytest.mark.parametrize(
        "raw,match",
        [
            ("this is not json at all", "Cannot (parse|normalize)"),
            (12345, "Cannot normalize"),
            ([42], "Cannot normalize patch item"),
            ([{"op": "invalid_op", "path": "/x"}], "Cannot parse patch dict"),
        ],
        ids=["bad-string", "bad-type", "bad-item", "invalid-op"],
    )
    def test_normalize_error(self, raw, match):
        with pytest.raises(PatchError, match=match):
            normalize_patches(raw)


# ===================================================================
# apply_patch_and_validate
# ===================================================================


class TestApplyPatchAndValidate:
    def test_validates_various_inputs(self):
        # dict input
        result = apply_patch_and_validate(
            {"name": "Alice", "age": 28},
            [JsonPatchOp(op="replace", path="/age", value=29)],
            User,
        )
        assert isinstance(result.value, User)
        assert result.value.age == 29 and result.value.name == "Alice"
        # ParseResult shape
        assert result.flags == () and result.score == 0

        # BaseModel input
        result2 = apply_patch_and_validate(
            User(name="Alice", age=28),
            [JsonPatchOp(op="replace", path="/age", value=29)],
            User,
        )
        assert result2.value.age == 29

        # TypeAdapter
        result3 = apply_patch_and_validate(
            {"name": "Alice", "age": 28},
            [JsonPatchOp(op="replace", path="/age", value=30)],
            TypeAdapter(User),
        )
        assert result3.value.age == 30

    def test_nested_model_validation(self):
        result = apply_patch_and_validate(
            {"name": "Alice", "age": 28, "address": {"city": "NYC", "zip": "10001"}},
            [JsonPatchOp(op="replace", path="/address/city", value="LA")],
            User,
        )
        assert result.value.address.city == "LA" and result.value.address.zip == "10001"

    def test_validation_failure(self):
        with pytest.raises(ValidationError):
            apply_patch_and_validate(
                {"name": "Alice", "age": 28},
                [JsonPatchOp(op="replace", path="/age", value="not_a_number")],
                User,
            )


# ===================================================================
# Deep copy verification
# ===================================================================


class TestDeepCopy:
    @pytest.mark.parametrize(
        "doc,patches,check_result,check_original",
        [
            (
                {"name": "Alice", "nested": {"x": 1}},
                [
                    JsonPatchOp(op="replace", path="/name", value="Bob"),
                    JsonPatchOp(op="replace", path="/nested/x", value=99),
                ],
                lambda r: r["name"] == "Bob" and r["nested"]["x"] == 99,
                lambda d: d["name"] == "Alice" and d["nested"]["x"] == 1,
            ),
            (
                {"items": [1, 2, 3]},
                [JsonPatchOp(op="add", path="/items/-", value=4)],
                lambda r: len(r["items"]) == 4,
                lambda d: d["items"] == [1, 2, 3],
            ),
            (
                {"a": {"b": [1, 2]}},
                [JsonPatchOp(op="add", path="/a/b/-", value=3)],
                lambda r: r["a"]["b"] == [1, 2, 3],
                lambda d: d["a"]["b"] == [1, 2],
            ),
        ],
        ids=["nested-dict", "list-append", "nested-list"],
    )
    def test_original_unchanged(self, doc, patches, check_result, check_original):
        original = copy.deepcopy(doc)
        result = apply_patch(doc, patches)
        assert check_result(result)
        assert doc == original
        assert check_original(doc)


# ===================================================================
# Edge cases and error handling
# ===================================================================


class TestEdgeCases:
    @pytest.mark.parametrize(
        "doc,patches,match,policy",
        [
            ({"x": 1}, [JsonPatchOp(op="replace", path="x", value=2)], "must start with '/'", None),
            (
                {"x": 42},
                [JsonPatchOp(op="replace", path="/x/y", value=1)],
                "Cannot (traverse|replace)",
                None,
            ),
            (
                {"name": "Alice"},
                [JsonPatchOp(op="remove", path="/nonexistent")],
                "does not exist.*cannot remove",
                PatchPolicy(allow_remove=True),
            ),
        ],
        ids=["no-leading-slash", "traverse-scalar", "remove-missing"],
    )
    def test_error_cases(self, doc, patches, match, policy):
        with pytest.raises(PatchError, match=match):
            apply_patch(doc, patches, policy=policy) if policy else apply_patch(doc, patches)

    def test_various_happy_edge_cases(self):
        # empty patches
        assert apply_patch({"x": 1}, []) == {"x": 1}
        # add to empty list
        assert apply_patch({"items": []}, [JsonPatchOp(op="add", path="/items/-", value="first")])[
            "items"
        ] == ["first"]
        # multiple appends
        assert apply_patch(
            {"tags": ["a"]},
            [
                JsonPatchOp(op="add", path="/tags/-", value="b"),
                JsonPatchOp(op="add", path="/tags/-", value="c"),
            ],
        )["tags"] == ["a", "b", "c"]
        # add nested object
        assert apply_patch(
            {"user": {}},
            [JsonPatchOp(op="add", path="/user/profile", value={"name": "Alice", "active": True})],
        )["user"]["profile"] == {"name": "Alice", "active": True}
        # replace with None
        assert (
            apply_patch({"name": "Alice"}, [JsonPatchOp(op="replace", path="/name", value=None)])[
                "name"
            ]
            is None
        )

    def test_pydantic_models(self):
        pd = PatchDoc(
            json_doc_id="doc",
            planned_edits="Update name",
            patches=[JsonPatchOp(op="replace", path="/name", value="Bob")],
        )
        assert pd.json_doc_id == "doc" and len(pd.patches) == 1
        op = JsonPatchOp(op="add", path="/x", value=42)
        assert op.model_dump() == {"op": "add", "path": "/x", "value": 42}
        assert JsonPatchOp(op="remove", path="/x").value is None


# ===================================================================
# create_missing: list vs dict auto-detection
# ===================================================================


class TestCreateMissingListDetection:
    @pytest.mark.parametrize(
        "doc,path,value,check",
        [
            (
                {},
                "/items/0",
                {"name": "Widget"},
                lambda r: isinstance(r["items"], list) and r["items"][0] == {"name": "Widget"},
            ),
            ({}, "/items/0/name", "Widget", lambda r: r == {"items": [{"name": "Widget"}]}),
            (
                {"items": []},
                "/items/0/name",
                "Widget",
                lambda r: r == {"items": [{"name": "Widget"}]},
            ),
            (
                {},
                "/metadata/key",
                "val",
                lambda r: isinstance(r["metadata"], dict) and r["metadata"]["key"] == "val",
            ),
            (
                {},
                "/tags/-",
                "first",
                lambda r: isinstance(r["tags"], list) and r["tags"] == ["first"],
            ),
        ],
        ids=[
            "digit-creates-list",
            "nested-into-missing",
            "nested-into-empty",
            "string-creates-dict",
            "dash-creates-list",
        ],
    )
    def test_auto_create_container(self, doc, path, value, check):
        result = apply_patch(doc, [JsonPatchOp(op="add", path=path, value=value)])
        assert check(result)
