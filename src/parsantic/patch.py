"""Safe JSON Patch (RFC 6902) operations for LLM-generated structured updates.

This module intentionally supports a focused subset of RFC 6902 operations:
``add``, ``replace``, and ``remove``.  Unsupported operations (``move``,
``copy``, ``test``) raise :class:`PatchError`.

For LLM-forgiving flows, ``add`` paths auto-create missing intermediate
containers:

* missing object segments create ``dict`` containers
* missing list segments can append a placeholder container when the traversal
  index equals ``len(list)``
* list traversal beyond ``len(list)`` is rejected

Key features:

* **Policy enforcement** -- configurable limits on allowed operations, patch
  count, and path depth.
* **RFC 6901 JSON Pointer** resolution with proper ``~0``/``~1`` escaping.
* **String-concat edge case** -- when a patch targets ``/-`` on a *string*
  field the value is concatenated rather than raising a type error.  This
  matches the behaviour of the *trustcall* library and is the most common
  failure mode when an LLM tries to "append" to a text field.
* **Deep-copy semantics** -- the input document is never mutated.
* **Pydantic integration** -- :func:`apply_patch_and_validate` applies patches
  and validates the result against a Pydantic model / ``TypeAdapter``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from .api import ParseResult
from .json_pointer import parse_json_pointer

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PatchError(Exception):
    """Base exception for all patch-related errors."""


class PolicyViolationError(PatchError):
    """Raised when a patch set violates the active :class:`PatchPolicy`."""


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class JsonPatchOp(BaseModel):
    """A single RFC 6902 JSON Patch operation.

    Only ``add``, ``replace``, and ``remove`` are supported.  ``value`` is
    required for ``add`` and ``replace`` and ignored for ``remove``.
    """

    op: Literal["add", "replace", "remove"]
    path: str
    value: Any | None = None


class PatchDoc(BaseModel):
    """LLM-facing tool schema for submitting a batch of patches.

    ``json_doc_id``
        Identifier of the document being patched (use ``"doc"`` for
        single-document workflows).
    ``planned_edits``
        Short bullet-point plan produced by the LLM for debugging purposes;
        the plan is *not* executed.
    ``patches``
        Ordered list of :class:`JsonPatchOp` operations.
    """

    json_doc_id: str = Field(description="ID of the document to patch (use 'doc' if single).")
    planned_edits: str = Field(description="Short plan of edits (for debugging; not executed).")
    patches: list[JsonPatchOp] = Field(description="RFC 6902 add/replace/remove operations.")


@dataclass(slots=True)
class PatchPolicy:
    """Safety rails for patch application.

    Attributes
    ----------
    allow_remove : bool
        Whether ``remove`` operations are permitted.  Default ``False``
        to prevent accidental data loss.
    max_ops : int
        Maximum number of operations in a single patch set.
    max_path_depth : int
        Maximum number of path segments (``/``-separated tokens) allowed
        in a single JSON Pointer.
    allow_append : bool
        Whether ``/-`` (array append) is permitted.
    """

    allow_remove: bool = False
    max_ops: int = 50
    max_path_depth: int = 32
    allow_append: bool = True


# ---------------------------------------------------------------------------
# JSON Pointer helpers (RFC 6901)
# ---------------------------------------------------------------------------


def _parse_pointer(path: str) -> list[str]:
    """Split a JSON Pointer string into unescaped reference tokens.

    Returns an empty list for the root pointer ``""``.  Raises
    :class:`PatchError` if the pointer does not start with ``/``.
    """
    try:
        return parse_json_pointer(path)
    except ValueError as exc:
        raise PatchError(str(exc)) from exc


def _resolve_parent(
    doc: Any, tokens: list[str], *, max_depth: int, create_missing: bool = False
) -> tuple[Any, str]:
    """Walk *doc* following *tokens* and return ``(parent, last_token)``.

    Parameters
    ----------
    doc
        The mutable document (already deep-copied).
    tokens
        Reference tokens parsed from the JSON Pointer.
    max_depth
        Maximum allowed number of tokens (enforced by policy).
    create_missing
        If ``True``, create intermediate ``dict`` containers for missing
        keys.  Used by the ``add`` operation.

    Raises
    ------
    PatchError
        If a token cannot be resolved (missing key, bad array index, etc.).
    PolicyViolationError
        If the path exceeds *max_depth*.
    """
    if len(tokens) > max_depth:
        raise PolicyViolationError(
            f"Path depth {len(tokens)} exceeds policy max_path_depth={max_depth}"
        )
    if not tokens:
        raise PatchError("Cannot operate on the document root (empty path)")

    current = doc
    for i, tok in enumerate(tokens[:-1]):
        if isinstance(current, dict):
            if tok not in current:
                if create_missing:
                    # Look ahead: if the next token is a digit or "-",
                    # the path implies list indexing, so create a list.
                    next_tok = tokens[i + 1]
                    if next_tok.isdigit() or next_tok == "-":
                        current[tok] = []
                    else:
                        current[tok] = {}
                    current = current[tok]
                    continue
                raise PatchError(f"Key {tok!r} not found while resolving path")
            current = current[tok]
        elif isinstance(current, list):
            next_tok = tokens[i + 1]
            if tok == "-":
                if not create_missing:
                    raise PatchError("The '-' index is only valid for 'add' operations")
                idx = len(current)
            else:
                try:
                    idx = int(tok)
                except ValueError:
                    raise PatchError(f"Invalid array index: {tok!r}") from None
                if idx < 0:
                    raise PatchError(
                        f"Invalid array index: negative indices are not allowed ({idx})"
                    )

            if idx < len(current):
                current = current[idx]
                continue

            if create_missing and idx == len(current):
                current.append([] if (next_tok.isdigit() or next_tok == "-") else {})
                current = current[idx]
                continue

            raise PatchError(f"Array index {idx} out of bounds (length {len(current)})")
        else:
            raise PatchError(f"Cannot traverse into {type(current).__name__} with token {tok!r}")
    return current, tokens[-1]


def _list_index(token: str, lst: list[Any], *, allow_dash: bool, allow_append: bool = False) -> int:
    """Convert a JSON Pointer token to a list index.

    ``-`` is only allowed when *allow_dash* is ``True`` (used for ``add``).
    *allow_append* permits ``idx == len(lst)`` so that RFC 6902 ``add`` can
    insert at the end of a list using an explicit numeric index.
    """
    if token == "-":
        if allow_dash:
            return len(lst)
        raise PatchError("The '-' index is only valid for 'add' operations")
    try:
        idx = int(token)
    except ValueError:
        raise PatchError(f"Invalid array index: {token!r}") from None
    if idx < 0:
        raise PatchError(f"Invalid array index: negative indices are not allowed ({idx})")
    if idx > len(lst) or (idx == len(lst) and not allow_append):
        raise PatchError(f"Array index {idx} out of bounds (length {len(lst)})")
    return idx


# ---------------------------------------------------------------------------
# Individual operation handlers
# ---------------------------------------------------------------------------


def _apply_add(doc: Any, tokens: list[str], value: Any, *, max_depth: int) -> None:
    """Apply an ``add`` operation (mutates *doc* in-place)."""
    parent, last = _resolve_parent(doc, tokens, max_depth=max_depth, create_missing=True)
    if isinstance(parent, dict):
        parent[last] = value
    elif isinstance(parent, list):
        if last == "-":
            parent.append(value)
        else:
            idx = _list_index(last, parent, allow_dash=True, allow_append=True)
            if idx == len(parent):
                parent.append(value)
            else:
                parent.insert(idx, value)
    elif isinstance(parent, str):
        # Trustcall edge case: ``/-`` on a string field means concatenation.
        if last == "-":
            _string_concat_in_parent(doc, tokens, value)
        else:
            raise PatchError(f"Cannot add into a string value at {'/'.join(tokens[:-1])!r}")
    else:
        raise PatchError(f"Cannot add into {type(parent).__name__} at {'/'.join(tokens[:-1])!r}")


def _apply_replace(doc: Any, tokens: list[str], value: Any, *, max_depth: int) -> None:
    """Apply a ``replace`` operation (mutates *doc* in-place).

    The target path **must** already exist; ``replace`` never creates
    intermediate containers.
    """
    parent, last = _resolve_parent(doc, tokens, max_depth=max_depth, create_missing=False)

    # Handle string-concat edge case for ``/-`` on a string field.
    if isinstance(parent, str) and last == "-":
        _string_concat_in_parent(doc, tokens, value)
        return

    if isinstance(parent, dict):
        if last not in parent:
            raise PatchError(f"Path does not exist (cannot replace): key {last!r}")
        parent[last] = value
    elif isinstance(parent, list):
        idx = _list_index(last, parent, allow_dash=False)
        parent[idx] = value
    else:
        raise PatchError(f"Cannot replace in {type(parent).__name__}")


def _apply_remove(doc: Any, tokens: list[str], *, max_depth: int) -> None:
    """Apply a ``remove`` operation (mutates *doc* in-place)."""
    parent, last = _resolve_parent(doc, tokens, max_depth=max_depth, create_missing=False)
    if isinstance(parent, dict):
        if last not in parent:
            raise PatchError(f"Path does not exist (cannot remove): key {last!r}")
        del parent[last]
    elif isinstance(parent, list):
        idx = _list_index(last, parent, allow_dash=False)
        del parent[idx]
    else:
        raise PatchError(f"Cannot remove from {type(parent).__name__}")


# ---------------------------------------------------------------------------
# String-concat edge case  (trustcall compatibility)
# ---------------------------------------------------------------------------


def _string_concat_in_parent(doc: Any, tokens: list[str], value: Any) -> None:
    """Handle ``/-`` applied to a string field by concatenating *value*.

    Walks the document to the *grandparent* and replaces the string at the
    parent key with the concatenated result.

    This is a practical edge case documented by the *trustcall* library: LLMs
    sometimes emit ``{"op": "add", "path": "/bio/-", "value": " extra text"}``
    when ``bio`` is a ``str``, intending to append text.  Rather than raising
    a type error we interpret this as string concatenation.
    """
    # tokens looks like [..., parent_key, "-"]
    parent_tokens = tokens[:-1]
    if not parent_tokens:
        raise PatchError("Cannot apply string-concat on the document root")
    grandparent_tokens = parent_tokens[:-1]
    field_key = parent_tokens[-1]

    if grandparent_tokens:
        gp, _ = _resolve_parent(
            doc, parent_tokens, max_depth=len(parent_tokens), create_missing=False
        )
        container = gp
    else:
        container = doc

    if isinstance(container, dict):
        old = container.get(field_key, "")
        if not isinstance(old, str):
            raise PatchError(
                f"Expected string for concat at {'/'.join(parent_tokens)!r}, "
                f"got {type(old).__name__}"
            )
        container[field_key] = old + str(value)
    elif isinstance(container, list):
        idx = int(field_key)
        old = container[idx]
        if not isinstance(old, str):
            raise PatchError(
                f"Expected string for concat at {'/'.join(parent_tokens)!r}, "
                f"got {type(old).__name__}"
            )
        container[idx] = old + str(value)
    else:
        raise PatchError(f"Cannot apply string-concat: grandparent is {type(container).__name__}")


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def _validate_policy(patches: Sequence[JsonPatchOp], policy: PatchPolicy) -> None:
    """Check that *patches* conform to *policy*.

    Raises :class:`PolicyViolationError` on the first violation found.
    """
    if len(patches) > policy.max_ops:
        raise PolicyViolationError(
            f"Patch set contains {len(patches)} operations, "
            f"but policy allows at most {policy.max_ops}"
        )
    for patch in patches:
        # Disallow ``remove`` unless policy permits it.
        if patch.op == "remove" and not policy.allow_remove:
            raise PolicyViolationError("Remove operations are not allowed by the current policy")
        # Disallow ``/-`` append unless policy permits it.
        if patch.path.endswith("/-") and not policy.allow_append:
            raise PolicyViolationError(
                "Append (/-) operations are not allowed by the current policy"
            )
        # Check path depth.
        tokens = _parse_pointer(patch.path)
        if len(tokens) > policy.max_path_depth:
            raise PolicyViolationError(
                f"Path {patch.path!r} has depth {len(tokens)}, "
                f"exceeding policy max_path_depth={policy.max_path_depth}"
            )


# ---------------------------------------------------------------------------
# Public API: apply_patch
# ---------------------------------------------------------------------------


def apply_patch(
    doc: dict[str, Any],
    patches: Sequence[JsonPatchOp],
    *,
    policy: PatchPolicy | None = None,
) -> dict[str, Any]:
    """Apply a sequence of JSON Patch operations to *doc* and return the result.

    The input *doc* is **never mutated**; a deep copy is made before applying
    any operations.

    Parameters
    ----------
    doc
        The document to patch.
    patches
        Ordered list of :class:`JsonPatchOp` operations.
    policy
        Safety policy to enforce.  See :class:`PatchPolicy`.

    Returns
    -------
    dict
        The patched document.

    Raises
    ------
    PatchError
        If any operation fails (missing path, bad index, etc.).
    PolicyViolationError
        If the patch set violates the active policy.
    """
    policy = policy or PatchPolicy()
    _validate_policy(patches, policy)
    result = copy.deepcopy(doc)
    for patch in patches:
        tokens = _parse_pointer(patch.path)
        if patch.op == "add":
            _apply_add(result, tokens, patch.value, max_depth=policy.max_path_depth)
        elif patch.op == "replace":
            _apply_replace(result, tokens, patch.value, max_depth=policy.max_path_depth)
        elif patch.op == "remove":
            _apply_remove(result, tokens, max_depth=policy.max_path_depth)
        else:
            raise PatchError(f"Unsupported operation: {patch.op!r}")
    return result


# ---------------------------------------------------------------------------
# Public API: apply_patch_and_validate
# ---------------------------------------------------------------------------


def apply_patch_and_validate[T](
    doc: dict[str, Any] | BaseModel,
    patches: Sequence[JsonPatchOp],
    target: type[T] | TypeAdapter[T],
    *,
    policy: PatchPolicy | None = None,
) -> ParseResult[T]:
    """Apply patches then validate the result against a Pydantic type.

    If *doc* is a :class:`~pydantic.BaseModel` it is first converted to a
    ``dict`` via :meth:`~pydantic.BaseModel.model_dump`.

    Returns a :class:`~parsantic.api.ParseResult` with the validated
    value, empty flags, and a score of ``0``.

    Raises
    ------
    PatchError
        If any patch operation fails.
    PolicyViolationError
        If the patches violate the active policy.
    pydantic.ValidationError
        If the patched document does not conform to *target*.
    """
    if isinstance(doc, BaseModel):
        doc_dict: dict[str, Any] = doc.model_dump(mode="json")
    else:
        doc_dict = doc

    patched = apply_patch(doc_dict, patches, policy=policy)

    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    validated = adapter.validate_python(patched)
    return ParseResult(value=validated, flags=(), score=0)


# ---------------------------------------------------------------------------
# Public API: normalize_patches
# ---------------------------------------------------------------------------


def normalize_patches(patches: Any) -> list[JsonPatchOp]:
    """Coerce various patch representations into a list of :class:`JsonPatchOp`.

    Handles common LLM failure modes:

    * A ``list[dict]`` of raw patch dicts.
    * A JSON **string** containing a list of patches.
    * A ``dict`` with a ``"patches"`` key wrapping the actual list.
    * A JSON string wrapping a dict with a ``"patches"`` key.
    * A single ``dict`` (treated as a one-element list).
    * Already-validated :class:`JsonPatchOp` instances (pass-through).

    Raises
    ------
    PatchError
        If the input cannot be interpreted as a list of patches.
    """
    # 1) If it's a string, try to parse it as JSON first.
    if isinstance(patches, str):
        patches = _parse_json_string(patches)

    # 2) If it's a dict with a "patches" key, unwrap.
    if isinstance(patches, dict):
        if "patches" in patches:
            patches = patches["patches"]
        else:
            # Treat a single dict as a one-element list.
            patches = [patches]

    # 3) Must be iterable at this point.
    if not isinstance(patches, (list, tuple)):
        raise PatchError(
            f"Cannot normalize patches: expected list or dict, got {type(patches).__name__}"
        )

    result: list[JsonPatchOp] = []
    for item in patches:
        if isinstance(item, JsonPatchOp):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(JsonPatchOp(**item))
            except Exception as exc:
                raise PatchError(f"Cannot parse patch dict {item!r}: {exc}") from exc
        else:
            raise PatchError(
                f"Cannot normalize patch item: expected dict or JsonPatchOp, "
                f"got {type(item).__name__}"
            )
    return result


def _parse_json_string(text: str) -> Any:
    """Attempt to parse *text* as JSON, with SAP fallback.

    Tries ``json.loads`` first.  If that fails, attempts to import
    and use the SAP jsonish parser as a fallback for malformed JSON.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt SAP parse as a fallback for "JSON-ish" strings.
    try:
        from .jsonish import ParseOptions, parse_jsonish

        jv = parse_jsonish(
            text,
            options=ParseOptions(
                allow_markdown_json=True,
                allow_find_all_json_objects=True,
                allow_fixes=True,
                allow_as_string=False,
            ),
            is_done=True,
        )
        # Use the first candidate's value if available.
        if jv.candidates:
            return jv.candidates[0].value
        return jv.value
    except Exception:
        pass

    raise PatchError(f"Cannot parse patch string as JSON: {text!r}")


# ---------------------------------------------------------------------------
# Convenience re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "JsonPatchOp",
    "PatchDoc",
    "PatchError",
    "PatchPolicy",
    "PolicyViolationError",
    "apply_patch",
    "apply_patch_and_validate",
    "normalize_patches",
]
