"""pydantic-ai integration for SAP parsing and patch-based repair.

This module provides output processors that combine SAP (Schema-Aligned Parsing)
with pydantic-ai's output handling and retry mechanisms.  It is **import-safe**:
importing the module always succeeds, but functions that require ``pydantic-ai``
raise a clear :class:`ImportError` when the library is not installed.

Pure utility functions (:func:`validation_error_paths`, :func:`slice_schema_for_paths`,
:func:`slice_doc_for_paths`, :func:`build_patch_prompt`) work without ``pydantic-ai``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .api import ParseResult, parse
from .coerce import CoerceOptions
from .json_pointer import build_json_pointer, escape_json_pointer_token, parse_json_pointer
from .jsonish import ParseOptions
from .patch import PatchPolicy, apply_patch, normalize_patches

# ---------------------------------------------------------------------------
# Import guard for pydantic-ai
# ---------------------------------------------------------------------------

try:
    import pydantic_ai  # noqa: F401
    from pydantic_ai.exceptions import ModelRetry  # noqa: F401

    _HAS_PYDANTIC_AI = True
except ImportError:
    _HAS_PYDANTIC_AI = False


def _check_pydantic_ai() -> None:
    """Raise ``ImportError`` if ``pydantic-ai`` is not installed."""
    if not _HAS_PYDANTIC_AI:
        raise ImportError(
            "parsantic.ai requires pydantic-ai. Install with: pip install pydantic-ai"
        )


# ---------------------------------------------------------------------------
# D1) Schema / doc slicing utilities (no pydantic-ai dependency)
# ---------------------------------------------------------------------------


def _escape_json_pointer_token(token: str) -> str:
    """Escape a single JSON Pointer reference token (RFC 6901)."""
    return escape_json_pointer_token(token)


def validation_error_paths(error: ValidationError) -> list[str]:
    """Convert Pydantic ``ValidationError`` loc tuples to JSON Pointer paths.

    Each ``loc`` tuple like ``('user', 'pets', 0, 'age')`` is converted to
    ``/user/pets/0/age`` with proper RFC 6901 escaping for ``~`` and ``/``.

    Returns a deduplicated list of paths in the order they first appear.
    """
    seen: set[str] = set()
    paths: list[str] = []
    for err in error.errors():
        loc = err.get("loc", ())
        pointer = build_json_pointer([str(part) for part in loc])
        if pointer not in seen:
            seen.add(pointer)
            paths.append(pointer)
    return paths


def _pointer_to_segments(path: str) -> list[str]:
    """Split a JSON Pointer string into unescaped segments."""
    if not path:
        return []
    try:
        return parse_json_pointer(path)
    except ValueError:
        return [path]


def _parent_paths(path: str) -> list[str]:
    """Return the path itself and all parent paths.

    For ``/user/pets/0/age`` returns:
    ``['/user/pets/0/age', '/user/pets/0', '/user/pets', '/user']``
    """
    segments = _pointer_to_segments(path)
    result: list[str] = []
    for i in range(len(segments), 0, -1):
        result.append(build_json_pointer(segments[:i]))
    return result


def slice_schema_for_paths(schema_text: str, paths: list[str]) -> str:
    """Extract relevant schema fragments for the given JSON Pointer paths.

    Uses a simple heuristic: for each path, include lines from the schema text
    that mention any segment of the path or its parents.  This keeps the prompt
    small while providing enough context for the model to understand the target
    structure.

    Parameters
    ----------
    schema_text
        The full JSON schema or Pydantic model schema as a string.
    paths
        JSON Pointer paths derived from validation errors.

    Returns
    -------
    str
        A (possibly smaller) schema string containing only relevant fragments.
    """
    if not paths:
        return schema_text

    # Collect all relevant keywords from paths and their parents
    keywords: set[str] = set()
    for path in paths:
        for parent in _parent_paths(path):
            for seg in _pointer_to_segments(parent):
                # Skip numeric segments (array indices)
                if not seg.isdigit():
                    keywords.add(seg)

    if not keywords:
        return schema_text

    # Try to parse as JSON schema and extract relevant parts
    try:
        schema = json.loads(schema_text)
        sliced = _slice_json_schema(schema, keywords)
        return json.dumps(sliced, indent=2)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: line-based filtering
    lines = schema_text.splitlines()
    relevant: list[str] = []
    for line in lines:
        line_lower = line.lower()
        if any(kw.lower() in line_lower for kw in keywords):
            relevant.append(line)

    return "\n".join(relevant) if relevant else schema_text


def _slice_json_schema(schema: dict[str, Any], keywords: set[str]) -> dict[str, Any]:
    """Extract relevant properties from a JSON Schema dict."""
    result: dict[str, Any] = {}

    # Always keep top-level metadata
    for key in ("title", "type", "$defs", "definitions"):
        if key in schema:
            result[key] = schema[key]

    # Filter properties to only those matching keywords
    if "properties" in schema:
        filtered_props: dict[str, Any] = {}
        for prop_name, prop_schema in schema["properties"].items():
            if prop_name in keywords:
                filtered_props[prop_name] = prop_schema
        if filtered_props:
            result["properties"] = filtered_props

    # Keep required list filtered to relevant properties
    if "required" in schema and "properties" in result:
        relevant_required = [r for r in schema["required"] if r in result.get("properties", {})]
        if relevant_required:
            result["required"] = relevant_required

    return result if result else schema


def slice_doc_for_paths(doc: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """Extract relevant document fragments for the given paths.

    Returns a minimal dict containing values at the specified paths and
    their parent contexts.  This keeps the retry prompt focused on the
    parts of the document that need fixing.

    Parameters
    ----------
    doc
        The full document dict.
    paths
        JSON Pointer paths derived from validation errors.

    Returns
    -------
    dict
        A minimal dict with only the relevant fragments.
    """
    if not paths:
        return doc

    result: dict[str, Any] = {}
    for path in paths:
        segments = _pointer_to_segments(path)
        if not segments:
            return doc

        # Walk the document to extract the value and its parent context
        _insert_at_path(result, segments, _get_at_path(doc, segments))

    return result


def _get_at_path(doc: Any, segments: list[str]) -> Any:
    """Get the value at a JSON Pointer path in a document."""
    current = doc
    for seg in segments:
        if isinstance(current, dict):
            if seg in current:
                current = current[seg]
            else:
                return None
        elif isinstance(current, list):
            try:
                idx = int(seg)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _insert_at_path(target: dict[str, Any], segments: list[str], value: Any) -> None:
    """Insert a value into a nested dict at the given path segments."""
    if not segments:
        return

    current: Any = target
    for i, seg in enumerate(segments[:-1]):
        next_seg = segments[i + 1]
        # Determine if the next container should be a list or dict
        is_next_index = next_seg.isdigit()

        if isinstance(current, dict):
            if seg not in current:
                current[seg] = [] if is_next_index else {}
            current = current[seg]
        elif isinstance(current, list):
            try:
                idx = int(seg)
                while len(current) <= idx:
                    current.append({} if not is_next_index else [])
                current = current[idx]
            except ValueError:
                return
        else:
            return

    # Set the final value
    last = segments[-1]
    if isinstance(current, dict):
        current[last] = value
    elif isinstance(current, list):
        try:
            idx = int(last)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# D4) Patch prompt builder (no pydantic-ai dependency)
# ---------------------------------------------------------------------------


def _json_dumps_for_prompt(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def build_patch_prompt(
    current_doc: dict[str, Any],
    validation_errors: list[dict[str, Any]],
    schema_text: str | None = None,
    *,
    doc_slicing: bool = True,
) -> str:
    """Build a prompt asking the model to fix validation errors via patches.

    The prompt follows the trustcall-inspired structure:

    - Current document (or relevant slice)
    - Validation errors
    - Schema fragment (if provided)
    - Instructions for producing JSON Patch operations

    Parameters
    ----------
    current_doc
        The current (invalid) document as a dict.
    validation_errors
        List of validation error dicts (from ``ValidationError.errors()``).
    schema_text
        Optional schema text to include for context.
    doc_slicing
        If ``True`` (default), only include relevant fragments of the document
        and schema based on the error paths.

    Returns
    -------
    str
        A prompt string suitable for sending to an LLM.
    """
    # Derive error paths for slicing
    error_paths: list[str] = []
    for err in validation_errors:
        loc = err.get("loc", ())
        pointer = build_json_pointer([str(part) for part in loc])
        if pointer not in error_paths:
            error_paths.append(pointer)

    # Prepare document fragment
    if doc_slicing and error_paths:
        doc_fragment = slice_doc_for_paths(current_doc, error_paths)
    else:
        doc_fragment = current_doc

    # Prepare schema fragment
    schema_section = ""
    if schema_text:
        if doc_slicing and error_paths:
            schema_fragment = slice_schema_for_paths(schema_text, error_paths)
        else:
            schema_fragment = schema_text
        schema_section = (
            f"\n## Target Schema (relevant fragment)\n```json\n{schema_fragment}\n```\n"
        )

    # Format errors
    error_lines: list[str] = []
    for err in validation_errors:
        loc = err.get("loc", ())
        msg = err.get("msg", "unknown error")
        err_type = err.get("type", "")
        path_str = " -> ".join(str(p) for p in loc) if loc else "(root)"
        line = f"- **{path_str}**: {msg}"
        if err_type:
            line += f" (type: {err_type})"
        error_lines.append(line)

    errors_text = "\n".join(error_lines)

    prompt = f"""The following JSON document has validation errors that need to be fixed.

## Current Document
```json
{_json_dumps_for_prompt(doc_fragment)}
```
{schema_section}
## Validation Errors
{errors_text}

## Instructions
Fix the validation errors by providing RFC 6902 JSON Patch operations.
Return a JSON object with:
- "json_doc_id": "doc"
- "planned_edits": a short description of what you will fix
- "patches": a list of patch operations

Each patch operation should have:
- "op": one of "replace", "add" (prefer "replace" for existing fields)
- "path": JSON Pointer to the field (e.g., "/user/age")
- "value": the corrected value

Ordering guidance:
1. "replace" operations first
2. "add" operations last (use "/-" to append to arrays)

Example patch item:
{{"op": "replace", "path": "/user/age", "value": 42}}"""

    return prompt


# ---------------------------------------------------------------------------
# D2) sap_text_output (requires pydantic-ai)
# ---------------------------------------------------------------------------


def sap_text_output[T](
    target: type[T],
    *,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
) -> Callable[[str], T]:
    """Create an output processor that uses SAP parsing for text output.

    The returned callable accepts raw text from a pydantic-ai model run and
    returns a validated instance of *target* by running it through the SAP
    parse pipeline (JSON-ish extraction, coercion, Pydantic validation).

    Parameters
    ----------
    target
        The Pydantic model class or type to parse into.
    parse_options
        Options for the JSON-ish parser.
    coerce_options
        Options for schema-aligned coercion.

    Returns
    -------
    Callable[[str], T]
        A processor function: ``(text: str) -> T``.

    Raises
    ------
    ImportError
        If ``pydantic-ai`` is not installed.
    """
    _check_pydantic_ai()

    def processor(text: str) -> T:
        result: ParseResult[T] = parse(
            text,
            target,
            parse_options=parse_options,
            coerce_options=coerce_options,
        )
        return result.value

    return processor


# ---------------------------------------------------------------------------
# D3) patch_repair_output (requires pydantic-ai)
# ---------------------------------------------------------------------------


def patch_repair_output[T](
    target: type[T],
    *,
    policy: PatchPolicy | None = None,
    max_attempts: int = 3,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
) -> Callable[[str], T]:
    """Create an output processor with patch-based repair loop.

    The returned callable:

    1. Parses the text with SAP into a Python dict.
    2. Validates against *target*.
    3. On validation failure, raises ``ModelRetry`` with a patch-repair prompt
       so that pydantic-ai's retry mechanism can ask the model to fix the
       errors using JSON Patch operations.

    Parameters
    ----------
    target
        The Pydantic model class to parse/validate into.
    policy
        Patch safety policy.  Defaults to :class:`PatchPolicy` defaults.
    max_attempts
        Maximum number of patch-repair attempts before giving up.
    parse_options
        Options for the JSON-ish parser.
    coerce_options
        Options for schema-aligned coercion.

    Returns
    -------
    Callable[[str], T]
        A processor function: ``(text: str) -> T``.

    Raises
    ------
    ImportError
        If ``pydantic-ai`` is not installed.
    """
    _check_pydantic_ai()

    effective_policy = policy or PatchPolicy()
    adapter: TypeAdapter[T] = TypeAdapter(target)

    @dataclass(slots=True)
    class _RepairState:
        attempts: int = 0
        prev_doc: dict[str, Any] | None = None

    state = _RepairState()

    # Pre-render schema text for prompts
    try:
        schema_text = json.dumps(adapter.json_schema(), indent=2)
    except Exception:
        schema_text = None

    def _get_model_retry() -> type:
        """Lazily import ModelRetry only when actually needed (validation failure)."""
        from pydantic_ai.exceptions import ModelRetry as _MR

        return _MR

    def processor(text: str) -> T:
        # Try to parse as patch operations first (for repair retries)
        current_doc: dict[str, Any] | None = None
        if state.attempts > 0 and state.prev_doc is not None:
            try:
                patches = normalize_patches(text)
                patched = apply_patch(state.prev_doc, patches, policy=effective_policy)
                validated = adapter.validate_python(patched)
                state.attempts = 0
                state.prev_doc = None
                return validated
            except Exception:
                pass

        # Parse the raw text with SAP
        try:
            result: ParseResult[T] = parse(
                text,
                target,
                parse_options=parse_options,
                coerce_options=coerce_options,
            )
            state.attempts = 0
            state.prev_doc = None
            return result.value
        except ValidationError:
            # SAP parsed successfully but validation failed
            # Try to extract the raw dict for patch repair
            try:
                raw_result = parse(
                    text,
                    dict,
                    parse_options=parse_options,
                )
                current_doc = raw_result.value if isinstance(raw_result.value, dict) else None
            except Exception:
                current_doc = None
        except Exception as exc:
            # SAP parsing itself failed
            try:
                raw_result = parse(text, dict, parse_options=parse_options)
                current_doc = raw_result.value if isinstance(raw_result.value, dict) else None
            except Exception:
                current_doc = None

            if current_doc is None:
                if state.attempts < max_attempts:
                    state.attempts += 1
                    MR = _get_model_retry()
                    raise MR(
                        f"Failed to parse output. Please return valid JSON matching the schema.\n"
                        f"Parse error: {exc}"
                    ) from exc
                raise

        if current_doc is None:
            raise ValueError("Failed to extract document from model output")

        # Try validating the raw dict
        try:
            validated = adapter.validate_python(current_doc)
            state.attempts = 0
            state.prev_doc = None
            return validated
        except ValidationError as val_err:
            if state.attempts >= max_attempts:
                state.attempts = 0
                state.prev_doc = None
                raise

            state.attempts += 1
            state.prev_doc = current_doc

            prompt = build_patch_prompt(
                current_doc,
                val_err.errors(),
                schema_text=schema_text,
            )
            MR = _get_model_retry()
            raise MR(prompt) from val_err

    return processor


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "build_patch_prompt",
    "patch_repair_output",
    "sap_text_output",
    "slice_doc_for_paths",
    "slice_schema_for_paths",
    "validation_error_paths",
]
