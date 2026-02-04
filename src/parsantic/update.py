"""High-level LLM-powered update for existing objects via JSON Patch.

Given an existing document (dict or Pydantic model), an instruction describing
what changed, and a target schema, :func:`update` calls an LLM to produce
RFC 6902 JSON Patch operations, applies them, validates the result, and
optionally retries on validation failure.

Requires the ``[ai]`` extra (``pip install "parsantic[ai]"``).

Example::

    from parsantic import update

    result = update(
        existing={"name": "Alex", "role": "Engineer", "skills": ["Python"]},
        instruction="Alex got promoted to Senior Engineer and picked up Rust.",
        target=User,
        model="openai:gpt-4o-mini",
    )
    result.value   # User(name='Alex', role='Senior Engineer', skills=['Python', 'Rust'])
    result.patches # [JsonPatchOp(op='replace', path='/role', ...), ...]
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from .api import ParseResult, coerce, parse
from .patch import JsonPatchOp, PatchPolicy, apply_patch, normalize_patches

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "openai:gpt-4o-mini"


@dataclass(frozen=True, slots=True)
class UpdateResult[T]:
    """Result of an :func:`update` call.

    Attributes
    ----------
    value
        The validated updated object.
    patches
        All patches applied (accumulated across retries).
    doc_before
        The original document as a dict (before any patches).
    doc_after
        The final patched document as a dict (after all patches).
    raw_text
        Raw LLM output from the last successful call.
    attempts
        Number of LLM calls made (1 means no retries were needed).
    """

    value: T
    patches: list[JsonPatchOp]
    doc_before: dict[str, Any]
    doc_after: dict[str, Any]
    raw_text: str
    attempts: int


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_update_prompt(
    doc: dict[str, Any],
    instruction: str,
    schema_text: str,
) -> str:
    """Build the initial prompt asking the LLM to produce patches."""
    return f"""You are updating a JSON document based on new information.

## Current Document
```json
{json.dumps(doc, indent=2, default=str)}
```

## Target Schema
```json
{schema_text}
```

## Instruction
{instruction}

## Rules
- Return ONLY a JSON array of RFC 6902 JSON Patch operations.
- Use "replace" for existing fields, "add" for new fields or array appends (use "/-" to append).
- Do NOT change fields unless the instruction implies it.
- Keep values JSON-serializable and conformant to the schema above.
- Order: "replace" operations first, then "add" operations.

Return the JSON array now:"""


def _build_retry_prompt(
    patched_doc: dict[str, Any],
    validation_errors: list[dict[str, Any]],
    instruction: str,
    schema_text: str,
) -> str:
    """Build a retry prompt when patches produced invalid output."""
    error_lines: list[str] = []
    for err in validation_errors:
        loc = err.get("loc", ())
        msg = err.get("msg", "unknown error")
        path_str = " -> ".join(str(p) for p in loc) if loc else "(root)"
        error_lines.append(f"- {path_str}: {msg}")
    errors_text = "\n".join(error_lines)

    return f"""The patches you produced resulted in validation errors.

## Current Document (after patches)
```json
{json.dumps(patched_doc, indent=2, default=str)}
```

## Validation Errors
{errors_text}

## Target Schema
```json
{schema_text}
```

## Original Instruction (for context)
{instruction}

Return ONLY a JSON array of additional RFC 6902 JSON Patch operations to fix these errors:"""


# ---------------------------------------------------------------------------
# Provider helpers (reuse extract's provider system)
# ---------------------------------------------------------------------------


def _resolve_model(model: str | Any | None) -> str | Any:
    if model is not None:
        return model
    return os.environ.get("PARSANTIC_MODEL", _DEFAULT_MODEL)


def _create_provider(model: str | Any | None, provider_kwargs: dict[str, Any] | None):
    """Create or pass through a provider."""
    resolved = _resolve_model(model)
    if not isinstance(resolved, str):
        return resolved
    from .extract.providers.base import ProviderConfig
    from .extract.providers.factory import create_provider

    return create_provider(ProviderConfig(model_id=resolved, provider_kwargs=provider_kwargs or {}))


def _existing_to_dict(existing: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(existing, BaseModel):
        return existing.model_dump(mode="json")
    return dict(existing)


# ---------------------------------------------------------------------------
# Core update loop
# ---------------------------------------------------------------------------


def _run_update[T](
    doc: dict[str, Any],
    instruction: str,
    target: type[T] | TypeAdapter[T],
    adapter: TypeAdapter[T],
    schema_text: str,
    provider: Any,
    policy: PatchPolicy,
    max_retries: int,
) -> UpdateResult[T]:
    """Synchronous update loop."""
    all_patches: list[JsonPatchOp] = []
    current_doc = doc
    last_raw = ""

    for attempt in range(1 + max_retries):
        # Build prompt
        if attempt == 0:
            prompt = _build_update_prompt(current_doc, instruction, schema_text)
        else:
            prompt = _build_retry_prompt(
                current_doc, last_errors, instruction, schema_text  # noqa: F821
            )

        # Call LLM
        outputs = provider.infer([prompt])
        raw = outputs[0] if outputs else ""
        last_raw = raw

        # Parse patches from messy LLM output
        try:
            parsed = parse(raw, list).value
        except Exception:
            # Fallback: try normalize_patches directly on the raw text
            try:
                parsed = raw
            except Exception:
                if attempt < max_retries:
                    last_errors = [{"loc": (), "msg": f"Failed to parse LLM output as patch list: {raw[:200]}"}]  # noqa: F841
                    continue
                raise ValueError(f"Failed to parse LLM output as patches after {attempt + 1} attempts: {raw[:200]}")

        try:
            patches = normalize_patches(parsed)
        except Exception as exc:
            if attempt < max_retries:
                last_errors = [{"loc": (), "msg": f"Failed to normalize patches: {exc}"}]  # noqa: F841
                continue
            raise

        # Apply patches
        try:
            patched = apply_patch(current_doc, patches, policy=policy)
        except Exception as exc:
            if attempt < max_retries:
                last_errors = [{"loc": (), "msg": f"Patch application failed: {exc}"}]  # noqa: F841
                continue
            raise

        all_patches.extend(patches)

        # Validate with coerce (handles type mismatches locally)
        try:
            result: ParseResult[T] = coerce(patched, target)
            return UpdateResult(
                value=result.value,
                patches=all_patches,
                doc_before=doc,
                doc_after=patched,
                raw_text=last_raw,
                attempts=attempt + 1,
            )
        except (ValidationError, Exception) as exc:
            current_doc = patched
            if isinstance(exc, ValidationError):
                last_errors = exc.errors()  # noqa: F841
            else:
                last_errors = [{"loc": (), "msg": str(exc)}]  # noqa: F841
            if attempt >= max_retries:
                raise

    # Should not reach here, but just in case
    raise ValueError(f"Update failed after {max_retries + 1} attempts")


async def _arun_update[T](
    doc: dict[str, Any],
    instruction: str,
    target: type[T] | TypeAdapter[T],
    adapter: TypeAdapter[T],
    schema_text: str,
    provider: Any,
    policy: PatchPolicy,
    max_retries: int,
) -> UpdateResult[T]:
    """Async update loop."""
    all_patches: list[JsonPatchOp] = []
    current_doc = doc
    last_raw = ""

    for attempt in range(1 + max_retries):
        # Build prompt
        if attempt == 0:
            prompt = _build_update_prompt(current_doc, instruction, schema_text)
        else:
            prompt = _build_retry_prompt(
                current_doc, last_errors, instruction, schema_text  # noqa: F821
            )

        # Call LLM (async)
        if hasattr(provider, "ainfer"):
            outputs = await provider.ainfer([prompt])
        else:
            outputs = await asyncio.to_thread(provider.infer, [prompt])
        raw = outputs[0] if outputs else ""
        last_raw = raw

        # Parse patches from messy LLM output
        try:
            parsed = parse(raw, list).value
        except Exception:
            try:
                parsed = raw
            except Exception:
                if attempt < max_retries:
                    last_errors = [{"loc": (), "msg": f"Failed to parse LLM output as patch list: {raw[:200]}"}]  # noqa: F841
                    continue
                raise ValueError(f"Failed to parse LLM output as patches after {attempt + 1} attempts: {raw[:200]}")

        try:
            patches = normalize_patches(parsed)
        except Exception as exc:
            if attempt < max_retries:
                last_errors = [{"loc": (), "msg": f"Failed to normalize patches: {exc}"}]  # noqa: F841
                continue
            raise

        # Apply patches
        try:
            patched = apply_patch(current_doc, patches, policy=policy)
        except Exception as exc:
            if attempt < max_retries:
                last_errors = [{"loc": (), "msg": f"Patch application failed: {exc}"}]  # noqa: F841
                continue
            raise

        all_patches.extend(patches)

        # Validate with coerce
        try:
            result: ParseResult[T] = coerce(patched, target)
            return UpdateResult(
                value=result.value,
                patches=all_patches,
                doc_before=doc,
                doc_after=patched,
                raw_text=last_raw,
                attempts=attempt + 1,
            )
        except (ValidationError, Exception) as exc:
            current_doc = patched
            if isinstance(exc, ValidationError):
                last_errors = exc.errors()  # noqa: F841
            else:
                last_errors = [{"loc": (), "msg": str(exc)}]  # noqa: F841
            if attempt >= max_retries:
                raise

    raise ValueError(f"Update failed after {max_retries + 1} attempts")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update[T](
    existing: dict[str, Any] | BaseModel,
    instruction: str,
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    policy: PatchPolicy | None = None,
    max_retries: int = 2,
    provider_kwargs: dict[str, Any] | None = None,
) -> UpdateResult[T]:
    """Update an existing object with new information using an LLM.

    Calls an LLM to generate RFC 6902 JSON Patch operations, applies them
    to *existing*, validates against *target*, and retries on failure.

    Parameters
    ----------
    existing
        The current object as a dict or Pydantic model instance.
    instruction
        Natural language description of what changed.
    target
        The Pydantic model class or TypeAdapter to validate against.
    model
        Model string (e.g. ``"openai:gpt-4o-mini"``) or a provider instance.
    policy
        Patch safety policy. Defaults to no ``remove``, max 50 ops.
    max_retries
        Maximum number of retry attempts on validation failure.
    provider_kwargs
        Extra kwargs passed to the provider constructor.

    Returns
    -------
    UpdateResult[T]
        The validated updated object with patch metadata.
    """
    doc = _existing_to_dict(existing)
    effective_policy = policy or PatchPolicy()
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)

    try:
        schema_text = json.dumps(adapter.json_schema(), indent=2)
    except Exception:
        schema_text = "{}"

    provider = _create_provider(model, provider_kwargs)

    return _run_update(
        doc=doc,
        instruction=instruction,
        target=target,
        adapter=adapter,
        schema_text=schema_text,
        provider=provider,
        policy=effective_policy,
        max_retries=max_retries,
    )


async def aupdate[T](
    existing: dict[str, Any] | BaseModel,
    instruction: str,
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    policy: PatchPolicy | None = None,
    max_retries: int = 2,
    provider_kwargs: dict[str, Any] | None = None,
) -> UpdateResult[T]:
    """Async version of :func:`update`."""
    doc = _existing_to_dict(existing)
    effective_policy = policy or PatchPolicy()
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)

    try:
        schema_text = json.dumps(adapter.json_schema(), indent=2)
    except Exception:
        schema_text = "{}"

    provider = _create_provider(model, provider_kwargs)

    return await _arun_update(
        doc=doc,
        instruction=instruction,
        target=target,
        adapter=adapter,
        schema_text=schema_text,
        provider=provider,
        policy=effective_policy,
        max_retries=max_retries,
    )


__all__ = [
    "UpdateResult",
    "aupdate",
    "update",
]
