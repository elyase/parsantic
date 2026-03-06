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
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from .api import ParseResult, coerce
from .config import resolve_model
from .patch import JsonPatchOp, PatchPolicy, apply_patch, normalize_patches
from .prompts import build_retry_prompt, build_update_prompt
from .provider_output import normalize_text_outputs
from .retry import RetryPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


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
    doc_after_patches
        The final patched document as a dict (after all patches).
    raw_text
        Raw LLM output from the last successful call.
    attempts
        Number of LLM calls made (1 means no retries were needed).
    """

    value: T
    patches: list[JsonPatchOp]
    doc_before: dict[str, Any]
    doc_after_patches: dict[str, Any]
    raw_text: str
    attempts: int

    @property
    def doc_after(self) -> dict[str, Any]:
        """Backward-compatible alias for ``doc_after_patches``."""
        return self.doc_after_patches


# ---------------------------------------------------------------------------
# Provider helpers (reuse extract's provider system)
# ---------------------------------------------------------------------------


def _create_provider(model: str | Any | None, provider_kwargs: dict[str, Any] | None) -> Any:
    """Create or pass through a provider."""
    resolved = resolve_model(model)
    if not isinstance(resolved, str):
        return resolved
    from .extract.providers.base import ProviderConfig
    from .extract.providers.factory import create_provider

    return create_provider(ProviderConfig(model_id=resolved, provider_kwargs=provider_kwargs or {}))


def _existing_to_dict(existing: dict[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(existing, BaseModel):
        return existing.model_dump(mode="json")
    return copy.deepcopy(existing)


# ---------------------------------------------------------------------------
# Core update loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _UpdateState:
    current_doc: dict[str, Any]
    all_patches: list[JsonPatchOp] = field(default_factory=list)
    last_raw: str = ""
    last_errors: list[dict[str, Any]] = field(default_factory=list)


def _build_attempt_prompt(
    *,
    attempt: int,
    state: _UpdateState,
    instruction: str,
    schema_text: str,
    policy: PatchPolicy,
) -> str:
    if attempt == 0:
        return build_update_prompt(state.current_doc, instruction, schema_text, policy)
    return build_retry_prompt(
        state.current_doc,
        state.last_errors,
        instruction,
        schema_text,
        policy,
    )


def _schema_text_for_target[T](target: type[T] | TypeAdapter[T]) -> str:
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    try:
        return json.dumps(adapter.json_schema(), indent=2)
    except Exception:
        logger.debug("Failed to generate schema text for target %r", target, exc_info=True)
        return "{}"


def _process_update_attempt[T](
    *,
    attempt: int,
    max_retries: int,
    state: _UpdateState,
    doc_before: dict[str, Any],
    raw: str,
    target: type[T] | TypeAdapter[T],
    policy: PatchPolicy,
) -> UpdateResult[T] | None:
    # Parse patches from messy LLM output
    try:
        patches = normalize_patches(raw)
    except Exception as exc:
        state.last_errors = [{"loc": (), "msg": f"Failed to normalize patches: {exc}"}]
        if attempt >= max_retries:
            raise
        return None

    try:
        patched = apply_patch(state.current_doc, patches, policy=policy)
        logger.debug("Applied %d patches", len(patches))
    except Exception as exc:
        state.last_errors = [{"loc": (), "msg": f"Patch application failed: {exc}"}]
        if attempt >= max_retries:
            raise
        return None

    state.all_patches.extend(patches)

    try:
        logger.debug("Validating patched document")
        result: ParseResult[T] = coerce(patched, target)
        return UpdateResult(
            value=result.value,
            patches=state.all_patches,
            doc_before=doc_before,
            doc_after_patches=patched,
            raw_text=state.last_raw,
            attempts=attempt + 1,
        )
    except ValidationError as exc:
        state.current_doc = patched
        state.last_errors = exc.errors()
        if attempt >= max_retries:
            raise
        return None
    except Exception as exc:
        state.current_doc = patched
        state.last_errors = [{"loc": (), "msg": str(exc)}]
        if attempt >= max_retries:
            raise
        return None


def _infer_one(provider: Any, prompt: str) -> str:
    outputs = normalize_text_outputs(
        provider.infer([prompt]),
        expected_count=1,
        context="provider.infer",
    )
    return outputs[0]


async def _ainfer_one(provider: Any, prompt: str) -> str:
    if hasattr(provider, "ainfer"):
        raw_outputs = await provider.ainfer([prompt])
        outputs = normalize_text_outputs(raw_outputs, expected_count=1, context="provider.ainfer")
    else:
        raw_outputs = await asyncio.to_thread(provider.infer, [prompt])
        outputs = normalize_text_outputs(raw_outputs, expected_count=1, context="provider.infer")
    return outputs[0]


def _run_update[T](
    doc: dict[str, Any],
    instruction: str,
    target: type[T] | TypeAdapter[T],
    schema_text: str,
    provider: Any,
    policy: PatchPolicy,
    policy_retry: RetryPolicy,
) -> UpdateResult[T]:
    """Synchronous update loop."""
    state = _UpdateState(current_doc=doc)

    for attempt in range(1 + policy_retry.max_retries):
        logger.debug("Update attempt %d/%d", attempt + 1, policy_retry.max_retries + 1)
        if attempt > 0:
            policy_retry.wait(attempt - 1)
        prompt = _build_attempt_prompt(
            attempt=attempt,
            state=state,
            instruction=instruction,
            schema_text=schema_text,
            policy=policy,
        )

        raw = _infer_one(provider, prompt)
        state.last_raw = raw

        result = _process_update_attempt(
            attempt=attempt,
            max_retries=policy_retry.max_retries,
            state=state,
            doc_before=doc,
            raw=raw,
            target=target,
            policy=policy,
        )
        if result is not None:
            return result

    # Defensive: _process_update_attempt always raises on the final attempt,
    # so this line is normally unreachable.
    raise ValueError(
        f"Update failed after {policy_retry.max_retries + 1} attempts"
    )  # pragma: no cover


async def _arun_update[T](
    doc: dict[str, Any],
    instruction: str,
    target: type[T] | TypeAdapter[T],
    schema_text: str,
    provider: Any,
    policy: PatchPolicy,
    policy_retry: RetryPolicy,
) -> UpdateResult[T]:
    """Async update loop."""
    state = _UpdateState(current_doc=doc)

    for attempt in range(1 + policy_retry.max_retries):
        if attempt > 0:
            await policy_retry.async_wait(attempt - 1)
        prompt = _build_attempt_prompt(
            attempt=attempt,
            state=state,
            instruction=instruction,
            schema_text=schema_text,
            policy=policy,
        )

        raw = await _ainfer_one(provider, prompt)
        state.last_raw = raw

        result = _process_update_attempt(
            attempt=attempt,
            max_retries=policy_retry.max_retries,
            state=state,
            doc_before=doc,
            raw=raw,
            target=target,
            policy=policy,
        )
        if result is not None:
            return result

    # Defensive: _process_update_attempt always raises on the final attempt,
    # so this line is normally unreachable.
    raise ValueError(
        f"Update failed after {policy_retry.max_retries + 1} attempts"
    )  # pragma: no cover


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
    retry: RetryPolicy | None = None,
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
    effective_retry = retry or RetryPolicy(max_retries=max_retries)
    schema_text = _schema_text_for_target(target)

    provider = _create_provider(model, provider_kwargs)

    return _run_update(
        doc=doc,
        instruction=instruction,
        target=target,
        schema_text=schema_text,
        provider=provider,
        policy=effective_policy,
        policy_retry=effective_retry,
    )


async def aupdate[T](
    existing: dict[str, Any] | BaseModel,
    instruction: str,
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    policy: PatchPolicy | None = None,
    max_retries: int = 2,
    retry: RetryPolicy | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> UpdateResult[T]:
    """Async version of :func:`update`."""
    doc = _existing_to_dict(existing)
    effective_policy = policy or PatchPolicy()
    effective_retry = retry or RetryPolicy(max_retries=max_retries)
    schema_text = _schema_text_for_target(target)

    provider = _create_provider(model, provider_kwargs)

    return await _arun_update(
        doc=doc,
        instruction=instruction,
        target=target,
        schema_text=schema_text,
        provider=provider,
        policy=effective_policy,
        policy_retry=effective_retry,
    )


__all__ = [
    "UpdateResult",
    "aupdate",
    "update",
]
