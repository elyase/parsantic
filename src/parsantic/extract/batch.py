from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic_core import to_jsonable_python

from .options import ExtractOptions
from .pipeline import (
    Extractor,
    _align_evidence,
    _build_extraction_context,
    _default_merged_value,
    _parse_with_repair,
    _render_prompt,
)
from .prompt import Prompt
from .providers.base import BaseProvider, InferenceRequest, SupportsMediaInfer
from .types import Document, ExtractResult, FieldEvidence


@runtime_checkable
class SupportsBatchInfer(Protocol):
    def submit_batch(self, requests: Sequence[InferenceRequest], **kwargs: Any) -> str:
        """Submit a batch job, return a batch_id."""

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """Check batch status."""

    def retrieve_batch(self, batch_id: str) -> Sequence[str]:
        """Retrieve completed batch results."""


@dataclass(slots=True)
class BatchStatus:
    batch_id: str
    state: Literal["pending", "in_progress", "completed", "failed", "expired"]
    completed_count: int = 0
    total_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class BatchResult[T]:
    results: list[ExtractResult[T]]
    batch_id: str | None
    used_batch_api: bool
    total_documents: int


def _normalize_batch_outputs(outputs: Any, *, expected_count: int) -> list[str]:
    if outputs is None:
        raise TypeError("provider.retrieve_batch returned None, expected a list of strings")
    if isinstance(outputs, str):
        normalized: list[Any] = [outputs]
    elif isinstance(outputs, (dict, set, frozenset)):
        raise TypeError(
            f"provider.retrieve_batch returned {type(outputs).__name__}, expected a list of strings"
        )
    else:
        normalized = list(outputs)

    for idx, item in enumerate(normalized):
        if not isinstance(item, str):
            raise TypeError(
                f"provider.retrieve_batch output at index {idx} must be str, got {type(item).__name__}"
            )

    if len(normalized) != expected_count:
        raise ValueError(
            f"provider.retrieve_batch returned {len(normalized)} outputs for {expected_count} documents"
        )
    return normalized


def _coerce_batch_status(value: BatchStatus | Mapping[str, Any], *, batch_id: str) -> BatchStatus:
    if isinstance(value, BatchStatus):
        return value
    if isinstance(value, Mapping):
        state = value.get("state")
        if state not in {"pending", "in_progress", "completed", "failed", "expired"}:
            raise TypeError("provider.poll_batch returned an invalid state")
        return BatchStatus(
            batch_id=str(value.get("batch_id") or batch_id),
            state=state,
            completed_count=int(value.get("completed_count", 0)),
            total_count=int(value.get("total_count", 0)),
            error=str(value["error"]) if value.get("error") is not None else None,
        )
    raise TypeError("provider.poll_batch must return BatchStatus or a mapping")


def _batch_kwargs(ctx: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if ctx.use_native_schema and ctx.target_type is not None:
        kwargs["target_type"] = ctx.target_type
        kwargs["structured_output"] = ctx.opts.structured_output
    return kwargs


def _build_batch_requests(ctx: Any) -> list[InferenceRequest]:
    requests: list[InferenceRequest] = []
    for doc_idx, doc in enumerate(ctx.documents):
        prompt_text = _render_prompt(
            ctx.prompt_obj,
            schema_text=ctx.schema_text,
            examples=ctx.normalized_examples,
            question=doc.text or "Extract structured data from this document.",
            format_handler=ctx.format_handler,
            additional_context=doc.additional_context,
            output_kind=ctx.output_kind,
            native_mode=ctx.use_native_schema,
        )
        requests.append(
            InferenceRequest(
                prompt=prompt_text,
                attachments=doc.attachments,
                document_id=doc.document_id,
                document_index=doc_idx,
            )
        )
    return requests


def _parse_batch_output[T](
    ctx: Any, doc: Document, raw_output: str, target: type[T]
) -> ExtractResult[T]:
    cleaned = raw_output.strip()
    flags: tuple[str, ...] = ()
    score = 0

    if cleaned:
        parsed = _parse_with_repair(
            raw_output,
            target,
            parse_options=None,
            coerce_options=None,
            repair=ctx.opts.repair,
            allow_partial=False,
        )
        value = parsed.value
        flags = tuple(sorted(parsed.flags))
        score = parsed.score
    else:
        value = ctx.adapter.validate(_default_merged_value(ctx.output_kind))

    value_json = to_jsonable_python(value)
    evidence: list[FieldEvidence] = _align_evidence(
        doc.text,
        value_json,
        tokenizer=ctx.opts.tokenizer,
        alignment=ctx.opts.alignment,
        offset=0,
    )

    return ExtractResult(
        value=value,
        document_id=doc.document_id,
        raw_text=raw_output or None,
        flags=flags,
        score=score,
        evidence=evidence,
    )


def _check_batch_status(status: BatchStatus, batch_id: str) -> bool:
    """Return True if completed, raise on terminal states."""
    if status.state == "completed":
        return True
    if status.state in {"failed", "expired"}:
        detail = f": {status.error}" if status.error else ""
        raise RuntimeError(f"Batch {batch_id} ended in state '{status.state}'{detail}")
    return False


def _compute_sleep(
    poll_interval: float,
    timeout: float | None,
    start: float,
    batch_id: str,
) -> float:
    if timeout is not None and time.monotonic() - start >= timeout:
        raise TimeoutError(f"Timed out waiting for batch {batch_id} to complete")
    sleep_for = poll_interval
    if timeout is not None:
        remaining = timeout - (time.monotonic() - start)
        sleep_for = max(0.0, min(sleep_for, remaining))
    if sleep_for <= 0:
        raise TimeoutError(f"Timed out waiting for batch {batch_id} to complete")
    return sleep_for


def _wait_for_batch_completion(
    provider: SupportsBatchInfer,
    batch_id: str,
    *,
    poll_interval: float,
    timeout: float | None,
) -> BatchStatus:
    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")

    start = time.monotonic()
    while True:
        status = _coerce_batch_status(provider.poll_batch(batch_id), batch_id=batch_id)
        if _check_batch_status(status, batch_id):
            return status
        time.sleep(_compute_sleep(poll_interval, timeout, start, batch_id))


async def _await_batch_completion(
    provider: SupportsBatchInfer,
    batch_id: str,
    *,
    poll_interval: float,
    timeout: float | None,
) -> BatchStatus:
    if poll_interval <= 0:
        raise ValueError("poll_interval must be > 0")

    start = time.monotonic()
    while True:
        polled = await asyncio.to_thread(provider.poll_batch, batch_id)
        status = _coerce_batch_status(polled, batch_id=batch_id)
        if _check_batch_status(status, batch_id):
            return status
        await asyncio.sleep(_compute_sleep(poll_interval, timeout, start, batch_id))


def extract_batch[T](
    documents: Sequence[Document],
    target: type[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    poll_interval: float = 30.0,
    timeout: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> BatchResult[T]:
    if not documents:
        return BatchResult(results=[], batch_id=None, used_batch_api=False, total_documents=0)

    ctx = _build_extraction_context(
        documents,
        target,
        model=model,
        prompt=prompt,
        options=options,
        provider_kwargs=provider_kwargs,
    )

    provider: BaseProvider | SupportsMediaInfer | SupportsBatchInfer = ctx.provider
    if not isinstance(provider, SupportsBatchInfer):
        extractor = Extractor(
            model=provider,
            prompt=prompt,
            options=options,
            provider_kwargs=provider_kwargs,
        )
        fallback_results: list[ExtractResult[T]] = []
        for doc in ctx.documents:
            result = extractor.extract(doc, target)
            if isinstance(result, list):
                raise TypeError("extract() returned list for a single document")
            fallback_results.append(result)
        return BatchResult(
            results=fallback_results,
            batch_id=None,
            used_batch_api=False,
            total_documents=len(ctx.documents),
        )

    requests = _build_batch_requests(ctx)
    batch_id = provider.submit_batch(requests, **_batch_kwargs(ctx))
    _wait_for_batch_completion(provider, batch_id, poll_interval=poll_interval, timeout=timeout)
    raw_outputs = _normalize_batch_outputs(
        provider.retrieve_batch(batch_id),
        expected_count=len(ctx.documents),
    )

    results = [
        _parse_batch_output(ctx, doc, raw, target)
        for doc, raw in zip(ctx.documents, raw_outputs, strict=True)
    ]
    return BatchResult(
        results=results,
        batch_id=batch_id,
        used_batch_api=True,
        total_documents=len(ctx.documents),
    )


async def aextract_batch[T](
    documents: Sequence[Document],
    target: type[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    poll_interval: float = 30.0,
    timeout: float | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> BatchResult[T]:
    if not documents:
        return BatchResult(results=[], batch_id=None, used_batch_api=False, total_documents=0)

    ctx = _build_extraction_context(
        documents,
        target,
        model=model,
        prompt=prompt,
        options=options,
        provider_kwargs=provider_kwargs,
    )

    provider: BaseProvider | SupportsMediaInfer | SupportsBatchInfer = ctx.provider
    if not isinstance(provider, SupportsBatchInfer):
        extractor = Extractor(
            model=provider,
            prompt=prompt,
            options=options,
            provider_kwargs=provider_kwargs,
        )

        async def _extract_one(doc: Document) -> ExtractResult[T]:
            result = await extractor.aextract(doc, target)
            if isinstance(result, list):
                raise TypeError("aextract() returned list for a single document")
            return result

        fallback_results = list(await asyncio.gather(*[_extract_one(d) for d in ctx.documents]))
        return BatchResult(
            results=fallback_results,
            batch_id=None,
            used_batch_api=False,
            total_documents=len(ctx.documents),
        )

    requests = _build_batch_requests(ctx)
    batch_id = await asyncio.to_thread(provider.submit_batch, requests, **_batch_kwargs(ctx))
    await _await_batch_completion(provider, batch_id, poll_interval=poll_interval, timeout=timeout)
    raw_outputs = _normalize_batch_outputs(
        await asyncio.to_thread(provider.retrieve_batch, batch_id),
        expected_count=len(ctx.documents),
    )

    results = [
        _parse_batch_output(ctx, doc, raw, target)
        for doc, raw in zip(ctx.documents, raw_outputs, strict=True)
    ]
    return BatchResult(
        results=results,
        batch_id=batch_id,
        used_batch_api=True,
        total_documents=len(ctx.documents),
    )
