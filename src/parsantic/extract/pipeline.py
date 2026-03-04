from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from parsantic.api import ParseResult, parse
from parsantic.coerce import CoerceOptions
from parsantic.config import resolve_model
from parsantic.json_pointer import escape_json_pointer_token
from parsantic.jsonish import ParseOptions
from parsantic.provider_output import normalize_text_outputs

from .alignment import AlignmentOptions, align_value_to_text, merge_evidence
from .chunking import TextChunk, iter_chunks
from .formatting import FormatHandler
from .media.chunking import MediaChunk, chunk_attachments, needs_media
from .options import ExtractOptions
from .prompt import Example, Prompt, PromptValidationLevel
from .providers.base import (
    InferenceRequest,
    ProviderConfig,
    SupportsAsyncMediaInfer,
    SupportsMediaInfer,
)
from .providers.factory import create_provider
from .schema import PydanticSchemaAdapter
from .tokenizer import Tokenizer, TokenizerName, get_tokenizer
from .types import (
    AlignmentStatus,
    ChunkDebug,
    Document,
    ExtractDebug,
    ExtractResult,
    FieldEvidence,
    MergeConflict,
)

logger = logging.getLogger(__name__)


_DEFAULT_DESCRIPTION = "Extract structured data that matches the provided schema."


def _iter_leaf_values(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, dict):
        for key, val in value.items():
            next_path = f"{path}/{escape_json_pointer_token(str(key))}"
            yield from _iter_leaf_values(val, next_path)
        return
    if isinstance(value, list):
        for idx, val in enumerate(value):
            next_path = f"{path}/{idx}"
            yield from _iter_leaf_values(val, next_path)
        return
    if isinstance(value, (str, int, float, bool)):
        yield (path or "/", str(value))


def _normalize_prompt(prompt: Prompt | str | None) -> Prompt:
    if prompt is None:
        return Prompt(description=_DEFAULT_DESCRIPTION)
    if isinstance(prompt, str):
        return Prompt(description=prompt)
    return prompt


def _schema_root_kind(schema: dict[str, Any]) -> Literal["object", "array"] | None:
    schema_type = schema.get("type")
    if schema_type == "array":
        return "array"
    if schema_type == "object":
        return "object"
    return None


def _validate_examples(
    prompt: Prompt,
    adapter: PydanticSchemaAdapter,
    *,
    tokenizer: TokenizerName | Tokenizer | None,
    alignment: AlignmentOptions,
    level: PromptValidationLevel,
) -> None:
    if level == PromptValidationLevel.OFF:
        return
    tok = get_tokenizer(tokenizer)
    errors: list[str] = []
    for idx, ex in enumerate(prompt.examples):
        try:
            validated = adapter.validate(ex.output)
            dumped = adapter.dump(validated)
        # Keep broad behavior to continue collecting prompt validation issues.
        except Exception as exc:  # pragma: no cover - surfaced in tests
            errors.append(f"example#{idx} failed schema validation: {exc}")
            continue
        tokenized_source = tok.tokenize(ex.text)
        for path, text in _iter_leaf_values(dumped):
            evidence = align_value_to_text(
                ex.text,
                path,
                text,
                tokenizer=tok,
                options=alignment,
                tokenized_source=tokenized_source,
            )
            if evidence.char_interval is None:
                errors.append(f"example#{idx} path {path} value '{text}' not found in example text")
    if errors and level == PromptValidationLevel.ERROR:
        raise ValueError("Prompt validation failed: " + "; ".join(errors))
    if errors and level == PromptValidationLevel.WARNING:
        import warnings

        warnings.warn("Prompt validation warnings: " + "; ".join(errors), stacklevel=2)


def _render_prompt(
    prompt: Prompt,
    *,
    schema_text: str | None,
    examples: Sequence[Example],
    question: str,
    format_handler: FormatHandler,
    additional_context: str | None,
    output_kind: Literal["object", "array"] | None = None,
    native_mode: bool = False,
) -> str:
    lines: list[str] = [prompt.description.strip(), ""]
    if additional_context:
        lines.append(additional_context)
        lines.append("")

    if not native_mode:
        # E5: Add explicit output format instructions
        fmt = format_handler.options.format.lower() if format_handler.options else "json"
        if fmt == "json":
            expected_kind = output_kind or "object"
            if format_handler.options and format_handler.options.wrapper_key:
                expected_kind = "object"
            lines.append(
                f"Output a single JSON {expected_kind}. "
                "Do not include any surrounding prose or commentary."
            )
            if format_handler.options and format_handler.options.wrapper_key:
                lines.append(
                    f'Wrap the result list under the key "{format_handler.options.wrapper_key}".'
                )
            lines.append("")
        elif fmt == "yaml":
            lines.append("Output YAML only. Do not include any surrounding prose or commentary.")
            if format_handler.options and format_handler.options.wrapper_key:
                lines.append(
                    f'Wrap the result list under the key "{format_handler.options.wrapper_key}".'
                )
            lines.append("")

        if schema_text:
            lines.append("Schema:")
            lines.append(schema_text)
            lines.append("")

    if examples:
        if native_mode:
            for ex in examples:
                formatted = format_handler.format_example(ex.output)
                lines.append("Example:")
                lines.append(f"{ex.text} \u2192 {formatted}")
                lines.append("")
        else:
            lines.append("Examples")
            for ex in examples:
                formatted = format_handler.format_example(ex.output)
                lines.append(f"Q: {ex.text}")
                lines.append("A: " + formatted)
                lines.append("")

    if native_mode:
        lines.append("---")
        lines.append(question)
    else:
        lines.append(f"Q: {question}")
        lines.append("A:")
    return "\n".join(lines)


def _merge_values(
    base: Any,
    other: Any,
    *,
    strategy: Literal["first_wins", "last_wins", "prefer_non_null"] = "first_wins",
    conflicts: list[MergeConflict] | None = None,
    path: str = "",
    page_index: int | None = None,
) -> Any:
    if base is None:
        return other
    if other is None:
        return base
    if isinstance(base, list) and isinstance(other, list):
        merged = list(base)
        for item in other:
            if item not in merged:
                merged.append(item)
        return merged
    if isinstance(base, dict) and isinstance(other, dict):
        merged = dict(base)
        for key, val in other.items():
            child_path = f"{path}/{escape_json_pointer_token(str(key))}"
            if key in merged:
                merged[key] = _merge_values(
                    merged[key],
                    val,
                    strategy=strategy,
                    conflicts=conflicts,
                    path=child_path,
                    page_index=page_index,
                )
            else:
                merged[key] = val
        return merged
    # Scalar conflict: both base and other are non-None scalars with different values
    if base != other and conflicts is not None:
        conflicts.append(
            MergeConflict(
                path=path or "/",
                existing_preview=str(base)[:80],
                incoming_preview=str(other)[:80],
                page_index=page_index,
            )
        )
    if strategy == "last_wins":
        return other
    if strategy == "prefer_non_null":
        if base in (None, ""):
            return other
        if other in (None, ""):
            return base
    return base


def _align_evidence(
    source_text: str,
    value: Any,
    *,
    tokenizer: TokenizerName | Tokenizer | None,
    alignment: AlignmentOptions,
    offset: int,
) -> list[FieldEvidence]:
    tok = get_tokenizer(tokenizer)
    tokenized_source = tok.tokenize(source_text)
    evidence: list[FieldEvidence] = []
    for path, text in _iter_leaf_values(value):
        ev = align_value_to_text(
            source_text,
            path,
            text,
            tokenizer=tok,
            options=alignment,
            tokenized_source=tokenized_source,
        )
        if ev.char_interval:
            start, end = ev.char_interval
            ev = FieldEvidence(
                path=ev.path,
                value_preview=ev.value_preview,
                char_interval=(start + offset, end + offset),
                # token_interval stays chunk-relative: offsetting would require
                # retokenising the full document (token boundaries differ).
                token_interval=ev.token_interval,
                alignment_status=ev.alignment_status,
            )
        evidence.append(ev)
    return evidence


@dataclass(slots=True)
class _ExtractionContext:
    """Shared setup computed once for both sync and async extraction paths."""

    documents: list[Document]
    prompt_obj: Prompt
    opts: ExtractOptions
    adapter: PydanticSchemaAdapter[Any]
    format_handler: FormatHandler
    schema_text: str | None
    output_kind: Literal["object", "array"] | None
    normalized_examples: list[Example]
    provider: Any
    target_type: type[Any] | None = None
    use_native_schema: bool = False


def _build_extraction_context(
    text_or_documents: str | Document | Iterable[Document],
    target: type[Any] | TypeAdapter[Any],
    *,
    model: str | Any | None,
    prompt: Prompt | str | None,
    options: ExtractOptions | None,
    provider_kwargs: dict[str, Any] | None,
) -> _ExtractionContext:
    """Shared setup logic for both sync and async extraction."""
    if isinstance(text_or_documents, str):
        documents = [Document(text=text_or_documents)]
    elif isinstance(text_or_documents, Document):
        documents = [text_or_documents]
    else:
        documents = list(text_or_documents)

    prompt_obj = _normalize_prompt(prompt)
    opts = options or ExtractOptions()
    adapter = PydanticSchemaAdapter.from_target(target)
    format_handler = FormatHandler(opts.format)
    schema_obj = adapter.adapter.json_schema()

    _validate_examples(
        prompt_obj,
        adapter,
        tokenizer=opts.tokenizer,
        alignment=opts.alignment,
        level=opts.prompt_validation,
    )

    schema_text: str | None = None
    if prompt_obj.include_schema:
        if opts.schema_mode == "compact":
            schema_text = json.dumps(schema_obj, ensure_ascii=False)
        else:
            schema_text = json.dumps(schema_obj, indent=2, ensure_ascii=False)
    output_kind = _schema_root_kind(schema_obj)

    normalized_examples = [
        Example(text=ex.text, output=adapter.dump(adapter.validate(ex.output)))
        for ex in prompt_obj.examples
    ]

    resolved_model = resolve_model(model)
    logger.debug(
        "Building extraction context: model=%s, documents=%d",
        resolved_model if isinstance(resolved_model, str) else type(resolved_model).__name__,
        len(documents),
    )
    provider = (
        resolved_model
        if not isinstance(resolved_model, str)
        else create_provider(
            ProviderConfig(model_id=resolved_model, provider_kwargs=provider_kwargs or {})
        )
    )

    # Determine if we should use native structured output
    structured_mode = opts.structured_output
    supports = getattr(provider, "supports_native_structured_output", None)
    provider_supports = callable(supports) and supports()

    use_native = False
    if structured_mode == "native":
        if provider_supports:
            use_native = True
        else:
            logger.warning(
                "structured_output='native' requested but provider %s does not "
                "support native structured output; falling back to prompt mode",
                type(provider).__name__,
            )
    elif structured_mode == "auto":
        use_native = provider_supports

    # Resolve target_type for native structured output
    target_type: type[Any] | None = None
    if use_native:
        if isinstance(target, type):
            target_type = target
        elif isinstance(target, TypeAdapter):
            target_type = None  # Can't resolve type from TypeAdapter reliably
        if target_type is None:
            use_native = False

    if use_native:
        logger.debug("Native structured output enabled for %s", type(provider).__name__)
        schema_text = None  # redundant when provider constrains output via its own schema

    return _ExtractionContext(
        documents=documents,
        prompt_obj=prompt_obj,
        opts=opts,
        adapter=adapter,
        format_handler=format_handler,
        schema_text=schema_text,
        output_kind=output_kind,
        normalized_examples=normalized_examples,
        provider=provider,
        target_type=target_type,
        use_native_schema=use_native,
    )


@dataclass(slots=True)
class Extractor:
    model: str | Any | None = None
    prompt: Prompt | str | None = None
    options: ExtractOptions | None = None
    provider_kwargs: dict[str, Any] | None = None

    def __enter__(self) -> Extractor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    async def __aenter__(self) -> Extractor:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def extract[T](
        self,
        text: str | Document | Iterable[Document],
        target: type[T] | TypeAdapter[T],
        *,
        parse_options: ParseOptions | None = None,
        coerce_options: CoerceOptions | None = None,
        debug: bool = False,
    ) -> ExtractResult[T]:
        return extract(
            text,
            target,
            model=self.model,
            prompt=self.prompt,
            options=self.options,
            provider_kwargs=self.provider_kwargs,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

    async def aextract[T](
        self,
        text_or_documents: str | Document | Iterable[Document],
        target: type[T] | TypeAdapter[T],
        *,
        parse_options: ParseOptions | None = None,
        coerce_options: CoerceOptions | None = None,
        debug: bool = False,
    ) -> ExtractResult[T] | list[ExtractResult[T]]:
        return await aextract(
            text_or_documents,
            target,
            model=self.model,
            prompt=self.prompt,
            options=self.options,
            provider_kwargs=self.provider_kwargs,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )


def extract[T](
    text_or_documents: str | Document | Iterable[Document],
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
    debug: bool = False,
) -> ExtractResult[T] | list[ExtractResult[T]]:
    results = list(
        extract_iter(
            text_or_documents,
            target,
            model=model,
            prompt=prompt,
            options=options,
            provider_kwargs=provider_kwargs,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )
    )
    if isinstance(text_or_documents, (str, Document)):
        return results[0]
    return results


def _infer_batch(
    provider: Any,
    prompts: Sequence[str],
    batch_length: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Call provider.infer in batches of *batch_length* and concatenate results."""
    logger.debug("Batched inference: %d prompts, batch_length=%d", len(prompts), batch_length)
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_length):
        batch = prompts[i : i + batch_length]
        outputs = normalize_text_outputs(
            provider.infer(batch, **infer_kwargs),
            expected_count=len(batch),
            context="provider.infer",
        )
        all_outputs.extend(outputs)
    return all_outputs


def _infer_batch_parallel(
    provider: Any,
    prompts: Sequence[str],
    batch_length: int,
    max_workers: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Call provider.infer in batches of *batch_length* with ThreadPoolExecutor."""
    batches: list[Sequence[str]] = []
    batch_length = max(1, batch_length)
    for i in range(0, len(prompts), batch_length):
        batches.append(prompts[i : i + batch_length])
    logger.debug(
        "Parallel batched inference: %d batches, max_workers=%d", len(batches), max_workers
    )

    batch_results: dict[int, list[str]] = {}
    max_workers = max(1, max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(provider.infer, batch, **infer_kwargs): idx
            for idx, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch_result = normalize_text_outputs(
                future.result(),
                expected_count=len(batches[idx]),
                context="provider.infer",
            )
            batch_results[idx] = batch_result

    all_outputs: list[str] = []
    for idx in range(len(batches)):
        all_outputs.extend(batch_results[idx])
    return all_outputs


def _provider_supports_native_pdf(provider: Any) -> bool:
    """Check if provider explicitly advertises native PDF support."""
    kinds = getattr(provider, "supported_attachment_kinds", None)
    if kinds is not None:
        return "pdf" in kinds
    return False


def _check_media_capability(provider: Any, *, is_async: bool = False) -> None:
    """Raise if provider doesn't support media inference.

    When *is_async* is False (sync path), only ``SupportsMediaInfer`` is
    accepted.  When True (async path), either protocol is fine.
    """
    if is_async:
        if not isinstance(provider, (SupportsMediaInfer, SupportsAsyncMediaInfer)):
            provider_name = type(provider).__name__
            raise TypeError(
                f"Provider {provider_name} does not support media inference. "
                f"Use a provider that implements SupportsMediaInfer or "
                f"SupportsAsyncMediaInfer."
            )
    else:
        if not isinstance(provider, SupportsMediaInfer):
            provider_name = type(provider).__name__
            if isinstance(provider, SupportsAsyncMediaInfer):
                raise TypeError(
                    f"Provider {provider_name} only supports async media inference "
                    f"(ainfer_media). Use aextract() or extract_aiter() for async "
                    f"extraction, or implement infer_media() on the provider for "
                    f"sync support."
                )
            raise TypeError(
                f"Provider {provider_name} does not support media inference. "
                f"Use a provider that implements SupportsMediaInfer (e.g. "
                f"PydanticAIProvider with a vision-capable model like "
                f"openai:gpt-4o or gemini:gemini-2.5-flash)."
            )


def _build_media_inference_requests(
    ctx: _ExtractionContext,
    doc: Document,
    media_chunks: list[MediaChunk],
) -> list[InferenceRequest]:
    """Build InferenceRequest objects for media chunks."""
    requests: list[InferenceRequest] = []
    for chunk in media_chunks:
        prompt_text = _render_prompt(
            ctx.prompt_obj,
            schema_text=ctx.schema_text,
            examples=ctx.normalized_examples,
            question=chunk.text or doc.text or "Extract structured data from this document.",
            format_handler=ctx.format_handler,
            additional_context=doc.additional_context,
            output_kind=ctx.output_kind,
            native_mode=ctx.use_native_schema,
        )
        requests.append(
            InferenceRequest(
                prompt=prompt_text,
                attachments=(chunk.attachment,),
                document_id=doc.document_id,
                attachment_index=chunk.attachment_index,
                page_index=(chunk.page_index + 1) if chunk.page_index is not None else None,
            )
        )
    return requests


def _infer_media_batch(
    provider: Any,
    requests: Sequence[InferenceRequest],
    batch_length: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Call provider.infer_media in batches."""
    logger.debug(
        "Media batched inference: %d requests, batch_length=%d", len(requests), batch_length
    )
    all_outputs: list[str] = []
    for i in range(0, len(requests), batch_length):
        batch = requests[i : i + batch_length]
        outputs = normalize_text_outputs(
            provider.infer_media(batch, **infer_kwargs),
            expected_count=len(batch),
            context="provider.infer_media",
        )
        all_outputs.extend(outputs)
    return all_outputs


async def _ainfer_media_batch(
    provider: Any,
    requests: Sequence[InferenceRequest],
    batch_length: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Async version of media batched inference."""
    all_outputs: list[str] = []
    for i in range(0, len(requests), batch_length):
        batch = requests[i : i + batch_length]
        if isinstance(provider, SupportsAsyncMediaInfer):
            outputs = await provider.ainfer_media(batch, **infer_kwargs)
        else:
            outputs = await asyncio.to_thread(provider.infer_media, batch, **infer_kwargs)
        outputs = normalize_text_outputs(
            outputs,
            expected_count=len(batch),
            context="provider.ainfer_media"
            if isinstance(provider, SupportsAsyncMediaInfer)
            else "provider.infer_media",
        )
        all_outputs.extend(outputs)
    return all_outputs


def _infer_media_batch_parallel(
    provider: Any,
    requests: Sequence[InferenceRequest],
    batch_length: int,
    max_workers: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Call provider.infer_media in batches with ThreadPoolExecutor."""
    batch_length = max(1, batch_length)
    batches: list[Sequence[InferenceRequest]] = []
    for i in range(0, len(requests), batch_length):
        batches.append(requests[i : i + batch_length])
    max_workers = max(1, max_workers)
    logger.debug("Parallel media inference: %d batches, max_workers=%d", len(batches), max_workers)
    batch_results: dict[int, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(provider.infer_media, batch, **infer_kwargs): idx
            for idx, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch_results[idx] = normalize_text_outputs(
                future.result(),
                expected_count=len(batches[idx]),
                context="provider.infer_media",
            )
    all_outputs: list[str] = []
    for idx in range(len(batches)):
        all_outputs.extend(batch_results[idx])
    return all_outputs


async def _ainfer_media_batch_parallel(
    provider: Any,
    requests: Sequence[InferenceRequest],
    batch_length: int,
    max_workers: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Async parallel media inference with semaphore."""
    batch_length = max(1, batch_length)
    batches: list[Sequence[InferenceRequest]] = []
    for i in range(0, len(requests), batch_length):
        batches.append(requests[i : i + batch_length])
    max_workers = max(1, max_workers)
    sem = asyncio.Semaphore(max_workers)

    async def _run_batch(batch: Sequence[InferenceRequest]) -> list[str]:
        async with sem:
            if isinstance(provider, SupportsAsyncMediaInfer):
                raw = await provider.ainfer_media(batch, **infer_kwargs)
            else:
                raw = await asyncio.to_thread(provider.infer_media, batch, **infer_kwargs)
            return normalize_text_outputs(
                raw,
                expected_count=len(batch),
                context="provider.ainfer_media"
                if isinstance(provider, SupportsAsyncMediaInfer)
                else "provider.infer_media",
            )

    results = await asyncio.gather(*[_run_batch(b) for b in batches])
    all_outputs: list[str] = []
    for batch_result in results:
        all_outputs.extend(batch_result)
    return all_outputs


def _build_single_inference_request(
    ctx: _ExtractionContext,
    doc: Document,
    media_chunks: list[MediaChunk],
) -> InferenceRequest:
    """Build a single InferenceRequest bundling all media chunks."""
    all_attachments = tuple(chunk.attachment for chunk in media_chunks)
    # Collect unique text hints from all chunks (e.g. "Focus on pages: ...").
    base_question = doc.text or "Extract structured data from this document."
    extra_hints: list[str] = []
    for chunk in media_chunks:
        if chunk.text and chunk.text != doc.text and chunk.text not in extra_hints:
            extra_hints.append(chunk.text)
    question = "\n\n".join(extra_hints) if extra_hints else base_question
    prompt_text = _render_prompt(
        ctx.prompt_obj,
        schema_text=ctx.schema_text,
        examples=ctx.normalized_examples,
        question=question,
        format_handler=ctx.format_handler,
        additional_context=doc.additional_context,
        output_kind=ctx.output_kind,
        native_mode=ctx.use_native_schema,
    )
    return InferenceRequest(
        prompt=prompt_text,
        attachments=all_attachments,
        document_id=doc.document_id,
    )


def _resolve_page_strategy(strategy: str, media_chunks: list[MediaChunk]) -> str:
    """Resolve 'auto' page_strategy to concrete strategy."""
    if strategy != "auto":
        return strategy
    if not media_chunks:
        return "map_reduce"
    from .media.attachments import AttachmentKind

    all_native_pdf = all(chunk.attachment.kind is AttachmentKind.PDF for chunk in media_chunks)
    return "single" if all_native_pdf else "map_reduce"


def _try_pdf_text_extraction(doc: Document, media_options: Any) -> Document | None:
    """If pdf_mode='auto' and PDF has text layer, return a text-only Document."""
    if media_options.pdf_mode != "auto":
        return None
    if not doc.attachments:
        return None
    from .media.attachments import AttachmentKind

    if any(a.kind is not AttachmentKind.PDF for a in doc.attachments):
        return None
    try:
        from .media.preprocessing import has_text_layer

        if not any(has_text_layer(a.source) for a in doc.attachments):
            return None
        import fitz

        texts: list[str] = []
        for att in doc.attachments:
            data = att.source if isinstance(att.source, bytes) else att.source.read_bytes()
            pdf_doc = fitz.open(stream=data, filetype="pdf")
            try:
                page_indices = (
                    att.page_indices if att.page_indices is not None else tuple(range(len(pdf_doc)))
                )
                for pi in page_indices:
                    if 0 <= pi < len(pdf_doc):
                        page_text = pdf_doc[pi].get_text().strip()
                        if page_text:
                            texts.append(page_text)
            finally:
                pdf_doc.close()
        if not texts:
            return None
        extracted = "\n\n".join(texts)
        combined = f"{doc.text}\n\n{extracted}" if doc.text else extracted
        return Document(
            text=combined,
            document_id=doc.document_id,
            additional_context=doc.additional_context,
            attachments=(),
        )
    except ImportError:
        return None
    except Exception:
        logger.debug("PDF text extraction failed, falling back to media path", exc_info=True)
        return None


def _parse_with_repair[T](
    raw: str,
    target: type[T] | TypeAdapter[T],
    *,
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    repair: str,
    allow_partial: bool = False,
) -> ParseResult[T]:
    """Parse raw output, optionally applying local repair on validation failure."""
    try:
        return parse(
            raw,
            target,
            parse_options=parse_options,
            coerce_options=coerce_options,
            is_done=True,
            allow_partial=allow_partial,
        )
    except ValidationError:
        if repair != "local":
            raise
        # Local repair: retry with relaxed coercion (enable substring enum matching)
        relaxed = replace(coerce_options or CoerceOptions(), allow_substring_enum_match=True)
        return parse(
            raw,
            target,
            parse_options=parse_options,
            coerce_options=relaxed,
            is_done=True,
            allow_partial=allow_partial,
        )


def _default_merged_value(output_kind: Literal["object", "array"] | None) -> Any:
    if output_kind == "array":
        return []
    if output_kind == "object":
        return {}
    return None


def _is_unusable_partial(
    parsed: ParseResult[Any],
    output_kind: Literal["object", "array"] | None,
) -> bool:
    if output_kind not in {"object", "array"}:
        return False
    return "partial_unvalidated" in parsed.flags


def _process_inferred_chunks[T](
    *,
    chunks: Sequence[TextChunk],
    inferred: Sequence[str],
    target: type[T] | TypeAdapter[T],
    opts: ExtractOptions,
    output_kind: Literal["object", "array"] | None,
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
) -> tuple[list[Any], list[FieldEvidence], list[str], set[str], int, list[ChunkDebug]]:
    chunk_values: list[Any] = []
    chunk_evidence: list[FieldEvidence] = []
    chunk_outputs: list[str] = []
    pass_flags: set[str] = set()
    pass_worst_score: int = 0
    debug_entries: list[ChunkDebug] = []
    logger.debug("Processing %d chunks", len(chunks))

    for chunk_idx, (chunk, raw) in enumerate(zip(chunks, inferred, strict=True)):
        if not raw:
            continue
        chunk_outputs.append(raw)

        parse_error: str | None = None
        parsed: ParseResult[T] | None = None
        chunk_value: Any | None = None
        try:
            parsed = _parse_with_repair(
                raw,
                target,
                parse_options=parse_options,
                coerce_options=coerce_options,
                repair=opts.repair,
                allow_partial=True,
            )
            if _is_unusable_partial(parsed, output_kind):
                raise ValueError(
                    "Chunk parse produced unvalidated partial output "
                    "that does not match the target root kind"
                )
            chunk_value = to_jsonable_python(parsed.value)
            chunk_values.append(chunk_value)
            pass_flags.update(parsed.flags)
            pass_worst_score = max(pass_worst_score, parsed.score)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.debug("Chunk parse failed: %s", exc)
            parse_error = str(exc)
            parsed = None
            chunk_value = None
            if opts.chunk_error == "raise":
                raise

        if chunk_value is not None:
            chunk_evidence.extend(
                _align_evidence(
                    chunk.text,
                    chunk_value,
                    tokenizer=opts.tokenizer,
                    alignment=opts.alignment,
                    offset=chunk.start,
                )
            )

        if debug:
            debug_entries.append(
                ChunkDebug(
                    chunk_index=chunk_idx,
                    chunk_text_preview=chunk.text[:100],
                    raw_output=raw,
                    flags=parsed.flags if parsed is not None else (),
                    score=parsed.score if parsed is not None else 0,
                    error=parse_error,
                )
            )

    return (
        chunk_values,
        chunk_evidence,
        chunk_outputs,
        pass_flags,
        pass_worst_score,
        debug_entries,
    )


def _prepare_document_chunks_and_prompts(
    ctx: _ExtractionContext,
    doc: Document,
) -> tuple[list[TextChunk], list[str]]:
    chunks = list(
        iter_chunks(
            doc.text,
            max_char_buffer=ctx.opts.max_char_buffer,
            tokenizer=ctx.opts.tokenizer,
            overlap_chars=ctx.opts.overlap_chars,
        )
    )
    chunk_prompts = [
        _render_prompt(
            ctx.prompt_obj,
            schema_text=ctx.schema_text,
            examples=ctx.normalized_examples,
            question=chunk.text,
            format_handler=ctx.format_handler,
            additional_context=doc.additional_context,
            output_kind=ctx.output_kind,
            native_mode=ctx.use_native_schema,
        )
        for chunk in chunks
    ]
    return chunks, chunk_prompts


def _build_extract_result[T](
    *,
    ctx: _ExtractionContext,
    doc: Document,
    merged_value: Any,
    raw_outputs: list[str],
    all_flags: set[str],
    worst_score: int,
    doc_evidence: list[FieldEvidence],
    chunk_debug_entries: list[ChunkDebug],
    rendered_prompt_preview: str | None,
    debug: bool,
    conflicts: list[MergeConflict] | None = None,
) -> ExtractResult[T]:
    validated = ctx.adapter.validate(merged_value)
    debug_info = (
        ExtractDebug(
            prompt=ctx.prompt_obj.description,
            raw_outputs=raw_outputs,
            chunks=chunk_debug_entries,
            rendered_prompt_preview=rendered_prompt_preview,
        )
        if debug
        else None
    )
    return ExtractResult(
        value=validated,
        document_id=doc.document_id,
        raw_text=raw_outputs[-1] if raw_outputs else None,
        flags=tuple(sorted(all_flags)),
        score=worst_score,
        evidence=doc_evidence,
        debug=debug_info,
        conflicts=conflicts or [],
    )


@dataclass(slots=True)
class _DocumentState:
    """Mutable accumulator for per-document extraction across passes."""

    merged_value: Any = None
    raw_outputs: list[str] = field(default_factory=list)
    all_flags: set[str] = field(default_factory=set)
    worst_score: int = 0
    chunk_debug_entries: list[ChunkDebug] = field(default_factory=list)
    doc_evidence: list[FieldEvidence] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)


def _accumulate_pass[T](
    *,
    pass_index: int,
    state: _DocumentState,
    chunks: Sequence[TextChunk],
    inferred: list[str],
    target: type[T] | TypeAdapter[T],
    ctx: _ExtractionContext,
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
) -> None:
    """Process one inference pass and accumulate into *state*.

    May raise if ``chunk_error="raise"`` — this preserves fail-fast semantics
    so that subsequent passes are never invoked after a failure.
    """
    logger.debug("Extract pass %d: processing %d chunks", pass_index, len(chunks))
    (
        chunk_values,
        chunk_evidence,
        chunk_outputs,
        pass_flags,
        pass_worst_score,
        pass_debug_entries,
    ) = _process_inferred_chunks(
        chunks=chunks,
        inferred=inferred,
        target=target,
        opts=ctx.opts,
        output_kind=ctx.output_kind,
        parse_options=parse_options,
        coerce_options=coerce_options,
        debug=debug,
    )

    state.all_flags.update(pass_flags)
    state.worst_score = max(state.worst_score, pass_worst_score)
    if debug:
        state.chunk_debug_entries.extend(pass_debug_entries)

    for chunk_value in chunk_values:
        state.merged_value = _merge_values(
            state.merged_value,
            chunk_value,
            strategy=ctx.opts.merge_strategy,
            conflicts=state.conflicts,
        )

    if state.merged_value is None:
        state.merged_value = _default_merged_value(ctx.output_kind)

    if pass_index == 0:
        state.doc_evidence = chunk_evidence
        state.raw_outputs = chunk_outputs
    else:
        state.doc_evidence = merge_evidence(state.doc_evidence, chunk_evidence)
        state.raw_outputs.extend(chunk_outputs)


def _accumulate_media_pass[T](
    *,
    pass_index: int,
    state: _DocumentState,
    media_requests: list[InferenceRequest],
    media_chunks: list[MediaChunk],
    inferred: list[str],
    target: type[T] | TypeAdapter[T],
    ctx: _ExtractionContext,
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
) -> _DocumentState:
    """Process one media inference pass — parse outputs, skip text alignment,
    and produce vision-sourced FieldEvidence instead."""
    for req_idx, (req, chunk, raw) in enumerate(
        zip(media_requests, media_chunks, inferred, strict=True)
    ):
        if not raw:
            continue

        state.raw_outputs.append(raw)
        parse_error: str | None = None
        parsed: ParseResult[T] | None = None
        chunk_value: Any | None = None
        try:
            parsed = _parse_with_repair(
                raw,
                target,
                parse_options=parse_options,
                coerce_options=coerce_options,
                repair=ctx.opts.repair,
                allow_partial=True,
            )
            if _is_unusable_partial(parsed, ctx.output_kind):
                raise ValueError(
                    "Media chunk parse produced unvalidated partial output "
                    "that does not match the target root kind"
                )
            chunk_value = to_jsonable_python(parsed.value)
            state.all_flags.update(parsed.flags)
            state.worst_score = max(state.worst_score, parsed.score)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.debug("Media chunk parse failed: %s", exc)
            parse_error = str(exc)
            if ctx.opts.chunk_error == "raise":
                raise

        if chunk_value is not None:
            # Vision evidence: no text alignment, set source="vision"
            for path, text in _iter_leaf_values(chunk_value):
                state.doc_evidence.append(
                    FieldEvidence(
                        path=path,
                        value_preview=text[:80] if text else "",
                        char_interval=None,
                        token_interval=None,
                        alignment_status=AlignmentStatus.UNMATCHED,
                        source="vision",
                        attachment_index=chunk.attachment_index,
                        page_index=req.page_index,
                        grounding_method="unmatched",
                    )
                )
            state.merged_value = _merge_values(
                state.merged_value,
                chunk_value,
                strategy=ctx.opts.merge_strategy,
                conflicts=state.conflicts,
                page_index=req.page_index,
            )

        if debug:
            state.chunk_debug_entries.append(
                ChunkDebug(
                    chunk_index=req_idx,
                    chunk_text_preview=raw[:100],
                    raw_output=raw,
                    flags=parsed.flags if parsed is not None else (),
                    score=parsed.score if parsed is not None else 0,
                    error=parse_error,
                )
            )

    if state.merged_value is None:
        state.merged_value = _default_merged_value(ctx.output_kind)

    # Deduplicate evidence across passes
    if pass_index > 0:
        seen: set[tuple] = set()
        deduped: list[FieldEvidence] = []
        for ev in state.doc_evidence:
            key = (ev.path, ev.source, ev.attachment_index, ev.page_index, ev.value_preview)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        state.doc_evidence = deduped

    return state


def extract_iter[T](
    text_or_documents: str | Document | Iterable[Document],
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
    debug: bool = False,
) -> Iterator[ExtractResult[T]]:
    ctx = _build_extraction_context(
        text_or_documents,
        target,
        model=model,
        prompt=prompt,
        options=options,
        provider_kwargs=provider_kwargs,
    )

    for doc in ctx.documents:
        has_media = needs_media(doc.attachments)

        if has_media:
            # Try text extraction for PDFs with text layers (pdf_mode=auto)
            text_doc = _try_pdf_text_extraction(doc, ctx.opts.media)
            if text_doc is not None:
                doc = text_doc
                has_media = False

        # Build kwargs for native structured output (ignored by non-PydanticAI providers)
        _native_kwargs: dict[str, Any] = {}
        if ctx.use_native_schema and ctx.target_type is not None:
            _native_kwargs["target_type"] = ctx.target_type
            _native_kwargs["structured_output"] = ctx.opts.structured_output

        if has_media:
            _check_media_capability(ctx.provider, is_async=False)
            native_pdf = _provider_supports_native_pdf(ctx.provider)
            media_chunks = chunk_attachments(
                doc.attachments,
                text=doc.text,
                media_options=ctx.opts.media,
                provider_supports_native_pdf=native_pdf,
            )

            strategy = _resolve_page_strategy(ctx.opts.media.page_strategy, media_chunks)
            state = _DocumentState()
            batch_length = max(1, ctx.opts.batch_length)
            max_workers = max(1, ctx.opts.max_workers)

            if strategy == "single":
                single_req = _build_single_inference_request(ctx, doc, media_chunks)
                rendered_prompt_preview: str | None = single_req.prompt[:500] if debug else None
                for _pass in range(max(1, ctx.opts.passes)):
                    inferred = _infer_media_batch(
                        ctx.provider, [single_req], batch_length, **_native_kwargs
                    )
                    state = _accumulate_media_pass(
                        pass_index=_pass,
                        state=state,
                        media_requests=[single_req],
                        media_chunks=media_chunks[:1] if media_chunks else media_chunks,
                        inferred=inferred,
                        target=target,
                        ctx=ctx,
                        parse_options=parse_options,
                        coerce_options=coerce_options,
                        debug=debug,
                    )
            else:
                media_requests = _build_media_inference_requests(ctx, doc, media_chunks)
                rendered_prompt_preview = (
                    media_requests[0].prompt[:500] if (debug and media_requests) else None
                )
                for _pass in range(max(1, ctx.opts.passes)):
                    if max_workers <= 1:
                        inferred = _infer_media_batch(
                            ctx.provider, media_requests, batch_length, **_native_kwargs
                        )
                    else:
                        inferred = _infer_media_batch_parallel(
                            ctx.provider,
                            media_requests,
                            batch_length,
                            max_workers,
                            **_native_kwargs,
                        )
                    state = _accumulate_media_pass(
                        pass_index=_pass,
                        state=state,
                        media_requests=media_requests,
                        media_chunks=media_chunks,
                        inferred=inferred,
                        target=target,
                        ctx=ctx,
                        parse_options=parse_options,
                        coerce_options=coerce_options,
                        debug=debug,
                    )
        else:
            chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
            rendered_prompt_preview = chunk_prompts[0][:500] if (debug and chunk_prompts) else None
            state = _DocumentState()
            batch_length = max(1, ctx.opts.batch_length)
            max_workers = max(1, ctx.opts.max_workers)

            for _pass in range(max(1, ctx.opts.passes)):
                if max_workers <= 1:
                    inferred = _infer_batch(
                        ctx.provider, chunk_prompts, batch_length, **_native_kwargs
                    )
                else:
                    inferred = _infer_batch_parallel(
                        ctx.provider, chunk_prompts, batch_length, max_workers, **_native_kwargs
                    )
                _accumulate_pass(
                    pass_index=_pass,
                    state=state,
                    chunks=chunks,
                    inferred=inferred,
                    target=target,
                    ctx=ctx,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    debug=debug,
                )

        yield _build_extract_result(
            ctx=ctx,
            doc=doc,
            merged_value=state.merged_value,
            raw_outputs=state.raw_outputs,
            all_flags=state.all_flags,
            worst_score=state.worst_score,
            doc_evidence=state.doc_evidence,
            chunk_debug_entries=state.chunk_debug_entries,
            rendered_prompt_preview=rendered_prompt_preview,
            debug=debug,
            conflicts=state.conflicts,
        )


async def _ainfer_batch(
    provider: Any,
    prompts: Sequence[str],
    batch_length: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Async version of batched inference (sequential)."""
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_length):
        batch = prompts[i : i + batch_length]
        if hasattr(provider, "ainfer"):
            outputs = await provider.ainfer(batch, **infer_kwargs)
        else:
            outputs = await asyncio.to_thread(provider.infer, batch, **infer_kwargs)
        outputs = normalize_text_outputs(
            outputs,
            expected_count=len(batch),
            context="provider.ainfer" if hasattr(provider, "ainfer") else "provider.infer",
        )
        all_outputs.extend(outputs)
    return all_outputs


async def _ainfer_batch_parallel(
    provider: Any,
    prompts: Sequence[str],
    batch_length: int,
    max_workers: int,
    **infer_kwargs: Any,
) -> list[str]:
    """Async version of batched inference with concurrency via asyncio.gather."""
    batch_length = max(1, batch_length)
    batches: list[Sequence[str]] = []
    for i in range(0, len(prompts), batch_length):
        batches.append(prompts[i : i + batch_length])
    max_workers = max(1, max_workers)
    logger.debug(
        "Async parallel batched inference: %d batches, max_workers=%d", len(batches), max_workers
    )

    sem = asyncio.Semaphore(max_workers)

    async def _run_batch(batch: Sequence[str]) -> list[str]:
        async with sem:
            if hasattr(provider, "ainfer"):
                raw = await provider.ainfer(batch, **infer_kwargs)
            else:
                raw = await asyncio.to_thread(provider.infer, batch, **infer_kwargs)
            return normalize_text_outputs(
                raw,
                expected_count=len(batch),
                context="provider.ainfer" if hasattr(provider, "ainfer") else "provider.infer",
            )

    results = await asyncio.gather(*[_run_batch(b) for b in batches])

    all_outputs: list[str] = []
    for batch_result in results:
        all_outputs.extend(batch_result)
    return all_outputs


async def extract_aiter[T](
    text_or_documents: str | Document | Iterable[Document],
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
    debug: bool = False,
) -> AsyncIterator[ExtractResult[T]]:
    ctx = _build_extraction_context(
        text_or_documents,
        target,
        model=model,
        prompt=prompt,
        options=options,
        provider_kwargs=provider_kwargs,
    )

    for doc in ctx.documents:
        has_media = needs_media(doc.attachments)

        if has_media:
            text_doc = _try_pdf_text_extraction(doc, ctx.opts.media)
            if text_doc is not None:
                doc = text_doc
                has_media = False

        # Build kwargs for native structured output (ignored by non-PydanticAI providers)
        _native_kwargs: dict[str, Any] = {}
        if ctx.use_native_schema and ctx.target_type is not None:
            _native_kwargs["target_type"] = ctx.target_type
            _native_kwargs["structured_output"] = ctx.opts.structured_output

        if has_media:
            _check_media_capability(ctx.provider, is_async=True)
            native_pdf = _provider_supports_native_pdf(ctx.provider)
            media_chunks = chunk_attachments(
                doc.attachments,
                text=doc.text,
                media_options=ctx.opts.media,
                provider_supports_native_pdf=native_pdf,
            )

            strategy = _resolve_page_strategy(ctx.opts.media.page_strategy, media_chunks)
            state = _DocumentState()
            batch_length = max(1, ctx.opts.batch_length)
            max_workers = max(1, ctx.opts.max_workers)

            if strategy == "single":
                single_req = _build_single_inference_request(ctx, doc, media_chunks)
                rendered_prompt_preview: str | None = single_req.prompt[:500] if debug else None
                for _pass in range(max(1, ctx.opts.passes)):
                    inferred = await _ainfer_media_batch(
                        ctx.provider, [single_req], batch_length, **_native_kwargs
                    )
                    state = _accumulate_media_pass(
                        pass_index=_pass,
                        state=state,
                        media_requests=[single_req],
                        media_chunks=media_chunks[:1] if media_chunks else media_chunks,
                        inferred=inferred,
                        target=target,
                        ctx=ctx,
                        parse_options=parse_options,
                        coerce_options=coerce_options,
                        debug=debug,
                    )
            else:
                media_requests = _build_media_inference_requests(ctx, doc, media_chunks)
                rendered_prompt_preview = (
                    media_requests[0].prompt[:500] if (debug and media_requests) else None
                )
                for _pass in range(max(1, ctx.opts.passes)):
                    if max_workers <= 1:
                        inferred = await _ainfer_media_batch(
                            ctx.provider, media_requests, batch_length, **_native_kwargs
                        )
                    else:
                        inferred = await _ainfer_media_batch_parallel(
                            ctx.provider,
                            media_requests,
                            batch_length,
                            max_workers,
                            **_native_kwargs,
                        )
                    state = _accumulate_media_pass(
                        pass_index=_pass,
                        state=state,
                        media_requests=media_requests,
                        media_chunks=media_chunks,
                        inferred=inferred,
                        target=target,
                        ctx=ctx,
                        parse_options=parse_options,
                        coerce_options=coerce_options,
                        debug=debug,
                    )
        else:
            chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
            rendered_prompt_preview = chunk_prompts[0][:500] if (debug and chunk_prompts) else None
            state = _DocumentState()
            batch_length = max(1, ctx.opts.batch_length)
            max_workers = max(1, ctx.opts.max_workers)

            for _pass in range(max(1, ctx.opts.passes)):
                if max_workers <= 1:
                    inferred = await _ainfer_batch(
                        ctx.provider, chunk_prompts, batch_length, **_native_kwargs
                    )
                else:
                    inferred = await _ainfer_batch_parallel(
                        ctx.provider, chunk_prompts, batch_length, max_workers, **_native_kwargs
                    )
                _accumulate_pass(
                    pass_index=_pass,
                    state=state,
                    chunks=chunks,
                    inferred=inferred,
                    target=target,
                    ctx=ctx,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    debug=debug,
                )

        yield _build_extract_result(
            ctx=ctx,
            doc=doc,
            merged_value=state.merged_value,
            raw_outputs=state.raw_outputs,
            all_flags=state.all_flags,
            worst_score=state.worst_score,
            doc_evidence=state.doc_evidence,
            chunk_debug_entries=state.chunk_debug_entries,
            rendered_prompt_preview=rendered_prompt_preview,
            debug=debug,
            conflicts=state.conflicts,
        )


async def aextract[T](
    text_or_documents: str | Document | Iterable[Document],
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
    debug: bool = False,
) -> ExtractResult[T] | list[ExtractResult[T]]:
    results = [
        r
        async for r in extract_aiter(
            text_or_documents,
            target,
            model=model,
            prompt=prompt,
            options=options,
            provider_kwargs=provider_kwargs,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )
    ]
    if isinstance(text_or_documents, (str, Document)):
        return results[0]
    return results
