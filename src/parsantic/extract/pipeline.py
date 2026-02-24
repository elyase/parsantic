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
from .options import ExtractOptions
from .prompt import Example, Prompt, PromptValidationLevel
from .providers.base import ProviderConfig
from .providers.factory import create_provider
from .schema import PydanticSchemaAdapter
from .tokenizer import Tokenizer, TokenizerName, get_tokenizer
from .types import ChunkDebug, Document, ExtractDebug, ExtractResult, FieldEvidence

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
) -> str:
    lines: list[str] = [prompt.description.strip(), ""]
    if additional_context:
        lines.append(additional_context)
        lines.append("")

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
        lines.append("Examples")
        for ex in examples:
            formatted = format_handler.format_example(ex.output)
            lines.append(f"Q: {ex.text}")
            lines.append("A: " + formatted)
            lines.append("")
    lines.append(f"Q: {question}")
    lines.append("A:")
    return "\n".join(lines)


def _merge_values(
    base: Any,
    other: Any,
    *,
    strategy: Literal["first_wins", "last_wins", "prefer_non_null"] = "first_wins",
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
            if key in merged:
                merged[key] = _merge_values(merged[key], val, strategy=strategy)
            else:
                merged[key] = val
        return merged
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
        text: str,
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
        text: str,
        target: type[T] | TypeAdapter[T],
        *,
        parse_options: ParseOptions | None = None,
        coerce_options: CoerceOptions | None = None,
        debug: bool = False,
    ) -> ExtractResult[T]:
        return await aextract(
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
) -> list[str]:
    """Call provider.infer in batches of *batch_length* and concatenate results."""
    logger.debug("Batched inference: %d prompts, batch_length=%d", len(prompts), batch_length)
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_length):
        batch = prompts[i : i + batch_length]
        outputs = normalize_text_outputs(
            provider.infer(batch),
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
            executor.submit(provider.infer, batch): idx for idx, batch in enumerate(batches)
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
        )

    if state.merged_value is None:
        state.merged_value = _default_merged_value(ctx.output_kind)

    if pass_index == 0:
        state.doc_evidence = chunk_evidence
        state.raw_outputs = chunk_outputs
    else:
        state.doc_evidence = merge_evidence(state.doc_evidence, chunk_evidence)
        state.raw_outputs.extend(chunk_outputs)


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
        chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
        rendered_prompt_preview: str | None = (
            chunk_prompts[0][:500] if (debug and chunk_prompts) else None
        )
        state = _DocumentState()
        batch_length = max(1, ctx.opts.batch_length)
        max_workers = max(1, ctx.opts.max_workers)

        for _pass in range(max(1, ctx.opts.passes)):
            if max_workers <= 1:
                inferred = _infer_batch(ctx.provider, chunk_prompts, batch_length)
            else:
                inferred = _infer_batch_parallel(
                    ctx.provider, chunk_prompts, batch_length, max_workers
                )
            # Process immediately to preserve fail-fast on chunk_error="raise"
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
        )


async def _ainfer_batch(
    provider: Any,
    prompts: Sequence[str],
    batch_length: int,
) -> list[str]:
    """Async version of batched inference (sequential)."""
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_length):
        batch = prompts[i : i + batch_length]
        if hasattr(provider, "ainfer"):
            outputs = await provider.ainfer(batch)
        else:
            outputs = await asyncio.to_thread(provider.infer, batch)
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
                raw = await provider.ainfer(batch)
            else:
                raw = await asyncio.to_thread(provider.infer, batch)
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
        chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
        rendered_prompt_preview: str | None = (
            chunk_prompts[0][:500] if (debug and chunk_prompts) else None
        )
        state = _DocumentState()
        batch_length = max(1, ctx.opts.batch_length)
        max_workers = max(1, ctx.opts.max_workers)

        for _pass in range(max(1, ctx.opts.passes)):
            if max_workers <= 1:
                inferred = await _ainfer_batch(ctx.provider, chunk_prompts, batch_length)
            else:
                inferred = await _ainfer_batch_parallel(
                    ctx.provider, chunk_prompts, batch_length, max_workers
                )
            # Process immediately to preserve fail-fast on chunk_error="raise"
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
        )


async def aextract[T](
    text: str,
    target: type[T] | TypeAdapter[T],
    *,
    model: str | Any | None = None,
    prompt: Prompt | str | None = None,
    options: ExtractOptions | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
    debug: bool = False,
) -> ExtractResult[T]:
    results = [
        r
        async for r in extract_aiter(
            text,
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
    return results[0]
