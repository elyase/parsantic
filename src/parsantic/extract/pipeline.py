from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import TypeAdapter

from parsantic.api import ParseResult, parse
from parsantic.coerce import CoerceOptions
from parsantic.jsonish import ParseOptions

from .alignment import AlignmentOptions, align_value_to_text, merge_evidence
from .chunking import iter_chunks
from .formatting import FormatHandler
from .options import ExtractOptions
from .prompt import Example, Prompt, PromptValidationLevel
from .providers.base import ProviderConfig
from .providers.factory import create_provider
from .schema import PydanticSchemaAdapter, SchemaAdapter
from .tokenizer import get_tokenizer
from .types import ChunkDebug, Document, ExtractDebug, ExtractResult, FieldEvidence

_DEFAULT_MODEL = "openai:gpt-4o-mini"


def _resolve_model(model: str | Any | None) -> str | Any:
    """Return an explicit model, falling back to env var or built-in default."""
    if model is not None:
        return model
    return os.environ.get("PARSANTIC_MODEL", _DEFAULT_MODEL)


_DEFAULT_DESCRIPTION = "Extract structured data that matches the provided schema."


def _escape_json_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _iter_leaf_values(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, dict):
        for key, val in value.items():
            next_path = f"{path}/{_escape_json_pointer(str(key))}"
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
    adapter: SchemaAdapter[Any],
    *,
    tokenizer: str | None,
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
    tokenizer: str | None,
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

    resolved_model = _resolve_model(model)
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
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_length):
        batch = prompts[i : i + batch_length]
        outputs = provider.infer(batch)
        if not isinstance(outputs, Sequence):
            outputs = list(outputs)
        all_outputs.extend(outputs)
    return all_outputs


def _parse_with_repair[T](
    raw: str,
    target: type[T] | TypeAdapter[T],
    *,
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    repair: str,
) -> ParseResult[T]:
    """Parse raw output, optionally applying local repair on validation failure."""
    try:
        return parse(
            raw,
            target,
            parse_options=parse_options,
            coerce_options=coerce_options,
            is_done=True,
        )
    except Exception:
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
        )


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
        doc_text = doc.text
        doc_evidence: list[FieldEvidence] = []
        merged_value: Any = None
        raw_outputs: list[str] = []
        all_flags: set[str] = set()
        worst_score: int = 0
        chunk_debug_entries: list[ChunkDebug] = []
        rendered_prompt_preview: str | None = None

        chunks = list(
            iter_chunks(
                doc_text,
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

        if debug and chunk_prompts and rendered_prompt_preview is None:
            rendered_prompt_preview = chunk_prompts[0][:500]

        for _pass in range(max(1, ctx.opts.passes)):
            chunk_values: list[Any] = []
            chunk_evidence: list[FieldEvidence] = []
            chunk_outputs: list[str] = []

            # Infer: use batching and optional parallelism
            batch_length = max(1, ctx.opts.batch_length)
            max_workers = max(1, ctx.opts.max_workers)

            if max_workers <= 1:
                # Sequential batched inference
                inferred = _infer_batch(ctx.provider, chunk_prompts, batch_length)
            else:
                # Parallel batched inference using ThreadPoolExecutor
                # Split prompts into batches, run batches in parallel, then reassemble
                # in input order after collecting completed futures.
                batches: list[Sequence[str]] = []
                for i in range(0, len(chunk_prompts), batch_length):
                    batches.append(chunk_prompts[i : i + batch_length])

                batch_results: dict[int, Sequence[str]] = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(ctx.provider.infer, batch): idx
                        for idx, batch in enumerate(batches)
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        batch_result = future.result()
                        if not isinstance(batch_result, Sequence):
                            batch_result = list(batch_result)
                        batch_results[idx] = batch_result

                inferred = []
                for idx in range(len(batches)):
                    inferred.extend(batch_results[idx])

            # Process each chunk's output in deterministic order
            for chunk_idx, (chunk, raw) in enumerate(zip(chunks, inferred, strict=True)):
                if not raw:
                    continue
                chunk_outputs.append(raw)

                parsed: ParseResult[T] = _parse_with_repair(
                    raw,
                    target,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    repair=ctx.opts.repair,
                )
                chunk_value = ctx.adapter.dump(parsed.value)
                chunk_values.append(chunk_value)

                chunk_evidence.extend(
                    _align_evidence(
                        chunk.text,
                        chunk_value,
                        tokenizer=ctx.opts.tokenizer,
                        alignment=ctx.opts.alignment,
                        offset=chunk.start,
                    )
                )
                all_flags.update(parsed.flags)
                worst_score = max(worst_score, parsed.score)

                # Collect per-chunk debug info
                if debug:
                    chunk_debug_entries.append(
                        ChunkDebug(
                            chunk_index=chunk_idx,
                            chunk_text_preview=chunk.text[:100],
                            raw_output=raw,
                            flags=parsed.flags,
                            score=parsed.score,
                        )
                    )

            for chunk_value in chunk_values:
                merged_value = _merge_values(
                    merged_value,
                    chunk_value,
                    strategy=ctx.opts.merge_strategy,
                )

            if merged_value is None:
                merged_value = {}

            if _pass == 0:
                doc_evidence = chunk_evidence
                raw_outputs = chunk_outputs
            else:
                doc_evidence = merge_evidence(doc_evidence, chunk_evidence)
                raw_outputs.extend(chunk_outputs)

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
        yield ExtractResult(
            value=validated,
            document_id=doc.document_id,
            raw_text=raw_outputs[-1] if raw_outputs else None,
            flags=tuple(sorted(all_flags)),
            score=worst_score,
            evidence=doc_evidence,
            debug=debug_info,
        )


async def _ainfer_batch(
    provider: Any,
    prompts: Sequence[str],
    batch_length: int,
) -> list[str]:
    """Async version of batched inference."""
    all_outputs: list[str] = []
    for i in range(0, len(prompts), batch_length):
        batch = prompts[i : i + batch_length]
        if hasattr(provider, "ainfer"):
            outputs = await provider.ainfer(batch)
        else:
            outputs = await asyncio.to_thread(provider.infer, batch)
        if not isinstance(outputs, Sequence):
            outputs = list(outputs)
        all_outputs.extend(outputs)
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
        doc_text = doc.text
        doc_evidence: list[FieldEvidence] = []
        merged_value: Any = None
        raw_outputs: list[str] = []
        all_flags: set[str] = set()
        worst_score: int = 0
        chunk_debug_entries: list[ChunkDebug] = []
        rendered_prompt_preview: str | None = None

        chunks = list(
            iter_chunks(
                doc_text,
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

        if debug and chunk_prompts and rendered_prompt_preview is None:
            rendered_prompt_preview = chunk_prompts[0][:500]

        for _pass in range(max(1, ctx.opts.passes)):
            chunk_values: list[Any] = []
            chunk_evidence: list[FieldEvidence] = []
            chunk_outputs: list[str] = []

            # Infer with batching
            batch_length = max(1, ctx.opts.batch_length)
            inferred = await _ainfer_batch(ctx.provider, chunk_prompts, batch_length)

            # Process each chunk's output in deterministic order
            for chunk_idx, (chunk, raw) in enumerate(zip(chunks, inferred, strict=True)):
                if not raw:
                    continue
                chunk_outputs.append(raw)

                parsed: ParseResult[T] = _parse_with_repair(
                    raw,
                    target,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    repair=ctx.opts.repair,
                )
                chunk_value = ctx.adapter.dump(parsed.value)
                chunk_values.append(chunk_value)

                chunk_evidence.extend(
                    _align_evidence(
                        chunk.text,
                        chunk_value,
                        tokenizer=ctx.opts.tokenizer,
                        alignment=ctx.opts.alignment,
                        offset=chunk.start,
                    )
                )
                all_flags.update(parsed.flags)
                worst_score = max(worst_score, parsed.score)

                # Collect per-chunk debug info
                if debug:
                    chunk_debug_entries.append(
                        ChunkDebug(
                            chunk_index=chunk_idx,
                            chunk_text_preview=chunk.text[:100],
                            raw_output=raw,
                            flags=parsed.flags,
                            score=parsed.score,
                        )
                    )

            for chunk_value in chunk_values:
                merged_value = _merge_values(
                    merged_value,
                    chunk_value,
                    strategy=ctx.opts.merge_strategy,
                )

            if merged_value is None:
                merged_value = {}

            if _pass == 0:
                doc_evidence = chunk_evidence
                raw_outputs = chunk_outputs
            else:
                doc_evidence = merge_evidence(doc_evidence, chunk_evidence)
                raw_outputs.extend(chunk_outputs)

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
        yield ExtractResult(
            value=validated,
            document_id=doc.document_id,
            raw_text=raw_outputs[-1] if raw_outputs else None,
            flags=tuple(sorted(all_flags)),
            score=worst_score,
            evidence=doc_evidence,
            debug=debug_info,
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
