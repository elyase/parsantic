from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from parsantic.api import ParseResult, parse
from parsantic.coerce import CoerceOptions
from parsantic.config import resolve_model
from parsantic.json_pointer import (
    build_json_pointer,
    escape_json_pointer_token,
    parse_json_pointer,
)
from parsantic.jsonish import ParseOptions
from parsantic.provider_output import normalize_text_outputs

from .alignment import AlignmentOptions, align_value_to_text
from .chunking import TextChunk, iter_chunks
from .formatting import FormatHandler
from .media.chunking import MediaChunk, chunk_attachments, needs_media
from .options import ExtractOptions, ResolvedStrategy
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
    SourceRef,
)

logger = logging.getLogger(__name__)


_DEFAULT_DESCRIPTION = "Extract structured data that matches the provided schema."
_MISSING = object()
_STRICT_PAGE_ALIGNMENT = AlignmentOptions(enable_fuzzy_alignment=False, accept_match_lesser=False)
type _RootKind = Literal["object", "array", "scalar"]


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


def _pointer_tokens(path: str) -> tuple[str, ...]:
    if path in {"", "/"}:
        return ()
    return tuple(parse_json_pointer(path))


def _pointer_from_tokens(tokens: Sequence[str]) -> str:
    return build_json_pointer(list(tokens)) or "/"


def _child_pointer(path: str, token: str | int) -> str:
    token_text = escape_json_pointer_token(str(token))
    if path in {"", "/"}:
        return f"/{token_text}"
    return f"{path}/{token_text}"


def _collect_leaf_path_map(
    value: Any,
    *,
    source_path: str,
    target_path: str,
) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        mapping: dict[str, str] = {}
        for key, child in value.items():
            mapping.update(
                _collect_leaf_path_map(
                    child,
                    source_path=_child_pointer(source_path, key),
                    target_path=_child_pointer(target_path, key),
                )
            )
        return mapping
    if isinstance(value, list):
        mapping: dict[str, str] = {}
        for idx, child in enumerate(value):
            mapping.update(
                _collect_leaf_path_map(
                    child,
                    source_path=_child_pointer(source_path, idx),
                    target_path=_child_pointer(target_path, idx),
                )
            )
        return mapping
    return {target_path or "/": source_path or "/"}


def _remap_leaf_evidence(
    evidence: Sequence[FieldEvidence],
    leaf_map: dict[str, str],
) -> list[FieldEvidence]:
    if not leaf_map:
        return []
    source_to_target = {source_path: target_path for target_path, source_path in leaf_map.items()}
    remapped: list[FieldEvidence] = []
    for ev in evidence:
        target_path = source_to_target.get(ev.path)
        if target_path is None:
            continue
        remapped.append(
            FieldEvidence(
                path=target_path,
                value_preview=ev.value_preview,
                char_interval=ev.char_interval,
                token_interval=ev.token_interval,
                alignment_status=ev.alignment_status,
                source=ev.source,
                attachment_index=ev.attachment_index,
                page_index=ev.page_index,
                bbox_norm=ev.bbox_norm,
                grounding_method=ev.grounding_method,
            )
        )
    return remapped


def _is_record_like_list(items: Sequence[Any]) -> bool:
    return any(isinstance(item, (dict, list)) for item in items)


def _merge_branch_values(
    base: Any,
    incoming: Any,
    *,
    strategy: Literal["first_wins", "last_wins", "prefer_non_null"] = "first_wins",
    conflicts: list[MergeConflict] | None = None,
    path: str = "/",
    page_index: int | None = None,
) -> tuple[Any, dict[str, str]]:
    if base is None:
        return incoming, _collect_leaf_path_map(incoming, source_path=path, target_path=path)
    if incoming is None:
        return base, {}

    if isinstance(base, list) and isinstance(incoming, list):
        if not _is_record_like_list([*base, *incoming]):
            merged = list(base)
            leaf_map: dict[str, str] = {}
            seen: Counter[str] = Counter()
            for item in merged:
                try:
                    seen[json.dumps(item, sort_keys=True, default=str)] += 1
                except (ValueError, TypeError):
                    continue
            for idx, item in enumerate(incoming):
                try:
                    key = json.dumps(item, sort_keys=True, default=str)
                except (ValueError, TypeError):
                    key = None
                if key is not None and seen[key] > 0:
                    seen[key] -= 1
                    continue
                target_index = len(merged)
                merged.append(item)
                leaf_map.update(
                    _collect_leaf_path_map(
                        item,
                        source_path=_child_pointer(path, idx),
                        target_path=_child_pointer(path, target_index),
                    )
                )
            return merged, leaf_map

        merged = list(base)
        leaf_map: dict[str, str] = {}
        offset = len(base)
        for idx, item in enumerate(incoming):
            merged.append(item)
            source_path = _child_pointer(path, idx)
            target_path = _child_pointer(path, offset + idx)
            leaf_map.update(
                _collect_leaf_path_map(
                    item,
                    source_path=source_path,
                    target_path=target_path,
                )
            )
        return merged, leaf_map

    if isinstance(base, dict) and isinstance(incoming, dict):
        merged = dict(base)
        leaf_map: dict[str, str] = {}
        for key, val in incoming.items():
            child_path = _child_pointer(path, key)
            if key in merged:
                merged[key], child_map = _merge_branch_values(
                    merged[key],
                    val,
                    strategy=strategy,
                    conflicts=conflicts,
                    path=child_path,
                    page_index=page_index,
                )
                leaf_map.update(child_map)
            else:
                merged[key] = val
                leaf_map.update(
                    _collect_leaf_path_map(
                        val,
                        source_path=child_path,
                        target_path=child_path,
                    )
                )
        return merged, leaf_map

    if base != incoming and conflicts is not None:
        conflicts.append(
            MergeConflict(
                path=path or "/",
                existing_preview=str(base)[:80],
                incoming_preview=str(incoming)[:80],
                page_index=page_index,
            )
        )

    use_incoming = False
    if strategy == "last_wins":
        use_incoming = True
    elif strategy == "prefer_non_null":
        if base in (None, ""):
            use_incoming = True
        elif incoming in (None, ""):
            use_incoming = False
        else:
            use_incoming = False

    chosen = incoming if use_incoming else base
    leaf_map = (
        _collect_leaf_path_map(incoming, source_path=path, target_path=path) if use_incoming else {}
    )
    return chosen, leaf_map


@dataclass(slots=True)
class _HybridResolution:
    value: Any
    page_leaf_map: dict[str, str] = field(default_factory=dict)
    whole_leaf_map: dict[str, str] = field(default_factory=dict)
    chosen_page_paths: set[str] = field(default_factory=set)
    chosen_whole_paths: set[str] = field(default_factory=set)


def _normalize_prompt(prompt: Prompt | str | None) -> Prompt:
    if prompt is None:
        return Prompt(description=_DEFAULT_DESCRIPTION)
    if isinstance(prompt, str):
        return Prompt(description=prompt)
    return prompt


def _schema_root_kind(schema: dict[str, Any]) -> _RootKind | None:
    schema_type = schema.get("type")
    if schema_type == "array":
        return "array"
    if schema_type == "object":
        return "object"
    if schema_type in {"string", "number", "integer", "boolean", "null"}:
        return "scalar"
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if not isinstance(variants, list) or not variants:
            continue
        variant_kinds = {_schema_root_kind(variant) for variant in variants}
        if len(variant_kinds) == 1 and None not in variant_kinds:
            return next(iter(variant_kinds))
        if variant_kinds and variant_kinds <= {"scalar", None}:
            return "scalar"
    if "const" in schema:
        const_value = schema["const"]
        if isinstance(const_value, (str, int, float, bool)) or const_value is None:
            return "scalar"
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        if all(
            isinstance(enum_value, (str, int, float, bool)) or enum_value is None
            for enum_value in schema["enum"]
        ):
            return "scalar"
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
    output_kind: _RootKind | None = None,
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
            expected_label = "value" if expected_kind == "scalar" else expected_kind
            lines.append(
                f"Output a single JSON {expected_label}. "
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
        if page_index is not None:
            return [*base, *other]

        merged = list(base)
        seen: Counter[str] = Counter()
        for item in merged:
            try:
                seen[json.dumps(item, sort_keys=True, default=str)] += 1
            except (ValueError, TypeError):
                continue
        for item in other:
            try:
                key = json.dumps(item, sort_keys=True, default=str)
            except (ValueError, TypeError):
                key = None
            if key is None:
                merged.append(item)
                continue
            if seen[key] > 0:
                seen[key] -= 1
                continue
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
    output_kind: _RootKind | None
    normalized_examples: list[Example]
    provider: Any
    resolved_strategy: ResolvedStrategy
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
    resolved_strategy = opts.resolve_runtime_strategy()
    opts = replace(
        opts,
        mode=None,
        document_input="auto",
        page_input="auto",
        media=resolved_strategy.media,
        strategy=None,
    )
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
        resolved_strategy=resolved_strategy,
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

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    async def __aenter__(self) -> Extractor:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

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


def _provider_supports_async_text(provider: Any) -> bool:
    """Duck-type check for providers that expose async text inference."""
    return callable(getattr(provider, "ainfer", None))


def _ensure_native_pdf_support(
    provider: Any,
    attachments: Sequence[Any],
    *,
    media_options: Any,
    branch_label: str,
) -> None:
    """Fail fast when the caller explicitly requested native PDF input."""
    if media_options.pdf_mode != "native":
        return

    from .media.attachments import AttachmentKind

    if not any(attachment.kind is AttachmentKind.PDF for attachment in attachments):
        return
    if _provider_supports_native_pdf(provider):
        return

    provider_name = type(provider).__name__
    raise TypeError(
        f"Provider {provider_name} does not support native PDF input, but {branch_label} "
        "requested it. Use document_input='image', mode='page', media.pdf_mode='raster', "
        "or a provider that advertises PDF attachments."
    )


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


def _run_parallel_pair[T, U](
    first: Callable[[], T],
    second: Callable[[], U],
) -> tuple[T, U]:
    """Run two independent sync callables concurrently and preserve result order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first)
        second_future = executor.submit(second)
        return first_future.result(), second_future.result()


async def _arun_parallel_pair[T, U](
    first: Callable[[], Awaitable[T]],
    second: Callable[[], Awaitable[U]],
) -> tuple[T, U]:
    """Run two independent async callables concurrently and preserve result order."""
    return await asyncio.gather(first(), second())


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


@dataclass(frozen=True, slots=True)
class _PdfPageText:
    attachment_index: int
    page_index: int
    text: str


def _extract_pdf_page_texts(doc: Document) -> list[_PdfPageText]:
    if not doc.attachments:
        return []
    from .media.attachments import AttachmentKind

    if any(attachment.kind is not AttachmentKind.PDF for attachment in doc.attachments):
        return []

    try:
        from .media.preprocessing import has_text_layer

        if not any(has_text_layer(attachment.source) for attachment in doc.attachments):
            return []
        import fitz

        texts: list[_PdfPageText] = []
        for attachment_index, attachment in enumerate(doc.attachments):
            data = (
                attachment.source
                if isinstance(attachment.source, bytes)
                else attachment.source.read_bytes()
            )
            pdf_doc = fitz.open(stream=data, filetype="pdf")
            try:
                page_indices = (
                    attachment.page_indices
                    if attachment.page_indices is not None
                    else tuple(range(len(pdf_doc)))
                )
                for page_index in page_indices:
                    if 0 <= page_index < len(pdf_doc):
                        page_text = pdf_doc[page_index].get_text().strip()
                        if page_text:
                            texts.append(
                                _PdfPageText(
                                    attachment_index=attachment_index,
                                    page_index=page_index + 1,
                                    text=page_text,
                                )
                            )
            finally:
                pdf_doc.close()
        return texts
    except ImportError:
        return []
    except Exception:
        logger.debug("PDF text extraction failed, falling back to media path", exc_info=True)
        return []


def _try_pdf_text_extraction(doc: Document, media_options: Any) -> Document | None:
    """If pdf_mode='auto' and PDF has text layer, return a text-only Document."""
    if media_options.pdf_mode != "auto":
        return None
    page_texts = _extract_pdf_page_texts(doc)
    if not page_texts:
        return None
    extracted = "\n\n".join(page.text for page in page_texts)
    combined = f"{doc.text}\n\n{extracted}" if doc.text else extracted
    return Document(
        text=combined,
        document_id=doc.document_id,
        additional_context=doc.additional_context,
        attachments=(),
    )


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


def _default_merged_value(output_kind: _RootKind | None) -> Any:
    if output_kind == "array":
        return []
    if output_kind == "object":
        return {}
    return None


def _is_unusable_partial(
    parsed: ParseResult[Any],
    output_kind: _RootKind | None,
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
    output_kind: _RootKind | None,
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
            if debug:
                debug_entries.append(
                    ChunkDebug(
                        chunk_index=chunk_idx,
                        chunk_text_preview=chunk.text[:100],
                        raw_output="",
                        flags=(),
                        score=0,
                        error="empty output",
                    )
                )
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


def _build_single_text_prompt(
    ctx: _ExtractionContext,
    doc: Document,
) -> str:
    question = doc.text or "Extract structured data from this document."
    return _render_prompt(
        ctx.prompt_obj,
        schema_text=ctx.schema_text,
        examples=ctx.normalized_examples,
        question=question,
        format_handler=ctx.format_handler,
        additional_context=doc.additional_context,
        output_kind=ctx.output_kind,
        native_mode=ctx.use_native_schema,
    )


def _build_sources(doc_evidence: Sequence[FieldEvidence]) -> dict[str, SourceRef]:
    pages_by_path: dict[str, set[int]] = {}
    document_only_paths: set[str] = set()

    for evidence in doc_evidence:
        if evidence.page_index is None:
            document_only_paths.add(evidence.path)
            continue
        pages_by_path.setdefault(evidence.path, set()).add(evidence.page_index)

    sources: dict[str, SourceRef] = {}
    for path in sorted(document_only_paths | set(pages_by_path)):
        pages = tuple(sorted(pages_by_path.get(path, ())))
        if pages:
            sources[path] = SourceRef(scope="page", pages=pages)
        else:
            sources[path] = SourceRef(scope="document")
    return sources


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
        sources=_build_sources(doc_evidence),
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
    pass_value: Any = None
    pass_evidence: list[FieldEvidence] = []
    pass_conflicts: list[MergeConflict] = []
    for chunk_idx, (chunk, raw) in enumerate(zip(chunks, inferred, strict=True)):
        if not raw:
            if debug:
                state.chunk_debug_entries.append(
                    ChunkDebug(
                        chunk_index=chunk_idx,
                        chunk_text_preview=chunk.text[:100],
                        raw_output="",
                        flags=(),
                        score=0,
                        error="empty output",
                    )
                )
            continue

        state.raw_outputs.append(raw)
        parse_error: str | None = None
        parsed: ParseResult[T] | None = None
        chunk_value: Any | None = None
        chunk_evidence: list[FieldEvidence] = []

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
                    "Chunk parse produced unvalidated partial output "
                    "that does not match the target root kind"
                )
            chunk_value = to_jsonable_python(parsed.value)
            state.all_flags.update(parsed.flags)
            state.worst_score = max(state.worst_score, parsed.score)
            chunk_evidence = _align_evidence(
                chunk.text,
                chunk_value,
                tokenizer=ctx.opts.tokenizer,
                alignment=ctx.opts.alignment,
                offset=chunk.start,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.debug("Chunk parse failed: %s", exc)
            parse_error = str(exc)
            if ctx.opts.chunk_error == "raise":
                raise

        if chunk_value is not None:
            pass_value, leaf_map = _merge_branch_values(
                pass_value,
                chunk_value,
                strategy=ctx.opts.merge_strategy,
                conflicts=pass_conflicts,
                path="/",
            )
            pass_evidence.extend(_remap_leaf_evidence(chunk_evidence, leaf_map))

        if debug:
            state.chunk_debug_entries.append(
                ChunkDebug(
                    chunk_index=chunk_idx,
                    chunk_text_preview=chunk.text[:100],
                    raw_output=raw,
                    flags=parsed.flags if parsed is not None else (),
                    score=parsed.score if parsed is not None else 0,
                    error=parse_error,
                )
            )

    if pass_value is None:
        pass_value = _default_merged_value(ctx.output_kind)
    state.merged_value = _merge_values(
        state.merged_value,
        pass_value,
        strategy=ctx.opts.merge_strategy,
        conflicts=state.conflicts,
    )
    state.conflicts.extend(pass_conflicts)
    if state.merged_value is None:
        state.merged_value = _default_merged_value(ctx.output_kind)
    if pass_index == 0:
        state.doc_evidence = pass_evidence
    else:
        state.doc_evidence = _dedupe_field_evidence([*state.doc_evidence, *pass_evidence])


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
    pass_value: Any = None
    pass_evidence: list[FieldEvidence] = []
    pass_conflicts: list[MergeConflict] = []
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
            chunk_evidence: list[FieldEvidence] = []
            for path, text in _iter_leaf_values(chunk_value):
                chunk_evidence.append(
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
            pass_value, leaf_map = _merge_branch_values(
                pass_value,
                chunk_value,
                strategy=ctx.opts.merge_strategy,
                conflicts=pass_conflicts,
                path="/",
                page_index=req.page_index,
            )
            pass_evidence.extend(_remap_leaf_evidence(chunk_evidence, leaf_map))

        if debug:
            state.chunk_debug_entries.append(
                ChunkDebug(
                    chunk_index=req_idx,
                    chunk_text_preview=(chunk.text or "")[:100],
                    raw_output=raw,
                    flags=parsed.flags if parsed is not None else (),
                    score=parsed.score if parsed is not None else 0,
                    error=parse_error,
                )
            )

    if pass_value is None:
        pass_value = _default_merged_value(ctx.output_kind)
    state.merged_value = _merge_values(
        state.merged_value,
        pass_value,
        strategy=ctx.opts.merge_strategy,
        conflicts=state.conflicts,
    )
    state.conflicts.extend(pass_conflicts)
    if state.merged_value is None:
        state.merged_value = _default_merged_value(ctx.output_kind)
    if pass_index == 0:
        state.doc_evidence = pass_evidence
    else:
        state.doc_evidence = _dedupe_field_evidence([*state.doc_evidence, *pass_evidence])

    return state


def _dedupe_field_evidence(evidence: Sequence[FieldEvidence]) -> list[FieldEvidence]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldEvidence] = []
    for ev in evidence:
        key = (
            ev.path,
            ev.value_preview,
            ev.char_interval,
            ev.token_interval,
            ev.alignment_status,
            ev.source,
            ev.attachment_index,
            ev.page_index,
            ev.bbox_norm,
            ev.grounding_method,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)
    return deduped


def _is_missing_hybrid_value(value: Any) -> bool:
    return value in (None, "", [], {})


@dataclass(slots=True)
class _HybridMergeTrace:
    page_paths: set[str] = field(default_factory=set)
    whole_paths: set[str] = field(default_factory=set)
    page_aliases: list[tuple[str, str]] = field(default_factory=list)
    whole_aliases: list[tuple[str, str]] = field(default_factory=list)

    def include_page_value(self, value: Any, path: str) -> None:
        self.page_paths.update(_leaf_paths_for_value(value, path))

    def include_whole_value(self, value: Any, path: str) -> None:
        self.whole_paths.update(_leaf_paths_for_value(value, path))

    def merge(self, other: _HybridMergeTrace) -> None:
        self.page_paths.update(other.page_paths)
        self.whole_paths.update(other.whole_paths)
        self.page_aliases.extend(other.page_aliases)
        self.whole_aliases.extend(other.whole_aliases)


def _pointer_tokens(path: str) -> tuple[str, ...]:
    if path in {"", "/"}:
        return ()
    return tuple(parse_json_pointer(path))


def _pointer_path(path: str) -> str:
    return path or "/"


def _pointer_with_child(path: str, token: str | int) -> str:
    return build_json_pointer([*_pointer_tokens(path), str(token)])


def _leaf_paths_for_value(value: Any, path: str) -> set[str]:
    leaf_paths = {leaf_path for leaf_path, _ in _iter_leaf_values(value, path)}
    if leaf_paths:
        return leaf_paths
    if value in ({}, []):
        return {_pointer_path(path)}
    return set()


def _evidence_preview(value: str) -> str:
    return value[:80]


def _evidence_value_index(
    evidence: Sequence[FieldEvidence],
) -> dict[str, set[str]]:
    values_by_path: dict[str, set[str]] = defaultdict(set)
    for item in evidence:
        if item.value_preview == "":
            continue
        values_by_path[item.path].add(item.value_preview)
    return values_by_path


def _conflicts_by_path(conflicts: Sequence[MergeConflict]) -> dict[str, list[MergeConflict]]:
    grouped: dict[str, list[MergeConflict]] = defaultdict(list)
    for conflict in conflicts:
        grouped[conflict.path].append(conflict)
    return grouped


def _relative_leaf_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    return {path or "/": preview for path, preview in _iter_leaf_values(value)}


def _normalize_leaf_preview(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def _normalized_relative_leaf_map(value: Any) -> dict[str, str]:
    return {
        path: _normalize_leaf_preview(preview)
        for path, preview in _relative_leaf_map(value).items()
    }


def _join_relative_pointer(base_path: str, relative_path: str) -> str:
    if relative_path in {"", "/"}:
        return _pointer_path(base_path)
    return build_json_pointer([*_pointer_tokens(base_path), *_pointer_tokens(relative_path)])


def _item_match_score(
    page_item: Any,
    whole_item: Any,
    *,
    page_item_path: str,
    field_scope: Any,
) -> float:
    if isinstance(page_item, (str, int, float, bool)) or page_item is None:
        return 1.0 if page_item == whole_item else 0.0
    if type(page_item) is not type(whole_item):
        return 0.0
    page_leaves = _normalized_relative_leaf_map(page_item)
    whole_leaves = _normalized_relative_leaf_map(whole_item)
    if not page_leaves and not whole_leaves:
        return 1.0 if page_item == whole_item else 0.0
    common_paths = set(page_leaves) & set(whole_leaves)
    if not common_paths:
        return 0.0

    identity_paths = {
        relative_path
        for relative_path in common_paths
        if field_scope.scope_for(_join_relative_pointer(page_item_path, relative_path)) == "auto"
    }
    paths_to_compare = identity_paths or common_paths
    if any(page_leaves[path] != whole_leaves[path] for path in paths_to_compare):
        return 0.0
    return len(paths_to_compare) / max(len(page_leaves), len(whole_leaves), 1)


def _add_alias(aliases: list[tuple[str, str]], source_path: str, final_path: str) -> None:
    aliases.append((_pointer_path(source_path), _pointer_path(final_path)))


def _remap_trace(
    trace: _HybridMergeTrace,
    aliases: Sequence[tuple[str, str]],
) -> _HybridMergeTrace:
    if not aliases:
        return trace
    return _HybridMergeTrace(
        page_paths={_apply_path_alias(path, aliases) for path in trace.page_paths},
        whole_paths={_apply_path_alias(path, aliases) for path in trace.whole_paths},
        page_aliases=[
            (_apply_path_alias(source_path, aliases), _apply_path_alias(target_path, aliases))
            for source_path, target_path in trace.page_aliases
        ],
        whole_aliases=[
            (_apply_path_alias(source_path, aliases), _apply_path_alias(target_path, aliases))
            for source_path, target_path in trace.whole_aliases
        ],
    )


def _apply_path_alias(path: str, aliases: Sequence[tuple[str, str]]) -> str:
    normalized_path = _pointer_path(path)
    best_source: str | None = None
    best_target: str | None = None
    for source_path, target_path in aliases:
        if normalized_path == source_path or normalized_path.startswith(source_path + "/"):
            if best_source is None or len(source_path) > len(best_source):
                best_source = source_path
                best_target = target_path
    if best_source is None or best_target is None:
        return normalized_path
    if normalized_path == best_source:
        return best_target
    suffix_tokens = _pointer_tokens(normalized_path)[len(_pointer_tokens(best_source)) :]
    return build_json_pointer([*_pointer_tokens(best_target), *suffix_tokens])


def _remap_evidence_paths(
    evidence: Sequence[FieldEvidence],
    aliases: Sequence[tuple[str, str]],
) -> list[FieldEvidence]:
    if not aliases:
        return list(evidence)
    remapped: list[FieldEvidence] = []
    for item in evidence:
        remapped.append(replace(item, path=_apply_path_alias(item.path, aliases)))
    return remapped


def _remap_conflict_paths(
    conflicts: Sequence[MergeConflict],
    aliases: Sequence[tuple[str, str]],
) -> list[MergeConflict]:
    if not aliases:
        return list(conflicts)
    remapped: list[MergeConflict] = []
    for conflict in conflicts:
        remapped.append(replace(conflict, path=_apply_path_alias(conflict.path, aliases)))
    return remapped


def _should_prefer_whole_for_auto_scalar(
    *,
    path: str,
    page_value: Any,
    whole_value: Any,
    page_values_by_path: dict[str, set[str]],
    page_conflicts_by_path: dict[str, list[MergeConflict]],
) -> bool:
    if _is_missing_hybrid_value(page_value):
        return True
    if _is_missing_hybrid_value(whole_value):
        return False
    if page_value == whole_value:
        return False
    if len(page_values_by_path.get(_pointer_path(path), set())) > 1:
        return True
    return bool(page_conflicts_by_path.get(_pointer_path(path)))


def _reconcile_hybrid_value(
    *,
    page_value: Any,
    whole_value: Any,
    path: str,
    field_scope: Any,
    page_values_by_path: dict[str, set[str]],
    page_conflicts_by_path: dict[str, list[MergeConflict]],
) -> tuple[Any, _HybridMergeTrace]:
    trace = _HybridMergeTrace()
    pointer_path = _pointer_path(path)
    scope = field_scope.scope_for(pointer_path)

    if _is_missing_hybrid_value(page_value):
        if whole_value is None:
            return None, trace
        trace.include_whole_value(whole_value, path)
        return whole_value, trace
    if _is_missing_hybrid_value(whole_value):
        trace.include_page_value(page_value, path)
        return page_value, trace

    if isinstance(page_value, dict) or isinstance(whole_value, dict):
        if not isinstance(page_value, dict):
            trace.include_whole_value(whole_value, path)
            return whole_value, trace
        if not isinstance(whole_value, dict):
            trace.include_page_value(page_value, path)
            return page_value, trace
        merged: dict[str, Any] = {}
        for key in sorted(set(page_value) | set(whole_value)):
            child_path = _pointer_with_child(path, key)
            child_value, child_trace = _reconcile_hybrid_value(
                page_value=page_value.get(key),
                whole_value=whole_value.get(key),
                path=child_path,
                field_scope=field_scope,
                page_values_by_path=page_values_by_path,
                page_conflicts_by_path=page_conflicts_by_path,
            )
            if child_value is not None:
                merged[key] = child_value
            trace.merge(child_trace)
        return merged, trace

    if isinstance(page_value, list) or isinstance(whole_value, list):
        if not isinstance(page_value, list):
            trace.include_whole_value(whole_value, path)
            return whole_value, trace
        if not isinstance(whole_value, list):
            trace.include_page_value(page_value, path)
            return page_value, trace
        merged_list, list_trace = _reconcile_hybrid_list(
            page_items=page_value,
            whole_items=whole_value,
            path=path,
            field_scope=field_scope,
            page_values_by_path=page_values_by_path,
            page_conflicts_by_path=page_conflicts_by_path,
        )
        trace.merge(list_trace)
        return merged_list, trace

    prefer_whole = scope == "global" or (
        scope == "auto"
        and _should_prefer_whole_for_auto_scalar(
            path=pointer_path,
            page_value=page_value,
            whole_value=whole_value,
            page_values_by_path=page_values_by_path,
            page_conflicts_by_path=page_conflicts_by_path,
        )
    )
    prefer_page = scope in {"local", "span"}

    if page_value == whole_value:
        trace.page_paths.add(pointer_path)
        trace.whole_paths.add(pointer_path)
        return whole_value, trace
    if prefer_whole:
        trace.whole_paths.add(pointer_path)
        return whole_value, trace
    if prefer_page or scope == "auto":
        trace.page_paths.add(pointer_path)
        return page_value, trace
    trace.whole_paths.add(pointer_path)
    return whole_value, trace


def _reconcile_hybrid_list(
    *,
    page_items: list[Any],
    whole_items: list[Any],
    path: str,
    field_scope: Any,
    page_values_by_path: dict[str, set[str]],
    page_conflicts_by_path: dict[str, list[MergeConflict]],
) -> tuple[list[Any], _HybridMergeTrace]:
    trace = _HybridMergeTrace()
    scope = field_scope.scope_for(_pointer_path(path))
    if scope == "global":
        primary_branch = "whole"
        primary_items = whole_items
        secondary_items = page_items
    else:
        primary_branch = "page"
        primary_items = page_items
        secondary_items = whole_items

    used_secondary: set[int] = set()
    merged_items: list[Any] = []

    def _leaf_map_is_subset(subset_map: dict[str, str], superset_map: dict[str, str]) -> bool:
        return bool(subset_map) and all(
            superset_map.get(path) == value for path, value in subset_map.items()
        )

    def _duplicate_relation(
        existing_item: Any, candidate_item: Any
    ) -> Literal["equal", "existing_superset", "candidate_superset"] | None:
        if type(existing_item) is not type(candidate_item):
            return None
        if isinstance(existing_item, (str, int, float, bool)) or existing_item is None:
            return (
                "equal"
                if _normalize_leaf_preview(str(existing_item))
                == _normalize_leaf_preview(str(candidate_item))
                else None
            )
        existing_leaves = _normalized_relative_leaf_map(existing_item)
        candidate_leaves = _normalized_relative_leaf_map(candidate_item)
        if existing_leaves == candidate_leaves:
            return "equal"
        if isinstance(existing_item, dict) and isinstance(candidate_item, dict):
            if _leaf_map_is_subset(candidate_leaves, existing_leaves):
                return "existing_superset"
            if _leaf_map_is_subset(existing_leaves, candidate_leaves):
                return "candidate_superset"
        return None

    def _find_safe_duplicate(
        candidate_item: Any,
    ) -> tuple[int, Literal["equal", "existing_superset", "candidate_superset"]] | None:
        matches: list[tuple[int, Literal["equal", "existing_superset", "candidate_superset"]]] = []
        for idx, existing_item in enumerate(merged_items):
            relation = _duplicate_relation(existing_item, candidate_item)
            if relation is not None:
                matches.append((idx, relation))
        if len(matches) != 1:
            return None
        return matches[0]

    for primary_index, primary_item in enumerate(primary_items):
        best_secondary_index: int | None = None
        best_score = 0.0
        for secondary_index, secondary_item in enumerate(secondary_items):
            if secondary_index in used_secondary:
                continue
            page_item_path = _pointer_with_child(
                path,
                primary_index if primary_branch == "page" else secondary_index,
            )
            score = _item_match_score(
                primary_item if primary_branch == "page" else secondary_item,
                secondary_item if primary_branch == "page" else primary_item,
                page_item_path=page_item_path,
                field_scope=field_scope,
            )
            if score > best_score:
                best_score = score
                best_secondary_index = secondary_index

        final_index = len(merged_items)
        final_item_path = _pointer_with_child(path, final_index)
        primary_item_path = _pointer_with_child(path, primary_index)
        item_trace = _HybridMergeTrace()
        if primary_branch == "page":
            _add_alias(item_trace.page_aliases, primary_item_path, final_item_path)
        else:
            _add_alias(item_trace.whole_aliases, primary_item_path, final_item_path)

        if best_secondary_index is None or best_score <= 0.0:
            if primary_branch == "page":
                item_value, value_trace = _reconcile_hybrid_value(
                    page_value=primary_item,
                    whole_value=None,
                    path=final_item_path,
                    field_scope=field_scope,
                    page_values_by_path=page_values_by_path,
                    page_conflicts_by_path=page_conflicts_by_path,
                )
            else:
                item_value, value_trace = _reconcile_hybrid_value(
                    page_value=None,
                    whole_value=primary_item,
                    path=final_item_path,
                    field_scope=field_scope,
                    page_values_by_path=page_values_by_path,
                    page_conflicts_by_path=page_conflicts_by_path,
                )
        else:
            used_secondary.add(best_secondary_index)
            secondary_item = secondary_items[best_secondary_index]
            secondary_item_path = _pointer_with_child(path, best_secondary_index)
            if primary_branch == "page":
                _add_alias(item_trace.whole_aliases, secondary_item_path, final_item_path)
                item_value, value_trace = _reconcile_hybrid_value(
                    page_value=primary_item,
                    whole_value=secondary_item,
                    path=final_item_path,
                    field_scope=field_scope,
                    page_values_by_path=page_values_by_path,
                    page_conflicts_by_path=page_conflicts_by_path,
                )
            else:
                _add_alias(item_trace.page_aliases, secondary_item_path, final_item_path)
                item_value, value_trace = _reconcile_hybrid_value(
                    page_value=secondary_item,
                    whole_value=primary_item,
                    path=final_item_path,
                    field_scope=field_scope,
                    page_values_by_path=page_values_by_path,
                    page_conflicts_by_path=page_conflicts_by_path,
                )
        merged_items.append(item_value)
        item_trace.merge(value_trace)
        trace.merge(item_trace)

    for secondary_index, secondary_item in enumerate(secondary_items):
        if secondary_index in used_secondary:
            continue
        final_index = len(merged_items)
        final_item_path = _pointer_with_child(path, final_index)
        secondary_item_path = _pointer_with_child(path, secondary_index)
        item_trace = _HybridMergeTrace()
        if primary_branch == "page":
            _add_alias(item_trace.whole_aliases, secondary_item_path, final_item_path)
            item_value, value_trace = _reconcile_hybrid_value(
                page_value=None,
                whole_value=secondary_item,
                path=final_item_path,
                field_scope=field_scope,
                page_values_by_path=page_values_by_path,
                page_conflicts_by_path=page_conflicts_by_path,
            )
        else:
            _add_alias(item_trace.page_aliases, secondary_item_path, final_item_path)
            item_value, value_trace = _reconcile_hybrid_value(
                page_value=secondary_item,
                whole_value=None,
                path=final_item_path,
                field_scope=field_scope,
                page_values_by_path=page_values_by_path,
                page_conflicts_by_path=page_conflicts_by_path,
            )
        item_trace.merge(value_trace)
        duplicate = _find_safe_duplicate(item_value)
        if duplicate is not None:
            duplicate_index, relation = duplicate
            duplicate_path = _pointer_with_child(path, duplicate_index)
            item_trace = _remap_trace(item_trace, [(final_item_path, duplicate_path)])
            if relation == "candidate_superset":
                merged_items[duplicate_index] = item_value
            trace.merge(item_trace)
            continue

        merged_items.append(item_value)
        trace.merge(item_trace)

    return merged_items, trace


def _paths_overlap(path: str, leaf_paths: set[str]) -> bool:
    normalized = _pointer_path(path)
    for leaf_path in leaf_paths:
        if leaf_path == normalized or leaf_path.startswith(normalized + "/"):
            return True
        if normalized.startswith(leaf_path + "/"):
            return True
    return False


def _build_hybrid_evidence(
    *,
    merged_value: Any,
    page_evidence: Sequence[FieldEvidence],
    whole_evidence: Sequence[FieldEvidence],
    trace: _HybridMergeTrace,
    doc: Document | None = None,
) -> list[FieldEvidence]:
    remapped_page = _remap_evidence_paths(page_evidence, trace.page_aliases)
    remapped_whole = _remap_evidence_paths(whole_evidence, trace.whole_aliases)
    doc_page_texts = _extract_pdf_page_texts_for_backfill(doc)
    page_by_path: dict[str, list[FieldEvidence]] = defaultdict(list)
    whole_by_path: dict[str, list[FieldEvidence]] = defaultdict(list)
    for item in remapped_page:
        page_by_path[item.path].append(item)
    for item in remapped_whole:
        whole_by_path[item.path].append(item)

    selected: list[FieldEvidence] = []
    for path, raw_value in _iter_leaf_values(merged_value):
        pointer_path = _pointer_path(path)
        preview = _evidence_preview(raw_value)
        page_exact = [
            item for item in page_by_path.get(pointer_path, []) if item.value_preview == preview
        ]
        whole_exact = [
            item for item in whole_by_path.get(pointer_path, []) if item.value_preview == preview
        ]
        if page_exact:
            selected.extend(page_exact)
            continue
        doc_backfill = _backfill_document_page_evidence(
            path=pointer_path,
            preview=preview,
            whole_candidates=whole_exact or whole_by_path.get(pointer_path, []),
            doc_page_texts=doc_page_texts,
        )
        if doc_backfill:
            selected.extend(doc_backfill)
            continue
        if pointer_path in trace.page_paths and page_by_path.get(pointer_path):
            selected.extend(page_by_path[pointer_path])
            continue
        if whole_exact:
            selected.extend(whole_exact)
            continue
        if pointer_path in trace.whole_paths and whole_by_path.get(pointer_path):
            selected.extend(whole_by_path[pointer_path])
            continue
        if page_by_path.get(pointer_path):
            selected.extend(page_by_path[pointer_path])
            continue
        if whole_by_path.get(pointer_path):
            selected.extend(whole_by_path[pointer_path])
    return _dedupe_field_evidence(selected)


def _extract_pdf_page_texts_for_backfill(doc: Document | None) -> list[tuple[int, str]]:
    if doc is None or not doc.attachments:
        return []
    try:
        import fitz
    except ImportError:
        return []

    from .media.attachments import AttachmentKind

    page_texts: list[tuple[int, str]] = []
    for attachment in doc.attachments:
        if attachment.kind is not AttachmentKind.PDF:
            continue
        data = (
            attachment.source
            if isinstance(attachment.source, bytes)
            else attachment.source.read_bytes()
        )
        pdf_doc = fitz.open(stream=data, filetype="pdf")
        try:
            page_indices = (
                attachment.page_indices
                if attachment.page_indices is not None
                else tuple(range(len(pdf_doc)))
            )
            for page_index in page_indices:
                if 0 <= page_index < len(pdf_doc):
                    page_texts.append((page_index + 1, pdf_doc[page_index].get_text()))
        finally:
            pdf_doc.close()
    return page_texts


def _page_text_matches_preview(page_text: str, preview: str) -> bool:
    if not preview:
        return False
    if preview in page_text:
        return True

    normalized_preview = " ".join(preview.split())
    normalized_page_text = " ".join(page_text.split())
    if normalized_preview and normalized_preview in normalized_page_text:
        return True

    try:
        expected_number = float(preview.replace(",", ""))
    except ValueError:
        return False
    for match in re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?", page_text):
        try:
            actual_number = float(match.group().replace(",", ""))
        except ValueError:
            continue
        if actual_number == expected_number:
            return True
    return False


def _backfill_document_page_evidence(
    *,
    path: str,
    preview: str,
    whole_candidates: Sequence[FieldEvidence],
    doc_page_texts: Sequence[tuple[int, str]],
) -> list[FieldEvidence]:
    if not whole_candidates or not doc_page_texts:
        return []
    matched_pages = [
        page_index
        for page_index, page_text in doc_page_texts
        if _page_text_matches_preview(page_text, preview)
    ]
    if not matched_pages:
        return []
    template = whole_candidates[0]
    return [
        replace(
            template,
            path=path,
            value_preview=preview,
            page_index=page_index,
        )
        for page_index in matched_pages
    ]


def _build_hybrid_conflicts(
    *,
    merged_value: Any,
    page_conflicts: Sequence[MergeConflict],
    whole_conflicts: Sequence[MergeConflict],
    trace: _HybridMergeTrace,
) -> list[MergeConflict]:
    final_leaf_paths = _leaf_paths_for_value(merged_value, "")
    remapped_page = _remap_conflict_paths(page_conflicts, trace.page_aliases)
    remapped_whole = _remap_conflict_paths(whole_conflicts, trace.whole_aliases)
    filtered = [
        conflict
        for conflict in [*remapped_page, *remapped_whole]
        if _paths_overlap(conflict.path, final_leaf_paths)
    ]
    return filtered


def _branch_has_contribution(
    *,
    branch: Literal["page", "whole"],
    trace: _HybridMergeTrace,
    evidence: Sequence[FieldEvidence],
    merged_value: Any,
) -> bool:
    if branch == "page" and trace.page_paths:
        return True
    if branch == "whole" and trace.whole_paths:
        return True
    branch_evidence = (
        _remap_evidence_paths(evidence, trace.page_aliases)
        if branch == "page"
        else _remap_evidence_paths(evidence, trace.whole_aliases)
    )
    selected_paths = {path for path, value in _iter_leaf_values(merged_value) if value is not None}
    return any(item.path in selected_paths for item in branch_evidence)


def _merge_hybrid_states(
    *,
    page_state: _DocumentState,
    whole_state: _DocumentState,
    field_scope: Any,
    doc: Document | None = None,
    output_kind: _RootKind | None = None,
) -> _DocumentState:
    page_values_by_path = _evidence_value_index(page_state.doc_evidence)
    page_conflicts_by_path = _conflicts_by_path(page_state.conflicts)
    merged_value, trace = _reconcile_hybrid_value(
        page_value=page_state.merged_value,
        whole_value=whole_state.merged_value,
        path="",
        field_scope=field_scope,
        page_values_by_path=page_values_by_path,
        page_conflicts_by_path=page_conflicts_by_path,
    )
    if merged_value is None:
        merged_value = _default_merged_value(output_kind)

    doc_evidence = _build_hybrid_evidence(
        merged_value=merged_value,
        page_evidence=page_state.doc_evidence,
        whole_evidence=whole_state.doc_evidence,
        trace=trace,
        doc=doc,
    )
    conflicts = _build_hybrid_conflicts(
        merged_value=merged_value,
        page_conflicts=page_state.conflicts,
        whole_conflicts=whole_state.conflicts,
        trace=trace,
    )
    page_contributed = _branch_has_contribution(
        branch="page",
        trace=trace,
        evidence=page_state.doc_evidence,
        merged_value=merged_value,
    )
    whole_contributed = _branch_has_contribution(
        branch="whole",
        trace=trace,
        evidence=whole_state.doc_evidence,
        merged_value=merged_value,
    )
    contributing_states: list[_DocumentState] = []
    if page_contributed:
        contributing_states.append(page_state)
    if whole_contributed:
        contributing_states.append(whole_state)

    return _DocumentState(
        merged_value=merged_value,
        raw_outputs=[output for state in contributing_states for output in state.raw_outputs],
        all_flags=set().union(*(state.all_flags for state in contributing_states)),
        worst_score=max((state.worst_score for state in contributing_states), default=0),
        chunk_debug_entries=[
            entry for state in contributing_states for entry in state.chunk_debug_entries
        ],
        doc_evidence=doc_evidence,
        conflicts=conflicts,
    )


def _ensure_hybrid_output_kind_supported(output_kind: _RootKind | None) -> None:
    if output_kind in {"object", "array", "scalar"}:
        return
    raise NotImplementedError(
        "strategy.plan='hybrid' currently supports object, array, and scalar schemas only"
    )


def _run_hybrid_media_extraction[T](
    *,
    ctx: _ExtractionContext,
    doc: Document,
    page_media_chunks: list[MediaChunk],
    whole_media_chunks: list[MediaChunk],
    target: type[T] | TypeAdapter[T],
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
    native_kwargs: dict[str, Any],
) -> tuple[_DocumentState, str | None]:
    _ensure_hybrid_output_kind_supported(ctx.output_kind)

    media_requests = _build_media_inference_requests(ctx, doc, page_media_chunks)
    single_req = _build_single_inference_request(ctx, doc, whole_media_chunks)
    rendered_prompt_preview = single_req.prompt[:500] if debug else None
    batch_length = max(1, ctx.opts.batch_length)
    max_workers = max(1, ctx.opts.max_workers)
    page_state = _DocumentState()
    whole_state = _DocumentState()
    aggregate_chunk = MediaChunk(
        attachment=whole_media_chunks[0].attachment,
        attachment_index=None,
        page_index=None,
        text=whole_media_chunks[0].text,
    )

    for pass_index in range(max(1, ctx.opts.passes)):

        def _run_whole_branch() -> list[str]:
            return _infer_media_batch(ctx.provider, [single_req], batch_length, **native_kwargs)

        def _run_page_branch() -> list[str]:
            if max_workers <= 1:
                return _infer_media_batch(
                    ctx.provider, media_requests, batch_length, **native_kwargs
                )
            return _infer_media_batch_parallel(
                ctx.provider,
                media_requests,
                batch_length,
                max_workers,
                **native_kwargs,
            )

        if isinstance(ctx.provider, SupportsAsyncMediaInfer):
            whole_inferred = _run_whole_branch()
            page_inferred = _run_page_branch()
        else:
            whole_inferred, page_inferred = _run_parallel_pair(_run_whole_branch, _run_page_branch)
        whole_state = _accumulate_media_pass(
            pass_index=pass_index,
            state=whole_state,
            media_requests=[single_req],
            media_chunks=[aggregate_chunk],
            inferred=whole_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

        page_state = _accumulate_media_pass(
            pass_index=pass_index,
            state=page_state,
            media_requests=media_requests,
            media_chunks=page_media_chunks,
            inferred=page_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

    return (
        _merge_hybrid_states(
            page_state=page_state,
            whole_state=whole_state,
            field_scope=ctx.resolved_strategy.field_scope,
            doc=doc,
            output_kind=ctx.output_kind,
        ),
        rendered_prompt_preview,
    )


async def _arun_hybrid_media_extraction[T](
    *,
    ctx: _ExtractionContext,
    doc: Document,
    page_media_chunks: list[MediaChunk],
    whole_media_chunks: list[MediaChunk],
    target: type[T] | TypeAdapter[T],
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
    native_kwargs: dict[str, Any],
) -> tuple[_DocumentState, str | None]:
    _ensure_hybrid_output_kind_supported(ctx.output_kind)

    media_requests = _build_media_inference_requests(ctx, doc, page_media_chunks)
    single_req = _build_single_inference_request(ctx, doc, whole_media_chunks)
    rendered_prompt_preview = single_req.prompt[:500] if debug else None
    batch_length = max(1, ctx.opts.batch_length)
    max_workers = max(1, ctx.opts.max_workers)
    page_state = _DocumentState()
    whole_state = _DocumentState()
    aggregate_chunk = MediaChunk(
        attachment=whole_media_chunks[0].attachment,
        attachment_index=None,
        page_index=None,
        text=whole_media_chunks[0].text,
    )

    for pass_index in range(max(1, ctx.opts.passes)):

        def _run_whole_branch() -> Any:
            return _ainfer_media_batch(ctx.provider, [single_req], batch_length, **native_kwargs)

        def _run_page_branch() -> Any:
            if max_workers <= 1:
                return _ainfer_media_batch(
                    ctx.provider, media_requests, batch_length, **native_kwargs
                )
            return _ainfer_media_batch_parallel(
                ctx.provider,
                media_requests,
                batch_length,
                max_workers,
                **native_kwargs,
            )

        whole_inferred, page_inferred = await _arun_parallel_pair(
            _run_whole_branch,
            _run_page_branch,
        )
        whole_state = _accumulate_media_pass(
            pass_index=pass_index,
            state=whole_state,
            media_requests=[single_req],
            media_chunks=[aggregate_chunk],
            inferred=whole_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

        page_state = _accumulate_media_pass(
            pass_index=pass_index,
            state=page_state,
            media_requests=media_requests,
            media_chunks=page_media_chunks,
            inferred=page_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

    return (
        _merge_hybrid_states(
            page_state=page_state,
            whole_state=whole_state,
            field_scope=ctx.resolved_strategy.field_scope,
            doc=doc,
            output_kind=ctx.output_kind,
        ),
        rendered_prompt_preview,
    )


def _run_hybrid_text_extraction[T](
    *,
    ctx: _ExtractionContext,
    doc: Document,
    target: type[T] | TypeAdapter[T],
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
    native_kwargs: dict[str, Any],
) -> tuple[_DocumentState, str | None]:
    _ensure_hybrid_output_kind_supported(ctx.output_kind)

    chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
    whole_prompt = _build_single_text_prompt(ctx, doc)
    rendered_prompt_preview = whole_prompt[:500] if debug else None
    batch_length = max(1, ctx.opts.batch_length)
    max_workers = max(1, ctx.opts.max_workers)
    page_state = _DocumentState()
    whole_state = _DocumentState()
    aggregate_chunk = TextChunk(text=doc.text, start=0, end=len(doc.text))

    for pass_index in range(max(1, ctx.opts.passes)):

        def _run_whole_branch() -> list[str]:
            return _infer_batch(ctx.provider, [whole_prompt], batch_length, **native_kwargs)

        def _run_page_branch() -> list[str]:
            if max_workers <= 1:
                return _infer_batch(ctx.provider, chunk_prompts, batch_length, **native_kwargs)
            return _infer_batch_parallel(
                ctx.provider,
                chunk_prompts,
                batch_length,
                max_workers,
                **native_kwargs,
            )

        if _provider_supports_async_text(ctx.provider):
            whole_inferred = _run_whole_branch()
            page_inferred = _run_page_branch()
        else:
            whole_inferred, page_inferred = _run_parallel_pair(_run_whole_branch, _run_page_branch)
        _accumulate_pass(
            pass_index=pass_index,
            state=whole_state,
            chunks=[aggregate_chunk],
            inferred=whole_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

        _accumulate_pass(
            pass_index=pass_index,
            state=page_state,
            chunks=chunks,
            inferred=page_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

    return (
        _merge_hybrid_states(
            page_state=page_state,
            whole_state=whole_state,
            field_scope=ctx.resolved_strategy.field_scope,
            doc=doc,
            output_kind=ctx.output_kind,
        ),
        rendered_prompt_preview,
    )


async def _arun_hybrid_text_extraction[T](
    *,
    ctx: _ExtractionContext,
    doc: Document,
    target: type[T] | TypeAdapter[T],
    parse_options: ParseOptions | None,
    coerce_options: CoerceOptions | None,
    debug: bool,
    native_kwargs: dict[str, Any],
) -> tuple[_DocumentState, str | None]:
    _ensure_hybrid_output_kind_supported(ctx.output_kind)

    chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
    whole_prompt = _build_single_text_prompt(ctx, doc)
    rendered_prompt_preview = whole_prompt[:500] if debug else None
    batch_length = max(1, ctx.opts.batch_length)
    max_workers = max(1, ctx.opts.max_workers)
    page_state = _DocumentState()
    whole_state = _DocumentState()
    aggregate_chunk = TextChunk(text=doc.text, start=0, end=len(doc.text))

    for pass_index in range(max(1, ctx.opts.passes)):

        def _run_whole_branch() -> Any:
            return _ainfer_batch(ctx.provider, [whole_prompt], batch_length, **native_kwargs)

        def _run_page_branch() -> Any:
            if max_workers <= 1:
                return _ainfer_batch(ctx.provider, chunk_prompts, batch_length, **native_kwargs)
            return _ainfer_batch_parallel(
                ctx.provider,
                chunk_prompts,
                batch_length,
                max_workers,
                **native_kwargs,
            )

        whole_inferred, page_inferred = await _arun_parallel_pair(
            _run_whole_branch,
            _run_page_branch,
        )
        _accumulate_pass(
            pass_index=pass_index,
            state=whole_state,
            chunks=[aggregate_chunk],
            inferred=whole_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

        _accumulate_pass(
            pass_index=pass_index,
            state=page_state,
            chunks=chunks,
            inferred=page_inferred,
            target=target,
            ctx=ctx,
            parse_options=parse_options,
            coerce_options=coerce_options,
            debug=debug,
        )

    return (
        _merge_hybrid_states(
            page_state=page_state,
            whole_state=whole_state,
            field_scope=ctx.resolved_strategy.field_scope,
            doc=doc,
            output_kind=ctx.output_kind,
        ),
        rendered_prompt_preview,
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
        has_media = needs_media(doc.attachments)

        if has_media and ctx.resolved_strategy.plan != "hybrid":
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

            if ctx.resolved_strategy.plan == "hybrid":
                page_media_options = ctx.resolved_strategy.page_media or ctx.opts.media
                document_media_options = ctx.resolved_strategy.document_media or ctx.opts.media
                _ensure_native_pdf_support(
                    ctx.provider,
                    doc.attachments,
                    media_options=document_media_options,
                    branch_label="the whole-document branch",
                )
                page_media_chunks = chunk_attachments(
                    doc.attachments,
                    text=doc.text,
                    media_options=page_media_options,
                    provider_supports_native_pdf=False,
                )
                whole_media_chunks = chunk_attachments(
                    doc.attachments,
                    text=doc.text,
                    media_options=document_media_options,
                    provider_supports_native_pdf=native_pdf,
                )
                state, rendered_prompt_preview = _run_hybrid_media_extraction(
                    ctx=ctx,
                    doc=doc,
                    page_media_chunks=page_media_chunks,
                    whole_media_chunks=whole_media_chunks,
                    target=target,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    debug=debug,
                    native_kwargs=_native_kwargs,
                )
            else:
                _ensure_native_pdf_support(
                    ctx.provider,
                    doc.attachments,
                    media_options=ctx.opts.media,
                    branch_label="the current extraction mode",
                )
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
                    rendered_prompt_preview = single_req.prompt[:500] if debug else None
                    for _pass in range(max(1, ctx.opts.passes)):
                        inferred = _infer_media_batch(
                            ctx.provider, [single_req], batch_length, **_native_kwargs
                        )
                        # For "single" mode, use a synthetic aggregate chunk with None
                        # indices to avoid misattributing evidence to the first attachment.
                        aggregate_chunk = MediaChunk(
                            attachment=media_chunks[0].attachment if media_chunks else media_chunks,
                            attachment_index=None,
                            page_index=None,
                            text=media_chunks[0].text if media_chunks else "",
                        )
                        state = _accumulate_media_pass(
                            pass_index=_pass,
                            state=state,
                            media_requests=[single_req],
                            media_chunks=[aggregate_chunk],
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
            if ctx.resolved_strategy.plan == "hybrid":
                state, rendered_prompt_preview = _run_hybrid_text_extraction(
                    ctx=ctx,
                    doc=doc,
                    target=target,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    debug=debug,
                    native_kwargs=_native_kwargs,
                )
            else:
                chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
                rendered_prompt_preview = (
                    chunk_prompts[0][:500] if (debug and chunk_prompts) else None
                )
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

        if has_media and ctx.resolved_strategy.plan != "hybrid":
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

            if ctx.resolved_strategy.plan == "hybrid":
                page_media_options = ctx.resolved_strategy.page_media or ctx.opts.media
                document_media_options = ctx.resolved_strategy.document_media or ctx.opts.media
                _ensure_native_pdf_support(
                    ctx.provider,
                    doc.attachments,
                    media_options=document_media_options,
                    branch_label="the whole-document branch",
                )
                page_media_chunks = chunk_attachments(
                    doc.attachments,
                    text=doc.text,
                    media_options=page_media_options,
                    provider_supports_native_pdf=False,
                )
                whole_media_chunks = chunk_attachments(
                    doc.attachments,
                    text=doc.text,
                    media_options=document_media_options,
                    provider_supports_native_pdf=native_pdf,
                )
                state, rendered_prompt_preview = await _arun_hybrid_media_extraction(
                    ctx=ctx,
                    doc=doc,
                    page_media_chunks=page_media_chunks,
                    whole_media_chunks=whole_media_chunks,
                    target=target,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    debug=debug,
                    native_kwargs=_native_kwargs,
                )
            else:
                _ensure_native_pdf_support(
                    ctx.provider,
                    doc.attachments,
                    media_options=ctx.opts.media,
                    branch_label="the current extraction mode",
                )
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
                    rendered_prompt_preview = single_req.prompt[:500] if debug else None
                    for _pass in range(max(1, ctx.opts.passes)):
                        inferred = await _ainfer_media_batch(
                            ctx.provider, [single_req], batch_length, **_native_kwargs
                        )
                        aggregate_chunk = MediaChunk(
                            attachment=media_chunks[0].attachment if media_chunks else media_chunks,
                            attachment_index=None,
                            page_index=None,
                            text=media_chunks[0].text if media_chunks else "",
                        )
                        state = _accumulate_media_pass(
                            pass_index=_pass,
                            state=state,
                            media_requests=[single_req],
                            media_chunks=[aggregate_chunk],
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
            if ctx.resolved_strategy.plan == "hybrid":
                state, rendered_prompt_preview = await _arun_hybrid_text_extraction(
                    ctx=ctx,
                    doc=doc,
                    target=target,
                    parse_options=parse_options,
                    coerce_options=coerce_options,
                    debug=debug,
                    native_kwargs=_native_kwargs,
                )
            else:
                chunks, chunk_prompts = _prepare_document_chunks_and_prompts(ctx, doc)
                rendered_prompt_preview = (
                    chunk_prompts[0][:500] if (debug and chunk_prompts) else None
                )
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
