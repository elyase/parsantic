from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from .media.preflight import PageQuality, PreflightResult

type SelectionMode = str

_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "body",
    "date",
    "description",
    "details",
    "document",
    "field",
    "fields",
    "for",
    "from",
    "has",
    "id",
    "if",
    "in",
    "into",
    "is",
    "it",
    "items",
    "list",
    "name",
    "of",
    "on",
    "or",
    "page",
    "record",
    "records",
    "schema",
    "section",
    "should",
    "status",
    "text",
    "that",
    "the",
    "their",
    "this",
    "to",
    "type",
    "value",
    "values",
    "with",
    "primary",
}

_MAX_SCHEMA_FIELDS = 12
_MAX_SCHEMA_TERMS = 32
_MAX_PAGE_MATCH_RATIO = 0.6
_MIN_TEXT_LAYER_QUALITY = 0.25


@dataclass(frozen=True, slots=True)
class PageSelection:
    page_indices: tuple[int, ...] | None
    fallback_reason: str | None
    reason_codes_by_page: dict[int, tuple[str, ...]]
    selected_page_count: int
    analyzed_page_count: int


def select_pdf_pages(
    analysis: PreflightResult,
    target: type[Any] | TypeAdapter[Any],
    *,
    window: int = 1,
    max_pages: int | None = None,
    mode: SelectionMode = "optimize",
) -> PageSelection:
    if mode != "optimize":
        raise ValueError("select_pdf_pages currently supports only mode='optimize'")
    if window < 0:
        raise ValueError("window must be >= 0")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be >= 1")

    page_count = analysis.page_count
    if page_count == 0:
        return _fallback(analysis, "no_pages")

    if analysis.text_layer_quality < _MIN_TEXT_LAYER_QUALITY:
        return _fallback(analysis, "low_text_quality")

    schema_fields, terms = _schema_terms(target)
    if schema_fields == 0:
        return _fallback(analysis, "no_schema_fields")
    if schema_fields > _MAX_SCHEMA_FIELDS:
        return _fallback(analysis, "broad_schema")
    if not terms:
        return _fallback(analysis, "weak_schema_terms")
    if len(terms) > _MAX_SCHEMA_TERMS:
        return _fallback(analysis, "too_many_schema_terms")

    reason_codes_by_page: dict[int, tuple[str, ...]] = {}
    matched_pages: set[int] = set()
    for page in analysis.pages:
        matched_terms = _matched_terms(page, terms)
        if not matched_terms:
            continue
        matched_pages.add(page.page_index)
        reason_codes_by_page[page.page_index] = tuple(matched_terms[:5])

    if not matched_pages:
        return _fallback(analysis, "no_matching_pages")

    if len(matched_pages) / page_count > _MAX_PAGE_MATCH_RATIO:
        return _fallback(analysis, "too_many_matches")
    selected_pages = _expand_pages(
        matched_pages,
        page_count=page_count,
        window=window,
    )
    if max_pages is not None and len(selected_pages) > max_pages:
        return _fallback(analysis, "too_many_matches")

    return PageSelection(
        page_indices=tuple(selected_pages),
        fallback_reason=None,
        reason_codes_by_page=reason_codes_by_page,
        selected_page_count=len(selected_pages),
        analyzed_page_count=page_count,
    )


def _fallback(analysis: PreflightResult, reason: str) -> PageSelection:
    return PageSelection(
        page_indices=None,
        fallback_reason=reason,
        reason_codes_by_page={},
        selected_page_count=analysis.page_count,
        analyzed_page_count=analysis.page_count,
    )


def _schema_terms(target: type[Any] | TypeAdapter[Any]) -> tuple[int, tuple[str, ...]]:
    adapter = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    schema = adapter.json_schema()
    texts: list[str] = []
    field_count = _collect_schema_text(schema, texts, defs=schema.get("$defs", {}))
    terms: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for term in _tokenize(text):
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return field_count, tuple(terms)


def _collect_schema_text(
    schema: Any,
    texts: list[str],
    *,
    defs: dict[str, Any],
    seen_refs: set[str] | None = None,
) -> int:
    if not isinstance(schema, dict):
        return 0

    resolved_ref = schema.get("$ref")
    if isinstance(resolved_ref, str):
        if seen_refs is None:
            seen_refs = set()
        if resolved_ref in seen_refs:
            return 0
        resolved = _resolve_ref(resolved_ref, defs)
        if resolved is None:
            return 0
        return _collect_schema_text(
            resolved,
            texts,
            defs=defs,
            seen_refs={*seen_refs, resolved_ref},
        )

    for subschema in schema.get("allOf", ()):
        field_count = _collect_schema_text(
            subschema,
            texts,
            defs=defs,
            seen_refs=seen_refs,
        )
        if field_count:
            return field_count

    field_count = 0
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        for field_name, subschema in properties.items():
            field_count += 1
            texts.append(str(field_name))
            title = subschema.get("title")
            description = subschema.get("description")
            if title:
                texts.append(str(title))
            if description:
                texts.append(str(description))
            field_count += _collect_schema_text(
                subschema,
                texts,
                defs=defs,
                seen_refs=seen_refs,
            )
    elif schema_type == "array":
        field_count += _collect_schema_text(
            schema.get("items"),
            texts,
            defs=defs,
            seen_refs=seen_refs,
        )
    else:
        title = schema.get("title")
        description = schema.get("description")
        if title:
            texts.append(str(title))
        if description:
            texts.append(str(description))

    return field_count


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        return None
    key = ref[len(prefix) :]
    return defs.get(key)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
    normalized: list[str] = []
    for token in tokens:
        token = token.replace("_", " ")
        for part in token.split():
            if part in _STOPWORDS:
                continue
            normalized.append(part)
    return normalized


def _matched_terms(page: PageQuality, terms: tuple[str, ...]) -> list[str]:
    preview = page.text_preview.lower()
    hits: list[str] = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", preview):
            hits.append(f"term:{term}")
    return hits


def _expand_pages(pages: set[int], *, page_count: int, window: int) -> list[int]:
    expanded: set[int] = set()
    for page in pages:
        for offset in range(-window, window + 1):
            candidate = page + offset
            if 0 <= candidate < page_count:
                expanded.add(candidate)
    return sorted(expanded)
