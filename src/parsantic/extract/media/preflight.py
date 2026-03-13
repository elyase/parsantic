from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .attachments import Attachment
from .preprocessing import prepare_pdf, score_text_quality


@dataclass(slots=True)
class PageQuality:
    page_index: int
    text_char_count: int
    text_quality_score: float
    has_tables: bool
    has_images: bool
    is_scanned: bool
    recommended_mode: Literal["text_only", "image_only", "fused"]
    text_preview: str = ""


@dataclass(slots=True)
class PreflightResult:
    page_count: int
    pages: list[PageQuality]
    has_text_layer: bool
    text_layer_quality: float
    recommended_plan: Literal["fused", "text_only", "image_only", "hybrid"]
    estimated_tokens: int


def analyze_pdf(
    source: Path | bytes,
    *,
    page_indices: tuple[int, ...] | None = None,
) -> PreflightResult:
    prepared = prepare_pdf(source, page_indices=page_indices)
    pages: list[PageQuality] = []
    for page in prepared.pages:
        text_quality = score_text_quality(page.text)
        is_scanned = page.has_images and text_quality < 0.15
        if is_scanned:
            recommended_mode: Literal["text_only", "image_only", "fused"] = "image_only"
        elif text_quality >= 0.7 and not page.has_images and not page.has_tables:
            recommended_mode = "text_only"
        elif text_quality <= 0.25 and page.has_images:
            recommended_mode = "image_only"
        else:
            recommended_mode = "fused"
        pages.append(
            PageQuality(
                page_index=page.page_index,
                text_char_count=len(page.text),
                text_quality_score=text_quality,
                text_preview=_page_preview(page.text),
                has_tables=page.has_tables,
                has_images=page.has_images,
                is_scanned=is_scanned,
                recommended_mode=recommended_mode,
            )
        )

    recommended_modes = {page.recommended_mode for page in pages}
    if recommended_modes == {"text_only"}:
        recommended_plan: Literal["fused", "text_only", "image_only", "hybrid"] = "text_only"
    elif recommended_modes == {"image_only"}:
        recommended_plan = "image_only"
    elif recommended_modes == {"fused"}:
        recommended_plan = "fused"
    else:
        recommended_plan = "hybrid"

    text_layer_quality = (
        sum(page.text_quality_score for page in pages) / len(pages) if pages else 0.0
    )
    estimated_tokens = sum(
        (page.text_char_count // 4) + (800 if page.recommended_mode != "text_only" else 0)
        for page in pages
    )
    return PreflightResult(
        page_count=len(pages),
        pages=pages,
        has_text_layer=any(page.text_char_count > 10 for page in pages),
        text_layer_quality=text_layer_quality,
        recommended_plan=recommended_plan,
        estimated_tokens=estimated_tokens,
    )


def analyze_pdf_source(
    source: Path | bytes,
    *,
    page_indices: tuple[int, ...] | None = None,
) -> PreflightResult:
    return analyze_pdf(source, page_indices=page_indices)


def analyze_pdf_attachment(attachment: Attachment) -> PreflightResult:
    return analyze_pdf(attachment.source, page_indices=attachment.page_indices)


def _page_preview(text: str, *, head_chars: int = 220, tail_chars: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= head_chars + tail_chars + 5:
        return normalized
    return f"{normalized[:head_chars]} ... {normalized[-tail_chars:]}"
