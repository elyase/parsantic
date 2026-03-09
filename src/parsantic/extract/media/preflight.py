from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .attachments import Attachment
from .preprocessing import extract_pdf_page_texts, score_text_quality


@dataclass(slots=True)
class PageQuality:
    page_index: int
    text_char_count: int
    text_quality_score: float
    has_tables: bool
    has_images: bool
    is_scanned: bool
    recommended_mode: Literal["text_only", "image_only", "fused"]


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
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyMuPDF required for PDF preflight. Install with: pip install parsantic[vision]"
        ) from exc

    data = source if isinstance(source, bytes) else source.read_bytes()
    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        selected_pages = list(page_indices) if page_indices is not None else list(range(len(pdf)))
        pages: list[PageQuality] = []
        page_texts = dict(extract_pdf_page_texts(source, page_indices=page_indices))

        for page_index in selected_pages:
            page = pdf[page_index]
            text = page_texts.get(page_index, "")
            text_quality = score_text_quality(text)
            has_images = bool(page.get_images(full=True))
            has_tables = _page_has_tables(page, text)
            is_scanned = has_images and text_quality < 0.15
            if is_scanned:
                recommended_mode: Literal["text_only", "image_only", "fused"] = "image_only"
            elif text_quality >= 0.7 and not has_images and not has_tables:
                recommended_mode = "text_only"
            elif text_quality <= 0.25 and has_images:
                recommended_mode = "image_only"
            else:
                recommended_mode = "fused"
            pages.append(
                PageQuality(
                    page_index=page_index,
                    text_char_count=len(text),
                    text_quality_score=text_quality,
                    has_tables=has_tables,
                    has_images=has_images,
                    is_scanned=is_scanned,
                    recommended_mode=recommended_mode,
                )
            )
    finally:
        pdf.close()

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


def _page_has_tables(page: object, page_text: str) -> bool:
    find_tables = getattr(page, "find_tables", None)
    if callable(find_tables):
        try:
            tables = find_tables()
            if bool(getattr(tables, "tables", ())):
                return True
        except Exception:
            pass
    lines = [line for line in page_text.splitlines() if line.strip()]
    return sum(1 for line in lines if line.count("  ") >= 2 or "\t" in line) >= 3
