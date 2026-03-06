from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .attachments import Attachment, AttachmentKind

if TYPE_CHECKING:
    from parsantic.extract.options import MediaOptions

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MediaChunk:
    """A single chunk for multimodal extraction - one page or one image.

    When ``attachment_index`` is ``None``, the chunk represents an aggregate
    of multiple attachments (e.g., the "single" media strategy).
    """

    attachment: Attachment
    attachment_index: int | None
    page_index: int | None
    text: str = ""


def _should_rasterize_pdf(
    pdf_mode: Literal["auto", "native", "raster"],
    provider_supports_native_pdf: bool,
) -> bool:
    """Decide whether to rasterize a PDF attachment to images."""
    if pdf_mode == "native":
        return False
    if pdf_mode == "raster":
        return True
    # auto: rasterize unless provider supports native PDF
    return not provider_supports_native_pdf


def _rasterize_pdf_chunks(
    attachment: Attachment,
    att_idx: int,
    *,
    dpi: int,
    text: str,
    strict: bool = False,
    raster_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> list[MediaChunk]:
    """Rasterize PDF pages into image MediaChunks."""
    try:
        from .preprocessing import rasterize_pdf

        source = attachment.source
        page_indices = attachment.page_indices
        pages = rasterize_pdf(
            source,
            dpi=dpi,
            page_indices=page_indices,
            raster_format=raster_format,
            jpeg_quality=jpeg_quality,
        )
    except ImportError:
        if strict:
            raise ImportError(
                "pdf_mode='raster' requires vision dependencies. "
                'Install with: pip install "parsantic[vision]"'
            ) from None
        logger.warning(
            "Vision deps not installed; sending PDF natively. "
            "Install parsantic[vision] for PDF rasterization."
        )
        return _native_pdf_chunks(attachment, att_idx, text=text)

    chunks: list[MediaChunk] = []
    for page_idx, image_bytes in pages:
        mime = "image/jpeg" if raster_format == "jpeg" else "image/png"
        page_attachment = Attachment.image(image_bytes, mime_type=mime, name=attachment.name)
        chunks.append(
            MediaChunk(
                attachment=page_attachment,
                attachment_index=att_idx,
                page_index=page_idx,
                text=text,
            )
        )
    return chunks


def _native_pdf_chunks(
    attachment: Attachment,
    att_idx: int,
    *,
    text: str,
) -> list[MediaChunk]:
    """Create a single chunk for a PDF sent natively (no rasterization)."""
    hint_text: str = text
    if attachment.page_indices is not None:
        pages_display = ", ".join(str(p + 1) for p in attachment.page_indices)
        hint_text = (
            f"{text}\n\nFocus on pages: {pages_display}."
            if text
            else f"Focus on pages: {pages_display}."
        )
    return [
        MediaChunk(
            attachment=attachment,
            attachment_index=att_idx,
            page_index=None,
            text=hint_text,
        )
    ]


def _normalize_image_chunk(
    attachment: Attachment,
    att_idx: int,
    *,
    max_dim: int,
    text: str,
) -> MediaChunk:
    """Optionally normalize an image attachment before creating a chunk."""
    if max_dim > 0:
        source = attachment.source
        if isinstance(source, Path):
            source = source.read_bytes()
        if isinstance(source, bytes):
            try:
                from .preprocessing import normalize_image

                normalized = normalize_image(source, max_dim=max_dim)
                attachment = Attachment.image(
                    normalized, mime_type="image/png", name=attachment.name
                )
            except ImportError:
                pass  # No vision deps — send image as-is
            except Exception:
                logger.debug("Image normalization failed, sending as-is", exc_info=True)

    return MediaChunk(
        attachment=attachment,
        attachment_index=att_idx,
        page_index=None,
        text=text,
    )


def chunk_attachments(
    attachments: Sequence[Attachment],
    *,
    text: str = "",
    media_options: MediaOptions | None = None,
    provider_supports_native_pdf: bool = True,
) -> list[MediaChunk]:
    """Split attachments into individual media chunks.

    When *media_options* is provided and vision deps are installed, PDFs are
    rasterized to per-page image chunks and images are normalized.
    """
    # Import here to avoid circular imports at module level
    from parsantic.extract.options import MediaOptions as _MO

    opts = media_options or _MO()
    chunks: list[MediaChunk] = []

    for att_idx, attachment in enumerate(attachments):
        if attachment.kind is AttachmentKind.IMAGE:
            chunks.append(
                _normalize_image_chunk(attachment, att_idx, max_dim=opts.max_image_dim, text=text)
            )
        elif attachment.kind is AttachmentKind.PDF:
            if _should_rasterize_pdf(opts.pdf_mode, provider_supports_native_pdf):
                chunks.extend(
                    _rasterize_pdf_chunks(
                        attachment,
                        att_idx,
                        dpi=opts.raster_dpi,
                        text=text,
                        strict=(opts.pdf_mode == "raster"),
                        raster_format=opts.raster_format,
                        jpeg_quality=opts.jpeg_quality,
                    )
                )
            else:
                chunks.extend(_native_pdf_chunks(attachment, att_idx, text=text))

    return chunks


def needs_media(attachments: Sequence[Attachment]) -> bool:
    """Return True if any attachments are present."""
    return bool(attachments)
