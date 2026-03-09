from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock


@dataclass(frozen=True, slots=True)
class PreparedPdfPage:
    page_index: int
    text: str
    has_images: bool
    has_tables: bool


@dataclass(slots=True)
class PreparedPdf:
    source_hash: str
    page_count: int
    pages: tuple[PreparedPdfPage, ...]
    raster_cache: dict[
        tuple[int, tuple[int, ...] | None, str, int], tuple[tuple[int, bytes], ...]
    ] = field(default_factory=dict)


_PREPARED_PDF_CACHE: dict[tuple[str, tuple[int, ...] | None], PreparedPdf] = {}
_PREPARED_PDF_LOCK = Lock()


def _check_pymupdf() -> None:
    """Raise ImportError if PyMuPDF (fitz) is not installed."""
    try:
        import fitz  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyMuPDF required for PDF operations. Install with: pip install parsantic[vision]"
        ) from None


def _check_pillow() -> None:
    """Raise ImportError if Pillow (PIL) is not installed."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise ImportError(
            "Pillow required for image operations. Install with: pip install parsantic[vision]"
        ) from None


def score_text_quality(text: str) -> float:
    """Return a coarse 0-1 score for extracted page text."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    total = len(stripped)
    alpha_ratio = sum(char.isalpha() for char in stripped) / total
    whitespace_ratio = sum(char.isspace() for char in stripped) / total
    char_count_score = min(total / 500.0, 1.0)
    score = (char_count_score * 0.45) + (alpha_ratio * 0.35) + (whitespace_ratio * 0.20)
    return max(0.0, min(score, 1.0))


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


def prepare_pdf(
    source: Path | bytes,
    *,
    page_indices: tuple[int, ...] | None = None,
) -> PreparedPdf:
    _check_pymupdf()
    import fitz

    data = source if isinstance(source, bytes) else source.read_bytes()
    key = (file_hash(data), tuple(page_indices) if page_indices is not None else None)
    cached = _PREPARED_PDF_CACHE.get(key)
    if cached is not None:
        return cached

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        selected_pages = list(page_indices) if page_indices is not None else list(range(len(doc)))
        pages: list[PreparedPdfPage] = []
        for page_index in selected_pages:
            if page_index < 0 or page_index >= len(doc):
                raise ValueError(
                    f"Page index {page_index} out of range (document has {len(doc)} pages)"
                )
            page = doc[page_index]
            text = page.get_text().strip()
            pages.append(
                PreparedPdfPage(
                    page_index=page_index,
                    text=text,
                    has_images=bool(page.get_images(full=True)),
                    has_tables=_page_has_tables(page, text),
                )
            )
        prepared = PreparedPdf(
            source_hash=key[0],
            page_count=len(selected_pages),
            pages=tuple(pages),
        )
    finally:
        doc.close()

    with _PREPARED_PDF_LOCK:
        cached = _PREPARED_PDF_CACHE.get(key)
        if cached is not None:
            return cached
        _PREPARED_PDF_CACHE[key] = prepared
    return prepared


def extract_pdf_page_texts(
    source: Path | bytes,
    *,
    page_indices: tuple[int, ...] | None = None,
) -> list[tuple[int, str]]:
    prepared = prepare_pdf(source, page_indices=page_indices)
    return [(page.page_index, page.text) for page in prepared.pages]


def score_pdf_text_quality(
    source: Path | bytes,
    *,
    page_indices: tuple[int, ...] | None = None,
) -> float:
    prepared = prepare_pdf(source, page_indices=page_indices)
    if not prepared.pages:
        return 0.0
    return sum(score_text_quality(page.text) for page in prepared.pages) / len(prepared.pages)


def has_text_layer(source: Path | bytes) -> bool:
    """Check if a PDF has a usable text layer."""
    prepared = prepare_pdf(source)
    if not prepared.pages:
        return False
    return any(score_text_quality(page.text) > 0.1 for page in prepared.pages)


def rasterize_pdf(
    source: Path | bytes,
    *,
    dpi: int = 200,
    page_indices: tuple[int, ...] | None = None,
    raster_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> list[tuple[int, bytes]]:
    """Rasterize PDF pages to PNG or JPEG bytes.

    Returns list of (page_index, image_bytes) tuples. page_index is 0-based.
    """
    _check_pymupdf()
    import fitz

    data = source if isinstance(source, bytes) else source.read_bytes()
    page_key = tuple(page_indices) if page_indices is not None else None
    prepared = prepare_pdf(source, page_indices=page_indices)
    cache_key = (dpi, page_key, raster_format, jpeg_quality)
    cached = prepared.raster_cache.get(cache_key)
    if cached is not None:
        return list(cached)
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        pages_to_render = list(page_indices) if page_indices is not None else list(range(len(doc)))
        results: list[tuple[int, bytes]] = []
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_idx in pages_to_render:
            if page_idx < 0 or page_idx >= len(doc):
                raise ValueError(
                    f"Page index {page_idx} out of range (document has {len(doc)} pages)"
                )
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix)
            if raster_format == "jpeg":
                from PIL import Image

                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                results.append((page_idx, buf.getvalue()))
            else:
                results.append((page_idx, pix.tobytes("png")))
        prepared.raster_cache[cache_key] = tuple(results)
        return results
    finally:
        doc.close()


def normalize_image(
    data: bytes,
    *,
    max_dim: int = 2048,
) -> bytes:
    """Normalize an image: RGB conversion, EXIF orientation fix, resize if needed.

    Returns PNG bytes.
    """
    _check_pillow()
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def file_hash(data: bytes) -> str:
    """Return hex SHA-256 of *data*, useful for caching."""
    return hashlib.sha256(data).hexdigest()
