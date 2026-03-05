from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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


def has_text_layer(source: Path | bytes) -> bool:
    """Check if a PDF has a usable text layer.

    Returns True if any page has extractable text (>10 chars after strip).
    """
    _check_pymupdf()
    import fitz

    data = source if isinstance(source, bytes) else source.read_bytes()
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for page in doc:
            text = page.get_text().strip()
            if len(text) > 10:
                return True
        return False
    finally:
        doc.close()


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
                # Convert pixmap to PIL Image for JPEG.
                from PIL import Image

                # Handle alpha channel: drop it before creating RGB image.
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                results.append((page_idx, buf.getvalue()))
            else:
                results.append((page_idx, pix.tobytes("png")))
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
        # Fix EXIF orientation.
        img = ImageOps.exif_transpose(img)

        # Convert to RGB (handles CMYK, RGBA, palette, etc.).
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Resize if largest dimension exceeds max_dim.
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
    """Return hex SHA-256 of data, useful for caching."""
    return hashlib.sha256(data).hexdigest()
