from __future__ import annotations

# ruff: noqa: E402
import io
import string

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from parsantic.extract.media.preprocessing import (  # noqa: E402
    extract_pdf_page_texts,
    file_hash,
    has_text_layer,
    normalize_image,
    prepare_pdf,
    rasterize_pdf,
)


def _make_text_pdf(text: str = "Hello World test document") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_image_only_pdf() -> bytes:
    image = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=image_bytes)
    data = doc.tobytes()
    doc.close()
    return data


def _make_multipage_text_pdf(pages: int = 3) -> bytes:
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {index}")
    data = doc.tobytes()
    doc.close()
    return data


def _make_test_image(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    color = "red" if mode != "RGBA" else (255, 0, 0, 127)
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_has_text_layer_returns_true_for_text_pdf():
    pdf = _make_text_pdf()
    assert has_text_layer(pdf) is True


def test_has_text_layer_returns_false_for_image_only_pdf():
    pdf = _make_image_only_pdf()
    assert has_text_layer(pdf) is False


def test_rasterize_pdf_returns_jpeg_bytes_for_each_page_by_default():
    pdf = _make_multipage_text_pdf(pages=2)
    rendered = rasterize_pdf(pdf)

    assert [index for index, _ in rendered] == [0, 1]
    for _, data in rendered:
        assert data.startswith(b"\xff\xd8\xff")


def test_rasterize_pdf_page_indices_select_specific_pages_as_png():
    pdf = _make_multipage_text_pdf(pages=4)
    rendered = rasterize_pdf(pdf, page_indices=(1, 3), raster_format="png")

    assert [index for index, _ in rendered] == [1, 3]
    for _, data in rendered:
        assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_rasterize_pdf_raises_for_out_of_range_index():
    pdf = _make_multipage_text_pdf(pages=2)

    with pytest.raises(ValueError, match="out of range"):
        rasterize_pdf(pdf, page_indices=(2,))


def test_normalize_image_resizes_oversized_image():
    image_bytes = _make_test_image(width=4000, height=1000)
    normalized = normalize_image(image_bytes, max_dim=2048)

    img = Image.open(io.BytesIO(normalized))
    assert img.size == (2048, 512)


def test_normalize_image_preserves_small_image_dimensions():
    image_bytes = _make_test_image(width=100, height=80)
    normalized = normalize_image(image_bytes, max_dim=2048)

    img = Image.open(io.BytesIO(normalized))
    assert img.size == (100, 80)


def test_normalize_image_converts_rgba_to_rgb():
    image_bytes = _make_test_image(mode="RGBA")
    normalized = normalize_image(image_bytes, max_dim=2048)

    img = Image.open(io.BytesIO(normalized))
    assert img.mode == "RGB"


def test_file_hash_returns_consistent_hex_string():
    payload = b"hash-me"
    digest_a = file_hash(payload)
    digest_b = file_hash(payload)

    assert digest_a == digest_b
    assert len(digest_a) == 64
    assert all(char in string.hexdigits for char in digest_a)


def test_prepare_pdf_caches_repeated_page_text_extraction(monkeypatch):
    pdf = _make_multipage_text_pdf(pages=2)
    open_calls = 0
    original_open = fitz.open

    def _counting_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(fitz, "open", _counting_open)

    first = prepare_pdf(pdf)
    second = prepare_pdf(pdf)
    texts = extract_pdf_page_texts(pdf)

    assert first is second
    assert [text for _, text in texts] == ["Page 0", "Page 1"]
    assert open_calls == 1
