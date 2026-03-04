from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from parsantic.extract.media.attachments import Attachment
from parsantic.extract.media.chunking import MediaChunk, chunk_attachments, needs_media


def test_chunk_attachments_single_image_creates_one_chunk():
    attachment = Attachment.image(b"fake-image-bytes", name="receipt")

    chunks = chunk_attachments([attachment])

    assert len(chunks) == 1
    assert chunks[0].attachment is attachment
    assert chunks[0].attachment_index == 0
    assert chunks[0].page_index is None


def test_chunk_attachments_pdf_with_page_indices_creates_single_native_chunk_with_hint():
    attachment = Attachment.pdf(b"%PDF-1.4", page_indices=(0, 2, 4))

    chunks = chunk_attachments([attachment], text="context")

    assert len(chunks) == 1
    assert chunks[0].attachment is attachment
    assert chunks[0].attachment_index == 0
    assert chunks[0].page_index is None
    assert chunks[0].text == "context\n\nFocus on pages: 1, 3, 5."


def test_chunk_attachments_pdf_without_page_indices_creates_single_chunk():
    attachment = Attachment.pdf(b"%PDF-1.4")

    chunks = chunk_attachments([attachment])

    assert len(chunks) == 1
    assert chunks[0].attachment is attachment
    assert chunks[0].attachment_index == 0
    assert chunks[0].page_index is None


def test_chunk_attachments_mixed_image_and_pdf():
    image = Attachment.image(b"image-data")
    pdf = Attachment.pdf(b"%PDF-1.4", page_indices=(1, 3))

    chunks = chunk_attachments([image, pdf])

    assert len(chunks) == 2
    assert chunks[0].attachment is image
    assert chunks[0].attachment_index == 0
    assert chunks[0].page_index is None
    assert chunks[1].attachment is pdf
    assert chunks[1].attachment_index == 1
    assert chunks[1].page_index is None
    assert chunks[1].text == "Focus on pages: 2, 4."


def test_chunk_attachments_empty_returns_empty_list():
    assert chunk_attachments([]) == []


def test_needs_media_with_attachments_returns_true():
    assert needs_media([Attachment.image(b"image-data")]) is True


def test_needs_media_with_no_attachments_returns_false():
    assert needs_media([]) is False


def test_media_chunk_is_immutable():
    chunk = MediaChunk(
        attachment=Attachment.image(b"image-data"),
        attachment_index=0,
        page_index=None,
    )

    with pytest.raises(FrozenInstanceError):
        chunk.text = "updated"


def test_media_chunk_carries_text_context():
    chunk = chunk_attachments([Attachment.image(b"image-data")], text="extract totals")[0]

    assert chunk.text == "extract totals"
