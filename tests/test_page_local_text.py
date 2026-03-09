from __future__ import annotations

import pytest
from pydantic import BaseModel

from parsantic.extract import Document, ExtractOptions
from parsantic.extract.media.attachments import Attachment
from parsantic.extract.media.chunking import chunk_attachments
from parsantic.extract.options import MediaOptions
from parsantic.extract.pipeline import _build_extraction_context, _build_media_inference_requests


class _PageModel(BaseModel):
    field: str = ""


class _MediaProvider:
    supported_attachment_kinds = frozenset({"image", "pdf"})

    def infer(self, batch_prompts, **kwargs):
        return ['{"field": "unused"}' for _ in batch_prompts]

    def infer_media(self, batch, **kwargs):
        return ['{"field": "unused"}' for _ in batch]


@pytest.mark.skipif(pytest.importorskip("fitz") is None, reason="PyMuPDF not installed")
def test_rasterized_pdf_chunks_use_page_local_text_and_preserve_caller_context():
    import fitz

    pdf = fitz.open()
    page_one = pdf.new_page()
    page_one.insert_text((72, 72), "Page one body")
    page_two = pdf.new_page()
    page_two.insert_text((72, 72), "Page two body")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    doc = Document.from_pdf(
        pdf_bytes,
        text="Extract diagnosis fields",
        additional_context="Keep payer metadata when present.",
    )
    chunks = chunk_attachments(
        [Attachment.pdf(pdf_bytes)],
        text=doc.text,
        media_options=MediaOptions(pdf_mode="raster"),
        provider_supports_native_pdf=False,
    )

    assert [chunk.text for chunk in chunks] == ["Page one body", "Page two body"]

    ctx = _build_extraction_context(
        doc,
        _PageModel,
        model=_MediaProvider(),
        prompt=None,
        options=ExtractOptions(media=MediaOptions(pdf_mode="raster")),
        provider_kwargs=None,
    )
    requests = _build_media_inference_requests(ctx, doc, chunks)

    assert "Page one body" in requests[0].prompt
    assert "Page two body" in requests[1].prompt
    assert "Extract diagnosis fields" in requests[0].prompt
    assert "Keep payer metadata when present." in requests[0].prompt
