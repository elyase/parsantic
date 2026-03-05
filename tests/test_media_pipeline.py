"""Tests for media-aware pipeline dispatch logic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from parsantic.extract import Document, extract
from parsantic.extract.media.attachments import Attachment, AttachmentKind
from parsantic.extract.options import ExtractOptions, MediaOptions
from parsantic.extract.pipeline import (
    _build_media_inference_requests,
    _check_media_capability,
    _infer_media_batch,
)
from parsantic.extract.providers.base import (
    InferenceRequest,
    SupportsAsyncMediaInfer,
    SupportsMediaInfer,
)
from parsantic.extract.types import AlignmentStatus

# -- Fixtures / helpers --------------------------------------------------------


class Invoice(BaseModel):
    total: float
    vendor: str = ""


@dataclass(slots=True)
class _TextOnlyProvider:
    """Provider that only supports text inference (no media)."""

    model_id: str | None = "test:text-only"

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 42.0, "vendor": "acme"}'] * len(batch_prompts)


@dataclass(slots=True)
class _MediaProvider:
    """Provider that supports both text and media inference."""

    model_id: str | None = "test:media"
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 42.0, "vendor": "acme"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        return ['{"total": 99.0, "vendor": "vision-corp"}'] * len(batch)


@dataclass(slots=True)
class _AsyncMediaProvider:
    """Provider with both sync and async media support."""

    model_id: str | None = "test:async-media"
    ainfer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 42.0, "vendor": "acme"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 99.0, "vendor": "vision-corp"}'] * len(batch)

    async def ainfer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.ainfer_media_calls.append(list(batch))
        return ['{"total": 99.0, "vendor": "vision-corp"}'] * len(batch)


@dataclass(slots=True)
class _AsyncOnlyMediaProvider:
    """Provider that supports ONLY async media inference (no sync infer_media)."""

    model_id: str | None = "test:async-only-media"
    ainfer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 42.0, "vendor": "acme"}'] * len(batch_prompts)

    async def ainfer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.ainfer_media_calls.append(list(batch))
        return ['{"total": 77.0, "vendor": "async-only-corp"}'] * len(batch)


# -- Tests: capability checking ------------------------------------------------


def test_media_capability_and_protocols():
    _check_media_capability(_MediaProvider())
    with pytest.raises(TypeError, match="does not support media inference"):
        _check_media_capability(_TextOnlyProvider())
    assert isinstance(_MediaProvider(), SupportsMediaInfer)
    assert not isinstance(_TextOnlyProvider(), SupportsMediaInfer)


# -- Tests: extract with text-only documents (existing path unchanged) ---------


def test_extract_text_only_uses_infer_not_infer_media():
    provider = _MediaProvider()
    result = extract(
        Document(text='{"total": 42.0, "vendor": "acme"}'),
        Invoice,
        model=provider,
    )
    assert result.value.total == 42.0
    # infer_media should NOT have been called
    assert len(provider.infer_media_calls) == 0


# -- Tests: extract with media documents (new path) ---------------------------


def test_extract_media_document_uses_infer_media():
    provider = _MediaProvider()
    doc = Document.from_image(b"\x89PNG", text="extract totals from this receipt")
    result = extract(doc, Invoice, model=provider)
    assert result.value.total == 99.0
    assert result.value.vendor == "vision-corp"
    assert len(provider.infer_media_calls) == 1
    # Verify InferenceRequest was constructed correctly
    req = provider.infer_media_calls[0][0]
    assert len(req.attachments) == 1
    assert req.attachments[0].kind == AttachmentKind.IMAGE


def test_extract_media_document_raises_for_text_only_provider():
    provider = _TextOnlyProvider()
    doc = Document.from_image(b"\x89PNG")
    with pytest.raises(TypeError, match="does not support media inference"):
        extract(doc, Invoice, model=provider)


def test_extract_pdf_document_with_page_indices():
    provider = _MediaProvider()
    doc = Document.from_pdf(b"%PDF", page_indices=[0, 1, 2])
    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(media=MediaOptions(pdf_mode="native")),
    )
    assert result.value.total == 99.0
    # Native mode should create a single request with the original PDF.
    assert len(provider.infer_media_calls) == 1
    assert len(provider.infer_media_calls[0]) == 1
    req = provider.infer_media_calls[0][0]
    assert req.page_index is None
    assert "Focus on pages: 1, 2, 3." in req.prompt


# -- Tests: async extract with media ------------------------------------------


def test_aextract_media_document_uses_ainfer_media():
    import asyncio

    from parsantic.extract import aextract

    async def _run() -> None:
        provider = _AsyncMediaProvider()
        doc = Document.from_image(b"\x89PNG", text="async receipt")
        result = await aextract(doc, Invoice, model=provider)
        assert result.value.total == 99.0
        assert len(provider.ainfer_media_calls) == 1

    asyncio.run(_run())


# -- Tests: mixed documents (text + media in same batch) -----------------------


def test_extract_mixed_documents_dispatches_correctly():
    provider = _MediaProvider()
    text_doc = Document(text='{"total": 1.0, "vendor": "text-co"}')
    media_doc = Document.from_image(b"\x89PNG", text="receipt image")

    results = extract([text_doc, media_doc], Invoice, model=provider)
    assert len(results) == 2
    # First doc should use text path (provider.infer returns 42.0)
    assert results[0].value.total == 42.0
    # Second doc should use media path (provider.infer_media returns 99.0)
    assert results[1].value.total == 99.0
    assert len(provider.infer_media_calls) == 1


# -- Tests: _infer_media_batch ------------------------------------------------


def test_infer_media_batch_batches_correctly():
    provider = _MediaProvider()
    requests = [InferenceRequest(prompt=f"request-{i}") for i in range(5)]
    outputs = _infer_media_batch(provider, requests, batch_length=2)
    assert len(outputs) == 5
    # Should have been 3 batches: [2, 2, 1]
    assert len(provider.infer_media_calls) == 3


# -- Tests: _build_media_inference_requests ------------------------------------


def test_build_media_inference_requests_page_index_is_1_based():
    from parsantic.extract.media.chunking import MediaChunk

    doc = Document.from_pdf(b"%PDF", page_indices=[0, 3])
    chunks = [
        MediaChunk(
            attachment=doc.attachments[0],
            attachment_index=0,
            page_index=0,
            text="page 0",
        ),
        MediaChunk(
            attachment=doc.attachments[0],
            attachment_index=0,
            page_index=3,
            text="page 3",
        ),
    ]

    from parsantic.extract.pipeline import _build_extraction_context

    ctx = _build_extraction_context(
        doc, Invoice, model=_MediaProvider(), prompt=None, options=None, provider_kwargs=None
    )
    requests = _build_media_inference_requests(ctx, doc, chunks)
    assert len(requests) == 2
    assert requests[0].page_index == 1  # 0-based -> 1-based
    assert requests[1].page_index == 4  # 3-based -> 4-based (1-indexed)
    assert requests[0].attachment_index == 0


# -- Tests: async-only provider accepted by capability check ------------------


def test_async_only_provider():
    import asyncio

    from parsantic.extract import aextract

    _check_media_capability(_AsyncOnlyMediaProvider(), is_async=True)
    with pytest.raises(TypeError, match="only supports async media inference"):
        _check_media_capability(_AsyncOnlyMediaProvider(), is_async=False)
    assert isinstance(_AsyncOnlyMediaProvider(), SupportsAsyncMediaInfer)
    assert not isinstance(_AsyncOnlyMediaProvider(), SupportsMediaInfer)

    async def _run() -> None:
        provider = _AsyncOnlyMediaProvider()
        result = await aextract(
            Document.from_image(b"\x89PNG", text="async-only receipt"), Invoice, model=provider
        )
        assert result.value.total == 77.0 and len(provider.ainfer_media_calls) == 1

    asyncio.run(_run())


# -- Tests: vision evidence sourcing ------------------------------------------


def test_evidence_source_tagging():
    provider = _MediaProvider()
    # vision evidence
    result = extract(Document.from_image(b"\x89PNG", text="receipt image"), Invoice, model=provider)
    assert len(result.evidence) > 0
    for ev in result.evidence:
        assert (
            ev.source == "vision"
            and ev.char_interval is None
            and ev.alignment_status == AlignmentStatus.UNMATCHED
        )
    # text evidence
    for ev in extract(
        Document(text='{"total": 42.0, "vendor": "acme"}'), Invoice, model=provider
    ).evidence:
        assert ev.source == "text"


# -- Tests: MergeConflict population -----------------------------------------


def test_merge_conflict_populated_on_scalar_conflict():
    """When two media chunks return conflicting scalar values, MergeConflict is recorded."""

    @dataclass(slots=True)
    class _ConflictingMediaProvider:
        model_id: str | None = "test:conflict"
        _call_count: int = 0

        def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
            return ['{"total": 42.0, "vendor": "acme"}'] * len(batch_prompts)

        def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
            results = []
            for _ in batch:
                self._call_count += 1
                if self._call_count == 1:
                    results.append('{"total": 100.0, "vendor": "page1-corp"}')
                else:
                    results.append('{"total": 200.0, "vendor": "page2-corp"}')
            return results

    provider = _ConflictingMediaProvider()
    # Use two images to get 2 chunks for map_reduce -> merge conflicts
    doc = Document(
        attachments=(
            Attachment.image(b"\x89PNG-page-1"),
            Attachment.image(b"\x89PNG-page-2"),
        )
    )
    opts = ExtractOptions(media=MediaOptions(page_strategy="map_reduce"))
    result = extract(doc, Invoice, model=provider, options=opts)
    # Should have conflicts for both total and vendor
    assert len(result.conflicts) >= 1
    conflict_paths = {c.path for c in result.conflicts}
    assert any("/total" in p for p in conflict_paths)


# -- Tests: chunking with MediaOptions (preprocessing wiring) ----------------


def test_chunk_attachments_respects_pdf_mode_native():
    """pdf_mode='native' should NOT rasterize, even without vision deps."""
    from parsantic.extract.media.chunking import chunk_attachments

    att = Attachment.pdf(b"%PDF", page_indices=[0, 1])
    opts = MediaOptions(pdf_mode="native")
    chunks = chunk_attachments((att,), text="test", media_options=opts)
    # Native mode: one chunk with page focus hint.
    assert len(chunks) == 1
    assert chunks[0].attachment.kind == AttachmentKind.PDF
    assert chunks[0].page_index is None
    assert chunks[0].text == "test\n\nFocus on pages: 1, 2."


def test_chunk_attachments_raster_mode_requires_vision_deps(monkeypatch: pytest.MonkeyPatch):
    """pdf_mode='raster' should raise when rasterization import path fails."""
    from parsantic.extract.media import preprocessing
    from parsantic.extract.media.chunking import chunk_attachments

    def _raise_import_error(*args: Any, **kwargs: Any) -> list[tuple[int, bytes]]:
        raise ImportError("missing deps")

    att = Attachment.pdf(b"%PDF", page_indices=[0, 1])
    opts = MediaOptions(pdf_mode="raster")
    monkeypatch.setattr(preprocessing, "rasterize_pdf", _raise_import_error)
    with pytest.raises(
        ImportError,
        match="pdf_mode=\"raster\"|pdf_mode='raster'",
    ):
        chunk_attachments((att,), text="test", media_options=opts)


def test_provider_supports_native_pdf():
    from parsantic.extract.pipeline import _provider_supports_native_pdf

    assert _provider_supports_native_pdf(_MediaProvider()) is False  # no attr

    class _ImageOnly:
        model_id = "test:image-only"
        supported_attachment_kinds = frozenset({"image"})

    assert _provider_supports_native_pdf(_ImageOnly()) is False


def test_extract_with_media_options_passed_through():
    """Verify MediaOptions from ExtractOptions are used in media path."""
    provider = _MediaProvider()
    doc = Document.from_image(b"\x89PNG", text="receipt")
    opts = ExtractOptions(media=MediaOptions(max_image_dim=0))  # 0 = skip normalization
    result = extract(doc, Invoice, model=provider, options=opts)
    assert result.value.total == 99.0
