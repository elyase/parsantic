"""Tests for media-aware pipeline dispatch logic."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from parsantic.extract import Document, extract
from parsantic.extract.media.attachments import Attachment, AttachmentKind
from parsantic.extract.options import (
    ExtractOptions,
    FieldScopePolicy,
    MediaOptions,
    Strategy,
)
from parsantic.extract.pipeline import (
    _build_media_inference_requests,
    _check_media_capability,
    _DocumentState,
    _infer_media_batch,
    _merge_hybrid_states,
)
from parsantic.extract.providers.base import (
    InferenceRequest,
    SupportsAsyncMediaInfer,
    SupportsMediaInfer,
)
from parsantic.extract.types import AlignmentStatus

# -- Fixtures / helpers --------------------------------------------------------

SAMPLE_INVOICE_PDF = Path(__file__).resolve().parents[1] / "examples" / "sample_invoice.pdf"


class Invoice(BaseModel):
    total: float
    vendor: str = ""


class Patient(BaseModel):
    name: str = ""
    dob: str = ""


class LineItem(BaseModel):
    code: str
    amount: int | None = None
    description: str | None = None


class Record(BaseModel):
    patient: Patient
    line_items: list[LineItem]


class AmountOnly(BaseModel):
    amount: float


def _vision_evidence(path: str, *, value_preview: str, page_index: int | None):
    from parsantic.extract.types import FieldEvidence

    return FieldEvidence(
        path=path,
        value_preview=value_preview,
        char_interval=None,
        token_interval=None,
        alignment_status=AlignmentStatus.UNMATCHED,
        source="vision",
        page_index=page_index,
        grounding_method="unmatched",
    )


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
    supported_attachment_kinds = frozenset({"image", "pdf"})
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 42.0, "vendor": "acme"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        return ['{"total": 99.0, "vendor": "vision-corp"}'] * len(batch)


@dataclass(slots=True)
class _ImageOnlyMediaProvider:
    """Provider that accepts images but explicitly rejects native PDFs."""

    model_id: str | None = "test:image-only-media"
    supported_attachment_kinds = frozenset({"image"})
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
    supported_attachment_kinds = frozenset({"image", "pdf"})
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


@dataclass(slots=True)
class _HybridMediaProvider:
    """Provider that returns different outputs for whole-doc vs per-page requests."""

    model_id: str | None = "test:hybrid-media"
    infer_calls: list[list[str]] = field(default_factory=list)
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        self.infer_calls.append(list(batch_prompts))
        return ['{"total": 1.0, "vendor": "text-path"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append('{"total": 555.0, "vendor": "global-vendor"}')
            elif request.page_index == 1:
                outputs.append('{"total": 99.0, "vendor": "page-1-vendor"}')
            else:
                outputs.append('{"vendor": "page-2-vendor"}')
        return outputs


@dataclass(slots=True)
class _SimpleModeHybridMediaProvider:
    """Provider used to verify document/page branch separation in simple mode."""

    model_id: str | None = "test:simple-hybrid-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 1.0, "vendor": "text-path"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append('{"total": 555.0, "vendor": "global-vendor"}')
            elif request.page_index == 1:
                outputs.append('{"total": 99.0}')
            else:
                outputs.append("{}")
        return outputs


@dataclass(slots=True)
class _RootArrayHybridMediaProvider:
    """Provider that returns array outputs for whole-doc vs per-page requests."""

    model_id: str | None = "test:root-array-hybrid-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return [
            '[{"code": "A", "description": "Widget A"}, {"code": "B", "description": "Widget B"}]'
        ] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append(
                    '[{"code": "A", "description": "Widget A"}, '
                    '{"code": "B", "description": "Widget B"}]'
                )
            elif request.page_index == 1:
                outputs.append('[{"code": "A", "amount": 10}]')
            else:
                outputs.append('[{"code": "B", "amount": 20}]')
        return outputs


@dataclass(slots=True)
class _RootArrayBackfillMediaProvider:
    """Provider that forces whole-doc selection and PDF page backfill for a root array."""

    model_id: str | None = "test:root-array-backfill-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['[{"amount": 85.0}]'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append('[{"amount": 85.0}]')
            else:
                outputs.append("[]")
        return outputs


@dataclass(slots=True)
class _ScalarHybridMediaProvider:
    """Provider that forces whole-doc selection for a scalar root."""

    model_id: str | None = "test:scalar-hybrid-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['"SETTLED"'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append('"SETTLED"')
            elif request.page_index == 1:
                outputs.append('"PENDING"')
            elif request.page_index == 2:
                outputs.append('"APPROVED"')
            else:
                outputs.append('"PENDING"')
        return outputs


@dataclass(slots=True)
class _ConcurrentHybridMediaProvider:
    """Provider that exposes whether whole-doc and page hybrid branches overlap."""

    model_id: str | None = "test:concurrent-hybrid-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    infer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)
    started_branches: set[str] = field(default_factory=set)
    ready: threading.Event = field(default_factory=threading.Event)
    overlap_detected: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 1.0, "vendor": "text-path"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.infer_media_calls.append(list(batch))
        branch = "whole" if batch and batch[0].page_index is None else "page"
        with self.lock:
            self.started_branches.add(branch)
            if len(self.started_branches) == 2:
                self.ready.set()

        if self.ready.wait(timeout=0.25):
            self.overlap_detected = True

        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append('{"total": 555.0, "vendor": "global-vendor"}')
            elif request.page_index == 1:
                outputs.append('{"total": 99.0}')
            else:
                outputs.append("{}")
        return outputs


@dataclass(slots=True)
class _AsyncConcurrentHybridMediaProvider:
    """Async provider that exposes whether whole-doc and page branches overlap."""

    model_id: str | None = "test:async-concurrent-hybrid-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    ainfer_media_calls: list[list[InferenceRequest]] = field(default_factory=list)
    started_branches: set[str] = field(default_factory=set)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    overlap_detected: bool = False

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ['{"total": 1.0, "vendor": "text-path"}'] * len(batch_prompts)

    async def ainfer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        self.ainfer_media_calls.append(list(batch))
        branch = "whole" if batch and batch[0].page_index is None else "page"
        self.started_branches.add(branch)
        if len(self.started_branches) == 2:
            self.ready.set()

        try:
            await asyncio.wait_for(self.ready.wait(), timeout=0.25)
            self.overlap_detected = True
        except TimeoutError:
            pass

        outputs: list[str] = []
        for request in batch:
            if request.page_index is None:
                outputs.append('{"total": 555.0, "vendor": "global-vendor"}')
            elif request.page_index == 1:
                outputs.append('{"total": 99.0}')
            else:
                outputs.append("{}")
        return outputs


def _create_hybrid_fixture_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "hybrid_fixture.pdf"
    pdf = fitz.open()
    pages = (
        "Page 1\nCase Status: PENDING\nPatient: Jane Roe\nCode: A\nDescription: Widget A\nAmount: 25",
        "Page 2\nCase Status: APPROVED\nCode: B\nDescription: Widget B\nAmount: 35",
        "Page 3\nCase Status: SETTLED\nFinal Total: 85\nSummary: approved for closure",
    )
    for page_text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), page_text)
    pdf.save(pdf_path)
    pdf.close()
    return pdf_path


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


def test_extract_pdf_document_auto_mode_uses_text_layer_extraction(tmp_path: Path):
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "text_layer.pdf"
    pdf = fitz.open()
    page1 = pdf.new_page()
    page1.insert_text((72, 72), "Invoice Number: INV-123")
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Vendor: Acme Corp\nTotal: 85.0")
    pdf.save(pdf_path)
    pdf.close()

    class _AutoPdfTextProvider:
        def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
            return ['{"total": 85.0, "vendor": "Acme Corp"}'] * len(batch_prompts)

    result = extract(
        Document.from_pdf(pdf_path),
        Invoice,
        model=_AutoPdfTextProvider(),
    )

    assert result.value.total == 85.0
    assert result.value.vendor == "Acme Corp"


def test_extract_pdf_document_with_native_strategy_preset():
    provider = _MediaProvider()
    doc = Document.from_pdf(b"%PDF", page_indices=[0, 1])

    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(strategy="native"),
    )

    assert result.value.total == 99.0
    assert len(provider.infer_media_calls) == 1
    assert len(provider.infer_media_calls[0]) == 1
    req = provider.infer_media_calls[0][0]
    assert req.page_index is None
    assert "Focus on pages: 1, 2." in req.prompt


def test_extract_pdf_document_rejects_native_pdf_when_provider_lacks_support():
    provider = _ImageOnlyMediaProvider()
    doc = Document.from_pdf(b"%PDF")

    with pytest.raises(TypeError, match="does not support native PDF input"):
        extract(
            doc,
            Invoice,
            model=provider,
            options=ExtractOptions(media=MediaOptions(pdf_mode="native")),
        )


def test_extract_pdf_document_hybrid_mode_rejects_native_document_branch_when_provider_lacks_support():
    provider = _ImageOnlyMediaProvider()
    doc = Document.from_pdf(b"%PDF")

    with pytest.raises(TypeError, match="does not support native PDF input"):
        extract(
            doc,
            Invoice,
            model=provider,
            options=ExtractOptions(
                mode="hybrid",
                document_input="native",
                page_input="image",
            ),
        )


def test_extract_pdf_document_with_auditable_strategy_uses_page_map_reduce(monkeypatch):
    provider = _MediaProvider()
    doc = Document.from_pdf(b"%PDF", page_indices=[0, 1])

    from parsantic.extract.media import preprocessing as preprocessing_mod

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices == (0, 1)
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(strategy="auditable"),
    )

    assert result.value.total == 99.0
    assert len(provider.infer_media_calls) == 1
    requests = provider.infer_media_calls[0]
    assert len(requests) == 2
    assert [req.page_index for req in requests] == [1, 2]
    assert all(req.attachments[0].kind == AttachmentKind.IMAGE for req in requests)


def test_extract_pdf_document_with_hybrid_strategy_routes_fields_by_scope(monkeypatch):
    provider = _HybridMediaProvider()
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(
            strategy=Strategy(
                plan="hybrid",
                field_scope=FieldScopePolicy(
                    by_path={
                        "/total": "local",
                        "/vendor": "global",
                    }
                ),
            )
        ),
    )

    assert result.value.total == 99.0
    assert result.value.vendor == "global-vendor"
    assert provider.infer_calls == []
    assert len(provider.infer_media_calls) == 2
    assert any([req.page_index for req in batch] == [1, 2] for batch in provider.infer_media_calls)
    assert any([req.page_index for req in batch] == [None] for batch in provider.infer_media_calls)

    evidence_by_path = {evidence.path: evidence for evidence in result.evidence}
    assert evidence_by_path["/total"].page_index == 1
    assert evidence_by_path["/vendor"].page_index is None


def test_extract_pdf_document_with_hybrid_mode_uses_native_document_branch_and_page_sources(
    monkeypatch,
):
    provider = _SimpleModeHybridMediaProvider()
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
        ),
    )

    assert result.value.total == 99.0
    assert result.value.vendor == "global-vendor"
    assert len(provider.infer_media_calls) == 2
    assert any([req.page_index for req in batch] == [1, 2] for batch in provider.infer_media_calls)
    assert any([req.page_index for req in batch] == [None] for batch in provider.infer_media_calls)
    page_batch = next(
        batch for batch in provider.infer_media_calls if batch[0].page_index is not None
    )
    whole_batch = next(batch for batch in provider.infer_media_calls if batch[0].page_index is None)
    assert all(req.attachments[0].kind == AttachmentKind.IMAGE for req in page_batch)
    assert whole_batch[0].attachments[0].kind == AttachmentKind.PDF
    assert result.sources["/total"].scope == "page"
    assert result.sources["/total"].pages == (1,)
    assert result.sources["/vendor"].scope == "document"
    assert result.sources["/vendor"].pages == ()


def test_extract_pdf_document_hybrid_mode_runs_whole_and_page_branches_concurrently(monkeypatch):
    provider = _ConcurrentHybridMediaProvider()
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
            max_workers=1,
        ),
    )

    assert result.value.total == 99.0
    assert result.value.vendor == "global-vendor"
    assert len(provider.infer_media_calls) == 2
    assert provider.overlap_detected is True


def test_extract_pdf_document_hybrid_mode_avoids_threaded_branch_overlap_for_async_media_providers(
    monkeypatch,
):
    provider = _AsyncMediaProvider()
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

    from parsantic.extract import pipeline as pipeline_mod
    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    def _unexpected_parallel(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("sync hybrid should not use threaded branch overlap")

    monkeypatch.setattr(pipeline_mod, "_run_parallel_pair", _unexpected_parallel)

    result = extract(
        doc,
        Invoice,
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
            max_workers=1,
        ),
    )

    assert result.value.total == 99.0
    assert result.value.vendor == "vision-corp"
    assert provider.ainfer_media_calls == []


def test_extract_pdf_document_hybrid_mode_supports_root_array_targets(monkeypatch):
    provider = _RootArrayHybridMediaProvider()
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        list[LineItem],
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
        ),
    )

    assert result.value == [
        LineItem(code="A", amount=10, description="Widget A"),
        LineItem(code="B", amount=20, description="Widget B"),
    ]
    assert len(provider.infer_media_calls) == 2
    assert result.sources["/0/amount"].scope == "page"
    assert result.sources["/0/amount"].pages == (1,)
    assert result.sources["/1/amount"].scope == "page"
    assert result.sources["/1/amount"].pages == (2,)
    assert result.sources["/0/description"].scope == "page"
    assert result.sources["/0/description"].pages == (1,)
    assert result.sources["/1/description"].scope == "page"
    assert result.sources["/1/description"].pages == (1,)


def test_extract_pdf_document_hybrid_mode_backfills_root_array_page_sources_for_whole_selected_leaf(
    monkeypatch,
):
    provider = _RootArrayBackfillMediaProvider()
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        list[AmountOnly],
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
        ),
    )

    assert result.value == [AmountOnly(amount=85.0)]
    assert len(provider.infer_media_calls) == 2
    assert result.sources["/0/amount"].scope == "page"
    assert result.sources["/0/amount"].pages == (2,)


def test_extract_pdf_document_hybrid_mode_supports_scalar_root_targets(monkeypatch, tmp_path: Path):
    provider = _ScalarHybridMediaProvider()
    doc = Document.from_pdf(_create_hybrid_fixture_pdf(tmp_path))

    from parsantic.extract.media import preprocessing as preprocessing_mod

    monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

    def _fake_rasterize_pdf(
        source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
    ):
        assert page_indices is None
        return [(0, b"page-1"), (1, b"page-2"), (2, b"page-3")]

    monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

    result = extract(
        doc,
        str,
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
        ),
    )

    assert result.value == "SETTLED"
    assert len(provider.infer_media_calls) == 2
    assert result.sources["/"].scope == "page"
    assert result.sources["/"].pages == (3,)
    assert [
        (evidence.path, evidence.value_preview, evidence.page_index) for evidence in result.evidence
    ] == [
        ("/", "SETTLED", 3),
    ]


def test_merge_hybrid_uses_whole_doc_value_for_missing_auto_root():
    page_state = _DocumentState(
        merged_value={"vendor": None},
        doc_evidence=[
            _vision_evidence("/vendor", value_preview="", page_index=1),
        ],
    )
    whole_state = _DocumentState(
        merged_value={"vendor": "ACME"},
        doc_evidence=[
            _vision_evidence("/vendor", value_preview="ACME", page_index=None),
        ],
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
    )

    assert merged.merged_value == {"vendor": "ACME"}
    assert [(ev.path, ev.value_preview, ev.page_index) for ev in merged.doc_evidence] == [
        ("/vendor", "ACME", None),
    ]


def test_merge_hybrid_discards_metadata_from_filtered_out_branches():
    page_state = _DocumentState(
        merged_value={"total": 99.0},
        raw_outputs=['{"total": 99.0}'],
        all_flags={"page_flag"},
        worst_score=1,
    )
    whole_state = _DocumentState(
        merged_value={"total": 555.0},
        raw_outputs=['{"total": 555.0}'],
        all_flags={"whole_flag"},
        worst_score=7,
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(by_path={"/total": "local"}),
    )

    assert merged.merged_value == {"total": 99.0}
    assert merged.raw_outputs == ['{"total": 99.0}']
    assert merged.all_flags == {"page_flag"}
    assert merged.worst_score == 1


def test_merge_hybrid_supports_nested_objects_and_repeated_arrays():
    page_state = _DocumentState(
        merged_value={
            "patient": {"dob": "2000-01-01"},
            "line_items": [
                {"code": "A", "amount": 10},
                {"code": "A", "amount": 10},
            ],
        },
        doc_evidence=[
            _vision_evidence("/patient/dob", value_preview="2000-01-01", page_index=1),
            _vision_evidence("/line_items/0/code", value_preview="A", page_index=1),
            _vision_evidence("/line_items/0/amount", value_preview="10", page_index=1),
            _vision_evidence("/line_items/1/code", value_preview="A", page_index=2),
            _vision_evidence("/line_items/1/amount", value_preview="10", page_index=2),
        ],
    )
    whole_state = _DocumentState(
        merged_value={
            "patient": {"name": "Alice"},
            "line_items": [
                {"code": "A", "description": "Widget"},
                {"code": "A", "description": "Widget"},
            ],
        },
        doc_evidence=[
            _vision_evidence("/patient/name", value_preview="Alice", page_index=None),
            _vision_evidence("/line_items/0/code", value_preview="A", page_index=None),
            _vision_evidence("/line_items/0/description", value_preview="Widget", page_index=None),
            _vision_evidence("/line_items/1/code", value_preview="A", page_index=None),
            _vision_evidence("/line_items/1/description", value_preview="Widget", page_index=None),
        ],
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(
            by_path={
                "/patient": "global",
                "/patient/dob": "local",
                "/line_items/*/amount": "local",
                "/line_items/*/description": "global",
            }
        ),
    )

    assert merged.merged_value == {
        "patient": {"name": "Alice", "dob": "2000-01-01"},
        "line_items": [
            {"code": "A", "amount": 10, "description": "Widget"},
            {"code": "A", "amount": 10, "description": "Widget"},
        ],
    }
    evidence = {(ev.path, ev.value_preview, ev.page_index) for ev in merged.doc_evidence}
    assert ("/patient/name", "Alice", None) in evidence
    assert ("/patient/dob", "2000-01-01", 1) in evidence
    assert ("/line_items/0/amount", "10", 1) in evidence
    assert ("/line_items/0/description", "Widget", None) in evidence
    assert ("/line_items/1/amount", "10", 2) in evidence
    assert ("/line_items/1/description", "Widget", None) in evidence


def test_merge_hybrid_dedupes_unambiguous_richer_repeated_array_items():
    page_state = _DocumentState(
        merged_value={
            "medications": [
                {
                    "medicationCodeableConcept": {"text": "Capecitabine"},
                }
            ]
        },
        doc_evidence=[
            _vision_evidence(
                "/medications/0/medicationCodeableConcept/text",
                value_preview="Capecitabine",
                page_index=4,
            )
        ],
    )
    whole_state = _DocumentState(
        merged_value={
            "medications": [
                {
                    "medicationCodeableConcept": {"text": "Capecitabine"},
                    "dosageInstruction": "1500 mg orally twice daily",
                    "route": "oral",
                },
                {
                    "medicationCodeableConcept": {"text": "  capecitabine  "},
                },
            ]
        },
        doc_evidence=[
            _vision_evidence(
                "/medications/0/medicationCodeableConcept/text",
                value_preview="Capecitabine",
                page_index=None,
            ),
            _vision_evidence(
                "/medications/0/dosageInstruction",
                value_preview="1500 mg orally twice daily",
                page_index=None,
            ),
            _vision_evidence("/medications/0/route", value_preview="oral", page_index=None),
            _vision_evidence(
                "/medications/1/medicationCodeableConcept/text",
                value_preview="  capecitabine  ",
                page_index=None,
            ),
        ],
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
    )

    assert merged.merged_value == {
        "medications": [
            {
                "medicationCodeableConcept": {"text": "Capecitabine"},
                "dosageInstruction": "1500 mg orally twice daily",
                "route": "oral",
            }
        ]
    }
    evidence_paths = {ev.path for ev in merged.doc_evidence}
    assert "/medications/0/medicationCodeableConcept/text" in evidence_paths
    assert "/medications/0/dosageInstruction" in evidence_paths
    assert "/medications/0/route" in evidence_paths
    assert all(not path.startswith("/medications/1") for path in evidence_paths)


def test_merge_hybrid_keeps_ambiguous_repeated_array_duplicates():
    page_state = _DocumentState(
        merged_value={
            "medications": [
                {
                    "medicationCodeableConcept": {"text": "Capecitabine"},
                    "route": "oral",
                },
                {
                    "medicationCodeableConcept": {"text": "Capecitabine"},
                    "route": "intravenous",
                },
            ]
        },
    )
    whole_state = _DocumentState(
        merged_value={
            "medications": [
                {"medicationCodeableConcept": {"text": "capecitabine"}},
                {"medicationCodeableConcept": {"text": "CAPECITABINE"}},
                {"medicationCodeableConcept": {"text": " Capecitabine "}},
            ]
        },
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
    )

    assert merged.merged_value == {
        "medications": [
            {
                "medicationCodeableConcept": {"text": "Capecitabine"},
                "route": "oral",
            },
            {
                "medicationCodeableConcept": {"text": "Capecitabine"},
                "route": "intravenous",
            },
            {
                "medicationCodeableConcept": {"text": " Capecitabine "},
            },
        ]
    }


def test_merge_hybrid_backfills_exact_page_provenance_for_whole_selected_value():
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)
    page_state = _DocumentState(
        merged_value={"total": 60.0},
        doc_evidence=[
            _vision_evidence("/total", value_preview="60.0", page_index=1),
            _vision_evidence("/total", value_preview="0.0", page_index=2),
        ],
    )
    whole_state = _DocumentState(
        merged_value={"total": 85.0},
        doc_evidence=[
            _vision_evidence("/total", value_preview="85.0", page_index=None),
        ],
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
        doc=doc,
    )

    assert merged.merged_value == {"total": 85.0}
    assert [(ev.path, ev.value_preview, ev.page_index) for ev in merged.doc_evidence] == [
        ("/total", "85.0", 2),
    ]


def test_merge_hybrid_backfills_exact_page_provenance_for_root_array_leaf_value():
    doc = Document.from_pdf(SAMPLE_INVOICE_PDF)
    page_state = _DocumentState(
        merged_value=[{"code": "TOTAL", "amount": 60.0}],
        doc_evidence=[
            _vision_evidence("/0/code", value_preview="TOTAL", page_index=1),
            _vision_evidence("/0/amount", value_preview="60.0", page_index=1),
        ],
    )
    whole_state = _DocumentState(
        merged_value=[{"code": "TOTAL", "amount": 85.0}],
        doc_evidence=[
            _vision_evidence("/0/code", value_preview="TOTAL", page_index=None),
            _vision_evidence("/0/amount", value_preview="85.0", page_index=None),
        ],
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(by_path={"/*/amount": "global"}),
        doc=doc,
    )

    assert merged.merged_value == [{"code": "TOTAL", "amount": 85.0}]
    evidence = [(ev.path, ev.value_preview, ev.page_index) for ev in merged.doc_evidence]
    assert ("/0/amount", "85.0", 2) in evidence
    assert ("/0/code", "TOTAL", 1) in evidence


def test_merge_hybrid_supports_scalar_root_values_with_page_backfill(tmp_path: Path):
    doc = Document.from_pdf(_create_hybrid_fixture_pdf(tmp_path))
    page_state = _DocumentState(
        merged_value="PENDING",
        doc_evidence=[
            _vision_evidence("/", value_preview="PENDING", page_index=1),
            _vision_evidence("/", value_preview="APPROVED", page_index=2),
        ],
    )
    whole_state = _DocumentState(
        merged_value="SETTLED",
        doc_evidence=[
            _vision_evidence("/", value_preview="SETTLED", page_index=None),
        ],
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
        doc=doc,
    )

    assert merged.merged_value == "SETTLED"
    assert [(ev.path, ev.value_preview, ev.page_index) for ev in merged.doc_evidence] == [
        ("/", "SETTLED", 3),
    ]


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


def test_aextract_document_mode_native_pdf_rejects_provider_without_pdf_support():
    from parsantic.extract import aextract

    async def _run() -> None:
        provider = _ImageOnlyMediaProvider()
        doc = Document.from_pdf(b"%PDF")
        with pytest.raises(TypeError, match="does not support native PDF input"):
            await aextract(
                doc,
                Invoice,
                model=provider,
                options=ExtractOptions(mode="document", document_input="native"),
            )

    asyncio.run(_run())


def test_aextract_pdf_document_hybrid_mode_runs_whole_and_page_branches_concurrently(
    monkeypatch,
):
    from parsantic.extract import aextract

    async def _run() -> None:
        provider = _AsyncConcurrentHybridMediaProvider()
        doc = Document.from_pdf(SAMPLE_INVOICE_PDF)

        from parsantic.extract.media import preprocessing as preprocessing_mod

        monkeypatch.setattr(preprocessing_mod, "has_text_layer", lambda source: True)

        def _fake_rasterize_pdf(
            source, *, dpi=200, page_indices=None, raster_format="jpeg", jpeg_quality=85
        ):
            assert page_indices is None
            return [(0, b"page-1"), (1, b"page-2")]

        monkeypatch.setattr(preprocessing_mod, "rasterize_pdf", _fake_rasterize_pdf)

        result = await aextract(
            doc,
            Invoice,
            model=provider,
            options=ExtractOptions(
                mode="hybrid",
                document_input="native",
                page_input="image",
                max_workers=1,
            ),
        )

        assert result.value.total == 99.0
        assert result.value.vendor == "global-vendor"
        assert len(provider.ainfer_media_calls) == 2
        assert provider.overlap_detected is True

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

    assert _provider_supports_native_pdf(_MediaProvider()) is True

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
