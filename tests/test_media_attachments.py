from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from parsantic.extract import AlignmentStatus, Document
from parsantic.extract.media import Attachment, AttachmentKind
from parsantic.extract.options import MediaOptions
from parsantic.extract.types import ExtractResult, FieldEvidence, MergeConflict


def test_attachment_image_factory_with_path():
    source = Path("invoice.png")
    attachment = Attachment.image(source, mime_type="image/png", name="invoice")

    assert attachment.kind == AttachmentKind.IMAGE
    assert attachment.source == source
    assert attachment.mime_type == "image/png"
    assert attachment.page_indices is None
    assert attachment.name == "invoice"


def test_attachment_image_factory_with_bytes():
    source = b"\x89PNG..."
    attachment = Attachment.image(source)

    assert attachment.kind == AttachmentKind.IMAGE
    assert attachment.source == source
    assert attachment.mime_type is None
    assert attachment.page_indices is None


def test_attachment_pdf_factory_with_and_without_page_indices():
    with_pages = Attachment.pdf(Path("contract.pdf"), page_indices=[0, 2], name="contract")
    without_pages = Attachment.pdf(b"%PDF-1.7")

    assert with_pages.kind == AttachmentKind.PDF
    assert with_pages.page_indices == (0, 2)
    assert with_pages.name == "contract"
    assert without_pages.kind == AttachmentKind.PDF
    assert without_pages.page_indices is None


def test_attachment_pdf_rejects_negative_page_indices():
    with pytest.raises(ValueError, match="page_indices must be >= 0"):
        Attachment.pdf(Path("contract.pdf"), page_indices=[-1])


def test_attachment_rejects_page_indices_on_non_pdf():
    with pytest.raises(ValueError, match="page_indices is only valid for PDF attachments"):
        Attachment(kind=AttachmentKind.IMAGE, source=b"\x89PNG...", page_indices=(0,))


def test_attachment_kind_enum_values():
    assert AttachmentKind.IMAGE.value == "image"
    assert AttachmentKind.PDF.value == "pdf"


def test_attachment_is_frozen():
    attachment = Attachment.image(Path("photo.jpg"))

    with pytest.raises(FrozenInstanceError):
        attachment.name = "other-name"  # type: ignore[misc]


def test_document_from_image_creates_attachment_document():
    doc = Document.from_image(
        Path("receipt.png"),
        text="receipt summary",
        document_id="doc-1",
        additional_context="vendor=acme",
        mime_type="image/png",
        name="receipt",
    )

    assert doc.text == "receipt summary"
    assert doc.document_id == "doc-1"
    assert doc.additional_context == "vendor=acme"
    assert len(doc.attachments) == 1
    assert doc.attachments[0].kind == AttachmentKind.IMAGE
    assert doc.attachments[0].source == Path("receipt.png")
    assert doc.attachments[0].mime_type == "image/png"
    assert doc.attachments[0].name == "receipt"


def test_document_from_pdf_creates_attachment_document_with_page_indices():
    doc = Document.from_pdf(
        b"%PDF-1.7",
        page_indices=[1, 3],
        document_id="doc-2",
        additional_context="priority=high",
        name="statement",
    )

    assert doc.text == ""
    assert doc.document_id == "doc-2"
    assert doc.additional_context == "priority=high"
    assert len(doc.attachments) == 1
    assert doc.attachments[0].kind == AttachmentKind.PDF
    assert doc.attachments[0].source == b"%PDF-1.7"
    assert doc.attachments[0].page_indices == (1, 3)
    assert doc.attachments[0].name == "statement"


def test_document_allows_empty_text_with_attachments():
    attachment = Attachment.image(Path("scan.png"))
    doc = Document(attachments=(attachment,))

    assert doc.text == ""
    assert doc.attachments == (attachment,)


def test_document_backwards_compat_text_only_still_works():
    doc = Document(text="hello")

    assert doc.text == "hello"
    assert doc.attachments == ()


def test_field_evidence_new_fields_default_values():
    evidence = FieldEvidence(
        path="/amount",
        value_preview="100",
        char_interval=(0, 3),
        token_interval=(0, 1),
        alignment_status=AlignmentStatus.MATCH_EXACT,
    )

    assert evidence.source == "text"
    assert evidence.attachment_index is None
    assert evidence.page_index is None
    assert evidence.bbox_norm is None
    assert evidence.grounding_method is None


def test_merge_conflict_creation():
    conflict = MergeConflict(
        path="/total",
        existing_preview="10",
        incoming_preview="12",
        page_index=2,
    )

    assert conflict.path == "/total"
    assert conflict.existing_preview == "10"
    assert conflict.incoming_preview == "12"
    assert conflict.page_index == 2


def test_extract_result_conflicts_defaults_to_empty_list():
    result = ExtractResult(
        value={"ok": True},
        document_id=None,
        raw_text=None,
        flags=(),
        score=0,
        evidence=[],
    )

    assert result.conflicts == []


def test_media_options_grounding_default_is_auto():
    options = MediaOptions()

    assert options.grounding == "auto"
