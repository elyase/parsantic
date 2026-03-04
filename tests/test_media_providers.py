from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from parsantic.extract.media.attachments import Attachment, AttachmentKind
from parsantic.extract.providers.base import (
    InferenceRequest,
    SupportsAsyncMediaInfer,
    SupportsMediaInfer,
)
from parsantic.extract.providers.pydantic_ai_provider import PydanticAIProvider


class _SupportsMediaImpl:
    def infer_media(self, batch: list[InferenceRequest], **kwargs: Any) -> list[str]:
        return [request.prompt for request in batch]


class _SupportsAsyncMediaImpl:
    async def ainfer_media(self, batch: list[InferenceRequest], **kwargs: Any) -> list[str]:
        return [request.prompt for request in batch]


def _provider_without_init() -> PydanticAIProvider:
    provider = object.__new__(PydanticAIProvider)
    provider.model_id = "openai:gpt-4o-mini"
    provider.api_key = None
    provider.base_url = None
    provider.max_concurrency = 8
    provider._agent = object()
    return provider


def test_inference_request_defaults():
    request = InferenceRequest(prompt="extract this")
    assert request.prompt == "extract this"
    assert request.attachments == ()
    assert request.document_id is None
    assert request.document_index is None
    assert request.attachment_index is None
    assert request.page_index is None
    assert request.meta == {}


def test_inference_request_with_attachments_and_metadata():
    attachment = Attachment(kind=AttachmentKind.IMAGE, source=b"image-bytes")
    request = InferenceRequest(
        prompt="extract",
        attachments=(attachment,),
        document_id="doc-1",
        document_index=0,
        attachment_index=0,
        page_index=1,
        meta={"confidence": 0.9, "has_image": True},
    )
    assert request.attachments == (attachment,)
    assert request.document_id == "doc-1"
    assert request.document_index == 0
    assert request.attachment_index == 0
    assert request.page_index == 1
    assert request.meta == {"confidence": 0.9, "has_image": True}


def test_inference_request_is_frozen():
    request = InferenceRequest(prompt="extract")
    with pytest.raises(FrozenInstanceError):
        request.prompt = "mutated"  # type: ignore[misc]


def test_supports_media_infer_runtime_checkable():
    assert isinstance(_SupportsMediaImpl(), SupportsMediaInfer)


def test_supports_async_media_infer_runtime_checkable():
    assert isinstance(_SupportsAsyncMediaImpl(), SupportsAsyncMediaInfer)


def test_pydantic_ai_provider_exposes_supported_attachment_kinds():
    assert PydanticAIProvider.supported_attachment_kinds == frozenset({"image", "pdf"})


def test_build_message_parts_image_bytes():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import BinaryContent

    provider = _provider_without_init()
    request = InferenceRequest(
        prompt="describe this image",
        attachments=(Attachment(kind=AttachmentKind.IMAGE, source=b"\x89PNG"),),
    )
    parts = provider._build_message_parts(request)

    assert len(parts) == 2
    assert isinstance(parts[0], BinaryContent)
    assert parts[0].data == b"\x89PNG"
    assert parts[0].media_type == "image/png"
    assert parts[1] == "describe this image"


def test_build_message_parts_pdf_bytes():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import BinaryContent

    provider = _provider_without_init()
    request = InferenceRequest(
        prompt="summarize this pdf",
        attachments=(Attachment(kind=AttachmentKind.PDF, source=b"%PDF-1.7"),),
    )
    parts = provider._build_message_parts(request)

    assert len(parts) == 2
    assert isinstance(parts[0], BinaryContent)
    assert parts[0].data == b"%PDF-1.7"
    assert parts[0].media_type == "application/pdf"
    assert parts[1] == "summarize this pdf"


def test_build_message_parts_custom_mime_type():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import BinaryContent

    provider = _provider_without_init()
    request = InferenceRequest(
        prompt="describe jpeg",
        attachments=(
            Attachment(
                kind=AttachmentKind.IMAGE,
                source=b"\xff\xd8\xff\xe0",
                mime_type="image/jpeg",
            ),
        ),
    )
    parts = provider._build_message_parts(request)

    assert len(parts) == 2
    assert isinstance(parts[0], BinaryContent)
    assert parts[0].media_type == "image/jpeg"
    assert parts[1] == "describe jpeg"


def test_build_message_parts_without_attachments():
    pytest.importorskip("pydantic_ai")
    provider = _provider_without_init()
    parts = provider._build_message_parts(InferenceRequest(prompt="text only"))
    assert parts == ["text only"]


def test_build_message_parts_reads_path_bytes(tmp_path: Path):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import BinaryContent

    provider = _provider_without_init()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-path")

    request = InferenceRequest(
        prompt="from path",
        attachments=(Attachment(kind=AttachmentKind.PDF, source=pdf_path),),
    )
    parts = provider._build_message_parts(request)

    assert len(parts) == 2
    assert isinstance(parts[0], BinaryContent)
    assert parts[0].data == b"%PDF-path"
    assert parts[0].media_type == "application/pdf"
    assert parts[1] == "from path"
