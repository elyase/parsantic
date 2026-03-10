from __future__ import annotations

import time
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


def test_inference_request_construction():
    # defaults
    req = InferenceRequest(prompt="extract this")
    assert req.prompt == "extract this"
    assert req.attachments == ()
    assert req.document_id is None
    assert req.meta == {}

    # with all fields
    att = Attachment(kind=AttachmentKind.IMAGE, source=b"image-bytes")
    req2 = InferenceRequest(
        prompt="extract",
        attachments=(att,),
        document_id="doc-1",
        document_index=0,
        attachment_index=0,
        page_index=1,
        meta={"confidence": 0.9},
    )
    assert req2.attachments == (att,)
    assert req2.document_id == "doc-1"
    assert req2.page_index == 1


def test_supports_media_protocols():
    assert isinstance(_SupportsMediaImpl(), SupportsMediaInfer)
    assert isinstance(_SupportsAsyncMediaImpl(), SupportsAsyncMediaInfer)


def test_pydantic_ai_provider_exposes_supported_attachment_kinds():
    # Default value before __post_init__ includes both image and pdf
    assert frozenset({"image", "pdf"}).issuperset(
        PydanticAIProvider.__dataclass_fields__["supported_attachment_kinds"].default
    )
    assert "image" in PydanticAIProvider.__dataclass_fields__["supported_attachment_kinds"].default


@pytest.mark.parametrize(
    "prompt_text,attachments,expected_mime,expected_data",
    [
        (
            "describe this image",
            (Attachment(kind=AttachmentKind.IMAGE, source=b"\x89PNG"),),
            "image/png",
            b"\x89PNG",
        ),
        (
            "summarize this pdf",
            (Attachment(kind=AttachmentKind.PDF, source=b"%PDF-1.7"),),
            "application/pdf",
            b"%PDF-1.7",
        ),
        (
            "describe jpeg",
            (
                Attachment(
                    kind=AttachmentKind.IMAGE, source=b"\xff\xd8\xff\xe0", mime_type="image/jpeg"
                ),
            ),
            "image/jpeg",
            b"\xff\xd8\xff\xe0",
        ),
    ],
    ids=["image-bytes", "pdf-bytes", "custom-mime"],
)
def test_build_message_parts_with_attachment(
    prompt_text, attachments, expected_mime, expected_data
):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import BinaryContent

    provider = _provider_without_init()
    request = InferenceRequest(prompt=prompt_text, attachments=attachments)
    parts = provider._build_message_parts(request)
    assert len(parts) == 2
    assert isinstance(parts[0], BinaryContent)
    assert parts[0].data == expected_data
    assert parts[0].media_type == expected_mime
    assert parts[1] == prompt_text


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


def test_sync_infer_runs_batch_prompts_concurrently(monkeypatch):
    provider = _provider_without_init()

    def _slow_run(prompt, **kwargs):
        time.sleep(0.2)
        return prompt

    monkeypatch.setattr(type(provider), "_run_with_native_fallback", staticmethod(_slow_run))

    started = time.perf_counter()
    outputs = provider.infer(["a", "b", "c"], max_concurrency=3)
    elapsed = time.perf_counter() - started

    assert outputs == ["a", "b", "c"]
    assert elapsed < 0.45


def test_sync_infer_media_runs_requests_concurrently(monkeypatch):
    provider = _provider_without_init()

    def _slow_run(prompt, **kwargs):
        time.sleep(0.2)
        return prompt[-1]

    monkeypatch.setattr(type(provider), "_run_with_native_fallback", staticmethod(_slow_run))

    requests = [
        InferenceRequest(prompt="p1"),
        InferenceRequest(prompt="p2"),
        InferenceRequest(prompt="p3"),
    ]

    started = time.perf_counter()
    outputs = provider.infer_media(requests, max_concurrency=3)
    elapsed = time.perf_counter() - started

    assert outputs == ["p1", "p2", "p3"]
    assert elapsed < 0.45
