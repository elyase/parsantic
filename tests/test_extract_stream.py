from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from parsantic.extract import (
    Document,
    ExtractOptions,
    MediaOptions,
    aextract_stream,
    extract_stream,
)
from parsantic.extract.providers.base import InferenceRequest


class Invoice(BaseModel):
    total: float
    vendor: str = ""


@dataclass(slots=True)
class _StreamingMediaProvider:
    model_id: str | None = "test:stream-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    snapshots: list[str] = field(
        default_factory=lambda: ['{"vendor":"vis"', '{"vendor":"vision-corp","total":99.0}']
    )
    infer_media_stream_calls: list[InferenceRequest] = field(default_factory=list)

    def supports_native_structured_output(self) -> bool:
        return True

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        del kwargs
        return ['{"total": 99.0, "vendor": "vision-corp"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        del kwargs
        return ['{"total": 99.0, "vendor": "vision-corp"}'] * len(batch)

    def infer_media_stream(self, request: InferenceRequest, **kwargs: Any) -> Iterator[str]:
        del kwargs
        self.infer_media_stream_calls.append(request)
        yield from self.snapshots


@dataclass(slots=True)
class _AsyncStreamingMediaProvider:
    model_id: str | None = "test:async-stream-media"
    supported_attachment_kinds = frozenset({"image", "pdf"})
    snapshots: list[str] = field(
        default_factory=lambda: ['{"vendor":"vis"', '{"vendor":"vision-corp","total":88.0}']
    )
    ainfer_media_stream_calls: list[InferenceRequest] = field(default_factory=list)

    def supports_native_structured_output(self) -> bool:
        return True

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        del kwargs
        return ['{"total": 88.0, "vendor": "vision-corp"}'] * len(batch_prompts)

    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        del kwargs
        return ['{"total": 88.0, "vendor": "vision-corp"}'] * len(batch)

    async def ainfer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]:
        del kwargs
        return ['{"total": 88.0, "vendor": "vision-corp"}'] * len(batch)

    async def ainfer_media_stream(
        self, request: InferenceRequest, **kwargs: Any
    ) -> AsyncIterator[str]:
        del kwargs
        self.ainfer_media_stream_calls.append(request)
        for snapshot in self.snapshots:
            yield snapshot


def test_extract_stream_pdf_native_yields_partial_then_final():
    provider = _StreamingMediaProvider()
    doc = Document.from_pdf(b"%PDF-stream", text="invoice body")

    events = list(
        extract_stream(
            doc,
            Invoice,
            model=provider,
            options=ExtractOptions(media=MediaOptions(pdf_mode="native", page_strategy="single")),
        )
    )

    assert [event.is_final for event in events] == [False, False, True]
    assert events[0].value.vendor == "vis"
    assert events[-1].result is not None
    assert events[-1].result.value.total == 99.0
    assert len(provider.infer_media_stream_calls) == 1


def test_extract_stream_rejects_multi_request_pdf_modes():
    provider = _StreamingMediaProvider()
    doc = Document.from_pdf(b"%PDF-stream", text="invoice body")

    with pytest.raises(NotImplementedError, match="single-request extraction"):
        list(
            extract_stream(
                doc,
                Invoice,
                model=provider,
                options=ExtractOptions(
                    media=MediaOptions(pdf_mode="native", page_strategy="map_reduce")
                ),
            )
        )


def test_extract_stream_requires_single_pass():
    provider = _StreamingMediaProvider()
    doc = Document.from_pdf(b"%PDF-stream", text="invoice body")

    with pytest.raises(NotImplementedError, match="passes=1"):
        list(
            extract_stream(
                doc,
                Invoice,
                model=provider,
                options=ExtractOptions(
                    passes=2,
                    media=MediaOptions(pdf_mode="native", page_strategy="single"),
                ),
            )
        )


def test_aextract_stream_pdf_native_yields_final_result():
    provider = _AsyncStreamingMediaProvider()
    doc = Document.from_pdf(b"%PDF-stream", text="invoice body")

    async def _run():
        return [
            event
            async for event in aextract_stream(
                doc,
                Invoice,
                model=provider,
                options=ExtractOptions(
                    media=MediaOptions(pdf_mode="native", page_strategy="single")
                ),
            )
        ]

    events = asyncio.run(_run())

    assert [event.is_final for event in events] == [False, False, True]
    assert events[-1].result is not None
    assert events[-1].result.value.total == 88.0
    assert len(provider.ainfer_media_stream_calls) == 1
