from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel

from parsantic.extract.batch import BatchResult, BatchStatus, aextract_batch, extract_batch
from parsantic.extract.providers.base import InferenceRequest
from parsantic.extract.types import Document


class Person(BaseModel):
    name: str


@dataclass(slots=True)
class _FakeBatchProvider:
    outputs: Sequence[str]
    statuses: list[BatchStatus]
    model_id: str | None = "fake-batch"
    submitted_requests: list[InferenceRequest] = field(default_factory=list)
    poll_calls: int = 0
    retrieve_calls: int = 0

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        if not self.outputs:
            return ["" for _ in batch_prompts]
        if len(self.outputs) == 1:
            return [self.outputs[0] for _ in batch_prompts]
        return list(self.outputs[: len(batch_prompts)])

    def submit_batch(self, requests: Sequence[InferenceRequest], **kwargs: Any) -> str:
        self.submitted_requests = list(requests)
        return "batch-123"

    def poll_batch(self, batch_id: str) -> BatchStatus:
        self.poll_calls += 1
        idx = min(self.poll_calls - 1, len(self.statuses) - 1)
        return self.statuses[idx]

    def retrieve_batch(self, batch_id: str) -> Sequence[str]:
        self.retrieve_calls += 1
        return list(self.outputs)


@dataclass(slots=True)
class _NeverCompleteBatchProvider:
    model_id: str | None = "never-complete"

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        return ["" for _ in batch_prompts]

    def submit_batch(self, requests: Sequence[InferenceRequest], **kwargs: Any) -> str:
        return "batch-timeout"

    def poll_batch(self, batch_id: str) -> BatchStatus:
        return BatchStatus(
            batch_id=batch_id,
            state="in_progress",
            completed_count=0,
            total_count=1,
        )

    def retrieve_batch(self, batch_id: str) -> Sequence[str]:
        raise AssertionError("retrieve_batch should not be called on timeout")


@dataclass(slots=True)
class _FakeProvider:
    output: str
    model_id: str | None = "fake"
    infer_calls: int = 0

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        self.infer_calls += 1
        return [self.output for _ in batch_prompts]


def test_extract_batch_uses_batch_api_when_supported():
    provider = _FakeBatchProvider(
        outputs=['{"name": "Alice"}', '{"name": "Bob"}'],
        statuses=[
            BatchStatus(batch_id="batch-123", state="pending", completed_count=0, total_count=2),
            BatchStatus(batch_id="batch-123", state="completed", completed_count=2, total_count=2),
        ],
    )
    docs = [
        Document(text="Alice", document_id="doc-1"),
        Document(text="Bob", document_id="doc-2"),
    ]

    result = extract_batch(docs, Person, model=provider, poll_interval=0.001, timeout=1.0)

    assert result.used_batch_api is True
    assert result.batch_id == "batch-123"
    assert result.total_documents == 2
    assert [item.document_id for item in result.results] == ["doc-1", "doc-2"]
    assert [item.value.name for item in result.results] == ["Alice", "Bob"]
    assert len(provider.submitted_requests) == 2
    assert all(isinstance(req, InferenceRequest) for req in provider.submitted_requests)
    assert provider.poll_calls >= 1
    assert provider.retrieve_calls == 1


def test_extract_batch_falls_back_to_sequential_extract():
    provider = _FakeProvider(output='{"name": "Fallback"}')
    docs = [
        Document(text="one", document_id="d1"),
        Document(text="two", document_id="d2"),
    ]

    result = extract_batch(docs, Person, model=provider)

    assert result.used_batch_api is False
    assert result.batch_id is None
    assert result.total_documents == 2
    assert [item.value.name for item in result.results] == ["Fallback", "Fallback"]
    assert provider.infer_calls >= len(docs)


def test_extract_batch_timeout_raises_timeout_error():
    provider = _NeverCompleteBatchProvider()
    docs = [Document(text="Alice", document_id="doc-timeout")]

    with pytest.raises(TimeoutError):
        extract_batch(docs, Person, model=provider, poll_interval=0.001, timeout=0.01)


def test_batch_dataclasses_can_be_created():
    status = BatchStatus(
        batch_id="batch-42",
        state="in_progress",
        completed_count=3,
        total_count=10,
        error=None,
    )
    batch_result: BatchResult[Person] = BatchResult(
        results=[],
        batch_id="batch-42",
        used_batch_api=True,
        total_documents=10,
    )

    assert status.batch_id == "batch-42"
    assert status.state == "in_progress"
    assert status.completed_count == 3
    assert batch_result.batch_id == "batch-42"
    assert batch_result.used_batch_api is True


def test_extract_batch_handles_empty_document_list():
    provider = _FakeBatchProvider(outputs=[], statuses=[])

    result = extract_batch([], Person, model=provider)

    assert result.results == []
    assert result.batch_id is None
    assert result.used_batch_api is False
    assert result.total_documents == 0


def test_aextract_batch_uses_batch_api_when_supported():
    provider = _FakeBatchProvider(
        outputs=['{"name": "Async Alice"}'],
        statuses=[
            BatchStatus(batch_id="batch-123", state="completed", completed_count=1, total_count=1)
        ],
    )

    async def _run() -> BatchResult[Person]:
        return await aextract_batch(
            [Document(text="Async Alice", document_id="doc-async")],
            Person,
            model=provider,
            poll_interval=0.001,
            timeout=1.0,
        )

    result = asyncio.run(_run())

    assert result.used_batch_api is True
    assert result.batch_id == "batch-123"
    assert result.total_documents == 1
    assert len(result.results) == 1
    assert result.results[0].value.name == "Async Alice"
