from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from parsantic.extract.media.attachments import Attachment


@runtime_checkable
class BaseProvider(Protocol):
    model_id: str | None

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]: ...


type JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """A single inference request for a provider.

    Attributes
    ----------
    page_index
        1-based page number within the attachment; ``None`` if not applicable.
    """

    prompt: str
    attachments: tuple[Attachment, ...] = ()
    document_id: str | None = None
    document_index: int | None = None
    attachment_index: int | None = None
    page_index: int | None = None
    meta: Mapping[str, JsonScalar] = field(default_factory=dict)


@runtime_checkable
class SupportsMediaInfer(Protocol):
    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]: ...


@runtime_checkable
class SupportsAsyncMediaInfer(Protocol):
    async def ainfer_media(
        self, batch: Sequence[InferenceRequest], **kwargs: Any
    ) -> Sequence[str]: ...


@runtime_checkable
class SupportsInferStream(Protocol):
    def infer_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]: ...


@runtime_checkable
class SupportsAsyncInferStream(Protocol):
    async def ainfer_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]: ...


@runtime_checkable
class SupportsMediaInferStream(Protocol):
    def infer_media_stream(self, request: InferenceRequest, **kwargs: Any) -> Iterator[str]: ...


@runtime_checkable
class SupportsAsyncMediaInferStream(Protocol):
    async def ainfer_media_stream(
        self, request: InferenceRequest, **kwargs: Any
    ) -> AsyncIterator[str]: ...


@dataclass(slots=True)
class ProviderConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
