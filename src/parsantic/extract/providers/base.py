from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from parsantic.extract.media.attachments import Attachment


class BaseProvider(Protocol):
    model_id: str | None

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]: ...


JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    prompt: str
    attachments: tuple[Attachment, ...] = ()
    document_id: str | None = None
    document_index: int | None = None
    attachment_index: int | None = None
    page_index: int | None = None  # 1-based
    meta: Mapping[str, JsonScalar] = field(default_factory=dict)


@runtime_checkable
class SupportsMediaInfer(Protocol):
    def infer_media(self, batch: Sequence[InferenceRequest], **kwargs: Any) -> Sequence[str]: ...


@runtime_checkable
class SupportsAsyncMediaInfer(Protocol):
    async def ainfer_media(
        self, batch: Sequence[InferenceRequest], **kwargs: Any
    ) -> Sequence[str]: ...


@dataclass(slots=True)
class ProviderConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
