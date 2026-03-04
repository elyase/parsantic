from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AttachmentKind(str, Enum):
    IMAGE = "image"
    PDF = "pdf"


@dataclass(frozen=True, slots=True)
class Attachment:
    kind: AttachmentKind
    source: Path | bytes
    mime_type: str | None = None
    page_indices: tuple[int, ...] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind is not AttachmentKind.PDF and self.page_indices is not None:
            raise ValueError("page_indices is only valid for PDF attachments")
        if self.page_indices is not None and any(i < 0 for i in self.page_indices):
            raise ValueError("page_indices must be >= 0")

    @staticmethod
    def image(
        source: Path | bytes,
        *,
        mime_type: str | None = None,
        name: str | None = None,
    ) -> Attachment:
        return Attachment(kind=AttachmentKind.IMAGE, source=source, mime_type=mime_type, name=name)

    @staticmethod
    def pdf(
        source: Path | bytes,
        *,
        page_indices: Sequence[int] | None = None,
        name: str | None = None,
    ) -> Attachment:
        pages = tuple(page_indices) if page_indices is not None else None
        return Attachment(kind=AttachmentKind.PDF, source=source, page_indices=pages, name=name)
