from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .media.attachments import Attachment


class AlignmentStatus(str, Enum):
    MATCH_EXACT = "match_exact"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"
    UNMATCHED = "unmatched"


@dataclass(slots=True)
class Document:
    text: str = ""
    document_id: str | None = None
    additional_context: str | None = None
    attachments: tuple[Attachment, ...] = ()

    @classmethod
    def from_image(
        cls,
        source: Path | bytes,
        *,
        text: str = "",
        document_id: str | None = None,
        additional_context: str | None = None,
        mime_type: str | None = None,
        name: str | None = None,
    ) -> Document:
        from .media.attachments import Attachment

        return cls(
            text=text,
            document_id=document_id,
            additional_context=additional_context,
            attachments=(Attachment.image(source, mime_type=mime_type, name=name),),
        )

    @classmethod
    def from_pdf(
        cls,
        source: Path | bytes,
        *,
        text: str = "",
        document_id: str | None = None,
        additional_context: str | None = None,
        page_indices: Sequence[int] | None = None,
        name: str | None = None,
    ) -> Document:
        from .media.attachments import Attachment

        return cls(
            text=text,
            document_id=document_id,
            additional_context=additional_context,
            attachments=(Attachment.pdf(source, page_indices=page_indices, name=name),),
        )


@dataclass(slots=True)
class FieldEvidence:
    path: str
    value_preview: str
    char_interval: tuple[int, int] | None
    token_interval: tuple[int, int] | None
    alignment_status: AlignmentStatus
    source: Literal["text", "vision"] = "text"
    attachment_index: int | None = None
    page_index: int | None = None
    bbox_norm: tuple[float, float, float, float] | None = None
    grounding_method: Literal["ocr_align", "model_bbox", "unmatched"] | None = None


@dataclass(slots=True)
class ChunkDebug:
    """Per-chunk debug information collected during extraction."""

    chunk_index: int
    chunk_text_preview: str  # first 100 chars
    raw_output: str
    flags: tuple[str, ...]
    score: int
    error: str | None = None


@dataclass(slots=True)
class ExtractDebug:
    prompt: str
    raw_outputs: list[str]
    chunks: list[ChunkDebug] = field(default_factory=list)
    rendered_prompt_preview: str | None = None  # first 500 chars of rendered prompt


@dataclass(slots=True)
class MergeConflict:
    path: str
    existing_preview: str
    incoming_preview: str
    page_index: int | None = None


@dataclass(slots=True)
class ExtractResult[T]:
    value: T
    document_id: str | None
    raw_text: str | None
    flags: tuple[str, ...]
    score: int
    evidence: list[FieldEvidence]
    debug: ExtractDebug | None = None
    conflicts: list[MergeConflict] = field(default_factory=list)
