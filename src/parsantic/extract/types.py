from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .diagnostics import FieldDiagnostic

if TYPE_CHECKING:
    from .media.attachments import Attachment


def _import_httpx() -> Any:
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for URL fetching. Install with: pip install parsantic[web]"
        ) from None


class AlignmentStatus(str, Enum):
    MATCH_EXACT = "match_exact"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"
    UNMATCHED = "unmatched"


class SupportStatus(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class DocumentPageSpan:
    attachment_index: int
    page_index: int
    start: int
    end: int


@dataclass(slots=True)
class Document:
    text: str = ""
    document_id: str | None = None
    additional_context: str | None = None
    attachments: tuple[Attachment, ...] = ()
    page_spans: tuple[DocumentPageSpan, ...] = ()

    @staticmethod
    def _is_url(text: str) -> bool:
        return text.startswith(("http://", "https://"))

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

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        document_id: str | None = None,
        additional_context: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> Document:
        """Fetch text content from a URL and create a Document.

        Requires httpx: pip install parsantic[web]
        """
        httpx = _import_httpx()
        response = httpx.get(url, timeout=timeout, headers=headers or {}, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith(
            ("text/", "application/json", "application/xml")
        ):
            raise ValueError(
                f"URL {url!r} returned content-type {content_type!r}; "
                "use Attachment for binary content like PDFs or images"
            )
        return cls(
            text=response.text,
            document_id=document_id or url,
            additional_context=additional_context,
        )

    @classmethod
    async def afrom_url(
        cls,
        url: str,
        *,
        document_id: str | None = None,
        additional_context: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> Document:
        """Async version of from_url."""
        httpx = _import_httpx()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=timeout,
                headers=headers or {},
                follow_redirects=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if content_type and not content_type.startswith(
                ("text/", "application/json", "application/xml")
            ):
                raise ValueError(
                    f"URL {url!r} returned content-type {content_type!r}; "
                    "use Attachment for binary content like PDFs or images"
                )
            text = response.text
        return cls(
            text=text,
            document_id=document_id or url,
            additional_context=additional_context,
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
    resolution: str | None = None


@dataclass(slots=True)
class FieldStatus:
    path: str
    support: SupportStatus
    confidence: float


@dataclass(slots=True)
class SourceRef:
    scope: Literal["document", "page"]
    pages: tuple[int, ...] = ()


@dataclass(slots=True)
class ExtractResult[T]:
    value: T
    document_id: str | None
    raw_text: str | None
    flags: tuple[str, ...]
    score: int
    evidence: list[FieldEvidence]
    field_statuses: list[FieldStatus] = field(default_factory=list)
    sources: dict[str, SourceRef] = field(default_factory=dict)
    diagnostics: dict[str, FieldDiagnostic] = field(default_factory=dict)
    debug: ExtractDebug | None = None
    conflicts: list[MergeConflict] = field(default_factory=list)


@dataclass(slots=True)
class ExtractStreamEvent[T]:
    value: Any
    document_id: str | None
    raw_text: str | None
    flags: tuple[str, ...]
    score: int
    is_final: bool = False
    result: ExtractResult[T] | None = None
    attachment_index: int | None = None
    page_index: int | None = None
