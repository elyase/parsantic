from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AlignmentStatus(str, Enum):
    MATCH_EXACT = "match_exact"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"
    UNMATCHED = "unmatched"


@dataclass(slots=True)
class Document:
    text: str
    document_id: str | None = None
    additional_context: str | None = None


@dataclass(slots=True)
class FieldEvidence:
    path: str
    value_preview: str
    char_interval: tuple[int, int] | None
    token_interval: tuple[int, int] | None
    alignment_status: AlignmentStatus


@dataclass(slots=True)
class ChunkDebug:
    """Per-chunk debug information collected during extraction."""

    chunk_index: int
    chunk_text_preview: str  # first 100 chars
    raw_output: str
    flags: tuple[str, ...]
    score: int


@dataclass(slots=True)
class ExtractDebug:
    prompt: str
    raw_outputs: list[str]
    chunks: list[ChunkDebug] = field(default_factory=list)
    rendered_prompt_preview: str | None = None  # first 500 chars of rendered prompt


@dataclass(slots=True)
class ExtractResult[T]:
    value: T
    document_id: str | None
    raw_text: str | None
    flags: tuple[str, ...]
    score: int
    evidence: list[FieldEvidence]
    debug: ExtractDebug | None = None
