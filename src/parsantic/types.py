from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompletionState(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class ScoredValue:
    value: Any
    flags: tuple[str, ...]
    score: int


@dataclass(frozen=True, slots=True)
class CandidateDebug:
    """Debug info for a single coercion candidate."""

    value_preview: Any
    flags: tuple[str, ...]
    score: int
    validation_error: str | None = None


@dataclass(frozen=True, slots=True)
class ParseDebug[T]:
    """Full debug trace for a parse/coerce operation."""

    raw_text: str | None
    candidates: list[CandidateDebug] = field(default_factory=list)
    chosen: CandidateDebug | None = None
    value: T | None = None


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
