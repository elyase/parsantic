from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class FieldState(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    EMPTY = "empty"
    ERROR = "error"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(slots=True)
class FieldDiagnostic:
    state: FieldState
    source: Literal["document", "page", "fused", "repair"] = "document"
    confidence: float = 0.0
    validation_errors: list[str] = field(default_factory=list)
