from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromptValidationLevel(str, Enum):
    OFF = "off"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class Example:
    text: str
    output: Any


@dataclass(slots=True)
class Prompt:
    description: str
    examples: list[Example] = field(default_factory=list)
    include_schema: bool = True
