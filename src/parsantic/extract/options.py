from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .alignment import AlignmentOptions
from .formatting import FormatOptions
from .prompt import PromptValidationLevel


@dataclass(slots=True)
class ExtractOptions:
    passes: int = 1
    max_char_buffer: int | None = None
    batch_length: int = 4
    max_workers: int = 1
    overlap_chars: int = 0
    tokenizer: str | None = None
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    format: FormatOptions = field(default_factory=FormatOptions)
    prompt_validation: PromptValidationLevel = PromptValidationLevel.WARNING
    schema_mode: str = "compact"
    repair: Literal["none", "local"] = "none"
