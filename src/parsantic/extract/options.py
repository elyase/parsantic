from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .alignment import AlignmentOptions
from .formatting import FormatOptions
from .prompt import PromptValidationLevel
from .tokenizer import Tokenizer, TokenizerName


@dataclass(slots=True)
class ExtractOptions:
    passes: int = 1
    max_char_buffer: int | None = None
    batch_length: int = 4
    max_workers: int = 1
    overlap_chars: int = 0
    tokenizer: TokenizerName | Tokenizer | None = None
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    format: FormatOptions = field(default_factory=FormatOptions)
    prompt_validation: PromptValidationLevel = PromptValidationLevel.WARNING
    schema_mode: Literal["compact", "pretty"] = "compact"
    repair: Literal["none", "local"] = "none"
    chunk_error: Literal["raise", "skip"] = "skip"
    merge_strategy: Literal["first_wins", "last_wins", "prefer_non_null"] = "first_wins"

    def __post_init__(self) -> None:
        if self.passes < 1:
            raise ValueError("passes must be >= 1")
        if self.batch_length < 1:
            raise ValueError("batch_length must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
