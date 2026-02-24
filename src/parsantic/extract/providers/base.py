from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class BaseProvider(Protocol):
    model_id: str | None

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]: ...


@dataclass(slots=True)
class ProviderConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
