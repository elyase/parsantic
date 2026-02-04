from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class BaseProvider(Protocol):
    model_id: str | None

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]: ...

    async def ainfer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]: ...

    @classmethod
    def get_schema_class(cls) -> type[Any] | None:
        return None

    def apply_schema(self, schema_instance: Any | None) -> None:
        return None


@dataclass(slots=True)
class ProviderConfig:
    model_id: str | None = None
    provider: str | None = None
    provider_kwargs: dict[str, Any] = field(default_factory=dict)
