from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .registry import register


@register(r"^static$", priority=100)
@dataclass(slots=True)
class StaticProvider:
    """Deterministic provider for local demos/tests.

    If outputs has a single item, it is repeated for every prompt.
    If outputs is shorter than prompts, the last output is repeated.
    """

    outputs: Sequence[str]
    model_id: str | None = "static"

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        if not self.outputs:
            return ["" for _ in batch_prompts]
        if len(self.outputs) == 1:
            return [self.outputs[0] for _ in batch_prompts]
        if len(self.outputs) >= len(batch_prompts):
            return list(self.outputs[: len(batch_prompts)])
        last = self.outputs[-1]
        return list(self.outputs) + [last] * (len(batch_prompts) - len(self.outputs))
