from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from parsantic.extract import ExtractOptions, extract, extract_iter


class _Record(BaseModel):
    name: str


@dataclass
class _CountingProvider:
    outputs: list[str]
    calls: int = 0

    def infer(self, batch_prompts: Sequence[str], **kwargs):
        self.calls += 1
        return [self.outputs[min(self.calls - 1, len(self.outputs) - 1)] for _ in batch_prompts]


def test_validation_gated_passes_stop_after_first_success_for_extract():
    provider = _CountingProvider(outputs=['{"name": "Ada"}'])

    result = extract(
        "Ada",
        _Record,
        model=provider,
        options=ExtractOptions(passes=3),
    )

    assert result.value.name == "Ada"
    assert provider.calls == 1


def test_validation_gated_passes_stop_after_first_success_for_extract_iter():
    provider = _CountingProvider(outputs=['{"name": "Ada"}'])

    results = list(
        extract_iter(
            "Ada",
            _Record,
            model=provider,
            options=ExtractOptions(passes=3),
        )
    )

    assert results[0].value.name == "Ada"
    assert provider.calls == 1


def test_targeted_repair_reprompts_with_validation_errors():
    provider = _CountingProvider(
        outputs=[
            '{"name": 42}',
            '{"name": "Ada"}',
        ]
    )

    result = extract(
        "Ada",
        _Record,
        model=provider,
        options=ExtractOptions(repair="targeted", max_repair_attempts=1),
    )

    assert result.value.name == "Ada"
    assert provider.calls == 2
