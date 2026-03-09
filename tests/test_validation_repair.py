from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from parsantic.extract import ExtractOptions, extract, extract_iter
from parsantic.extract.pipeline import (
    _apply_targeted_fragment,
    _build_extraction_context,
    _build_targeted_repair_prompt,
    _DocumentState,
)


class _Record(BaseModel):
    name: str


class _RepairRecord(BaseModel):
    a: int
    b: str


@dataclass
class _CountingProvider:
    outputs: list[str]
    calls: int = 0

    def infer(self, batch_prompts: Sequence[str], **kwargs):
        self.calls += 1
        return [self.outputs[min(self.calls - 1, len(self.outputs) - 1)] for _ in batch_prompts]


@dataclass
class _PromptRecordingProvider:
    outputs: list[str]
    prompts: list[str]
    calls: int = 0

    def infer(self, batch_prompts: Sequence[str], **kwargs):
        self.prompts.extend(batch_prompts)
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


def test_targeted_repair_scopes_prompt_and_keeps_untargeted_fields():
    provider = _PromptRecordingProvider(
        outputs=['{"a": "oops", "b": "keep"}'],
        prompts=[],
    )
    ctx = _build_extraction_context(
        "repair target",
        _RepairRecord,
        model=provider,
        prompt=None,
        options=ExtractOptions(repair="targeted", max_repair_attempts=1),
        provider_kwargs=None,
    )
    state = _DocumentState(merged_value={"a": "oops", "b": "keep"})
    prompt, target_paths = _build_targeted_repair_prompt(
        ctx=ctx,
        doc=ctx.documents[0],
        state=state,
        original_prompt="repair target",
    )
    repaired = _apply_targeted_fragment(state.merged_value, {"a": 7}, target_paths)

    assert repaired == {"a": 7, "b": "keep"}
    assert target_paths == ["/a"]
    assert "Only repair these JSON Pointer paths:" in prompt
    assert "- /a" in prompt
    assert '"b": "keep"' not in prompt
