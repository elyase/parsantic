"""Tests for lean prompt rendering in native structured output mode."""

from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from parsantic.extract.formatting import FormatHandler
from parsantic.extract.options import ExtractOptions
from parsantic.extract.pipeline import _render_prompt
from parsantic.extract.prompt import Example, Prompt

# ── helpers ──────────────────────────────────────────────────────────────

_DEFAULT_DESCRIPTION = "Extract structured data that matches the provided schema."
_SCHEMA_TEXT = '{"type":"object","properties":{"name":{"type":"string"}}}'
_QUESTION = "John is 25 years old."


class _SampleModel(BaseModel):
    name: str
    age: int


def _make_prompt(**kwargs: object) -> Prompt:
    defaults: dict[str, object] = {"description": _DEFAULT_DESCRIPTION}
    defaults.update(kwargs)
    return Prompt(**defaults)  # type: ignore[arg-type]


# ── _render_prompt with native_mode=True ─────────────────────────────────


class TestRenderPromptNativeMode:
    """_render_prompt with native_mode=True produces lean prompts."""

    def _render_native(self, **kwargs):
        defaults = dict(
            prompt_obj=_make_prompt(),
            schema_text=None,
            examples=[],
            question=_QUESTION,
            format_handler=FormatHandler(),
            additional_context=None,
            native_mode=True,
        )
        defaults.update(kwargs)
        prompt_obj = defaults.pop("prompt_obj")
        return _render_prompt(prompt_obj, **defaults)

    def test_strips_schema_and_format_instructions(self):
        prompt = self._render_native(schema_text=_SCHEMA_TEXT)
        assert "Schema:" not in prompt
        assert _SCHEMA_TEXT not in prompt
        assert "Output a single JSON" not in prompt
        assert "Do not include any surrounding prose" not in prompt

    def test_no_qa_framing_and_uses_separator(self):
        prompt = self._render_native()
        assert "Q: " not in prompt
        assert "\nA:" not in prompt
        assert "\n---\n" in prompt
        assert prompt.endswith(_QUESTION)

    def test_keeps_description(self):
        prompt = self._render_native(prompt_obj=_make_prompt(description="Extract person info."))
        assert prompt.startswith("Extract person info.")

    def test_keeps_additional_context(self):
        prompt = self._render_native(additional_context="Ignore hypothetical people.")
        assert "Ignore hypothetical people." in prompt

    def test_keeps_examples_without_qa(self):
        examples = [Example(text="Bob, age 30", output={"name": "Bob", "age": 30})]
        prompt = self._render_native(examples=examples)
        assert "Example:" in prompt
        assert "\u2192" in prompt
        assert "Q: Bob" not in prompt
        assert "A: " not in prompt


# ── _render_prompt with native_mode=False (regression) ───────────────────


class TestRenderPromptPromptModeUnchanged:
    """_render_prompt with native_mode=False is unchanged (regression)."""

    def _render_prompt_mode(self, **kwargs):
        defaults = dict(
            prompt_obj=_make_prompt(),
            schema_text=_SCHEMA_TEXT,
            examples=[],
            question=_QUESTION,
            format_handler=FormatHandler(),
            additional_context=None,
            native_mode=False,
        )
        defaults.update(kwargs)
        prompt_obj = defaults.pop("prompt_obj")
        return _render_prompt(prompt_obj, **defaults)

    def test_has_format_and_schema(self):
        prompt = self._render_prompt_mode()
        assert "Output a single JSON object" in prompt
        assert "Schema:" in prompt
        assert _SCHEMA_TEXT in prompt

    def test_has_qa_framing(self):
        prompt = self._render_prompt_mode(schema_text=None)
        assert f"Q: {_QUESTION}" in prompt
        assert prompt.rstrip().endswith("A:")

    def test_examples_use_qa(self):
        examples = [Example(text="Bob, age 30", output={"name": "Bob", "age": 30})]
        prompt = self._render_prompt_mode(schema_text=None, examples=examples)
        assert "Q: Bob, age 30" in prompt
        assert "A: " in prompt


# ── _build_extraction_context ────────────────────────────────────────────


class TestBuildExtractionContextNativeSchema:
    """_build_extraction_context sets schema_text=None when use_native=True."""

    def test_schema_text_none_when_native(self):
        from unittest.mock import patch

        from parsantic.extract.pipeline import _build_extraction_context

        mock_provider = MagicMock()
        mock_provider.supports_native_structured_output.return_value = True

        with patch("parsantic.extract.pipeline.create_provider", return_value=mock_provider):
            ctx = _build_extraction_context(
                "some text",
                _SampleModel,
                model="openai:gpt-4o",
                prompt=None,
                options=ExtractOptions(structured_output="native"),
                provider_kwargs=None,
            )
        assert ctx.use_native_schema is True
        assert ctx.schema_text is None

    def test_schema_text_present_when_prompt_mode(self):
        from unittest.mock import patch

        from parsantic.extract.pipeline import _build_extraction_context

        mock_provider = MagicMock()
        mock_provider.supports_native_structured_output.return_value = True

        with patch("parsantic.extract.pipeline.create_provider", return_value=mock_provider):
            ctx = _build_extraction_context(
                "some text",
                _SampleModel,
                model="openai:gpt-4o",
                prompt=None,
                options=ExtractOptions(structured_output="prompt"),
                provider_kwargs=None,
            )
        assert ctx.use_native_schema is False
        assert ctx.schema_text is not None
