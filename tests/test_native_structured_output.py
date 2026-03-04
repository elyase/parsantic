"""Tests for native structured output support in PydanticAIProvider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from parsantic.extract.options import ExtractOptions
from parsantic.extract.providers.pydantic_ai_provider import (
    PydanticAIProvider,
    _extract_raw_json_from_messages,
    _parse_model_spec,
    _result_to_json_string,
)

# ── helpers ──────────────────────────────────────────────────────────────


class _SampleModel(BaseModel):
    name: str
    age: int


def _make_provider(*, supports_native: bool = False) -> PydanticAIProvider:
    """Create a PydanticAIProvider without hitting the network."""
    provider = object.__new__(PydanticAIProvider)
    provider.model_id = "openai:gpt-4o-mini"
    provider.api_key = None
    provider.base_url = None
    provider.project_id = None
    provider.region = None
    provider.service_account_file = None
    provider.max_concurrency = 8
    provider._agent = MagicMock()
    provider._supports_native = supports_native
    return provider


# ── _parse_model_spec ────────────────────────────────────────────────────


class TestParseModelSpec:
    def test_explicit_provider(self):
        assert _parse_model_spec("openai:gpt-4o") == ("openai", "gpt-4o")

    def test_bare_gpt(self):
        assert _parse_model_spec("gpt-4o") == ("openai", "gpt-4o")

    def test_bare_claude(self):
        assert _parse_model_spec("claude-3-opus") == ("anthropic", "claude-3-opus")

    def test_bare_gemini(self):
        assert _parse_model_spec("gemini-2.0-flash") == ("gemini", "gemini-2.0-flash")

    def test_unknown_bare(self):
        assert _parse_model_spec("custom-model") == ("openai", "custom-model")


# ── _result_to_json_string ───────────────────────────────────────────────


class TestResultToJsonString:
    def test_string_passthrough(self):
        assert _result_to_json_string('{"a":1}') == '{"a":1}'

    def test_dict(self):
        result = _result_to_json_string({"name": "Alice", "age": 30})
        parsed = json.loads(result)
        assert parsed == {"name": "Alice", "age": 30}

    def test_list(self):
        result = _result_to_json_string([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_pydantic_model(self):
        model = _SampleModel(name="Bob", age=25)
        result = _result_to_json_string(model)
        parsed = json.loads(result)
        assert parsed == {"name": "Bob", "age": 25}

    def test_unicode_preserved(self):
        result = _result_to_json_string({"text": "日本語"})
        assert "日本語" in result

    def test_dict_with_non_json_native_values(self):
        """to_jsonable_python handles datetime/Decimal inside dicts."""
        from datetime import datetime
        from decimal import Decimal

        result = _result_to_json_string({"ts": datetime(2025, 1, 1), "val": Decimal("3.14")})
        parsed = json.loads(result)
        assert "2025" in parsed["ts"]
        assert float(parsed["val"]) == pytest.approx(3.14)


# ── _extract_raw_json_from_messages ──────────────────────────────────────


class TestExtractRawJsonFromMessages:
    def test_empty_messages(self):
        assert _extract_raw_json_from_messages([]) is None

    def test_extracts_tool_call_args(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        tool_part = ToolCallPart(
            tool_name="final_result",
            args='{"name":"Alice","age":30}',
            tool_call_id="tc_1",
        )
        msg = ModelResponse(parts=[tool_part])
        result = _extract_raw_json_from_messages([msg])
        assert result == '{"name":"Alice","age":30}'

    def test_extracts_text_content(self):
        from pydantic_ai.messages import ModelResponse, TextPart

        text_part = TextPart(content='{"name":"Bob"}')
        msg = ModelResponse(parts=[text_part])
        result = _extract_raw_json_from_messages([msg])
        assert result == '{"name":"Bob"}'

    def test_skips_empty_text(self):
        from pydantic_ai.messages import ModelResponse, TextPart

        text_part = TextPart(content="   ")
        msg = ModelResponse(parts=[text_part])
        assert _extract_raw_json_from_messages([msg]) is None

    def test_tool_call_prioritized_over_text(self):
        """ToolCallPart should be preferred even when TextPart comes first."""
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

        # TextPart first, ToolCallPart second — tool call should win
        parts = [
            TextPart(content="some reasoning"),
            ToolCallPart(
                tool_name="final_result",
                args='{"from":"tool"}',
                tool_call_id="tc_2",
            ),
        ]
        msg = ModelResponse(parts=parts)
        result = _extract_raw_json_from_messages([msg])
        assert result == '{"from":"tool"}'

    def test_scans_from_last_message(self):
        from pydantic_ai.messages import ModelResponse, TextPart

        msg1 = ModelResponse(parts=[TextPart(content='{"old":true}')])
        msg2 = ModelResponse(parts=[TextPart(content='{"new":true}')])
        result = _extract_raw_json_from_messages([msg1, msg2])
        assert result == '{"new":true}'


# ── supports_native_structured_output ────────────────────────────────────


class TestSupportsNativeStructuredOutput:
    def test_returns_false_by_default(self):
        provider = _make_provider(supports_native=False)
        assert provider.supports_native_structured_output() is False

    def test_returns_true_when_set(self):
        provider = _make_provider(supports_native=True)
        assert provider.supports_native_structured_output() is True


# ── _resolve_output_type ─────────────────────────────────────────────────


class TestResolveOutputType:
    def test_returns_none_for_prompt_mode(self):
        provider = _make_provider(supports_native=True)
        result = provider._resolve_output_type(_SampleModel, "prompt")
        assert result is None

    def test_returns_none_when_no_target(self):
        provider = _make_provider(supports_native=True)
        result = provider._resolve_output_type(None, "native")
        assert result is None

    def test_returns_native_output_for_native_mode(self):
        from pydantic_ai import NativeOutput

        provider = _make_provider(supports_native=False)  # doesn't matter for "native"
        result = provider._resolve_output_type(_SampleModel, "native")
        assert isinstance(result, NativeOutput)

    def test_returns_native_output_for_auto_when_supported(self):
        from pydantic_ai import NativeOutput

        provider = _make_provider(supports_native=True)
        result = provider._resolve_output_type(_SampleModel, "auto")
        assert isinstance(result, NativeOutput)

    def test_returns_none_for_auto_when_not_supported(self):
        provider = _make_provider(supports_native=False)
        result = provider._resolve_output_type(_SampleModel, "auto")
        assert result is None


# ── infer with native structured output ──────────────────────────────────


class TestInferWithNativeOutput:
    def test_infer_passes_kwargs_to_run_sync(self):
        provider = _make_provider(supports_native=False)
        mock_result = MagicMock()
        mock_result.output = '{"name":"Alice","age":30}'
        provider._agent.run_sync.return_value = mock_result

        results = provider.infer(
            ["extract name"],
            target_type=_SampleModel,
            structured_output="prompt",
        )
        assert results == ['{"name":"Alice","age":30}']
        provider._agent.run_sync.assert_called_once()

    def test_infer_with_native_mode_uses_output_type_override(self):
        provider = _make_provider(supports_native=False)
        mock_result = MagicMock()
        mock_result.output = _SampleModel(name="Alice", age=30)
        provider._agent.run_sync.return_value = mock_result

        results = provider.infer(
            ["extract name"],
            target_type=_SampleModel,
            structured_output="native",
        )

        # Should serialize the pydantic model back to JSON
        parsed = json.loads(results[0])
        assert parsed == {"name": "Alice", "age": 30}

        # run_sync should have been called with output_type override
        call_kwargs = provider._agent.run_sync.call_args
        assert call_kwargs.kwargs.get("output_type") is not None

    def test_infer_native_failure_reraises_when_no_raw_json(self):
        """When native fails with no message history, should re-raise."""
        provider = _make_provider(supports_native=False)
        provider._agent.run_sync.side_effect = RuntimeError("native failed")

        with pytest.raises(RuntimeError, match="native failed"):
            provider.infer(
                ["extract name"],
                target_type=_SampleModel,
                structured_output="native",
            )

    def test_infer_native_fallback_extracts_raw_json_from_captured_messages(self):
        """When native fails, extract raw JSON from capture_run_messages."""
        from unittest.mock import patch

        from pydantic_ai.messages import ModelResponse, ToolCallPart

        provider = _make_provider(supports_native=False)

        # Mock capture_run_messages to return messages with tool call
        captured_messages = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args='{"name":"Alice","age":"thirty"}',
                        tool_call_id="tc_1",
                    )
                ]
            )
        ]

        # run_sync raises on native call
        provider._agent.run_sync.side_effect = RuntimeError("validation failed")

        with patch(
            "parsantic.extract.providers.pydantic_ai_provider.capture_run_messages"
        ) as mock_capture:
            # Make the context manager yield our captured messages list
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=captured_messages)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_capture.return_value = mock_ctx

            results = provider.infer(
                ["extract name"],
                target_type=_SampleModel,
                structured_output="native",
            )
            assert results == ['{"name":"Alice","age":"thirty"}']


# ── ExtractOptions structured_output field ───────────────────────────────


class TestExtractOptionsStructuredOutput:
    def test_default_is_auto(self):
        opts = ExtractOptions()
        assert opts.structured_output == "auto"

    def test_accepts_valid_values(self):
        for val in ("auto", "native", "prompt"):
            opts = ExtractOptions(structured_output=val)
            assert opts.structured_output == val
