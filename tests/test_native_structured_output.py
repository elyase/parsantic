"""Tests for native structured output support in PydanticAIProvider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

pydantic_ai = pytest.importorskip("pydantic_ai")

from parsantic.extract.options import ExtractOptions  # noqa: E402
from parsantic.extract.providers.pydantic_ai_provider import (  # noqa: E402
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


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("openai:gpt-4o", ("openai", "gpt-4o")),
        ("gpt-4o", ("openai", "gpt-4o")),
        ("claude-3-opus", ("anthropic", "claude-3-opus")),
        ("gemini-2.0-flash", ("gemini", "gemini-2.0-flash")),
        ("custom-model", ("openai", "custom-model")),
    ],
    ids=["explicit", "bare-gpt", "bare-claude", "bare-gemini", "unknown-bare"],
)
def test_parse_model_spec(spec, expected):
    assert _parse_model_spec(spec) == expected


# ── _result_to_json_string ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_val,check",
    [
        ('{"a":1}', lambda r: r == '{"a":1}'),
        ({"name": "Alice", "age": 30}, lambda r: json.loads(r) == {"name": "Alice", "age": 30}),
        ([1, 2, 3], lambda r: json.loads(r) == [1, 2, 3]),
        (_SampleModel(name="Bob", age=25), lambda r: json.loads(r) == {"name": "Bob", "age": 25}),
        ({"text": "日本語"}, lambda r: "日本語" in r),
    ],
    ids=["string-passthrough", "dict", "list", "pydantic-model", "unicode"],
)
def test_result_to_json_string(input_val, check):
    assert check(_result_to_json_string(input_val))


def test_result_to_json_string_non_native_values():
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


def test_supports_native_structured_output():
    assert _make_provider(supports_native=False).supports_native_structured_output() is False
    assert _make_provider(supports_native=True).supports_native_structured_output() is True


# ── _resolve_output_type ─────────────────────────────────────────────────


class TestResolveOutputType:
    def test_returns_none_when_appropriate(self):
        provider = _make_provider(supports_native=True)
        assert provider._resolve_output_type(_SampleModel, "prompt") is None
        assert provider._resolve_output_type(None, "native") is None
        assert (
            _make_provider(supports_native=False)._resolve_output_type(_SampleModel, "auto") is None
        )

    def test_returns_native_output_when_appropriate(self):
        from pydantic_ai import NativeOutput

        assert isinstance(
            _make_provider(supports_native=False)._resolve_output_type(_SampleModel, "native"),
            NativeOutput,
        )
        assert isinstance(
            _make_provider(supports_native=True)._resolve_output_type(_SampleModel, "auto"),
            NativeOutput,
        )


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


def test_extract_options_structured_output():
    assert ExtractOptions().structured_output == "auto"
    for val in ("auto", "native", "prompt"):
        assert ExtractOptions(structured_output=val).structured_output == val
