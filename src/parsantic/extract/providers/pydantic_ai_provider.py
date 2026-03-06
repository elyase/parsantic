"""Provider backed by pydantic-ai for multi-model LLM support.

Supports model strings like ``openai:gpt-4o-mini``, ``anthropic:claude-sonnet``,
``gemini:gemini-2.0-flash``, etc.  Requires ``pydantic-ai`` to be installed::

    pip install pydantic-ai
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic_core import to_jsonable_python

try:
    from pydantic_ai import Agent, NativeOutput, capture_run_messages

    _HAS_PYDANTIC_AI = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC_AI = False

from parsantic.config import DEFAULT_MODEL

from .base import InferenceRequest
from .registry import register

logger = logging.getLogger(__name__)

# Match any "provider:model" string for common providers, plus bare model names.
_PATTERNS: tuple[str, ...] = (
    r"^(openai|anthropic|gemini|ollama|vertex|mistral|groq|bedrock|deepseek):",
    r"^gpt-",
    r"^claude-",
    r"^gemini-",
)


def _parse_model_spec(model_spec: str) -> tuple[str, str]:
    """Split ``'provider:model_name'`` into (provider, model_name).

    Bare model names (e.g. ``gpt-4o``) are mapped to their default provider.
    """
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
        return provider, model_name
    if model_spec.startswith("gpt-"):
        return "openai", model_spec
    if model_spec.startswith("claude-"):
        return "anthropic", model_spec
    if model_spec.startswith("gemini-"):
        return "gemini", model_spec
    return "openai", model_spec


def _requires_explicit_api_key(model_spec: str) -> bool:
    provider_name, _ = _parse_model_spec(model_spec)
    return provider_name == "gemini"


def _build_model_with_credentials(
    model_spec: str,
    api_key: str | None = None,
    base_url: str | None = None,
    **extra_kwargs: Any,
) -> Any:
    """Create a pydantic-ai Model object with explicit credentials.

    Falls back to the plain ``model_spec`` string when the provider-specific
    classes are unavailable (letting pydantic-ai resolve env vars).
    """
    provider_name, model_name = _parse_model_spec(model_spec)

    provider_kwargs: dict[str, Any] = {}
    if api_key:
        provider_kwargs["api_key"] = api_key
    if base_url:
        provider_kwargs["base_url"] = base_url

    def _build_provider(provider_cls: type[Any], kwargs: dict[str, Any]) -> Any:
        try:
            return provider_cls(**kwargs)
        except TypeError:
            # Some provider classes may not accept ``base_url``.
            if "base_url" in kwargs:
                reduced = {k: v for k, v in kwargs.items() if k != "base_url"}
                return provider_cls(**reduced)
            raise

    try:
        if provider_name == "openai":
            from pydantic_ai.models.openai import OpenAIModel
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = _build_provider(OpenAIProvider, provider_kwargs)
            return OpenAIModel(model_name, provider=provider)

        if provider_name == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            provider = _build_provider(AnthropicProvider, provider_kwargs)
            return AnthropicModel(model_name, provider=provider)

        if provider_name == "gemini":
            from pydantic_ai.models.google import GoogleModel

            try:
                from pydantic_ai.providers.google import GoogleProvider
            except ImportError:
                from pydantic_ai.providers.google_gla import GoogleGLAProvider as GoogleProvider

            gla_kwargs = {k: v for k, v in provider_kwargs.items() if k == "api_key"}
            provider = _build_provider(GoogleProvider, gla_kwargs)
            return GoogleModel(model_name, provider=provider)

        if provider_name == "vertex":
            from pydantic_ai.models.google import GoogleModel

            try:
                from pydantic_ai.providers.google import GoogleProvider as _GProvider

                vertex_kwargs: dict[str, Any] = {"vertexai": True}
                if extra_kwargs.get("project_id"):
                    vertex_kwargs["project"] = extra_kwargs["project_id"]
                if extra_kwargs.get("region"):
                    vertex_kwargs["location"] = extra_kwargs["region"]
                if extra_kwargs.get("service_account_file"):
                    import google.auth

                    creds, _ = google.auth.load_credentials_from_file(
                        extra_kwargs["service_account_file"]
                    )
                    vertex_kwargs["credentials"] = creds
                provider = _GProvider(**vertex_kwargs)
            except (ImportError, TypeError):
                from pydantic_ai.providers.google_vertex import GoogleVertexProvider

                vertex_kwargs = {}
                if extra_kwargs.get("project_id"):
                    vertex_kwargs["project_id"] = extra_kwargs["project_id"]
                if extra_kwargs.get("region"):
                    vertex_kwargs["region"] = extra_kwargs["region"]
                if extra_kwargs.get("service_account_file"):
                    vertex_kwargs["service_account_file"] = extra_kwargs["service_account_file"]
                provider = GoogleVertexProvider(**vertex_kwargs)

            return GoogleModel(model_name, provider=provider)
    except (ImportError, TypeError) as exc:
        logger.debug(
            "Could not build model object for %r (%s), falling back to string spec",
            model_spec,
            exc,
        )

    # Unknown provider or import failure — let pydantic-ai resolve via env vars.
    return model_spec


def _result_to_json_string(output: Any) -> str:
    """Serialize a pydantic-ai result back to a JSON string."""
    if isinstance(output, str):
        return output
    return json.dumps(to_jsonable_python(output), ensure_ascii=False)


def _extract_raw_json_from_messages(messages: list[Any]) -> str | None:
    """Extract raw JSON string from pydantic-ai message history.

    On validation failure, we can recover the raw LLM output from messages
    so parsantic's repair pipeline can attempt to fix it.

    Prioritizes ToolCallPart (structured JSON) over TextPart to avoid
    returning reasoning text when the actual payload is in a tool call.
    """
    try:
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    except ImportError:
        return None

    for msg in reversed(messages):
        if not isinstance(msg, ModelResponse):
            continue
        # First pass: look for tool calls (these contain structured JSON)
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                return part.args_as_json_str()
        # Second pass: fall back to text content
        for part in msg.parts:
            if isinstance(part, TextPart) and part.content.strip():
                return part.content
    return None


@register(*_PATTERNS, priority=10)
@dataclass(slots=True)
class PydanticAIProvider:
    """Provider that delegates to pydantic-ai for model execution.

    By default uses ``output_type=str`` so parsantic's own pipeline handles
    parsing/coercion/alignment. When ``structured_output='native'`` or
    ``'auto'`` with a capable provider, uses pydantic-ai's ``NativeOutput``
    to leverage provider-native JSON schema constraints (Gemini response_schema,
    OpenAI response_format, etc.), then serializes back to JSON string.
    """

    model_id: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    project_id: str | None = None
    region: str | None = None
    service_account_file: str | None = None
    max_concurrency: int = 8
    supported_attachment_kinds: ClassVar[frozenset[str]] = frozenset({"image", "pdf"})
    _agent: Any = field(default=None, repr=False, init=False)
    _supports_native: bool = field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        if not _HAS_PYDANTIC_AI:
            raise ImportError(
                "pydantic-ai is required for this model. Install with: pip install pydantic-ai"
            )
        model_spec = self.model_id or DEFAULT_MODEL

        has_credentials = (
            self.api_key
            or self.base_url
            or self.project_id
            or self.region
            or self.service_account_file
        )
        if has_credentials:
            model = _build_model_with_credentials(
                model_spec,
                api_key=self.api_key,
                base_url=self.base_url,
                project_id=self.project_id,
                region=self.region,
                service_account_file=self.service_account_file,
            )
            self._agent = Agent(model, output_type=str)
        else:
            if _requires_explicit_api_key(model_spec):
                raise ValueError(
                    "Gemini models require GEMINI_API_KEY or an explicit api_key provider kwarg"
                )
            self._agent = Agent(model_spec, output_type=str)

        # Detect native structured output capability from model profile
        try:
            profile = self._agent.model.profile
            self._supports_native = getattr(profile, "supports_json_schema_output", False)
        except AttributeError:
            self._supports_native = False

    def supports_native_structured_output(self) -> bool:
        """Whether the underlying model supports native JSON schema output."""
        return self._supports_native

    def _resolve_output_type(
        self,
        target_type: type[Any] | None,
        structured_output: Literal["auto", "native", "prompt"],
    ) -> Any | None:
        """Resolve the pydantic-ai output_type override for this call.

        Returns NativeOutput(target_type) when native mode is active,
        or None to use the default str output.
        """
        if not _HAS_PYDANTIC_AI or target_type is None:
            return None

        use_native = structured_output == "native" or (
            structured_output == "auto" and self._supports_native
        )
        if not use_native:
            return None

        try:
            return NativeOutput(target_type)
        except (TypeError, ValueError):
            logger.debug("Failed to build NativeOutput for %s, falling back to str", target_type)
            return None

    def _run_with_native_fallback(
        self,
        user_prompt: Any,
        *,
        target_type: type[Any] | None,
        structured_output: Literal["auto", "native", "prompt"],
        **kwargs: Any,
    ) -> str:
        """Run a single prompt, trying native structured output with fallback.

        If native output fails, attempts to extract raw JSON from the captured
        message history for parsantic's repair pipeline. If no raw JSON is
        recovered, re-raises the exception (the prompt still contains schema
        text, so the pipeline can retry at a higher level if needed).
        """
        output_type_override = self._resolve_output_type(target_type, structured_output)

        if output_type_override is not None:
            with capture_run_messages() as captured:
                try:
                    result = self._agent.run_sync(
                        user_prompt, output_type=output_type_override, **kwargs
                    )
                    return _result_to_json_string(result.output)
                except Exception as exc:
                    raw = _extract_raw_json_from_messages(captured)

                    if raw is not None:
                        logger.debug(
                            "Native structured output validation failed, "
                            "returning raw JSON for repair: %s",
                            exc,
                        )
                        return raw

                    # No raw JSON recovered — re-raise
                    raise

        # prompt mode: plain str output (current behavior)
        result = self._agent.run_sync(user_prompt, **kwargs)
        return result.output

    async def _arun_with_native_fallback(
        self,
        user_prompt: Any,
        *,
        target_type: type[Any] | None,
        structured_output: Literal["auto", "native", "prompt"],
        **kwargs: Any,
    ) -> str:
        """Async version of _run_with_native_fallback."""
        output_type_override = self._resolve_output_type(target_type, structured_output)

        if output_type_override is not None:
            with capture_run_messages() as captured:
                try:
                    result = await self._agent.run(
                        user_prompt, output_type=output_type_override, **kwargs
                    )
                    return _result_to_json_string(result.output)
                except Exception as exc:
                    raw = _extract_raw_json_from_messages(captured)

                    if raw is not None:
                        logger.debug(
                            "Native structured output validation failed, "
                            "returning raw JSON for repair: %s",
                            exc,
                        )
                        return raw

                    raise

        result = await self._agent.run(user_prompt, **kwargs)
        return result.output

    def _iter_stream_with_native_fallback(
        self,
        user_prompt: Any,
        *,
        target_type: type[Any] | None,
        structured_output: Literal["auto", "native", "prompt"],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream cumulative output snapshots for a single prompt."""
        debounce_by = kwargs.pop("debounce_by", 0.1)
        output_type_override = self._resolve_output_type(target_type, structured_output)
        last_snapshot: str | None = None

        if output_type_override is not None:
            result = self._agent.run_stream_sync(
                user_prompt, output_type=output_type_override, **kwargs
            )
            for output in result.stream_output(debounce_by=debounce_by):
                snapshot = _result_to_json_string(output)
                if snapshot != last_snapshot:
                    last_snapshot = snapshot
                    yield snapshot
            final_snapshot = _result_to_json_string(result.get_output())
            if final_snapshot != last_snapshot:
                yield final_snapshot
            return

        result = self._agent.run_stream_sync(user_prompt, **kwargs)
        for text in result.stream_text(delta=False, debounce_by=debounce_by):
            if text != last_snapshot:
                last_snapshot = text
                yield text
        final_snapshot = _result_to_json_string(result.get_output())
        if final_snapshot != last_snapshot:
            yield final_snapshot

    async def _aiter_stream_with_native_fallback(
        self,
        user_prompt: Any,
        *,
        target_type: type[Any] | None,
        structured_output: Literal["auto", "native", "prompt"],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async version of _iter_stream_with_native_fallback."""
        debounce_by = kwargs.pop("debounce_by", 0.1)
        output_type_override = self._resolve_output_type(target_type, structured_output)
        last_snapshot: str | None = None

        if output_type_override is not None:
            async with self._agent.run_stream(
                user_prompt, output_type=output_type_override, **kwargs
            ) as result:
                async for output in result.stream_output(debounce_by=debounce_by):
                    snapshot = _result_to_json_string(output)
                    if snapshot != last_snapshot:
                        last_snapshot = snapshot
                        yield snapshot
                final_snapshot = _result_to_json_string(await result.get_output())
                if final_snapshot != last_snapshot:
                    yield final_snapshot
            return

        async with self._agent.run_stream(user_prompt, **kwargs) as result:
            async for text in result.stream_text(delta=False, debounce_by=debounce_by):
                if text != last_snapshot:
                    last_snapshot = text
                    yield text
            final_snapshot = _result_to_json_string(await result.get_output())
            if final_snapshot != last_snapshot:
                yield final_snapshot

    def infer(
        self,
        batch_prompts: Sequence[str],
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> Sequence[str]:
        results: list[str] = []
        for prompt in batch_prompts:
            output = self._run_with_native_fallback(
                prompt,
                target_type=target_type,
                structured_output=structured_output,
                **kwargs,
            )
            results.append(output)
        return results

    def infer_stream(
        self,
        prompt: str,
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> Iterator[str]:
        yield from self._iter_stream_with_native_fallback(
            prompt,
            target_type=target_type,
            structured_output=structured_output,
            **kwargs,
        )

    def _build_message_parts(self, request: InferenceRequest) -> list[Any]:
        """Convert InferenceRequest to pydantic-ai message parts (text + BinaryContent)."""
        from pydantic_ai.messages import BinaryContent

        from parsantic.extract.media.attachments import AttachmentKind

        parts: list[Any] = []
        for attachment in request.attachments:
            if isinstance(attachment.source, bytes):
                data = attachment.source
            else:
                data = attachment.source.read_bytes()

            media_type = attachment.mime_type
            if media_type is None:
                if attachment.kind is AttachmentKind.PDF:
                    media_type = "application/pdf"
                elif attachment.kind is AttachmentKind.IMAGE:
                    media_type = "image/png"
                else:
                    media_type = "application/octet-stream"

            parts.append(BinaryContent(data=data, media_type=media_type))

        parts.append(request.prompt)
        return parts

    def infer_media(
        self,
        batch: Sequence[InferenceRequest],
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> Sequence[str]:
        results: list[str] = []
        for request in batch:
            parts = self._build_message_parts(request)
            output = self._run_with_native_fallback(
                parts,
                target_type=target_type,
                structured_output=structured_output,
                **kwargs,
            )
            results.append(output)
        return results

    def _make_semaphore(self, kwargs: dict[str, Any]) -> asyncio.Semaphore:
        concurrency = kwargs.pop("max_concurrency", self.max_concurrency)
        return asyncio.Semaphore(max(1, int(concurrency)))

    def infer_media_stream(
        self,
        request: InferenceRequest,
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> Iterator[str]:
        parts = self._build_message_parts(request)
        yield from self._iter_stream_with_native_fallback(
            parts,
            target_type=target_type,
            structured_output=structured_output,
            **kwargs,
        )

    async def ainfer(
        self,
        batch_prompts: Sequence[str],
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> Sequence[str]:
        semaphore = self._make_semaphore(kwargs)

        async def _run_prompt(prompt: str) -> str:
            async with semaphore:
                return await self._arun_with_native_fallback(
                    prompt,
                    target_type=target_type,
                    structured_output=structured_output,
                    **kwargs,
                )

        return list(await asyncio.gather(*(_run_prompt(prompt) for prompt in batch_prompts)))

    async def ainfer_stream(
        self,
        prompt: str,
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for snapshot in self._aiter_stream_with_native_fallback(
            prompt,
            target_type=target_type,
            structured_output=structured_output,
            **kwargs,
        ):
            yield snapshot

    async def ainfer_media(
        self,
        batch: Sequence[InferenceRequest],
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> Sequence[str]:
        semaphore = self._make_semaphore(kwargs)

        async def _run_request(request: InferenceRequest) -> str:
            async with semaphore:
                parts = self._build_message_parts(request)
                return await self._arun_with_native_fallback(
                    parts,
                    target_type=target_type,
                    structured_output=structured_output,
                    **kwargs,
                )

        return list(await asyncio.gather(*(_run_request(request) for request in batch)))

    async def ainfer_media_stream(
        self,
        request: InferenceRequest,
        *,
        target_type: type[Any] | None = None,
        structured_output: Literal["auto", "native", "prompt"] = "prompt",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        parts = self._build_message_parts(request)
        async for snapshot in self._aiter_stream_with_native_fallback(
            parts,
            target_type=target_type,
            structured_output=structured_output,
            **kwargs,
        ):
            yield snapshot
