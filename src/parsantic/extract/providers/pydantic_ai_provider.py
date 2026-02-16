"""Provider backed by pydantic-ai for multi-model LLM support.

Supports model strings like ``openai:gpt-4o-mini``, ``anthropic:claude-sonnet``,
``gemini:gemini-2.0-flash``, etc.  Requires ``pydantic-ai`` to be installed::

    uv add pydantic-ai
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic_ai import Agent

    _HAS_PYDANTIC_AI = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC_AI = False

from .registry import register

# Match any "provider:model" string for common providers, plus bare model names.
_PATTERNS = (
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


def _build_model_with_credentials(
    model_spec: str,
    api_key: str | None,
    base_url: str | None,
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

    try:
        if provider_name == "openai":
            from pydantic_ai.models.openai import OpenAIModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIModel(model_name, provider=OpenAIProvider(**provider_kwargs))

        if provider_name == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(model_name, provider=AnthropicProvider(**provider_kwargs))

        if provider_name == "gemini":
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google_gla import GoogleGLAProvider

            gla_kwargs = {k: v for k, v in provider_kwargs.items() if k == "api_key"}
            return GoogleModel(model_name, provider=GoogleGLAProvider(**gla_kwargs))
    except (ImportError, TypeError):
        pass

    # Unknown provider or import failure — let pydantic-ai resolve via env vars.
    return model_spec


@register(*_PATTERNS, priority=10)
@dataclass(slots=True)
class PydanticAIProvider:
    """Provider that delegates to pydantic-ai for model execution.

    The provider asks for raw text (``output_type=str``) so that
    parsantic's own parsing/coercion/alignment pipeline handles
    the structured output extraction downstream.
    """

    model_id: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_concurrency: int = 8
    _agent: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not _HAS_PYDANTIC_AI:
            raise ImportError(
                "pydantic-ai is required for this model. Install with:  uv add pydantic-ai"
            )
        model_spec = self.model_id or "openai:gpt-4o-mini"

        if self.api_key or self.base_url:
            model = _build_model_with_credentials(model_spec, self.api_key, self.base_url)
            self._agent = Agent(model, output_type=str)
        else:
            self._agent = Agent(model_spec, output_type=str)

    def infer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        results: list[str] = []
        for prompt in batch_prompts:
            result = self._agent.run_sync(prompt, **kwargs)
            results.append(result.output)
        return results

    async def ainfer(self, batch_prompts: Sequence[str], **kwargs: Any) -> Sequence[str]:
        concurrency = kwargs.pop("max_concurrency", self.max_concurrency)
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def _run_prompt(prompt: str) -> str:
            async with semaphore:
                result = await self._agent.run(prompt, **kwargs)
                return result.output

        return list(await asyncio.gather(*(_run_prompt(prompt) for prompt in batch_prompts)))
