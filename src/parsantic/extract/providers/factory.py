from __future__ import annotations

import os
import warnings
from typing import Any

from .base import ProviderConfig
from .plugins import load_plugins_once
from .registry import resolve, resolve_provider


def _kwargs_with_environment_defaults(
    model_id: str | None, kwargs: dict[str, Any]
) -> dict[str, Any]:
    resolved = dict(kwargs)
    lowered = model_id.lower() if model_id else ""

    if "api_key" not in resolved:
        env_sources = []
        for env_var in ("PARSANTIC_API_KEY",):
            if os.getenv(env_var):
                env_sources.append((env_var, os.getenv(env_var)))

        if model_id:
            if "openai" in lowered or "gpt" in lowered:
                if os.getenv("OPENAI_API_KEY"):
                    env_sources.insert(0, ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
            if "anthropic" in lowered or "claude" in lowered:
                if os.getenv("ANTHROPIC_API_KEY"):
                    env_sources.insert(0, ("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY")))
            if "gemini" in lowered:
                if os.getenv("GEMINI_API_KEY"):
                    env_sources.insert(0, ("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")))
            if "mistral" in lowered:
                if os.getenv("MISTRAL_API_KEY"):
                    env_sources.insert(0, ("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY")))
            if "groq" in lowered:
                if os.getenv("GROQ_API_KEY"):
                    env_sources.insert(0, ("GROQ_API_KEY", os.getenv("GROQ_API_KEY")))

        if env_sources:
            resolved["api_key"] = env_sources[0][1]
            if len(env_sources) > 1:
                keys = ", ".join(k for k, _ in env_sources)
                warnings.warn(
                    f"Multiple API keys detected ({keys}); using {env_sources[0][0]}",
                    stacklevel=2,
                )

    if model_id and "ollama" in model_id.lower() and "base_url" not in resolved:
        resolved["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if "base_url" not in resolved:
        if "openai" in lowered and os.getenv("OPENAI_BASE_URL"):
            resolved["base_url"] = os.getenv("OPENAI_BASE_URL")
        elif "anthropic" in lowered and os.getenv("ANTHROPIC_BASE_URL"):
            resolved["base_url"] = os.getenv("ANTHROPIC_BASE_URL")
        elif "mistral" in lowered and os.getenv("MISTRAL_BASE_URL"):
            resolved["base_url"] = os.getenv("MISTRAL_BASE_URL")
        elif "groq" in lowered and os.getenv("GROQ_BASE_URL"):
            resolved["base_url"] = os.getenv("GROQ_BASE_URL")

    return resolved


def create_provider(config: ProviderConfig) -> Any:
    if not config.model_id and not config.provider:
        raise ValueError("Either model_id or provider must be specified")

    load_plugins_once()

    if config.provider:
        provider_class = resolve_provider(config.provider)
    else:
        provider_class = resolve(config.model_id or "")

    kwargs = _kwargs_with_environment_defaults(config.model_id, config.provider_kwargs)
    if config.model_id:
        kwargs["model_id"] = config.model_id

    return provider_class(**kwargs)
