from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from parsantic.extract.providers import plugins
from parsantic.extract.providers.factory import _kwargs_with_environment_defaults
from parsantic.extract.providers.pydantic_ai_provider import PydanticAIProvider


@dataclass
class _BrokenEntryPoint:
    name: str = "broken_provider"

    def load(self) -> None:
        raise RuntimeError("boom")


def test_load_plugins_once_warns_on_failures(monkeypatch: pytest.MonkeyPatch):
    plugins.load_plugins_once.cache_clear()
    monkeypatch.delenv("PARSANTIC_DISABLE_PLUGINS", raising=False)
    monkeypatch.setattr(plugins, "_safe_entry_points", lambda _group: [_BrokenEntryPoint()])
    with pytest.warns(UserWarning, match="broken_provider"):
        plugins.load_plugins_once()
    plugins.load_plugins_once.cache_clear()


def test_factory_picks_provider_specific_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PARSANTIC_API_KEY", "fallback")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    resolved = _kwargs_with_environment_defaults("openai:gpt-4o-mini", {})
    assert resolved["api_key"] == "openai-key"


def test_factory_supports_additional_provider_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    resolved = _kwargs_with_environment_defaults("anthropic:claude-sonnet", {})
    assert resolved["api_key"] == "anthropic-key"

    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    resolved = _kwargs_with_environment_defaults("mistral:mistral-large", {})
    assert resolved["api_key"] == "mistral-key"

    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    resolved = _kwargs_with_environment_defaults("groq:llama-3.3-70b", {})
    assert resolved["api_key"] == "groq-key"


def test_factory_supports_provider_base_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://mistral.example")
    resolved = _kwargs_with_environment_defaults("mistral:mistral-large", {})
    assert resolved["base_url"] == "https://mistral.example"


@dataclass
class _TrackingAgent:
    active: int = 0
    max_active: int = 0
    seen_kwargs: list[dict[str, Any]] = field(default_factory=list)

    def run_sync(self, prompt: str, **kwargs: Any):
        self.seen_kwargs.append(kwargs)
        return SimpleNamespace(output=prompt.upper())

    async def run(self, prompt: str, **kwargs: Any):
        self.seen_kwargs.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return SimpleNamespace(output=prompt.upper())


def _provider_with_tracking_agent(
    max_concurrency: int = 2,
) -> tuple[PydanticAIProvider, _TrackingAgent]:
    provider = object.__new__(PydanticAIProvider)
    provider.model_id = "openai:gpt-4o-mini"
    provider.api_key = None
    provider.base_url = None
    provider.max_concurrency = max_concurrency
    agent = _TrackingAgent()
    provider._agent = agent
    return provider, agent


def test_pydantic_ai_provider_infer_passes_kwargs():
    provider, agent = _provider_with_tracking_agent()
    outputs = PydanticAIProvider.infer(provider, ["hello"], temperature=0.2)
    assert outputs == ["HELLO"]
    assert agent.seen_kwargs[-1]["temperature"] == 0.2


def test_pydantic_ai_provider_ainfer_runs_concurrently_and_passes_kwargs():
    provider, agent = _provider_with_tracking_agent(max_concurrency=2)
    outputs = asyncio.run(
        PydanticAIProvider.ainfer(
            provider,
            ["a", "b", "c", "d"],
            max_concurrency=2,
            temperature=0.1,
        )
    )
    assert outputs == ["A", "B", "C", "D"]
    assert agent.max_active <= 2
    assert all(kw.get("temperature") == 0.1 for kw in agent.seen_kwargs)
