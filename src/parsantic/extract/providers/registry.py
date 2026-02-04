from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegistryEntry:
    patterns: tuple[re.Pattern[str], ...]
    provider: type[Any]
    priority: int


_REGISTRY: list[RegistryEntry] = []


def register(*patterns: str, priority: int = 0) -> Callable[[type[Any]], type[Any]]:
    def decorator(cls: type[Any]) -> type[Any]:
        compiled = tuple(re.compile(p) for p in patterns)
        _REGISTRY.append(RegistryEntry(patterns=compiled, provider=cls, priority=priority))
        _REGISTRY.sort(key=lambda e: e.priority, reverse=True)
        return cls

    return decorator


def resolve(model_id: str) -> type[Any]:
    for entry in _REGISTRY:
        if any(p.search(model_id) for p in entry.patterns):
            return entry.provider
    raise ValueError(f"No provider registered for model_id={model_id!r}")


def resolve_provider(name: str) -> type[Any]:
    lname = name.lower()
    for entry in _REGISTRY:
        if entry.provider.__name__.lower() == lname:
            return entry.provider
        if lname in entry.provider.__name__.lower():
            return entry.provider
    raise ValueError(f"No provider registered with name={name!r}")
