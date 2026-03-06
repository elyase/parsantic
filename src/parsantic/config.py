"""Centralized configuration and defaults for parsantic."""

from __future__ import annotations

import os

DEFAULT_MODEL = "openai:gpt-4o-mini"
"""Default model used when none is specified and PARSANTIC_MODEL is not set."""


def resolve_model(model: str | object | None = None) -> str | object:
    """Return an explicit model or provider, falling back to env var or built-in default.

    When *model* is a non-string object (e.g. a provider instance), it is
    returned as-is.  When *model* is ``None``, the ``PARSANTIC_MODEL`` env var
    or :data:`DEFAULT_MODEL` is used.
    """
    if model is not None:
        return model
    env = os.environ.get("PARSANTIC_MODEL", "").strip()
    return env or DEFAULT_MODEL


__all__ = ["DEFAULT_MODEL", "resolve_model"]
