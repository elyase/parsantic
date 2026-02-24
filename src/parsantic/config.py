"""Centralized configuration and defaults for parsantic."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = "openai:gpt-4o-mini"
"""Default model used when none is specified and PARSANTIC_MODEL is not set."""


def resolve_model(model: str | Any | None = None) -> str | Any:
    """Return an explicit model or provider, falling back to env var or built-in default.

    When *model* is a non-string object (e.g. a provider instance), it is
    returned as-is.  When *model* is ``None``, the ``PARSANTIC_MODEL`` env var
    or :data:`DEFAULT_MODEL` is used.
    """
    if model is not None:
        return model
    return os.environ.get("PARSANTIC_MODEL", DEFAULT_MODEL)
