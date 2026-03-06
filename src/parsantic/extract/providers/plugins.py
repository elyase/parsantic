from __future__ import annotations

import functools
import logging
import os
import warnings
from importlib import metadata
from importlib.metadata import EntryPoint

logger = logging.getLogger(__name__)


def _safe_entry_points(group: str) -> list[EntryPoint]:
    return list(metadata.entry_points().select(group=group))


@functools.cache
def load_plugins_once() -> None:
    if os.getenv("PARSANTIC_DISABLE_PLUGINS") == "1":
        return
    for ep in _safe_entry_points("parsantic.providers"):
        try:
            ep.load()
        except Exception as exc:
            warnings.warn(
                f"Failed to load parsantic provider plugin {ep.name!r}: {exc}",
                stacklevel=1,
            )
            logger.warning(
                "Failed to load parsantic provider plugin %r: %s",
                ep.name,
                exc,
            )
