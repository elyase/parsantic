from __future__ import annotations

import functools
import logging
import os
from importlib import metadata

logger = logging.getLogger(__name__)


def _safe_entry_points(group: str) -> list:
    eps = metadata.entry_points()
    try:
        return list(eps.select(group=group))
    except AttributeError:
        return list(eps.get(group, []))


@functools.lru_cache(maxsize=1)
def load_plugins_once() -> None:
    if os.getenv("PARSANTIC_DISABLE_PLUGINS") == "1":
        return
    for ep in _safe_entry_points("parsantic.providers"):
        try:
            ep.load()
        except Exception as exc:
            # best-effort: plugin import failure shouldn't crash core
            import warnings

            warnings.warn(
                f"Failed to load parsantic provider plugin {ep.name!r}: {exc}",
                stacklevel=2,
            )
            logger.warning(
                "Failed to load parsantic provider plugin %r: %s",
                ep.name,
                exc,
            )
