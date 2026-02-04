from __future__ import annotations

import functools
import os
from importlib import metadata


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
        except Exception:
            # best-effort: plugin import failure shouldn't crash core
            continue
