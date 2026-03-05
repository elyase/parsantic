from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def normalize_text_outputs(
    outputs: Any,
    *,
    expected_count: int,
    context: str = "provider",
) -> list[str]:
    """Normalize provider output into a ``list[str]`` with strict shape checks."""
    if outputs is None:
        raise TypeError(f"{context} returned None, expected a list of strings")
    if isinstance(outputs, str):
        normalized: list[Any] = [outputs]
    elif isinstance(outputs, (dict, set, frozenset)):
        raise TypeError(f"{context} returned {type(outputs).__name__}, expected a list of strings")
    elif isinstance(outputs, Sequence):
        normalized = list(outputs)
    else:
        normalized = list(outputs)

    for idx, item in enumerate(normalized):
        if not isinstance(item, str):
            raise TypeError(
                f"{context} output at index {idx} must be str, got {type(item).__name__}"
            )

    if len(normalized) != expected_count:
        raise ValueError(
            f"{context} returned {len(normalized)} outputs for {expected_count} prompts"
        )

    return normalized
