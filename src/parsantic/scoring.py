"""Candidate scoring engine for schema-aligned parsing.

Flag weights (lower is better) control how coercion candidates are ranked.
Inspired by ``engine/baml-lib/jsonish/src/deserializer/score.rs``.
"""

from __future__ import annotations

from collections.abc import Iterable

from .types import ScoredValue

# Flag weights (lower is better).
FLAG_WEIGHTS: dict[str, int] = {
    "fixed_json": 0,
    "fixed_array": 1,
    "markdown": 0,
    "markdown_array": 1,
    "closed_unclosed": 0,
    "grepped_json": 0,
    "grepped_array": 1,
    "markdown_tail": 5,
    "inferred_array": 5,
    "as_string": 2,
    "single_to_array": 1,
    "object_to_string": 2,
    "string_to_int": 1,
    "string_to_float": 1,
    "string_to_bool": 1,
    "float_to_int": 1,
    "extra_key": 1,
    "substring_match": 2,
    "strip_punct": 3,
    "case_insensitive": 3,
    "accent_insensitive": 2,
    "key_normalized": 3,
    "implied_key": 2,
    "partial_model": 0,
    "partial_unvalidated": 50,
    "ambiguous_key": 10,
    "ambiguous_key_kept": 8,
    "ambiguous_enum": 20,
    "key_collision": 5,
    "max_depth_exceeded": 50,
}

_UNKNOWN_FLAG_PENALTY = 5


def score_flags(flags: Iterable[str]) -> int:
    """Compute total penalty score for a set of flags."""
    return sum(FLAG_WEIGHTS.get(f, _UNKNOWN_FLAG_PENALTY) for f in flags)


def pick_best(scored: list[ScoredValue]) -> ScoredValue:
    """Select the best candidate by (score, flag count, generation order)."""
    return sorted(
        enumerate(scored),
        key=lambda pair: (pair[1].score, len(pair[1].flags), pair[0]),
    )[0][1]
