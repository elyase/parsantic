from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .coerce import CoerceOptions, coerce_jsonish_to_python
from .jsonish import JsonishValue, ParseOptions, parse_jsonish
from .streaming import StreamParser
from .types import CandidateDebug, CompletionState, ParseDebug

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParseResult[T]:
    value: T
    flags: tuple[str, ...]
    score: int


def parse[T](
    text: str,
    target: type[T] | TypeAdapter[T],
    *,
    is_done: bool = True,
    allow_partial: bool = False,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
) -> ParseResult[T]:
    """
    Parse raw model output into a value validated against a Pydantic v2 type.
    """
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    if is_done:
        try:
            validated = adapter.validate_json(text)
            return ParseResult(value=validated, flags=(), score=0)
        except (ValidationError, ValueError):
            logger.debug("Direct JSON validation failed, falling back to jsonish pipeline")
    jsonish_value = parse_jsonish(text, options=parse_options or ParseOptions(), is_done=is_done)
    coerced = coerce_jsonish_to_python(
        jsonish_value,
        adapter,
        options=coerce_options or CoerceOptions(),
        allow_partial=allow_partial,
    )
    return ParseResult(value=coerced.value, flags=tuple(coerced.flags), score=coerced.score)


def coerce[T](
    value: Any,
    target: type[T] | TypeAdapter[T],
    *,
    options: CoerceOptions | None = None,
    allow_partial: bool = False,
) -> ParseResult[T]:
    """
    Coerce a Python object to match the target schema.

    Use this when you already have python objects (e.g., tool call args)
    but still want schema-aligned coercions + scoring.
    """
    opts = options or CoerceOptions()
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)

    # Fast path: already valid
    try:
        validated = adapter.validate_python(value)
        return ParseResult(value=validated, flags=(), score=0)
    except ValidationError:
        pass

    jv = JsonishValue(
        value=value,
        completion=CompletionState.COMPLETE,
        raw=str(value),
    )
    sv = coerce_jsonish_to_python(jv, adapter, options=opts, allow_partial=allow_partial)
    return ParseResult(value=sv.value, flags=tuple(sv.flags), score=sv.score)


def parse_debug[T](
    text: str,
    target: type[T] | TypeAdapter[T],
    *,
    is_done: bool = True,
    allow_partial: bool = False,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
) -> ParseDebug[T]:
    """
    Parse raw model output with full debug trace.

    Returns a ParseDebug with all candidate interpretations, the selected
    candidate, and the final value.
    """
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    opts = coerce_options or CoerceOptions()

    candidates_debug: list[CandidateDebug] = []

    # Try direct JSON parse only when the buffer is complete.
    if is_done:
        direct_error: str | None = None
        try:
            validated = adapter.validate_json(text)
            chosen = CandidateDebug(
                value_preview=validated,
                flags=(),
                score=0,
            )
            candidates_debug.append(chosen)
            return ParseDebug(
                raw_text=text,
                candidates=candidates_debug,
                chosen=chosen,
                value=validated,
            )
        # Keep direct JSON parse failures as a debug candidate before falling back to jsonish.
        except (ValidationError, ValueError) as e:
            direct_error = str(e)
            candidates_debug.append(
                CandidateDebug(
                    value_preview=text[:200] if len(text) > 200 else text,
                    flags=(),
                    score=-1,
                    validation_error=direct_error,
                )
            )

    # Parse through jsonish
    jsonish_value = parse_jsonish(text, options=parse_options or ParseOptions(), is_done=is_done)

    # Collect all candidates from jsonish
    if jsonish_value.candidates:
        for cand in jsonish_value.candidates:
            try:
                sv = coerce_jsonish_to_python(
                    cand,
                    adapter,
                    options=opts,
                    allow_partial=allow_partial,
                )
                candidates_debug.append(
                    CandidateDebug(
                        value_preview=sv.value,
                        flags=tuple(sv.flags),
                        score=sv.score,
                    )
                )
            # If a candidate fails to coerce, keep the raw representation for debugging.
            except (ValidationError, ValueError, TypeError) as e:
                candidates_debug.append(
                    CandidateDebug(
                        value_preview=cand.value,
                        flags=tuple(cand.fixes),
                        score=-1,
                        validation_error=str(e),
                    )
                )

    logger.debug("parse_debug: collected %d candidates", len(candidates_debug))

    # Get the actual result
    try:
        coerced = coerce_jsonish_to_python(
            jsonish_value,
            adapter,
            options=opts,
            allow_partial=allow_partial,
        )
        chosen = CandidateDebug(
            value_preview=coerced.value,
            flags=tuple(coerced.flags),
            score=coerced.score,
        )
        if chosen not in candidates_debug:
            candidates_debug.append(chosen)
        return ParseDebug(
            raw_text=text,
            candidates=candidates_debug,
            chosen=chosen,
            value=coerced.value,
        )
    except (ValidationError, ValueError, TypeError):
        return ParseDebug(
            raw_text=text,
            candidates=candidates_debug,
            chosen=None,
            value=None,
        )


def coerce_debug[T](
    value: Any,
    target: type[T] | TypeAdapter[T],
    *,
    options: CoerceOptions | None = None,
) -> ParseDebug[T]:
    """
    Coerce a Python object with full debug trace.

    Returns a ParseDebug with candidate info and the final value.
    """
    opts = options or CoerceOptions()
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    candidates_debug: list[CandidateDebug] = []

    # Try direct validation
    try:
        validated = adapter.validate_python(value)
        chosen = CandidateDebug(
            value_preview=validated,
            flags=(),
            score=0,
        )
        candidates_debug.append(chosen)
        return ParseDebug(
            raw_text=None,
            candidates=candidates_debug,
            chosen=chosen,
            value=validated,
        )
    except (ValidationError, ValueError, TypeError) as e:
        candidates_debug.append(
            CandidateDebug(
                value_preview=value,
                flags=(),
                score=-1,
                validation_error=str(e),
            )
        )

    # Coerce
    try:
        result = coerce(value, target, options=opts)
        chosen = CandidateDebug(
            value_preview=result.value,
            flags=result.flags,
            score=result.score,
        )
        candidates_debug.append(chosen)
        return ParseDebug(
            raw_text=None,
            candidates=candidates_debug,
            chosen=chosen,
            value=result.value,
        )
    except (ValidationError, ValueError, TypeError) as e:
        candidates_debug.append(
            CandidateDebug(
                value_preview=value,
                flags=(),
                score=-1,
                validation_error=str(e),
            )
        )
        return ParseDebug(
            raw_text=None,
            candidates=candidates_debug,
            chosen=None,
            value=None,
        )


def parse_stream[T](
    target: type[T] | TypeAdapter[T],
    *,
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
    max_buffer_chars: int | None = None,
) -> StreamParser[T]:
    """
    Create a streaming parser. Feed text chunks into the returned StreamParser.
    """
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    return StreamParser(
        adapter=adapter,
        parse_options=parse_options or ParseOptions(),
        coerce_options=coerce_options or CoerceOptions(),
        max_buffer_chars=max_buffer_chars,
    )
