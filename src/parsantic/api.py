from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from .coerce import CoerceOptions, _coerce_to_type, coerce_jsonish_to_python
from .jsonish import JsonishValue, ParseOptions, parse_jsonish
from .streaming import StreamParser
from .types import CandidateDebug, CompletionState, ParseDebug


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
    parse_options: ParseOptions | None = None,
    coerce_options: CoerceOptions | None = None,
) -> ParseResult[T]:
    """
    Parse raw model output into a value validated against a Pydantic v2 type.
    """
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    if is_done:
        _missing = object()
        try:
            validated = adapter.validate_json(text)
        except Exception:
            validated = _missing
        if validated is not _missing:
            return ParseResult(value=validated, flags=(), score=0)
    jsonish_value = parse_jsonish(text, options=parse_options or ParseOptions(), is_done=is_done)
    coerced = coerce_jsonish_to_python(
        jsonish_value, adapter, options=coerce_options or CoerceOptions()
    )
    validated = adapter.validate_python(coerced.value)
    return ParseResult(value=validated, flags=tuple(coerced.flags), score=coerced.score)


def coerce[T](
    value: Any,
    target: type[T] | TypeAdapter[T],
    *,
    options: CoerceOptions | None = None,
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
    except Exception:
        pass

    # Use the jsonish coercion path for dicts (model-level coercion)
    if isinstance(value, dict):
        jv = JsonishValue(
            value=value,
            completion=CompletionState.COMPLETE,
            raw=str(value),
        )
        sv = coerce_jsonish_to_python(jv, adapter, options=opts)
        validated = adapter.validate_python(sv.value)
        return ParseResult(value=validated, flags=tuple(sv.flags), score=sv.score)

    # Use recursive coercion for everything else
    target_type = target if isinstance(target, type) else getattr(adapter, "_type", type(value))
    sv = _coerce_to_type(value, target_type, opts)
    return ParseResult(value=sv.value, flags=tuple(sv.flags), score=sv.score)


def parse_debug[T](
    text: str,
    target: type[T] | TypeAdapter[T],
    *,
    is_done: bool = True,
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

    # Try direct JSON parse
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
    except Exception as e:
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
                sv = coerce_jsonish_to_python(cand, adapter, options=opts)
                candidates_debug.append(
                    CandidateDebug(
                        value_preview=sv.value,
                        flags=tuple(sv.flags),
                        score=sv.score,
                    )
                )
            except Exception as e:
                candidates_debug.append(
                    CandidateDebug(
                        value_preview=cand.value,
                        flags=tuple(cand.fixes),
                        score=-1,
                        validation_error=str(e),
                    )
                )

    # Get the actual result
    try:
        coerced = coerce_jsonish_to_python(jsonish_value, adapter, options=opts)
        validated = adapter.validate_python(coerced.value)
        chosen = CandidateDebug(
            value_preview=validated,
            flags=tuple(coerced.flags),
            score=coerced.score,
        )
        if chosen not in candidates_debug:
            candidates_debug.append(chosen)
        return ParseDebug(
            raw_text=text,
            candidates=candidates_debug,
            chosen=chosen,
            value=validated,
        )
    except Exception:
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
    except Exception as e:
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
    except Exception as e:
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
) -> StreamParser[T]:
    """
    Create a streaming parser. Feed text chunks into the returned StreamParser.
    """
    adapter: TypeAdapter[T] = target if isinstance(target, TypeAdapter) else TypeAdapter(target)
    return StreamParser(
        adapter=adapter,
        parse_options=parse_options or ParseOptions(),
        coerce_options=coerce_options or CoerceOptions(),
    )
