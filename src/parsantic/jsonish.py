from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .fixing_parser import parse_fixing
from .types import CompletionState


@dataclass(slots=True)
class ParseOptions:
    allow_markdown_json: bool = True
    allow_find_all_json_objects: bool = True
    allow_fixes: bool = True
    allow_as_string: bool = True
    depth_limit: int = 100
    max_candidates: int = 64


@dataclass(frozen=True, slots=True)
class JsonishValue:
    """
    A JSON-ish value with completion state metadata and candidate provenance.

    This is intentionally lightweight vs the Rust implementation. It carries:
    - `value`: python scalar/list/dict
    - `completion`: COMPLETE/INCOMPLETE
    - `candidates`: optional alternate interpretations (AnyOf)
    - `raw`: original text the candidates came from
    - `fixes`: applied fixes (repair markers)
    """

    value: Any
    completion: CompletionState
    raw: str
    candidates: tuple[JsonishValue, ...] = ()
    fixes: tuple[str, ...] = ()

    def is_anyof(self) -> bool:
        return bool(self.candidates)


# Match opening markdown fences:
# - backtick (```) or tilde (~~~) fences
# - optional language tag (e.g. json, JSON, jsonc, application/json, json5, ...)
# - case-insensitive matching of the tag is handled downstream; the regex just captures it.
_MD_START = re.compile(r"(?m)^[ \t]*(?:```|~~~)([a-zA-Z0-9/_ .-]*)?(?:\n|$)")
_MD_END = re.compile(r"(?m)^[ \t]*(?:```|~~~)(?:\n|$)")


def parse_jsonish(
    text: str, *, options: ParseOptions, is_done: bool, _depth: int = 0
) -> JsonishValue:
    """
    Best-effort parse for LLM output.

    Mirrors the Rust jsonish pipeline:
    - strict JSON
    - markdown fenced blocks
    - find balanced JSON substrings
    - fixing/repair parser
    - fallback to string
    """
    if options.depth_limit <= 0:
        raise ValueError("depth_limit must be > 0")
    if _depth >= options.depth_limit:
        raise RecursionError(f"jsonish parse depth limit ({options.depth_limit}) exceeded")

    # 1) strict JSON
    try:
        val = json.loads(text)
    except Exception:
        val = None
    else:
        # match Rust: top-level numbers are treated as incomplete (streaming-friendly)
        completion = (
            CompletionState.INCOMPLETE
            if isinstance(val, (int, float))
            else CompletionState.COMPLETE
        )
        candidate = JsonishValue(value=val, completion=completion, raw=text)
        return JsonishValue(value=text, completion=completion, raw=text, candidates=(candidate,))

    candidates: list[JsonishValue] = []

    # 2) markdown fenced blocks
    if options.allow_markdown_json:
        candidates.extend(
            _parse_markdown_blocks(text, options=options, is_done=False, _depth=_depth)
        )

    # 3) balanced JSON substrings
    if options.allow_find_all_json_objects:
        candidates.extend(_parse_all_json_objects(text, options=options, is_done=False))

    # 4) fixing parser
    if options.allow_fixes:
        try:
            candidates.extend(_fixing_parse(text, is_done=is_done))
        except Exception:
            pass

    # 4b) fallback: try closing unclosed brackets/braces and re-parsing
    if not candidates and options.allow_fixes:
        closed = _close_unclosed_json(text)
        if closed != text:
            try:
                val = json.loads(closed)
                candidates.append(
                    JsonishValue(
                        value=val,
                        completion=CompletionState.INCOMPLETE,
                        raw=text,
                        fixes=("closed_unclosed",),
                    )
                )
            except Exception:
                pass

    # 5) fallback string
    if not candidates and options.allow_as_string:
        return JsonishValue(
            value=text,
            completion=CompletionState.COMPLETE if is_done else CompletionState.INCOMPLETE,
            raw=text,
        )

    if not candidates:
        raise ValueError("failed to parse")

    # Wrap as AnyOf – include original string as a last-resort scalar candidate.
    if options.allow_as_string:
        candidates.append(
            JsonishValue(
                value=text,
                completion=CompletionState.COMPLETE if is_done else CompletionState.INCOMPLETE,
                raw=text,
                fixes=("as_string",),
            )
        )

    # Truncate candidate list to max_candidates to prevent candidate explosion.
    # Prefer complete structured candidates with fewer fixes before truncating.
    if len(candidates) > options.max_candidates:
        candidates = _truncate_candidates(candidates, options.max_candidates)

    return JsonishValue(
        value=text, completion=_anyof_completion(candidates), raw=text, candidates=tuple(candidates)
    )


def _anyof_completion(candidates: list[JsonishValue]) -> CompletionState:
    if any(c.completion == CompletionState.INCOMPLETE for c in candidates):
        return CompletionState.INCOMPLETE
    return CompletionState.COMPLETE


def _candidate_priority(candidate: JsonishValue, idx: int) -> tuple[int, int, int, int, int]:
    completion_rank = 0 if candidate.completion == CompletionState.COMPLETE else 1
    if isinstance(candidate.value, dict):
        type_rank = 0
    elif isinstance(candidate.value, list):
        type_rank = 1
    else:
        type_rank = 2
    try:
        size_rank = -len(json.dumps(candidate.value, ensure_ascii=False, default=str))
    except Exception:
        size_rank = 0
    return (
        completion_rank,
        type_rank,
        len(candidate.fixes),
        size_rank,
        idx,
    )


def _truncate_candidates(candidates: list[JsonishValue], max_candidates: int) -> list[JsonishValue]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: _candidate_priority(pair[1], pair[0]),
    )
    return [candidate for _, candidate in ranked[:max_candidates]]


def _parse_markdown_blocks(
    text: str, *, options: ParseOptions, is_done: bool, _depth: int = 0
) -> list[JsonishValue]:
    values: list[JsonishValue] = []
    block_candidates: list[JsonishValue] = []
    remaining = text
    while True:
        start = _MD_START.search(remaining)
        if not start:
            break
        after_start = remaining[start.end() :]
        ends = list(_MD_END.finditer(after_start))
        if not ends:
            md_content = after_start.strip()
            remaining = ""
        else:
            chosen_end = ends[0]
            md_content = after_start[: chosen_end.start()].strip()
            # Prefer first closing fence that yields a successful parse
            for end in ends:
                candidate = after_start[: end.start()].strip()
                try:
                    parsed = parse_jsonish(
                        candidate,
                        options=ParseOptions(
                            allow_markdown_json=False,
                            allow_find_all_json_objects=False,
                            allow_fixes=True,
                            allow_as_string=True,
                            depth_limit=options.depth_limit,
                        ),
                        is_done=is_done,
                        _depth=_depth + 1,
                    )
                except Exception:
                    continue
                md_content = candidate
                chosen_end = end
                break
            remaining = remaining[start.end() + chosen_end.end() :]

        try:
            parsed = parse_jsonish(
                md_content,
                options=ParseOptions(
                    allow_markdown_json=False,
                    allow_find_all_json_objects=False,
                    allow_fixes=True,
                    allow_as_string=True,
                    depth_limit=options.depth_limit,
                ),
                is_done=is_done,
                _depth=_depth + 1,
            )
        except Exception:
            continue
        # Flatten nested AnyOf: markdown is a wrapper, not a candidate set by itself.
        if parsed.candidates:
            for cand in parsed.candidates:
                jv = JsonishValue(
                    value=cand.value,
                    completion=cand.completion,
                    raw=text,
                    fixes=("markdown",) + cand.fixes,
                )
                values.append(jv)
                block_candidates.append(jv)
        else:
            jv = JsonishValue(
                value=parsed.value,
                completion=parsed.completion,
                raw=text,
                fixes=("markdown",) + parsed.fixes,
            )
            values.append(jv)
            block_candidates.append(jv)

    if len(block_candidates) > 1:
        # Mirror Rust behavior: provide an "array of all markdown blocks" candidate.
        values.append(
            JsonishValue(
                value=[c.value for c in block_candidates],
                completion=CompletionState.INCOMPLETE,
                raw=text,
                fixes=("markdown_array",),
            )
        )

    if values and remaining.strip():
        # Mirror Rust behavior: keep the tail text after the final fenced block as a separate
        # candidate. This helps avoid "re-parsing" the entire input when multiple blocks exist.
        values.append(
            JsonishValue(
                value=remaining,
                completion=CompletionState.INCOMPLETE,
                raw=text,
                fixes=("markdown_tail",),
            )
        )
    return values


def _parse_all_json_objects(
    text: str, *, options: ParseOptions, is_done: bool
) -> list[JsonishValue]:
    """
    Grep balanced {...} and [...] regions, then strict-parse each region.
    """
    results: list[JsonishValue] = []
    for substring, completion in _balanced_json_substrings(text):
        try:
            parsed = json.loads(substring)
        except Exception:
            continue
        cmpl = CompletionState.COMPLETE if completion else CompletionState.INCOMPLETE
        results.append(
            JsonishValue(value=parsed, completion=cmpl, raw=text, fixes=("grepped_json",))
        )

    # If multiple objects found, include a list-of-them candidate
    if len(results) > 1:
        arr = [r.value for r in results]
        results.append(
            JsonishValue(
                value=arr, completion=CompletionState.INCOMPLETE, raw=text, fixes=("grepped_array",)
            )
        )
    return results


def _balanced_json_substrings(text: str) -> Iterator[tuple[str, bool]]:
    """
    Yield (substring, is_complete) for balanced brace/bracket regions.

    This is intentionally simpler than the Rust version; we avoid braces inside JSON strings.
    """
    stack: list[str] = []
    start_idx: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            opening = stack[-1]
            expected = "{" if ch == "}" else "["
            if opening != expected:
                # mismatched, reset
                stack.clear()
                start_idx = None
                continue
            stack.pop()
            if not stack and start_idx is not None:
                end = i + 1
                yield text[start_idx:end], True
                start_idx = None
    # Incomplete tail region
    if stack and start_idx is not None:
        yield text[start_idx:], False


def _fixing_parse(text: str, *, is_done: bool) -> list[JsonishValue]:
    # Full state-machine JSON-ish fixer, ported from BAML.
    raw = text
    try:
        candidates = parse_fixing(text)
    except Exception:
        return []

    out: list[JsonishValue] = []
    for cand in candidates:
        fixes = tuple(cand.fixes) + (
            ("closed_unclosed",) if cand.completion == CompletionState.INCOMPLETE else ()
        )
        out.append(
            JsonishValue(
                value=cand.value,
                completion=cand.completion,
                raw=raw,
                fixes=fixes,
            )
        )
    if len(out) > 1:
        out.append(
            JsonishValue(
                value=[c.value for c in out],
                completion=CompletionState.INCOMPLETE,
                raw=raw,
                fixes=("fixed_array",),
            )
        )
    return out


def _close_unclosed_json(s: str) -> str:
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            opening = stack[-1]
            expected = "{" if ch == "}" else "["
            if opening == expected:
                stack.pop()
            else:
                # ignore mismatch
                continue
    if not stack:
        return s
    # If we ended inside a quoted string, close it so the JSON becomes parseable.
    if in_string:
        s = s + '"'
    # Close in reverse order.
    closing = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return s + closing
