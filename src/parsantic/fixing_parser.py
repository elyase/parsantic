from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Literal

from .types import CompletionState

# This module is a Python port (with some pragmatic simplifications) of
# `engine/baml-lib/jsonish/src/jsonish/parser/fixing_parser/*`.
#
# Core idea:
# - parse *invalid* JSON-ish text using a character-level state machine
# - tolerate missing commas/colons, comments, unterminated collections/strings
# - produce one or more candidate Python values


Fix = Literal[
    "fixed_json",
    "inferred_array",
]


@dataclass(slots=True)
class FixingCandidate:
    value: Any
    completion: CompletionState
    fixes: tuple[Fix, ...]


@dataclass(slots=True)
class _Object:
    keys: list[str]
    values: list[Any]
    completion: CompletionState


@dataclass(slots=True)
class _Array:
    values: list[Any]
    completion: CompletionState


@dataclass(slots=True)
class _QuotedString:
    quote: Literal['"', "'", "`"]
    buf: list[str]
    completion: CompletionState


@dataclass(slots=True)
class _TripleQuotedString:
    quote: Literal['"', "`"]
    buf: list[str]
    completion: CompletionState


@dataclass(slots=True)
class _UnquotedString:
    buf: list[str]
    completion: CompletionState


@dataclass(slots=True)
class _LineComment:
    buf: list[str]
    completion: CompletionState


@dataclass(slots=True)
class _BlockComment:
    buf: list[str]
    completion: CompletionState


_Collection = (
    _Object
    | _Array
    | _QuotedString
    | _TripleQuotedString
    | _UnquotedString
    | _LineComment
    | _BlockComment
)


@dataclass(slots=True)
class _Frame:
    col: _Collection
    fixes: list[Fix]


class FixingParser:
    def __init__(self) -> None:
        self.stack: list[_Frame] = []
        # completed values at the top level.
        # we keep the "kind" to emulate the Rust behavior (special-case all-strings).
        self.completed: list[tuple[str, Any, CompletionState, list[Fix]]] = []

    def parse(self, text: str) -> list[FixingCandidate]:
        i = 0
        while i < len(text):
            skip = self._process_char(text, i)
            i += 1 + skip

        # Close anything still open as incomplete
        while self.stack:
            self._complete_top(CompletionState.INCOMPLETE)

        if not self.completed:
            raise ValueError("No JSON objects found")

        if len(self.completed) == 1:
            kind, value, completion, fixes = self.completed[0]
            return [FixingCandidate(value=value, completion=completion, fixes=tuple(fixes))]

        all_strings = all(kind == "string" for kind, *_rest in self.completed)
        if all_strings:
            # If all the values are strings, return them as an array of strings,
            # mirroring Rust `Fixes::InferredArray`.
            items: list[Any] = [value for _kind, value, _cmpl, _fixes in self.completed]
            completion = CompletionState.INCOMPLETE
            return [
                FixingCandidate(
                    value=items,
                    completion=completion,
                    fixes=("inferred_array",),
                )
            ]

        # Otherwise, return only objects and arrays.
        candidates: list[FixingCandidate] = []
        for kind, value, completion, fixes in self.completed:
            if kind in {"object", "array"}:
                candidates.append(
                    FixingCandidate(value=value, completion=completion, fixes=tuple(fixes))
                )
        if not candidates:
            raise ValueError("No JSON objects found")
        return candidates

    # ----------------------------
    # Parsing
    # ----------------------------

    def _process_char(self, text: str, idx: int) -> int:
        token = text[idx]
        look = text[idx + 1 :]

        if not self.stack:
            return self._find_any_starting_value(token, look, in_object=False)

        top = self.stack[-1].col

        if isinstance(top, _Object):
            if token == "}":
                self._complete_top(CompletionState.COMPLETE)
                return 0
            if token in {",", ":"}:
                return 0
            return self._find_any_starting_value(token, look, in_object=True)

        if isinstance(top, _Array):
            if token == "]":
                self._complete_top(CompletionState.COMPLETE)
                return 0
            if token == ",":
                return 0
            return self._find_any_starting_value(token, look, in_object=False)

        if isinstance(top, _TripleQuotedString):
            # Close on """ or ``` depending on quote.
            if token == top.quote:
                if look.startswith(top.quote * 2):
                    # For triple-backticks, avoid closing on sequences like ```json
                    # which commonly appear inside the content. Prefer closing fences
                    # that look like terminators: ```<ws|,|}|]|eof>
                    if top.quote == "`":
                        after = look[2:3]
                        if after and (after.isalnum() or after in {"_", "-"}):
                            top.buf.append(token)
                            return 0
                    self._complete_top(CompletionState.COMPLETE)
                    return 2
            top.buf.append(token)
            return 0

        if isinstance(top, _QuotedString):
            if token == top.quote:
                if self._should_close_string(look, closing=top.quote):
                    self._complete_top(CompletionState.COMPLETE)
                    return 0
                top.buf.append(token)
                return 0

            if top.quote == '"' and token == "\\":
                # handle escape sequences like Rust does (consume next char)
                if not look:
                    top.buf.append(token)
                    return 0
                n = look[0]
                if n == "n":
                    top.buf.append("\n")
                    return 1
                if n == "t":
                    top.buf.append("\t")
                    return 1
                if n == "r":
                    top.buf.append("\r")
                    return 1
                if n == "b":
                    top.buf.append("\b")
                    return 1
                if n == "f":
                    top.buf.append("\f")
                    return 1
                if n == "\\":
                    top.buf.append("\\")
                    return 1
                if n == '"':
                    top.buf.append('"')
                    return 1
                if n == "u":
                    # Keep raw \uXXXX
                    hex_part = look[1:5]
                    top.buf.append("\\u" + hex_part)
                    return 1 + len(hex_part)
                # default: treat as literal backslash
                top.buf.append(token)
                return 0

            top.buf.append(token)
            return 0

        if isinstance(top, _UnquotedString):
            top.buf.append(token)
            if self._should_close_unquoted(top, look):
                # When we close due to a delimiter that exists in lookahead, this value is "complete".
                completion = CompletionState.COMPLETE if look else CompletionState.INCOMPLETE
                self._complete_top(completion)
            return 0

        if isinstance(top, _LineComment):
            if token == "\n":
                self._complete_top(CompletionState.COMPLETE)
                return 0
            top.buf.append(token)
            return 0

        if isinstance(top, _BlockComment):
            if token == "*" and look.startswith("/"):
                self._complete_top(CompletionState.COMPLETE)
                return 1
            top.buf.append(token)
            return 0

        raise AssertionError(f"unhandled token {token!r} in {top!r}")

    def _find_any_starting_value(self, token: str, look: str, *, in_object: bool) -> int:
        if token == "{":
            self.stack.append(
                _Frame(_Object(keys=[], values=[], completion=CompletionState.INCOMPLETE), fixes=[])
            )
            return 0
        if token == "[":
            self.stack.append(
                _Frame(_Array(values=[], completion=CompletionState.INCOMPLETE), fixes=[])
            )
            return 0
        if token == '"':
            if look.startswith('""'):
                self.stack.append(
                    _Frame(
                        _TripleQuotedString(
                            quote='"', buf=[], completion=CompletionState.INCOMPLETE
                        ),
                        fixes=[],
                    )
                )
                return 2
            self.stack.append(
                _Frame(
                    _QuotedString(quote='"', buf=[], completion=CompletionState.INCOMPLETE),
                    fixes=[],
                )
            )
            return 0
        if token == "'":
            self.stack.append(
                _Frame(
                    _QuotedString(quote="'", buf=[], completion=CompletionState.INCOMPLETE),
                    fixes=[],
                )
            )
            return 0
        if token == "`":
            if look.startswith("``"):
                self.stack.append(
                    _Frame(
                        _TripleQuotedString(
                            quote="`", buf=[], completion=CompletionState.INCOMPLETE
                        ),
                        fixes=[],
                    )
                )
                return 2
            self.stack.append(
                _Frame(
                    _QuotedString(quote="`", buf=[], completion=CompletionState.INCOMPLETE),
                    fixes=[],
                )
            )
            return 0
        if token == "/":
            if look.startswith("/"):
                self.stack.append(
                    _Frame(_LineComment(buf=[], completion=CompletionState.INCOMPLETE), fixes=[])
                )
                return 1
            if look.startswith("*"):
                self.stack.append(
                    _Frame(_BlockComment(buf=[], completion=CompletionState.INCOMPLETE), fixes=[])
                )
                return 1
            # If we're in an object, allow paths like /tmp/foo as unquoted strings.
            if in_object:
                self.stack.append(
                    _Frame(
                        _UnquotedString(buf=[token], completion=CompletionState.INCOMPLETE),
                        fixes=[],
                    )
                )
            return 0
        if token == "#":
            self.stack.append(
                _Frame(_LineComment(buf=[], completion=CompletionState.INCOMPLETE), fixes=[])
            )
            return 0
        if token.isspace():
            return 0

        # default: start unquoted token (numbers, identifiers, etc.)
        unq = _UnquotedString(buf=[token], completion=CompletionState.INCOMPLETE)
        self.stack.append(_Frame(unq, fixes=[]))
        # Mirror Rust behavior: a single-token unquoted string may be complete immediately
        # if the next character is a structural delimiter.
        if self._should_close_unquoted(unq, look):
            completion = CompletionState.COMPLETE if look else CompletionState.INCOMPLETE
            self._complete_top(completion)
        return 0

    # ----------------------------
    # Close heuristics
    # ----------------------------

    def _parent_context(self) -> Literal["none", "object_key", "object_value", "array", "unknown"]:
        if len(self.stack) < 2:
            return "none"
        parent = self.stack[-2].col
        if isinstance(parent, _Object):
            return "object_key" if len(parent.keys) == len(parent.values) else "object_value"
        if isinstance(parent, _Array):
            return "array"
        return "unknown"

    def _peek_next_non_ws(self, look: str) -> str | None:
        for ch in look:
            if not ch.isspace():
                return ch
        return None

    def _should_close_string(self, look: str, *, closing: str) -> bool:
        nxt = self._peek_next_non_ws(look)
        if nxt is None:
            return True
        ls = look.lstrip()
        if ls.startswith("//") or ls.startswith("/*") or ls.startswith("#"):
            return True
        ctx = self._parent_context()
        if ctx == "object_key":
            return nxt in {":", "}"}
        if ctx in {"object_value", "array"}:
            return nxt in {",", "}", "]"}
        return nxt in {",", "}", "]", ":"}

    def _should_close_unquoted(self, current: _UnquotedString, look: str) -> bool:
        nxt = self._peek_next_non_ws(look)
        if nxt is None:
            return True
        ctx = self._parent_context()

        # Always close on structural delimiters that imply end-of-token.
        if ctx == "object_key":
            if nxt in {":", "}"}:
                return True
        elif ctx == "object_value":
            if nxt == "}":
                return True
            if nxt == ",":
                # `,` normally terminates a value, but allow commas inside unquoted
                # multi-line prose (common in model output) when the comma doesn't
                # look like a field separator.
                ls = look.lstrip()
                if ls.startswith(","):
                    after = ls[1:]
                    after_ws = after.lstrip()
                    # trailing comma before end-of-object
                    if after_ws.startswith("}"):
                        return True
                    # comma + next key (`foo: ...`) before any other terminator
                    colon_idx = after.find(":")
                    if colon_idx != -1:
                        stops = [
                            i
                            for i in (after.find(","), after.find("}"), after.find("]"))
                            if i != -1
                        ]
                        stop_idx = min(stops) if stops else len(after)
                        if colon_idx < stop_idx:
                            return True
                return False
        elif ctx == "array":
            if nxt in {",", "]"}:
                return True
        elif ctx == "none":
            if nxt in {"{", "["}:
                return True

        # Heuristic for missing commas/colons:
        # If current value looks like a compact "atomic" token (number/bool/null/identifier),
        # and the next significant character looks like the beginning of another token (quote, brace, identifier),
        # treat the current token as done.
        s = "".join(current.buf).strip()
        is_numeric = _looks_like_number(s)
        is_bool = s.lower() in {"true", "false"}
        is_null = s.lower() == "null"
        is_identifier = (" " not in s) and ("(" not in s)
        is_possible_value = is_numeric or is_bool or is_null or is_identifier
        if not is_possible_value:
            return False

        if ctx == "object_value":
            # likely start of a new key or a comment
            if nxt == "/":
                ls = look.lstrip()
                if ls.startswith("//") or ls.startswith("/*"):
                    return True
                if is_numeric and ls.startswith("/"):
                    rest = ls[1:].lstrip()
                    if rest and (rest[0].isdigit() or rest[0] == "."):
                        return False
                return True
            if nxt in {'"', "'", "{", "[", "#"}:
                return True
            if look and look[0].isspace() and (nxt.isalpha() or nxt in {"_"}):
                # Allow unquoted multi-word string values like:
                #   key: value with spaces,
                # and only split on whitespace when it looks like the start
                # of a new key (e.g. `b: 2` before any `,`/`}`/`]`).
                rest = look.lstrip()
                colon_idx = rest.find(":")
                if colon_idx != -1:
                    stops = [i for i in (rest.find(","), rest.find("}"), rest.find("]")) if i != -1]
                    stop_idx = min(stops) if stops else len(rest)
                    if colon_idx < stop_idx:
                        return True

        if ctx == "array":
            if nxt == "/":
                ls = look.lstrip()
                if ls.startswith("//") or ls.startswith("/*"):
                    return True
                if is_numeric and ls.startswith("/"):
                    rest = ls[1:].lstrip()
                    if rest and (rest[0].isdigit() or rest[0] == "."):
                        return False
                return True
            if nxt in {'"', "'", "{", "[", "#"}:
                return True
            if look and look[0].isspace() and (nxt.isalpha() or nxt in {"_"}):
                return True

        return False

    # ----------------------------
    # Completion / conversion
    # ----------------------------

    def _complete_top(self, completion: CompletionState) -> None:
        frame = self.stack.pop()
        col = frame.col

        kind: str
        value: Any | None

        if isinstance(col, _Object):
            kind = "object"
            col.completion = completion
            value = {k: v for k, v in zip(col.keys, col.values, strict=False)}
        elif isinstance(col, _Array):
            kind = "array"
            col.completion = completion
            value = list(col.values)
        elif isinstance(col, _QuotedString):
            kind = "string"
            col.completion = completion
            value = "".join(col.buf)
        elif isinstance(col, _TripleQuotedString):
            kind = "string"
            col.completion = completion
            raw = "".join(col.buf)
            if col.quote == '"':
                value = textwrap.dedent(raw).strip("\n")
            else:
                # triple backticks: drop first line like fenced code blocks, then dedent.
                if "\n" not in raw:
                    value = raw.strip("\n")
                else:
                    _info, rest = raw.split("\n", 1)
                    value = textwrap.dedent(rest).strip("\n")
        elif isinstance(col, _UnquotedString):
            kind = "unquoted"
            col.completion = completion
            value = _coerce_unquoted("".join(col.buf))
            # classify "string-ness" for inferred-array behavior
            if isinstance(value, str):
                kind = "string"
        elif isinstance(col, _LineComment) or isinstance(col, _BlockComment):
            # Comments are discarded.
            return
        else:
            raise AssertionError(f"unknown collection type: {type(col)}")

        # attach "fixed_json" to any value produced by this parser
        frame.fixes.append("fixed_json")

        if self.stack:
            parent = self.stack[-1].col
            if isinstance(parent, _Object):
                if len(parent.keys) == len(parent.values):
                    parent.keys.append(_as_key_string(value))
                else:
                    parent.values.append(value)
                return
            if isinstance(parent, _Array):
                parent.values.append(value)
                return
            # If we are inside a string/comment, this should not happen.
            return

        self.completed.append((kind, value, completion, frame.fixes))


def parse_fixing(text: str) -> list[FixingCandidate]:
    return FixingParser().parse(text)


def _as_key_string(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_unquoted(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False
    if sl == "null":
        return None
    # ints
    try:
        return int(s)
    except Exception:
        pass
    # floats
    try:
        return float(s)
    except Exception:
        return s


def _looks_like_number(s: str) -> bool:
    if not s:
        return False
    # Accept leading dot (".5") and basic exponent forms.
    try:
        float(s)
        return True
    except Exception:
        return False
