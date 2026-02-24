from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(slots=True)
class Token:
    text: str
    start: int
    end: int
    first_token_after_newline: bool = False


@dataclass(slots=True)
class TokenizedText:
    text: str
    tokens: list[Token]


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> TokenizedText: ...


_WORD_OR_PUNCT = re.compile(r"[\w']+|[^\w\s]+", re.UNICODE)
TokenizerName = Literal["regex", "unicode"]

try:
    import regex as _regex  # type: ignore

    _GRAPHEME_RE = _regex.compile(r"\X")
except ImportError:  # pragma: no cover - exercised when missing
    _regex = None
    _GRAPHEME_RE = None


class RegexTokenizer:
    def tokenize(self, text: str) -> TokenizedText:
        tokens: list[Token] = []
        previous_end = 0
        for match in _WORD_OR_PUNCT.finditer(text):
            start, end = match.span()
            token_text = match.group(0)
            first_after_newline = False
            if start > previous_end:
                gap = text[previous_end:start]
                if "\n" in gap or "\r" in gap:
                    first_after_newline = True
            tokens.append(
                Token(
                    text=token_text,
                    start=start,
                    end=end,
                    first_token_after_newline=first_after_newline,
                )
            )
            previous_end = end
        return TokenizedText(text=text, tokens=tokens)


class UnicodeTokenizer:
    def tokenize(self, text: str) -> TokenizedText:
        if _regex is None or _GRAPHEME_RE is None:
            raise ImportError(
                "UnicodeTokenizer requires the 'regex' package. Install with: pip install regex"
            )

        tokens: list[Token] = []
        previous_end = 0
        for match in _GRAPHEME_RE.finditer(text):
            grapheme = match.group(0)
            start, end = match.span()
            if grapheme.isspace():
                continue
            first_after_newline = False
            if start > previous_end:
                gap = text[previous_end:start]
                if "\n" in gap or "\r" in gap:
                    first_after_newline = True
            tokens.append(
                Token(
                    text=grapheme,
                    start=start,
                    end=end,
                    first_token_after_newline=first_after_newline,
                )
            )
            previous_end = end
        return TokenizedText(text=text, tokens=tokens)


def get_tokenizer(tokenizer: TokenizerName | Tokenizer | None) -> Tokenizer:
    if tokenizer is None or tokenizer == "regex":
        return RegexTokenizer()
    if tokenizer == "unicode":
        return UnicodeTokenizer()
    if isinstance(tokenizer, str):
        raise ValueError(f"Unknown tokenizer {tokenizer!r}. Expected one of: 'regex', 'unicode'.")
    return tokenizer


def tokens_lower(tokenized: TokenizedText) -> list[str]:
    return [t.text.lower() for t in tokenized.tokens]
