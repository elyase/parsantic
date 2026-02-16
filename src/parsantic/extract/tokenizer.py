from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


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
        try:
            import regex as uni_regex  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised when missing
            raise ImportError(
                "UnicodeTokenizer requires the 'regex' package. Install with: pip install regex"
            ) from exc

        tokens: list[Token] = []
        previous_end = 0
        for match in uni_regex.finditer(r"\X", text):
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


def get_tokenizer(tokenizer: str | Tokenizer | None) -> Tokenizer:
    if tokenizer is None or tokenizer == "regex":
        return RegexTokenizer()
    if tokenizer == "unicode":
        return UnicodeTokenizer()
    return tokenizer


def tokens_lower(tokenized: TokenizedText) -> list[str]:
    return [t.text.lower() for t in tokenized.tokens]
