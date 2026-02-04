from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .tokenizer import TokenizedText, Tokenizer, get_tokenizer

_SENTENCE_END = re.compile(r"[.!?。！？]$")


@dataclass(slots=True)
class TextChunk:
    text: str
    start: int
    end: int


def _is_sentence_end(token_text: str) -> bool:
    return bool(_SENTENCE_END.search(token_text))


def iter_chunks(
    text: str,
    *,
    max_char_buffer: int | None,
    tokenizer: Tokenizer | None = None,
    overlap_chars: int = 0,
) -> Iterator[TextChunk]:
    """Iterate over text chunks respecting sentence/newline boundaries.

    Parameters
    ----------
    overlap_chars:
        When > 0, each chunk (except the first) starts ``overlap_chars``
        characters before where it would normally start.  This reduces
        boundary loss in extraction by providing context overlap between
        adjacent chunks.
    """
    if max_char_buffer is None or max_char_buffer <= 0:
        yield TextChunk(text=text, start=0, end=len(text))
        return

    tok = get_tokenizer(tokenizer)
    tokenized: TokenizedText = tok.tokenize(text)

    if not tokenized.tokens:
        yield TextChunk(text=text, start=0, end=len(text))
        return

    is_first_chunk = True
    start_idx = 0
    while start_idx < len(tokenized.tokens):
        start_char = tokenized.tokens[start_idx].start
        # Oversized token handling
        if tokenized.tokens[start_idx].end - start_char > max_char_buffer:
            end_char = tokenized.tokens[start_idx].end
            yield TextChunk(text=text[start_char:end_char], start=start_char, end=end_char)
            start_idx += 1
            is_first_chunk = False
            continue

        end_idx = start_idx
        last_break_idx = None
        while end_idx < len(tokenized.tokens):
            end_char = tokenized.tokens[end_idx].end
            if end_char - start_char > max_char_buffer:
                break
            if tokenized.tokens[end_idx].first_token_after_newline:
                last_break_idx = end_idx
            if _is_sentence_end(tokenized.tokens[end_idx].text):
                last_break_idx = end_idx + 1
            end_idx += 1

        if end_idx == start_idx:
            end_idx = start_idx + 1
        if last_break_idx is not None and last_break_idx > start_idx and last_break_idx <= end_idx:
            end_idx = last_break_idx

        chunk_start = tokenized.tokens[start_idx].start
        chunk_end = tokenized.tokens[end_idx - 1].end

        # Apply overlap: shift the chunk start backward for non-first chunks
        if not is_first_chunk and overlap_chars > 0:
            overlap_start = max(0, chunk_start - overlap_chars)
            chunk_start = overlap_start

        yield TextChunk(text=text[chunk_start:chunk_end], start=chunk_start, end=chunk_end)
        start_idx = end_idx
        is_first_chunk = False
