from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass

from .tokenizer import TokenizedText, Tokenizer, get_tokenizer, tokens_lower
from .types import AlignmentStatus, FieldEvidence


@dataclass(slots=True)
class AlignmentOptions:
    enable_fuzzy_alignment: bool = True
    fuzzy_threshold: float = 0.75
    accept_match_lesser: bool = True


def _token_interval_from_chars(
    tokenized: TokenizedText, start: int, end: int
) -> tuple[int, int] | None:
    if not tokenized.tokens:
        return None
    start_idx = None
    end_idx = None
    for idx, token in enumerate(tokenized.tokens):
        if start_idx is None and token.end > start:
            start_idx = idx
        if token.start < end:
            end_idx = idx
    if start_idx is None or end_idx is None:
        return None
    return (start_idx, end_idx + 1)


def _exact_substring_align(source: str, target: str) -> tuple[int, int] | None:
    if not target:
        return None
    idx = source.find(target)
    if idx == -1:
        return None
    return (idx, idx + len(target))


def _exact_token_align(
    source_tokens: list[str], target_tokens: list[str]
) -> tuple[int, int] | None:
    if not target_tokens:
        return None
    t_len = len(target_tokens)
    for start in range(0, len(source_tokens) - t_len + 1):
        if source_tokens[start : start + t_len] == target_tokens:
            return (start, start + t_len)
    return None


def _longest_common_block(
    source_tokens: list[str], target_tokens: list[str]
) -> tuple[int, int, int] | None:
    matcher = difflib.SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    blocks = matcher.get_matching_blocks()
    if not blocks:
        return None
    best = max(blocks, key=lambda b: b.size)
    if best.size == 0:
        return None
    return (best.a, best.b, best.size)


def _fuzzy_align_window(
    source_tokens: list[str], target_tokens: list[str], threshold: float
) -> tuple[int, int, float] | None:
    if not target_tokens:
        return None
    t_len = len(target_tokens)
    best_ratio = 0.0
    best_span: tuple[int, int] | None = None
    matcher = difflib.SequenceMatcher(autojunk=False, b=target_tokens)
    for start in range(0, len(source_tokens) - t_len + 1):
        window = source_tokens[start : start + t_len]
        matcher.set_seq1(window)
        matches = sum(size for _, _, size in matcher.get_matching_blocks())
        ratio = matches / t_len if t_len else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (start, start + t_len)
    if best_span and best_ratio >= threshold:
        return (best_span[0], best_span[1], best_ratio)
    return None


def align_value_to_text(
    source_text: str,
    path: str,
    value: str,
    *,
    tokenizer: Tokenizer | None = None,
    options: AlignmentOptions | None = None,
    tokenized_source: TokenizedText | None = None,
) -> FieldEvidence:
    tok = get_tokenizer(tokenizer)
    opts = options or AlignmentOptions()
    tokenized = tokenized_source or tok.tokenize(source_text)

    if not value:
        return FieldEvidence(
            path=path,
            value_preview=value,
            char_interval=None,
            token_interval=None,
            alignment_status=AlignmentStatus.UNMATCHED,
        )

    # 1) exact substring
    exact_span = _exact_substring_align(source_text, value)
    if exact_span:
        token_interval = _token_interval_from_chars(tokenized, *exact_span)
        return FieldEvidence(
            path=path,
            value_preview=value,
            char_interval=exact_span,
            token_interval=token_interval,
            alignment_status=AlignmentStatus.MATCH_EXACT,
        )

    source_tokens = tokens_lower(tokenized)
    target_tokenized = tok.tokenize(value)
    target_tokens = [t.text.lower() for t in target_tokenized.tokens]

    # 2) exact token alignment
    exact_token_span = _exact_token_align(source_tokens, target_tokens)
    if exact_token_span:
        start_idx, end_idx = exact_token_span
        start_char = tokenized.tokens[start_idx].start
        end_char = tokenized.tokens[end_idx - 1].end
        return FieldEvidence(
            path=path,
            value_preview=value,
            char_interval=(start_char, end_char),
            token_interval=(start_idx, end_idx),
            alignment_status=AlignmentStatus.MATCH_EXACT,
        )

    # 3) lesser match (longest common block)
    if opts.accept_match_lesser:
        block = _longest_common_block(source_tokens, target_tokens)
        if block:
            start_idx, _target_idx, size = block
            end_idx = start_idx + size
            start_char = tokenized.tokens[start_idx].start
            end_char = tokenized.tokens[end_idx - 1].end
            return FieldEvidence(
                path=path,
                value_preview=value,
                char_interval=(start_char, end_char),
                token_interval=(start_idx, end_idx),
                alignment_status=AlignmentStatus.MATCH_LESSER,
            )

    # 4) fuzzy match
    if opts.enable_fuzzy_alignment:
        fuzzy = _fuzzy_align_window(source_tokens, target_tokens, opts.fuzzy_threshold)
        if fuzzy:
            start_idx, end_idx, _ratio = fuzzy
            start_char = tokenized.tokens[start_idx].start
            end_char = tokenized.tokens[end_idx - 1].end
            return FieldEvidence(
                path=path,
                value_preview=value,
                char_interval=(start_char, end_char),
                token_interval=(start_idx, end_idx),
                alignment_status=AlignmentStatus.MATCH_FUZZY,
            )

    return FieldEvidence(
        path=path,
        value_preview=value,
        char_interval=None,
        token_interval=None,
        alignment_status=AlignmentStatus.UNMATCHED,
    )


def merge_evidence(
    primary: Iterable[FieldEvidence],
    secondary: Iterable[FieldEvidence],
) -> list[FieldEvidence]:
    merged = list(primary)
    for ev in secondary:
        overlaps = False
        if ev.char_interval:
            for existing in merged:
                if not existing.char_interval:
                    continue
                s1, e1 = existing.char_interval
                s2, e2 = ev.char_interval
                if s1 < e2 and s2 < e1:
                    overlaps = True
                    break
        if not overlaps:
            merged.append(ev)
    return merged
