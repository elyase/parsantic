from __future__ import annotations

from parsantic.extract.alignment import (
    AlignmentOptions,
    Resolver,
    TokenAlignmentResolver,
    align_value_to_text,
    get_resolver,
)
from parsantic.extract.types import AlignmentStatus, FieldEvidence


def test_token_alignment_resolver_matches_align_value_to_text_exact():
    source = "Name: Ada Lovelace"
    path = "/name"
    value = "Ada Lovelace"

    resolver = TokenAlignmentResolver()
    expected = align_value_to_text(source, path, value)
    actual = resolver.resolve(source, path, value)

    assert actual == expected
    assert actual.alignment_status == AlignmentStatus.MATCH_EXACT


def test_token_alignment_resolver_matches_align_value_to_text_fuzzy():
    source = "Ada Lovelace wrote analytical engine notes."
    path = "/summary"
    value = "Ada Lovelace wrote analytical engin notes."
    options = AlignmentOptions(accept_match_lesser=False, fuzzy_threshold=0.8)

    resolver = TokenAlignmentResolver(options=options)
    expected = align_value_to_text(source, path, value, options=options)
    actual = resolver.resolve(source, path, value)

    assert actual == expected
    assert actual.alignment_status == AlignmentStatus.MATCH_FUZZY


def test_token_alignment_resolver_matches_align_value_to_text_unmatched():
    source = "Ada Lovelace wrote analytical engine notes."
    path = "/person"
    value = "Charles Babbage built a machine."
    options = AlignmentOptions(enable_fuzzy_alignment=False, accept_match_lesser=False)

    resolver = TokenAlignmentResolver(options=options)
    expected = align_value_to_text(source, path, value, options=options)
    actual = resolver.resolve(source, path, value)

    assert actual == expected
    assert actual.alignment_status == AlignmentStatus.UNMATCHED


class CustomResolver:
    def resolve(
        self,
        source_text: str,
        path: str,
        value: str,
        *,
        tokenized_source=None,
    ) -> FieldEvidence:
        return FieldEvidence(
            path=path,
            value_preview=value,
            char_interval=(0, 1),
            token_interval=(0, 1),
            alignment_status=AlignmentStatus.MATCH_EXACT,
        )


def test_custom_resolver_implements_protocol_and_is_used_when_passed():
    custom = CustomResolver()
    assert isinstance(custom, Resolver)

    resolved = get_resolver(custom)
    assert resolved is custom

    evidence = resolved.resolve("source", "/field", "value")
    assert evidence.path == "/field"
    assert evidence.value_preview == "value"
    assert evidence.alignment_status == AlignmentStatus.MATCH_EXACT


def test_get_resolver_creates_default_token_alignment_resolver():
    resolved = get_resolver()

    assert isinstance(resolved, TokenAlignmentResolver)
    assert resolved.options == AlignmentOptions()
    assert resolved.tokenizer is None


def test_get_resolver_passes_options_and_tokenizer_through():
    options = AlignmentOptions(enable_fuzzy_alignment=False, fuzzy_threshold=0.95)
    resolved = get_resolver(options=options, tokenizer="regex")

    assert isinstance(resolved, TokenAlignmentResolver)
    assert resolved.options is options
    assert resolved.tokenizer == "regex"


def test_align_value_to_text_backward_compatible():
    evidence = align_value_to_text("Ada Lovelace", "/name", "Ada Lovelace")

    assert evidence.path == "/name"
    assert evidence.value_preview == "Ada Lovelace"
    assert evidence.alignment_status == AlignmentStatus.MATCH_EXACT
    assert evidence.char_interval is not None


def test_alignment_options_backward_compatible():
    options = AlignmentOptions(enable_fuzzy_alignment=False, fuzzy_threshold=0.9)

    assert options.enable_fuzzy_alignment is False
    assert options.fuzzy_threshold == 0.9
    assert options.accept_match_lesser is True
