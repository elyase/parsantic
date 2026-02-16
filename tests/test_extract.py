"""Consolidated extraction pipeline tests.

Covers: smoke tests, async iteration, chunking, batching, parallelism,
prompt rendering, local repair, debug info, and prompt validation.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from parsantic.extract import (
    AlignmentStatus,
    ChunkDebug,
    Document,
    Example,
    ExtractDebug,
    ExtractOptions,
    StaticProvider,
    extract,
    extract_aiter,
    extract_iter,
)
from parsantic.extract.chunking import iter_chunks
from parsantic.extract.formatting import FormatHandler, FormatOptions
from parsantic.extract.pipeline import _render_prompt, aextract
from parsantic.extract.prompt import Prompt

# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------


class Resume(BaseModel):
    name: str
    email: str | None = None


class Items(BaseModel):
    items: list[str]


class Person(BaseModel):
    name: str
    age: int | None = None


class NameOnly(BaseModel):
    name: str


class EventRecord(BaseModel):
    happened_at: datetime


# ---------------------------------------------------------------------------
# Helper providers
# ---------------------------------------------------------------------------


@dataclass
class FakeProvider:
    def infer(self, batch_prompts):
        return ['{"name": "Ada Lovelace", "email": "ada@example.com"}' for _ in batch_prompts]


@dataclass
class EchoProvider:
    def infer(self, batch_prompts):
        outputs = []
        for prompt in batch_prompts:
            if "Alpha" in prompt:
                outputs.append('{"items": ["Alpha"]}')
            else:
                outputs.append('{"items": ["Beta"]}')
        return outputs


@dataclass
class ChunkAwareProvider:
    """Returns different JSON depending on what text appears in the prompt."""

    mappings: list[tuple[str, str]]
    fallback: str = '{"items": []}'

    def infer(self, batch_prompts: Sequence[str]) -> Sequence[str]:
        results: list[str] = []
        for prompt in batch_prompts:
            matched = False
            for substr, response in self.mappings:
                if substr in prompt:
                    results.append(response)
                    matched = True
                    break
            if not matched:
                results.append(self.fallback)
        return results


@dataclass
class BatchRecordingProvider:
    """Records each infer() call's batch size, returns static output."""

    output: str = '{"name": "Ada", "email": "ada@example.com"}'
    batch_sizes: list[int] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str]) -> Sequence[str]:
        self.batch_sizes.append(len(batch_prompts))
        return [self.output for _ in batch_prompts]


@dataclass
class ThreadRecordingProvider:
    """Records thread IDs for each infer() call to verify parallelism."""

    output: str = '{"name": "Ada", "email": "ada@example.com"}'
    thread_ids: list[int] = field(default_factory=list)

    def infer(self, batch_prompts: Sequence[str]) -> Sequence[str]:
        self.thread_ids.append(threading.current_thread().ident)
        return [self.output for _ in batch_prompts]


# ===========================================================================
# Smoke tests (from test_extract_smoke.py)
# ===========================================================================


def test_extract_smoke():
    provider = FakeProvider()
    result = extract(
        "Ada Lovelace <ada@example.com>",
        Resume,
        model=provider,
    )
    assert result.value.name == "Ada Lovelace"
    assert result.value.email == "ada@example.com"
    by_path = {ev.path: ev for ev in result.evidence}
    assert "/name" in by_path
    assert "/email" in by_path
    assert by_path["/name"].alignment_status == AlignmentStatus.MATCH_EXACT
    assert by_path["/email"].alignment_status == AlignmentStatus.MATCH_EXACT


def test_chunking_and_multipass_merge():
    provider = EchoProvider()
    options = ExtractOptions(max_char_buffer=7, passes=2)
    result = extract("Alpha.\nBeta.", Items, model=provider, options=options)
    assert result.value.items == ["Alpha", "Beta"]


def test_streaming_iter_order():
    provider = FakeProvider()
    docs = [
        Document(text="Ada Lovelace <ada@example.com>", document_id="doc1"),
        Document(text="Ada Lovelace <ada@example.com>", document_id="doc2"),
    ]
    results = list(extract_iter(docs, Resume, model=provider))
    assert [r.document_id for r in results] == ["doc1", "doc2"]


# ===========================================================================
# Async iteration (from test_extract_aiter.py)
# ===========================================================================


def test_extract_aiter_yields_results():
    provider = StaticProvider(outputs=['{"name": "Ada Lovelace", "email": "ada@example.com"}'])

    async def _run():
        results = []
        async for r in extract_aiter(
            "Ada Lovelace <ada@example.com>",
            Resume,
            model=provider,
        ):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert len(results) == 1
    assert results[0].value.name == "Ada Lovelace"
    assert results[0].value.email == "ada@example.com"


def test_extract_aiter_order_multiple_documents():
    provider = StaticProvider(outputs=['{"name": "Ada Lovelace", "email": "ada@example.com"}'])
    docs = [
        Document(text="Ada Lovelace <ada@example.com>", document_id="doc1"),
        Document(text="Ada Lovelace <ada@example.com>", document_id="doc2"),
        Document(text="Ada Lovelace <ada@example.com>", document_id="doc3"),
    ]

    async def _run():
        results = []
        async for r in extract_aiter(docs, Resume, model=provider):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert [r.document_id for r in results] == ["doc1", "doc2", "doc3"]


def test_extract_aiter_multi_chunk_merge():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7)

    async def _run():
        results = []
        async for r in extract_aiter(
            "Alpha.\nBeta.",
            Items,
            model=provider,
            options=options,
        ):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert len(results) == 1
    assert "Alpha" in results[0].value.items
    assert "Beta" in results[0].value.items


def test_extract_aiter_flags_accumulated_across_chunks():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7)

    async def _run():
        results = []
        async for r in extract_aiter(
            "Alpha.\nBeta.",
            Items,
            model=provider,
            options=options,
        ):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert len(results) == 1
    result = results[0]
    assert isinstance(result.flags, tuple)
    assert result.flags == tuple(sorted(result.flags))
    assert isinstance(result.score, int)
    assert result.score >= 0


def test_extract_iter_flags_accumulated_across_chunks():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7)
    result = extract("Alpha.\nBeta.", Items, model=provider, options=options)
    assert isinstance(result.flags, tuple)
    assert result.flags == tuple(sorted(result.flags))
    assert isinstance(result.score, int)
    assert result.score >= 0


def test_aextract_convenience_function():
    provider = StaticProvider(outputs=['{"name": "Grace Hopper", "email": "grace@example.com"}'])

    async def _run():
        return await aextract(
            "Grace Hopper <grace@example.com>",
            Resume,
            model=provider,
        )

    result = asyncio.run(_run())
    assert result.value.name == "Grace Hopper"
    assert result.value.email == "grace@example.com"


def test_extract_aiter_with_asyncio_to_thread_fallback():
    provider = StaticProvider(outputs=['{"name": "Alan Turing", "email": null}'])

    async def _run():
        results = []
        async for r in extract_aiter(
            "Alan Turing",
            Resume,
            model=provider,
        ):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert len(results) == 1
    assert results[0].value.name == "Alan Turing"


# ===========================================================================
# Chunk overlap (from test_extract_upgrades.py — E2)
# ===========================================================================


def test_chunk_overlap_basic():
    text = "Hello world. Goodbye world."
    chunks_no_overlap = list(iter_chunks(text, max_char_buffer=15, overlap_chars=0))
    chunks_with_overlap = list(iter_chunks(text, max_char_buffer=15, overlap_chars=10))
    assert len(chunks_no_overlap) >= 2
    assert len(chunks_with_overlap) >= 2
    assert chunks_no_overlap[0].start == chunks_with_overlap[0].start
    if len(chunks_no_overlap) >= 2 and len(chunks_with_overlap) >= 2:
        assert chunks_with_overlap[1].start < chunks_no_overlap[1].start


def test_chunk_overlap_first_chunk_unaffected():
    text = "AAAA. BBBB. CCCC."
    chunks = list(iter_chunks(text, max_char_buffer=8, overlap_chars=5))
    assert chunks[0].start == 0


def test_chunk_overlap_chars_parameter():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = list(iter_chunks(text, max_char_buffer=25, overlap_chars=10))
    if len(chunks) >= 2:
        chunks_no_overlap = list(iter_chunks(text, max_char_buffer=25, overlap_chars=0))
        if len(chunks_no_overlap) >= 2:
            diff = chunks_no_overlap[1].start - chunks[1].start
            assert diff > 0
            assert diff <= 10


def test_chunk_overlap_zero_is_default():
    text = "Hello world. Goodbye world."
    chunks_default = list(iter_chunks(text, max_char_buffer=15))
    chunks_zero = list(iter_chunks(text, max_char_buffer=15, overlap_chars=0))
    assert len(chunks_default) == len(chunks_zero)
    for a, b in zip(chunks_default, chunks_zero, strict=True):
        assert a.start == b.start
        assert a.end == b.end
        assert a.text == b.text


# ===========================================================================
# Batch inference (from test_extract_upgrades.py — E3)
# ===========================================================================


def test_batch_length_controls_batch_sizes():
    provider = BatchRecordingProvider(output='{"items": ["x"]}')
    text = "Alpha.\nBeta.\nGamma.\nDelta."
    options = ExtractOptions(max_char_buffer=8, batch_length=2, max_workers=1)
    extract(text, Items, model=provider, options=options)
    for batch_size in provider.batch_sizes:
        assert batch_size <= 2


def test_batch_length_single():
    provider = BatchRecordingProvider(output='{"items": ["x"]}')
    text = "Alpha.\nBeta.\nGamma."
    options = ExtractOptions(max_char_buffer=8, batch_length=1, max_workers=1)
    extract(text, Items, model=provider, options=options)
    for batch_size in provider.batch_sizes:
        assert batch_size == 1


def test_max_workers_greater_than_one():
    provider = ThreadRecordingProvider()
    text = "Alpha.\nBeta.\nGamma.\nDelta."
    options = ExtractOptions(max_char_buffer=8, batch_length=1, max_workers=4)
    result = extract(text, Resume, model=provider, options=options)
    assert len(provider.thread_ids) >= 1
    assert result.value.name == "Ada"


# ===========================================================================
# Prompt rendering (from test_extract_upgrades.py — E5)
# ===========================================================================


def test_prompt_rendering_json_format_instructions():
    prompt = Prompt(description="Extract data.")
    format_handler = FormatHandler(FormatOptions(format="json"))
    rendered = _render_prompt(
        prompt,
        schema_text=None,
        examples=[],
        question="Some text",
        format_handler=format_handler,
        additional_context=None,
    )
    assert "Output a single JSON object" in rendered
    assert "Do not include any surrounding prose or commentary" in rendered


def test_prompt_rendering_wrapper_key_instruction():
    prompt = Prompt(description="Extract data.")
    format_handler = FormatHandler(FormatOptions(format="json", wrapper_key="extractions"))
    rendered = _render_prompt(
        prompt,
        schema_text=None,
        examples=[],
        question="Some text",
        format_handler=format_handler,
        additional_context=None,
    )
    assert '"extractions"' in rendered
    assert "Wrap the result list" in rendered


def test_prompt_rendering_no_wrapper_key():
    prompt = Prompt(description="Extract data.")
    format_handler = FormatHandler(FormatOptions(format="json", wrapper_key=None))
    rendered = _render_prompt(
        prompt,
        schema_text=None,
        examples=[],
        question="Some text",
        format_handler=format_handler,
        additional_context=None,
    )
    assert "Wrap the result list" not in rendered


def test_prompt_rendering_json_array_format_instructions():
    prompt = Prompt(description="Extract data.")
    format_handler = FormatHandler(FormatOptions(format="json"))
    rendered = _render_prompt(
        prompt,
        schema_text=None,
        examples=[],
        question="Some text",
        format_handler=format_handler,
        additional_context=None,
        output_kind="array",
    )
    assert "Output a single JSON array" in rendered


def test_extract_prompt_uses_array_instruction_for_list_targets():
    provider = StaticProvider(outputs=['[{"name": "Ada", "email": "ada@example.com"}]'])
    result = extract("Ada", list[Resume], model=provider, debug=True)
    assert result.debug is not None
    assert result.debug.rendered_prompt_preview is not None
    assert "Output a single JSON array" in result.debug.rendered_prompt_preview


# ===========================================================================
# Local repair (from test_extract_upgrades.py — E6)
# ===========================================================================


def test_repair_none_is_default():
    provider = StaticProvider(outputs=['{"name": "Ada Lovelace", "email": "ada@example.com"}'])
    options = ExtractOptions(repair="none")
    result = extract("Ada", Resume, model=provider, options=options)
    assert result.value.name == "Ada Lovelace"


def test_repair_local_recovers_clean_output():
    provider = StaticProvider(outputs=['{"name": "Ada Lovelace", "email": "ada@example.com"}'])
    options = ExtractOptions(repair="local")
    result = extract("Ada", Resume, model=provider, options=options)
    assert result.value.name == "Ada Lovelace"


def test_repair_local_handles_slightly_malformed():
    provider = StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com",}'])
    options = ExtractOptions(repair="local")
    result = extract("Ada", Resume, model=provider, options=options)
    assert result.value.name == "Ada"


def test_repair_local_with_markdown_fenced_output():
    provider = StaticProvider(outputs=['```json\n{"name": "Ada", "email": "ada@example.com"}\n```'])
    options = ExtractOptions(repair="local")
    result = extract("Ada", Resume, model=provider, options=options)
    assert result.value.name == "Ada"


def test_repair_local_preserves_existing_coerce_options():
    provider = StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com",}'])
    options = ExtractOptions(repair="local")
    result = extract("Ada", Resume, model=provider, options=options)
    assert result.value.name == "Ada"


# ===========================================================================
# Debug info (from test_extract_upgrades.py — E7)
# ===========================================================================


def test_debug_info_populated():
    provider = StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}'])
    result = extract("Ada Lovelace", Resume, model=provider, debug=True)
    assert result.debug is not None
    assert isinstance(result.debug, ExtractDebug)
    assert len(result.debug.raw_outputs) > 0


def test_debug_rendered_prompt_preview():
    provider = StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}'])
    result = extract("Ada Lovelace", Resume, model=provider, debug=True)
    assert result.debug is not None
    assert result.debug.rendered_prompt_preview is not None
    assert len(result.debug.rendered_prompt_preview) <= 500
    assert (
        "Ada Lovelace" in result.debug.rendered_prompt_preview
        or "Extract" in result.debug.rendered_prompt_preview
    )


def test_chunk_debug_populated_multi_chunk():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7)
    result = extract("Alpha.\nBeta.", Items, model=provider, options=options, debug=True)
    assert result.debug is not None
    assert len(result.debug.chunks) >= 2
    for chunk_debug in result.debug.chunks:
        assert isinstance(chunk_debug, ChunkDebug)
        assert isinstance(chunk_debug.chunk_index, int)
        assert isinstance(chunk_debug.chunk_text_preview, str)
        assert len(chunk_debug.chunk_text_preview) <= 100
        assert isinstance(chunk_debug.raw_output, str)
        assert isinstance(chunk_debug.flags, tuple)
        assert isinstance(chunk_debug.score, int)


def test_chunk_debug_not_populated_without_debug():
    provider = StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}'])
    result = extract("Ada", Resume, model=provider, debug=False)
    assert result.debug is None


def test_chunk_debug_via_aiter():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7)

    async def _run():
        results = []
        async for r in extract_aiter(
            "Alpha.\nBeta.",
            Items,
            model=provider,
            options=options,
            debug=True,
        ):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert len(results) == 1
    result = results[0]
    assert result.debug is not None
    assert len(result.debug.chunks) >= 2
    assert result.debug.rendered_prompt_preview is not None


# ===========================================================================
# Order preservation (from test_extract_upgrades.py)
# ===========================================================================


def test_parallel_processing_preserves_order():
    provider = StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}'])
    docs = [Document(text=f"Doc {i}", document_id=f"doc{i}") for i in range(10)]
    options = ExtractOptions(max_workers=4, batch_length=2)
    results = list(extract_iter(docs, Resume, model=provider, options=options))
    assert [r.document_id for r in results] == [f"doc{i}" for i in range(10)]


def test_parallel_chunk_processing_preserves_order():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
            ("Gamma", '{"items": ["Gamma"]}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=8, batch_length=1, max_workers=4)
    result = extract("Alpha.\nBeta.\nGamma.", Items, model=provider, options=options)
    assert "Alpha" in result.value.items
    assert "Beta" in result.value.items
    assert "Gamma" in result.value.items


# ===========================================================================
# Overlap wiring (from test_extract_upgrades.py)
# ===========================================================================


def test_overlap_chars_wired_through_options():
    provider = StaticProvider(outputs=['{"items": ["Overlap test"]}'])
    options = ExtractOptions(max_char_buffer=15, overlap_chars=5)
    result = extract(
        "First sentence here. Second sentence here.", Items, model=provider, options=options
    )
    assert result.value.items is not None


def test_unicode_tokenizer_respects_newline_boundaries():
    chunks = list(iter_chunks("Alpha\nBeta", max_char_buffer=10, tokenizer="unicode"))
    assert len(chunks) >= 2
    assert chunks[0].text == "Alpha"
    assert chunks[1].text == "Beta"


def test_merge_strategy_last_wins_for_scalar_conflicts():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"name": "Alpha"}'),
            ("Beta", '{"name": "Beta"}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7, merge_strategy="last_wins")
    result = extract("Alpha.\nBeta.", NameOnly, model=provider, options=options)
    assert result.value.name == "Beta"


def test_extract_handles_datetime_examples_without_json_crash():
    provider = StaticProvider(outputs=['{"happened_at": "2024-01-01T00:00:00Z"}'])
    prompt = Prompt(
        description="Extract the event timestamp.",
        examples=[
            Example(
                text="Happened at 2024-01-01T00:00:00Z",
                output={"happened_at": datetime(2024, 1, 1, tzinfo=UTC)},
            )
        ],
    )
    result = extract("Happened at 2024-01-01T00:00:00Z", EventRecord, model=provider, prompt=prompt)
    assert result.value.happened_at == datetime(2024, 1, 1, tzinfo=UTC)


# ===========================================================================
# Prompt validation (from test_prompt_validation_extract.py)
# ===========================================================================


@dataclass
class _PromptValidationProvider:
    def infer(self, batch_prompts):
        return ['{"name": "Alice"}' for _ in batch_prompts]


def test_prompt_validation_error():
    from parsantic.extract import Example, PromptValidationLevel
    from parsantic.extract import ExtractOptions as EO

    provider = _PromptValidationProvider()
    prompt = Prompt(
        description="Extract name.",
        examples=[Example(text="Bob is here", output={"name": "Alice"})],
    )
    options = EO(prompt_validation=PromptValidationLevel.ERROR)
    with pytest.raises(ValueError):
        extract("Alice is here", Person, model=provider, prompt=prompt, options=options)
