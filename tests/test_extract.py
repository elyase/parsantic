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
from pydantic import BaseModel, ValidationError

from parsantic.extract import (
    AlignmentStatus,
    ChunkDebug,
    Document,
    Example,
    ExtractDebug,
    ExtractOptions,
    FieldScopePolicy,
    StaticProvider,
    Strategy,
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


class LineItem(BaseModel):
    code: str
    amount: int | None = None
    description: str | None = None


class Order(BaseModel):
    line_items: list[LineItem]


class Person(BaseModel):
    name: str
    age: int | None = None


class NameOnly(BaseModel):
    name: str


class NameAge(BaseModel):
    name: str
    age: int


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
class SingleStringProvider:
    output: str

    def infer(self, batch_prompts):
        return self.output


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


@dataclass
class ConcurrentHybridTextProvider:
    """Exposes whether whole-doc and chunk hybrid branches overlap."""

    started_calls: int = 0
    ready: threading.Event = field(default_factory=threading.Event)
    overlap_detected: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def infer(self, batch_prompts: Sequence[str]) -> Sequence[str]:
        with self.lock:
            self.started_calls += 1
            if self.started_calls == 2:
                self.ready.set()

        if self.ready.wait(timeout=0.25):
            self.overlap_detected = True

        outputs: list[str] = []
        for prompt in batch_prompts:
            if "Alpha.\nBeta." in prompt:
                outputs.append('"SETTLED"')
            elif "Alpha" in prompt:
                outputs.append('"PENDING"')
            else:
                outputs.append('"APPROVED"')
        return outputs


@dataclass
class AsyncCapableTextProvider:
    output: str = '{"name": "Ada", "age": 42}'

    def infer(self, batch_prompts: Sequence[str]) -> Sequence[str]:
        return [self.output for _ in batch_prompts]

    async def ainfer(self, batch_prompts: Sequence[str]) -> Sequence[str]:
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
    assert result.sources["/name"].scope == "document"
    assert result.sources["/name"].pages == ()
    assert result.sources["/email"].scope == "document"
    assert result.sources["/email"].pages == ()


def test_extract_accepts_single_string_provider_output():
    provider = SingleStringProvider(output='{"name": "Ada Lovelace", "email": "ada@example.com"}')
    result = extract("Ada Lovelace <ada@example.com>", Resume, model=provider)
    assert result.value.name == "Ada Lovelace"
    assert result.value.email == "ada@example.com"


def test_chunking_and_multipass_merge():
    provider = EchoProvider()
    options = ExtractOptions(max_char_buffer=7, passes=2)
    result = extract("Alpha.\nBeta.", Items, model=provider, options=options)
    assert result.value.items == ["Alpha", "Beta"]


def test_chunk_merge_preserves_repeated_object_items():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"line_items": [{"code": "A", "amount": 10}]}'),
            ("Beta", '{"line_items": [{"code": "A", "amount": 10}]}'),
        ]
    )

    result = extract(
        "Alpha.\nBeta.",
        Order,
        model=provider,
        options=ExtractOptions(max_char_buffer=7),
    )

    assert result.value.line_items == [
        LineItem(code="A", amount=10, description=None),
        LineItem(code="A", amount=10, description=None),
    ]


def test_hybrid_text_supports_root_array_targets():
    provider = ChunkAwareProvider(
        mappings=[
            (
                "Alpha.\nBeta.",
                (
                    '[{"code": "A", "description": "Widget"}, '
                    '{"code": "A", "description": "Widget"}]'
                ),
            ),
            ("Alpha", '[{"code": "A", "amount": 10}]'),
            ("Beta", '[{"code": "A", "amount": 10}]'),
        ]
    )

    result = extract(
        "Alpha.\nBeta.",
        list[LineItem],
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            max_char_buffer=7,
            strategy=None,
        ),
    )

    assert result.value == [
        LineItem(code="A", amount=10, description="Widget"),
        LineItem(code="A", amount=10, description="Widget"),
    ]
    assert result.sources["/0/amount"].scope == "document"
    assert result.sources["/0/description"].scope == "document"


def test_hybrid_text_supports_root_array_field_scope_paths():
    provider = ChunkAwareProvider(
        mappings=[
            (
                "Alpha.\nBeta.",
                (
                    '[{"code": "A", "description": "Widget"}, '
                    '{"code": "A", "description": "Widget"}]'
                ),
            ),
            ("Alpha", '[{"code": "A", "amount": 10}]'),
            ("Beta", '[{"code": "A", "amount": 10}]'),
        ]
    )

    result = extract(
        "Alpha.\nBeta.",
        list[LineItem],
        model=provider,
        options=ExtractOptions(
            strategy=Strategy(
                plan="hybrid",
                field_scope=FieldScopePolicy(
                    by_path={
                        "/*/amount": "local",
                        "/*/description": "global",
                    }
                ),
            ),
            max_char_buffer=7,
        ),
    )

    assert result.value == [
        LineItem(code="A", amount=10, description="Widget"),
        LineItem(code="A", amount=10, description="Widget"),
    ]


def test_hybrid_text_root_array_sources_use_root_index_paths():
    provider = ChunkAwareProvider(
        mappings=[
            (
                "Alpha.\nBeta.",
                (
                    '[{"code": "A", "description": "Widget A"}, '
                    '{"code": "B", "description": "Widget B"}]'
                ),
            ),
            ("Alpha", '[{"code": "A", "amount": 10}]'),
            ("Beta", '[{"code": "B", "amount": 20}]'),
        ]
    )

    result = extract(
        "Alpha.\nBeta.",
        list[LineItem],
        model=provider,
        options=ExtractOptions(
            strategy=Strategy(
                plan="hybrid",
                field_scope=FieldScopePolicy(
                    by_path={
                        "/*/amount": "local",
                        "/*/description": "global",
                    }
                ),
            ),
            max_char_buffer=7,
        ),
    )

    assert result.value == [
        LineItem(code="A", amount=10, description="Widget A"),
        LineItem(code="B", amount=20, description="Widget B"),
    ]
    assert set(result.sources) >= {
        "/0/code",
        "/0/amount",
        "/0/description",
        "/1/code",
        "/1/amount",
        "/1/description",
    }
    assert result.sources["/0/amount"].scope == "document"
    assert result.sources["/1/amount"].scope == "document"
    assert result.sources["/0/description"].scope == "document"
    assert result.sources["/1/description"].scope == "document"


def test_hybrid_text_supports_scalar_root_targets():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha.\nBeta.", '"SETTLED"'),
            ("Alpha", '"PENDING"'),
            ("Beta", '"APPROVED"'),
        ],
        fallback='"PENDING"',
    )

    result = extract(
        "Alpha.\nBeta.",
        str,
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            max_char_buffer=7,
            strategy=None,
        ),
    )

    assert result.value == "SETTLED"
    assert result.sources["/"].scope == "document"


def test_hybrid_text_supports_scalar_root_scope_rules():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha.\nBeta.", '"SETTLED"'),
            ("Alpha", '"PENDING"'),
            ("Beta", '"PENDING"'),
        ],
        fallback='"PENDING"',
    )

    result = extract(
        "Alpha.\nBeta.",
        str,
        model=provider,
        options=ExtractOptions(
            strategy=Strategy(
                plan="hybrid",
                field_scope=FieldScopePolicy(by_path={"/": "global"}),
            ),
            max_char_buffer=7,
        ),
    )

    assert result.value == "SETTLED"
    assert result.sources["/"].scope == "document"


def test_hybrid_text_runs_whole_and_chunk_branches_concurrently():
    provider = ConcurrentHybridTextProvider()

    result = extract(
        "Alpha.\nBeta.",
        str,
        model=provider,
        options=ExtractOptions(
            mode="hybrid",
            max_char_buffer=7,
            max_workers=1,
        ),
    )

    assert result.value == "SETTLED"
    assert result.sources["/"].scope == "document"
    assert provider.overlap_detected is True


def test_sync_text_extraction_bypasses_threaded_timeout_for_async_capable_provider(monkeypatch):
    from parsantic.extract import pipeline as pipeline_mod

    def _unexpected_timeout(*args, **kwargs):
        raise AssertionError("sync text path should not use threaded timeout wrapper here")

    monkeypatch.setattr(pipeline_mod, "_run_sync_call_with_timeout", _unexpected_timeout)

    result = extract(
        "Ada is 42 years old",
        NameAge,
        model=AsyncCapableTextProvider(),
        options=ExtractOptions(per_call_timeout_s=1.0),
    )

    assert result.value == NameAge(name="Ada", age=42)


def test_chunk_parsing_allows_partial_then_final_validates():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"name": "Ada"}'),
            ("Beta", '{"age": 36}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7)
    result = extract("Alpha.\nBeta.", NameAge, model=provider, options=options)
    assert result.value.name == "Ada"
    assert result.value.age == 36


def test_chunk_error_skip_continues_on_bad_chunk():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", "not json"),
            ("Beta", '{"name": "Beta", "age": 42}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7, chunk_error="skip")
    result = extract("Alpha.\nBeta.", NameAge, model=provider, options=options, debug=True)
    assert result.value.name == "Beta"
    assert result.value.age == 42
    assert result.debug is not None
    assert any(chunk.error for chunk in result.debug.chunks)


def test_chunk_error_raise_fails_on_bad_chunk():
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", "not json"),
            ("Beta", '{"name": "Beta", "age": 42}'),
        ]
    )
    options = ExtractOptions(max_char_buffer=7, chunk_error="raise")
    with pytest.raises((ValidationError, ValueError)):
        extract("Alpha.\nBeta.", NameAge, model=provider, options=options)


def test_empty_extraction_defaults_to_empty_array_for_list_targets():
    provider = StaticProvider(outputs=[""])
    result = extract("No entities here.", list[str], model=provider)
    assert result.value == []


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


def test_extract_aiter_basic_and_order():
    provider = StaticProvider(outputs=['{"name": "Ada Lovelace", "email": "ada@example.com"}'])

    async def _run_single():
        return [
            r async for r in extract_aiter("Ada Lovelace <ada@example.com>", Resume, model=provider)
        ]

    results = asyncio.run(_run_single())
    assert len(results) == 1 and results[0].value.name == "Ada Lovelace"

    # multiple documents preserve order
    docs = [
        Document(text="Ada Lovelace <ada@example.com>", document_id=f"doc{i}") for i in range(1, 4)
    ]

    async def _run_multi():
        return [r async for r in extract_aiter(docs, Resume, model=provider)]

    assert [r.document_id for r in asyncio.run(_run_multi())] == ["doc1", "doc2", "doc3"]


def test_extract_aiter_multi_chunk_merge():
    provider = ChunkAwareProvider(
        mappings=[("Alpha", '{"items": ["Alpha"]}'), ("Beta", '{"items": ["Beta"]}')]
    )

    async def _run():
        return [
            r
            async for r in extract_aiter(
                "Alpha.\nBeta.", Items, model=provider, options=ExtractOptions(max_char_buffer=7)
            )
        ]

    results = asyncio.run(_run())
    assert (
        len(results) == 1 and "Alpha" in results[0].value.items and "Beta" in results[0].value.items
    )


def test_flags_accumulated_across_chunks():
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


def test_aextract_and_thread_fallback():
    async def _run():
        result = await aextract(
            "Grace Hopper <grace@example.com>",
            Resume,
            model=StaticProvider(
                outputs=['{"name": "Grace Hopper", "email": "grace@example.com"}']
            ),
        )
        assert result.value.name == "Grace Hopper" and result.value.email == "grace@example.com"
        # thread fallback
        results = [
            r
            async for r in extract_aiter(
                "Alan Turing",
                Resume,
                model=StaticProvider(outputs=['{"name": "Alan Turing", "email": null}']),
            )
        ]
        assert len(results) == 1 and results[0].value.name == "Alan Turing"

    asyncio.run(_run())


# ===========================================================================
# Chunk overlap (from test_extract_upgrades.py — E2)
# ===========================================================================


def test_chunk_overlap_basic_and_first_unaffected():
    text = "Hello world. Goodbye world."
    chunks_no = list(iter_chunks(text, max_char_buffer=15, overlap_chars=0))
    chunks_yes = list(iter_chunks(text, max_char_buffer=15, overlap_chars=10))
    assert len(chunks_no) >= 2
    assert len(chunks_yes) >= 2
    assert chunks_no[0].start == chunks_yes[0].start == 0
    if len(chunks_no) >= 2 and len(chunks_yes) >= 2:
        assert chunks_yes[1].start < chunks_no[1].start


def test_chunk_overlap_zero_is_default():
    text = "Hello world. Goodbye world."
    chunks_default = list(iter_chunks(text, max_char_buffer=15))
    chunks_zero = list(iter_chunks(text, max_char_buffer=15, overlap_chars=0))
    assert len(chunks_default) == len(chunks_zero)
    for a, b in zip(chunks_default, chunks_zero, strict=True):
        assert a.start == b.start
        assert a.end == b.end


# ===========================================================================
# Batch inference (from test_extract_upgrades.py — E3)
# ===========================================================================


def test_batch_length_controls_sizes():
    text = "Alpha.\nBeta.\nGamma.\nDelta."
    # batch_length=2
    p1 = BatchRecordingProvider(output='{"items": ["x"]}')
    extract(
        text,
        Items,
        model=p1,
        options=ExtractOptions(max_char_buffer=8, batch_length=2, max_workers=1),
    )
    assert all(bs <= 2 for bs in p1.batch_sizes)
    # batch_length=1
    p2 = BatchRecordingProvider(output='{"items": ["x"]}')
    extract(
        "Alpha.\nBeta.\nGamma.",
        Items,
        model=p2,
        options=ExtractOptions(max_char_buffer=8, batch_length=1, max_workers=1),
    )
    assert all(bs == 1 for bs in p2.batch_sizes)


def test_max_workers_sync_and_async():
    text = "Alpha.\nBeta.\nGamma.\nDelta."
    options = ExtractOptions(max_char_buffer=8, batch_length=1, max_workers=4)
    # sync
    p = ThreadRecordingProvider()
    result = extract(text, Resume, model=p, options=options)
    assert len(p.thread_ids) >= 1 and result.value.name == "Ada"

    # async
    async def _run():
        return [r async for r in extract_aiter(text, Resume, model=FakeProvider(), options=options)]

    results = asyncio.run(_run())
    assert len(results) == 1 and results[0].value.name == "Ada Lovelace"


# ===========================================================================
# Prompt rendering (from test_extract_upgrades.py — E5)
# ===========================================================================


def test_prompt_rendering_format_instructions():
    prompt = Prompt(description="Extract data.")

    def _render(**kwargs):
        defaults = dict(
            prompt=prompt,
            schema_text=None,
            examples=[],
            question="Some text",
            format_handler=FormatHandler(FormatOptions(format="json")),
            additional_context=None,
        )
        defaults.update(kwargs)
        return _render_prompt(defaults.pop("prompt"), **defaults)

    # JSON object format
    rendered = _render()
    assert (
        "Output a single JSON object" in rendered
        and "Do not include any surrounding prose" in rendered
    )
    # wrapper key
    assert '"extractions"' in _render(
        format_handler=FormatHandler(FormatOptions(format="json", wrapper_key="extractions"))
    )
    assert "Wrap the result list" not in _render(
        format_handler=FormatHandler(FormatOptions(format="json", wrapper_key=None))
    )
    # JSON array format
    assert "Output a single JSON array" in _render(output_kind="array")


def test_extract_prompt_uses_array_instruction_for_list_targets():
    result = extract(
        "Ada",
        list[Resume],
        model=StaticProvider(outputs=['[{"name": "Ada", "email": "ada@example.com"}]']),
        debug=True,
    )
    assert "Output a single JSON array" in result.debug.rendered_prompt_preview


def test_render_prompt_uses_scalar_instruction_for_scalar_targets():
    rendered = _render_prompt(
        Prompt(description="Extract."),
        schema_text=None,
        examples=[],
        question="Alpha",
        format_handler=FormatHandler(FormatOptions(format="json")),
        additional_context=None,
        output_kind="scalar",
    )

    assert "Output a single JSON value" in rendered


# ===========================================================================
# Local repair (from test_extract_upgrades.py — E6)
# ===========================================================================


@pytest.mark.parametrize(
    "output,repair_mode",
    [
        ('{"name": "Ada Lovelace", "email": "ada@example.com"}', "none"),
        ('{"name": "Ada Lovelace", "email": "ada@example.com"}', "local"),
        ('{"name": "Ada", "email": "ada@example.com",}', "local"),
        ('```json\n{"name": "Ada", "email": "ada@example.com"}\n```', "local"),
    ],
    ids=["none-clean", "local-clean", "local-trailing-comma", "local-markdown"],
)
def test_repair_modes(output, repair_mode):
    provider = StaticProvider(outputs=[output])
    options = ExtractOptions(repair=repair_mode)
    result = extract("Ada", Resume, model=provider, options=options)
    assert result.value.name is not None


# ===========================================================================
# Debug info (from test_extract_upgrades.py — E7)
# ===========================================================================


def test_debug_info_populated():
    result = extract(
        "Ada Lovelace",
        Resume,
        model=StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}']),
        debug=True,
    )
    assert isinstance(result.debug, ExtractDebug)
    assert len(result.debug.raw_outputs) > 0
    assert (
        result.debug.rendered_prompt_preview is not None
        and len(result.debug.rendered_prompt_preview) <= 500
    )
    # debug=False
    assert (
        extract(
            "Ada",
            Resume,
            model=StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}']),
            debug=False,
        ).debug
        is None
    )


def test_chunk_debug_sync_and_async():
    provider = ChunkAwareProvider(
        mappings=[("Alpha", '{"items": ["Alpha"]}'), ("Beta", '{"items": ["Beta"]}')]
    )
    options = ExtractOptions(max_char_buffer=7)
    # sync
    result = extract("Alpha.\nBeta.", Items, model=provider, options=options, debug=True)
    assert len(result.debug.chunks) >= 2
    for chunk_debug in result.debug.chunks:
        assert isinstance(chunk_debug, ChunkDebug)
        assert (
            isinstance(chunk_debug.chunk_index, int) and len(chunk_debug.chunk_text_preview) <= 100
        )

    # async
    async def _run():
        results = []
        async for r in extract_aiter(
            "Alpha.\nBeta.", Items, model=provider, options=options, debug=True
        ):
            results.append(r)
        return results

    results = asyncio.run(_run())
    assert len(results) == 1 and len(results[0].debug.chunks) >= 2


# ===========================================================================
# Order preservation (from test_extract_upgrades.py)
# ===========================================================================


def test_parallel_processing_preserves_order():
    # multi-document order
    docs = [Document(text=f"Doc {i}", document_id=f"doc{i}") for i in range(10)]
    results = list(
        extract_iter(
            docs,
            Resume,
            model=StaticProvider(outputs=['{"name": "Ada", "email": "ada@example.com"}']),
            options=ExtractOptions(max_workers=4, batch_length=2),
        )
    )
    assert [r.document_id for r in results] == [f"doc{i}" for i in range(10)]
    # multi-chunk order
    provider = ChunkAwareProvider(
        mappings=[
            ("Alpha", '{"items": ["Alpha"]}'),
            ("Beta", '{"items": ["Beta"]}'),
            ("Gamma", '{"items": ["Gamma"]}'),
        ]
    )
    result = extract(
        "Alpha.\nBeta.\nGamma.",
        Items,
        model=provider,
        options=ExtractOptions(max_char_buffer=8, batch_length=1, max_workers=4),
    )
    assert all(x in result.value.items for x in ["Alpha", "Beta", "Gamma"])


# ===========================================================================
# Overlap wiring (from test_extract_upgrades.py)
# ===========================================================================


def test_overlap_and_tokenizer():
    # overlap wired through options
    result = extract(
        "First sentence here. Second sentence here.",
        Items,
        model=StaticProvider(outputs=['{"items": ["Overlap test"]}']),
        options=ExtractOptions(max_char_buffer=15, overlap_chars=5),
    )
    assert result.value.items is not None
    # unicode tokenizer
    chunks = list(iter_chunks("Alpha\nBeta", max_char_buffer=10, tokenizer="unicode"))
    assert len(chunks) >= 2 and chunks[0].text == "Alpha" and chunks[1].text == "Beta"
    # unknown tokenizer
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        list(iter_chunks("Alpha Beta", max_char_buffer=10, tokenizer="unknown"))  # type: ignore[arg-type]


def test_merge_strategy_and_datetime_examples():
    # last_wins
    provider = ChunkAwareProvider(
        mappings=[("Alpha", '{"name": "Alpha"}'), ("Beta", '{"name": "Beta"}')]
    )
    assert (
        extract(
            "Alpha.\nBeta.",
            NameOnly,
            model=provider,
            options=ExtractOptions(max_char_buffer=7, merge_strategy="last_wins"),
        ).value.name
        == "Beta"
    )
    # datetime examples
    prompt = Prompt(
        description="Extract timestamp.",
        examples=[
            Example(
                text="Happened at 2024-01-01T00:00:00Z",
                output={"happened_at": datetime(2024, 1, 1, tzinfo=UTC)},
            )
        ],
    )
    result = extract(
        "Happened at 2024-01-01T00:00:00Z",
        EventRecord,
        model=StaticProvider(outputs=['{"happened_at": "2024-01-01T00:00:00Z"}']),
        prompt=prompt,
    )
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
