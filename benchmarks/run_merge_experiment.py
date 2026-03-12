#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["parsantic[ai,vision]"]
# ///
"""Experiment: evaluate merge fix alternatives for per-page PDF extraction.

Tests each proposed fix against scanned/mixed PDFs from both corpora.
Uses native Gemini API (gemini-2.5-flash-lite) to avoid gateway image issues.

Usage:
    source ~/.env  # GEMINI_API_KEY
    uv run python benchmarks/run_merge_experiment.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, create_model  # noqa: E402

from parsantic.extract import Document, ExtractOptions, aextract  # noqa: E402
from parsantic.extract.options import MediaOptions  # noqa: E402

# Load env
_ENV_FILE = Path.home() / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


# ---------------------------------------------------------------------------
# Benchmark scoring (inline to keep self-contained)
# ---------------------------------------------------------------------------
from benchmarks.metrics import score_case  # noqa: E402

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
CORPORA_ROOT = ROOT / "benchmarks" / "corpus"


@dataclass
class TestCase:
    name: str
    corpus: str
    pdf_path: Path
    schema_cls: type[BaseModel]
    expected: dict[str, Any]
    prompt: str


def _load_cases() -> list[TestCase]:
    """Load scanned + mixed cases from both corpora."""
    import importlib

    cases: list[TestCase] = []
    for corpus_name, schema_module, schema_class, truth_file, prompt in [
        (
            "oncology",
            "benchmarks.corpus.oncology.snapshot_schema",
            "OncologySnapshot",
            "snapshot_truth.json",
            "Extract the oncology snapshot into the flat schema. Use exact values from the document.",
        ),
        (
            "nasal_melanoma",
            "benchmarks.corpus.nasal_melanoma.snapshot_schema",
            "NasalMelanomaSnapshot",
            "ground_truth.json",
            "Extract the nasal melanoma snapshot into the flat schema. Use exact values from the document.",
        ),
    ]:
        corpus_dir = CORPORA_ROOT / corpus_name
        mod = importlib.import_module(schema_module)
        cls = getattr(mod, schema_class)
        expected = json.loads((corpus_dir / truth_file).read_text())
        for variant in ("scanned", "mixed"):
            pdf_path = corpus_dir / "generated" / f"{corpus_name}_{variant}.pdf"
            if pdf_path.exists():
                cases.append(
                    TestCase(
                        name=f"{corpus_name}_{variant}",
                        corpus=corpus_name,
                        pdf_path=pdf_path,
                        schema_cls=cls,
                        expected=expected,
                        prompt=prompt,
                    )
                )
    return cases


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
MODEL_ID = "gemini:gemini-2.5-flash-lite"


def _make_provider():
    from parsantic.extract.providers.base import ProviderConfig
    from parsantic.extract.providers.factory import create_provider

    return create_provider(ProviderConfig(model_id=MODEL_ID))


# ---------------------------------------------------------------------------
# Fix implementations
# ---------------------------------------------------------------------------


def _make_optional_schema(cls: type[BaseModel]) -> type[BaseModel]:
    """Create a copy of cls with all fields Optional[T] = None."""
    field_defs: dict[str, Any] = {}
    for name, field_info in cls.model_fields.items():
        annotation = field_info.annotation
        # Wrap in Optional
        field_defs[name] = (annotation | None, None)
    return create_model(f"{cls.__name__}Optional", **field_defs)


def _patch_merge_for_expanded_null():
    """Monkey-patch _merge_branch_values to treat sentinel values as null."""
    from parsantic.extract import pipeline

    original = pipeline._merge_branch_values

    _SENTINEL_STRINGS = {
        "",
        "n/a",
        "none",
        "unknown",
        "none listed",
        "not specified",
        "not available",
        "not found",
        "<unknown>",
        "null",
    }

    def _is_sentinel(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in _SENTINEL_STRINGS:
            return True
        if isinstance(value, (int, float)) and value in (0, 0.0, -1, -1.0):
            return True
        if isinstance(value, (list, dict)) and not value:
            return True
        return False

    def _patched(
        base, incoming, *, strategy="first_wins", conflicts=None, path="/", page_index=None
    ):
        # For prefer_non_null strategy, expand null detection
        if strategy == "prefer_non_null":
            # Let the original handle dicts/lists/None
            if base is None or incoming is None:
                return original(
                    base,
                    incoming,
                    strategy=strategy,
                    conflicts=conflicts,
                    path=path,
                    page_index=page_index,
                )
            if isinstance(base, (dict, list)) or isinstance(incoming, (dict, list)):
                return original(
                    base,
                    incoming,
                    strategy=strategy,
                    conflicts=conflicts,
                    path=path,
                    page_index=page_index,
                )
            # Leaf scalar: use expanded null detection
            if base != incoming and conflicts is not None:
                from parsantic.extract.pipeline import MergeConflict

                conflicts.append(
                    MergeConflict(
                        path=path or "/",
                        existing_preview=str(base)[:80],
                        incoming_preview=str(incoming)[:80],
                        page_index=page_index,
                    )
                )
            if _is_sentinel(base) and not _is_sentinel(incoming):
                from parsantic.extract.pipeline import _collect_leaf_path_map

                return incoming, _collect_leaf_path_map(
                    incoming, source_path=path, target_path=path
                )
            if _is_sentinel(incoming) and not _is_sentinel(base):
                return base, {}
            return original(
                base,
                incoming,
                strategy=strategy,
                conflicts=conflicts,
                path=path,
                page_index=page_index,
            )
        return original(
            base, incoming, strategy=strategy, conflicts=conflicts, path=path, page_index=page_index
        )

    pipeline._merge_branch_values = _patched
    return original


def _patch_prompt_for_per_page():
    """Monkey-patch _media_chunk_question to add per-page null instructions."""
    from parsantic.extract import pipeline

    original = pipeline._media_chunk_question

    def _patched(doc, chunk):
        base = original(doc, chunk)
        page_hint = ""
        if chunk.page_index is not None:
            page_hint = (
                "\n\nIMPORTANT: This is a single page from a multi-page document. "
                "Only extract field values that are EXPLICITLY VISIBLE on this page. "
                "For any field whose value is NOT shown on this page, you MUST return null. "
                "Do NOT guess, infer, or use placeholder values like 0, 0.0, -1, "
                "empty string, 'N/A', 'Unknown', or 'None listed'."
            )
        return base + page_hint

    pipeline._media_chunk_question = _patched
    return original


def _restore_patch(module, attr, original):
    setattr(module, attr, original)


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    name: str
    description: str
    options_fn: Any  # Callable[[type[BaseModel]], tuple[ExtractOptions, type[BaseModel]]]
    setup_fn: Any = None  # Optional monkey-patch setup
    teardown_fn: Any = None  # Optional cleanup


def _baseline_options(schema_cls):
    """Baseline: force map_reduce with first_wins (the problematic default)."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="first_wins",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _single_options(schema_cls):
    """Single: bundle all pages in one request."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="single", pdf_mode="raster"),
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _prefer_non_null_options(schema_cls):
    """map_reduce + prefer_non_null merge."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="prefer_non_null",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _last_wins_options(schema_cls):
    """map_reduce + last_wins merge."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="last_wins",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _no_native_output_options(schema_cls):
    """map_reduce + prompt-only (no native structured output)."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="first_wins",
        structured_output="prompt",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _optional_schema_options(schema_cls):
    """A: Dynamic schema optionality — make all fields Optional for per-page calls."""
    optional_cls = _make_optional_schema(schema_cls)
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="prefer_non_null",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, optional_cls


def _expanded_null_options(schema_cls):
    """C: Expanded null detection in merge (sentinel values → null)."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="prefer_non_null",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _hybrid_options(schema_cls):
    """F: Hybrid mode (whole-doc + per-page with smart reconciliation)."""
    opts = ExtractOptions(
        mode="hybrid",
        document_input="native",
        page_input="image",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _prompt_fix_options(schema_cls):
    """E: Stronger per-page prompts (no code change, just prompts)."""
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="prefer_non_null",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, schema_cls


def _optional_plus_prompt_options(schema_cls):
    """A+E: Optional schema + stronger prompts."""
    optional_cls = _make_optional_schema(schema_cls)
    opts = ExtractOptions(
        media=MediaOptions(page_strategy="map_reduce", pdf_mode="raster"),
        merge_strategy="prefer_non_null",
        repair="targeted",
        max_repair_attempts=1,
        per_call_timeout_s=60,
        per_document_timeout_s=180,
    )
    return opts, optional_cls


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="baseline_map_reduce",
        description="Baseline: map_reduce + first_wins (current default for scanned)",
        options_fn=_baseline_options,
    ),
    ExperimentConfig(
        name="single_bundled",
        description="Bundle all pages in one request (avoids merge entirely)",
        options_fn=_single_options,
    ),
    ExperimentConfig(
        name="map_reduce_prefer_non_null",
        description="map_reduce + prefer_non_null merge",
        options_fn=_prefer_non_null_options,
    ),
    ExperimentConfig(
        name="map_reduce_last_wins",
        description="map_reduce + last_wins merge",
        options_fn=_last_wins_options,
    ),
    ExperimentConfig(
        name="B_no_native_output",
        description="map_reduce + prompt-only output (disable native structured output)",
        options_fn=_no_native_output_options,
    ),
    ExperimentConfig(
        name="A_optional_schema",
        description="Dynamic schema optionality + prefer_non_null",
        options_fn=_optional_schema_options,
    ),
    ExperimentConfig(
        name="C_expanded_null",
        description="Expanded null detection (sentinel values treated as null)",
        options_fn=_expanded_null_options,
        setup_fn=lambda: _patch_merge_for_expanded_null(),
        teardown_fn=lambda orig: _restore_patch(
            __import__("parsantic.extract.pipeline", fromlist=["_merge_branch_values"]),
            "_merge_branch_values",
            orig,
        ),
    ),
    ExperimentConfig(
        name="E_better_prompts",
        description="Stronger per-page null prompts + prefer_non_null",
        options_fn=_prompt_fix_options,
        setup_fn=lambda: _patch_prompt_for_per_page(),
        teardown_fn=lambda orig: _restore_patch(
            __import__("parsantic.extract.pipeline", fromlist=["_media_chunk_question"]),
            "_media_chunk_question",
            orig,
        ),
    ),
    ExperimentConfig(
        name="F_hybrid",
        description="Hybrid mode (whole-doc native + per-page images)",
        options_fn=_hybrid_options,
    ),
    ExperimentConfig(
        name="AE_optional_plus_prompt",
        description="A+E: Optional schema + stronger prompts + prefer_non_null",
        options_fn=_optional_plus_prompt_options,
        setup_fn=lambda: _patch_prompt_for_per_page(),
        teardown_fn=lambda orig: _restore_patch(
            __import__("parsantic.extract.pipeline", fromlist=["_media_chunk_question"]),
            "_media_chunk_question",
            orig,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    experiment: str
    case_name: str
    exact_accuracy: float
    fuzzy_accuracy: float
    wrong_present_rate: float
    latency_s: float
    error: str | None = None
    output: dict[str, Any] | None = None


async def run_case(
    case: TestCase,
    experiment: ExperimentConfig,
    provider: Any,
) -> CaseResult:
    """Run a single case with a specific experiment configuration."""
    opts, schema_cls = experiment.options_fn(case.schema_cls)
    doc = Document.from_pdf(
        case.pdf_path,
        text=case.prompt,
        document_id=case.name,
    )
    start = time.perf_counter()
    try:
        result = await aextract(doc, schema_cls, model=provider, options=opts)
        output = (
            result.value.model_dump(mode="json")
            if hasattr(result.value, "model_dump")
            else result.value
        )
        metrics = score_case(case.expected, output)
        latency = time.perf_counter() - start
        return CaseResult(
            experiment=experiment.name,
            case_name=case.name,
            exact_accuracy=metrics.exact_accuracy,
            fuzzy_accuracy=metrics.fuzzy_accuracy,
            wrong_present_rate=metrics.wrong_present_rate,
            latency_s=latency,
            output=output,
        )
    except Exception as exc:
        latency = time.perf_counter() - start
        return CaseResult(
            experiment=experiment.name,
            case_name=case.name,
            exact_accuracy=0.0,
            fuzzy_accuracy=0.0,
            wrong_present_rate=1.0,
            latency_s=latency,
            error=str(exc),
        )


async def main():
    cases = _load_cases()
    print(f"Loaded {len(cases)} test cases: {[c.name for c in cases]}")
    print(f"Model: {MODEL_ID}")
    print()

    all_results: list[CaseResult] = []

    for experiment in EXPERIMENTS:
        print(f"{'=' * 70}")
        print(f"Experiment: {experiment.name}")
        print(f"  {experiment.description}")
        print(f"{'=' * 70}")

        # Setup monkey-patches if needed
        orig = None
        if experiment.setup_fn:
            orig = experiment.setup_fn()

        provider = _make_provider()

        for case in cases:
            print(f"  Running {case.name}...", end=" ", flush=True)
            result = await run_case(case, experiment, provider)
            all_results.append(result)
            if result.error:
                print(f"ERROR: {result.error[:80]}")
            else:
                print(
                    f"exact={result.exact_accuracy:.1%} "
                    f"fuzzy={result.fuzzy_accuracy:.1%} "
                    f"wrong={result.wrong_present_rate:.1%} "
                    f"latency={result.latency_s:.1f}s"
                )

            # Rate limit pause between cases
            await asyncio.sleep(2)

        # Teardown monkey-patches
        if experiment.teardown_fn and orig is not None:
            experiment.teardown_fn(orig)

        # Pause between experiments
        await asyncio.sleep(5)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()

    # Group by experiment
    experiments_seen: list[str] = []
    for r in all_results:
        if r.name not in experiments_seen:
            experiments_seen.append(r.experiment)

    # Per-experiment averages
    header = f"{'Experiment':<30} {'Exact':>8} {'Fuzzy':>8} {'Wrong':>8} {'Latency':>8}"
    print(header)
    print("-" * len(header))
    for exp_name in dict.fromkeys(r.experiment for r in all_results):
        exp_results = [r for r in all_results if r.experiment == exp_name]
        avg_exact = sum(r.exact_accuracy for r in exp_results) / len(exp_results)
        avg_fuzzy = sum(r.fuzzy_accuracy for r in exp_results) / len(exp_results)
        avg_wrong = sum(r.wrong_present_rate for r in exp_results) / len(exp_results)
        avg_latency = sum(r.latency_s for r in exp_results) / len(exp_results)
        print(
            f"{exp_name:<30} {avg_exact:>7.1%} {avg_fuzzy:>7.1%} {avg_wrong:>7.1%} {avg_latency:>7.1f}s"
        )

    print()

    # Per-corpus breakdown
    for corpus in ("oncology", "nasal_melanoma"):
        print(f"\n--- {corpus} ---")
        header = f"{'Experiment':<30} {'Exact':>8} {'Fuzzy':>8} {'Wrong':>8}"
        print(header)
        print("-" * len(header))
        for exp_name in dict.fromkeys(r.experiment for r in all_results):
            exp_results = [
                r for r in all_results if r.experiment == exp_name and corpus in r.case_name
            ]
            if not exp_results:
                continue
            avg_exact = sum(r.exact_accuracy for r in exp_results) / len(exp_results)
            avg_fuzzy = sum(r.fuzzy_accuracy for r in exp_results) / len(exp_results)
            avg_wrong = sum(r.wrong_present_rate for r in exp_results) / len(exp_results)
            print(f"{exp_name:<30} {avg_exact:>7.1%} {avg_fuzzy:>7.1%} {avg_wrong:>7.1%}")

    # Save detailed results
    output_path = CORPORA_ROOT / "merge_experiment_results.json"
    payload = [
        {
            "experiment": r.experiment,
            "case": r.case_name,
            "exact_accuracy": r.exact_accuracy,
            "fuzzy_accuracy": r.fuzzy_accuracy,
            "wrong_present_rate": r.wrong_present_rate,
            "latency_s": r.latency_s,
            "error": r.error,
            "output": r.output,
        }
        for r in all_results
    ]
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
