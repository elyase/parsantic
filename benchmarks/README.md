# Benchmarks

The benchmark harness tracks:

- Field-level exact accuracy
- Field-level fuzzy accuracy
- Schema completeness
- Latency
- Token count
- API call count

Default maintained benchmarks:

- Oncology strategy benchmark
- Nasal melanoma model benchmark

Default oncology benchmark:

```bash
uv run python benchmarks/run_oncology_default.py
```

This default benchmark uses:
- Model: `gemini:gemini-2.5-flash-lite`
- Real generated oncology PDFs under [`benchmarks/corpus/oncology`](/Users/yaser/parsantic/benchmarks/corpus/oncology)
- Snapshot schema optimized for repeatable strategy comparisons

Artifacts:
- Default manifest: [`manifest.default.json`](/Users/yaser/parsantic/benchmarks/corpus/oncology/manifest.default.json)
- Latest saved results: [`RESULTS.default.md`](/Users/yaser/parsantic/benchmarks/corpus/oncology/RESULTS.default.md)

Default nasal melanoma model sweep:

```bash
uv run python benchmarks/run_nasal_melanoma_models.py
```

Artifacts:
- Corpus: [`benchmarks/corpus/nasal_melanoma`](/Users/yaser/parsantic/benchmarks/corpus/nasal_melanoma)
- Latest saved results: [`RESULTS.models.md`](/Users/yaser/parsantic/benchmarks/corpus/nasal_melanoma/RESULTS.models.md)

## Default Rationale

The library defaults are intentionally conservative:

- Default model: `gemini:gemini-2.5-flash-lite`
- Default repair mode: `targeted`
- Default extraction strategy: automatic whole-document path (`document_auto` behavior)
- `max_repair_attempts`: `2`

Why:

- On the oncology strategy benchmark, `hybrid_targeted` had the best quality, but `document_auto` had nearly the same quality with much lower latency.
- On the more realistic nasal melanoma benchmark, `document_auto` was the most stable strategy family. `hybrid_targeted` hit timeouts and runtime bugs, and `fused_targeted` was materially worse.
- On the nasal melanoma model sweep, `gemini:gemini-2.5-flash-lite` matched `gemini:gemini-3.1-flash-lite-preview` on quality and was faster, while `gemini:gemini-2.5-flash` was somewhat more accurate but much slower.

That combination makes `gemini:gemini-2.5-flash-lite` plus targeted repair the safest default, while leaving the strategy on the stable document-first path.

## Strategy Guide

`document_auto`
- The default behavior.
- If the PDF has a text layer, extract from text first.
- Otherwise fall back to media/image processing.
- Best speed and best stability in the current benchmarks.

`hybrid_targeted`
- Whole-document extraction plus page-level extraction, then merge, then targeted repair.
- Best quality on the oncology strategy benchmark.
- Not the default because it is slower and currently less stable on heavier documents.

`fused_targeted`
- Page-by-page extraction with page image plus page-local text in one prompt, then targeted repair.
- Promising for some layouts, but underperformed on important local fields in the current benchmarks.
- Not recommended as the default.

When to use each:

- Use `document_auto` when you want the best default tradeoff.
- Use `hybrid_targeted` when you care most about quality and can afford more latency.
- Use `fused_targeted` only when you are explicitly testing page-fused behavior.

For custom runs:

```bash
uv run python -m benchmarks.run_benchmarks path/to/manifest.json
```
