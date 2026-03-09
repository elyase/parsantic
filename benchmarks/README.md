# Benchmarks

This folder keeps two maintained benchmarks:

- `oncology`: strategy benchmark
- `nasal_melanoma`: model and strategy benchmark

Default recommendation:

- Default model: `gemini:gemini-2.5-flash-lite`
- Default strategy: `document_auto`
- Default repair mode: `targeted`

Why:
- `document_auto` is the most stable strategy and usually the fastest good option.
- `hybrid_targeted` can improve quality, but it is slower and not the best default tradeoff.
- `fused_targeted` was consistently weaker in the current benchmarks.
- `gemini:gemini-2.5-flash-lite` matched the main alternative lite Gemini on quality and was faster.

Benchmarks in plain terms:
- `oncology`: 4 generated PDFs, about 4-5 pages each, small oncology snapshot.
- `nasal_melanoma`: 4 generated PDFs, about 2-3 pages each, more realistic imaging/surgery/pathology note.

What the strategy names mean:
- `document_auto`: whole-document extraction first; uses text layer when available, otherwise media.
- `hybrid_targeted`: whole-document pass plus page-level pass, merge, then validation-guided repair.
- `fused_targeted`: page-level extraction with page image plus page-local text together, then repair.

What the metrics mean:
- `exact`: fields that match ground truth exactly.
- `fuzzy`: fields that are acceptably close after normalization.
- `completeness`: expected fields that were present at all.
- `total latency`: wall-clock time for the whole benchmark slice.

### Oncology Strategy Benchmark

Model used: `gemini:gemini-2.5-flash-lite`

| Strategy | Exact | Fuzzy | Completeness | Total latency |
| --- | ---: | ---: | ---: | ---: |
| `document_auto` | `0.639` | `0.667` | `1.000` | `7.81s` |
| `hybrid_targeted` | `0.917` | `0.917` | `1.000` | `66.67s` |
| `fused_targeted` | `0.583` | `0.583` | `1.000` | `50.11s` |

Takeaway:
- Best quality: `hybrid_targeted`
- Best default tradeoff: `document_auto`

### Nasal Melanoma Model Benchmark

Strategy used: `document_auto`

| Model | Exact | Fuzzy | Completeness | Total latency | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemini:gemini-3.1-flash-lite-preview` | `0.823` | `0.823` | `0.976` | `19.71s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `0.823` | `0.823` | `0.976` | `16.25s` | succeeded |
| `gemini:gemini-2.5-flash` | `0.815` | `0.823` | `0.976` | `34.91s` | succeeded |
| `gemini:gemini-3-flash` | `0.000` | `0.000` | `0.000` | `9.96s` | unsupported (`404`) |

Takeaway:
- `gemini-2.5-flash-lite` and `gemini-3.1-flash-lite-preview` were tied on quality here.
- `gemini-2.5-flash-lite` was faster, so it is the better default.

### Nasal Melanoma Strategy Matrix

| Model | Strategy | Exact | Fuzzy | Completeness | Total latency | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `gemini:gemini-3.1-flash-lite-preview` | `document_auto` | `0.823` | `0.823` | `0.976` | `21.00s` | succeeded |
| `gemini:gemini-3.1-flash-lite-preview` | `hybrid_targeted` | `0.806` | `0.806` | `1.000` | `56.81s` | succeeded |
| `gemini:gemini-3.1-flash-lite-preview` | `fused_targeted` | `0.500` | `0.516` | `1.000` | `33.75s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `document_auto` | `0.831` | `0.831` | `0.976` | `16.17s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `hybrid_targeted` | `0.815` | `0.815` | `1.000` | `49.90s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `fused_targeted` | `0.484` | `0.492` | `1.000` | `26.23s` | succeeded |
| `gemini:gemini-2.5-flash` | `document_auto` | `0.815` | `0.823` | `0.976` | `38.00s` | succeeded |
| `gemini:gemini-2.5-flash` | `hybrid_targeted` | `0.766` | `0.790` | `1.000` | `181.73s` | succeeded |
| `gemini:gemini-2.5-flash` | `fused_targeted` | `0.573` | `0.573` | `1.000` | `132.44s` | succeeded |

Takeaway:
- On the heavier melanoma case, `document_auto` is the strongest default.
- `hybrid_targeted` is slower and does not beat `document_auto` here.
- `fused_targeted` is the weakest strategy.

Run:

- `uv run python benchmarks/run_oncology_default.py`
- `uv run python benchmarks/run_nasal_melanoma_models.py`
- `uv run python benchmarks/run_nasal_melanoma_matrix.py`
- `uv run python -m benchmarks.run_benchmarks path/to/manifest.json`
