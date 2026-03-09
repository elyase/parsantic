# Benchmarks

This folder keeps three maintained benchmarks:

- `oncology`: strategy benchmark
- `nasal_melanoma`: model and strategy benchmark
- `oncology_page_scale`: latency-vs-pages benchmark

Default recommendation:

- Default model: `gemini:gemini-2.5-flash-lite`
- Default strategy: `document_auto`
- Default repair mode: `targeted`
- If page-level provenance matters, prefer `hybrid_targeted`

Benchmarks:
- `oncology`: 4 generated PDFs, about 4-5 pages each, small oncology snapshot.
- `nasal_melanoma`: 4 generated PDFs, about 2-3 pages each, more realistic imaging/surgery/pathology note.
- `oncology_page_scale`: 3 generated scanned PDFs at 5, 10, and 15 pages. The first 5 pages contain the oncology snapshot and the remaining pages are irrelevant appendices, so the benchmark isolates latency growth as total PDF pages increase.

Strategies:
- `document_auto`: this is the library default and is equivalent to calling `ExtractOptions()` with no explicit `mode` or `strategy`.
  For PDFs it picks the cheapest workable path in this order:
  1. if the PDF has a text layer, extract the document as text and run one whole-document text extraction pass
  2. otherwise, if the provider supports native PDF input, send the PDF natively
  3. otherwise, rasterize the PDF to page images and process the image path
  This is the default because it was the most stable strategy and usually the fastest good option.
- `hybrid_targeted`: whole-document pass plus page-level pass, merge, then validation-guided repair.
  This can improve quality, but it is slower and not the best default tradeoff.
- `fused_targeted`: page-level extraction with page image plus page-local text together, then repair.
  This was consistently weaker in the current benchmarks.

Metrics:
- `exact`: fields that match ground truth exactly.
- `fuzzy`: fields that are acceptably close after normalization.
- `completeness`: expected fields that were present at all.
- `provenance`: fields whose source matched the expected scope/page exactly.
- `page coverage`: expected page-local fields that came back with any page-local source.
- `total latency`: wall-clock time for the whole benchmark slice.

### Oncology Strategy Benchmark

Model used: `gemini:gemini-2.5-flash-lite`

| Strategy | Exact | Fuzzy | Completeness | Provenance | Page coverage | Total latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `document_auto` | `0.639` | `0.667` | `1.000` | `0.000` | `0.000` | `19.17s` |
| `hybrid_targeted` | `0.917` | `0.917` | `1.000` | `0.833` | `0.889` | `141.19s` |
| `fused_targeted` | `0.583` | `0.583` | `1.000` | `0.667` | `1.000` | `75.86s` |

Takeaway:
- Best quality: `hybrid_targeted`
- Best default tradeoff: `document_auto`
- Best provenance: `hybrid_targeted`

### Nasal Melanoma Model Benchmark

Strategy used: `document_auto`

| Model | Exact | Fuzzy | Completeness | Provenance | Page coverage | Total latency | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gemini:gemini-3.1-flash-lite-preview` | `0.823` | `0.823` | `0.976` | `0.000` | `0.000` | `29.39s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `0.831` | `0.831` | `0.976` | `0.000` | `0.000` | `17.01s` | succeeded |
| `gemini:gemini-2.5-flash` | `0.315` | `0.395` | `0.476` | `0.000` | `0.000` | `45.63s` | degraded by transient connect errors |
| `gemini:gemini-3-flash` | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | `7.51s` | unsupported (`404`) |

Takeaway:
- `gemini-2.5-flash-lite` and `gemini-3.1-flash-lite-preview` are effectively tied on quality here.
- `gemini-2.5-flash-lite` was faster, so it is the better default.
- `document_auto` does not provide strong page-level provenance regardless of model.

### Nasal Melanoma Strategy Matrix

| Model | Strategy | Exact | Fuzzy | Completeness | Provenance | Page coverage | Total latency | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gemini:gemini-3.1-flash-lite-preview` | `document_auto` | `0.823` | `0.823` | `0.976` | `0.000` | `0.000` | `21.00s` | succeeded |
| `gemini:gemini-3.1-flash-lite-preview` | `hybrid_targeted` | `0.581` | `0.581` | `0.750` | `0.427` | `0.573` | `109.48s` | partial failures |
| `gemini:gemini-3.1-flash-lite-preview` | `fused_targeted` | `0.516` | `0.532` | `1.000` | `0.169` | `0.500` | `62.71s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `document_auto` | `0.831` | `0.831` | `0.976` | `0.000` | `0.000` | `16.77s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `hybrid_targeted` | `0.815` | `0.839` | `1.000` | `0.556` | `0.823` | `48.50s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `fused_targeted` | `0.524` | `0.524` | `1.000` | `0.185` | `0.500` | `27.64s` | succeeded |
| `gemini:gemini-2.5-flash` | `document_auto` | `0.758` | `0.823` | `0.976` | `0.000` | `0.000` | `90.87s` | succeeded |
| `gemini:gemini-2.5-flash` | `hybrid_targeted` | `0.823` | `0.823` | `1.000` | `0.492` | `0.806` | `187.00s` | succeeded |
| `gemini:gemini-2.5-flash` | `fused_targeted` | `0.540` | `0.548` | `1.000` | `0.185` | `0.500` | `148.60s` | succeeded |

Takeaway:
- On the heavier melanoma case, `document_auto` is the strongest default.
- If page-level provenance matters, `hybrid_targeted` is the best option despite the latency cost.
- `fused_targeted` is the weakest strategy.

### Oncology Page-Scale Benchmark

Document type: scanned PDF with a fixed 5-page oncology core plus irrelevant appendix pages

| Model | Strategy | 5 pages | 10 pages | 15 pages | Slope (s/page) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `gemini:gemini-2.5-flash-lite` | `document_auto` | `6.86s` | `12.27s` | `18.20s` | `1.13` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `fused_targeted` | `12.16s` | `18.53s` | `41.87s` | `2.97` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `hybrid_targeted` | `17.15s` | `34.67s` | `47.74s` | `3.06` | succeeded |
| `gemini:gemini-3.1-flash-lite-preview` | `document_auto` | `10.40s` | `16.75s` | `21.78s` | `1.14` | succeeded |
| `gemini:gemini-3.1-flash-lite-preview` | `fused_targeted` | `11.91s` | `24.46s` | `39.94s` | `2.80` | succeeded |
| `gemini:gemini-3.1-flash-lite-preview` | `hybrid_targeted` | `28.44s` | `42.97s` | `61.83s` | `3.34` | succeeded |

Takeaway:
- `document_auto` scales best with page count on scanned PDFs for both tested models, at about `1.1s` per added page in this setup.
- `hybrid_targeted` remains the slowest strategy as pages increase, and its latency slope is about 3x `document_auto`.
- `gemini:gemini-2.5-flash-lite` stayed faster than `gemini:gemini-3.1-flash-lite-preview` across every strategy in this benchmark.

Run:

- `uv run python benchmarks/run_oncology_default.py`
- `uv run python benchmarks/run_oncology_page_scale.py`
- `uv run python benchmarks/run_nasal_melanoma_models.py`
- `uv run python benchmarks/run_nasal_melanoma_matrix.py`
- `uv run python -m benchmarks.run_benchmarks path/to/manifest.json`
