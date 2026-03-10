# Benchmarks

## Start Here

| Goal | Use |
| --- | --- |
| Best default | `gemini:gemini-2.5-flash-lite` + default `ExtractOptions(...)` |
| Best provenance | `mode="hybrid"` with `document_input="native"` and `page_input="image"` |
| Fastest | `gemini:gemini-2.5-flash-lite` + default path + fewer repair attempts |

Best default:

```python
from parsantic.extract import ExtractOptions, extract

result = extract(
    document,
    Schema,
    model="gemini:gemini-2.5-flash-lite",
    options=ExtractOptions(
        repair="targeted",
        max_repair_attempts=2,
    ),
)
```

Best provenance:

```python
from parsantic.extract import ExtractOptions, extract

result = extract(
    document,
    Schema,
    model="gemini:gemini-2.5-flash-lite",
    options=ExtractOptions(
        mode="hybrid",
        document_input="native",
        page_input="image",
        repair="targeted",
        max_repair_attempts=2,
    ),
)
```

Fastest:

```python
from parsantic.extract import ExtractOptions, extract

result = extract(
    document,
    Schema,
    model="gemini:gemini-2.5-flash-lite",
    options=ExtractOptions(
        repair="targeted",
        max_repair_attempts=1,
    ),
)
```

## What The Benchmark Labels Mean

| Benchmark label | Library config |
| --- | --- |
| `document_auto` | default whole-document path from `ExtractOptions(...)` |
| `document_grounded` | `ExtractOptions(strategy=Strategy(plan="document_grounded"))` |
| `hybrid_targeted` | `ExtractOptions(mode="hybrid", document_input="native", page_input="image", repair="targeted")` |

## Strategies

### `document_auto`

Single-pass whole-document extraction.

For PDFs it uses:

1. extracted text if a usable text layer exists
2. native PDF input if the provider supports it
3. rasterized page images otherwise

Choose this when you want the best overall tradeoff.

### `document_grounded`

Whole-document extraction with page-aware evidence grounding when page text boundaries are available.

Choose this when you want the whole-document path with explicit page grounding.

### `hybrid_targeted`

Runs a whole-document branch and a page-level branch, merges them, then applies targeted repair.

Choose this when provenance matters more than latency.

## Multi-Model Results

All models routed through a single OpenAI-compatible gateway. Each corpus includes clean, table, scanned, and mixed PDF variants. Accuracy is fuzzy field match rate across all variants.

### Oncology (9 fields, 4 documents)

| Model | Accuracy | Provenance | Latency |
| --- | ---: | ---: | ---: |
| grok-4.1-fast | 83.3% | 75.0% | 10.0s |
| gemini-2.5-flash-lite | 83.3% | 75.0% | 10.7s |
| gemini-3.1-flash-lite | 83.3% | 75.0% | 14.6s |
| claude-haiku-4.5 | 83.3% | 75.0% | 20.5s |
| claude-sonnet-4.6 | 83.3% | 75.0% | 32.6s |
| gpt-5.1-instant | 83.3% | 75.0% | 38.4s |
| gemini-2.5-flash | 83.3% | 75.0% | 48.2s |
| gpt-5-mini | 83.3% | 75.0% | 75.5s |
| gpt-5-nano | 83.3% | 75.0% | 99.0s |
| gemini-3.1-pro | 83.3% | 75.0% | 140.1s |
| glm-4.7-flash | 44.4% | 38.9% | 5.7s |

### Nasal Melanoma (31 fields, 4 documents)

| Model | Accuracy | Provenance | Latency |
| --- | ---: | ---: | ---: |
| claude-sonnet-4.6 | 79.0% | 52.4% | 38.8s |
| gpt-5.1-instant | 79.0% | 52.4% | 36.1s |
| gpt-5-mini | 78.2% | 52.4% | 92.2s |
| gpt-5-nano | 77.4% | 53.2% | 155.9s |
| claude-haiku-4.5 | 77.4% | 51.6% | 22.7s |
| gemini-3.1-pro | 76.6% | 49.2% | 85.8s |
| gemini-2.5-flash-lite | 76.6% | 50.0% | 11.8s |
| gemini-2.5-flash | 75.8% | 50.8% | 45.3s |
| gemini-3.1-flash-lite | 74.2% | 46.8% | 14.9s |
| grok-4.1-fast | 69.4% | 50.8% | 9.2s |
| glm-4.7-flash | 48.4% | 28.2% | 7.9s |

Clean and table PDFs reach ~100% accuracy across all models. The overall numbers are brought down by scanned/mixed variants, which require vision-based extraction through rasterized images.

### Page Scale

Model: `gemini:gemini-2.5-flash-lite`

| Strategy | 5 pages | 10 pages | 15 pages | Slope (s/page) |
| --- | ---: | ---: | ---: | ---: |
| `document_auto` | 7.12s | 10.79s | 16.06s | 0.89 |
| `document_grounded` | 6.47s | 10.57s | 15.04s | 0.86 |

## Useful Knobs

Quality:

- `model`
- prompt wording
- prompt examples
- `repair`
- `max_repair_attempts`
- `structured_output`

Provenance:

- `strategy` or `mode`
- `document_input`
- `page_input`

Latency:

- `model`
- `max_repair_attempts`
- `max_workers`
- whether you use whole-document or hybrid extraction

## Full Results

Full metrics stay in the JSON result files:

- [oncology multi-model](corpus/oncology/results_multimodel.json)
- [nasal melanoma multi-model](corpus/nasal_melanoma/results_multimodel.json)
- [oncology strategy comparison](corpus/oncology/results.default.json)
- [nasal melanoma strategy comparison](corpus/nasal_melanoma/results.default.json)
- [page-scale results](corpus/oncology_page_scale/results.page_scale.json)

## Run

- `uv run python runner.py --manifest corpus/oncology/manifest.default.json --config document_auto --config document_grounded --output corpus/oncology/results.default.json`
- `uv run python runner.py --manifest corpus/nasal_melanoma/manifest.default.json --config document_auto --config document_grounded --output corpus/nasal_melanoma/results.default.json`
- `uv run python run_oncology_page_scale.py --model gemini:gemini-2.5-flash-lite --strategy document_auto --strategy document_grounded --skip-generate --output corpus/oncology_page_scale/results.page_scale.json`
