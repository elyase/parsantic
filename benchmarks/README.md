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

## Current Snapshot

Columns:

- `Accuracy`: exact field match rate
- `Wrong values`: rate of returned fields that were wrong
- `Source grounding`: correct source scope/page attribution
- `Latency`: total runtime

Higher is better for `Accuracy` and `Source grounding`. Lower is better for `Wrong values` and `Latency`.

Why only two models here:

- strategy snapshots keep the model fixed so strategy differences are easier to read
- page-scale keeps the current default model fixed so latency growth is easier to read
- if you want a different model, duplicate a manifest config and rerun

### Oncology

Model: `gemini:gemini-3.1-flash-lite-preview`

| Strategy | Accuracy | Wrong values | Source grounding | Latency |
| --- | ---: | ---: | ---: | ---: |
| `document_auto` | `0.639` | `0.333` | `0.611` | `7.23s` |
| `document_grounded` | `0.639` | `0.333` | `0.611` | `9.97s` |

### Nasal Melanoma

Model: `gemini:gemini-3.1-flash-lite-preview`

| Strategy | Accuracy | Wrong values | Source grounding | Latency |
| --- | ---: | ---: | ---: | ---: |
| `document_auto` | `0.831` | `0.158` | `0.419` | `14.02s` |
| `document_grounded` | `0.831` | `0.158` | `0.419` | `15.68s` |

### Page Scale

Model: `gemini:gemini-2.5-flash-lite`

| Strategy | 5 pages | 10 pages | 15 pages | Slope (s/page) |
| --- | ---: | ---: | ---: | ---: |
| `document_auto` | `7.12s` | `10.79s` | `16.06s` | `0.89` |
| `document_grounded` | `6.47s` | `10.57s` | `15.04s` | `0.86` |

The compact tables above focus on the whole-document family. If you care most about provenance, benchmark the hybrid recipe on your own corpus, because the cost depends heavily on page count and document type.

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

- [oncology results](corpus/oncology/results.default.json)
- [nasal melanoma results](corpus/nasal_melanoma/results.default.json)
- [page-scale results](corpus/oncology_page_scale/results.page_scale.json)

## Run

- `uv run python runner.py --manifest corpus/oncology/manifest.default.json --config document_auto --config document_grounded --output corpus/oncology/results.default.json`
- `uv run python runner.py --manifest corpus/nasal_melanoma/manifest.default.json --config document_auto --config document_grounded --output corpus/nasal_melanoma/results.default.json`
- `uv run python run_oncology_page_scale.py --model gemini:gemini-2.5-flash-lite --strategy document_auto --strategy document_grounded --skip-generate --output corpus/oncology_page_scale/results.page_scale.json`
