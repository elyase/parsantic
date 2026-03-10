# Benchmarks

## Start Here

| Goal | Use |
| --- | --- |
| Best default | `gemini:gemini-2.5-flash-lite` + default `ExtractOptions(...)` |
| Best provenance | `mode="hybrid"` with `document_input="native"` and `page_input="image"` |
| Fastest | `gemini:gemini-2.5-flash-lite` + default path + fewer repair attempts |

!!! tip
    Start with the default recipe unless page-level provenance or strict latency targets are the main requirement.

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

## Benchmark Labels

| Benchmark label | Library config |
| --- | --- |
| `document_auto` | default whole-document path from `ExtractOptions(...)` |
| `document_grounded` | `ExtractOptions(strategy=Strategy(plan="document_grounded"))` |
| `hybrid_targeted` | `ExtractOptions(mode="hybrid", document_input="native", page_input="image", repair="targeted")` |

## How To Read The Tables

- `Accuracy`: exact field match rate
- `Wrong values`: rate of returned fields that were wrong
- `Source grounding`: correct source scope/page attribution
- `Latency`: total runtime

Higher is better for `Accuracy` and `Source grounding`. Lower is better for `Wrong values` and `Latency`.

The strategy snapshots keep the model fixed so strategy differences are easier to read. The page-scale snapshot keeps the default model fixed so latency growth is easier to read.

## Current Snapshot

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

- [oncology results](https://github.com/elyase/parsantic/blob/main/benchmarks/corpus/oncology/results.default.json)
- [nasal melanoma results](https://github.com/elyase/parsantic/blob/main/benchmarks/corpus/nasal_melanoma/results.default.json)
- [page-scale results](https://github.com/elyase/parsantic/blob/main/benchmarks/corpus/oncology_page_scale/results.page_scale.json)
