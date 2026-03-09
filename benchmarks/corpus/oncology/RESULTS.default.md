# Default Oncology Benchmark

Default benchmark:
- Model: `gemini:gemini-3.1-flash-lite-preview`
- Corpus: 4 generated PDFs
  - `oncology_clean.pdf`
  - `oncology_table.pdf`
  - `oncology_scanned.pdf`
  - `oncology_mixed.pdf`
- Schema: `OncologySnapshot`

Current measured results:

| Strategy | Exact | Fuzzy | Completeness | Total latency |
| --- | ---: | ---: | ---: | ---: |
| `document_auto` | `0.972` | `1.000` | `1.000` | `19.6s` |
| `hybrid_targeted` | `1.000` | `1.000` | `1.000` | `58.9s` |
| `fused_targeted` | `0.667` | `0.667` | `1.000` | `43.7s` |

Per-case notes:

- `document_auto`
  - Perfect on `clean`, `table`, and `scanned`
  - On `mixed`, only missed capitalization for `primary_medication` (`capecitabine` vs `Capecitabine`)

- `hybrid_targeted`
  - Perfect on all four PDFs
  - Highest latency due to the extra whole-document + page pass

- `fused_targeted`
  - Structurally complete on all four PDFs
  - Consistently missed three fields:
    - `hemoglobin_g_dl`
    - `creatinine_mg_dl`
    - `primary_medication`

Interpretation:

- Best quality: `hybrid_targeted`
- Best speed/quality tradeoff: `document_auto`
- Not recommended as default for this corpus: `fused_targeted`

Run the default benchmark with:

```bash
uv run python benchmarks/run_oncology_default.py
```
