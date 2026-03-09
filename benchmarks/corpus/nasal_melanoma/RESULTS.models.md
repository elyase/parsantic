# Nasal Melanoma Model Sweep

Corpus:
- `nasal_melanoma_clean.pdf`
- `nasal_melanoma_table.pdf`
- `nasal_melanoma_scanned.pdf`
- `nasal_melanoma_mixed.pdf`

Schema:
- `NasalMelanomaSnapshot`

Run with:

```bash
uv run python benchmarks/run_nasal_melanoma_models.py
```

Summary:

| Model | Exact | Fuzzy | Completeness | Total latency | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `gemini:gemini-3.1-flash-lite-preview` | `0.823` | `0.823` | `0.976` | `17.63s` | succeeded |
| `gemini:gemini-2.5-flash-lite` | `0.823` | `0.823` | `0.976` | `14.72s` | succeeded |
| `gemini:gemini-2.5-flash` | `0.855` | `0.863` | `0.976` | `38.06s` | succeeded |
| `gemini:gemini-3-flash` | `0.000` | `0.000` | `0.000` | `9.63s` | unsupported on current direct Gemini API path |

Interpretation:

- `gemini:gemini-2.5-flash` had the best extraction quality on this case.
- `gemini:gemini-2.5-flash-lite` matched `gemini:gemini-3.1-flash-lite-preview` on quality and was faster.
- `gemini:gemini-3-flash` is not usable here because the direct API returned `404 model not found`.

Notes:

- This sweep uses fresh Python processes per case to avoid cross-run async state contamination in Gemini clients.
- These results are for the model sweep only. Strategy comparison should still use the dedicated corpus manifests.
