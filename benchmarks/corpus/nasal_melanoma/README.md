# Nasal Melanoma Corpus

Realistic benchmark corpus derived from a narrative oncology note describing
left nasal cavity mucosal melanoma workup, imaging, surgery, and pathology.

Files:
- `ground_truth.json`: structured reference output
- `snapshot_schema.py`: benchmark schema
- `generate.py`: produces PDF variants from the source narrative
- `manifest.default.json`: ready-to-run benchmark manifest
- `RESULTS.models.md`: current direct-Gemini model sweep summary

Generated variants:
- `nasal_melanoma_clean.pdf`
- `nasal_melanoma_table.pdf`
- `nasal_melanoma_scanned.pdf`
- `nasal_melanoma_mixed.pdf`

Model sweep:

```bash
uv run python /Users/yaser/parsantic/benchmarks/run_nasal_melanoma_models.py
```
