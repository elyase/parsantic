# Oncology Benchmark Corpus

This corpus is generated from a single ground-truth oncology FHIR bundle in
[ground_truth.json](/Users/yaser/parsantic/benchmarks/corpus/oncology/ground_truth.json).

Generated variants:

- `oncology_clean.pdf`: native text-layer narrative pages
- `oncology_table.pdf`: text-layer pages rendered in table-heavy layouts
- `oncology_scanned.pdf`: image-only PDF derived from rendered pages
- `oncology_mixed.pdf`: mixed text, table, and scanned-like pages

Regenerate with:

```bash
uv run python /Users/yaser/parsantic/benchmarks/corpus/oncology/generate.py
```
