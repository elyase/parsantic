# Benchmark Datasets

The benchmark runner expects a JSON manifest with a repository-relative root and
an explicit list of benchmark cases.

## Manifest Shape

```json
{
  "name": "pdf-extract-regressions",
  "root": "../../tests/fixtures",
  "cases": [
    {
      "name": "oncology-fhir",
      "document": "oncology.pdf",
      "schema": "my_project.schemas:OncologyRecord",
      "expected": "oncology.expected.json",
      "prompt": "Extract the structured oncology record.",
      "additional_context": "Prefer normalized dates.",
      "tags": ["oncology", "fhir"],
      "expected_api_calls": 1,
      "expected_tokens": 1800,
      "expected_cost_usd": 0.02
    }
  ]
}
```

## Recommended Coverage

The improvement plan calls out four core document categories:

- oncology / FHIR-like PDFs
- table-heavy PDFs
- scanned PDFs
- mixed-quality PDFs

Store the document, its schema fixture, and the ground-truth JSON close together
so benchmark additions stay reviewable.
