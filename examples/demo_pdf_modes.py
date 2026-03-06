"""Compare the main PDF extraction modes on a synthetic oncology summary.

This is a mode matrix demo. For the minimal recommended path, run
`examples/demo_pdf.py` instead.

Requires:
  uv sync --extra ai --extra vision
  GEMINI_API_KEY env var by default, or set PARSANTIC_MODEL explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from parsantic.extract import Document, ExtractOptions, extract


class OncologyModeSummary(BaseModel):
    patient_name: str = ""
    diagnosis_stage: str = ""
    pathology_conclusion: str = ""
    hemoglobin_g_dl: float = 0.0
    primary_therapy: str = ""
    care_plan: str = ""


def main() -> None:
    model = os.getenv("PARSANTIC_MODEL", "gemini:gemini-3.1-flash-lite-preview")
    pdf_path = Path(__file__).with_name("sample_oncology_summary.pdf")

    cases = (
        ("1. auto", None),
        ("2. document/native", ExtractOptions(mode="document", document_input="native")),
        ("3. page", ExtractOptions(mode="page")),
        (
            "4. hybrid (native + image pages)",
            ExtractOptions(
                mode="hybrid",
                document_input="native",
                page_input="image",
            ),
        ),
    )

    for name, options in cases:
        result = extract(
            Document.from_pdf(pdf_path),
            OncologyModeSummary,
            model=model,
            options=options,
        )
        summary = ", ".join(
            (
                f"patient={result.value.patient_name!r}",
                f"stage={result.value.diagnosis_stage!r}",
                f"hemoglobin={result.value.hemoglobin_g_dl!r}",
                f"therapy={result.value.primary_therapy!r}",
                f"care_plan={result.value.care_plan!r}",
            )
        )
        print(name)
        print(f"  summary: {summary}")
        if result.sources:
            provenance = ", ".join(
                (
                    f"{path}=page {', '.join(str(page) for page in source.pages)}"
                    if source.scope == "page"
                    else f"{path}=document"
                )
                for path, source in sorted(result.sources.items())
            )
            print(f"  provenance: {provenance}")
        print()


if __name__ == "__main__":
    main()
