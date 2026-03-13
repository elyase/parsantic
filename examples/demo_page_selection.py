"""Demonstrate deterministic page selection before extraction.

Requires:
  uv sync --extra ai --extra vision
  GEMINI_API_KEY env var by default, or set PARSANTIC_MODEL explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from parsantic import extract
from parsantic.extract import Document, analyze_pdf_source, select_pdf_pages


class LabsOnly(BaseModel):
    hemoglobin_g_dl: float = Field(description="Hemoglobin lab value in g/dL")
    creatinine_mg_dl: float = Field(description="Creatinine lab value in mg/dL")


def main() -> None:
    model = os.getenv("PARSANTIC_MODEL", "gemini:gemini-2.5-flash-lite")
    pdf_path = Path(__file__).with_name("sample_oncology_summary.pdf")

    analysis = analyze_pdf_source(pdf_path)
    selection = select_pdf_pages(analysis, LabsOnly, window=1, max_pages=4)
    result = extract(
        Document.from_pdf(pdf_path, page_indices=selection.page_indices),
        LabsOnly,
        model=model,
    )

    print("Selected pages:", selection.page_indices)
    print("Fallback reason:", selection.fallback_reason)
    print(result.value.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
