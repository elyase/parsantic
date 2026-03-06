"""parsantic PDF extraction demo — extract structured data from a PDF.

Public PDF controls:

  mode:
    "auto"     — text-layer PDFs use text extraction, otherwise pages are rasterized
    "document" — one whole-document extraction
    "page"     — page-by-page extraction
    "hybrid"   — whole-document branch + page branch, then merge

  document_input:
    "auto"   — let parsantic choose
    "native" — send the raw PDF directly to the model
    "image"  — rasterize pages and bundle them as one whole-document request

  page_input:
    "auto" / "image" — page-grounded extraction currently uses page images

Requires: uv sync --extra ai --extra vision
Requires: GEMINI_API_KEY env var by default,
or set PARSANTIC_MODEL explicitly.
"""

import os
from pathlib import Path

from pydantic import BaseModel

from parsantic.extract import Document, ExtractOptions, extract


class Invoice(BaseModel):
    invoice_number: str = ""
    date: str = ""
    vendor: str = ""
    total: float = 0.0


pdf = (Path(__file__).parent / "sample_invoice.pdf").read_bytes()
model = os.getenv("PARSANTIC_MODEL", "gemini:gemini-2.5-flash")


# ── 1. Auto: detects text layer → extracts text, skips vision entirely ─

result = extract(Document.from_pdf(pdf), Invoice, model=model)
print("1. mode='auto' (default — text layer detected, no vision call):")
print(f"   {result.value!r}\n")


# ── 2. Whole-document native PDF: one direct PDF call ──────────────────

result = extract(
    Document.from_pdf(pdf),
    Invoice,
    model=model,
    options=ExtractOptions(mode="document", document_input="native"),
)
print("2. mode='document', document_input='native' (full PDF in one call):")
print(f"   {result.value!r}\n")


# ── 3. Page mode: each page → image, one call per page ────────────────

result = extract(
    Document.from_pdf(pdf),
    Invoice,
    model=model,
    options=ExtractOptions(mode="page"),
)
print("3. mode='page' (page-by-page vision):")
print(f"   {result.value!r}\n")


# ── 4. Hybrid: native full PDF + page images with page/document sources ─

result = extract(
    Document.from_pdf(pdf),
    Invoice,
    model=model,
    options=ExtractOptions(
        mode="hybrid",
        document_input="native",
        page_input="image",
    ),
)
print("4. mode='hybrid', document_input='native', page_input='image':")
print(f"   {result.value!r}")
print(f"   sources: {result.sources!r}")
