"""parsantic PDF extraction demo — extract structured data from a PDF.

Two knobs control how PDFs are processed:

  pdf_mode: how the PDF is prepared
    "auto"   — if the PDF has a text layer, extract text (no vision needed);
               otherwise rasterize to images
    "native" — send the raw PDF binary directly to the model
    "raster" — convert each page to a JPEG image first

  page_strategy: how pages are dispatched to the model
    "auto"      — native PDF → single call; rasterized → one call per page
    "single"    — bundle all pages in one model call
    "map_reduce" — one model call per page, results merged

Requires: uv sync --extra ai --extra vision
Requires: GEMINI_API_KEY or OPENAI_API_KEY env var
"""

from pathlib import Path

from pydantic import BaseModel

from parsantic.extract import Document, extract
from parsantic.extract.options import ExtractOptions, MediaOptions


class Invoice(BaseModel):
    invoice_number: str = ""
    date: str = ""
    vendor: str = ""
    total: float = 0.0


pdf = (Path(__file__).parent / "sample_invoice.pdf").read_bytes()
model = "gemini:gemini-2.5-flash"  # or "openai:gpt-4o-mini"


# ── 1. Auto: detects text layer → extracts text, skips vision entirely ─

result = extract(Document.from_pdf(pdf), Invoice, model=model)
print("1. pdf_mode='auto' (default — text layer detected, no vision call):")
print(f"   {result.value!r}\n")


# ── 2. Native: send the full PDF as-is in a single model call ──────────

result = extract(
    Document.from_pdf(pdf),
    Invoice,
    model=model,
    options=ExtractOptions(media=MediaOptions(pdf_mode="native")),
)
print("2. pdf_mode='native' (full PDF sent to model in one call):")
print(f"   {result.value!r}\n")


# ── 3. Raster + map_reduce: each page → JPEG, one call per page ───────

result = extract(
    Document.from_pdf(pdf),
    Invoice,
    model=model,
    options=ExtractOptions(media=MediaOptions(pdf_mode="raster", page_strategy="map_reduce")),
)
print("3. pdf_mode='raster', page_strategy='map_reduce' (page-by-page vision):")
print(f"   {result.value!r}\n")


# ── 4. Raster + single: all pages → JPEGs, bundled in one call ────────

result = extract(
    Document.from_pdf(pdf),
    Invoice,
    model=model,
    options=ExtractOptions(media=MediaOptions(pdf_mode="raster", page_strategy="single")),
)
print("4. pdf_mode='raster', page_strategy='single' (all page images in one call):")
print(f"   {result.value!r}")
