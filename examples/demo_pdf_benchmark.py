"""Benchmark page and hybrid PDF extraction with and without parallelism.

This is a speed demo, not the main product demo. It expands the synthetic
oncology PDF into a temporary multi-page packet, then measures async
extraction latency for sequential vs parallel page fanout.

Requires:
  uv sync --extra ai --extra vision
  GEMINI_API_KEY env var by default, or set PARSANTIC_MODEL explicitly.

Optional env vars:
  PARSANTIC_BENCH_REPEATS  Number of times to repeat the 5-page sample (default: 2)
  PARSANTIC_BENCH_WORKERS  Parallel worker count for the parallel cases (default: 8)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import fitz
from pydantic import BaseModel

from parsantic.extract import Document, ExtractOptions, aextract


class OncologyBenchmarkSummary(BaseModel):
    patient_name: str = ""
    diagnosis_stage: str = ""
    pathology_conclusion: str = ""
    hemoglobin_g_dl: float = 0.0
    primary_therapy: str = ""
    care_plan: str = ""


def _build_expanded_pdf(source_pdf: Path, *, repeats: int) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="parsantic-bench-"))
    output_path = temp_dir / f"sample_oncology_summary_x{repeats}.pdf"

    source = fitz.open(source_pdf)
    expanded = fitz.open()
    try:
        for _ in range(max(1, repeats)):
            expanded.insert_pdf(source)
        expanded.save(output_path)
    finally:
        expanded.close()
        source.close()

    return output_path


def _summarize(result: object) -> str:
    value = result.value
    return ", ".join(
        (
            f"patient={value.patient_name!r}",
            f"stage={value.diagnosis_stage!r}",
            f"hemoglobin={value.hemoglobin_g_dl!r}",
            f"therapy={value.primary_therapy!r}",
        )
    )


async def _time_case(
    *,
    name: str,
    pdf_path: Path,
    model: str,
    options: ExtractOptions,
) -> tuple[str, float, str]:
    started = time.perf_counter()
    result = await aextract(
        Document.from_pdf(pdf_path),
        OncologyBenchmarkSummary,
        model=model,
        options=options,
    )
    elapsed = time.perf_counter() - started
    return name, elapsed, _summarize(result)


async def main() -> None:
    model = os.getenv("PARSANTIC_MODEL", "gemini:gemini-3.1-flash-lite-preview")
    repeats = max(1, int(os.getenv("PARSANTIC_BENCH_REPEATS", "2")))
    workers = max(1, int(os.getenv("PARSANTIC_BENCH_WORKERS", "8")))
    source_pdf = Path(__file__).with_name("sample_oncology_summary.pdf")
    expanded_pdf = _build_expanded_pdf(source_pdf, repeats=repeats)
    page_count = repeats * 5

    cases = (
        (
            "page / sequential",
            ExtractOptions(mode="page", max_workers=1),
        ),
        (
            "page / parallel",
            ExtractOptions(mode="page", max_workers=workers),
        ),
        (
            "hybrid / sequential pages",
            ExtractOptions(
                mode="hybrid",
                document_input="native",
                page_input="image",
                max_workers=1,
            ),
        ),
        (
            "hybrid / parallel pages",
            ExtractOptions(
                mode="hybrid",
                document_input="native",
                page_input="image",
                max_workers=workers,
            ),
        ),
    )

    print(f"Model: {model}")
    print(f"Benchmark PDF: {expanded_pdf.name} ({page_count} pages)")
    print()

    baseline: float | None = None
    for name, options in cases:
        case_name, elapsed, summary = await _time_case(
            name=name,
            pdf_path=expanded_pdf,
            model=model,
            options=options,
        )
        if baseline is None:
            baseline = elapsed
            speedup_text = "1.00x"
        else:
            speedup_text = f"{baseline / elapsed:.2f}x"
        print(f"{case_name:26} {elapsed:>6.2f}s  {speedup_text}")
        print(f"  summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
