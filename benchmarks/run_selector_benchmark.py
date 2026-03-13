from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from parsantic.extract import Document, aextract, analyze_pdf_source, select_pdf_pages

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus" / "oncology_page_scale"
GENERATED = CORPUS / "generated"
LABELS = json.loads((CORPUS / "selector_labels.json").read_text())
OUTPUT = CORPUS / "results.selector.json"


class LabsOnly(BaseModel):
    hemoglobin_g_dl: float = Field(description="Hemoglobin lab value in g/dL")
    creatinine_mg_dl: float = Field(description="Creatinine lab value in mg/dL")


class MedicationOnly(BaseModel):
    primary_medication: str = Field(description="Primary chemotherapy medication name")


@dataclass(frozen=True, slots=True)
class SelectorCase:
    name: str
    pdf_path: Path
    schema: type[BaseModel]
    oracle_pages: tuple[int, ...]


def _load_cases() -> list[SelectorCase]:
    cases: list[SelectorCase] = []
    for file_name, labels in sorted(LABELS.items()):
        pdf_path = GENERATED / file_name
        cases.append(
            SelectorCase(
                name=f"{file_name}:labs_only",
                pdf_path=pdf_path,
                schema=LabsOnly,
                oracle_pages=tuple(labels["labs_only"]),
            )
        )
        cases.append(
            SelectorCase(
                name=f"{file_name}:medication_only",
                pdf_path=pdf_path,
                schema=MedicationOnly,
                oracle_pages=tuple(labels["medication_only"]),
            )
        )
    return cases


def _page_recall(
    selected: tuple[int, ...] | None, oracle: tuple[int, ...], page_count: int
) -> float:
    if selected is None:
        selected = tuple(range(page_count))
    if not oracle:
        return 1.0
    overlap = len(set(selected) & set(oracle))
    return overlap / len(oracle)


async def _run_case(case: SelectorCase, model: str) -> dict[str, object]:
    analysis = analyze_pdf_source(case.pdf_path)
    selection = select_pdf_pages(analysis, case.schema, window=1, max_pages=4)
    page_recall = _page_recall(selection.page_indices, case.oracle_pages, analysis.page_count)

    full_started = asyncio.get_running_loop().time()
    await aextract(Document.from_pdf(case.pdf_path), case.schema, model=model)
    full_latency = asyncio.get_running_loop().time() - full_started

    selected_started = asyncio.get_running_loop().time()
    await aextract(
        Document.from_pdf(case.pdf_path, page_indices=selection.page_indices),
        case.schema,
        model=model,
    )
    selected_latency = asyncio.get_running_loop().time() - selected_started

    selected_pages = (
        analysis.page_count if selection.page_indices is None else len(selection.page_indices)
    )
    return {
        "case": case.name,
        "page_count": analysis.page_count,
        "selected_page_count": selected_pages,
        "fallback_reason": selection.fallback_reason,
        "page_recall": round(page_recall, 3),
        "full_latency_s": round(full_latency, 3),
        "selected_latency_s": round(selected_latency, 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deterministic page selection.")
    parser.add_argument(
        "--model",
        default=os.getenv("PARSANTIC_MODEL", "gemini:gemini-2.5-flash-lite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    results = [await _run_case(case, args.model) for case in _load_cases()]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
