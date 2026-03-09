from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus" / "nasal_melanoma"
TRUTH = CORPUS / "ground_truth.json"
EXPECTED_SOURCES = CORPUS / "expected_sources.json"
OUTPUT = CORPUS / "results.models.json"
HOME_ENV = Path.home() / ".env"
MODELS = [
    "gemini:gemini-3.1-flash-lite-preview",
    "gemini:gemini-2.5-flash-lite",
    "gemini:gemini-2.5-flash",
    "gemini:gemini-3-flash",
]
PDFS = [
    "generated/nasal_melanoma_clean.pdf",
    "generated/nasal_melanoma_table.pdf",
    "generated/nasal_melanoma_scanned.pdf",
    "generated/nasal_melanoma_mixed.pdf",
]


def _load_home_env() -> None:
    if "GEMINI_API_KEY" in os.environ or not HOME_ENV.exists():
        return
    for line in HOME_ENV.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "GEMINI_API_KEY" and "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = value.strip()


def main() -> int:
    _load_home_env()
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY is required. Put it in ~/.env or export it in the shell.")

    script = r"""
import json, time
from pathlib import Path
from pydantic import BaseModel
from parsantic.extract import Document, ExtractOptions, extract
from benchmarks.metrics import score_case, score_provenance_case

class NasalMelanomaSnapshot(BaseModel):
    age_years: int
    sex: str
    comorbidities: list[str]
    presenting_symptom_month: str
    primary_site: str
    diagnosis: str
    ct_sinus_date: str
    ct_lesion_size_cm: list[float]
    mri_face_date: str
    mri_mass_size_cm: list[float]
    left_submandibular_node_cm: float
    pet_ct_date: str
    pet_hypermetabolic_nasal_mass: bool
    metastatic_disease_on_pet: bool
    left_apical_lung_nodule_present: bool
    surgery_date: str
    surgery_procedures: list[str]
    pathology_size_cm: float
    margins_negative: bool
    lymphovascular_invasion: bool
    perineural_invasion: bool
    nodes_positive: int
    nodes_examined: int
    pathologic_t_stage: str
    pathologic_n_stage: str

truth = json.loads(Path({truth!r}).read_text())
expected_sources = json.loads(Path({expected_sources!r}).read_text())
pdf_path = Path({pdf!r})
model = {model!r}
start = time.perf_counter()
output = {{}}
error = None
sources = {{}}
try:
    result = extract(
        Document.from_pdf(
            pdf_path,
            text='Extract the nasal melanoma snapshot into the flat schema. Use exact values from the document.',
            document_id=pdf_path.stem,
        ),
        NasalMelanomaSnapshot,
        model=model,
        options=ExtractOptions(repair='targeted', max_repair_attempts=1, per_call_timeout_s=60, per_document_timeout_s=180),
    )
    output = result.value.model_dump(mode='json')
    sources = result.sources
except Exception as exc:
    error = f"{{type(exc).__name__}}: {{exc}}"
elapsed = time.perf_counter() - start
metrics = score_case(truth, output)
provenance_accuracy, page_coverage = score_provenance_case(expected_sources, sources)
print(json.dumps({{
    'model': model,
    'pdf': pdf_path.name,
    'exact_accuracy': metrics.exact_accuracy,
    'fuzzy_accuracy': metrics.fuzzy_accuracy,
    'completeness': metrics.completeness,
    'provenance_accuracy': provenance_accuracy,
    'page_coverage': page_coverage,
    'latency_s': elapsed,
    'error': error,
    'output': output,
}}))
"""

    rows: list[dict[str, object]] = []
    for model in MODELS:
        for pdf in PDFS:
            payload = script.format(
                truth=str(TRUTH),
                expected_sources=str(EXPECTED_SOURCES),
                pdf=str(CORPUS / pdf),
                model=model,
            )
            completed = subprocess.run(
                [sys.executable, "-c", payload],
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            stdout = completed.stdout.strip().splitlines()
            if stdout:
                row = json.loads(stdout[-1])
            else:
                row = {
                    "model": model,
                    "pdf": Path(pdf).name,
                    "exact_accuracy": 0.0,
                    "fuzzy_accuracy": 0.0,
                    "completeness": 0.0,
                    "provenance_accuracy": 0.0,
                    "page_coverage": 0.0,
                    "latency_s": 0.0,
                    "error": completed.stderr.strip() or "No output",
                    "output": {},
                }
            rows.append(row)
            print(
                row["model"],
                row["pdf"],
                round(float(row["exact_accuracy"]), 3),
                round(float(row["fuzzy_accuracy"]), 3),
                round(float(row["completeness"]), 3),
                round(float(row["provenance_accuracy"]), 3),
                round(float(row["page_coverage"]), 3),
                round(float(row["latency_s"]), 2),
                str(row["error"] or "")[:120],
                flush=True,
            )

    summaries: list[dict[str, object]] = []
    for model in MODELS:
        group = [row for row in rows if row["model"] == model]
        summaries.append(
            {
                "model": model,
                "exact_accuracy": sum(float(row["exact_accuracy"]) for row in group) / len(group),
                "fuzzy_accuracy": sum(float(row["fuzzy_accuracy"]) for row in group) / len(group),
                "completeness": sum(float(row["completeness"]) for row in group) / len(group),
                "provenance_accuracy": sum(float(row["provenance_accuracy"]) for row in group)
                / len(group),
                "page_coverage": sum(float(row["page_coverage"]) for row in group) / len(group),
                "total_latency_s": sum(float(row["latency_s"]) for row in group),
                "all_cases_succeeded": all(not row["error"] for row in group),
            }
        )

    OUTPUT.write_text(json.dumps({"summary": summaries, "cases": rows}, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
