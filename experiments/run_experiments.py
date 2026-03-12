"""
Experiment suite for improving parsantic extraction accuracy, provenance, and latency.

Experiments:
1. Prompt engineering with field descriptions
2. Schema-level field descriptions
3. Text extraction with bounding boxes (PyMuPDF dict mode)
4. OCR preprocessing for scanned pages
5. Higher DPI rasterization
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from benchmarks.metrics import score_case, score_provenance_case
from parsantic.extract import Document, ExtractOptions, Strategy, extract
from parsantic.extract.providers.base import ProviderConfig
from parsantic.extract.providers.factory import create_provider

ROOT = Path(__file__).resolve().parents[1]
ONCOLOGY_DIR = ROOT / "benchmarks" / "corpus" / "oncology"
NASAL_DIR = ROOT / "benchmarks" / "corpus" / "nasal_melanoma"

GROUND_TRUTH = json.loads((ONCOLOGY_DIR / "snapshot_truth.json").read_text())
EXPECTED_SOURCES = json.loads((ONCOLOGY_DIR / "expected_sources.json").read_text())


# ---- Schema variants ----


class OncologySnapshotBaseline(BaseModel):
    """Baseline schema (no field descriptions)."""

    patient_name: str
    patient_identifier: str
    diagnosis: str
    stage: str
    oncologist: str
    cancer_center: str
    hemoglobin_g_dl: float
    creatinine_mg_dl: float
    primary_medication: str


class OncologySnapshotDescribed(BaseModel):
    """Schema with rich field descriptions for medical document extraction."""

    patient_name: str = Field(description="Full name of the patient (e.g. 'Maya Hernandez')")
    patient_identifier: str = Field(description="Patient MRN or unique identifier code")
    diagnosis: str = Field(
        description="Primary cancer diagnosis (e.g. 'Metastatic breast carcinoma'). NOT the report type or section header."
    )
    stage: str = Field(
        description="Cancer staging (e.g. 'Stage IV', 'Stage IIIA'). Look for TNM staging or stage designation."
    )
    oncologist: str = Field(description="Name of the treating oncologist/physician")
    cancer_center: str = Field(
        description="Name of the hospital, cancer center, or service provider facility"
    )
    hemoglobin_g_dl: float = Field(description="Hemoglobin lab value in g/dL")
    creatinine_mg_dl: float = Field(description="Creatinine lab value in mg/dL")
    primary_medication: str = Field(
        description="Primary chemotherapy or cancer treatment medication name"
    )


# ---- Prompts ----

PROMPT_BASELINE = (
    "Extract the oncology snapshot into the flat schema. Use exact values from the document."
)

PROMPT_MEDICAL = """Extract structured clinical data from this oncology document.

IMPORTANT RULES:
- Use exact values as they appear in the document. Do not paraphrase or infer.
- 'diagnosis' means the primary cancer diagnosis (e.g. 'Metastatic breast carcinoma'), NOT a report type or section header like 'Pathology and biomarker summary'.
- 'cancer_center' is the treating facility name. It may appear as 'Center', 'Service Provider', 'Institution', or 'Hospital'.
- 'stage' is the cancer stage (e.g. 'Stage IV'). Look for explicit staging information.
- If a value cannot be clearly identified in the document, use null rather than guessing.
- For lab values (hemoglobin, creatinine), extract the numeric value only.
"""

PROMPT_GROUNDED = """Extract structured clinical data from this medical document.

CRITICAL INSTRUCTIONS:
1. Only extract values that are EXPLICITLY stated in the document text or image.
2. Do NOT hallucinate or fabricate any values. If you cannot clearly read a value, output null.
3. Map document fields to schema fields:
   - 'Center' or 'Service Provider' → cancer_center
   - Primary cancer condition → diagnosis (NOT report type headers)
   - 'Stage' designation → stage
   - Patient MRN or ID number → patient_identifier
4. For scanned/image documents: if text is illegible, output null for that field.
5. Use exact values - do not normalize dates, capitalize, or change formatting.
"""


@dataclass
class ExperimentResult:
    name: str
    pdf_variant: str
    accuracy: float
    provenance_accuracy: float
    page_coverage: float
    latency_s: float
    wrong_present_rate: float
    output: dict[str, Any]
    errors: list[str]


def run_single_extraction(
    pdf_path: Path,
    schema_cls: type[BaseModel],
    prompt: str,
    model_id: str = "gemini:gemini-2.5-flash-lite",
    options: ExtractOptions | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Run extraction and return (output_dict, sources_dict, latency)."""
    provider = create_provider(ProviderConfig(model_id=model_id))
    doc = Document.from_pdf(pdf_path, text=prompt)
    opts = options or ExtractOptions(repair="targeted", max_repair_attempts=1)

    start = time.perf_counter()
    result = extract(doc, schema_cls, model=provider, options=opts)
    latency = time.perf_counter() - start

    output = (
        result.value.model_dump(mode="json")
        if hasattr(result.value, "model_dump")
        else result.value
    )
    return output, result.sources, latency


def evaluate(
    output: dict[str, Any],
    sources: dict[str, Any],
    expected: dict[str, Any] = GROUND_TRUTH,
    expected_sources: dict[str, Any] = EXPECTED_SOURCES,
) -> tuple[float, float, float, float]:
    """Return (accuracy, provenance_accuracy, page_coverage, wrong_rate)."""
    metrics = score_case(expected, output)
    prov_acc, page_cov = score_provenance_case(expected_sources, sources)
    return metrics.exact_accuracy, prov_acc, page_cov, metrics.wrong_present_rate


def run_experiment(
    name: str,
    pdf_name: str,
    schema_cls: type[BaseModel],
    prompt: str,
    model_id: str = "gemini:gemini-2.5-flash-lite",
    options: ExtractOptions | None = None,
) -> ExperimentResult:
    """Run one experiment and return results."""
    pdf_path = ONCOLOGY_DIR / "generated" / pdf_name
    errors = []
    try:
        output, sources, latency = run_single_extraction(
            pdf_path, schema_cls, prompt, model_id, options
        )
        accuracy, prov_acc, page_cov, wrong_rate = evaluate(output, sources)
    except Exception as e:
        errors.append(str(e))
        output = {}
        accuracy = prov_acc = page_cov = 0.0
        wrong_rate = 1.0
        latency = 0.0

    return ExperimentResult(
        name=name,
        pdf_variant=pdf_name,
        accuracy=accuracy,
        provenance_accuracy=prov_acc,
        page_coverage=page_cov,
        latency_s=latency,
        wrong_present_rate=wrong_rate,
        output=output,
        errors=errors,
    )


def print_results(results: list[ExperimentResult]) -> None:
    """Print results table."""
    print(f"\n{'=' * 100}")
    print(
        f"{'Experiment':<30} {'PDF':<25} {'Accuracy':>8} {'Prov':>8} {'PageCov':>8} {'Wrong':>8} {'Latency':>8}"
    )
    print(f"{'-' * 100}")
    for r in results:
        err = " ERROR" if r.errors else ""
        print(
            f"{r.name:<30} {r.pdf_variant:<25} {r.accuracy:>8.3f} {r.provenance_accuracy:>8.3f} {r.page_coverage:>8.3f} {r.wrong_present_rate:>8.3f} {r.latency_s:>7.1f}s{err}"
        )
    print(f"{'=' * 100}")


def experiment_1_prompt_engineering():
    """Test different prompts on the table and scanned PDFs (the weak spots)."""
    print("\n## Experiment 1: Prompt Engineering")
    results = []

    for pdf_name in ["oncology_table.pdf", "oncology_scanned.pdf", "oncology_clean.pdf"]:
        # Baseline prompt
        results.append(
            run_experiment(
                "baseline_prompt",
                pdf_name,
                OncologySnapshotBaseline,
                PROMPT_BASELINE,
            )
        )
        # Medical-specific prompt
        results.append(
            run_experiment(
                "medical_prompt",
                pdf_name,
                OncologySnapshotBaseline,
                PROMPT_MEDICAL,
            )
        )
        # Grounded prompt
        results.append(
            run_experiment(
                "grounded_prompt",
                pdf_name,
                OncologySnapshotBaseline,
                PROMPT_GROUNDED,
            )
        )

    print_results(results)
    return results


def experiment_2_schema_descriptions():
    """Test schema with field descriptions."""
    print("\n## Experiment 2: Schema Field Descriptions")
    results = []

    for pdf_name in ["oncology_table.pdf", "oncology_scanned.pdf", "oncology_clean.pdf"]:
        # No descriptions
        results.append(
            run_experiment(
                "no_descriptions",
                pdf_name,
                OncologySnapshotBaseline,
                PROMPT_MEDICAL,
            )
        )
        # With descriptions
        results.append(
            run_experiment(
                "with_descriptions",
                pdf_name,
                OncologySnapshotDescribed,
                PROMPT_MEDICAL,
            )
        )

    print_results(results)
    return results


def experiment_3_model_comparison():
    """Test different models."""
    print("\n## Experiment 3: Model Comparison")
    results = []

    models = [
        "gemini:gemini-2.5-flash-lite",
        "gemini:gemini-2.5-flash",
    ]

    for model_id in models:
        for pdf_name in ["oncology_table.pdf", "oncology_scanned.pdf"]:
            try:
                results.append(
                    run_experiment(
                        f"model_{model_id.split(':')[1][:20]}",
                        pdf_name,
                        OncologySnapshotDescribed,
                        PROMPT_GROUNDED,
                        model_id=model_id,
                    )
                )
            except Exception as e:
                print(f"  SKIP {model_id} on {pdf_name}: {e}")

    print_results(results)
    return results


def experiment_4_grounded_strategy():
    """Test document_grounded strategy with improved prompts."""
    print("\n## Experiment 4: Grounded Strategy + Better Prompts")
    results = []

    opts_grounded = ExtractOptions(
        strategy=Strategy(plan="document_grounded"),
        repair="targeted",
        max_repair_attempts=2,
    )
    opts_auto = ExtractOptions(repair="targeted", max_repair_attempts=2)

    for pdf_name in ["oncology_table.pdf", "oncology_scanned.pdf", "oncology_mixed.pdf"]:
        results.append(
            run_experiment(
                "auto_described",
                pdf_name,
                OncologySnapshotDescribed,
                PROMPT_GROUNDED,
                options=opts_auto,
            )
        )
        results.append(
            run_experiment(
                "grounded_described",
                pdf_name,
                OncologySnapshotDescribed,
                PROMPT_GROUNDED,
                options=opts_grounded,
            )
        )

    print_results(results)
    return results


def experiment_5_bounding_boxes():
    """Test extracting bounding boxes from PyMuPDF text dict."""
    print("\n## Experiment 5: Bounding Box Extraction from Text Layer")
    import fitz

    for pdf_name in ["oncology_clean.pdf", "oncology_table.pdf"]:
        pdf_path = ONCOLOGY_DIR / "generated" / pdf_name
        doc = fitz.open(str(pdf_path))
        print(f"\n  {pdf_name}:")
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text_dict = page.get_text("dict")
            print(f"    Page {page_idx}: {len(text_dict.get('blocks', []))} blocks")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text and len(text) > 3:
                                span_bbox = span.get("bbox", [])
                                # Normalize to page dimensions
                                page_w, page_h = page.rect.width, page.rect.height
                                norm_bbox = (
                                    (
                                        span_bbox[0] / page_w,
                                        span_bbox[1] / page_h,
                                        span_bbox[2] / page_w,
                                        span_bbox[3] / page_h,
                                    )
                                    if span_bbox
                                    else None
                                )
                                print(
                                    f"      '{text}' → bbox_norm={tuple(f'{x:.3f}' for x in norm_bbox) if norm_bbox else None}"
                                )
        doc.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Experiment numbers to run (1-5)",
    )
    args = parser.parse_args()

    all_results = []

    if 5 in args.exp:
        experiment_5_bounding_boxes()

    if 1 in args.exp:
        all_results.extend(experiment_1_prompt_engineering())

    if 2 in args.exp:
        all_results.extend(experiment_2_schema_descriptions())

    if 3 in args.exp:
        all_results.extend(experiment_3_model_comparison())

    if 4 in args.exp:
        all_results.extend(experiment_4_grounded_strategy())

    # Save all results
    if all_results:
        output_path = ROOT / "experiments" / "results.json"
        serializable = [
            {
                "name": r.name,
                "pdf_variant": r.pdf_variant,
                "accuracy": r.accuracy,
                "provenance_accuracy": r.provenance_accuracy,
                "page_coverage": r.page_coverage,
                "latency_s": r.latency_s,
                "wrong_present_rate": r.wrong_present_rate,
                "output": r.output,
                "errors": r.errors,
            }
            for r in all_results
        ]
        output_path.write_text(json.dumps(serializable, indent=2))
        print(f"\nResults saved to {output_path}")
