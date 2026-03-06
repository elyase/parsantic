"""Minimal oncology PDF demo with page provenance.

Runs a synthetic oncology patient summary through the recommended hybrid PDF
flow: whole-document native PDF + page images for page-grounded provenance.

Requires:
  uv sync --extra ai --extra vision
  GEMINI_API_KEY env var by default, or set PARSANTIC_MODEL explicitly.

For a side-by-side mode comparison, run `examples/demo_pdf_modes.py`.
For a speed comparison, run `examples/demo_pdf_benchmark.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from parsantic.extract import Document, ExtractOptions, extract


class CodeableConcept(BaseModel):
    text: str = ""


class Quantity(BaseModel):
    value: float = 0.0
    unit: str = ""


class PatientResource(BaseModel):
    resourceType: Literal["Patient"] = "Patient"
    identifier: str = ""
    name: str = ""
    gender: str = ""
    birthDate: str = ""


class ConditionResource(BaseModel):
    resourceType: Literal["Condition"] = "Condition"
    code: CodeableConcept = CodeableConcept()
    clinicalStatus: str = ""
    stage: str = ""


class DiagnosticReportResource(BaseModel):
    resourceType: Literal["DiagnosticReport"] = "DiagnosticReport"
    code: CodeableConcept = CodeableConcept()
    effectiveDateTime: str = ""
    conclusion: str = ""


class ObservationResource(BaseModel):
    resourceType: Literal["Observation"] = "Observation"
    code: CodeableConcept = CodeableConcept()
    effectiveDateTime: str = ""
    valueQuantity: Quantity = Quantity()


class MedicationRequestResource(BaseModel):
    resourceType: Literal["MedicationRequest"] = "MedicationRequest"
    status: str = ""
    medicationCodeableConcept: CodeableConcept = CodeableConcept()
    dosageInstruction: str = ""
    route: str = ""


class CarePlanResource(BaseModel):
    resourceType: Literal["CarePlan"] = "CarePlan"
    status: str = ""
    intent: str = ""
    description: str = ""


class OncologyFHIRBundle(BaseModel):
    resourceType: Literal["Bundle"] = "Bundle"
    type: Literal["collection"] = "collection"
    patient: PatientResource = PatientResource()
    condition: ConditionResource = ConditionResource()
    diagnosticReport: DiagnosticReportResource = DiagnosticReportResource()
    observations: list[ObservationResource] = []
    medications: list[MedicationRequestResource] = []
    carePlan: CarePlanResource = CarePlanResource()


def main() -> None:
    model = os.getenv("PARSANTIC_MODEL", "gemini:gemini-3.1-flash-lite-preview")
    pdf_path = Path(__file__).with_name("sample_oncology_summary.pdf")

    result = extract(
        Document.from_pdf(pdf_path),
        OncologyFHIRBundle,
        model=model,
        options=ExtractOptions(
            mode="hybrid",
            document_input="native",
            page_input="image",
        ),
    )

    print("FHIR-shaped bundle:")
    print(result.value.model_dump_json(indent=2))
    print("\nSelected provenance:")

    selected_paths = [
        ("Patient name", "/patient/name"),
        ("Patient birth date", "/patient/birthDate"),
        ("Diagnosis stage", "/condition/stage"),
        ("Pathology conclusion", "/diagnosticReport/conclusion"),
        ("Care plan", "/carePlan/description"),
    ]

    for index, observation in enumerate(result.value.observations):
        if observation.code.text.strip().lower() == "hemoglobin":
            selected_paths.append(
                ("Latest hemoglobin", f"/observations/{index}/valueQuantity/value")
            )
            break

    for index, medication in enumerate(result.value.medications):
        if medication.medicationCodeableConcept.text.strip().lower() == "capecitabine":
            selected_paths.append(
                ("Primary cancer therapy", f"/medications/{index}/medicationCodeableConcept/text")
            )
            break

    for label, path in selected_paths:
        source = result.sources.get(path)
        if source is None:
            continue
        if source.scope == "page":
            print(f"  {label} ({path}): page {', '.join(str(page) for page in source.pages)}")
        else:
            print(f"  {label} ({path}): document")


if __name__ == "__main__":
    main()
