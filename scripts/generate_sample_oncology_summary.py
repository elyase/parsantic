from __future__ import annotations

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
PDF_PATH = EXAMPLES_DIR / "sample_oncology_summary.pdf"
JSON_PATH = FIXTURES_DIR / "sample_oncology_summary.fhir.json"


BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "patient": {
        "resourceType": "Patient",
        "identifier": "ONC-2026-0142",
        "name": "Maya Hernandez",
        "gender": "female",
        "birthDate": "1978-09-12",
    },
    "encounter": {
        "resourceType": "Encounter",
        "periodStart": "2026-02-18",
        "participant": "Dr. Priya Shah",
        "serviceProvider": "North Valley Cancer Center",
    },
    "condition": {
        "resourceType": "Condition",
        "code": {"text": "Metastatic breast carcinoma"},
        "clinicalStatus": "active",
        "stage": "Stage IV",
        "bodySite": "left breast with liver metastases",
    },
    "diagnosticReport": {
        "resourceType": "DiagnosticReport",
        "code": {"text": "Pathology and biomarker summary"},
        "effectiveDateTime": "2026-02-12",
        "conclusion": (
            "Metastatic adenocarcinoma consistent with breast primary; ER positive, "
            "HER2-low, PD-L1 CPS 12."
        ),
    },
    "observations": [
        {
            "resourceType": "Observation",
            "code": {"text": "Hemoglobin"},
            "effectiveDateTime": "2026-02-18",
            "valueQuantity": {"value": 11.2, "unit": "g/dL"},
        },
        {
            "resourceType": "Observation",
            "code": {"text": "Absolute neutrophil count"},
            "effectiveDateTime": "2026-02-18",
            "valueQuantity": {"value": 2.4, "unit": "x10^9/L"},
        },
        {
            "resourceType": "Observation",
            "code": {"text": "Creatinine"},
            "effectiveDateTime": "2026-02-18",
            "valueQuantity": {"value": 0.9, "unit": "mg/dL"},
        },
        {
            "resourceType": "Observation",
            "code": {"text": "CA 15-3"},
            "effectiveDateTime": "2026-02-18",
            "valueQuantity": {"value": 48.0, "unit": "U/mL"},
        },
    ],
    "medications": [
        {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Capecitabine"},
            "dosageInstruction": "1500 mg orally twice daily on days 1-14 of a 21-day cycle",
            "route": "oral",
        },
        {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Ondansetron"},
            "dosageInstruction": "8 mg every 8 hours as needed for nausea",
            "route": "oral",
        },
        {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": "Zoledronic acid"},
            "dosageInstruction": "4 mg every 28 days",
            "route": "intravenous",
        },
    ],
    "carePlan": {
        "resourceType": "CarePlan",
        "status": "active",
        "intent": "plan",
        "description": (
            "Continue capecitabine, repeat CT chest/abdomen/pelvis in 8 weeks, "
            "and return to clinic on 2026-03-11."
        ),
    },
}


PAGES = (
    (
        "Oncology Daily Summary\n\n"
        "Patient Demographics\n"
        "Name: Maya Hernandez\n"
        "MRN: ONC-2026-0142\n"
        "Date of Birth: 1978-09-12\n"
        "Gender: female\n\n"
        "Encounter\n"
        "Encounter Date: 2026-02-18\n"
        "Primary Oncologist: Dr. Priya Shah\n"
        "Cancer Center: North Valley Cancer Center\n\n"
        "Primary Diagnosis\n"
        "Diagnosis: Metastatic breast carcinoma\n"
        "Clinical Status: active\n"
        "Stage: Stage IV\n"
        "Body Site: left breast with liver metastases\n"
    ),
    (
        "Pathology And Biomarker Report\n\n"
        "Report Date: 2026-02-12\n"
        "Report Type: Pathology and biomarker summary\n"
        "Histology: Invasive ductal carcinoma\n"
        "ER: 90% positive\n"
        "PR: 20% positive\n"
        "HER2: IHC 1+ (HER2-low)\n"
        "PD-L1: CPS 12\n\n"
        "Conclusion: Metastatic adenocarcinoma consistent with breast primary; "
        "ER positive, HER2-low, PD-L1 CPS 12.\n"
    ),
    (
        "Recent Laboratory Results\n\n"
        "Collection Date: 2026-02-18\n"
        "Hemoglobin: 11.2 g/dL\n"
        "Absolute neutrophil count: 2.4 x10^9/L\n"
        "Platelets: 198 x10^9/L\n"
        "Creatinine: 0.9 mg/dL\n"
        "AST: 28 U/L\n"
        "ALT: 31 U/L\n"
        "CA 15-3: 48 U/mL\n"
    ),
    (
        "Active Medication List\n\n"
        "Medication 1\n"
        "Name: Capecitabine\n"
        "Status: active\n"
        "Dose: 1500 mg orally twice daily on days 1-14 of a 21-day cycle\n"
        "Route: oral\n\n"
        "Medication 2\n"
        "Name: Ondansetron\n"
        "Status: active\n"
        "Dose: 8 mg every 8 hours as needed for nausea\n"
        "Route: oral\n\n"
        "Medication 3\n"
        "Name: Zoledronic acid\n"
        "Status: active\n"
        "Dose: 4 mg every 28 days\n"
        "Route: intravenous\n"
    ),
    (
        "Assessment And Plan\n\n"
        "Current Line Of Therapy: second-line capecitabine\n"
        "Treatment Response: stable symptoms, labs acceptable for treatment\n"
        "Care Plan Status: active\n"
        "Care Plan Intent: plan\n"
        "Care Plan Description: Continue capecitabine, repeat CT chest/abdomen/pelvis "
        "in 8 weeks, and return to clinic on 2026-03-11.\n"
    ),
)


def _add_page(pdf: fitz.Document, page_number: int, body: str) -> None:
    page = pdf.new_page(width=612, height=792)
    page.insert_textbox(
        fitz.Rect(48, 48, 564, 744),
        body,
        fontname="helv",
        fontsize=11,
        lineheight=1.3,
    )
    page.insert_text((48, 768), f"Page {page_number}", fontname="helv", fontsize=9)


def main() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open()
    for page_number, body in enumerate(PAGES, start=1):
        _add_page(pdf, page_number, body)
    pdf.save(PDF_PATH)
    pdf.close()

    JSON_PATH.write_text(json.dumps(BUNDLE, indent=2) + "\n")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
