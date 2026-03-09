from __future__ import annotations

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
GENERATED = ROOT / "generated"
TRUTH = json.loads((ROOT / "ground_truth.json").read_text())


def _page_texts() -> tuple[str, ...]:
    observations = "\n".join(
        f"{item['code']['text']}: {item['valueQuantity']['value']} {item['valueQuantity']['unit']}"
        for item in TRUTH["observations"]
    )
    medications = "\n\n".join(
        f"Medication: {item['medicationCodeableConcept']['text']}\n"
        f"Status: {item['status']}\n"
        f"Dose: {item['dosageInstruction']}\n"
        f"Route: {item['route']}"
        for item in TRUTH["medications"]
    )
    return (
        (
            "Oncology Summary\n\n"
            f"Patient: {TRUTH['patient']['name']}\n"
            f"Identifier: {TRUTH['patient']['identifier']}\n"
            f"Birth Date: {TRUTH['patient']['birthDate']}\n"
            f"Gender: {TRUTH['patient']['gender']}\n\n"
            f"Encounter Date: {TRUTH['encounter']['periodStart']}\n"
            f"Oncologist: {TRUTH['encounter']['participant']}\n"
            f"Center: {TRUTH['encounter']['serviceProvider']}\n\n"
            f"Diagnosis: {TRUTH['condition']['code']['text']}\n"
            f"Stage: {TRUTH['condition']['stage']}\n"
            f"Body Site: {TRUTH['condition']['bodySite']}\n"
        ),
        (
            "Diagnostic Report\n\n"
            f"Report Type: {TRUTH['diagnosticReport']['code']['text']}\n"
            f"Effective: {TRUTH['diagnosticReport']['effectiveDateTime']}\n"
            f"Conclusion: {TRUTH['diagnosticReport']['conclusion']}\n"
        ),
        "Laboratory Results\n\n" + observations,
        "Medication List\n\n" + medications,
        (
            "Care Plan\n\n"
            f"Status: {TRUTH['carePlan']['status']}\n"
            f"Intent: {TRUTH['carePlan']['intent']}\n"
            f"Description: {TRUTH['carePlan']['description']}\n"
        ),
    )


def _table_pages() -> tuple[str, ...]:
    observation_rows = "\n".join(
        f"{item['code']['text']:<28} | {item['effectiveDateTime']} | {item['valueQuantity']['value']} | {item['valueQuantity']['unit']}"
        for item in TRUTH["observations"]
    )
    medication_rows = "\n".join(
        f"{item['medicationCodeableConcept']['text']:<18} | {item['status']:<6} | {item['route']:<12} | {item['dosageInstruction']}"
        for item in TRUTH["medications"]
    )
    return (
        (
            "Oncology Intake Table\n\n"
            "Field                        | Value\n"
            "---------------------------- | -----------------------------------------\n"
            f"Patient                      | {TRUTH['patient']['name']}\n"
            f"Identifier                   | {TRUTH['patient']['identifier']}\n"
            f"Birth Date                   | {TRUTH['patient']['birthDate']}\n"
            f"Oncologist                   | {TRUTH['encounter']['participant']}\n"
            f"Service Provider             | {TRUTH['encounter']['serviceProvider']}\n"
            f"Diagnosis                    | {TRUTH['condition']['code']['text']}\n"
            f"Stage                        | {TRUTH['condition']['stage']}\n"
        ),
        (
            "Observation Table\n\n"
            "Observation                  | Effective   | Value | Unit\n"
            "---------------------------- | ----------- | ----- | ---------\n"
            f"{observation_rows}\n"
        ),
        (
            "Medication Table\n\n"
            "Medication           | Status | Route        | Dosage\n"
            "-------------------- | ------ | ------------ | ----------------------------------------------\n"
            f"{medication_rows}\n\n"
            f"Care Plan: {TRUTH['carePlan']['description']}\n"
        ),
    )


def _add_text_page(pdf: fitz.Document, body: str) -> None:
    page = pdf.new_page(width=612, height=792)
    page.insert_textbox(
        fitz.Rect(48, 48, 564, 744),
        body,
        fontname="helv",
        fontsize=11,
        lineheight=1.3,
    )


def _save_text_pdf(path: Path, pages: tuple[str, ...]) -> None:
    pdf = fitz.open()
    for body in pages:
        _add_text_page(pdf, body)
    pdf.save(path)
    pdf.close()


def _save_scanned_pdf(path: Path, pages: tuple[str, ...], *, mixed: bool = False) -> None:
    text_pdf = fitz.open()
    for body in pages:
        _add_text_page(text_pdf, body)

    scanned = fitz.open()
    for index, page in enumerate(text_pdf):
        target = scanned.new_page(width=612, height=792)
        if mixed and index % 2 == 0:
            target.insert_textbox(
                fitz.Rect(48, 48, 564, 744),
                pages[index],
                fontname="helv",
                fontsize=11,
                lineheight=1.3,
            )
            continue
        pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
        target.insert_image(target.rect, stream=pix.tobytes("png"))
    scanned.save(path)
    scanned.close()
    text_pdf.close()


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    narrative = _page_texts()
    table = _table_pages()
    _save_text_pdf(GENERATED / "oncology_clean.pdf", narrative)
    _save_text_pdf(GENERATED / "oncology_table.pdf", table)
    _save_scanned_pdf(GENERATED / "oncology_scanned.pdf", narrative)
    _save_scanned_pdf(GENERATED / "oncology_mixed.pdf", narrative, mixed=True)


if __name__ == "__main__":
    main()
