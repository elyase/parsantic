from __future__ import annotations

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
TRUTH = json.loads((ROOT / "ground_truth.json").read_text())


def _clean_pages() -> tuple[str, ...]:
    return (
        (
            "Initial Presentation\n\n"
            "56 year old female with a history of hyperlipidemia and tobacco abuse.\n"
            "Patient initially noted a growth in her nose in 11/2025 associated with increased "
            "difficulty breathing through her nose.\n"
            "CT sinus without contrast on 12/21/25 showed a polypoid lesion in the anterior left "
            "naris measuring 1.8 x 1.4 cm and abuts the septum.\n\n"
            "ENT exam revealed nasal fullness of the left lower lateral cartilage. "
            "Anterior rhinoscopy showed a mass from the left nasal ala and lateral nasal sidewall "
            "filling the anterior nasal cavity with the septum bowed to the right. "
            "Lesion was pink and purple with overlying telangiectasias.\n"
        ),
        (
            "MRI / PET Imaging\n\n"
            "MRI face on 1/15/26 revealed a 3.0 x 1.3 x 2.5 cm well-circumscribed anterior "
            "nasal cavity mass abutting the lateral nasal wall and septum. "
            "Left submandibular lymph node measured up to 1.3 cm and was indeterminate.\n\n"
            "PET/CT at Touchstone on 2/3/26 showed a hypermetabolic mass in the left nasal cavity, "
            "consistent with known melanoma. No PET evidence for metastatic disease. "
            "Millimetric left apical lung nodule was present below PET resolution and recommended "
            "for follow-up.\n"
        ),
        (
            "Surgery And Pathology\n\n"
            "Patient underwent left partial rhinectomy, left maxillectomy, and cervical "
            "lymphadenectomy on 2/16/26.\n"
            "Procedure described a massive mucosal melanoma of the left nasal cavity involving "
            "the piriform aperture, nasal floor mucosa, inferior turbinate head, inferior meatus "
            "mucosa, and lateral nasal wall. No obvious invasion of septum, columella, or middle turbinate.\n\n"
            "Pathology was positive for mucosal malignant melanoma of the left paranasal sinus, "
            "maxillary, measuring 3.1 cm. Lymphovascular and perineural invasion were not "
            "identified. Margins were negative. Regional lymph nodes uninvolved, 0/27. "
            "Pathologic stage: pT4a pN0.\n"
        ),
    )


def _table_pages() -> tuple[str, ...]:
    return (
        (
            "Clinical Timeline\n\n"
            "Item                         | Value\n"
            "---------------------------- | --------------------------------------------\n"
            f"Age                          | {TRUTH['age_years']}\n"
            f"Sex                          | {TRUTH['sex']}\n"
            f"Comorbidities                | {', '.join(TRUTH['comorbidities'])}\n"
            f"Symptom onset                | {TRUTH['presenting_symptom_month']}\n"
            f"Primary site                 | {TRUTH['primary_site']}\n"
            f"Diagnosis                    | {TRUTH['diagnosis']}\n"
            f"CT sinus date                | {TRUTH['ct_sinus_date']}\n"
            f"CT lesion size (cm)          | {TRUTH['ct_lesion_size_cm'][0]} x {TRUTH['ct_lesion_size_cm'][1]}\n"
            f"MRI date                     | {TRUTH['mri_face_date']}\n"
            f"MRI mass size (cm)           | {TRUTH['mri_mass_size_cm'][0]} x {TRUTH['mri_mass_size_cm'][1]} x {TRUTH['mri_mass_size_cm'][2]}\n"
            f"Left submandibular node (cm) | {TRUTH['left_submandibular_node_cm']}\n"
            f"PET/CT date                  | {TRUTH['pet_ct_date']}\n"
        ),
        (
            "Operative / Pathology Summary\n\n"
            "Finding                      | Value\n"
            "---------------------------- | --------------------------------------------\n"
            f"Hypermetabolic nasal mass    | {TRUTH['pet_hypermetabolic_nasal_mass']}\n"
            f"Metastatic disease on PET    | {TRUTH['metastatic_disease_on_pet']}\n"
            f"Left apical lung nodule      | {TRUTH['left_apical_lung_nodule_present']}\n"
            f"Surgery date                 | {TRUTH['surgery_date']}\n"
            f"Procedures                   | {', '.join(TRUTH['surgery_procedures'])}\n"
            f"Pathology size (cm)          | {TRUTH['pathology_size_cm']}\n"
            f"Margins negative             | {TRUTH['margins_negative']}\n"
            f"Lymphovascular invasion      | {TRUTH['lymphovascular_invasion']}\n"
            f"Perineural invasion          | {TRUTH['perineural_invasion']}\n"
            f"Nodes positive               | {TRUTH['nodes_positive']}\n"
            f"Nodes examined               | {TRUTH['nodes_examined']}\n"
            f"Pathologic T stage           | {TRUTH['pathologic_t_stage']}\n"
            f"Pathologic N stage           | {TRUTH['pathologic_n_stage']}\n"
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
    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    clean = _clean_pages()
    table = _table_pages()
    _save_text_pdf(generated / "nasal_melanoma_clean.pdf", clean)
    _save_text_pdf(generated / "nasal_melanoma_table.pdf", table)
    _save_scanned_pdf(generated / "nasal_melanoma_scanned.pdf", clean)
    _save_scanned_pdf(generated / "nasal_melanoma_mixed.pdf", clean, mixed=True)


if __name__ == "__main__":
    main()
