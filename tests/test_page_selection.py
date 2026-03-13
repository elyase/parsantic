from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from parsantic.extract import Document, ExtractOptions, Strategy, extract, select_pdf_pages
from parsantic.extract.media import PageQuality, PreflightResult, analyze_pdf_source


class LabsOnly(BaseModel):
    hemoglobin_g_dl: float = Field(description="Hemoglobin lab value in g/dL")
    creatinine_mg_dl: float = Field(description="Creatinine lab value in mg/dL")


class MedicationOnly(BaseModel):
    primary_medication: str = Field(description="Primary chemotherapy medication name")


class WeakSchema(BaseModel):
    aa: str = ""
    bb: str = ""


class BroadSchema(BaseModel):
    field_01: str = Field(description="alpha")
    field_02: str = Field(description="beta")
    field_03: str = Field(description="gamma")
    field_04: str = Field(description="delta")
    field_05: str = Field(description="epsilon")
    field_06: str = Field(description="zeta")
    field_07: str = Field(description="eta")
    field_08: str = Field(description="theta")
    field_09: str = Field(description="iota")
    field_10: str = Field(description="kappa")
    field_11: str = Field(description="lambda")
    field_12: str = Field(description="mu")
    field_13: str = Field(description="nu")


class NestedOuter(BaseModel):
    labs: LabsOnly


def _analysis(previews: list[str], *, quality: float = 0.9) -> PreflightResult:
    return PreflightResult(
        page_count=len(previews),
        pages=[
            PageQuality(
                page_index=index,
                text_char_count=len(preview),
                text_quality_score=quality,
                text_preview=preview,
                has_tables=False,
                has_images=False,
                is_scanned=False,
                recommended_mode="text_only",
            )
            for index, preview in enumerate(previews)
        ],
        has_text_layer=True,
        text_layer_quality=quality,
        recommended_plan="text_only",
        estimated_tokens=42,
    )


def test_select_pdf_pages_picks_strict_subset_for_obvious_cues():
    analysis = _analysis(
        [
            "Administrative cover page and appointment notes",
            "Laboratory results hemoglobin g/dL and creatinine mg/dL",
            "Insurance appendix and parking instructions",
        ]
    )

    selection = select_pdf_pages(analysis, LabsOnly, window=0)

    assert selection.page_indices == (1,)
    assert selection.fallback_reason is None
    assert selection.selected_page_count == 1
    assert selection.reason_codes_by_page[1]


def test_select_pdf_pages_falls_back_for_broad_schema():
    analysis = _analysis(
        [
            "alpha beta gamma delta epsilon",
            "lambda mu nu theta iota kappa",
            "appendix page",
        ]
    )

    selection = select_pdf_pages(analysis, BroadSchema)

    assert selection.page_indices is None
    assert selection.fallback_reason == "broad_schema"
    assert selection.selected_page_count == analysis.page_count


def test_select_pdf_pages_expands_neighbor_pages():
    analysis = _analysis(
        [
            "Administrative intro",
            "Primary chemotherapy medication name: capecitabine",
            "Continuation of dosing instructions",
            "Billing appendix",
        ]
    )

    selection = select_pdf_pages(analysis, MedicationOnly, window=1)

    assert selection.page_indices == (0, 1, 2)


def test_select_pdf_pages_falls_back_when_too_many_pages_match():
    analysis = _analysis(
        [
            "Primary chemotherapy medication name and dose",
            "Primary chemotherapy medication name and route",
            "Primary chemotherapy medication name and status",
            "Primary chemotherapy medication name and refill",
            "Primary chemotherapy medication name and scheduling",
        ]
    )

    selection = select_pdf_pages(analysis, MedicationOnly, window=0)

    assert selection.page_indices is None
    assert selection.fallback_reason == "too_many_matches"


def test_select_pdf_pages_falls_back_for_low_text_quality():
    analysis = _analysis(
        [
            "Hemoglobin creatinine",
            "Medication capecitabine",
        ],
        quality=0.1,
    )

    selection = select_pdf_pages(analysis, LabsOnly)

    assert selection.page_indices is None
    assert selection.fallback_reason == "low_text_quality"


def test_select_pdf_pages_is_deterministic():
    analysis = _analysis(
        [
            "Administrative page",
            "Laboratory results hemoglobin g/dL and creatinine mg/dL",
            "Medication page",
        ]
    )

    first = select_pdf_pages(analysis, LabsOnly, window=0)
    second = select_pdf_pages(analysis, LabsOnly, window=0)

    assert first == second


def test_select_pdf_pages_accepts_type_adapter_input():
    analysis = _analysis(
        [
            "Administrative cover page and appointment notes",
            "Laboratory results hemoglobin g/dL and creatinine mg/dL",
        ]
    )

    selection = select_pdf_pages(analysis, TypeAdapter(LabsOnly), window=0)

    assert selection.page_indices == (1,)


def test_select_pdf_pages_traverses_nested_schema_refs():
    analysis = _analysis(
        [
            "Administrative cover page and appointment notes",
            "Laboratory results hemoglobin g/dL and creatinine mg/dL",
        ]
    )

    selection = select_pdf_pages(analysis, NestedOuter, window=0)

    assert selection.page_indices == (1,)


@pytest.mark.skipif(pytest.importorskip("fitz") is None, reason="PyMuPDF not installed")
def test_extract_using_selected_page_indices_preserves_correct_result():
    import fitz

    class LabResult(BaseModel):
        body_text: str = Field(description="Page two body laboratory field")

    class _ExactProvider:
        def infer(self, batch_prompts, **kwargs):
            outputs = []
            for prompt in batch_prompts:
                if "Page two body" in prompt:
                    outputs.append('{"body_text": "Page two body"}')
                else:
                    outputs.append('{"body_text": "fallback"}')
            return outputs

    pdf = fitz.open()
    page_one = pdf.new_page()
    page_one.insert_text((72, 72), "Administrative cover page")
    page_two = pdf.new_page()
    page_two.insert_text((72, 72), "Page two body\nHemoglobin g/dL 11.2")
    page_three = pdf.new_page()
    page_three.insert_text((72, 72), "Parking desk and scheduling")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    analysis = analyze_pdf_source(pdf_bytes)
    selection = select_pdf_pages(analysis, LabResult, window=0, max_pages=1)
    assert selection.page_indices == (1,)

    result = extract(
        Document.from_pdf(pdf_bytes, page_indices=selection.page_indices, text="Extract the field"),
        LabResult,
        model=_ExactProvider(),
        options=ExtractOptions(strategy=Strategy(plan="document_grounded")),
    )

    assert result.value.body_text == "Page two body"
