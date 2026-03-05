from __future__ import annotations

import re

from parsantic.extract.types import AlignmentStatus, ExtractResult, FieldEvidence
from parsantic.extract.visualization import _PALETTE, _build_highlighted_html, visualize


def _ev(
    path: str,
    value_preview: str,
    char_interval: tuple[int, int] | None,
    status: AlignmentStatus,
    *,
    source: str = "text",
) -> FieldEvidence:
    return FieldEvidence(
        path=path,
        value_preview=value_preview,
        char_interval=char_interval,
        token_interval=None,
        alignment_status=status,
        source=source,
    )


def _result(
    *,
    document_id: str | None,
    raw_text: str | None,
    evidence: list[FieldEvidence],
) -> ExtractResult[dict[str, str]]:
    return ExtractResult(
        value={"status": "ok"},
        document_id=document_id,
        raw_text=raw_text,
        flags=(),
        score=100,
        evidence=evidence,
    )


def test_visualize_simple_exact_matches():
    raw_text = "Ada Lovelace lives in London."
    result = _result(
        document_id="doc-1",
        raw_text=raw_text,
        evidence=[
            _ev("/name", "Ada Lovelace", (0, 12), AlignmentStatus.MATCH_EXACT),
            _ev("/city", "London", (22, 28), AlignmentStatus.MATCH_EXACT),
        ],
    )

    html = visualize(result)

    assert "<html" in html.lower()
    assert "<body" in html.lower()
    assert "</body>" in html.lower()
    assert "</html>" in html.lower()
    assert 'data-field-path="/name"' in html
    assert 'data-field-path="/city"' in html
    assert "match_exact" in html
    assert "/name" in html
    assert _PALETTE[0] in html


def test_visualize_fuzzy_and_unmatched_evidence():
    result = _result(
        document_id="doc-fuzzy",
        raw_text="Name: Ada",
        evidence=[
            _ev("/name", "Ada", (6, 9), AlignmentStatus.MATCH_FUZZY),
            _ev("/email", "unknown", None, AlignmentStatus.UNMATCHED, source="vision"),
        ],
    )

    html = visualize(result)

    assert "match_fuzzy" in html
    assert "unmatched" in html
    assert "Evidence Without Character Intervals" in html
    assert "doc-fuzzy" in html
    assert "No evidence without character intervals." not in html


def test_visualize_multiple_results_batch():
    result_a = _result(
        document_id="doc-a",
        raw_text="Alpha",
        evidence=[_ev("/name", "Alpha", (0, 5), AlignmentStatus.MATCH_EXACT)],
    )
    result_b = _result(
        document_id="doc-b",
        raw_text="Beta",
        evidence=[_ev("/name", "Beta", (0, 4), AlignmentStatus.MATCH_EXACT)],
    )

    html = visualize([result_a, result_b])

    assert "Result 1" in html
    assert "Result 2" in html
    assert "doc-a" in html
    assert "doc-b" in html
    assert html.count('class="pv-card pv-result"') == 2


def test_visualize_output_path_writes_file(tmp_path):
    result = _result(
        document_id="doc-path",
        raw_text="Ada",
        evidence=[_ev("/name", "Ada", (0, 3), AlignmentStatus.MATCH_EXACT)],
    )
    output_path = tmp_path / "visualization.html"

    rendered = visualize(result, output_path=output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == rendered


def test_visualize_empty_evidence_produces_valid_html():
    result = _result(document_id=None, raw_text="Nothing to align", evidence=[])

    html = visualize(result)

    assert "No field evidence to highlight." in html
    assert "<html" in html.lower()
    assert "<body" in html.lower()
    assert "</body>" in html.lower()
    assert "</html>" in html.lower()


def test_build_highlighted_html_handles_overlaps_with_nested_spans():
    highlighted = _build_highlighted_html(
        "ABCDE",
        [
            _ev("/outer", "ABCD", (0, 4), AlignmentStatus.MATCH_EXACT),
            _ev("/inner", "CDE", (2, 5), AlignmentStatus.MATCH_EXACT),
        ],
        _PALETTE,
    )

    assert 'data-field-path="/outer"' in highlighted
    assert 'data-field-path="/inner"' in highlighted
    nested = re.compile(
        r'data-field-path="/outer"[^>]*><span class="pv-highlight"[^>]*data-field-path="/inner"[^>]*>CD</span></span>'
    )
    assert nested.search(highlighted)


def test_visualize_contains_css_classes_and_tooltip_data_attributes():
    result = _result(
        document_id="doc-css",
        raw_text="Ada",
        evidence=[_ev("/name", "Ada", (0, 3), AlignmentStatus.MATCH_EXACT)],
    )

    html = visualize(result)

    assert ".pv-highlight" in html
    assert "pv-text-container" in html
    assert "data-tooltip=" in html
    assert 'data-field-value="Ada"' in html
    assert 'data-alignment-status="match_exact"' in html
    assert "pv-tooltip" in html
