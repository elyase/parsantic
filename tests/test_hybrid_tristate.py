from __future__ import annotations

from pydantic import BaseModel

from parsantic.extract.diagnostics import FieldDiagnostic, FieldState
from parsantic.extract.options import FieldScopePolicy
from parsantic.extract.pipeline import _DocumentState, _merge_hybrid_states


class EmptyPayload(BaseModel):
    tags: list[str] | None = None
    notes: str | None = None
    metadata: dict[str, str] | None = None


def test_merge_hybrid_treats_empty_values_without_diagnostics_as_missing():
    page_state = _DocumentState(
        merged_value={"tags": [], "notes": "", "metadata": {}},
    )
    whole_state = _DocumentState(
        merged_value={"tags": ["oncology"], "notes": "present", "metadata": {"site": "A"}},
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
    )

    assert EmptyPayload.model_validate(merged.merged_value) == EmptyPayload(
        tags=["oncology"],
        notes="present",
        metadata={"site": "A"},
    )


def test_merge_hybrid_preserves_intentional_empty_values_with_diagnostics():
    empty = FieldDiagnostic(state=FieldState.EMPTY, source="page", confidence=1.0)
    page_state = _DocumentState(
        merged_value={"tags": [], "notes": "", "metadata": {}},
        diagnostics={
            "/tags": empty,
            "/notes": empty,
            "/metadata": empty,
        },
    )
    whole_state = _DocumentState(
        merged_value={"tags": ["oncology"], "notes": "present", "metadata": {"site": "A"}},
        diagnostics={
            "/tags": FieldDiagnostic(state=FieldState.PRESENT, source="document", confidence=1.0),
            "/notes": FieldDiagnostic(state=FieldState.PRESENT, source="document", confidence=1.0),
            "/metadata": FieldDiagnostic(
                state=FieldState.PRESENT,
                source="document",
                confidence=1.0,
            ),
        },
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
    )

    assert EmptyPayload.model_validate(merged.merged_value) == EmptyPayload(
        tags=[],
        notes="",
        metadata={},
    )
    assert merged.diagnostics["/tags"].state == FieldState.EMPTY
    assert merged.diagnostics["/notes"].state == FieldState.EMPTY
    assert merged.diagnostics["/metadata"].state == FieldState.EMPTY
