from __future__ import annotations

from parsantic.extract.options import FieldScopePolicy
from parsantic.extract.pipeline import _DocumentState, _merge_hybrid_states


def test_merge_hybrid_list_matching_is_order_invariant_for_ambiguous_identities():
    page_state = _DocumentState(
        merged_value={
            "medications": [
                {
                    "medicationCodeableConcept": {"text": "Capecitabine"},
                    "route": "oral",
                },
                {
                    "medicationCodeableConcept": {"text": "Capecitabine"},
                    "route": "intravenous",
                },
            ]
        }
    )
    whole_state = _DocumentState(
        merged_value={
            "medications": [
                {"medicationCodeableConcept": {"text": " Capecitabine "}},
                {"medicationCodeableConcept": {"text": "CAPECITABINE"}},
                {"medicationCodeableConcept": {"text": "capecitabine"}},
            ]
        }
    )

    merged = _merge_hybrid_states(
        page_state=page_state,
        whole_state=whole_state,
        field_scope=FieldScopePolicy(),
    )

    assert merged.merged_value == {
        "medications": [
            {
                "medicationCodeableConcept": {"text": "Capecitabine"},
                "route": "oral",
            },
            {
                "medicationCodeableConcept": {"text": "Capecitabine"},
                "route": "intravenous",
            },
            {
                "medicationCodeableConcept": {"text": "capecitabine"},
            },
        ]
    }
