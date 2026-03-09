from __future__ import annotations

import pytest

from parsantic.extract import ExtractOptions
from parsantic.extract.options import (
    FieldScopePolicy,
    MediaOptions,
    ProvenancePolicy,
    Strategy,
)


def test_extract_options_strategy_preset_native_resolves_to_runtime_media():
    options = ExtractOptions(strategy="native")

    resolved = options.resolve_runtime_strategy()

    assert resolved.represent == "native"
    assert resolved.plan == "whole_document"
    assert resolved.media == MediaOptions(
        pdf_mode="native", page_strategy="single", grounding="off"
    )
    assert resolved.provenance == ProvenancePolicy(mode="sidecar", cite_by="none", strict=False)


def test_extract_options_document_mode_uses_whole_document_runtime():
    options = ExtractOptions(mode="document", document_input="native")

    resolved = options.resolve_runtime_strategy()

    assert resolved.mode == "document"
    assert resolved.plan == "whole_document"
    assert resolved.document_input == "native"
    assert resolved.page_input == "auto"
    assert resolved.media == MediaOptions(
        pdf_mode="native", page_strategy="single", grounding="off"
    )
    assert resolved.document_media == resolved.media
    assert resolved.page_media is None


def test_extract_options_page_mode_uses_page_runtime():
    options = ExtractOptions(mode="page")

    resolved = options.resolve_runtime_strategy()

    assert resolved.mode == "page"
    assert resolved.plan == "page_map_reduce"
    assert resolved.page_input == "auto"
    assert resolved.document_input == "auto"
    assert resolved.media == MediaOptions(
        pdf_mode="raster",
        page_strategy="map_reduce",
        grounding="off",
    )
    assert resolved.page_media == resolved.media
    assert resolved.document_media is None


def test_extract_options_hybrid_mode_splits_document_and_page_inputs():
    options = ExtractOptions(
        mode="hybrid",
        document_input="native",
        page_input="image",
    )

    resolved = options.resolve_runtime_strategy()

    assert resolved.mode == "hybrid"
    assert resolved.plan == "hybrid"
    assert resolved.document_input == "native"
    assert resolved.page_input == "image"
    assert resolved.media == MediaOptions(
        pdf_mode="raster",
        page_strategy="map_reduce",
        grounding="off",
    )
    assert resolved.document_media == MediaOptions(
        pdf_mode="native",
        page_strategy="single",
        grounding="off",
    )
    assert resolved.page_media == MediaOptions(
        pdf_mode="raster",
        page_strategy="map_reduce",
        grounding="off",
    )


def test_extract_options_strategy_preset_auditable_uses_page_aware_runtime_without_structural_claims():
    options = ExtractOptions(strategy="auditable")

    resolved = options.resolve_runtime_strategy()

    assert resolved.represent == "raster"
    assert resolved.plan == "page_map_reduce"
    assert resolved.media == MediaOptions(
        pdf_mode="raster",
        page_strategy="map_reduce",
        grounding="off",
    )
    assert resolved.provenance == ProvenancePolicy(
        mode="sidecar",
        cite_by="none",
        strict=False,
    )


def test_document_grounded_strategy_resolves_to_document_runtime_with_identity_keys():
    options = ExtractOptions(
        strategy=Strategy(
            plan="document_grounded",
            identity_keys={"/line_items": "code"},
        )
    )

    resolved = options.resolve_runtime_strategy()

    assert resolved.plan == "document_grounded"
    assert resolved.mode == "document"
    assert resolved.media == MediaOptions(
        pdf_mode="auto",
        page_strategy="single",
        grounding="off",
    )
    assert resolved.identity_keys == {"/line_items": ("/code",)}


def test_fused_strategy_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        resolved = ExtractOptions(strategy=Strategy(plan="fused")).resolve_runtime_strategy()

    assert resolved.plan == "fused"


def test_strategy_representation_fallback_skips_unsupported_backends_with_warning():
    options = ExtractOptions(
        strategy=Strategy(
            represent=("docling", "raster", "native"),
            plan="page_map_reduce",
        )
    )

    with pytest.warns(UserWarning, match="docling"):
        resolved = options.resolve_runtime_strategy()

    assert resolved.represent == "raster"
    assert resolved.media.pdf_mode == "raster"
    assert resolved.media.page_strategy == "map_reduce"


def test_hybrid_representation_fallback_skips_incompatible_native_and_uses_raster():
    options = ExtractOptions(
        strategy=Strategy(
            represent=("docling", "native", "raster"),
            plan="hybrid",
        )
    )

    with pytest.warns(UserWarning, match="docling, native"):
        resolved = options.resolve_runtime_strategy()

    assert resolved.represent == "raster"
    assert resolved.media.pdf_mode == "raster"
    assert resolved.plan == "hybrid"


def test_extract_options_strategy_disallows_mixing_with_custom_legacy_media():
    with pytest.raises(ValueError, match="strategy cannot be combined with custom media options"):
        ExtractOptions(
            strategy="native",
            media=MediaOptions(pdf_mode="raster"),
        )


def test_extract_options_mode_requires_mode_for_input_overrides():
    with pytest.raises(ValueError, match="require mode"):
        ExtractOptions(document_input="native")


def test_extract_options_document_mode_rejects_page_input_override():
    with pytest.raises(ValueError, match="page_input"):
        ExtractOptions(mode="document", page_input="image")


def test_extract_options_page_mode_rejects_document_input_override():
    with pytest.raises(ValueError, match="document_input"):
        ExtractOptions(mode="page", document_input="native")


def test_extract_options_mode_disallows_mixing_with_strategy():
    with pytest.raises(ValueError, match="cannot be combined with strategy"):
        ExtractOptions(mode="hybrid", strategy="native")


def test_extract_options_mode_disallows_mixing_with_custom_legacy_media():
    with pytest.raises(ValueError, match="cannot be combined with custom media options"):
        ExtractOptions(
            mode="hybrid",
            document_input="native",
            media=MediaOptions(pdf_mode="native"),
        )


def test_field_scope_policy_uses_explicit_path_then_default():
    policy = FieldScopePolicy(
        default_scope="auto",
        by_path={
            "/vendor": "global",
            "/line_items": "span",
        },
    )

    assert policy.scope_for("/vendor") == "global"
    assert policy.scope_for("/line_items") == "span"
    assert policy.scope_for("/total") == "auto"


def test_field_scope_policy_supports_nested_and_wildcard_paths():
    policy = FieldScopePolicy(
        default_scope="auto",
        by_path={
            "/patient": "global",
            "/patient/dob": "local",
            "/line_items/*/amount": "local",
            "/line_items/*/description": "global",
        },
    )

    assert policy.scope_for("/patient/name") == "global"
    assert policy.scope_for("/patient/dob") == "local"
    assert policy.scope_for("/line_items/0/amount") == "local"
    assert policy.scope_for("/line_items/4/description") == "global"
    assert policy.scope_for("/line_items/4/code") == "auto"
    assert policy.has_descendant_rule("/patient") is True
    assert policy.has_descendant_rule("/line_items") is True
    assert policy.has_descendant_rule("/total") is False


def test_retrieve_strategy_is_rejected_until_runtime_support_exists():
    options = ExtractOptions(strategy="retrieve")

    with pytest.raises(NotImplementedError, match="retrieve"):
        options.resolve_runtime_strategy()


def test_hybrid_strategy_resolves_to_page_aware_runtime_media():
    options = ExtractOptions(
        strategy=Strategy(
            plan="hybrid",
            field_scope=FieldScopePolicy(by_path={"/vendor": "global"}),
        )
    )

    resolved = options.resolve_runtime_strategy()

    assert resolved.represent == "raster"
    assert resolved.plan == "hybrid"
    assert resolved.media == MediaOptions(
        pdf_mode="raster",
        page_strategy="map_reduce",
        grounding="off",
    )
    assert resolved.field_scope.scope_for("/vendor") == "global"


def test_explicit_structural_provenance_request_falls_back_until_grounding_exists():
    options = ExtractOptions(
        strategy=Strategy(
            plan="hybrid",
            provenance=ProvenancePolicy(mode="sidecar", cite_by="structural", strict=False),
        )
    )

    with pytest.warns(UserWarning, match="Structural provenance"):
        resolved = options.resolve_runtime_strategy()

    assert resolved.provenance == ProvenancePolicy(mode="sidecar", cite_by="none", strict=False)
    assert resolved.media.grounding == "off"


def test_hybrid_strategy_rejects_native_representation():
    options = ExtractOptions(
        strategy=Strategy(
            represent="native",
            plan="hybrid",
        )
    )

    with pytest.raises(NotImplementedError, match="native"):
        options.resolve_runtime_strategy()


def test_hybrid_strategy_allows_nested_field_scope_paths():
    options = ExtractOptions(
        strategy=Strategy(
            plan="hybrid",
            field_scope=FieldScopePolicy(
                by_path={
                    "/line_items/*/description": "local",
                    "/patient/dob": "global",
                }
            ),
        )
    )

    resolved = options.resolve_runtime_strategy()

    assert resolved.plan == "hybrid"
    assert resolved.field_scope.scope_for("/line_items/0/description") == "local"
    assert resolved.field_scope.scope_for("/patient/dob") == "global"
