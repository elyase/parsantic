from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from parsantic.json_pointer import parse_json_pointer

from .alignment import AlignmentOptions
from .concurrency import ConcurrencyConfig
from .formatting import FormatOptions
from .prompt import PromptValidationLevel
from .tokenizer import Tokenizer, TokenizerName

if TYPE_CHECKING:
    from .alignment import Resolver

type FieldScope = Literal["auto", "local", "global", "span"]
type StrategyPreset = Literal["native", "balanced", "auditable", "retrieve"]
type ExtractionMode = Literal["auto", "document", "page", "hybrid"]
type DocumentInput = Literal["auto", "native", "image"]
type PageInput = Literal["auto", "image"]
type Representation = Literal[
    "auto",
    "native",
    "text_layer",
    "raster",
    "docling",
    "unstructured",
    "azure_di",
]
type ExecutionPlan = Literal[
    "auto",
    "document_grounded",
    "whole_document",
    "page_map_reduce",
    "page_windows",
    "retrieve_then_extract",
    "hybrid",
    "fused",
]
type ProvenanceMode = Literal["none", "sidecar"]
type CitationKind = Literal["none", "structural", "chunk_id", "offset", "quote", "model"]


@dataclass(slots=True)
class MediaOptions:
    pdf_mode: Literal["auto", "native", "raster"] = "auto"
    raster_dpi: int = 200
    max_image_dim: int = 2048
    page_strategy: Literal["auto", "single", "map_reduce"] = "auto"
    grounding: Literal["off", "auto", "force"] = "auto"
    raster_format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = 85


@dataclass(slots=True)
class ProvenancePolicy:
    mode: ProvenanceMode = "sidecar"
    cite_by: CitationKind = "none"
    strict: bool = False


@dataclass(slots=True)
class FieldScopePolicy:
    default_scope: FieldScope = "auto"
    by_path: dict[str, FieldScope] = field(default_factory=dict)

    def scope_for(self, path: str) -> FieldScope:
        best_scope = self.default_scope
        best_score: tuple[int, int] = (-1, -1)
        path_tokens = _pointer_tokens(path)
        for candidate, scope in self.by_path.items():
            candidate_tokens = _pointer_tokens(candidate)
            if len(candidate_tokens) > len(path_tokens):
                continue
            for candidate_token, path_token in zip(candidate_tokens, path_tokens, strict=False):
                if candidate_token == "*":
                    continue
                if candidate_token != path_token:
                    break
            else:
                score = (
                    sum(1 for token in candidate_tokens if token != "*"),
                    len(candidate_tokens),
                )
                if score > best_score:
                    best_scope = scope
                    best_score = score
        return best_scope

    def has_descendant_rule(self, path: str) -> bool:
        path_tokens = _pointer_tokens(path)
        for candidate in self.by_path:
            candidate_tokens = _pointer_tokens(candidate)
            if len(candidate_tokens) <= len(path_tokens):
                continue
            for candidate_token, path_token in zip(candidate_tokens, path_tokens, strict=False):
                if candidate_token != "*" and candidate_token != path_token:
                    break
            else:
                return True
        return False


def _pointer_tokens(path: str) -> tuple[str, ...]:
    if path in {"", "/"}:
        return ()
    return tuple(parse_json_pointer(path))


@dataclass(slots=True)
class Strategy:
    represent: Representation | tuple[Representation, ...] = "auto"
    plan: ExecutionPlan = "auto"
    provenance: ProvenancePolicy = field(default_factory=ProvenancePolicy)
    field_scope: FieldScopePolicy = field(default_factory=FieldScopePolicy)
    identity_keys: dict[str, tuple[str, ...] | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.represent, str):
            self.represent = tuple(self.represent)
            if not self.represent:
                raise ValueError("strategy.represent must not be empty")
        normalized_identity_keys: dict[str, tuple[str, ...]] = {}
        for path, raw_value in self.identity_keys.items():
            values = (raw_value,) if isinstance(raw_value, str) else tuple(raw_value)
            normalized: list[str] = []
            for value in values:
                candidate = value.strip()
                if not candidate:
                    raise ValueError("strategy.identity_keys entries must not be empty")
                normalized.append(candidate if candidate.startswith("/") else f"/{candidate}")
            if isinstance(raw_value, str):
                normalized_identity_keys[path] = tuple(normalized)
            else:
                normalized_identity_keys[path] = tuple(normalized)
        self.identity_keys = normalized_identity_keys


@dataclass(frozen=True, slots=True)
class ResolvedStrategy:
    represent: Literal["auto", "native", "raster"]
    plan: Literal[
        "auto",
        "document_grounded",
        "whole_document",
        "page_map_reduce",
        "hybrid",
        "fused",
    ]
    media: MediaOptions
    provenance: ProvenancePolicy
    field_scope: FieldScopePolicy
    identity_keys: dict[str, tuple[str, ...]]
    mode: ExtractionMode = "auto"
    document_input: DocumentInput = "auto"
    page_input: PageInput = "auto"
    document_media: MediaOptions | None = None
    page_media: MediaOptions | None = None
    preset: StrategyPreset | None = None


def _media_for_document_input(
    document_input: DocumentInput,
    base_media: MediaOptions,
) -> MediaOptions:
    pdf_mode: Literal["auto", "native", "raster"]
    if document_input == "native":
        pdf_mode = "native"
    elif document_input == "image":
        pdf_mode = "raster"
    else:
        pdf_mode = "auto"
    return replace(base_media, pdf_mode=pdf_mode, page_strategy="single", grounding="off")


def _media_for_page_input(
    page_input: PageInput,
    base_media: MediaOptions,
) -> MediaOptions:
    if page_input not in {"auto", "image"}:
        raise ValueError(f"Unsupported page_input {page_input!r}")
    # v1 page-grounded extraction uses page images consistently; text-page
    # variants can be added later without changing the public knob.
    return replace(base_media, pdf_mode="raster", page_strategy="map_reduce", grounding="off")


def resolve_runtime_mode(
    *,
    mode: ExtractionMode,
    document_input: DocumentInput,
    page_input: PageInput,
    base_media: MediaOptions,
) -> ResolvedStrategy:
    default_provenance = ProvenancePolicy(mode="sidecar", cite_by="none", strict=False)

    if mode == "auto":
        media = replace(base_media, grounding="off")
        return ResolvedStrategy(
            represent=media.pdf_mode,
            plan="auto",
            media=media,
            provenance=default_provenance,
            field_scope=FieldScopePolicy(),
            identity_keys={},
            mode="auto",
            document_input="auto",
            page_input="auto",
            preset=None,
        )

    if mode == "document":
        document_media = _media_for_document_input(document_input, base_media)
        return ResolvedStrategy(
            represent=document_media.pdf_mode,
            plan="whole_document",
            media=document_media,
            provenance=default_provenance,
            field_scope=FieldScopePolicy(),
            identity_keys={},
            mode="document",
            document_input=document_input,
            page_input="auto",
            document_media=document_media,
            preset=None,
        )

    if mode == "page":
        page_media = _media_for_page_input(page_input, base_media)
        return ResolvedStrategy(
            represent=page_media.pdf_mode,
            plan="page_map_reduce",
            media=page_media,
            provenance=default_provenance,
            field_scope=FieldScopePolicy(),
            identity_keys={},
            mode="page",
            document_input="auto",
            page_input=page_input,
            page_media=page_media,
            preset=None,
        )

    if mode == "hybrid":
        document_media = _media_for_document_input(document_input, base_media)
        page_media = _media_for_page_input(page_input, base_media)
        return ResolvedStrategy(
            represent=page_media.pdf_mode,
            plan="hybrid",
            media=page_media,
            provenance=default_provenance,
            field_scope=FieldScopePolicy(),
            identity_keys={},
            mode="hybrid",
            document_input=document_input,
            page_input=page_input,
            document_media=document_media,
            page_media=page_media,
            preset=None,
        )

    raise ValueError(f"Unknown extraction mode {mode!r}")


def _strategy_from_preset(preset: StrategyPreset) -> Strategy:
    if preset == "native":
        return Strategy(
            represent="native",
            plan="whole_document",
            provenance=ProvenancePolicy(mode="sidecar", cite_by="none", strict=False),
        )
    if preset == "balanced":
        return Strategy(
            represent="auto",
            plan="auto",
            provenance=ProvenancePolicy(mode="sidecar", cite_by="none", strict=False),
        )
    if preset == "auditable":
        return Strategy(
            represent="raster",
            plan="page_map_reduce",
            provenance=ProvenancePolicy(mode="sidecar", cite_by="none", strict=False),
        )
    if preset == "retrieve":
        raise NotImplementedError("strategy='retrieve' is not supported by the current runtime yet")
    raise ValueError(f"Unknown strategy preset {preset!r}")


def _plan_to_page_strategy(
    plan: Literal[
        "auto", "document_grounded", "whole_document", "page_map_reduce", "hybrid", "fused"
    ],
) -> Literal["auto", "single", "map_reduce"]:
    if plan == "auto":
        return "auto"
    if plan in {"whole_document", "document_grounded"}:
        return "single"
    return "map_reduce"


def _supports_structural_provenance(
    represent: Literal["auto", "native", "raster"],
    plan: Literal[
        "auto", "document_grounded", "whole_document", "page_map_reduce", "hybrid", "fused"
    ],
) -> bool:
    return plan == "document_grounded" and represent in {"auto", "native"}


def _resolve_representation(
    represent: Representation | tuple[Representation, ...],
    *,
    plan: ExecutionPlan,
) -> tuple[Literal["auto", "native", "raster"], list[str]]:
    supported: tuple[Literal["auto", "native", "raster"], ...] = ("auto", "native", "raster")
    if isinstance(represent, str):
        reps = (represent,)
    else:
        reps = represent

    skipped: list[str] = []
    first_supported: Literal["auto", "native", "raster"] | None = None
    for candidate in reps:
        if candidate not in supported:
            skipped.append(candidate)
            continue
        if first_supported is None:
            first_supported = candidate
        if not (plan in {"hybrid", "fused"} and candidate == "native"):
            return candidate, skipped
        skipped.append(candidate)

    if first_supported is not None:
        return first_supported, skipped
    if skipped:
        raise NotImplementedError(
            "None of the requested representations are supported by the current runtime: "
            + ", ".join(skipped)
        )
    raise ValueError("strategy.represent must not be empty")


def _resolve_provenance(
    provenance: ProvenancePolicy,
    *,
    represent: Literal["auto", "native", "raster"],
    plan: Literal[
        "auto", "document_grounded", "whole_document", "page_map_reduce", "hybrid", "fused"
    ],
) -> ProvenancePolicy:
    if provenance.mode == "none":
        return ProvenancePolicy(mode="none", cite_by="none", strict=provenance.strict)

    if provenance.cite_by in {"chunk_id", "offset", "quote", "model"}:
        if provenance.strict:
            raise NotImplementedError(
                f"provenance.cite_by={provenance.cite_by!r} is not supported by the current runtime"
            )
        warnings.warn(
            f"provenance.cite_by={provenance.cite_by!r} is not supported yet; "
            "falling back to the closest runtime-supported provenance mode",
            stacklevel=3,
        )
        if _supports_structural_provenance(represent, plan):
            return ProvenancePolicy(mode="sidecar", cite_by="structural", strict=False)
        return ProvenancePolicy(mode="sidecar", cite_by="none", strict=False)

    if provenance.cite_by == "structural" and not _supports_structural_provenance(represent, plan):
        if provenance.strict:
            raise NotImplementedError(
                "Structural provenance is not supported for the resolved represent/plan combination"
            )
        warnings.warn(
            "Structural provenance is not available for the resolved represent/plan "
            "combination; falling back to evidence-only sidecar output",
            stacklevel=3,
        )
        return ProvenancePolicy(mode="sidecar", cite_by="none", strict=False)

    return provenance


def resolve_runtime_strategy(
    strategy: StrategyPreset | Strategy | None,
) -> ResolvedStrategy:
    if strategy is None:
        return ResolvedStrategy(
            represent="auto",
            plan="auto",
            media=MediaOptions(),
            provenance=ProvenancePolicy(mode="sidecar", cite_by="structural", strict=False),
            field_scope=FieldScopePolicy(),
            identity_keys={},
            mode="auto",
            document_input="auto",
            page_input="auto",
            preset=None,
        )

    declared = _strategy_from_preset(strategy) if isinstance(strategy, str) else strategy
    preset = strategy if isinstance(strategy, str) else None

    if declared.plan in {"page_windows", "retrieve_then_extract"}:
        raise NotImplementedError(
            f"strategy.plan={declared.plan!r} is not supported by the current runtime yet"
        )
    if declared.plan == "fused":
        warnings.warn(
            "strategy.plan='fused' is deprecated and will be removed in a future release; "
            "prefer strategy.plan='document_grounded' or mode='hybrid'",
            DeprecationWarning,
            stacklevel=2,
        )

    resolved_represent, skipped = _resolve_representation(
        declared.represent,
        plan=declared.plan,
    )
    if not isinstance(declared.represent, str):
        if skipped:
            warnings.warn(
                "Skipping unsupported or incompatible representations: " + ", ".join(skipped),
                stacklevel=2,
            )

    resolved_plan = declared.plan
    if resolved_plan in {"hybrid", "fused"} and resolved_represent == "auto":
        resolved_represent = "raster"
    if resolved_plan in {"hybrid", "fused"} and resolved_represent == "native":
        raise NotImplementedError(
            f"strategy.plan={resolved_plan!r} requires a page-aware representation; "
            "native PDF input is not supported by the current runtime"
        )
    media = MediaOptions(
        pdf_mode=resolved_represent,
        page_strategy=_plan_to_page_strategy(resolved_plan),
        grounding="force"
        if declared.provenance.cite_by == "structural" and declared.provenance.strict
        else "auto"
        if declared.provenance.cite_by == "structural"
        else "off",
    )
    provenance = _resolve_provenance(
        declared.provenance,
        represent=resolved_represent,
        plan=resolved_plan,
    )
    if provenance.cite_by != "structural":
        media = replace(media, grounding="off")

    return ResolvedStrategy(
        represent=resolved_represent,
        plan=resolved_plan,
        media=media,
        provenance=provenance,
        field_scope=declared.field_scope,
        identity_keys=declared.identity_keys,
        mode=(
            "document"
            if resolved_plan in {"whole_document", "document_grounded"}
            else "page"
            if resolved_plan in {"page_map_reduce", "fused"}
            else "hybrid"
            if resolved_plan == "hybrid"
            else "auto"
        ),
        document_input=(
            "native"
            if media.pdf_mode == "native"
            else "image"
            if media.pdf_mode == "raster"
            else "auto"
        ),
        page_input="image" if resolved_plan in {"page_map_reduce", "hybrid", "fused"} else "auto",
        document_media=media
        if resolved_plan in {"whole_document", "document_grounded", "hybrid"}
        else None,
        page_media=media if resolved_plan in {"page_map_reduce", "hybrid", "fused"} else None,
        preset=preset,
    )


@dataclass(slots=True)
class ExtractOptions:
    mode: ExtractionMode | None = None
    document_input: DocumentInput = "auto"
    page_input: PageInput = "auto"
    passes: int = 1
    max_repair_attempts: int = 2
    max_char_buffer: int | None = None
    batch_length: int = 4
    max_workers: int = 1
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    max_pages: int | None = None
    max_pdf_bytes: int | None = None
    max_api_calls: int | None = None
    per_call_timeout_s: float | None = None
    per_document_timeout_s: float | None = None
    overlap_chars: int = 0
    tokenizer: TokenizerName | Tokenizer | None = None
    alignment: AlignmentOptions = field(default_factory=AlignmentOptions)
    media: MediaOptions = field(default_factory=MediaOptions)
    format: FormatOptions = field(default_factory=FormatOptions)
    prompt_validation: PromptValidationLevel = PromptValidationLevel.WARNING
    schema_mode: Literal["compact", "pretty"] = "compact"
    structured_output: Literal["auto", "native", "prompt"] = "auto"
    repair: Literal["none", "local", "targeted"] = "targeted"
    chunk_error: Literal["raise", "skip"] = "skip"
    merge_strategy: Literal["first_wins", "last_wins", "prefer_non_null"] = "first_wins"
    resolver: Resolver | None = None
    strategy: StrategyPreset | Strategy | None = None

    def __post_init__(self) -> None:
        if self.passes < 1:
            raise ValueError("passes must be >= 1")
        if self.max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        if self.batch_length < 1:
            raise ValueError("batch_length must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.concurrency.network_workers < 1:
            raise ValueError("concurrency.network_workers must be >= 1")
        if self.concurrency.cpu_workers < 1:
            raise ValueError("concurrency.cpu_workers must be >= 1")
        if self.concurrency.max_inflight_image_bytes < 1:
            raise ValueError("concurrency.max_inflight_image_bytes must be >= 1")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must be >= 0")
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        if self.max_pdf_bytes is not None and self.max_pdf_bytes < 1:
            raise ValueError("max_pdf_bytes must be >= 1")
        if self.max_api_calls is not None and self.max_api_calls < 1:
            raise ValueError("max_api_calls must be >= 1")
        if self.per_call_timeout_s is not None and self.per_call_timeout_s <= 0:
            raise ValueError("per_call_timeout_s must be > 0")
        if self.per_document_timeout_s is not None and self.per_document_timeout_s <= 0:
            raise ValueError("per_document_timeout_s must be > 0")
        uses_simple_mode = (
            self.mode is not None or self.document_input != "auto" or self.page_input != "auto"
        )
        if self.mode is None and (self.document_input != "auto" or self.page_input != "auto"):
            raise ValueError("document_input/page_input require mode to be set")
        if self.mode == "document" and self.page_input != "auto":
            raise ValueError("page_input is only valid for mode='page' or mode='hybrid'")
        if self.mode == "page" and self.document_input != "auto":
            raise ValueError("document_input is only valid for mode='document' or mode='hybrid'")
        if self.strategy is not None and (
            self.media.pdf_mode != "auto"
            or self.media.page_strategy != "auto"
            or self.media.grounding != "auto"
        ):
            raise ValueError(
                "strategy cannot be combined with custom media options for "
                "pdf_mode, page_strategy, or grounding"
            )
        if uses_simple_mode and self.strategy is not None:
            raise ValueError("mode/document_input/page_input cannot be combined with strategy")
        if uses_simple_mode and (
            self.media.pdf_mode != "auto"
            or self.media.page_strategy != "auto"
            or self.media.grounding != "auto"
        ):
            raise ValueError(
                "mode/document_input/page_input cannot be combined with custom media "
                "options for pdf_mode, page_strategy, or grounding"
            )
        if self.passes > 1:
            warnings.warn(
                "ExtractOptions.passes>1 is deprecated; use repair='targeted' and "
                "max_repair_attempts instead",
                DeprecationWarning,
                stacklevel=2,
            )

    def resolve_runtime_strategy(self) -> ResolvedStrategy:
        if self.mode is not None:
            return resolve_runtime_mode(
                mode=self.mode,
                document_input=self.document_input,
                page_input=self.page_input,
                base_media=self.media,
            )
        if self.strategy is None:
            provenance = (
                ProvenancePolicy(mode="sidecar", cite_by="none", strict=False)
                if self.media.grounding == "off"
                else ProvenancePolicy(
                    mode="sidecar",
                    cite_by="structural",
                    strict=(self.media.grounding == "force"),
                )
            )
            return ResolvedStrategy(
                represent=self.media.pdf_mode,
                plan=(
                    "whole_document"
                    if self.media.page_strategy == "single"
                    else "page_map_reduce"
                    if self.media.page_strategy == "map_reduce"
                    else "auto"
                ),
                media=self.media,
                provenance=provenance,
                field_scope=FieldScopePolicy(),
                identity_keys={},
                mode=(
                    "document"
                    if self.media.page_strategy == "single"
                    else "page"
                    if self.media.page_strategy == "map_reduce"
                    else "auto"
                ),
                document_input=(
                    "native"
                    if self.media.pdf_mode == "native"
                    else "image"
                    if self.media.pdf_mode == "raster"
                    else "auto"
                ),
                page_input="image" if self.media.page_strategy == "map_reduce" else "auto",
                document_media=self.media if self.media.page_strategy == "single" else None,
                page_media=self.media if self.media.page_strategy == "map_reduce" else None,
                preset=None,
            )

        resolved = resolve_runtime_strategy(self.strategy)
        media = replace(
            resolved.media,
            raster_dpi=self.media.raster_dpi,
            max_image_dim=self.media.max_image_dim,
            raster_format=self.media.raster_format,
            jpeg_quality=self.media.jpeg_quality,
        )
        return replace(resolved, media=media)

    def resolved_media_options(self) -> MediaOptions:
        return self.resolve_runtime_strategy().media
