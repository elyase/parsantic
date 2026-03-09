from __future__ import annotations

from .alignment import AlignmentOptions, Resolver, TokenAlignmentResolver, get_resolver
from .batch import BatchResult, BatchStatus, SupportsBatchInfer, aextract_batch, extract_batch
from .chunking import TextChunk
from .concurrency import ConcurrencyConfig
from .diagnostics import FieldDiagnostic, FieldState
from .formatting import FormatOptions
from .media import (
    Attachment,
    AttachmentKind,
    PageQuality,
    PreflightResult,
    analyze_pdf_attachment,
    analyze_pdf_source,
)
from .options import (
    ExtractOptions,
    FieldScopePolicy,
    MediaOptions,
    ProvenancePolicy,
    ResolvedStrategy,
    Strategy,
    resolve_runtime_strategy,
)
from .pipeline import (
    Extractor,
    aextract,
    aextract_stream,
    extract,
    extract_aiter,
    extract_iter,
    extract_stream,
)
from .prompt import Example, Prompt, PromptValidationLevel
from .providers.static import StaticProvider
from .types import (
    AlignmentStatus,
    ChunkDebug,
    Document,
    ExtractDebug,
    ExtractResult,
    ExtractStreamEvent,
    FieldEvidence,
    MergeConflict,
    SourceRef,
)
from .visualization import visualize

__all__ = [
    "AlignmentOptions",
    "AlignmentStatus",
    "Attachment",
    "AttachmentKind",
    "BatchResult",
    "BatchStatus",
    "ChunkDebug",
    "ConcurrencyConfig",
    "Document",
    "Example",
    "ExtractDebug",
    "ExtractOptions",
    "ExtractResult",
    "ExtractStreamEvent",
    "FieldDiagnostic",
    "Extractor",
    "FieldEvidence",
    "FieldState",
    "FieldScopePolicy",
    "FormatOptions",
    "MediaOptions",
    "MergeConflict",
    "Prompt",
    "PageQuality",
    "PreflightResult",
    "ProvenancePolicy",
    "PromptValidationLevel",
    "ResolvedStrategy",
    "Resolver",
    "SourceRef",
    "StaticProvider",
    "Strategy",
    "SupportsBatchInfer",
    "TextChunk",
    "TokenAlignmentResolver",
    "aextract",
    "aextract_stream",
    "aextract_batch",
    "analyze_pdf_attachment",
    "analyze_pdf_source",
    "extract",
    "extract_aiter",
    "extract_batch",
    "extract_iter",
    "extract_stream",
    "get_resolver",
    "resolve_runtime_strategy",
    "visualize",
]
