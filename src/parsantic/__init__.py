from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parsantic")
except PackageNotFoundError:  # pragma: no cover - local source tree without installed metadata
    __version__ = "0.0.0-dev"

from .api import coerce, coerce_debug, parse, parse_debug, parse_stream
from .coerce import CoerceOptions
from .extract import (
    Extractor,
    aextract,
    aextract_stream,
    extract,
    extract_aiter,
    extract_batch,
    extract_iter,
    extract_stream,
    visualize,
)
from .jsonish import ParseOptions
from .patch import (
    JsonPatchOp,
    PatchDoc,
    PatchError,
    PatchPolicy,
    PolicyViolationError,
    apply_patch,
    apply_patch_and_validate,
    normalize_patches,
)
from .retry import RetryPolicy
from .types import CandidateDebug, ParseDebug
from .update import UpdateResult, aupdate, update

_LAZY_AI_EXPORTS = {
    "build_patch_prompt",
    "slice_doc_for_paths",
    "slice_schema_for_paths",
    "validation_error_paths",
}


def __getattr__(name: str):
    if name in _LAZY_AI_EXPORTS:
        from . import ai

        value = getattr(ai, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CandidateDebug",
    "CoerceOptions",
    "Extractor",
    "JsonPatchOp",
    "ParseDebug",
    "ParseOptions",
    "PatchDoc",
    "PatchError",
    "PatchPolicy",
    "PolicyViolationError",
    "RetryPolicy",
    "UpdateResult",
    "aextract",
    "aextract_stream",
    "apply_patch",
    "apply_patch_and_validate",
    "aupdate",
    "build_patch_prompt",
    "coerce",
    "coerce_debug",
    "extract",
    "extract_aiter",
    "extract_batch",
    "extract_iter",
    "extract_stream",
    "normalize_patches",
    "parse",
    "parse_debug",
    "parse_stream",
    "slice_doc_for_paths",
    "slice_schema_for_paths",
    "update",
    "validation_error_paths",
    "visualize",
]
