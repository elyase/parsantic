from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("parsantic")
except PackageNotFoundError:  # pragma: no cover - local source tree without installed metadata
    __version__ = "0.2.0"

# The ai module is always importable (import-safe), but functions that
# require pydantic-ai will raise ImportError at call time when the
# library is not installed.  We expose pure utilities unconditionally.
from .ai import (
    build_patch_prompt,
    slice_doc_for_paths,
    slice_schema_for_paths,
    validation_error_paths,
)
from .api import coerce, coerce_debug, parse, parse_debug, parse_stream
from .coerce import CoerceOptions
from .extract import Extractor, extract, extract_aiter, extract_iter
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
from .types import CandidateDebug, ParseDebug
from .update import UpdateResult, aupdate, update

__all__ = [
    "CandidateDebug",
    "CoerceOptions",
    "Extractor",
    "JsonPatchOp",
    "ParseDebug",
    "PatchDoc",
    "PatchError",
    "PatchPolicy",
    "PolicyViolationError",
    "apply_patch",
    "apply_patch_and_validate",
    "build_patch_prompt",
    "coerce",
    "coerce_debug",
    "extract",
    "extract_aiter",
    "extract_iter",
    "normalize_patches",
    "parse",
    "ParseOptions",
    "parse_debug",
    "parse_stream",
    "slice_doc_for_paths",
    "slice_schema_for_paths",
    "update",
    "UpdateResult",
    "aupdate",
    "validation_error_paths",
]
