"""Optional document-representation adapters."""

from .docling import (
    DoclingDocument,
    DoclingImportError,
    DoclingPage,
    extract_docling_representation,
    is_docling_available,
)

__all__ = [
    "DoclingDocument",
    "DoclingImportError",
    "DoclingPage",
    "extract_docling_representation",
    "is_docling_available",
]
