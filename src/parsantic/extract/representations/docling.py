from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DoclingImportError(ImportError):
    """Raised when the optional Docling dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class DoclingPage:
    page_index: int
    text: str
    tables: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DoclingDocument:
    text: str
    pages: tuple[DoclingPage, ...]
    meta: dict[str, Any] = field(default_factory=dict)


def is_docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def _import_docling() -> Any:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise DoclingImportError(
            "Docling support is optional and opt-in. Install it separately before "
            "calling extract_docling_representation()."
        ) from exc
    return DocumentConverter


def _page_texts_from_export(export: Any) -> tuple[DoclingPage, ...]:
    pages: list[DoclingPage] = []
    exported_pages = getattr(export, "pages", None) or []
    for page_index, page in enumerate(exported_pages):
        text = getattr(page, "text", "") or ""
        table_texts = tuple(str(table) for table in getattr(page, "tables", []) or ())
        image_refs = tuple(str(image) for image in getattr(page, "pictures", []) or ())
        pages.append(
            DoclingPage(
                page_index=page_index,
                text=text,
                tables=table_texts,
                images=image_refs,
            )
        )
    return tuple(pages)


def extract_docling_representation(source: Path | bytes) -> DoclingDocument:
    """Convert a PDF into a Docling-backed page representation.

    This adapter is intentionally standalone and non-default. Runtime extraction
    must opt into it explicitly in future work after benchmarks justify the
    dependency and behavior trade-offs.
    """

    DocumentConverter = _import_docling()
    converter = DocumentConverter()

    if isinstance(source, bytes):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(source)
            handle.flush()
            result = converter.convert(handle.name)
    else:
        result = converter.convert(str(source))

    export = getattr(result, "document", result)
    if hasattr(export, "export_to_markdown"):
        text = export.export_to_markdown()
    else:
        text = str(export)

    pages = _page_texts_from_export(export)
    return DoclingDocument(
        text=text,
        pages=pages,
        meta={"adapter": "docling", "page_count": len(pages)},
    )
