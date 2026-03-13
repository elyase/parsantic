# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.7.3 - 2026-03-12

### Added

- Added deterministic page selection via `select_pdf_pages(...)` for opt-in
  page pruning before extraction.
- Added selector benchmark coverage and a checked-in selector snapshot for the
  oncology page-scale corpus.
- Added docs and an example for page-selection-first extraction.

### Changed

- Native PDF extraction with `page_indices` now uploads a physically subsetted
  PDF when local PDF rewriting support is available.
- Nested schema selection now resolves `$defs` / `$ref` and accepts
  `TypeAdapter` inputs directly.

### Fixed

- Fixed native PDF prompt / payload semantics so selected-page native requests
  no longer mix full-document hints with subsetted uploads.
- Added normalization / no-op handling for subsetted page ranges.

## 0.7.2 - 2026-03-12

### Added

- Added a fail-closed safety gate for multi-PDF extraction on a single
  `Document` while attachment-aware provenance is under development.
- Added regression tests covering multi-PDF safety gating and bounded async
  batch fallback concurrency.

### Changed

- Updated the async batch fallback path to share a provider-call budget and
  bound document concurrency instead of faning out all documents at once.
- Updated the README and docs to recommend one `Document` per PDF plus
  `extract_batch()` / `aextract_batch()` for current multi-document workflows.

### Fixed

- Prevented misleading page-level provenance from being produced by bundled
  multi-PDF extraction paths that cannot yet preserve attachment identity.
