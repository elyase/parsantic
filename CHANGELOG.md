# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
