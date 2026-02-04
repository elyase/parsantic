.PHONY: install
install: ## Install dependencies and git hooks
	uv sync --frozen
	uv run pre-commit install

.PHONY: fmt
fmt: ## Auto-format and auto-fix lint issues
	uv run ruff format
	uv run ruff check --fix

.PHONY: check
check: ## Run CI-style checks (lockfile + pre-commit)
	uv lock --locked
	uv run ruff format --check
	uv run ruff check
	uv run pre-commit run -a

.PHONY: test
test: ## Run tests
	uv run pytest

.PHONY: bench
bench: ## Run microbenchmarks
	uv run python benchmarks/bench.py

.PHONY: changelog
changelog: ## Regenerate CHANGELOG.md (requires maintain deps)
	uv sync --group maintain --frozen
	uv run --group maintain git-changelog -o CHANGELOG.md

.PHONY: api-check
api-check: ## Load API and run griffe checks (requires maintain deps)
	uv sync --group maintain --frozen
	uv run --group maintain griffe check parsantic --search src --base-ref HEAD --against HEAD

.PHONY: docs
docs: ## Serve docs locally (requires docs deps)
	uv run mkdocs serve

.PHONY: docs-test
docs-test: ## Build docs (requires docs deps)
	uv run mkdocs build -s

.PHONY: build
build: ## Build sdist + wheel into dist/
	uv build

.PHONY: help
help:
	@python -c "import re; \
	[[print(f'{m[0]:<12} {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
