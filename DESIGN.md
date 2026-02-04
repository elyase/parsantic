# parsantic: Typed JSON extraction + safe patch updates for LLMs

## Motivation
LLMs are great at *reasoning* over structured data and unreliable at *serializing* it:
- Raw outputs are “JSON-ish” (markdown fences, trailing commas, comments, partial objects, multiple objects, etc.).
- Tool calling helps, but tool args still come back wrong/incomplete and schema validation still fails.
- “Update by rewrite” is dangerous: regenerating large nested objects is slow/expensive and often drops unchanged fields.

This project’s goal: make structured extraction and updates feel like Pydantic/FastAPI:
type-safe, deterministic, debuggable, and cheap to retry.

## Prior art
This project draws from several existing libraries and concepts:

- **BAML** ([github.com/BoundaryML/baml](https://github.com/BoundaryML/baml)): the original SAP algorithm with scoring, fuzzy matching, and streaming semantics (implemented in Rust)
- **trustcall** ([github.com/hinthornw/trustcall](https://github.com/hinthornw/trustcall)): JSON Patch retry patterns and practical patch application for LLM tool-calling workflows
- **pydantic-ai** ([github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai)): LLM calling, retries, and output validation plumbing

Optional conceptual background: BoundaryML's SAP write-up at `https://boundaryml.com/blog/schema-aligned-parsing`.

## Executive summary (defaults & decisions)
- **Packaging**: default install includes `pydantic-ai`, but a core install exists without it (see §1).
- **Extraction default**:
  - Try structured output/tooling once when available.
  - On failure, switch to SAP + patch-based repair (bounded; no “rewrite the world” retries).
- **Update default**:
  - Patch-based updates are first-class.
  - `remove` operations are disabled by default to prevent accidental data loss (enable explicitly).

## Success criteria (how we know it’s “best-in-class”)
- **Most outputs fixed locally**: common JSON-ish failures are repaired without another model call.
- **Bounded retries**: when a retry is needed, it is capped and uses patch payloads instead of full regeneration.
- **No silent data loss**: updates do not delete fields unless explicitly allowed.
- **Actionable debugging**: every failure has a deterministic trace (candidates, flags, scores, validation errors).

---

## 0) Design principles

1. **Correctness first**: return a value that Pydantic validates (or raise with rich diagnostics).
2. **Determinism**: parsing/coercion should be stable given the same input text/object.
3. **Bounded cost**: retries must be bounded and targeted; default retry strategy should avoid regenerating whole documents.
4. **Provider-agnostic**: core should work without any specific LLM SDK.
5. **Composable**: users can adopt only the parser/coercer, or also the patch/update layer, or also the pydantic-ai integration.

---

## 1) Packaging & dependency strategy

### Install
- `pip install parsantic` → core parsing, coercion, and patch utilities (deps: `pydantic>=2`, `regex`, `pyyaml`).
- `pip install "parsantic[ai]"` → adds `pydantic-ai` for LLM extraction with provider support.

### Patch dependency stance
- RFC 6902 **`add`/`replace`/`remove` + JSON Pointer** is implemented in core (no external deps).
- A compatibility layer for `jsonpatch/jsonpointer` can be added later if users need additional RFC operations like `test`, `move`, `copy`.

---

## 2) Core mode defaults (what happens out of the box)

### Default extraction mode (recommended)
**Tools-first, then SAP+patch fallback**.

Why:
- Tools/structured output is the fastest when it works.
- SAP is the best “last mile” when the output is messy or the schema is complex.
- Patch retries are the cheapest way to correct errors without regenerating huge documents.

Concrete default for the `parsantic.ai` helper layer:
1. **Try once** with pydantic-ai’s native structured output (tool/schema mode).
2. If it fails: re-run using **text output parsed by SAP**, and use **JSON Patch** for subsequent repair retries.

> Implementation note for contributors: pydantic-ai’s built-in retries for native structured output re-generate the whole object. Patch-based retries are easiest to implement inside a `TextOutput` processor (see `reference/pydantic-ai/pydantic_ai_slim/pydantic_ai/_output.py`).

### Default update mode (recommended)
**Patch-first updates**, with safe defaults:
- `add`/`replace` allowed
- `remove` disabled by default (enable explicitly)

Reason: most update use cases prioritize “never drop existing facts unless asked”.

---

## 3) Public API design

This API aims to:
- Be usable *without* pydantic-ai.
- Expose the “debug surface” so users can trust behavior.
- Support both **text parsing** and **tool-call/object coercion**.

### 3.1 `parsantic` (core parse + coerce)

Keep existing:
- `parse(text: str, target: type[T] | TypeAdapter[T], ...) -> ParseResult[T]`
- `parse_stream(target: type[T] | TypeAdapter[T], ...) -> StreamParser[T]`

Add (high ROI):
- `coerce(value: Any, target: type[T] | TypeAdapter[T], *, options: CoerceOptions | None = None) -> ParseResult[T]`
  - Use this when you already have python objects (e.g., tool call args) but still want schema-aligned coercions + scoring.
- `parse_debug(...) -> ParseDebug[T]`
- `coerce_debug(...) -> ParseDebug[T]`

`ParseDebug[T]` should include:
- All candidate interpretations (value, flags, score)
- The selected candidate
- Any validation errors encountered per candidate

Example (core; no LLM SDK required):

```python
from pydantic import BaseModel

from parsantic import parse, parse_stream


class Resume(BaseModel):
    name: str
    email: str | None = None


# 1) Parse messy model text (markdown fences, trailing commas, etc.)
resume = parse("```json\n{name: Ada, email: ada@example.com,}\n```", Resume).value

# 2) Streaming-friendly parsing
sp = parse_stream(Resume)
sp.feed('{"name": "Ada')
partial = sp.parse_partial().value  # best-effort partial dict/model
sp.feed(' Lovelace", "email": "ada@example.com"}')
final = sp.finish().value

# 3) Tool-call args / python dict → schema-aligned coercion (planned API)
#    Useful when you already have a JSON object from a provider’s tool calling.
from parsantic import coerce

resume2 = coerce({"name": "Ada Lovelace", "email": "ada@example.com"}, Resume).value
```

Implementation pointers for contributors:
- `parsantic/src/parsantic/api.py` (entrypoints `parse`, `parse_stream`)
- `parsantic/src/parsantic/jsonish.py` (candidate generation)
- `parsantic/src/parsantic/coerce.py` (schema-aligned coercion + scoring)

### 3.2 `parsantic.patch` (JSON Patch primitives + safe apply)

Core concepts:
- `JsonPatchOp`: RFC6902 ops (`add`, `remove`, `replace`) + JSON Pointer `path`
- `PatchPolicy`: safety rails + limits

API:
- `apply_patch(doc: dict, patches: Sequence[JsonPatchOp], *, policy: PatchPolicy = PatchPolicy()) -> dict`
- `apply_patch_and_validate(doc: dict | BaseModel, patches: Sequence[JsonPatchOp], target: type[T] | TypeAdapter[T], *, policy: PatchPolicy = PatchPolicy()) -> ParseResult[T]`
- `normalize_patches(patches: Any) -> list[JsonPatchOp]`
  - Handles the “patch list accidentally returned as a string” failure mode (trustcall has a helper like this).

PatchPolicy defaults:
- `allow_remove: bool = False`
- `max_ops: int = 50` (tunable)
- `max_path_depth: int = 32`
- `allow_append: bool = True` (allow `/-` array append)

Implementation notes:
- Borrow trustcall’s practical edge-case fix: when patching `"/-"` into a string field, interpret as “string concatenation” rather than array append.
- Provide friendly exceptions for conflicts/out-of-bounds pointers.

Example (update an existing document safely):

```python
from pydantic import BaseModel

from parsantic.patch import JsonPatchOp, PatchPolicy, apply_patch_and_validate


class User(BaseModel):
    preferred_name: str
    age: int


existing = {"preferred_name": "Alex", "age": 28}
patches = [
    JsonPatchOp(op="replace", path="/age", value=29),
    # NOTE: `remove` is disabled by default; enabling it is an explicit choice.
]

updated = apply_patch_and_validate(
    existing,
    patches=patches,
    target=User,
    policy=PatchPolicy(allow_remove=False),
).value
```

Where to copy behavior from:
- Patch parsing / “patch list inside a string”: `reference/trustcall/trustcall/_base.py` (`_ensure_patches`)
- Patch application + string concat edge case: `reference/trustcall/trustcall/_base.py` (`_apply_patch`) and `reference/trustcall/tests/unit_tests/test_utils.py`

### 3.3 `parsantic.ai` (pydantic-ai integration)

This module exists in the codebase, but should be import-safe when pydantic-ai is missing:
- `import parsantic.ai` should raise a clear `ImportError` instructing `pip install "parsantic[ai]"` to get pydantic-ai support.

Core utilities:
- `sap_text_output(target: type[T] | TypeAdapter[T], *, parse_options=None, coerce_options=None) -> pydantic_ai.TextOutput[T]`
  - For providers/models where you want plain text output but still get typed results.
- `patch_repair_output(target: type[T], *, policy=PatchPolicy(), max_attempts=3, schema_renderer="compact") -> pydantic_ai.TextOutput[T]`
  - Runs “patch-don’t-post” loop *inside* the output processor using `ModelRetry`.

High-level helpers (optional, but great UX):
- `extract(agent: pydantic_ai.Agent[..., Any], prompt: str, target: type[T], *, strategy: Literal["auto","tools","text"]="auto", ...) -> T`
- `update(agent: pydantic_ai.Agent[..., Any], prompt: str, existing: dict | BaseModel, target: type[T], *, allow_remove=False, ...) -> T`

Example (pydantic-ai integration; bounded retries with patch fallback):

```python
from pydantic import BaseModel
from pydantic_ai import Agent

from parsantic.ai import extract


class Resume(BaseModel):
    name: str
    email: str | None = None


agent = Agent("openai:gpt-5")  # any pydantic-ai model/provider string works

resume = await extract(
    agent,
    prompt="Extract name and email from: Ada Lovelace <ada@example.com>",
    target=Resume,
    strategy="auto",  # tools-first once, then SAP+patch fallback
)
```

Where to copy pydantic-ai idioms from:
- Output plumbing (`TextOutput`, validators, retry errors): `reference/pydantic-ai/pydantic_ai_slim/pydantic_ai/_output.py`
- `TextOutput` marker type: `reference/pydantic-ai/pydantic_ai_slim/pydantic_ai/output.py`
- Retry mechanism (`ModelRetry`): `reference/pydantic-ai/pydantic_ai_slim/pydantic_ai/exceptions.py`

---

### 3.4 `parsantic.extract` (primary extraction API; LangExtract‑inspired)

This is the **main** end‑user API for “extract structured data from text.” It builds on core SAP parsing, adds source‑grounding, long‑doc handling, and provider routing. It should feel **simple by default** and scale to advanced control through progressive disclosure.

Rationale & sources:
- Source‑grounded spans + alignment quality: `reference/langextract/langextract/core/data.py`, `reference/langextract/langextract/resolver.py`
- Prompt/example validation: `reference/langextract/langextract/prompt_validation.py`
- Chunking + tokenizer design: `reference/langextract/langextract/chunking.py`, `reference/langextract/langextract/core/tokenizer.py`
- Multi‑pass extraction + overlap merge: `reference/langextract/langextract/annotation.py`
- Format handling (fences/wrappers/index order): `reference/langextract/langextract/core/format_handler.py`
- Provider registry + plugins + env defaults + schema: `reference/langextract/langextract/plugins.py`, `reference/langextract/langextract/factory.py`, `reference/langextract/langextract/providers/README.md`
- Batch/parallel knobs: `reference/langextract/langextract/annotation.py`, `reference/langextract/langextract/providers/openai.py`
- SAP parsing/repair: `reference/baml_repo/...jsonish` (scoring/coercion), `reference/trustcall/...` (patch repair)
- pydantic‑ai retry & output plumbing: `reference/pydantic-ai/pydantic_ai_slim/...`

#### 3.4.1 Public API (progressive disclosure)

Level 0: **One‑liner extraction** (schema‑first)

```python
from pydantic import BaseModel
from parsantic.extract import extract

class Resume(BaseModel):
    name: str
    email: str | None = None

result = extract(
    "Ada Lovelace <ada@example.com>",
    Resume,
    model="openai:gpt-4o-mini",  # or "gemini-2.5-flash"
)
resume = result.value
```

Level 1: **Add instructions + examples** (few‑shot)

```python
from parsantic.extract import Example, Prompt, extract

prompt = Prompt(
    description="Extract name and email exactly as written.",
    examples=[
        Example(
            text="Grace Hopper <grace@navy.mil>",
            output={"name": "Grace Hopper", "email": "grace@navy.mil"},
        )
    ],
)

result = extract(
    "Ada Lovelace <ada@example.com>",
    Resume,
    prompt=prompt,
    model="openai:gpt-4o-mini",
)
```

Level 2: **Batch/streaming over many docs** (sync + async)

```python
from parsantic.extract import Document, extract_iter

docs = [
    Document(text="Ada Lovelace <ada@example.com>", document_id="doc1"),
    Document(text="Alan Turing <alan@bletchley.uk>", document_id="doc2"),
]

for res in extract_iter(docs, Resume, model="openai:gpt-4o-mini"):
    print(res.document_id, res.value)
```

```python
from parsantic.extract import Document, extract_aiter

async for res in extract_aiter(docs, Resume, model="openai:gpt-4o-mini"):
    ...
```

Level 3: **Reusable Extractor (context manager)**
Use for repeated calls, shared provider clients, caching, and stable defaults.

```python
from parsantic.extract import Extractor, Prompt

prompt = Prompt(description="Extract name + email.")

with Extractor(model="openai:gpt-4o-mini", prompt=prompt) as ex:
    r1 = ex.extract("Ada <ada@example.com>", Resume)
    r2 = ex.extract("Grace <grace@navy.mil>", Resume)
```

Async variant:

```python
from parsantic.extract import Extractor

async with Extractor(model="openai:gpt-4o-mini") as ex:
    r = await ex.aextract("Ada <ada@example.com>", Resume)
```

Level 4: **Full control (options object)**

```python
from parsantic.extract import ExtractOptions, extract

opts = ExtractOptions(
    passes=2,
    max_char_buffer=1000,
    batch_length=10,
    max_workers=10,
    tokenizer="unicode",
    alignment=AlignmentOptions(fuzzy_threshold=0.75),
    format=FormatOptions(format="json", wrapper_key="extractions"),
    prompt_validation="warning",
)

result = extract("Ada <ada@example.com>", Resume, options=opts)
```

#### 3.4.2 Core types (what developers implement)

**Schema‑agnostic contract** (future‑proofing beyond Pydantic):
- `SchemaAdapter[T]` protocol:
  - `render_schema(mode: Literal["compact","json_schema"]) -> str`
  - `validate(value: Any) -> T` (returns typed model)
  - `normalize(value: Any) -> dict` (for prompts + alignment)
- Default adapter: `PydanticSchemaAdapter` (from `type[BaseModel]` or `TypeAdapter`).
- Future adapters: dataclasses, attrs, JSON Schema, Zod‑style, etc.

**Result objects** (SAP + extraction metadata):
- `ExtractResult[T]`:
  - `value: T`
  - `document_id: str | None`
  - `raw_text: str | None` (model output)
  - `candidates: list[ScoredValue]` (from SAP)
  - `evidence: list[FieldEvidence]` (source‑grounded spans)
  - `debug: ExtractDebug | None`
- `FieldEvidence`:
  - `path: str` (JSON Pointer to field)
  - `value_preview: str`
  - `char_interval: tuple[int, int] | None`
  - `token_interval: tuple[int, int] | None`
  - `alignment_status: Literal["match_exact","match_lesser","match_fuzzy","unmatched"]`

Rationale: mirrors LangExtract’s span tracking (`char_interval`, `token_interval`, `alignment_status`) while keeping SAP’s candidate scoring visible.
Sources: LangExtract data/resolver for spans; BAML jsonish scoring for candidate ranking.

#### 3.4.3 Prompt + example model (few‑shot done right)

**Prompt model**:
- `Prompt(description: str, examples: list[Example] = [])`
- `Example(text: str, output: dict | BaseModel | Any)`
- `PromptValidationLevel = {off, warning, error}`

Validation behavior:
- Align each example output’s **string leaf values** back to example text.
- Warn or raise if values don’t align (prevents broken few‑shot examples).

Sources:
- Prompt validation: `reference/langextract/langextract/prompt_validation.py`
- Alignment: `reference/langextract/langextract/resolver.py`

#### 3.4.4 Alignment & tokenization

Alignment flow:
1. Tokenize source text.
2. Tokenize extracted string values.
3. Match with difflib SequenceMatcher.
4. If no exact match, apply fuzzy window match.

Options:
- `alignment.enable_fuzzy_alignment` (default True)
- `alignment.fuzzy_threshold` (default 0.75)
- `alignment.accept_match_lesser` (default True)
- `tokenizer="regex" | "unicode" | Tokenizer`

Sources:
- Alignment algorithm: `reference/langextract/langextract/resolver.py`
- Tokenizers: `reference/langextract/langextract/core/tokenizer.py`
- Japanese / non‑spaced example: `reference/langextract/docs/examples/japanese_extraction.md`

#### 3.4.5 Chunking, long‑docs, and multi‑pass recall

Chunking:
- Sentence‑aware chunker that respects newlines; oversized tokens become their own chunk.
- Config via `max_char_buffer` and `tokenizer`.

Multi‑pass:
- `passes=N`: run extraction N times, merge non‑overlapping spans; first‑pass wins.

Sources:
- Chunking: `reference/langextract/langextract/chunking.py`
- Pass merge: `reference/langextract/langextract/annotation.py`

#### 3.4.6 Provider interface + registry

Provider base:
- `BaseProvider.infer(batch_prompts: Sequence[str], **kwargs) -> Sequence[str]`
- Optional `get_schema_class()` + `apply_schema()` for structured outputs.

Registry:
- `@register(patterns..., priority=...)` decorator
- plugin discovery via entry points
- `SAP_DISABLE_PLUGINS=1` to disable 3rd‑party providers

Env defaults:
- Resolve API keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`, `PARSANTIC_API_KEY`)
- Warn on conflicts

Sources:
- Registry + plugins: `reference/langextract/langextract/plugins.py`, `reference/langextract/langextract/providers/router.py`
- Env defaults: `reference/langextract/langextract/factory.py`

#### 3.4.7 Format & ordering conventions

Format options:
- `format="json" | "yaml"`
- `use_fences: bool`
- `wrapper_key: str | None` (default `extractions`)
- `attribute_suffix: str` (default `_attributes`)
- `index_suffix: str | None` for ordering

Why: standardizes output and parsing, improves determinism and schema‑based routing.
Source: `reference/langextract/langextract/core/format_handler.py`

#### 3.4.8 Streaming‑friendly emission

When extracting over large doc sets:
- `extract_iter` yields completed documents as soon as they’re fully processed.
- Avoids materializing all results; memory stays bounded.

Source: `reference/langextract/langextract/annotation.py`

#### 3.4.9 Implementation roadmap (for devs)

Phase 1 — Types + prompts
- Implement `SchemaAdapter` + Pydantic adapter.
- Implement `Prompt`, `Example`, `PromptValidationLevel`.
- Wire prompt rendering (schema + examples) and validation.

Phase 2 — Provider plumbing
- Implement provider base + registry + plugin discovery.
- Add env‑driven defaults + warnings.
- Add schema‑aware structured output negotiation.

Phase 3 — Extraction pipeline
- Implement chunking + tokenizers.
- Implement alignment + evidence mapping.
- Implement multi‑pass merge.

Phase 4 — Streaming & performance
- Implement `extract_iter` / `extract_aiter`.
- Add `batch_length` + `max_workers` flow‑through.

Phase 5 — Debug + tests
- Alignments, validation, chunking, multi‑pass, tokenizer parity.
- Validate deterministic ordering and stable scoring.

Phase 6 — Docs + examples
- Quick‑start, long‑doc extraction, non‑spaced language example, provider plugin guide.

---

## 4) “Patch-don’t-post” retry strategy (trustcall ideas, adapted to pydantic-ai)

### When to use patch retries
Use patch retries when:
- Validation fails after a first attempt (tool output or text parse).
- Updating an existing document.
- The target schema is large/nested.

### Prompt contract for patches
Adopt trustcall’s proven prompt structure:
- Provide:
  - `json_doc_id` (even if single doc, keep for future multi-doc support)
  - current document (or relevant slice)
  - schema (or relevant slice)
  - validation errors (or relevant subset)
- Ask for:
  - `planned_edits` (short bullet plan; do not require chain-of-thought)
  - `patches` (list of ops)
- Strong guidance on op ordering:
  1. `replace`
  2. `remove` (arrays: highest index first)
  3. `add` (`/-` to append)

Minimal patch tool schema (language-model facing; modeled after trustcall):

```python
from pydantic import BaseModel, Field

from parsantic.patch import JsonPatchOp


class PatchDoc(BaseModel):
    json_doc_id: str = Field(description="ID of the document to patch (use 'doc' if single).")
    planned_edits: str = Field(description="Short plan of edits (for debugging; not executed).")
    patches: list[JsonPatchOp] = Field(description="RFC6902 add/replace/remove operations.")
```

Repair loop sketch (what `patch_repair_output(...)` should do internally):

```text
best = best_candidate_from_sap(raw_text)
for attempt in 1..max_attempts:
  try validate(best) -> return best
  except ValidationError as e:
    prompt = build_patch_prompt(best, schema_slice(e), errors=e)
    patches = parse_patches(model(prompt))  # use SAP again here
    best = apply_patch(best, patches, policy)
raise last error with debug trace
```

### Key improvement opportunity: schema/doc slicing
BoundaryML’s SAP post points out naive retries are expensive. We can do better by sending **only what’s relevant**:
- From `ValidationError.errors()`, derive JSON pointer-like paths.
- Slice:
  - schema fragment(s) for those paths
  - doc fragment(s) for those paths
- Retry prompt contains only:
  - the errors
  - the fragments
  - minimal context

This reduces tokens, increases accuracy, and makes retries bounded and predictable.

Implementation detail (Pydantic error paths → JSON Pointer):
- Pydantic v2 `ValidationError.errors()` returns items with `loc` tuples like `('user', 'pets', 0, 'age')`.
- Convert that to JSON Pointer `/user/pets/0/age` (with RFC6901 escaping for `~` and `/`).
- When slicing, include parents (e.g., `/user/pets/0`) so the model has enough context to patch correctly.

---

## 5) SAP engine improvements (high ROI)

The current `parsantic` core is strong, but there are several BAML-inspired features that should be completed to reach “best in class”.

### 5.1 Enum/Literal matching (missing today)
Implement BAML-style string matching for:
- `Enum` values
- `Literal[...]` values

Features to port:
- Accent-insensitive matching
- Punctuation-insensitive matching
- Optional substring match (guarded; can be dangerous)
- Deterministic tie handling + “ambiguous match” flag

References for contributors:
- Matching algorithm: `reference/baml_repo/engine/baml-lib/jsonish/src/deserializer/coercer/match_string.rs`
- Scoring flags: `reference/baml_repo/engine/baml-lib/jsonish/src/deserializer/score.rs`

Consistency check to fix in this repo:
- `parsantic/src/parsantic/coerce.py` already defines score weights for flags like `substring_match`, `strip_punct`, `case_insensitive`, `accent_insensitive`, but the coercer does not yet emit all of them. Either implement the matching logic or remove unused flags to keep scoring honest.

### 5.2 Better key mapping for Pydantic models
Current `_coerce_model_keys()` is a good start, but extend it to:
- Respect field aliases more strongly (Pydantic v2 alias generators, alias priority).
- Optionally keep extras for `extra="allow"` models rather than dropping them.
- Provide “collision visibility”: if multiple fields normalize to the same key, surface that clearly.

References for contributors:
- Current implementation: `parsantic/src/parsantic/coerce.py` (`normalize_key`, `_coerce_model_keys`)
- BAML alias-focused tests for inspiration: `reference/baml_repo/engine/baml-lib/jsonish/src/tests/test_aliases.rs`

### 5.3 Better diagnostics
Add a stable debug structure:
- “Why did this candidate win?”
- “Which coercions happened?”
- “What got dropped and why?”

### 5.4 Streaming ergonomics
Expose structured partials:
- `StreamParser.parse_partial()` already returns `ScoredValue` with a dict for models.
- Add a convenience method:
  - `StreamParser.parse_partial_model()` that returns `dict` + flags + score + completion state.

---

## 6) Tests & quality gates

### Core (no LLM)
- Candidate selection determinism tests (ordering and scoring stable).
- Enum/Literal match tests (case, accents, punctuation, ambiguous ties).
- Patch apply tests:
  - nested add/replace
  - array remove ordering
  - pointer edge cases
  - policy enforcement (`allow_remove=False`)
  - string concat edge case (see `reference/trustcall/tests/unit_tests/test_utils.py`)

### Extraction layer (no LLM)
- Alignment tests:
  - exact match, lesser match, fuzzy match, unmatched
  - alignment thresholds + “accept_match_lesser” behavior
- Prompt/example validation tests (warn/error)
- Chunking tests:
  - newline boundaries
  - oversized tokens
  - multi‑sentence packing
- Multi‑pass merge tests (first‑pass wins)
- Tokenizer parity tests (Regex vs Unicode offsets)
- Format handling tests (wrapper, fences, index ordering)

### Verification (developer checklist)
Focus on a few **core tests** that prove the system works end‑to‑end. These should be fast and runnable without network access.

**V1 smoke tests (must pass)**
1. **Extraction E2E (fake provider)**
   - Use a deterministic in‑repo provider stub that returns a fixed JSON/YAML payload.
   - Verify `extract()` returns a typed model, plus evidence spans + alignment status.
   - Confirms: provider plumbing → prompt render → parse → alignment → typed output.

2. **Prompt validation**
   - Provide an example with a deliberate mismatch.
   - Verify warning in `warning` mode and exception in `error` mode.
   - Confirms: example alignment gate is active.

3. **Chunking + multi‑pass merge**
   - Long text with newline boundaries + an oversized token.
   - Run `passes=2` and verify “first‑pass wins” for overlapping spans.
   - Confirms: chunking behavior + merge logic.

4. **Streaming iterator**
   - Use `extract_iter` (and `extract_aiter` if available).
   - Verify documents are yielded in input order and before all docs finish.
   - Confirms: streaming emission works and is memory‑bounded.

**Suggested test files (names are flexible)**
- `tests/extract_smoke_test.py` (fake provider E2E)
- `tests/prompt_validation_test.py`
- `tests/chunking_multipass_test.py`
- `tests/streaming_iter_test.py`

**Expected pass criteria**
- All four tests green on a clean install.
- No network calls (providers are stubbed/mocked).
- Run time < 5s on a laptop.

### pydantic-ai integration (no real network calls)
- Use pydantic-ai’s test utilities or fake models to simulate retries:
  - invalid output → retry → patch output → success

---

## 7) Documentation & examples (ship as part of v1 quality)

Add docs pages and examples for:
1. Extraction (tool mode and text+SAP mode)
2. Streaming extraction
3. Updating an existing doc (patch mode)
4. Debugging/trace output
5. “Safety defaults” (no deletes; bounded patch ops)

---

## 8) Execution roadmap (milestones)

### Milestone A — Core API + diagnostics
- Add `coerce()`/`coerce_debug()`
- Add `parse_debug()`
- Improve flag/score reporting

### Milestone B — Patch primitives (no LLM)
- Implement `parsantic.patch` module
- Patch apply + validate + policy
- Tests for all patch behaviors

### Milestone C — pydantic-ai integration
- Implement `parsantic.ai` module (import-safe)
- `sap_text_output()` + `patch_repair_output()`
- Tests using fake models (no network)

### Milestone D — Schema/doc slicing for retries (token+latency win)
- Implement `slice_schema_for_error_paths()`
- Implement `slice_doc_for_error_paths()`
- Integrate into patch retry prompts

### Milestone E — Complete SAP coercions
- Enum/Literal matching
- More coercions aligned with BAML scoring

### Milestone F — Extraction layer (LangExtract‑inspired)
- `parsantic.extract` data model + alignment
- Prompt/example validation
- Chunking + Unicode tokenizer
- Multi‑pass extraction + merge
- Provider registry + env defaults + schema‑aware outputs
- Batch/parallel knobs + streaming emission

---

## 9) Non-goals (for now)

- Building a full agent framework (pydantic-ai already covers this).
- Constrained decoding / CFG grammars (could be a future “advanced” mode).
- Full JSON Schema compiler/optimizer (focus on Pydantic → compact prompt rendering + slicing).
