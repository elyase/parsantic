# parsantic

[![CI](https://github.com/elyase/parsantic/actions/workflows/ci.yml/badge.svg)](https://github.com/elyase/parsantic/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/parsantic)](https://pypi.org/project/parsantic/)
[![Python](https://img.shields.io/pypi/pyversions/parsantic)](https://pypi.org/project/parsantic/)
[![License](https://img.shields.io/pypi/l/parsantic)](https://github.com/elyase/parsantic/blob/main/LICENSE)

The structured extraction toolkit: parse, stream, extract, update, patch,
and coerce LLM output — locally, deterministically, with one clean API.

## Install

```bash
uv add parsantic
```

For LLM extraction and update features (OpenAI, Anthropic, Gemini, etc.):

```bash
uv add "parsantic[ai]"
```

## What it does

LLM output is messy. Models wrap JSON in markdown, add trailing commas, use
wrong-case enum values, and return partial objects mid-stream. Most tools
deal with this by retrying the LLM call. `parsantic` fixes it locally in
one pass:

```python
from enum import Enum
from pydantic import BaseModel
from parsantic import parse

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task(BaseModel):
    title: str
    priority: Priority
    days_left: int
    done: bool = False

# The LLM returned this mess:
llm_output = """
Sure! Here's the task you requested:

```json
{
    // Task details
    "Title": "Fix the login bug",
    "priority": "HIGH",
    "Days-Left": "3",
    "done": false,
}
```

Let me know if you need anything else!
"""

task = parse(llm_output, Task).value
# Task(title='Fix the login bug', priority=<Priority.HIGH: 'high'>, days_left=3, done=False)
```

One call. No retry. Markdown fences, comments, surrounding prose,
wrong-case keys, kebab-to-snake key normalization, enum coercion,
string-to-int coercion, trailing commas — all handled.

It works even when the JSON is still arriving. Feed tokens as they come and
get valid, typed partial objects back *while the LLM is still generating*:

```python
from parsantic import parse_stream

stream = parse_stream(Task)

stream.feed('{"title": "Stre')
print(stream.parse_partial().value)
# TaskPartial(title='Stre', priority=None, days_left=None, done=None)  ← partial but typed

stream.feed('aming task", "pr')
print(stream.parse_partial().value)
# TaskPartial(title='Streaming task', priority=None, days_left=None, done=None)

stream.feed('iority": "low", "days_left": 5}')
task = stream.finish().value
# Task(title='Streaming task', priority=<Priority.LOW: 'low'>, days_left=5, done=False)
```

Every call to `parse_partial()` returns a valid Pydantic object (a generated
`TaskPartial` with all-optional fields) with whatever values are available so far.
No waiting for the full response.

## Extract from text (requires `[ai]` extra)

Turn unstructured text into typed objects — with source grounding:

```python
from pydantic import BaseModel
from parsantic import extract

class Person(BaseModel):
    name: str
    role: str
    years_experience: int

result = extract(
    "Dr. Sarah Chen is a principal ML engineer at Anthropic (3 years).",
    Person,
    model="openai:gpt-4o-mini",
)
result.value
# Person(name='Sarah Chen', role='principal ML engineer', years_experience=3)

# Every extracted value is grounded back to the source text
result.evidence[0]
# FieldEvidence(path='/name', value_preview='Sarah Chen', char_interval=(4, 14), ...)
```

## Coerce tool arguments

LLM tool calls return raw dicts with wrong types and casing.
`coerce()` fixes them against your schema — no string parsing needed:

```python
from parsantic import coerce

# Raw dict from an LLM tool call
tool_args = {"title": "Deploy", "priority": "HIGH", "days_left": "2", "done": "true"}

task = coerce(tool_args, Task).value
# Task(title='Deploy', priority=<Priority.HIGH: 'high'>, days_left=2, done=True)
```

The coercion engine handles case-insensitive and accent-insensitive enum
matching, string-to-number conversion, key normalization, and more — each
tracked with a penalty score so the least-edited interpretation always wins.

## Update existing objects (requires `[ai]` extra)

Once you've extracted a large object, new information may arrive. Asking the
LLM to regenerate all 50 fields risks silently dropping data it wasn't
paying attention to. `update()` handles this — it asks the LLM to produce
only the changes as JSON Patch operations, applies them, and validates the
result:

```python
from pydantic import BaseModel
from parsantic import update

class User(BaseModel):
    name: str
    role: str
    skills: list[str]
    years_experience: int

profile = {
    "name": "Alex Chen",
    "role": "Software Engineer",
    "skills": ["Python", "TypeScript", "SQL"],
    "years_experience": 3,
}

result = update(
    existing=profile,
    instruction="Alex got promoted to Senior Engineer and picked up Rust.",
    target=User,
    model="openai:gpt-4o-mini",
)
result.value
# User(name='Alex Chen', role='Senior Software Engineer',
#      skills=['Python', 'TypeScript', 'SQL', 'Rust'], years_experience=5)
result.patches
# [JsonPatchOp(op='replace', path='/role', value='Senior Software Engineer'),
#  JsonPatchOp(op='replace', path='/years_experience', value=5),
#  JsonPatchOp(op='add', path='/skills/-', value='Rust')]
```

The original document is never mutated. Under the hood, `update()` prompts
the LLM for RFC 6902 patches, parses the messy response with `parse()`,
applies the patches with safety rails (`remove` disabled by default), and
validates the result with schema-aware coercion. If validation fails, it
automatically retries with the error context.

## Candidate scoring

When the input is ambiguous, `parsantic` generates multiple candidate
interpretations and picks the one requiring the fewest transformations:

```python
from parsantic import parse_debug

debug = parse_debug('{"title": "Review PR", "priority": "Critical", "days_left": 1}', Task)
for c in debug.candidates:
    print(f"  score={c.score}  flags={c.flags}")
# score=-1  flags=()                    ← direct JSON parse (failed validation)
# score=3   flags=('case_insensitive',) ← coerced "Critical" → Priority.CRITICAL
print(debug.value)
# Task(title='Review PR', priority=<Priority.CRITICAL: 'critical'>, days_left=0, done=False)
```

Every coercion is tagged with a flag and a cost. You can inspect exactly
what happened and why.

## Comparison with similar libraries

`parsantic` focuses on one thing: getting a valid typed object from messy
LLM text with the least effort and fewest LLM calls.

### Quick comparison

| | parsantic | BAML | trustcall | llguidance | LangExtract |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Approach** | Fix output locally | Fix output locally | Patch via LLM retry | Prevent at token level | LLM extraction pipeline |
| **Handles invalid JSON text** | Yes (local repair) | Yes (local repair) | N/A (tool calling) | N/A (constrained decoding) | No (expects valid JSON/YAML; can strip fences) |
| **Streaming** | Typed partial objects (Partial model) | Typed partial objects (generated Partial types) | — | Token-level masks | — |
| **Updates** | JSON Patch (targeted) | — | JSON Patch (LLM-generated) | — | Re-run extraction |
| **Source grounding** | Char/token spans + alignment | — | — | — | Char spans + fuzzy alignment |
| **Schema** | Pydantic models | `.baml` DSL | Pydantic / functions / JSON Schema | JSON Schema (subset) / CFG / regex | Example-driven (own format) |
| **Candidate scoring** | Weighted flags, inspectable | Scoring heuristics (internal) | — | — | — |
| **Install** | `pip install` | `pip install` (+ BAML CLI / codegen) | `pip install` | `pip install` (Rust-backed) | `pip install` |
| **Extra LLM calls on validation failure** | 0 | 0 | Yes (patch retries; configurable) | 0 | 0 |

### Detailed comparison

Each tool takes a fundamentally different approach to the structured-output
problem. Here is how they differ in practice.

#### [BAML](https://github.com/BoundaryML/baml) — Schema-Aligned Parsing in Rust

BAML is the closest in philosophy: let the LLM generate freely, then fix the
output locally. Its Rust-based parser handles the same classes of breakage
(markdown fences, trailing commas, wrong-case keys, partial objects) and
applies schema-aware coercion with a cost function.

**Where BAML goes further:**
- Dedicated `.baml` DSL with multi-language code generation (Python, TS, Ruby, Go, etc.)
- VS Code playground for live prompt testing
- Compact prompt schema (BAML docs claim ~80 % fewer tokens than JSON Schema; varies by schema)
- `@check` / `@assert` validators on output fields
- Dynamic types via TypeBuilder for runtime schema changes
- Multi-modal support (images, audio as first-class prompt inputs)
- Retry policies with exponential backoff and fallback client chains

**Where parsantic goes further:**
- Pure Python — no DSL, no code generation step
- Native Pydantic models as the schema (no new language to learn)
- JSON Patch support for targeted updates without full regeneration
- Source grounding with character/token-level evidence alignment
- Transparent candidate scoring with inspectable flags and costs
- Multi-pass extraction with non-overlapping span merging

#### [trustcall](https://github.com/hinthornw/trustcall) — Patch-Based Retry via LangChain

trustcall wraps LLM tool-calling with automatic validation and repair. When
a tool call fails Pydantic validation, it asks the LLM to generate JSON Patch
operations to fix the error rather than regenerating the entire output.

**Where trustcall goes further:**
- Simultaneous updates and insertions in one pass (`enable_inserts=True`)
- Works with LangChain chat models that support tool calling (broad provider coverage)
- Supports Pydantic models, plain functions, JSON Schema, and LangChain tools as input
- Graph-based execution with parallel tool-call validation (LangGraph)
- Optional deletes + policies for existing docs (`enable_deletes`, `existing_schema_policy`)

**Where parsantic goes further:**
- Zero extra LLM calls — all repairs are deterministic and local
- Streaming partial objects while the LLM is still generating
- Schema-aware coercion (enum matching, type conversion) without LLM involvement
- Candidate scoring shows exactly what was changed and why
- Source grounding ties extracted values back to source text positions
- No LangChain / LangGraph dependency

#### [llguidance](https://github.com/guidance-ai/llguidance) — Constrained Decoding at the Token Level

llguidance takes the opposite approach: instead of fixing broken output, it
prevents invalid output from being generated. At each decoding step it
computes a bitmask of valid tokens and blocks everything else.

**Where llguidance goes further:**
- Guarantees output that conforms to the provided grammar — no post-processing needed
- Supports context-free grammars (Lark-like syntax) beyond JSON Schema
- Parametric grammars for combinatorial structures (permutations, unique lists)
- ~50 μs per token mask for a 128k tokenizer (highly optimized Rust; depends on grammar)
- Powers OpenAI Structured Outputs; integrated into vLLM, SGLang, llama.cpp, and Chromium

**Where parsantic goes further:**
- Works with any LLM API — no inference-engine access needed (though llguidance
  is also available transparently via OpenAI's Structured Outputs API)
- Handles output that is *already generated* (logs, cached responses, tool-call results)
- JSON Patch updates, streaming partial objects, source grounding
- Candidate scoring with transparent coercion flags
- Pure Python, no Rust compilation or special deployment
- Handles messy real-world output (markdown, comments, surrounding text) that constrained
  decoding never produces but APIs frequently return

#### [LangExtract](https://github.com/google/langextract) — Extraction Pipeline with Visualization

LangExtract (Google) is an extraction-focused pipeline. It chunks long
documents, runs few-shot prompting in parallel, and aligns results back to
source text with interactive HTML visualization.

**Where LangExtract goes further:**
- Interactive HTML visualization with hover tooltips and colored highlighting
- Native Vertex AI Batch API integration for cost-efficient large-scale extraction
- Provider plugin system; Gemini provider supports schema-constrained output
- Schema derived from examples (no separate schema definition needed)

**Where parsantic goes further:**
- Local JSON fixing and schema-aware coercion (LangExtract parses JSON/YAML but does not repair invalid JSON)
- Streaming partial objects during generation
- JSON Patch for targeted document updates
- Candidate scoring with inspectable coercion trace
- Pydantic models as the schema (type-safe, IDE-friendly)
- Works as a standalone parser without any LLM — useful for cached/logged responses

### When to use what

| Scenario | Recommended tool |
| :--- | :--- |
| Parse messy LLM output into Pydantic models, no extra LLM calls | **parsantic** |
| Apply small updates to existing objects without regeneration | **parsantic** (JSON Patch) or **trustcall** (LLM-assisted) |
| Need source-grounded evidence spans from extracted data | **parsantic** or **LangExtract** |
| Guaranteed valid structure via cloud API (no self-hosting) | **llguidance** (via OpenAI Structured Outputs) |
| Own the inference engine and want grammar-level control | **llguidance** |
| Want a full DSL with code generation, VS Code tooling, and multi-modal | **BAML** |
| Production LLM orchestration with retries and fallback chains | **BAML** |
| Complex nested schemas that fail standard tool calling | **BAML** (SAP parsing) or **trustcall** (patch retries) |
| Validate and repair multiple tool calls in parallel | **trustcall** |
| Large-scale batch extraction with Vertex AI | **LangExtract** |
| Streaming typed partial objects during generation | **parsantic** or **BAML** |

## Development

```bash
uv sync
make test        # 345 tests
make check       # lint + format
make fmt         # auto-fix
```

## License

Apache-2.0
