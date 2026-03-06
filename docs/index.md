# parsantic

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

````python
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
````

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

## Extract from text

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

## Update existing objects

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

## Extract from PDFs and images

Pass a `Document` instead of a string to extract structured data from visual
content — scanned invoices, screenshots, research papers:

```python
from pathlib import Path
from pydantic import BaseModel
from parsantic import extract
from parsantic.extract import Document

class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    total: float

result = extract(
    Document.from_pdf(Path("invoice.pdf")),
    Invoice,
    model="gemini:gemini-2.5-flash",
)
result.value
# Invoice(invoice_number='INV-2024-001', vendor='Acme Corp', total=1250.00)
```

Images work the same way:

```python
result = extract(
    Document.from_image(Path("receipt.jpg")),
    Invoice,
    model="openai:gpt-4o-mini",
)
```

By default, PDFs with a text layer are extracted as text (no vision cost);
otherwise pages are rasterized to images.

Use `mode` when you want to force a higher-level PDF strategy:

```python
from parsantic.extract import ExtractOptions

# Whole document in one call.
result = extract(
    Document.from_pdf(pdf_bytes),
    Invoice,
    model="gemini:gemini-2.5-flash",
    options=ExtractOptions(mode="document", document_input="native"),
)

# Page-by-page vision with page provenance.
result = extract(
    Document.from_pdf(pdf_bytes),
    Invoice,
    model="gemini:gemini-2.5-flash",
    options=ExtractOptions(mode="page"),
)

# Hybrid: whole-document native PDF + page images for page-grounded fields.
result = extract(
    Document.from_pdf(pdf_bytes),
    Invoice,
    model="gemini:gemini-2.5-flash",
    options=ExtractOptions(
        mode="hybrid",
        document_input="native",
        page_input="image",
    ),
)

print(result.sources["/total"])   # SourceRef(scope="page", pages=(1,))
print(result.sources["/vendor"])  # SourceRef(scope="document", pages=())
```

For lower-level PDF/image control, `MediaOptions` is still available:

```python
from parsantic.extract.options import ExtractOptions, MediaOptions

result = extract(
    Document.from_pdf(pdf_bytes),
    Invoice,
    model="openai:gpt-4o-mini",
    options=ExtractOptions(
        media=MediaOptions(pdf_mode="raster", page_strategy="map_reduce"),
    ),
)
```

| `mode` | Behavior |
| :--- | :--- |
| `"auto"` | Use text extraction for text-layer PDFs, otherwise rasterize pages (default) |
| `"document"` | Run one whole-document extraction |
| `"page"` | Run page-by-page extraction |
| `"hybrid"` | Run both a whole-document branch and a page branch, then merge |

| `document_input` | Behavior |
| :--- | :--- |
| `"auto"` | Let parsantic choose the whole-document representation |
| `"native"` | Send the raw PDF binary to the model |
| `"image"` | Rasterize the PDF and bundle page images into one whole-document request |

| `page_input` | Behavior |
| :--- | :--- |
| `"auto"` | Use the default page-grounded representation |
| `"image"` | Rasterize each PDF page to an image |

Advanced `MediaOptions`:

| `pdf_mode` | Behavior |
| :--- | :--- |
| `"auto"` | Text layer → text extraction; otherwise rasterize (default) |
| `"native"` | Send raw PDF binary to the model |
| `"raster"` | Convert every page to JPEG/PNG |

## Vertex AI support

Use `vertex:` prefix with any Gemini model to route through Vertex AI:

```python
result = extract(
    "Dr. Sarah Chen is a principal ML engineer at Anthropic.",
    Person,
    model="vertex:gemini-2.5-flash",
    provider_kwargs={"project_id": "my-project", "region": "us-central1"},
)
```

Credentials are resolved automatically from environment variables
(`VERTEX_PROJECT_ID`, `VERTEX_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`)
or from `gcloud auth application-default login`.

## Native structured output

When the model supports it (Gemini, OpenAI, etc.), parsantic can use the
provider's native JSON schema constraints instead of prompt-based extraction.
This is enabled by default (`"auto"`) and falls back to prompt mode
transparently:

```python
result = extract(
    text,
    MySchema,
    model="gemini:gemini-2.5-flash",
    options=ExtractOptions(structured_output="native"),  # or "auto" (default)
)
```

| `structured_output` | Behavior |
| :--- | :--- |
| `"auto"` | Use native mode if the model supports it, otherwise prompt (default) |
| `"native"` | Force native JSON schema constraints |
| `"prompt"` | Always use prompt-based extraction |

If native mode fails validation, parsantic automatically recovers the raw
JSON from the response and runs it through the local repair pipeline.

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
