"""parsantic demo — parse messy LLM output into typed Python objects."""

from enum import Enum

from pydantic import BaseModel

import parsantic as sap

# ── Define a schema ──────────────────────────────────────────────────


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    title: str
    priority: Priority
    done: bool = False


# ── 1. Parse: messy LLM text → typed object ─────────────────────────
#
# LLMs wrap JSON in markdown, add trailing commas, use wrong-case keys,
# and return fuzzy enum values. parse() handles all of it in one call.

llm_output = """
Here is your task:
```json
{
    "Title": "Fix the login bug",
    "priority": "HIGH",
    "done": false,
}
```
"""

result = sap.parse(llm_output, Task)
print("1) parse — messy LLM text → Task")
print(f"   {result.value!r}")
print(f"   flags={result.flags}  score={result.score}")
print()


# ── 2. Parse without markdown — just broken JSON ────────────────────

broken_json = '{"title": "Deploy v2", "priority": "Medium",}'
result2 = sap.parse(broken_json, Task)
print("2) parse — trailing comma + case-insensitive enum")
print(f"   {result2.value!r}")
print(f"   flags={result2.flags}")
print()


# ── 3. Debug: see all candidates and why one was chosen ──────────────

debug = sap.parse_debug(
    '{"title": "Review PR", "priority": "Critical", "done": true}',
    Task,
)
print("3) parse_debug — inspect candidates")
for c in debug.candidates:
    status = " ← chosen" if c is debug.chosen else (" (failed)" if c.score < 0 else "")
    print(f"   score={c.score:>2}  flags={c.flags}{status}")
print(f"   value={debug.value!r}")
print()


# ── 4. Patch: safe RFC 6902 updates ──────────────────────────────────
#
# Instead of asking the LLM to regenerate the whole object, apply
# small JSON patches. The original document is never mutated.

doc = {"title": "Fix login bug", "priority": "high", "done": False}

ops = sap.normalize_patches(
    [
        {"op": "replace", "path": "/priority", "value": "critical"},
        {"op": "replace", "path": "/done", "value": True},
    ]
)
patched = sap.apply_patch(doc, ops)
print("4) apply_patch — safe partial update")
print(f"   before: {doc}")
print(f"   after:  {patched}")
print()


# ── 5. Patch + validate against schema ───────────────────────────────

validated = sap.apply_patch_and_validate(doc, ops, Task)
print("5) apply_patch_and_validate — patch then check schema")
print(f"   {validated.value!r}")
print()


# ── 6. Normalize LLM patch output ────────────────────────────────────
#
# LLMs return patches in all kinds of formats. normalize_patches()
# handles JSON strings, nested {"patches": [...]}, raw dicts, etc.

raw_llm = '{"patches": [{"op": "replace", "path": "/title", "value": "Updated"}]}'
normalized = sap.normalize_patches(raw_llm)
print("6) normalize_patches — handle messy LLM patch formats")
print(f"   input:  {raw_llm}")
print(f"   output: {normalized}")
print()


# ── 7. Streaming: parse token-by-token ───────────────────────────────
#
# Feed tokens as they arrive from the LLM. Get partial objects back.

stream = sap.parse_stream(Task)
tokens = [
    '{"title": "Stre',
    'aming task", "pr',
    'iority": "low"}',
]
print("7) parse_stream — incremental parsing")
for tok in tokens:
    stream.feed(tok)
    partial = stream.parse_partial()
    print(f"   feed {tok!r:30s} → {partial.value!r}")
final = stream.finish()
print(f"   final: {final.value!r}")
