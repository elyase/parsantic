"""Prompt templates for LLM interactions.

Centralizes all prompt text to make it easy to version, test, and customize.
"""

from __future__ import annotations

import json
from typing import Any

from .patch import PatchPolicy


def render_policy_lines(policy: PatchPolicy) -> str:
    """Render patch policy as bullet points for prompt inclusion."""
    return "\n".join(
        [
            f"- remove operations allowed: {'yes' if policy.allow_remove else 'no'}",
            f"- max operations: {policy.max_ops}",
            f"- max path depth: {policy.max_path_depth}",
            f"- append (/-) allowed: {'yes' if policy.allow_append else 'no'}",
        ]
    )


def build_update_prompt(
    doc: dict[str, Any],
    instruction: str,
    schema_text: str,
    policy: PatchPolicy,
) -> str:
    """Build the initial prompt asking the LLM to produce patches."""
    policy_lines = render_policy_lines(policy)
    return f"""You are updating a JSON document based on new information.

## Current Document
```json
{json.dumps(doc, indent=2, default=str)}
```

## Target Schema
```json
{schema_text}
```

## Instruction
{instruction}

## Rules
- Return ONLY a JSON array of RFC 6902 JSON Patch operations.
- Use "replace" for existing fields, "add" for new fields or array appends (use "/-" to append).
- Do NOT change fields unless the instruction implies it.
- Keep values JSON-serializable and conformant to the schema above.
- Order: "replace" operations first, then "add" operations.

## Patch Policy
{policy_lines}

Return the JSON array now:"""


def build_retry_prompt(
    patched_doc: dict[str, Any],
    validation_errors: list[dict[str, Any]],
    instruction: str,
    schema_text: str,
    policy: PatchPolicy,
) -> str:
    """Build a retry prompt when patches produced invalid output."""
    error_lines: list[str] = []
    for err in validation_errors:
        loc = err.get("loc", ())
        msg = err.get("msg", "unknown error")
        path_str = " -> ".join(str(p) for p in loc) if loc else "(root)"
        error_lines.append(f"- {path_str}: {msg}")
    errors_text = "\n".join(error_lines)
    policy_lines = render_policy_lines(policy)

    return f"""The patches you produced resulted in validation errors.

## Current Document (after patches)
```json
{json.dumps(patched_doc, indent=2, default=str)}
```

## Validation Errors
{errors_text}

## Target Schema
```json
{schema_text}
```

## Original Instruction (for context)
{instruction}

## Patch Policy
{policy_lines}

Return ONLY a JSON array of additional RFC 6902 JSON Patch operations to fix these errors:"""
