from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from parsantic.ai import validation_error_paths


@dataclass(slots=True)
class RepairContext:
    original_prompt: str = ""
    original_document_text: str = ""
    additional_context: str | None = None
    schema_text: str | None = None
    raw_output: str = ""
    errors: list[str] = field(default_factory=list)
    attempt_index: int = 0
    prior_output: str = ""
    validation_errors: list[str] = field(default_factory=list)

    def effective_output(self) -> str:
        return self.raw_output or self.prior_output

    def effective_errors(self) -> list[str]:
        return self.errors or self.validation_errors

    def effective_prompt(self) -> str:
        return self.original_prompt or self.original_document_text


def collect_validation_errors(error: ValidationError) -> list[str]:
    paths = validation_error_paths(error)
    details: list[str] = []
    error_list = error.errors()
    for index, item in enumerate(error_list):
        path = paths[index] if index < len(paths) else "/"
        details.append(f"{path}: {item.get('msg', 'validation error')}")
    return details


def build_repair_prompt(
    context: RepairContext | None = None,
    *,
    original_question: str | None = None,
    additional_context: str | None = None,
    schema_text: str | None = None,
    repair_context: RepairContext | None = None,
) -> str:
    ctx = repair_context or context
    if ctx is None:
        raise ValueError("build_repair_prompt requires a RepairContext")
    question = (
        original_question or ctx.effective_prompt() or "Extract structured data from this document."
    )
    rendered_context = (
        additional_context if additional_context is not None else ctx.additional_context
    )
    rendered_schema = schema_text if schema_text is not None else ctx.schema_text
    broken_output = ctx.effective_output()
    errors = ctx.effective_errors()

    lines = [
        "Repair the previous structured extraction result.",
        "",
        "Original extraction context:",
        question.strip(),
        "",
    ]
    if ctx.original_document_text:
        lines.extend(["Document text:", ctx.original_document_text, ""])
    if rendered_context:
        lines.extend([rendered_context, ""])
    if rendered_schema:
        lines.extend(["Schema:", rendered_schema, ""])
    lines.extend(
        [
            "Previous invalid JSON:",
            broken_output,
            "",
            "Validation errors:",
            *[f"- {error}" for error in errors],
            "",
            "Return the complete corrected JSON only.",
        ]
    )
    return "\n".join(lines)


def format_broken_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
