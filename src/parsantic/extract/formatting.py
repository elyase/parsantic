from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic_core import to_jsonable_python


@dataclass(slots=True)
class FormatOptions:
    format: Literal["json", "yaml"] = "json"
    use_fences: bool = True
    wrapper_key: str | None = None

    def __post_init__(self) -> None:
        normalized = self.format.lower()
        if normalized not in {"json", "yaml"}:
            raise ValueError("format must be either 'json' or 'yaml'")
        self.format = cast(Literal["json", "yaml"], normalized)


class FormatHandler:
    def __init__(self, options: FormatOptions | None = None) -> None:
        self.options = options or FormatOptions()

    def format_example(self, value: Any) -> str:
        payload = value
        if self.options.wrapper_key and isinstance(value, list):
            payload = {self.options.wrapper_key: value}
        payload = _to_json_safe(payload)
        if self.options.format == "yaml":
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "YAML output requested but PyYAML is not installed. Install with: pip install pyyaml"
                ) from exc
            text = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return self._add_fences(text) if self.options.use_fences else text

    def _add_fences(self, text: str) -> str:
        return f"```{self.options.format}\n{text.strip()}\n```"


def _to_json_safe(value: Any) -> Any:
    try:
        return to_jsonable_python(value)
    except (ValueError, TypeError):
        return value
