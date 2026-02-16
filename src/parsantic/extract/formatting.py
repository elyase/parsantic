from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FormatOptions:
    format: str = "json"
    use_fences: bool = True
    wrapper_key: str | None = None
    attribute_suffix: str = "_attributes"
    index_suffix: str | None = None


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
            except Exception as exc:  # pragma: no cover
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
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return value
