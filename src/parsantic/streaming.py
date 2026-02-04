from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, TypeAdapter, create_model

from .coerce import CoerceOptions, coerce_jsonish_to_python
from .jsonish import ParseOptions, parse_jsonish
from .types import ScoredValue


@lru_cache(maxsize=128)
def _partial_model_for(model_type: type[BaseModel]) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, field in model_type.model_fields.items():
        annotation = field.annotation if field.annotation is not None else Any
        fields[field_name] = (annotation | None, None)
    return create_model(f"{model_type.__name__}Partial", __base__=BaseModel, **fields)


@dataclass(slots=True)
class StreamParser[T]:
    """
    Incremental SAP parser.

    This is a pragmatic streaming interface:
    - callers feed chunks (as strings)
    - parser maintains a buffer
    - `parse_partial()` attempts to coerce the *current buffer* (is_done=False)
    - `finish()` validates final result (is_done=True)

    This mirrors the Rust approach where `raw_string_is_done` controls completion state.
    """

    adapter: TypeAdapter[T]
    parse_options: ParseOptions
    coerce_options: CoerceOptions
    _partial_adapter: TypeAdapter[Any] | None = None
    _buffer: str = ""

    def __post_init__(self) -> None:
        if self._partial_adapter is not None:
            return
        model_type = getattr(self.adapter, "_type", None)
        if (
            model_type is None
            or not isinstance(model_type, type)
            or not issubclass(model_type, BaseModel)
        ):
            return
        self._partial_adapter = TypeAdapter(_partial_model_for(model_type))

    def feed(self, chunk: str) -> None:
        self._buffer += chunk

    @property
    def buffer(self) -> str:
        return self._buffer

    @property
    def partial_type(self) -> type[BaseModel] | None:
        """Type returned by :meth:`parse_partial` (if available)."""
        if self._partial_adapter is None:
            return None
        partial_type = getattr(self._partial_adapter, "_type", None)
        return (
            partial_type
            if isinstance(partial_type, type) and issubclass(partial_type, BaseModel)
            else None
        )

    def parse_partial(self) -> ScoredValue:
        jsonish_value = parse_jsonish(self._buffer, options=self.parse_options, is_done=False)
        sv = coerce_jsonish_to_python(
            jsonish_value, self.adapter, options=self.coerce_options, allow_partial=True
        )
        if self._partial_adapter is None or not isinstance(sv.value, dict):
            return sv
        # The coercer only keeps per-field validated values, so the dict is safe to
        # validate into a "Partial" model (all fields optional).
        partial = self._partial_adapter.validate_python(sv.value)
        return ScoredValue(value=partial, flags=sv.flags, score=sv.score)

    def finish(self) -> ScoredValue:
        _missing = object()
        try:
            validated = self.adapter.validate_json(self._buffer)
        except Exception:
            validated = _missing
        if validated is not _missing:
            return ScoredValue(value=validated, flags=(), score=0)
        jsonish_value = parse_jsonish(self._buffer, options=self.parse_options, is_done=True)
        coerced = coerce_jsonish_to_python(jsonish_value, self.adapter, options=self.coerce_options)
        validated = self.adapter.validate_python(coerced.value)
        return ScoredValue(value=validated, flags=coerced.flags, score=coerced.score)
