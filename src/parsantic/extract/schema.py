from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


class SchemaAdapter(Protocol[T]):
    def validate(self, value: Any) -> T: ...

    def dump(self, value: T) -> Any: ...

    def render_schema(self, mode: str = "compact") -> str: ...


@dataclass(slots=True)
class PydanticSchemaAdapter(SchemaAdapter[T]):
    adapter: TypeAdapter[T]

    @classmethod
    def from_target(cls, target: type[T] | TypeAdapter[T]) -> PydanticSchemaAdapter[T]:
        if isinstance(target, TypeAdapter):
            return cls(adapter=target)
        return cls(adapter=TypeAdapter(target))

    def validate(self, value: Any) -> T:
        return self.adapter.validate_python(value)

    def dump(self, value: T) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        return self.adapter.dump_python(value)

    def render_schema(self, mode: str = "compact") -> str:
        schema = self.adapter.json_schema()
        if mode == "compact":
            return json.dumps(schema, ensure_ascii=False)
        return json.dumps(schema, indent=2, ensure_ascii=False)
