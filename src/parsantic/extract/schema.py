from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


@dataclass(slots=True)
class PydanticSchemaAdapter:
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
            return value.model_dump(mode="json")
        return self.adapter.dump_python(value, mode="json")
