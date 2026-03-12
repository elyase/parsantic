from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic.fields import FieldInfo

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


@lru_cache(maxsize=256)
def _partial_model_type(model_type: type[BaseModel]) -> type[BaseModel]:
    """Build a "partial" version of a BaseModel type.

    All fields become optional (nullable, default None) while preserving field
    metadata (aliases, constraints, descriptions) in the generated JSON schema.
    Validators are intentionally *not* inherited: this type exists only to
    relax provider-side structured output constraints for per-chunk extraction.
    """

    fields: dict[str, tuple[Any, FieldInfo]] = {}
    override = Field(default=None, default_factory=None)
    for name, info in model_type.model_fields.items():
        annotation = info.annotation or Any
        merged_info = FieldInfo.merge_field_infos(info, override)
        fields[name] = (annotation | None, merged_info)

    return create_model(
        f"{model_type.__name__}Partial",
        __config__=model_type.model_config,
        __module__=model_type.__module__,
        **fields,
    )


def partial_target_type(target_type: type[Any]) -> type[Any]:
    """Return a relaxed schema type for chunk/page-level native structured output."""
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        return _partial_model_type(target_type)
    return target_type
