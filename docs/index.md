# parsantic

Schema-aligned parsing (SAP) for Pydantic v2.

This project is a self-contained Python implementation inspired by BAML's SAP
approach, but designed to work directly with Pydantic schemas (no BAML DSL).

## Install

```bash
pip install parsantic
```

## Usage

```python
from pydantic import BaseModel
from parsantic import parse


class Resume(BaseModel):
    name: str
    email: str | None = None


resume = parse('{"name": "Ada Lovelace", "email": "ada@example.com"}', Resume).value
```

## Streaming

```python
from parsantic import parse_stream

sp = parse_stream(Resume)
sp.feed('{"name": "Ada')
partial = sp.parse_partial()  # best-effort partial result
sp.feed(' Lovelace", "email": "ada@example.com"}')
final = sp.finish().value
```

## Extraction (LLM‑style)

```python
from pydantic import BaseModel

from parsantic.extract import extract


class Resume(BaseModel):
    name: str
    email: str | None = None


result = extract(
    "Ada Lovelace <ada@example.com>",
    Resume,
    model="openai:gpt-4o-mini",
    provider_kwargs={"api_key": "..."},
)
print(result.value)
```

If you don’t have a provider plugin, you can pass a custom provider instance that implements
`infer(batch_prompts)`.

## Development

```bash
uv sync --group dev --group docs
make check
make test
make docs
```
