from __future__ import annotations

import pytest

from parsantic.jsonish import ParseOptions, parse_jsonish


def test_markdown_json_block_extracted():
    text = """```json
{
  "a": 1
}
```
"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    assert any(isinstance(c.value, dict) and c.value.get("a") == 1 for c in v.candidates)


def test_markdown_multi_item_includes_trailing_text_candidate_like_baml():
    # Mirrors intent of Rust test `test_markdown_multi_item_does_not_reparse_entire_input_as_string`.
    text = """```json
{"a": 1}
```

```json
{"b": 2}
```

i"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=False)
    assert v.candidates
    assert any(isinstance(c.value, str) and c.value.strip() == "i" for c in v.candidates), v


def test_multi_codeblocks_extract_multiple_json_values_like_baml():
    # Mirrors Rust `multi_json_parser::test_parse` / `markdown_parser::basic_parse` intent:
    # we should find multiple JSON structures within one response.
    text = """```json
{
  "a": 1
}
```

Also we've got a few more!
```python
print("Hello, world!")
```

```test json
["This is a test"]
```
"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    assert any(isinstance(c.value, dict) and c.value.get("a") == 1 for c in v.candidates)
    assert any(isinstance(c.value, list) and c.value == ["This is a test"] for c in v.candidates)


def test_utf8_between_blocks_like_baml():
    # Port of Rust `utf8_between_blocks` intent:
    # ensure non-ascii text between blocks doesn't break parsing.
    text = r"""
lorem ipsum

```json
"block1"
```

🌅🌞🏖️🏊‍♀️🐚🌴🍹🌺🏝️🌊👒😎👙🩴🐠🚤🍉🎣🎨📸🎉💃🕺🌙🌠🍽️🎶✨🌌🏕️🔥🌲🌌🌟💤

```json
"block2"
```

dolor sit amet
"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    vals = [c.value for c in v.candidates]
    assert "block1" in vals
    assert "block2" in vals
    assert any(isinstance(x, str) and x.strip() == "dolor sit amet" for x in vals)


def test_fence_like_text_inside_triple_backtick_string_parses_like_baml():
    # Port of Rust `fence_like_sequence_inside_triple_backtick_string_does_not_split_markdown_blocks`.
    text = r"""
```json
{
  "type": "code",
  "code": ```
  inside
  ```json
  not a markdown block
  ```,
}
```
"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert dicts, v
    code_vals = [d.get("code") for d in dicts]
    assert any(
        isinstance(s, str) and "```json" in s and "not a markdown block" in s for s in code_vals
    ), code_vals


def test_multiple_codeblocks_not_merged_when_fence_like_text_present_like_baml():
    text = r"""
```json
{
  "type": "code",
  "code": ```
  first block
  ```json
  still content
  ```,
}
```

```json
{"type": "code", "code": "second block"}
```
"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    # We should see at least two separate objects.
    assert len(dicts) >= 2


# ---- A3: plain fences, tilde fences, case-insensitive language tags ----


@pytest.mark.parametrize(
    "fence_open,fence_close,key,val",
    [
        ("```", "```", "key", "value"),
        ("~~~", "~~~", "x", 42),
        ("~~~json", "~~~", "x", 42),
        ("```JSON", "```", "a", 1),
        ("```Json", "```", "b", 2),
        ("```jsonc", "```", "c", 3),
        ("```json5", "```", "d", 4),
        ("```application/json", "```", "e", 5),
    ],
    ids=[
        "backtick-no-lang",
        "tilde-no-lang",
        "tilde-json",
        "JSON-upper",
        "Json-mixed",
        "jsonc",
        "json5",
        "application-json",
    ],
)
def test_fence_variants_extract_json(fence_open, fence_close, key, val):
    text = f'{fence_open}\n{{"{key}": {val if isinstance(val, int) else chr(34) + str(val) + chr(34)}}}\n{fence_close}\n'
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    assert any(isinstance(c.value, dict) and c.value.get(key) == val for c in v.candidates)


def test_multiple_json_blocks_with_commentary():
    """Multiple JSON blocks separated by prose text should all be extracted."""
    text = """Here is the first result:

```json
{"first": true}
```

And here is another analysis:

```
{"second": true}
```

That concludes the report.
"""
    v = parse_jsonish(text, options=ParseOptions(), is_done=True)
    assert v.candidates
    vals = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("first") is True for d in vals), f"Missing first block in {vals}"
    assert any(d.get("second") is True for d in vals), f"Missing second block in {vals}"
