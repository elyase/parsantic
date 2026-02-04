"""Corpus tests ported from BAML's Rust test suite.

Covers: malformed JSON repairs, code-as-string edge cases, and international/unicode handling.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from parsantic import parse
from parsantic.jsonish import ParseOptions, parse_jsonish

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_has_candidate(v, expected):
    assert v.candidates
    assert any(c.value == expected for c in v.candidates), v


def _assert_has_code_candidate(v, expected_code: str) -> None:
    assert v.candidates
    dicts = [c.value for c in v.candidates if isinstance(c.value, dict)]
    assert any(d.get("type") == "code" and d.get("code") == expected_code for d in dicts), dicts


def _assert_has_string_candidate(v, expected: str) -> None:
    assert v.candidates
    assert any(c.value == expected for c in v.candidates), v


# ===========================================================================
# Basics — ported from engine/baml-lib/jsonish/src/tests/test_basics.rs
# ===========================================================================


@pytest.mark.parametrize(
    ("raw", "expected", "is_done"),
    [
        pytest.param("[1, 2, 3,]", [1, 2, 3], True, id="trailing-comma-array"),
        pytest.param('{"key": "value",}', {"key": "value"}, True, id="trailing-comma-object"),
        pytest.param("[1, 2, 3", [1, 2, 3], False, id="unterminated-array"),
        pytest.param(
            '{"key": [1, 2, 3',
            {"key": [1, 2, 3]},
            False,
            id="unterminated-array-in-object",
        ),
        pytest.param('{"key": "value', {"key": "value"}, False, id="unterminated-string-in-object"),
    ],
)
def test_corpus_basics_malformed_json_repairs(raw: str, expected: object, is_done: bool):
    v = parse_jsonish(raw, options=ParseOptions(), is_done=is_done)
    _assert_has_candidate(v, expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            r"""
some text
```json
{
  "key": "value",
  "array": [1, 2, 3,],
  "object": {
    "key": "value"
  }
}
```
""",
            {"key": "value", "array": [1, 2, 3], "object": {"key": "value"}},
            id="markdown-fence-inner-trailing-comma",
        ),
        pytest.param(
            r"""
some text
```json
{
  key: "value",
  array: [1, 2, 3, 'some stinrg'   with quotes' /* test */],
  object: { // Test comment
    key: "value"
  },
}
```
""",
            {
                "key": "value",
                "array": [1, 2, 3, "some stinrg'   with quotes"],
                "object": {"key": "value"},
            },
            id="unquoted-keys-single-quotes-and-comments",
        ),
        pytest.param(
            r"""
{
  key: value with space,
  array: [1, 2, 3],
  object: {
    key: value
  }
}
""",
            {"key": "value with space", "array": [1, 2, 3], "object": {"key": "value"}},
            id="unquoted-values-with-spaces",
        ),
        pytest.param(
            r"""
{
  key: "test a long
thing with new

lines",
  array: [1, 2, 3],
  object: {
    key: value
  }
}
""",
            {
                "key": "test a long\nthing with new\n\nlines",
                "array": [1, 2, 3],
                "object": {"key": "value"},
            },
            id="quoted-multiline-string-in-unquoted-jsonish",
        ),
        pytest.param(
            r"""
{
  "my_field_0": true,
  "my_field_1": **First fragment, Another fragment**

Frag 2, frag 3. Frag 4, Frag 5, Frag 5.

Frag 6, the rest, of the sentence. Then i would quote something "like this" or this.

Then would add a summary of sorts.
}
""",
            {
                "my_field_0": True,
                "my_field_1": (
                    "**First fragment, Another fragment**\n\n"
                    "Frag 2, frag 3. Frag 4, Frag 5, Frag 5.\n\n"
                    'Frag 6, the rest, of the sentence. Then i would quote something "like this" or this.\n\n'
                    "Then would add a summary of sorts."
                ),
            },
            id="markdown-text-value-without-quotes",
        ),
    ],
)
def test_corpus_basics_unquoted_and_markdown_values(raw: str, expected: object):
    v = parse_jsonish(raw, options=ParseOptions(), is_done=True)
    _assert_has_candidate(v, expected)


# ===========================================================================
# Code — ported from engine/baml-lib/jsonish/src/tests/test_code.rs
# ===========================================================================


@pytest.mark.parametrize(
    ("raw", "expected_code"),
    [
        pytest.param(
            r"""
{
  "type": "code",
  "code": `print("Hello, world!")`
}
""",
            'print("Hello, world!")',
            id="backticks",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": 'print("Hello, world!")'
}
""",
            'print("Hello, world!")',
            id="single-quotes",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": "print(\"Hello, world!\")"
}
""",
            'print("Hello, world!")',
            id="double-quotes",
        ),
        pytest.param(
            r'''
{
  "type": "code",
  "code": """print("Hello, world!")"""
}
''',
            'print("Hello, world!")',
            id="triple-double-quotes",
        ),
        pytest.param(
            r'''
{
  "code": """
"Hello, world!"
"""
  "type": "code",
}
''',
            '"Hello, world!"',
            id="triple-quotes-contains-only-quoted-string",
        ),
        pytest.param(
            r'''
{
  "code": """
        def main():
          print("Hello, world!")
    """,
  "type": "code",
}
''',
            'def main():\n  print("Hello, world!")',
            id="triple-quotes-dedent",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": "print(\"Hello, world!
Goodbye, world!\")"
}
""",
            'print("Hello, world!\nGoodbye, world!")',
            id="unescaped-newline-double-quotes",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": `print("Hello, world!
Goodbye, world!")`
}
""",
            'print("Hello, world!\nGoodbye, world!")',
            id="unescaped-newline-backticks",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": 'print("Hello, world!
Goodbye, world!")'
}
""",
            'print("Hello, world!\nGoodbye, world!")',
            id="unescaped-newline-single-quotes",
        ),
        pytest.param(
            r'''
{
  "type": "code",
  "code": """print("Hello, world!
Goodbye, world!")"""
}
''',
            'print("Hello, world!\nGoodbye, world!")',
            id="unescaped-newline-triple-quotes",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": "print("Hello, world!")"
}
""",
            'print("Hello, world!")',
            id="unescaped-double-quotes-in-double-quotes",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": 'print('Hello, world!')'
}
""",
            "print('Hello, world!')",
            id="unescaped-single-quotes-in-single-quotes",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": `console.log(`Hello, world!`)`
}
""",
            "console.log(`Hello, world!`)",
            id="unescaped-backticks-in-backticks",
        ),
        pytest.param(
            r"""
{
  "type": "code",
  "code": `import { query } from './_generated/server';
import { v } from 'convex/values';

export default query(async (ctx) => {
  const posts = await ctx.db
    .query('posts')
    .order('desc')
    .collect();

  const postsWithDetails = await Promise.all(
    posts.map(async (post) => {
      // Fetch author information
      const author = await ctx.db.get(post.authorId);
      if (!author) {
        throw new Error('Author not found');
      }

      // Count upvotes
      const upvotes = await ctx.db
        .query('upvotes')
        .filter((q) => q.eq(q.field('postId'), post._id))
        .collect();

      return {
        id: post._id.toString(),
        title: post.title,
        content: post.content,
        author: {
          id: author._id.toString(),
          name: author.name,
        },
        upvoteCount: upvotes.length,
        createdAt: post._creationTime.toString(),
      };
    })
  );

  return postsWithDetails;
})`
}
""",
            """import { query } from './_generated/server';
import { v } from 'convex/values';

export default query(async (ctx) => {
  const posts = await ctx.db
    .query('posts')
    .order('desc')
    .collect();

  const postsWithDetails = await Promise.all(
    posts.map(async (post) => {
      // Fetch author information
      const author = await ctx.db.get(post.authorId);
      if (!author) {
        throw new Error('Author not found');
      }

      // Count upvotes
      const upvotes = await ctx.db
        .query('upvotes')
        .filter((q) => q.eq(q.field('postId'), post._id))
        .collect();

      return {
        id: post._id.toString(),
        title: post.title,
        content: post.content,
        author: {
          id: author._id.toString(),
          name: author.name,
        },
        upvoteCount: upvotes.length,
        createdAt: post._creationTime.toString(),
      };
    })
  );

  return postsWithDetails;
})""",
            id="large-backticks",
        ),
        pytest.param(
            r"""
Here's a comparison of TypeScript and Ruby code for checking the main Git branch using subprocesses:

{
  "code": ```
const { execSync } = require('child_process');

function getMainBranch(): string {
  try {
    // Try 'main' first
    const mainExists = execSync('git rev-parse --verify main 2>/dev/null', { encoding: 'utf8' });
    if (mainExists) return 'main';
  } catch {
    // Try 'master' if 'main' doesn't exist
    try {
      const masterExists = execSync('git rev-parse --verify master 2>/dev/null', { encoding: 'utf8' });
      if (masterExists) return 'master';
    } catch {
      throw new Error('Neither main nor master branch found');
    }
  }

  throw new Error('Unable to determine main branch');
}

// Usage
try {
  const mainBranch = getMainBranch();
  console.log(`Main branch is: ${mainBranch}`);
} catch (error) {
  console.error(`Error: ${error.message}`);
}
```,
  "type": "code",
}
""",
            """const { execSync } = require('child_process');

function getMainBranch(): string {
  try {
    // Try 'main' first
    const mainExists = execSync('git rev-parse --verify main 2>/dev/null', { encoding: 'utf8' });
    if (mainExists) return 'main';
  } catch {
    // Try 'master' if 'main' doesn't exist
    try {
      const masterExists = execSync('git rev-parse --verify master 2>/dev/null', { encoding: 'utf8' });
      if (masterExists) return 'master';
    } catch {
      throw new Error('Neither main nor master branch found');
    }
  }

  throw new Error('Unable to determine main branch');
}

// Usage
try {
  const mainBranch = getMainBranch();
  console.log(`Main branch is: ${mainBranch}`);
} catch (error) {
  console.error(`Error: ${error.message}`);
}""",
            id="triple-backticks-code-block",
        ),
        pytest.param(
            r"""
{
  "code": ```
`Hello, world!`
```,
  "type": "code",
}
""",
            "`Hello, world!`",
            id="triple-backticks-contains-backtick-string",
        ),
        pytest.param(
            r"""
{
  "code": ```typescript main.ts
    const async function main() {
      console.log("Hello, world!");
    }
```,
  "type": "code",
}
""",
            'const async function main() {\n  console.log("Hello, world!");\n}',
            id="triple-backticks-dedent-and-drop-info",
        ),
        pytest.param(
            r"""
{
  "code": ```
  { type: "code", code: "aaa", closing_terminators: }}}]])) }
```,
  "type": "code",
}
""",
            '{ type: "code", code: "aaa", closing_terminators: }}}]])) }',
            id="triple-backticks-json-terminators",
        ),
        pytest.param(
            r"""
```json
{
  "code": ```
  { type: "code", code: "aaa", closing_terminators: }}}]])) }
```,
  "type": "code",
}
```
""",
            '{ type: "code", code: "aaa", closing_terminators: }}}]])) }',
            id="triple-backticks-inside-json-fenced-block",
        ),
        pytest.param(
            r"""
```json
{
  "code": "```
const { execSync } = require('child_process');
```",
  "type": "code",
}
```
""",
            "```\nconst { execSync } = require('child_process');\n```",
            id="string-preserves-triple-backticks",
        ),
    ],
)
def test_corpus_code_code_as_string_variants(raw: str, expected_code: str):
    v = parse_jsonish(raw, options=ParseOptions(), is_done=True)
    _assert_has_code_candidate(v, expected_code)


# ===========================================================================
# I18n — ported from engine/baml-lib/jsonish/src/tests/test_international.rs
# ===========================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("¡Hola!", "¡Hola!", id="spanish-inverted-exclaim"),
        pytest.param("São Paulo", "São Paulo", id="portuguese-tilde"),
        pytest.param("Müller", "Müller", id="german-umlaut"),
        pytest.param("北京", "北京", id="chinese-characters"),
        pytest.param("السلام عليكم", "السلام عليكم", id="arabic-text"),
        pytest.param("Москва", "Москва", id="russian-cyrillic"),
        pytest.param("こんにちは", "こんにちは", id="japanese-hiragana"),
        pytest.param('"François"', "François", id="quoted-accented"),
        pytest.param("Café ☕", "Café ☕", id="accented-with-emoji"),
        pytest.param("naïve résumé", "naïve résumé", id="diacritics-combo"),
    ],
)
def test_corpus_i18n_unicode_strings_parse_jsonish(raw: str, expected: str):
    v = parse_jsonish(raw, options=ParseOptions(), is_done=True)
    _assert_has_string_candidate(v, expected)


class _Restaurant(BaseModel):
    name: str = Field(alias="nom")
    address: str = Field(alias="adresse")
    specialty: str = Field(alias="spécialité")
    stars: int = Field(alias="étoiles")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            r'{"nom": "Le Petit Café", "adresse": "Champs-Élysées", "spécialité": "crêpes bretonnes", "étoiles": 4}',
            ("Le Petit Café", "Champs-Élysées", "crêpes bretonnes", 4),
            id="restaurant-french-aliases-quoted",
        ),
        pytest.param(
            r'{nom: "Le Petit Café", adresse: Champs-Élysées, spécialité: "crêpes bretonnes", étoiles: 4}',
            ("Le Petit Café", "Champs-Élysées", "crêpes bretonnes", 4),
            id="restaurant-french-aliases-unquoted",
        ),
    ],
)
def test_corpus_i18n_unicode_field_aliases_via_parse(raw: str, expected: tuple[str, str, str, int]):
    res = parse(raw, _Restaurant, is_done=True)
    assert (res.value.name, res.value.address, res.value.specialty, res.value.stars) == expected


class _InternationalContact(BaseModel):
    first_name: str = Field(alias="prénom")
    family_name: str = Field(alias="família")
    city: str = Field(alias="città")
    street: str = Field(alias="straße")
    field: str = Field(alias="поле")
    data_field: str = Field(alias="フィールド")


@pytest.mark.parametrize(
    ("raw", "expected_first", "expected_city"),
    [
        pytest.param(
            r'{"prénom": "François", "família": "Silva", "città": "Milano", "straße": "Hauptstraße", "поле": "значение", "フィールド": "値"}',
            "François",
            "Milano",
            id="international-aliases-mixed-scripts",
        ),
        pytest.param(
            r"""Here is the contact information:
{
  "prénom": "José",
  "família": "González",
  "città": "Barcelona",
  "straße": "Königstraße",
  "поле": "текст",
  "フィールド": "データ"
}""",
            "José",
            "Barcelona",
            id="international-aliases-with-context",
        ),
    ],
)
def test_corpus_i18n_international_contact_aliases_via_parse(
    raw: str, expected_first: str, expected_city: str
):
    res = parse(raw, _InternationalContact, is_done=True)
    assert res.value.first_name == expected_first
    assert res.value.city == expected_city


class _Address(BaseModel):
    number: int = Field(alias="numéro")
    street: str = Field(alias="rue")
    city: str = Field(alias="ville")
    region: str = Field(alias="région")


class _Person(BaseModel):
    first_name: str = Field(alias="prénom")
    last_name: str = Field(alias="nom")
    age: int = Field(alias="âge")
    address: _Address = Field(alias="adresse")


def test_corpus_i18n_nested_models_with_accented_aliases_via_parse():
    raw = r'{"prénom": "François", "nom": "Müller", "âge": 35, "adresse": {"numéro": 42, "rue": "Champs-Élysées", "ville": "Paris", "région": "Île-de-France"}}'
    res = parse(raw, _Person, is_done=True)
    assert res.value.first_name == "François"
    assert res.value.last_name == "Müller"
    assert res.value.age == 35
    assert res.value.address.number == 42
    assert res.value.address.region == "Île-de-France"


class _FrenchProfile(BaseModel):
    first_name: str = Field(alias="prénom")
    last_name: str = Field(alias="nom")
    city: str = Field(alias="ville")
    profession: str = Field(alias="métier")


def test_corpus_i18n_unaccented_keys_match_accented_aliases_via_parse():
    raw = r'{"prenom": "François", "nom": "Dupont", "ville": "Paris", "metier": "Professeur"}'
    res = parse(raw, _FrenchProfile, is_done=True)
    assert res.value.first_name == "François"
    assert res.value.profession == "Professeur"


class _PortugueseData(BaseModel):
    location: str = Field(alias="localização")
    description: str = Field(alias="descrição")
    solution: str = Field(alias="solução")
    information: str = Field(alias="informação")


def test_corpus_i18n_unaccented_portuguese_keys_match_accented_aliases_via_parse():
    raw = r'{"localizacao": "São Paulo", "descricao": "Uma cidade grande", "solucao": "Transporte público", "informacao": "Dados importantes"}'
    res = parse(raw, _PortugueseData, is_done=True)
    assert res.value.location == "São Paulo"
    assert res.value.solution == "Transporte público"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("cafe", "café", id="unaccented-literal-cafe"),
        pytest.param("resume", "résumé", id="unaccented-literal-resume"),
        pytest.param("senor", "señor", id="unaccented-literal-senor"),
    ],
)
def test_corpus_i18n_unaccented_literals_match_accented_targets(raw: str, expected: str):
    res = parse(raw, Literal[expected], is_done=True)
    assert res.value == expected
