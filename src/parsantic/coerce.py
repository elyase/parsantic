from __future__ import annotations

import enum
import re
import types as py_types
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from .jsonish import JsonishValue
from .types import CompletionState, ScoredValue, is_scalar


@dataclass(slots=True)
class CoerceOptions:
    """
    Options for coercion & scoring.
    """

    allow_substring_enum_match: bool = False


# Flag weights (lower is better), inspired by `engine/baml-lib/jsonish/src/deserializer/score.rs`.
_FLAG_WEIGHTS: dict[str, int] = {
    "fixed_json": 0,
    "fixed_array": 1,
    "markdown": 0,
    "markdown_array": 1,
    "closed_unclosed": 0,
    "grepped_json": 0,
    "grepped_array": 1,
    "markdown_tail": 5,
    "inferred_array": 5,
    "as_string": 2,
    "single_to_array": 1,
    "object_to_string": 2,
    "string_to_int": 1,
    "string_to_float": 1,
    "string_to_bool": 1,
    "float_to_int": 1,
    "default_from_missing": 100,
    "extra_key": 1,
    "substring_match": 2,
    "strip_punct": 3,
    "case_insensitive": 3,
    "accent_insensitive": 2,
    "key_normalized": 3,
    "implied_key": 2,
    "partial_model": 0,
    "partial_unvalidated": 50,
    "ambiguous_key": 10,
    "ambiguous_key_kept": 8,
    "ambiguous_enum": 20,
    "key_collision": 5,
    "max_depth_exceeded": 50,
}


def _adapter_target_type(adapter: TypeAdapter[Any]) -> Any | None:
    # Pydantic does not expose this as a stable public API yet.
    return getattr(adapter, "_type", None)


def _validate(adapter: TypeAdapter[Any], value: Any) -> Any:
    # Pydantic v2 defaults to validating models "by alias". For SAP, we want to accept
    # both aliases and field names after key-normalization.
    return adapter.validate_python(value, by_alias=True, by_name=True)


def coerce_jsonish_to_python(
    value: JsonishValue,
    adapter: TypeAdapter[Any],
    *,
    options: CoerceOptions,
    allow_partial: bool = False,
) -> ScoredValue:
    """
    Coerce JsonishValue candidates to a python object validated by a TypeAdapter.

    Strategy:
    - if AnyOf: coerce each candidate, validate, pick best score.
    - otherwise: attempt direct validation; if it fails, apply light coercions and retry.
    """
    if value.candidates:
        scored: list[ScoredValue] = []
        for cand in value.candidates:
            try:
                sv = coerce_jsonish_to_python(
                    cand, adapter, options=options, allow_partial=allow_partial
                )
            except Exception:
                continue
            # Early exit: score 0 means perfect match with no coercion flags.
            if sv.score == 0:
                return sv
            scored.append(sv)
        if not scored:
            # last resort: validate raw string
            raw_val = value.value if is_scalar(value.value) else value.raw
            try:
                validated = _validate(adapter, raw_val)
            except Exception:
                if allow_partial:
                    flags = ("object_to_string", "partial_unvalidated")
                    return ScoredValue(value=raw_val, flags=flags, score=_score_flags(flags))
                raise
            return ScoredValue(
                value=validated,
                flags=("object_to_string",),
                score=_score_flags(("object_to_string",)),
            )
        return _pick_best(scored)

    base_flags = tuple(value.fixes)

    # Fast path: already valid
    try:
        validated = _validate(adapter, value.value)
        return ScoredValue(value=validated, flags=base_flags, score=_score_flags(base_flags))
    except Exception:
        pass

    # Enum/Literal matching for string inputs
    if isinstance(value.value, str):
        enum_result = _try_enum_literal_match(adapter, value.value, options)
        if enum_result is not None:
            validated, enum_flags = enum_result
            flags = base_flags + tuple(enum_flags)
            return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))

    # Single-field model "implied key" coercion (ported from BAML class coercer):
    # if a model has exactly one field, treat the entire value as that field when needed.
    implied = _try_implied_single_field_model(adapter, value.value)
    if implied is not None:
        validated, implied_flags = implied
        flags = base_flags + tuple(implied_flags)
        return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))

    # Schema-aligned dict coercion for Pydantic models: normalize/match keys before validating.
    if isinstance(value.value, dict):
        mapped, key_flags = _coerce_model_keys(adapter, value.value)
        if mapped is not value.value:
            try:
                validated = _validate(adapter, mapped)
                flags = base_flags + tuple(key_flags)
                return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))
            except Exception:
                pass

        # Schema-aligned value coercion for models: coerce individual field values based on the
        # Pydantic field annotations (e.g. fractions/currency strings -> floats).
        coerced_mapped, value_flags = _coerce_model_values(
            adapter, mapped, options=options, allow_partial=allow_partial
        )
        if coerced_mapped is not mapped:
            try:
                validated = _validate(adapter, coerced_mapped)
                flags = base_flags + tuple(key_flags) + tuple(value_flags)
                return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))
            except Exception:
                pass

    # Streaming/partial: for models, return a best-effort partial dict rather than error.
    if allow_partial and isinstance(value.value, dict):
        partial, partial_flags = _partial_model_dict(adapter, value.value, options=options)
        flags = base_flags + tuple(partial_flags) + ("partial_model",)
        return ScoredValue(value=partial, flags=flags, score=_score_flags(flags) + 1)

    # If the input is a string but schema expects number/bool/null, try casting.
    if isinstance(value.value, str):
        s = value.value.strip()
        # int
        i = _try_int(s)
        if i is not None:
            try:
                validated = _validate(adapter, i)
                flags = base_flags + ("string_to_int",)
                return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))
            except Exception:
                pass
        f = _try_float(s)
        if f is not None:
            try:
                validated = _validate(adapter, f)
                flags = base_flags + ("string_to_float",)
                return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))
            except Exception:
                pass
        b = _try_bool(s)
        if b is not None:
            try:
                validated = _validate(adapter, b)
                flags = base_flags + ("string_to_bool",)
                return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))
            except Exception:
                pass

    # Single-to-array coercion: if schema expects list, wrap and retry.
    if not isinstance(value.value, list):
        try:
            validated = _validate(adapter, [value.value])
        except Exception:
            validated = None
        if validated is not None:
            flags = base_flags + ("single_to_array",)
            return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))

    # Fallback: try validating raw string representation (object->string)
    raw_val = value.value if is_scalar(value.value) else value.raw
    try:
        validated = _validate(adapter, raw_val)
    except Exception:
        if allow_partial:
            flags = base_flags + ("object_to_string", "partial_unvalidated")
            return ScoredValue(value=raw_val, flags=flags, score=_score_flags(flags))
        raise
    flags = base_flags + ("object_to_string",)
    return ScoredValue(value=validated, flags=flags, score=_score_flags(flags))


# ---------------------------------------------------------------------------
# Recursive coercion for arbitrary types (B2)
# ---------------------------------------------------------------------------


def _coerce_to_type(
    value: Any, target_type: Any, options: CoerceOptions, *, _depth: int = 0
) -> ScoredValue:
    """
    Recursively coerce *value* toward *target_type*.

    Handles: scalars, Enum, Literal, list[T], dict[K,V], tuple[T,...],
    Union[A,B,...], and nested BaseModel.
    """
    if _depth > 100:
        return ScoredValue(
            value=value,
            flags=("max_depth_exceeded",),
            score=_score_flags(("max_depth_exceeded",)),
        )

    adapter = TypeAdapter(target_type)

    # Fast path: already valid
    try:
        validated = _validate(adapter, value)
        return ScoredValue(value=validated, flags=(), score=0)
    except Exception:
        pass

    origin = get_origin(target_type)
    args = get_args(target_type)

    # --- Union ---
    if origin in {Union, py_types.UnionType}:
        return _coerce_union(value, args, options, _depth=_depth + 1)

    # --- list / List[T] ---
    if origin is list and args:
        return _coerce_list(value, args[0], options, _depth=_depth + 1)

    # --- dict / Dict[K, V] ---
    if origin is dict and len(args) == 2:
        return _coerce_dict(value, args[0], args[1], options, _depth=_depth + 1)

    # --- tuple / Tuple[T, ...] ---
    if origin is tuple and args:
        return _coerce_tuple(value, args, options, _depth=_depth + 1)

    # --- Enum ---
    if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
        if isinstance(value, str):
            candidates = [m.value if isinstance(m.value, str) else m.name for m in target_type]
            matched, flags = _match_enum_value(value, candidates, options)
            if matched is not None:
                # find the enum member
                for m in target_type:
                    mval = m.value if isinstance(m.value, str) else m.name
                    if mval == matched:
                        return ScoredValue(value=m, flags=flags, score=_score_flags(flags))

    # --- Literal ---
    if origin is Literal:
        if isinstance(value, str) and args:
            str_args = [a for a in args if isinstance(a, str)]
            if str_args:
                matched, flags = _match_enum_value(value, str_args, options)
                if matched is not None:
                    return ScoredValue(value=matched, flags=flags, score=_score_flags(flags))

    # --- BaseModel ---
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        if isinstance(value, dict):
            jv = JsonishValue(value=value, completion=CompletionState.COMPLETE, raw=str(value))
            return coerce_jsonish_to_python(jv, adapter, options=options)

    # --- scalar coercions (string -> int/float/bool) ---
    if isinstance(value, str):
        s = value.strip()
        i = _try_int(s)
        if i is not None:
            try:
                validated = _validate(adapter, i)
                return ScoredValue(
                    value=validated,
                    flags=("string_to_int",),
                    score=_score_flags(("string_to_int",)),
                )
            except Exception:
                pass
        f = _try_float(s)
        if f is not None:
            try:
                validated = _validate(adapter, f)
                return ScoredValue(
                    value=validated,
                    flags=("string_to_float",),
                    score=_score_flags(("string_to_float",)),
                )
            except Exception:
                pass
        b = _try_bool(s)
        if b is not None:
            try:
                validated = _validate(adapter, b)
                return ScoredValue(
                    value=validated,
                    flags=("string_to_bool",),
                    score=_score_flags(("string_to_bool",)),
                )
            except Exception:
                pass

    # --- float -> int safe coercion ---
    if isinstance(value, float) and target_type is int:
        if abs(value - round(value)) < 1e-9:
            return ScoredValue(
                value=int(round(value)),
                flags=("float_to_int",),
                score=_score_flags(("float_to_int",)),
            )

    # Last resort: try adapter directly (may raise)
    validated = _validate(adapter, value)
    return ScoredValue(value=validated, flags=(), score=0)


def _coerce_union(
    value: Any, variants: tuple[Any, ...], options: CoerceOptions, *, _depth: int = 0
) -> ScoredValue:
    """Try each variant of a Union, return best-scoring result."""
    scored: list[ScoredValue] = []
    for vtype in variants:
        if vtype is type(None):
            if value is None:
                return ScoredValue(value=None, flags=(), score=0)
            continue
        try:
            sv = _coerce_to_type(value, vtype, options, _depth=_depth)
            scored.append(sv)
        except Exception:
            continue
    if not scored:
        raise ValueError(f"Cannot coerce {value!r} to Union{variants!r}")
    return _pick_best(scored)


def _coerce_list(
    value: Any, elem_type: Any, options: CoerceOptions, *, _depth: int = 0
) -> ScoredValue:
    """Coerce each element of a list toward elem_type."""
    if not isinstance(value, list):
        # wrap scalar as single-element list
        sv = _coerce_to_type(value, elem_type, options, _depth=_depth)
        combined = sv.flags + ("single_to_array",)
        return ScoredValue(
            value=[sv.value],
            flags=combined,
            score=_score_flags(combined),
        )
    out: list[Any] = []
    all_flags: list[str] = []
    total_score = 0
    for item in value:
        sv = _coerce_to_type(item, elem_type, options, _depth=_depth)
        out.append(sv.value)
        all_flags.extend(sv.flags)
        total_score += sv.score
    return ScoredValue(value=out, flags=tuple(all_flags), score=total_score)


def _coerce_dict(
    value: Any, key_type: Any, val_type: Any, options: CoerceOptions, *, _depth: int = 0
) -> ScoredValue:
    """Coerce dict values (and keys if K is not str)."""
    if not isinstance(value, dict):
        raise ValueError(f"Cannot coerce {type(value).__name__} to dict")
    out: dict[Any, Any] = {}
    all_flags: list[str] = []
    total_score = 0
    for k, v in value.items():
        if key_type is not str:
            sk = _coerce_to_type(k, key_type, options, _depth=_depth)
            k = sk.value
            all_flags.extend(sk.flags)
            total_score += sk.score
        sv = _coerce_to_type(v, val_type, options, _depth=_depth)
        out[k] = sv.value
        all_flags.extend(sv.flags)
        total_score += sv.score
    return ScoredValue(value=out, flags=tuple(all_flags), score=total_score)


def _coerce_tuple(
    value: Any, args: tuple[Any, ...], options: CoerceOptions, *, _depth: int = 0
) -> ScoredValue:
    """Coerce tuple elements."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Cannot coerce {type(value).__name__} to tuple")
    # Handle Tuple[T, ...] (homogeneous)
    if len(args) == 2 and args[1] is Ellipsis:
        elem_type = args[0]
        out: list[Any] = []
        all_flags: list[str] = []
        total_score = 0
        for item in value:
            sv = _coerce_to_type(item, elem_type, options, _depth=_depth)
            out.append(sv.value)
            all_flags.extend(sv.flags)
            total_score += sv.score
        return ScoredValue(value=tuple(out), flags=tuple(all_flags), score=total_score)
    # Fixed-length tuple
    if len(value) != len(args):
        raise ValueError(f"Tuple length mismatch: got {len(value)}, expected {len(args)}")
    out_fixed: list[Any] = []
    all_flags_f: list[str] = []
    total_score_f = 0
    for item, t in zip(value, args, strict=True):
        sv = _coerce_to_type(item, t, options, _depth=_depth)
        out_fixed.append(sv.value)
        all_flags_f.extend(sv.flags)
        total_score_f += sv.score
    return ScoredValue(value=tuple(out_fixed), flags=tuple(all_flags_f), score=total_score_f)


# ---------------------------------------------------------------------------
# Enum/Literal matching (B1)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _strip_accents(s: str) -> str:
    """Remove diacritical marks / combining characters."""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if not unicodedata.combining(ch))


def _strip_punct(s: str) -> str:
    """Strip punctuation (non-word, non-space) characters."""
    return _PUNCT_RE.sub("", s)


def _match_enum_value(
    input_str: str, candidates: list[str], options: CoerceOptions
) -> tuple[str | None, tuple[str, ...]]:
    """
    Graduated enum/literal matching.

    Returns (matched_value, flags) or (None, ()) if no match.

    Matching levels (in priority order):
    1. Exact match (no flags)
    2. Case-insensitive match (flag: case_insensitive)
    3. Punctuation-stripped match (flag: strip_punct)
    4. Accent-insensitive match (flag: accent_insensitive)
    5. Substring match (flag: substring_match) -- only if options.allow_substring_enum_match
    """
    # Level 1: exact match
    for c in candidates:
        if input_str == c:
            return c, ()

    # Level 2: case-insensitive
    input_lower = input_str.lower()
    matches = [c for c in candidates if c.lower() == input_lower]
    if len(matches) == 1:
        return matches[0], ("case_insensitive",)
    if len(matches) > 1:
        pick = sorted(matches)[0]
        return pick, ("case_insensitive", "ambiguous_enum")

    # Level 3: punctuation-stripped (case-insensitive)
    input_stripped = _strip_punct(input_str).lower()
    matches = [c for c in candidates if _strip_punct(c).lower() == input_stripped]
    if len(matches) == 1:
        return matches[0], ("strip_punct",)
    if len(matches) > 1:
        pick = sorted(matches)[0]
        return pick, ("strip_punct", "ambiguous_enum")

    # Level 4: accent-insensitive (case-insensitive, punct-stripped)
    input_no_accent = _strip_accents(_strip_punct(input_str)).lower()
    matches = [c for c in candidates if _strip_accents(_strip_punct(c)).lower() == input_no_accent]
    if len(matches) == 1:
        return matches[0], ("accent_insensitive",)
    if len(matches) > 1:
        pick = sorted(matches)[0]
        return pick, ("accent_insensitive", "ambiguous_enum")

    # Level 5: substring match
    if options.allow_substring_enum_match:
        matches = [c for c in candidates if input_lower in c.lower() or c.lower() in input_lower]
        if len(matches) == 1:
            return matches[0], ("substring_match",)
        if len(matches) > 1:
            pick = sorted(matches)[0]
            return pick, ("substring_match", "ambiguous_enum")

    return None, ()


def _try_enum_literal_match(
    adapter: TypeAdapter[Any], value: str, options: CoerceOptions
) -> tuple[Any, tuple[str, ...]] | None:
    """
    Try to match a string value against Enum or Literal target types in the adapter.
    Returns (validated_value, flags) or None.
    """
    target_type = _adapter_target_type(adapter)
    if target_type is None:
        return None

    # Check for Enum
    if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
        candidates_map: dict[str, Any] = {}
        for m in target_type:
            key = m.value if isinstance(m.value, str) else m.name
            candidates_map[key] = m
        matched, flags = _match_enum_value(value, list(candidates_map.keys()), options)
        if matched is not None:
            return candidates_map[matched], flags
        return None

    # Check for Literal
    origin = get_origin(target_type)
    if origin is Literal:
        args = get_args(target_type)
        str_args = [a for a in args if isinstance(a, str)]
        if str_args:
            matched, flags = _match_enum_value(value, str_args, options)
            if matched is not None:
                return matched, flags
    return None


# ---------------------------------------------------------------------------
# Scoring & selection
# ---------------------------------------------------------------------------


def _score_flags(flags: Iterable[str]) -> int:
    return sum(_FLAG_WEIGHTS.get(f, 5) for f in flags)


def _pick_best(scored: list[ScoredValue]) -> ScoredValue:
    # B5: Deterministic: sort by (score, len(flags), index) - use enumeration index
    # instead of repr(value) for deterministic tie-breaking by candidate generation order.
    return sorted(enumerate(scored), key=lambda pair: (pair[1].score, len(pair[1].flags), pair[0]))[
        0
    ][1]


# ---------------------------------------------------------------------------
# Scalar coercion helpers
# ---------------------------------------------------------------------------


def _try_int(s: str) -> int | None:
    """
    B3: Safe int coercion.
    - Direct int parse (e.g. "123")
    - Float-ish only when within epsilon of an integer (e.g. "3.0" -> 3, but "1.4" -> None)
    - Emit float_to_int flag implicitly (caller should add when float path used)
    """
    s = s.rstrip(",")
    try:
        return int(s)
    except Exception:
        pass
    # float-ish: only allow when the float is within epsilon of an integer
    try:
        fval = float(s)
        if abs(fval - round(fval)) < 1e-9:
            return int(round(fval))
        # NOT safe to round -- return None
        return None
    except Exception:
        return None


_FRACTION = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\s*$")


def _try_float(s: str) -> float | None:
    s = s.rstrip(",")
    try:
        return float(s)
    except Exception:
        pass
    m = _FRACTION.match(s)
    if m:
        num = float(m.group(1))
        den = float(m.group(2))
        if den == 0:
            return None
        return num / den
    comma = _float_from_comma_separated(s)
    if comma is not None:
        return comma
    return None


def _try_bool(s: str) -> bool | None:
    sl = s.lower()
    if sl in {"true", "yes", "y", "1"}:
        return True
    if sl in {"false", "no", "n", "0"}:
        return False
    return None


def normalize_key(s: str) -> str:
    # Match BAML's "strip punctuation + case/accents" spirit (see match_string.rs).
    s = s.strip()
    # For key matching we intentionally normalize separators aggressively
    # (spaces/underscores/hyphens collapse) to better match LLM output.
    s = "".join(ch for ch in s if ch.isalnum())
    # Handle ligatures/special cases similar to Rust `remove_accents`.
    s = (
        s.replace("\u00df", "ss")
        .replace("\u00e6", "ae")
        .replace("\u00c6", "AE")
        .replace("\u00f8", "o")
        .replace("\u00d8", "O")
        .replace("\u0153", "oe")
        .replace("\u0152", "OE")
    )
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


_NUM_WITH_COMMAS = re.compile(
    r"([-+]?)\$?(?:\d+(?:,\d+)*(?:\.\d+)?|\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?"
)


def _strip_currency_symbols(s: str) -> str:
    # Rough equivalent of Rust regex \p{Sc}.
    return "".join(ch for ch in s if unicodedata.category(ch) != "Sc")


def _float_from_comma_separated(text: str) -> float | None:
    """
    Port of Rust `float_from_comma_separated`:
    - find exactly one number-ish match in the input
    - remove commas and currency symbols
    - parse as float
    """
    matches = list(_NUM_WITH_COMMAS.finditer(text))
    if len(matches) != 1:
        return None
    number_str = matches[0].group(0)
    without_commas = number_str.replace(",", "")
    without_currency = _strip_currency_symbols(without_commas)
    try:
        return float(without_currency)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model-level coercion helpers
# ---------------------------------------------------------------------------


def _partial_model_dict(
    adapter: TypeAdapter[Any], value: dict[str, Any], *, options: CoerceOptions
) -> tuple[dict[str, Any], list[str]]:
    """
    Best-effort field-by-field validation for streaming partials.

    This intentionally does *not* attempt to create a BaseModel instance (since
    required fields may be missing). Instead it returns a dict containing only
    fields that validated successfully.
    """
    model_type = getattr(adapter, "_type", None)
    if (
        model_type is None
        or not isinstance(model_type, type)
        or not issubclass(model_type, BaseModel)
    ):
        return dict(value), []

    mapped, key_flags = _coerce_model_keys(adapter, value)
    out: dict[str, Any] = {}
    flags: list[str] = list(key_flags)
    for field_name, field in model_type.model_fields.items():
        if field_name not in mapped:
            continue
        raw_val = mapped[field_name]
        try:
            coerced = coerce_jsonish_to_python(
                JsonishValue(value=raw_val, completion=CompletionState.COMPLETE, raw=str(raw_val)),
                TypeAdapter(field.annotation),
                options=options,
            )
            out[field_name] = coerced.value
            flags.extend(coerced.flags)
        except Exception:
            continue
    return out, flags


def _coerce_model_values(
    adapter: TypeAdapter[Any],
    value: dict[str, Any],
    *,
    options: CoerceOptions,
    allow_partial: bool,
) -> tuple[dict[str, Any], list[str]]:
    model_type = getattr(adapter, "_type", None)
    if (
        model_type is None
        or not isinstance(model_type, type)
        or not issubclass(model_type, BaseModel)
    ):
        return value, []

    out: dict[str, Any] = dict(value)
    flags: list[str] = []

    for field_name, field in model_type.model_fields.items():
        if field_name not in value:
            continue
        raw_val = value[field_name]
        try:
            coerced = coerce_jsonish_to_python(
                JsonishValue(value=raw_val, completion=CompletionState.COMPLETE, raw=str(raw_val)),
                TypeAdapter(field.annotation),
                options=options,
                allow_partial=allow_partial,
            )
        except Exception:
            continue
        out[field_name] = coerced.value
        flags.extend(coerced.flags)

    if out == value:
        return value, flags if flags else []
    return out, flags


def _coerce_model_keys(
    adapter: TypeAdapter[Any], value: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """
    Best-effort mapping of LLM keys -> Pydantic field names.

    BAML does this with `match_string` + aliases; here we approximate by:
    - exact match on field name/alias/validation_alias (B4)
    - normalized match (strip punctuation, remove accents, lowercase)
    - collision handling: first match wins, flag `key_collision` (B4)
    - extra="allow" models preserve unmatched keys (B4)
    """
    model_type = getattr(adapter, "_type", None)
    if (
        model_type is None
        or not isinstance(model_type, type)
        or not issubclass(model_type, BaseModel)
    ):
        return value, []

    # Check if model allows extras
    model_config = getattr(model_type, "model_config", {})
    allows_extra = model_config.get("extra", None) == "allow"

    # Build normalized alias -> field_name map, tracking collisions.
    norm_to_field: dict[str, str] = {}
    collisions: set[str] = set()
    # Also build exact-match lookups for aliases and validation_aliases
    exact_alias_to_field: dict[str, str] = {}

    for field_name, field in model_type.model_fields.items():
        candidates: list[str] = [field_name]
        if field.alias and field.alias not in candidates:
            candidates.append(field.alias)
        # B4: also check validation_alias
        val_alias = field.validation_alias
        if val_alias is not None and isinstance(val_alias, str) and val_alias not in candidates:
            candidates.append(val_alias)

        for cand in candidates:
            # register exact alias lookup
            if cand != field_name:
                exact_alias_to_field[cand] = field_name
            nk = normalize_key(cand)
            if nk in norm_to_field and norm_to_field[nk] != field_name:
                collisions.add(nk)
            else:
                norm_to_field[nk] = field_name

    out: dict[str, Any] = {}
    flags: list[str] = []
    for k, v in value.items():
        # Prefer exact matches first (field name).
        if k in model_type.model_fields and k not in out:
            out[k] = v
            continue

        # Exact alias / validation_alias match
        if k in exact_alias_to_field:
            mapped_field = exact_alias_to_field[k]
            if mapped_field not in out:
                out[mapped_field] = v
                if mapped_field != k:
                    flags.append("key_normalized")
                continue
            else:
                # B4: key collision - first match wins
                flags.append("key_collision")
                continue

        nk = normalize_key(k)
        if nk in collisions:
            if allows_extra:
                # Preserve the original key for extra="allow" models
                out[k] = v
                flags.append("ambiguous_key_kept")
            else:
                flags.append("ambiguous_key")
            continue
        mapped = norm_to_field.get(nk)
        if mapped is None:
            if allows_extra:
                # B4: preserve unmatched keys for extra="allow" models
                out[k] = v
            else:
                flags.append("extra_key")
            continue
        if mapped in out:
            # B4: key collision - first match wins
            flags.append("key_collision")
            continue
        if mapped != k:
            flags.append("key_normalized")
        out[mapped] = v

    if out == value:
        return value, flags if flags else []
    return out, flags


def _try_implied_single_field_model(
    adapter: TypeAdapter[Any], value: Any
) -> tuple[Any, list[str]] | None:
    model_type = getattr(adapter, "_type", None)
    if (
        model_type is None
        or not isinstance(model_type, type)
        or not issubclass(model_type, BaseModel)
    ):
        return None
    if len(model_type.model_fields) != 1:
        return None

    (field_name, field) = next(iter(model_type.model_fields.items()))

    # If the value is already a dict that contains the field (or its alias), don't do implied-key.
    if isinstance(value, dict):
        if field_name in value or (field.alias and field.alias in value):
            return None

    try:
        validated = TypeAdapter(model_type).validate_python({field_name: value})
    except Exception:
        return None
    return validated, ["implied_key"]
