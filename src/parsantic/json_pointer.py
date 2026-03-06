from __future__ import annotations


def escape_json_pointer_token(token: str) -> str:
    """Escape a single JSON Pointer token per RFC 6901 Section 3.

    Order matters: ``~`` must be escaped before ``/``.
    """
    return token.replace("~", "~0").replace("/", "~1")


def unescape_json_pointer_token(token: str) -> str:
    """Unescape a single JSON Pointer token per RFC 6901 Section 3.

    Order matters: ``~1`` must be unescaped before ``~0``.
    """
    return token.replace("~1", "/").replace("~0", "~")


def parse_json_pointer(path: str) -> list[str]:
    """Split a JSON Pointer into unescaped tokens.

    The root pointer ``""`` returns an empty list.
    """
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError(f"JSON Pointer must start with '/' or be empty, got: {path!r}")
    return [unescape_json_pointer_token(tok) for tok in path[1:].split("/")]


def build_json_pointer(tokens: list[str]) -> str:
    """Build a JSON Pointer string from raw tokens."""
    if not tokens:
        return ""
    return "/" + "/".join(escape_json_pointer_token(token) for token in tokens)


__all__ = [
    "build_json_pointer",
    "escape_json_pointer_token",
    "parse_json_pointer",
    "unescape_json_pointer_token",
]
