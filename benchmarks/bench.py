from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import pydantic
from pydantic import BaseModel, TypeAdapter

from parsantic import parse, parse_stream
from parsantic.jsonish import ParseOptions, parse_jsonish


class BenchResult(NamedTuple):
    name: str
    iters: int
    seconds_per_iter: float


def _run_bench(
    name: str,
    fn: Callable[[], object],
    *,
    target_total_seconds: float = 0.25,
    repeats: int = 7,
    max_iters: int = 1_000_000,
) -> BenchResult:
    iters = 1
    while True:
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= target_total_seconds:
            break
        if iters >= max_iters:
            break
        iters *= 2

    per_iter_samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed = time.perf_counter() - start
        per_iter_samples.append(elapsed / iters)

    return BenchResult(name=name, iters=iters, seconds_per_iter=statistics.median(per_iter_samples))


def _fmt_seconds(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.3f} ms"
    return f"{s:.3f} s"


def _print_table(results: list[BenchResult]) -> None:
    name_w = max(len(r.name) for r in results)
    it_w = max(len(str(r.iters)) for r in results)
    print(
        f"{'scenario'.ljust(name_w)}  {'iters'.rjust(it_w)}  {'median/op'.rjust(12)}  {'ops/s'.rjust(12)}"
    )
    print(f"{'-' * name_w}  {'-' * it_w}  {'-' * 12}  {'-' * 12}")
    for r in results:
        ops = 1.0 / r.seconds_per_iter if r.seconds_per_iter else float("inf")
        print(
            f"{r.name.ljust(name_w)}  {str(r.iters).rjust(it_w)}  {str(_fmt_seconds(r.seconds_per_iter)).rjust(12)}  {ops:12.0f}"
        )


@dataclass(frozen=True, slots=True)
class Payloads:
    strict_small: str
    strict_medium: str
    markdown: str
    fixing_simple: str
    fixing_heavy: str
    multi_objects: str
    yapping: str


def _payloads() -> Payloads:
    strict_small = '{"a": 1, "b": 2}'
    strict_medium = (
        '{"user": {"id": 123, "name": "Ada Lovelace", "email": "ada@example.com", '
        '"tags": ["math", "programming", "history"], '
        '"prefs": {"newsletter": true, "theme": "dark"}}, '
        '"events": [{"type": "click", "ts": 1700000000, "meta": {"x": 1, "y": 2}},'
        '{"type": "scroll", "ts": 1700000001, "meta": {"dx": 3, "dy": 4}}]}'
    )
    markdown = f"Here you go:\n```json\n{strict_medium}\n```\n"
    fixing_simple = "{a: 1, b: 2,}"
    fixing_heavy = r"""
// comment
{
  a: "1",
  b: 2,
  // missing comma
  c: 1/2
  d: "$1,234.56",
  nested: {
    msg: "hello \"world\"\nnext",
    ok: true,
  },
}
""".strip()
    multi_objects = '{"a": 1}\n{"a": 2}\n{"a": 3}'
    yapping = f"I think the answer is:\n\n{fixing_heavy}\n\nThanks!"
    return Payloads(
        strict_small=strict_small,
        strict_medium=strict_medium,
        markdown=markdown,
        fixing_simple=fixing_simple,
        fixing_heavy=fixing_heavy,
        multi_objects=multi_objects,
        yapping=yapping,
    )


class ObjSmall(BaseModel):
    a: int
    b: int


class ObjMedium(BaseModel):
    user: dict
    events: list[dict]


class ObjA(BaseModel):
    a: int


class FixingNested(BaseModel):
    msg: str
    ok: bool


class ObjFixingHeavy(BaseModel):
    a: int
    b: int
    c: float
    d: float
    nested: FixingNested


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmarks for parsantic (SAP).")
    parser.add_argument("--target-seconds", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    payloads = _payloads()

    adapter_small = TypeAdapter(ObjSmall)
    adapter_medium = TypeAdapter(ObjMedium)
    adapter_fixing = TypeAdapter(ObjFixingHeavy)

    parse_opts = ParseOptions()

    def sap_parse_small_adapter() -> object:
        return parse(payloads.strict_small, adapter_small).value

    def sap_parse_small_type() -> object:
        return parse(payloads.strict_small, ObjSmall).value

    def sap_parse_medium_adapter() -> object:
        return parse(payloads.strict_medium, adapter_medium).value

    def sap_parse_markdown_adapter() -> object:
        return parse(payloads.markdown, adapter_medium).value

    def sap_parse_fixing_simple_adapter() -> object:
        return parse(payloads.fixing_simple, adapter_small).value

    def sap_parse_fixing_heavy_adapter() -> object:
        return parse(payloads.fixing_heavy, adapter_fixing).value

    def sap_parse_yapping_adapter() -> object:
        return parse(payloads.yapping, adapter_fixing).value

    def sap_parse_multi_objects_adapter() -> object:
        return parse(payloads.multi_objects, list[ObjA]).value

    def jsonish_only_fixing_heavy() -> object:
        return parse_jsonish(payloads.fixing_heavy, options=parse_opts, is_done=True).value

    def streaming_message_small() -> object:
        sp = parse_stream(adapter_small)
        s = payloads.strict_small
        chunks = [s[i : i + 3] for i in range(0, len(s), 3)]
        for ch in chunks[:-1]:
            sp.feed(ch)
            sp.parse_partial()
        sp.feed(chunks[-1])
        return sp.finish().value

    def pydantic_validate_json_small() -> object:
        return adapter_small.validate_json(payloads.strict_small)

    def pydantic_validate_json_medium() -> object:
        return adapter_medium.validate_json(payloads.strict_medium)

    def json_loads_then_validate_small() -> object:
        return adapter_small.validate_python(json.loads(payloads.strict_small))

    def json_loads_then_validate_medium() -> object:
        return adapter_medium.validate_python(json.loads(payloads.strict_medium))

    results: list[BenchResult] = []
    results.append(
        _run_bench(
            "sap.parse strict_small (TypeAdapter reuse)",
            sap_parse_small_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse strict_small (type -> adapter each call)",
            sap_parse_small_type,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "pydantic.validate_json strict_small",
            pydantic_validate_json_small,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "json.loads + validate_python strict_small",
            json_loads_then_validate_small,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse strict_medium (TypeAdapter reuse)",
            sap_parse_medium_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "pydantic.validate_json strict_medium",
            pydantic_validate_json_medium,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "json.loads + validate_python strict_medium",
            json_loads_then_validate_medium,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse markdown fenced JSON",
            sap_parse_markdown_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse fixing_simple (unquoted keys, trailing comma)",
            sap_parse_fixing_simple_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse fixing_heavy (comments, fractions, nesting)",
            sap_parse_fixing_heavy_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse yapping (prefix+suffix)",
            sap_parse_yapping_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "sap.parse multi_objects -> list[ObjA]",
            sap_parse_multi_objects_adapter,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "jsonish only: parse_jsonish fixing_heavy",
            jsonish_only_fixing_heavy,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )
    results.append(
        _run_bench(
            "streaming: chunks+parse_partial+finish (strict_small)",
            streaming_message_small,
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )

    print("Environment")
    print(f"- python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"- platform: {platform.platform()}")
    print(f"- pydantic: {pydantic.__version__}")
    try:
        import msgspec  # type: ignore
    except Exception:
        print("- msgspec: (not installed)")
    else:
        print(f"- msgspec: {msgspec.__version__}")
        results.extend(
            [
                _run_bench(
                    "msgspec.json.decode + validate_python strict_small",
                    lambda: adapter_small.validate_python(
                        msgspec.json.decode(payloads.strict_small)
                    ),
                    target_total_seconds=args.target_seconds,
                    repeats=args.repeats,
                ),
                _run_bench(
                    "msgspec.json.decode + validate_python strict_medium",
                    lambda: adapter_medium.validate_python(
                        msgspec.json.decode(payloads.strict_medium)
                    ),
                    target_total_seconds=args.target_seconds,
                    repeats=args.repeats,
                ),
            ]
        )
    print()

    _print_table(results)


if __name__ == "__main__":
    main()
