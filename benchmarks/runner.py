from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from parsantic.extract import Document, ExtractOptions, extract
from parsantic.extract.providers.base import InferenceRequest

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.metrics import BenchmarkMetrics, CaseMetrics, score_case, summarize_cases
else:
    from .metrics import BenchmarkMetrics, CaseMetrics, score_case, summarize_cases


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    document: str
    schema: str
    expected: str
    prompt: str = ""
    additional_context: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    model: str
    options: str | None = None
    prompt_prefix: str = ""
    additional_context: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    name: str
    root: str
    cases: tuple[BenchmarkCase, ...]
    configs: tuple[BenchmarkConfig, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCaseReport:
    config_name: str
    case: BenchmarkCase
    metrics: CaseMetrics
    latency_s: float
    api_calls: int
    token_count: int
    estimated_cost_usd: float
    output: Any
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkConfigReport:
    config: BenchmarkConfig
    summary: BenchmarkMetrics
    cases: tuple[BenchmarkCaseReport, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    manifest: BenchmarkManifest
    configs: tuple[BenchmarkConfigReport, ...] = field(default_factory=tuple)


class CountingProvider:
    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self.api_calls = 0
        self.estimated_prompt_tokens = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def _count_text_batch(self, prompts: list[str]) -> None:
        self.api_calls += 1
        self.estimated_prompt_tokens += sum(max(1, len(prompt) // 4) for prompt in prompts)

    def _count_media_batch(self, batch: list[InferenceRequest]) -> None:
        self.api_calls += 1
        self.estimated_prompt_tokens += sum(max(1, len(request.prompt) // 4) for request in batch)

    def infer(self, batch_prompts: list[str], **kwargs: Any) -> Any:
        self._count_text_batch(list(batch_prompts))
        return self._provider.infer(batch_prompts, **kwargs)

    async def ainfer(self, batch_prompts: list[str], **kwargs: Any) -> Any:
        self._count_text_batch(list(batch_prompts))
        if hasattr(self._provider, "ainfer"):
            return await self._provider.ainfer(batch_prompts, **kwargs)
        return self._provider.infer(batch_prompts, **kwargs)

    def infer_media(self, batch: list[InferenceRequest], **kwargs: Any) -> Any:
        self._count_media_batch(list(batch))
        return self._provider.infer_media(batch, **kwargs)

    async def ainfer_media(self, batch: list[InferenceRequest], **kwargs: Any) -> Any:
        self._count_media_batch(list(batch))
        if hasattr(self._provider, "ainfer_media"):
            return await self._provider.ainfer_media(batch, **kwargs)
        return self._provider.infer_media(batch, **kwargs)


def provider_from_model(model_id: str) -> Any:
    from parsantic.extract.providers.base import ProviderConfig
    from parsantic.extract.providers.factory import create_provider

    return create_provider(ProviderConfig(model_id=model_id))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_symbol(import_path: str) -> Any:
    module_name, _, symbol_name = import_path.partition(":")
    if not module_name or not symbol_name:
        raise ValueError(f"Expected import path in 'module:Symbol' format, got {import_path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _safe_eval_expression(expression: str) -> Any:
    parsantic_module = importlib.import_module("parsantic")
    extract_module = importlib.import_module("parsantic.extract")
    benchmarks_module = importlib.import_module("benchmarks")
    globals_map = {
        "__builtins__": {},
        "parsantic": parsantic_module,
        "extract": extract_module,
        "benchmarks": benchmarks_module,
        "provider_from_model": provider_from_model,
        "ExtractOptions": ExtractOptions,
    }
    return eval(expression, globals_map, {})


def _load_manifest(path: Path) -> BenchmarkManifest:
    data = _load_json(path)
    root = str((path.parent / data.get("root", ".")).resolve())
    cases = tuple(
        BenchmarkCase(
            name=item["name"],
            document=item["document"],
            schema=item["schema"],
            expected=item["expected"],
            prompt=item.get("prompt", ""),
            additional_context=item.get("additional_context"),
            tags=tuple(item.get("tags", [])),
        )
        for item in data.get("cases", [])
    )
    raw_configs = data.get("configs", [])
    configs = tuple(
        BenchmarkConfig(
            name=item["name"],
            model=item["model"],
            options=item.get("options"),
            prompt_prefix=item.get("prompt_prefix", ""),
            additional_context=item.get("additional_context"),
            tags=tuple(item.get("tags", [])),
        )
        for item in raw_configs
    )
    return BenchmarkManifest(
        name=data.get("name", path.stem), root=root, cases=cases, configs=configs
    )


def _join_text(*parts: str | None) -> str | None:
    rendered = [part.strip() for part in parts if part and part.strip()]
    if not rendered:
        return None
    return "\n\n".join(rendered)


def _load_document(case: BenchmarkCase, config: BenchmarkConfig, root: Path) -> Document:
    document_path = (root / case.document).resolve()
    prompt = (
        _join_text(
            f"Benchmark case: {case.name}",
            f"Benchmark config: {config.name}",
            config.prompt_prefix,
            case.prompt,
        )
        or ""
    )
    additional_context = _join_text(
        case.additional_context,
        config.additional_context,
        f"Benchmark case: {case.name}",
        f"Benchmark config: {config.name}",
    )
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        return Document.from_pdf(
            document_path,
            text=prompt,
            additional_context=additional_context,
            document_id=case.name,
        )
    return Document(
        text=document_path.read_text(),
        additional_context=additional_context,
        document_id=case.name,
    )


def _resolve_options(expr: str | None) -> ExtractOptions | None:
    if expr is None:
        return None
    value = _safe_eval_expression(expr)
    if value is None:
        return None
    if not isinstance(value, ExtractOptions):
        raise TypeError(
            f"options expression must return ExtractOptions, got {type(value).__name__}"
        )
    return value


def _resolve_model(expr: str) -> Any:
    value = _safe_eval_expression(expr)
    if isinstance(value, str):
        return provider_from_model(value)
    return value


def _zero_metrics(expected: Any) -> CaseMetrics:
    return score_case(expected, {})


def run_suite(
    manifest: BenchmarkManifest,
    *,
    configs: list[BenchmarkConfig] | None = None,
    fuzzy_threshold: float = 0.9,
) -> BenchmarkReport:
    root = Path(manifest.root)
    selected_configs = configs or list(manifest.configs)
    config_reports: list[BenchmarkConfigReport] = []

    for config in selected_configs:
        options = _resolve_options(config.options)
        case_reports: list[BenchmarkCaseReport] = []
        case_metrics: list[CaseMetrics] = []
        total_latency = 0.0
        total_api_calls = 0
        total_token_count = 0
        total_cost = 0.0

        for case in manifest.cases:
            provider = CountingProvider(_resolve_model(config.model))
            document = _load_document(case, config, root)
            target = _resolve_symbol(case.schema)
            expected = _load_json((root / case.expected).resolve())

            start = time.perf_counter()
            error: str | None = None
            output: Any = {}
            try:
                result = extract(document, target, model=provider, options=options)
                output = (
                    result.value.model_dump(mode="json")
                    if hasattr(result.value, "model_dump")
                    else result.value
                )
                metrics = score_case(expected, output, fuzzy_threshold=fuzzy_threshold)
            except Exception as exc:
                error = str(exc)
                metrics = _zero_metrics(expected)
            latency_s = time.perf_counter() - start

            total_latency += latency_s
            total_api_calls += provider.api_calls
            total_token_count += provider.estimated_prompt_tokens
            total_cost += 0.0
            case_metrics.append(metrics)
            case_reports.append(
                BenchmarkCaseReport(
                    config_name=config.name,
                    case=case,
                    metrics=metrics,
                    latency_s=latency_s,
                    api_calls=provider.api_calls,
                    token_count=provider.estimated_prompt_tokens,
                    estimated_cost_usd=0.0,
                    output=output,
                    error=error,
                )
            )

        summary = summarize_cases(
            case_metrics,
            latency_s=total_latency,
            api_calls=total_api_calls,
            token_count=total_token_count,
            estimated_cost_usd=total_cost,
        )
        config_reports.append(
            BenchmarkConfigReport(
                config=config,
                summary=summary,
                cases=tuple(case_reports),
            )
        )

    return BenchmarkReport(manifest=manifest, configs=tuple(config_reports))


def _report_to_json(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "manifest": {
            "name": report.manifest.name,
            "root": report.manifest.root,
            "cases": [asdict(case) for case in report.manifest.cases],
            "configs": [asdict(config) for config in report.manifest.configs],
        },
        "configs": [
            {
                "config": asdict(config_report.config),
                "summary": asdict(config_report.summary),
                "cases": [
                    {
                        "config_name": case_report.config_name,
                        "case": asdict(case_report.case),
                        "metrics": asdict(case_report.metrics),
                        "latency_s": case_report.latency_s,
                        "api_calls": case_report.api_calls,
                        "token_count": case_report.token_count,
                        "estimated_cost_usd": case_report.estimated_cost_usd,
                        "output": case_report.output,
                        "error": case_report.error,
                    }
                    for case_report in config_report.cases
                ],
            }
            for config_report in report.configs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Document-level extraction benchmark runner.")
    parser.add_argument(
        "--manifest", required=True, help="Path to the benchmark manifest JSON file."
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate the manifest without running extraction."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional config name(s) to run. Defaults to all manifest configs.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)

    if args.dry_run:
        payload = {
            "name": manifest.name,
            "root": manifest.root,
            "cases": [asdict(case) for case in manifest.cases],
            "configs": [asdict(config) for config in manifest.configs],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    selected_configs = list(manifest.configs)
    if args.config:
        wanted = set(args.config)
        selected_configs = [config for config in manifest.configs if config.name in wanted]
        missing = wanted - {config.name for config in selected_configs}
        if missing:
            raise SystemExit(f"Unknown config(s): {', '.join(sorted(missing))}")

    report = run_suite(manifest, configs=selected_configs, fuzzy_threshold=args.fuzzy_threshold)
    payload = _report_to_json(report)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
