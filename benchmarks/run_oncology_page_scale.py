from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus" / "oncology_page_scale"
MANIFEST = CORPUS / "manifest.page_scale.json"
GENERATOR = CORPUS / "generate.py"
OUTPUT = CORPUS / "results.page_scale.json"
HOME_ENV = Path.home() / ".env"


def _load_home_env() -> None:
    if "GEMINI_API_KEY" in os.environ or not HOME_ENV.exists():
        return
    for line in HOME_ENV.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key == "GEMINI_API_KEY" and key not in os.environ:
            os.environ[key] = value.strip()


def _tag_value(tags: list[str], prefix: str) -> str | None:
    for tag in tags:
        if tag.startswith(prefix):
            return tag.removeprefix(prefix)
    return None


def _selected_config_names(
    *,
    models: list[str],
    strategies: list[str],
) -> list[str]:
    manifest = json.loads(MANIFEST.read_text())
    selected: list[str] = []
    wanted_models = set(models)
    wanted_strategies = set(strategies)
    for config in manifest["configs"]:
        tags = config.get("tags", [])
        model = _tag_value(tags, "model:")
        strategy = _tag_value(tags, "strategy:")
        if wanted_models and model not in wanted_models:
            continue
        if wanted_strategies and strategy not in wanted_strategies:
            continue
        selected.append(config["name"])
    return selected


def _page_count(case: dict[str, Any]) -> int:
    tag = _tag_value(case.get("tags", []), "pages:")
    if tag is None:
        raise ValueError(f"Case {case['name']!r} is missing a pages tag")
    return int(tag)


def _slope(rows: list[dict[str, Any]]) -> float:
    xs = [float(row["pages"]) for row in rows]
    ys = [float(row["latency_s"]) for row in rows]
    if len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return 0.0 if denominator == 0 else numerator / denominator


def _print_table(summary_rows: list[dict[str, Any]]) -> None:
    print("model\tstrategy\t5p\t10p\t15p\tslope_s_per_page\tstatus")
    for row in summary_rows:
        latencies = {item["pages"]: item["latency_s"] for item in row["latencies"]}
        print(
            "\t".join(
                [
                    str(row["model"]),
                    str(row["strategy"]),
                    f"{latencies.get(5, 0.0):.2f}s",
                    f"{latencies.get(10, 0.0):.2f}s",
                    f"{latencies.get(15, 0.0):.2f}s",
                    f"{float(row['slope_s_per_page']):.2f}",
                    str(row["status"]),
                ]
            )
        )


def _augment_report(output_path: Path) -> None:
    payload = json.loads(output_path.read_text())
    summary_rows: list[dict[str, Any]] = []
    for config_report in payload["configs"]:
        config = config_report["config"]
        tags = config.get("tags", [])
        rows = []
        for case_report in config_report["cases"]:
            case = case_report["case"]
            rows.append(
                {
                    "pages": _page_count(case),
                    "latency_s": case_report["latency_s"],
                    "api_calls": case_report["api_calls"],
                    "token_count": case_report["token_count"],
                    "error": case_report["error"],
                }
            )
        rows.sort(key=lambda item: item["pages"])
        summary_rows.append(
            {
                "config_name": config["name"],
                "model": _tag_value(tags, "model:"),
                "strategy": _tag_value(tags, "strategy:"),
                "latencies": rows,
                "slope_s_per_page": _slope(rows),
                "status": "succeeded"
                if all(not row["error"] for row in rows)
                else "partial failures",
            }
        )

    summary_rows.sort(key=lambda item: (str(item["model"]), str(item["strategy"])))
    payload["page_scale_summary"] = summary_rows
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print_table(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the oncology PDF page-scale latency benchmark."
    )
    parser.add_argument("--output", default=str(OUTPUT), help="Path to the JSON output report.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Optional model tag value to include. Repeat to select multiple models.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Optional strategy tag value to include. Repeat to select multiple strategies.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip regenerating the page-scale PDFs before running the benchmark.",
    )
    args = parser.parse_args()

    _load_home_env()
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY is required. Put it in ~/.env or export it in the shell.")

    if not args.skip_generate:
        subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT.parent, check=True)

    selected_configs = _selected_config_names(models=args.model, strategies=args.strategy)
    if not selected_configs:
        raise SystemExit("No benchmark configs matched the requested filters.")

    output_path = Path(args.output).resolve()
    cmd = [
        sys.executable,
        str(ROOT / "runner.py"),
        "--manifest",
        str(MANIFEST),
        "--output",
        str(output_path),
    ]
    for config_name in selected_configs:
        cmd.extend(["--config", config_name])

    completed = subprocess.run(cmd, cwd=ROOT.parent)
    if completed.returncode != 0:
        return completed.returncode

    _augment_report(output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
