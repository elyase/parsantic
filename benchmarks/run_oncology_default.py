from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "corpus" / "oncology" / "manifest.default.json"
OUTPUT = ROOT / "corpus" / "oncology" / "results.default.json"
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


def main() -> int:
    _load_home_env()
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit("GEMINI_API_KEY is required. Put it in ~/.env or export it in the shell.")
    cmd = [
        sys.executable,
        str(ROOT / "runner.py"),
        "--manifest",
        str(MANIFEST),
        "--output",
        str(OUTPUT),
    ]
    completed = subprocess.run(cmd, cwd=ROOT.parent)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
