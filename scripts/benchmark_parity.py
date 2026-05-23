#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

SCORE_RE = re.compile(r"\[score=[^\]]+\]")
LOCATION_RE = re.compile(r"^## \d+\. ([^ ]+)", re.MULTILINE)


@dataclass(frozen=True)
class CommandCase:
    name: str
    args: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare semble-rs against the Python semble reference for CLI parity and speed."
    )
    parser.add_argument(
        "--python-repo",
        type=Path,
        default=None,
        help="Path to the Python semble checkout. May also be set via SEMBLE_PYTHON_REPO.",
    )
    parser.add_argument(
        "--rust-bin",
        type=Path,
        default=Path("target/release/semble"),
        help="Path to the built Rust semble binary.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=12,
        help="Measured iterations per command after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per command before measuring.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.20,
        help="Fail when Rust p50 latency exceeds Python p50 by this ratio.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the JSON summary.",
    )
    return parser.parse_args()


def resolve_python_repo(arg_value: Path | None) -> Path:
    if arg_value is not None:
        return arg_value.resolve()
    raw_env = os.environ.get("SEMBLE_PYTHON_REPO")
    if raw_env:
        return Path(raw_env).resolve()
    raise SystemExit(
        "Python reference repo not configured. Pass --python-repo or set SEMBLE_PYTHON_REPO."
    )


def write_fixture(root: Path) -> None:
    files = {
        "auth.py": """def authenticate(token):
    return token == "secret"


def login(username, password):
    return authenticate(password)
""",
        "users.py": """class UserService:
    def authenticate_user(self, token):
        return authenticate(token)
""",
        "utils.py": """def format_name(first, last):
    return f"{first} {last}"
""",
        "README.md": """# Sample repo

Authentication helpers for CLI parity tests.
""",
    }
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def command_cases(fixture: Path) -> list[CommandCase]:
    fixture_str = str(fixture)
    return [
        CommandCase(
            name="search_bm25",
            args=["search", "authenticate token", fixture_str, "--top-k", "3", "--mode", "bm25"],
        ),
        CommandCase(
            name="search_hybrid",
            args=["search", "authenticate token", fixture_str, "--top-k", "3", "--mode", "hybrid"],
        ),
        CommandCase(
            name="find_related",
            args=["find-related", "auth.py", "1", fixture_str, "--top-k", "1"],
        ),
    ]


def run_command(argv: list[str]) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip(), elapsed_ms


def normalize_output(text: str) -> str:
    return SCORE_RE.sub("[score=<score>]", text)


def extract_locations(text: str) -> list[str]:
    return LOCATION_RE.findall(text)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * pct)))
    return ordered[index]


def benchmark_case(argv: list[str], warmup: int, iterations: int) -> dict[str, object]:
    warmup_samples = [run_command(argv) for _ in range(warmup)]
    samples = [run_command(argv) for _ in range(iterations)]
    first_code, first_stdout, first_stderr, _ = samples[0]
    timings = [elapsed_ms for _, _, _, elapsed_ms in samples]
    return {
        "exit_code": first_code,
        "stdout": first_stdout,
        "stderr": first_stderr,
        "warmup_ms": [elapsed_ms for _, _, _, elapsed_ms in warmup_samples],
        "timings_ms": timings,
        "p50_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 0.95),
        "locations": extract_locations(first_stdout),
        "normalized_stdout": normalize_output(first_stdout),
    }


def main() -> int:
    args = parse_args()
    python_repo = resolve_python_repo(args.python_repo)
    rust_bin = args.rust_bin.resolve()
    python_cli = python_repo / ".venv" / "bin" / "semble"
    if not python_repo.exists():
        raise SystemExit(f"Python repo not found: {python_repo}")
    if not rust_bin.exists():
        raise SystemExit(f"Rust binary not found: {rust_bin}")
    if not python_cli.exists():
        raise SystemExit(f"Python CLI not found: {python_cli}")

    with tempfile.TemporaryDirectory(prefix="semble-parity-") as temp_dir:
        fixture = Path(temp_dir)
        write_fixture(fixture)
        summary: dict[str, object] = {
            "fixture": fixture.name,
            "python_repo": str(python_repo),
            "python_cli": str(python_cli),
            "rust_bin": str(rust_bin),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "cases": {},
            "failures": [],
        }

        for case in command_cases(fixture):
            rust_argv = [str(rust_bin), *case.args]
            python_argv = [str(python_cli), *case.args]
            rust_result = benchmark_case(rust_argv, args.warmup, args.iterations)
            python_result = benchmark_case(python_argv, args.warmup, args.iterations)

            case_summary = {
                "rust": rust_result,
                "python": python_result,
                "ratio_p50": (
                    rust_result["p50_ms"] / python_result["p50_ms"]
                    if python_result["p50_ms"]
                    else None
                ),
            }
            summary["cases"][case.name] = case_summary

            if rust_result["exit_code"] != python_result["exit_code"]:
                summary["failures"].append(
                    f"{case.name}: exit code mismatch rust={rust_result['exit_code']} python={python_result['exit_code']}"
                )

            if rust_result["normalized_stdout"] != python_result["normalized_stdout"]:
                summary["failures"].append(f"{case.name}: normalized stdout mismatch")

            if rust_result["stderr"] != python_result["stderr"]:
                summary["failures"].append(f"{case.name}: stderr mismatch")

            ratio = case_summary["ratio_p50"]
            if ratio is not None and ratio > args.max_ratio:
                summary["failures"].append(
                    f"{case.name}: rust p50 {rust_result['p50_ms']:.2f}ms exceeded python p50 {python_result['p50_ms']:.2f}ms by ratio {ratio:.2f}"
                )

        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            for case_name, case_summary in summary["cases"].items():
                rust = case_summary["rust"]
                python = case_summary["python"]
                ratio = case_summary["ratio_p50"]
                print(case_name)
                print(
                    f"  rust   p50={rust['p50_ms']:.2f}ms p95={rust['p95_ms']:.2f}ms exit={rust['exit_code']}"
                )
                print(
                    f"  python p50={python['p50_ms']:.2f}ms p95={python['p95_ms']:.2f}ms exit={python['exit_code']}"
                )
                print(f"  ratio  p50={ratio:.2f}" if ratio is not None else "  ratio  p50=n/a")
            if summary["failures"]:
                print("\nfailures:")
                for failure in summary["failures"]:
                    print(f"  - {failure}")
            else:
                print("\nparity and performance checks passed")

        return 1 if summary["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
