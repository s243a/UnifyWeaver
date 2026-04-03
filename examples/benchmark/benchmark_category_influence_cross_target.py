#!/usr/bin/env python3
"""
Benchmark category influence propagation across the C# query engine and
generated Rust/Go binaries.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
GENERATOR = ROOT / "examples" / "benchmark" / "generate_pipeline.py"
FACTS_PATH = BENCH_DIR / "10k" / "facts.pl"


@dataclass
class RunResult:
    target: str
    scale: str
    times: list[float]
    stdout_sha256: str
    row_count: int
    stderr: str

    @property
    def median(self) -> float:
        return statistics.median(self.times)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", default="300,1k,5k,10k")
    parser.add_argument(
        "--targets",
        default="csharp-query,rust-dfs,go-dfs",
        help="Comma-separated targets: csharp-query,rust-dfs,go-dfs",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, env=env)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def scale_sort_key(scale: str) -> tuple[int, str]:
    digits = "".join(ch for ch in scale if ch.isdigit())
    suffix = "".join(ch for ch in scale if not ch.isdigit())
    if not digits:
        return (0, scale)
    value = int(digits)
    if suffix.lower() == "k":
        value *= 1000
    return (value, scale)


def available_targets(requested: list[str]) -> list[str]:
    targets: list[str] = []
    for target in requested:
        if target == "csharp-query" and shutil.which("dotnet") is None:
            print("skip csharp-query: dotnet not found", file=sys.stderr)
            continue
        if target == "rust-dfs" and shutil.which("rustc") is None:
            print("skip rust-dfs: rustc not found", file=sys.stderr)
            continue
        if target == "go-dfs" and shutil.which("go") is None:
            print("skip go-dfs: go not found", file=sys.stderr)
            continue
        targets.append(target)
    return targets


def generate_pipeline_source(root: Path, target: str) -> Path:
    ext = {"csharp_query": ".cs", "rust": ".rs", "go": ".go"}[target]
    filename = f"category_influence{ext}"
    output = root / filename
    run(
        [
            sys.executable,
            str(GENERATOR),
            "--facts",
            str(FACTS_PATH),
            "--workload",
            "category_influence",
            "--target",
            target,
            "--output",
            str(output),
        ]
    )
    return output


def build_csharp_query(root: Path) -> list[str]:
    project_dir = root / "csharp_query"
    run(
        [
            sys.executable,
            str(GENERATOR),
            "--facts",
            str(FACTS_PATH),
            "--workload",
            "category_influence",
            "--target",
            "csharp_query",
            "--output-dir",
            str(project_dir),
        ]
    )
    run(["dotnet", "build", "benchmark_qe.csproj", "-c", "Release"], cwd=project_dir)
    return [
        "dotnet",
        str(project_dir / "bin" / "Release" / "net9.0" / "benchmark_qe.dll"),
    ]


def build_rust_dfs(root: Path) -> list[str]:
    project_dir = root / "rust_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = generate_pipeline_source(project_dir, "rust")
    binary = project_dir / "category_influence_rust"
    run(["rustc", "-O", str(source), "-o", str(binary)])
    return [str(binary)]


def build_go_dfs(root: Path) -> list[str]:
    project_dir = root / "go_dfs"
    project_dir.mkdir(parents=True, exist_ok=True)
    source = generate_pipeline_source(project_dir, "go")
    binary = project_dir / "category_influence_go"
    go_cache = project_dir / ".gocache"
    go_cache.mkdir(exist_ok=True)
    env = dict(os.environ, GOCACHE=str(go_cache))
    run(["go", "build", "-o", str(binary), str(source)], env=env)
    return [str(binary)]


def normalize_output(output: str) -> str:
    lines = output.splitlines()
    if not lines:
        return ""
    header = lines[0]
    rows: list[tuple[str, float]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rows.append((parts[0], round(float(parts[1]), 9)))
    rows.sort(key=lambda item: (-item[1], item[0]))
    normalized = [header]
    for root, score in rows:
        normalized.append(f"{root}\t{score:.9f}")
    return "\n".join(normalized)


def benchmark_target(command: list[str], scale: str, repetitions: int, target: str) -> RunResult:
    scale_dir = require_file(BENCH_DIR / scale / "category_parent.tsv").parent
    edge_path = scale_dir / "category_parent.tsv"
    article_path = scale_dir / "article_category.tsv"

    times: list[float] = []
    stdout = ""
    stderr = ""
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run(command + [str(edge_path), str(article_path)])
        times.append(time.perf_counter() - started)
        stdout = result.stdout
        stderr = result.stderr

    normalized = normalize_output(stdout)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    row_count = max(0, len(normalized.splitlines()) - 1)
    return RunResult(target, scale, times, digest, row_count, stderr)


def print_summary(results: list[RunResult]) -> None:
    by_scale: dict[str, list[RunResult]] = {}
    for result in results:
        by_scale.setdefault(result.scale, []).append(result)

    print("scale\ttarget\tmedian_s\tmin_s\tmax_s\trows\tstdout_sha256")
    for scale in sorted(by_scale.keys(), key=scale_sort_key):
        entries = sorted(by_scale[scale], key=lambda item: item.target)
        for result in entries:
            print(
                f"{scale}\t{result.target}\t{result.median:.3f}\t"
                f"{min(result.times):.3f}\t{max(result.times):.3f}\t"
                f"{result.row_count}\t{result.stdout_sha256[:12]}"
            )

        if len(entries) > 1:
            hashes = {item.stdout_sha256 for item in entries}
            print(f"{scale}\toutputs\t{'match' if len(hashes) == 1 else 'MISMATCH'}")
            csharp = next((item for item in entries if item.target == "csharp-query"), None)
            rust = next((item for item in entries if item.target == "rust-dfs"), None)
            go = next((item for item in entries if item.target == "go-dfs"), None)
            if csharp and rust:
                print(f"{scale}\tspeedup_vs_rust_dfs\t{rust.median / csharp.median:.2f}x")
            if csharp and go:
                print(f"{scale}\tspeedup_vs_go_dfs\t{go.median / csharp.median:.2f}x")
            if rust and go:
                faster = "rust-dfs" if rust.median < go.median else "go-dfs"
                speedup = (go.median / rust.median) if faster == "rust-dfs" else (rust.median / go.median)
                print(f"{scale}\tfaster_target\t{faster}")
                print(f"{scale}\tspeedup\t{speedup:.2f}x")
            if csharp and csharp.stderr:
                phase_lines = [line.strip() for line in csharp.stderr.splitlines() if "=" in line]
                if phase_lines:
                    print(f"{scale}\tcsharp-query-metrics\t" + " ".join(phase_lines))


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    targets = available_targets([part.strip() for part in args.targets.split(",") if part.strip()])
    if not targets:
        print("no benchmark targets available", file=sys.stderr)
        return 1

    temp_ctx = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-category-influence-"))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-category-influence-")
        temp_root = Path(temp_ctx.name)

    try:
        commands: dict[str, list[str]] = {}
        for target in targets:
            if target == "csharp-query":
                commands[target] = build_csharp_query(temp_root)
            elif target == "rust-dfs":
                commands[target] = build_rust_dfs(temp_root)
            elif target == "go-dfs":
                commands[target] = build_go_dfs(temp_root)
            else:
                raise ValueError(f"unsupported target: {target}")

        results: list[RunResult] = []
        for scale in scales:
            for target in targets:
                results.append(benchmark_target(commands[target], scale, args.repetitions, target))

        print_summary(results)
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
