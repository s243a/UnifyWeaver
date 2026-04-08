#!/usr/bin/env python3
"""
Compare Rust WAM seed-level profiles against direct Prolog evaluation for the
effective-distance benchmark.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
WAM_GENERATOR = ROOT / "examples" / "benchmark" / "generate_wam_effective_distance_benchmark.pl"
WORKLOAD = ROOT / "examples" / "benchmark" / "effective_distance.pl"

PROFILE_RE = re.compile(
    r"seed_progress category=(?P<category>\S+) elapsed_ms=(?P<elapsed_ms>\d+) "
    r"steps=(?P<steps>\d+) backtracks=(?P<backtracks>\d+) "
    r"solutions=(?P<solutions>\d+) weight_sum=(?P<weight_sum>[-+]?\d+(?:\.\d+)?)"
)

PROLOG_RE = re.compile(
    r"seed_profile category=(?P<category>\S+) root=(?P<root>\S+) "
    r"elapsed_ms=(?P<elapsed_ms>\d+) solutions=(?P<solutions>\d+) "
    r"weight_sum=(?P<weight_sum>[-+]?\d+(?:\.\d+)?)"
)


@dataclass
class WamSeedProfile:
    category: str
    elapsed_ms: int
    steps: int
    backtracks: int
    solutions: int
    weight_sum: float


@dataclass
class PrologSeedProfile:
    category: str
    root: str
    elapsed_ms: int
    solutions: int
    weight_sum: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default="300", help="Benchmark scale under data/benchmark/")
    parser.add_argument(
        "--seeds",
        required=True,
        help="Pipe-separated seed categories to profile",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=1_000_000,
        help="Rust WAM per-seed step cap",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-run timeout in seconds",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the generated WAM project directory",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def prolog_atom_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def build_wam_benchmark(scale: str, root: Path) -> Path:
    facts_path = BENCH_DIR / scale / "facts.pl"
    project_dir = root / "wam_rust" / scale
    project_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        ["swipl", "-q", "-s", str(WAM_GENERATOR), "--", str(facts_path), str(project_dir)],
        cwd=ROOT,
    )
    run_command(["cargo", "build", "--release"], cwd=project_dir)
    return project_dir / "target" / "release" / "hybrid_ed_bench"


def run_wam_seed(binary: Path, scale: str, seed: str, step_limit: int, timeout_s: int) -> WamSeedProfile:
    env = os.environ.copy()
    env["WAM_PROFILE"] = "1"
    env["WAM_SEED_FILTER"] = seed
    env["WAM_STEP_LIMIT"] = str(step_limit)
    result = run_command(
        ["timeout", str(timeout_s), str(binary), str(BENCH_DIR / scale)],
        cwd=ROOT,
        env=env,
    )
    for line in result.stderr.splitlines():
        match = PROFILE_RE.search(line.strip())
        if match:
            return WamSeedProfile(
                category=match.group("category"),
                elapsed_ms=int(match.group("elapsed_ms")),
                steps=int(match.group("steps")),
                backtracks=int(match.group("backtracks")),
                solutions=int(match.group("solutions")),
                weight_sum=float(match.group("weight_sum")),
            )
    raise RuntimeError(f"missing Rust WAM seed_progress line for {seed}\nstderr:\n{result.stderr}")


def run_prolog_seed(scale: str, seed: str, timeout_s: int) -> PrologSeedProfile:
    facts_path = BENCH_DIR / scale / "facts.pl"
    goal = (
        "statistics(walltime, [StartMs,_]), "
        "once(root_category(Root)), "
        f"Cat = {prolog_atom_literal(seed)}, "
        "aggregate_all(count, category_ancestor(Cat, Root, _, [Cat]), Count), "
        "dimension_n(N), NegN is -N, "
        "aggregate_all(sum(W), "
        "(category_ancestor(Cat, Root, Hops, [Cat]), D is Hops + 1, W is D ** NegN), "
        "WeightSum0), "
        "(var(WeightSum0) -> WeightSum = 0.0 ; WeightSum = WeightSum0), "
        "statistics(walltime, [EndMs,_]), "
        "ElapsedMs is EndMs - StartMs, "
        "format('seed_profile category=~w root=~w elapsed_ms=~w solutions=~w weight_sum=~6f~n', "
        "[Cat, Root, ElapsedMs, Count, WeightSum]), "
        "halt"
    )
    result = run_command(
        ["timeout", str(timeout_s), "swipl", "-q", "-l", str(WORKLOAD), "-l", str(facts_path), "-g", goal],
        cwd=ROOT,
    )
    for line in result.stdout.splitlines():
        match = PROLOG_RE.search(line.strip())
        if match:
            return PrologSeedProfile(
                category=match.group("category"),
                root=match.group("root"),
                elapsed_ms=int(match.group("elapsed_ms")),
                solutions=int(match.group("solutions")),
                weight_sum=float(match.group("weight_sum")),
            )
    raise RuntimeError(f"missing Prolog seed_profile line for {seed}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def print_row(seed: str, prolog: PrologSeedProfile, wam: WamSeedProfile) -> None:
    print(
        "\t".join(
            [
                seed,
                prolog.root,
                str(prolog.solutions),
                f"{prolog.weight_sum:.6f}",
                str(prolog.elapsed_ms),
                str(wam.solutions),
                f"{wam.weight_sum:.6f}",
                str(wam.elapsed_ms),
                str(wam.steps),
                str(wam.backtracks),
                "match" if prolog.solutions == wam.solutions and abs(prolog.weight_sum - wam.weight_sum) < 1e-9 else "mismatch",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    seeds = [seed.strip() for seed in args.seeds.split("|") if seed.strip()]
    if not seeds:
        raise SystemExit("no seeds supplied")

    temp_ctx = None
    scratch_root = ROOT / "output"
    scratch_root.mkdir(parents=True, exist_ok=True)
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="uw-wam-seed-", dir=scratch_root))
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="uw-wam-seed-", dir=scratch_root)
        temp_root = Path(temp_ctx.name)

    try:
        binary = build_wam_benchmark(args.scale, temp_root)
        print(
            "seed\troot\tprolog_solutions\tprolog_weight_sum\tprolog_elapsed_ms\t"
            "wam_solutions\twam_weight_sum\twam_elapsed_ms\twam_steps\twam_backtracks\tstatus"
        )
        for seed in seeds:
            prolog = run_prolog_seed(args.scale, seed, args.timeout)
            wam = run_wam_seed(binary, args.scale, seed, args.step_limit, args.timeout)
            print_row(seed, prolog, wam)
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
