#!/usr/bin/env python3
"""Prepare larger effective-distance benchmark fixtures and reusable artifacts.

This wrapper mirrors the Haskell scaling notes:

- 50k_cats: full SimpleWiki category hierarchy, first 50k categories as seeds.
- 100k_cats: full SimpleWiki category hierarchy, all content categories as seeds
  (historically about 84k categories / 197k edges after filtering).

It intentionally separates fixture/artifact preparation from timed benchmark
runs. Use benchmark_effective_distance.py with --build-root and --prepare-only
to compile runners and ingest reusable Elixir LMDB artifacts once.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
CATEGORY_ONLY_GENERATOR = ROOT / "examples" / "benchmark" / "generate_category_only_benchmark.py"
BENCHMARK = ROOT / "examples" / "benchmark" / "benchmark_effective_distance.py"
DEFAULT_DB_CANDIDATES = [
    ROOT / "data" / "simplewiki" / "simplewiki_categories.db",
    ROOT / "context" / "gemini" / "UnifyWeaver" / "data" / "simplewiki" / "simplewiki_categories.db",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default="50k_cats,100k_cats",
        help="Comma-separated large fixtures to prepare: 50k_cats,100k_cats",
    )
    parser.add_argument("--db", type=Path, default=None, help="Path to simplewiki_categories.db")
    parser.add_argument(
        "--build-root",
        type=Path,
        default=ROOT / "output" / "effective-distance-large",
        help="Persistent benchmark build/artifact root",
    )
    parser.add_argument(
        "--targets",
        default="wam-elixir-int-tuple,wam-elixir-lmdb-int-ids",
        help="Targets to prepare with benchmark_effective_distance.py --prepare-only",
    )
    parser.add_argument(
        "--skip-fixtures",
        action="store_true",
        help="Only prepare target artifacts; assume fixture directories already exist.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def resolve_db(path: Path | None) -> Path:
    if path is not None:
        if path.exists():
            return path
        raise FileNotFoundError(path)
    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in DEFAULT_DB_CANDIDATES)
    raise FileNotFoundError(
        "simplewiki_categories.db not found. Run examples/benchmark/parse_simplewiki_dump.py "
        "or pass --db.\nSearched:\n" + searched
    )


def scale_seed_cap(scale: str) -> int | None:
    if scale == "50k_cats":
        return 50_000
    if scale == "100k_cats":
        return None
    raise ValueError(f"unsupported scale: {scale}")


def prepare_fixture(scale: str, db_path: Path) -> None:
    output_dir = BENCH_DIR / scale
    if (output_dir / "category_parent.tsv").exists() and (output_dir / "facts.pl").exists():
        print(f"[fixture] reuse {output_dir}", file=sys.stderr)
        return
    cmd = [
        sys.executable,
        str(CATEGORY_ONLY_GENERATOR),
        "--output",
        str(output_dir),
        "--db",
        str(db_path),
    ]
    cap = scale_seed_cap(scale)
    if cap is not None:
        cmd.extend(["--max-seeds", str(cap)])
    print(f"[fixture] generate {scale}", file=sys.stderr)
    run(cmd)


def prepare_artifacts(scales: list[str], targets: str, build_root: Path) -> None:
    run(
        [
            sys.executable,
            str(BENCHMARK),
            "--scales",
            ",".join(scales),
            "--targets",
            targets,
            "--repetitions",
            "1",
            "--build-root",
            str(build_root),
            "--prepare-only",
        ]
    )


def main() -> int:
    args = parse_args()
    scales = [part.strip() for part in args.scales.split(",") if part.strip()]
    if not scales:
        raise SystemExit("expected at least one scale")
    if not args.skip_fixtures:
        db_path = resolve_db(args.db)
        for scale in scales:
            prepare_fixture(scale, db_path)
    prepare_artifacts(scales, args.targets, args.build_root)
    print(f"prepared large-scale artifacts under {args.build_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
