#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
DEFAULT_DUMP = ROOT / "data" / "enwiki" / "enwiki-latest-categorylinks.sql.gz"
PREPARE_SCRIPT = ROOT / "examples" / "benchmark" / "prepare_csharp_query_enwiki_category_fixture.py"
BACKEND_BENCHMARK = ROOT / "examples" / "benchmark" / "benchmark_csharp_query_effective_distance_artifact_backends.py"

HEADERS = [
    "scale",
    "max_edges",
    "prepare_s",
    "rows",
    "distinct_categories",
    "lookup_keys",
    "artifact_bytes",
    "open_ms",
    "lookup_ms",
    "bucket_ms",
    "scan_ms",
]


@dataclass(frozen=True)
class Measurement:
    values: dict[str, str]


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def run_checked(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def prepare_command(
    *,
    scale: str,
    max_edges: int,
    dump_path: Path,
    output_root: Path,
    map_size: int,
    refresh_lmdb: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--scale",
        scale,
        "--max-edges",
        str(max_edges),
        "--dump",
        str(dump_path),
        "--output-root",
        str(output_root),
        "--sink-lmdb",
        "--lmdb-sink-target",
        "rust",
        "--lmdb-map-size",
        str(map_size),
    ]
    if refresh_lmdb:
        command.append("--refresh-lmdb")
    return command


def benchmark_command(
    *,
    scale: str,
    output_root: Path,
    lookup_keys: int,
    lookup_repetitions: int,
) -> list[str]:
    return [
        sys.executable,
        str(BACKEND_BENCHMARK),
        "--scales",
        scale,
        "--benchmark-root",
        str(output_root),
        "--lookup-keys",
        str(lookup_keys),
        "--lookup-repetitions",
        str(lookup_repetitions),
        "--repetitions",
        "1",
        "--use-scale-lmdb-artifact",
        "--lmdb-only",
        "--format",
        "tsv",
    ]


def lmdb_row_from_tsv(output: str) -> dict[str, str]:
    rows = list(csv.DictReader(output.splitlines(), delimiter="\t"))
    if len(rows) != 1 or rows[0].get("mode") != "lmdb":
        raise RuntimeError(f"expected one lmdb row, got: {output}")
    return rows[0]


def measure(
    *,
    scale: str,
    max_edges: int,
    dump_path: Path,
    output_root: Path,
    map_size: int,
    refresh_lmdb: bool,
    lookup_keys: int,
    lookup_repetitions: int,
) -> Measurement:
    started = time.perf_counter()
    run_checked(
        prepare_command(
            scale=scale,
            max_edges=max_edges,
            dump_path=dump_path,
            output_root=output_root,
            map_size=map_size,
            refresh_lmdb=refresh_lmdb,
        ),
        timeout=3600,
    )
    prepare_s = time.perf_counter() - started

    benchmark = run_checked(
        benchmark_command(
            scale=scale,
            output_root=output_root,
            lookup_keys=lookup_keys,
            lookup_repetitions=lookup_repetitions,
        ),
        timeout=600,
    )
    row = lmdb_row_from_tsv(benchmark.stdout)
    return Measurement(
        {
            "scale": scale,
            "max_edges": str(max_edges),
            "prepare_s": f"{prepare_s:.3f}",
            "rows": row["rows"],
            "distinct_categories": row["distinct_categories"],
            "lookup_keys": row["lookup_keys"],
            "artifact_bytes": row["artifact_bytes"],
            "open_ms": row["open_ms"],
            "lookup_ms": row["lookup_ms"],
            "bucket_ms": row["bucket_ms"],
            "scan_ms": row["scan_ms"],
        }
    )


def render_tsv(rows: list[Measurement]) -> str:
    lines = ["\t".join(HEADERS)]
    lines.extend("\t".join(row.values[column] for column in HEADERS) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Rust mysql_stream_lmdb fixture ingestion plus C# query LMDB-only access."
    )
    parser.add_argument("--scale", required=True, help="fixture label to create under --output-root")
    parser.add_argument("--max-edges", type=parse_positive_int, required=True, help="subcat edge cap")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP, help="categorylinks SQL dump")
    parser.add_argument("--output-root", type=Path, default=BENCH_DIR, help="benchmark scale output root")
    parser.add_argument("--lmdb-map-size", type=parse_positive_int, default=1 << 30, help="LMDB map size in bytes")
    parser.add_argument("--lookup-keys", type=parse_positive_int, default=64, help="lookup key count")
    parser.add_argument("--lookup-repetitions", type=parse_positive_int, default=1, help="lookup repetitions")
    parser.add_argument("--refresh-lmdb", action="store_true", help="rebuild LMDB artifact before measuring")
    args = parser.parse_args(argv)

    if not args.dump.exists():
        raise FileNotFoundError(args.dump)

    measurement = measure(
        scale=args.scale,
        max_edges=args.max_edges,
        dump_path=args.dump,
        output_root=args.output_root,
        map_size=args.lmdb_map_size,
        refresh_lmdb=args.refresh_lmdb,
        lookup_keys=args.lookup_keys,
        lookup_repetitions=args.lookup_repetitions,
    )
    print(render_tsv([measurement]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
