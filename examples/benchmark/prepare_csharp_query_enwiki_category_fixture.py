#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, TextIO


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
MYSQL_STREAM_MANIFEST = ROOT / "src" / "unifyweaver" / "runtime" / "rust" / "mysql_stream" / "Cargo.toml"
DEFAULT_DUMP = ROOT / "data" / "enwiki" / "enwiki-latest-categorylinks.sql.gz"


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parser_command(dump_path: Path) -> list[str]:
    return [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(MYSQL_STREAM_MANIFEST),
        "--",
        str(dump_path),
    ]


def edge_rows_from_mysql_stream(lines: Iterable[str], max_edges: int) -> tuple[list[tuple[str, str]], int]:
    rows: list[tuple[str, str]] = []
    scanned = 0
    for line in lines:
        scanned += 1
        fields = line.rstrip("\n").split("\t")
        if len(fields) <= 6 or fields[4] != "subcat":
            continue
        rows.append((fields[0], fields[6]))
        if len(rows) >= max_edges:
            break
    return rows, scanned


def write_fixture(scale: str, edges: list[tuple[str, str]], scanned_rows: int, output_root: Path) -> Path:
    output_dir = output_root / scale
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "category_parent.tsv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("child\tparent\n")
        for child, parent in edges:
            handle.write(f"{child}\t{parent}\n")
    metadata = {
        "source": "English Wikipedia categorylinks dump via rust mysql_stream parser",
        "shape": "category_parent/2 backend fixture; child=cl_from page id, parent=cl_target_id",
        "scale": scale,
        "n_hierarchy_edges": len(edges),
        "mysql_rows_scanned": scanned_rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_dir


def prepare_from_stream(
    *,
    scale: str,
    max_edges: int,
    output_root: Path,
    stream: TextIO,
) -> Path:
    edges, scanned_rows = edge_rows_from_mysql_stream(stream, max_edges)
    if not edges:
        raise RuntimeError("parser stream produced no subcat edges")
    return write_fixture(scale, edges, scanned_rows, output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare capped enwiki category_parent/2 fixtures for the C# query "
            "effective-distance artifact-backend benchmark using the existing Rust mysql_stream parser."
        )
    )
    parser.add_argument("--scale", required=True, help="fixture label, e.g. 500k_cats or 1m_cats")
    parser.add_argument("--max-edges", type=parse_positive_int, required=True, help="maximum subcat edges to write")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP, help="path to enwiki-latest-categorylinks.sql.gz")
    parser.add_argument("--output-root", type=Path, default=BENCH_DIR, help="directory containing benchmark scales")
    parser.add_argument(
        "--from-stdin",
        action="store_true",
        help="read mysql_stream TSV from stdin instead of invoking cargo; useful for tests and pipelines",
    )
    args = parser.parse_args(argv)

    if args.from_stdin:
        output_dir = prepare_from_stream(
            scale=args.scale,
            max_edges=args.max_edges,
            output_root=args.output_root,
            stream=sys.stdin,
        )
    else:
        if not args.dump.exists():
            raise FileNotFoundError(args.dump)
        with subprocess.Popen(
            parser_command(args.dump),
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
        ) as proc:
            assert proc.stdout is not None
            try:
                output_dir = prepare_from_stream(
                    scale=args.scale,
                    max_edges=args.max_edges,
                    output_root=args.output_root,
                    stream=proc.stdout,
                )
            finally:
                proc.stdout.close()
            return_code = proc.wait()
            if return_code not in (0, -13):
                raise RuntimeError(f"mysql_stream parser exited with status {return_code}")

    print(f"wrote {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
