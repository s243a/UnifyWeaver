#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, TextIO


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "benchmark"
MYSQL_STREAM_MANIFEST = ROOT / "src" / "unifyweaver" / "runtime" / "rust" / "mysql_stream" / "Cargo.toml"
CSHARP_LMDB_INGEST_PROJECT = ROOT / "src" / "unifyweaver" / "runtime" / "csharp" / "lmdb_ingest" / "lmdb_ingest.csproj"
DEFAULT_DUMP = ROOT / "data" / "enwiki" / "enwiki-latest-categorylinks.sql.gz"
DEFAULT_LMDB_DBNAME = "main"


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lmdb_manifest(
    *,
    predicate_name: str,
    environment_path: str,
    database_name: str,
    row_count: int,
    source_path: Path,
) -> dict[str, object]:
    return {
        "Format": "unifyweaver.lmdb_relation.v1",
        "Version": 1,
        "Backend": "lmdb",
        "PredicateName": predicate_name,
        "Arity": 2,
        "EnvironmentPath": environment_path,
        "DatabaseName": database_name,
        "DupSort": True,
        "KeyEncoding": "utf8",
        "ValueEncoding": "utf8",
        "RowCount": row_count,
        "SourcePath": str(source_path),
        "SourceLength": source_path.stat().st_size,
        "SourceSha256": file_sha256(source_path),
    }


def lmdb_consumer_env(
    *,
    lmdb_path: Path,
    map_size: int,
    database_name: str = DEFAULT_LMDB_DBNAME,
) -> dict[str, str]:
    return {
        "UW_LMDB_PATH": str(lmdb_path),
        "UW_LMDB_MAP_SIZE": str(map_size),
        "UW_LMDB_DBNAME": database_name,
        "UW_LMDB_DUPSORT": "1",
        "UW_KEY_COL": "0",
        "UW_VAL_COL": "1",
        "UW_KEY_ENCODING": "utf8",
        "UW_VAL_ENCODING": "utf8",
        "UW_BATCH_SIZE": "50000",
    }


def lmdb_sink_input(edges: list[tuple[str, str]]) -> str:
    return "".join(f"{child}\t{parent}\n" for child, parent in edges)


def write_lmdb_artifact(
    *,
    fixture_dir: Path,
    edges: list[tuple[str, str]],
    refresh: bool,
    map_size: int,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet is not available; cannot sink C# LMDB artifact")
    if not CSHARP_LMDB_INGEST_PROJECT.exists():
        raise FileNotFoundError(CSHARP_LMDB_INGEST_PROJECT)

    lmdb_dir = fixture_dir / "category_parent.lmdb"
    manifest_path = fixture_dir / "category_parent.lmdb.manifest.json"
    if manifest_path.exists() and lmdb_dir.exists() and not refresh:
        print(f"[lmdb] reuse {manifest_path}", file=sys.stderr)
        return manifest_path

    if lmdb_dir.exists():
        if not refresh:
            raise RuntimeError(f"LMDB directory exists without manifest; pass --refresh-lmdb to rebuild: {lmdb_dir}")
        shutil.rmtree(lmdb_dir)
    if manifest_path.exists():
        if not refresh:
            raise RuntimeError(f"LMDB manifest exists without directory; pass --refresh-lmdb to rebuild: {manifest_path}")
        manifest_path.unlink()

    process_env = {**os.environ, **lmdb_consumer_env(lmdb_path=lmdb_dir, map_size=map_size)}
    result = run(
        ["dotnet", "run", "--project", str(CSHARP_LMDB_INGEST_PROJECT), "--configuration", "Release"],
        cwd=ROOT,
        text=True,
        input=lmdb_sink_input(edges),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=process_env,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "C# LMDB ingest failed with status "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    source_path = fixture_dir / "category_parent.tsv"
    manifest = lmdb_manifest(
        predicate_name="category_parent",
        environment_path=lmdb_dir.name,
        database_name=DEFAULT_LMDB_DBNAME,
        row_count=len(edges),
        source_path=source_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[lmdb] wrote {manifest_path}", file=sys.stderr)
    return manifest_path


def prepare_from_stream(
    *,
    scale: str,
    max_edges: int,
    output_root: Path,
    stream: TextIO,
    sink_lmdb: bool = False,
    refresh_lmdb: bool = False,
    lmdb_map_size: int = 1 << 30,
) -> Path:
    edges, scanned_rows = edge_rows_from_mysql_stream(stream, max_edges)
    if not edges:
        raise RuntimeError("parser stream produced no subcat edges")
    output_dir = write_fixture(scale, edges, scanned_rows, output_root)
    if sink_lmdb:
        write_lmdb_artifact(
            fixture_dir=output_dir,
            edges=edges,
            refresh=refresh_lmdb,
            map_size=lmdb_map_size,
        )
    return output_dir


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
    parser.add_argument(
        "--sink-lmdb",
        action="store_true",
        help="also sink the capped fixture rows into a reusable C# LMDB relation artifact under the scale directory",
    )
    parser.add_argument(
        "--refresh-lmdb",
        action="store_true",
        help="rebuild the LMDB artifact even when category_parent.lmdb.manifest.json already exists",
    )
    parser.add_argument(
        "--lmdb-map-size",
        type=parse_positive_int,
        default=1 << 30,
        help="LMDB map size in bytes for --sink-lmdb",
    )
    args = parser.parse_args(argv)

    if args.from_stdin:
        output_dir = prepare_from_stream(
            scale=args.scale,
            max_edges=args.max_edges,
            output_root=args.output_root,
            stream=sys.stdin,
            sink_lmdb=args.sink_lmdb,
            refresh_lmdb=args.refresh_lmdb,
            lmdb_map_size=args.lmdb_map_size,
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
                    sink_lmdb=args.sink_lmdb,
                    refresh_lmdb=args.refresh_lmdb,
                    lmdb_map_size=args.lmdb_map_size,
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
