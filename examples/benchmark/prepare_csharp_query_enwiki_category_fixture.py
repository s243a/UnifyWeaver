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


RELATION_SPECS = {
    "category_parent": {
        "cl_type": "subcat",
        "tsv": "category_parent.tsv",
        "lmdb": "category_parent.lmdb",
        "manifest": "category_parent.lmdb.manifest.json",
        "stats": "category_parent.lmdb.stats.json",
        "header": "child\tparent\n",
        "shape": "category_parent/2 backend fixture; child=cl_from page id, parent=cl_target_id",
        "count_key": "n_hierarchy_edges",
    },
    "article_category": {
        "cl_type": "page",
        "tsv": "article_category.tsv",
        "lmdb": "article_category.lmdb",
        "manifest": "article_category.lmdb.manifest.json",
        "stats": "article_category.lmdb.stats.json",
        "header": "article\tcategory\n",
        "shape": "article_category/2 backend fixture; article=cl_from page id, category=cl_target_id",
        "count_key": "n_article_category_edges",
    },
}


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


def rust_lmdb_sink_command(
    *,
    dump_path: Path,
    fixture_dir: Path,
    relation: str,
    max_edges: int,
    map_size: int,
    refresh: bool,
) -> list[str]:
    spec = RELATION_SPECS[relation]
    command = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(MYSQL_STREAM_MANIFEST),
        "--bin",
        "mysql_stream_lmdb",
        "--",
        str(dump_path),
        str(fixture_dir / spec["lmdb"]),
        "--manifest",
        str(fixture_dir / spec["manifest"]),
        "--predicate-name",
        relation,
        "--cl-type",
        str(spec["cl_type"]),
        "--max-edges",
        str(max_edges),
        "--map-size",
        str(map_size),
        "--fixture-tsv",
        str(fixture_dir / spec["tsv"]),
        "--fixture-header",
        str(spec["header"]).rstrip("\n").replace("\t", "\\t"),
        "--stats",
        str(fixture_dir / spec["stats"]),
    ]
    if refresh:
        command.append("--refresh")
    return command


def edge_rows_from_mysql_stream(
    lines: Iterable[str],
    *,
    relation: str,
    max_edges: int,
) -> tuple[list[tuple[str, str]], int]:
    spec = RELATION_SPECS[relation]
    rows: list[tuple[str, str]] = []
    scanned = 0
    for line in lines:
        scanned += 1
        fields = line.rstrip("\n").split("\t")
        if len(fields) <= 6 or fields[4] != spec["cl_type"]:
            continue
        rows.append((fields[0], fields[6]))
        if len(rows) >= max_edges:
            break
    return rows, scanned


def write_fixture(
    scale: str,
    *,
    relation: str,
    edges: list[tuple[str, str]],
    scanned_rows: int,
    output_root: Path,
) -> Path:
    spec = RELATION_SPECS[relation]
    output_dir = output_root / scale
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / spec["tsv"]).open("w", encoding="utf-8", newline="") as handle:
        handle.write(str(spec["header"]))
        for child, parent in edges:
            handle.write(f"{child}\t{parent}\n")
    write_metadata(output_dir, scale=scale, relation=relation, edge_count=len(edges), scanned_rows=scanned_rows)
    return output_dir


def write_metadata(output_dir: Path, *, scale: str, relation: str, edge_count: int, scanned_rows: int) -> None:
    spec = RELATION_SPECS[relation]
    metadata = {
        "source": "English Wikipedia categorylinks dump via rust mysql_stream parser",
        "shape": spec["shape"],
        "scale": scale,
        "relation": relation,
        spec["count_key"]: edge_count,
        "mysql_rows_scanned": scanned_rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def read_rust_lmdb_stats(fixture_dir: Path, *, relation: str) -> tuple[int, int]:
    stats = json.loads((fixture_dir / RELATION_SPECS[relation]["stats"]).read_text(encoding="utf-8"))
    return int(stats["edges_written"]), int(stats["rows_scanned"])


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
    relation: str,
    edges: list[tuple[str, str]],
    refresh: bool,
    map_size: int,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    if shutil.which("dotnet") is None:
        raise RuntimeError("dotnet is not available; cannot sink C# LMDB artifact")
    if not CSHARP_LMDB_INGEST_PROJECT.exists():
        raise FileNotFoundError(CSHARP_LMDB_INGEST_PROJECT)

    spec = RELATION_SPECS[relation]
    lmdb_dir = fixture_dir / spec["lmdb"]
    manifest_path = fixture_dir / spec["manifest"]
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

    source_path = fixture_dir / spec["tsv"]
    manifest = lmdb_manifest(
        predicate_name=relation,
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
    relation: str,
    max_edges: int,
    output_root: Path,
    stream: TextIO,
    sink_lmdb: bool = False,
    refresh_lmdb: bool = False,
    lmdb_map_size: int = 1 << 30,
) -> Path:
    edges, scanned_rows = edge_rows_from_mysql_stream(stream, relation=relation, max_edges=max_edges)
    if not edges:
        raise RuntimeError(f"parser stream produced no {RELATION_SPECS[relation]['cl_type']} edges")
    output_dir = write_fixture(scale, relation=relation, edges=edges, scanned_rows=scanned_rows, output_root=output_root)
    if sink_lmdb:
        write_lmdb_artifact(
            fixture_dir=output_dir,
            relation=relation,
            edges=edges,
            refresh=refresh_lmdb,
            map_size=lmdb_map_size,
        )
    return output_dir


def prepare_with_rust_lmdb_sink(
    *,
    scale: str,
    relation: str,
    max_edges: int,
    output_root: Path,
    dump_path: Path,
    refresh_lmdb: bool,
    lmdb_map_size: int,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    output_dir = output_root / scale
    output_dir.mkdir(parents=True, exist_ok=True)
    command = rust_lmdb_sink_command(
        dump_path=dump_path,
        fixture_dir=output_dir,
        relation=relation,
        max_edges=max_edges,
        map_size=lmdb_map_size,
        refresh=refresh_lmdb,
    )
    result = run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Rust mysql_stream_lmdb sink failed with status "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    edge_count, scanned_rows = read_rust_lmdb_stats(output_dir, relation=relation)
    if edge_count <= 0:
        raise RuntimeError(f"Rust mysql_stream_lmdb sink produced no {RELATION_SPECS[relation]['cl_type']} edges")
    write_metadata(output_dir, scale=scale, relation=relation, edge_count=edge_count, scanned_rows=scanned_rows)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare capped enwiki categorylinks-derived fixtures for the C# query "
            "effective-distance artifact-backend benchmark using the existing Rust mysql_stream parser."
        )
    )
    parser.add_argument("--scale", required=True, help="fixture label, e.g. 500k_cats or 1m_cats")
    parser.add_argument(
        "--relation",
        choices=tuple(RELATION_SPECS),
        default="category_parent",
        help="categorylinks-derived relation to prepare",
    )
    parser.add_argument("--max-edges", type=parse_positive_int, required=True, help="maximum relation rows to write")
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
        "--lmdb-sink-target",
        choices=("csharp", "rust"),
        default="csharp",
        help="implementation used by --sink-lmdb; rust writes TSV and LMDB in one parser pass when reading from --dump",
    )
    parser.add_argument(
        "--refresh-lmdb",
        action="store_true",
        help="rebuild the LMDB artifact even when the relation manifest already exists",
    )
    parser.add_argument(
        "--lmdb-map-size",
        type=parse_positive_int,
        default=1 << 30,
        help="LMDB map size in bytes for --sink-lmdb",
    )
    args = parser.parse_args(argv)

    if args.from_stdin and args.sink_lmdb and args.lmdb_sink_target == "rust":
        parser.error("--lmdb-sink-target rust requires --dump input, not --from-stdin")

    if args.from_stdin:
        output_dir = prepare_from_stream(
            scale=args.scale,
            relation=args.relation,
            max_edges=args.max_edges,
            output_root=args.output_root,
            stream=sys.stdin,
            sink_lmdb=args.sink_lmdb,
            refresh_lmdb=args.refresh_lmdb,
            lmdb_map_size=args.lmdb_map_size,
        )
    elif args.sink_lmdb and args.lmdb_sink_target == "rust":
        if not args.dump.exists():
            raise FileNotFoundError(args.dump)
        output_dir = prepare_with_rust_lmdb_sink(
            scale=args.scale,
            relation=args.relation,
            max_edges=args.max_edges,
            output_root=args.output_root,
            dump_path=args.dump,
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
                    relation=args.relation,
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
