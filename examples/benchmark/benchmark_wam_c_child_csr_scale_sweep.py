#!/usr/bin/env python3
"""
Run the WAM-C child-search CSR layout scale sweep.

This wrapper keeps the routine sweep compile-only: it builds the generated C
projects and reports artifact byte sizes without executing the expensive
child-search query body. Use the underlying matrix directly for full runtime
rows when a longer benchmark window is intentional.

Use --artifact-only for large category scales where generating and compiling the
full WAM-C query runner is not needed. That path builds only the reverse CSR
artifacts from the benchmark TSV inputs and reports their byte sizes.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "examples" / "benchmark" / "benchmark_effective_distance_matrix.py"
BENCH_DIR = ROOT / "data" / "benchmark"
CSR_TARGETS = [
    "c-wam-accumulated-child-csr",
    "c-wam-accumulated-child-csr-drop",
    "c-wam-accumulated-child-csr-lmdb-offset",
]
DEFAULT_SCALES = "10x,1k,5k,10k"
ARTIFACT_ONLY_TARGETS = [
    ("c-wam-accumulated-child-csr", "sorted_array"),
    ("c-wam-accumulated-child-csr-drop", "sorted_array"),
    ("c-wam-accumulated-child-csr-lmdb-offset", "lmdb_offset"),
]
IDX_RECORD = struct.Struct("<iQI")
OFFSET_RECORD = struct.Struct("<QI")
I32 = struct.Struct("<i")


@dataclass(frozen=True)
class ArtifactOnlyRow:
    scale: str
    target: str
    status: str
    build_s: float
    category_parent_tsv_bytes: int
    reverse_csr_index_bytes: int
    reverse_csr_values_bytes: int
    reverse_csr_offsets_lmdb_bytes: int
    parent_count: int
    edge_count: int
    category_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default=DEFAULT_SCALES,
        help=(
            f"Comma-separated benchmark scales. Default: {DEFAULT_SCALES}. "
            "Use --scales 50k_cats,100k_cats explicitly for the largest local artifacts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix command or artifact-only plan without running it.",
    )
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="Build only reverse CSR artifacts from benchmark TSVs; skip WAM-C generation and C compilation.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Directory for artifact-only outputs. Defaults to a temporary directory that is removed after the run.",
    )
    parser.add_argument(
        "matrix_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed after -- to the matrix runner.",
    )
    return parser.parse_args()


def matrix_command(scales: str, extra_args: list[str] | None = None) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX),
        "--scales",
        scales,
        "--target-sets",
        "c-wam-child-csr-layouts",
        "--compile-only-targets",
        ",".join(CSR_TARGETS),
        "--baseline-target",
        "c-wam-accumulated-child-csr",
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def scale_names(scales: str) -> list[str]:
    return [part.strip() for part in scales.split(",") if part.strip()]


def read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    return rows


def read_tsv_column(path: Path) -> list[str]:
    values: list[str] = []
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            value = line.rstrip("\n")
            if value:
                values.append(value.split("\t", 1)[0])
    return values


def category_id_map(
    category_parents: list[tuple[str, str]],
    article_categories: list[tuple[str, str]],
    root_categories: list[str],
) -> dict[str, int]:
    categories: set[str] = set(root_categories)
    for child, parent in category_parents:
        categories.add(child)
        categories.add(parent)
    for _article, category in article_categories:
        categories.add(category)
    return {category: index for index, category in enumerate(sorted(categories), start=1)}


def generated_category_parent_tsv_bytes(category_parents: list[tuple[str, str]]) -> int:
    return sum(len(f"{child}\t{parent}\n".encode("utf-8")) for child, parent in category_parents)


def file_tree_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def write_reverse_csr_artifact(
    category_parents: list[tuple[str, str]],
    ids: dict[str, int],
    out_dir: Path,
    index_backend: str,
) -> tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    edge_count = 0
    for child, parent in category_parents:
        children_by_parent[ids[parent]].append(ids[child])
        edge_count += 1

    idx_path = out_dir / "category_child.csr.idx"
    val_path = out_dir / "category_child.csr.val"
    offset_edges = 0
    with idx_path.open("wb") as idx_file, val_path.open("wb") as val_file:
        for parent_id in sorted(children_by_parent):
            children = sorted(children_by_parent[parent_id])
            idx_file.write(IDX_RECORD.pack(parent_id, offset_edges, len(children)))
            for child_id in children:
                val_file.write(I32.pack(child_id))
            offset_edges += len(children)

    if index_backend == "lmdb_offset":
        write_lmdb_offset_index(idx_path, out_dir / "category_child.csr.offsets.lmdb")
    elif index_backend != "sorted_array":
        raise ValueError(f"unknown index backend: {index_backend}")
    return len(children_by_parent), edge_count, len(ids)


def write_lmdb_offset_index(idx_path: Path, offset_lmdb_path: Path) -> None:
    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("artifact-only LMDB offset rows require the 'lmdb' Python package") from exc

    row_count = idx_path.stat().st_size // IDX_RECORD.size
    env = lmdb.open(
        str(offset_lmdb_path),
        map_size=max(row_count * 128, 1 << 20),
        max_dbs=2,
        subdir=True,
    )
    try:
        offsets_db = env.open_db(b"offsets")
        with env.begin(write=True) as txn:
            data = idx_path.read_bytes()
            for offset in range(0, len(data), IDX_RECORD.size):
                parent_id, offset_edges, count = IDX_RECORD.unpack_from(data, offset)
                txn.put(I32.pack(parent_id), OFFSET_RECORD.pack(offset_edges, count), db=offsets_db)
    finally:
        env.sync()
        env.close()


def build_artifact_only_rows(scales: list[str], artifact_root: Path) -> list[ArtifactOnlyRow]:
    rows: list[ArtifactOnlyRow] = []
    for scale in scales:
        scale_dir = BENCH_DIR / scale
        category_parent_path = scale_dir / "category_parent.tsv"
        if not category_parent_path.exists():
            raise FileNotFoundError(category_parent_path)

        category_parents = read_tsv_pairs(category_parent_path)
        article_categories = read_tsv_pairs(scale_dir / "article_category.tsv")
        root_categories = read_tsv_column(scale_dir / "root_categories.tsv")
        ids = category_id_map(category_parents, article_categories, root_categories)

        for target, index_backend in ARTIFACT_ONLY_TARGETS:
            started_at = time.perf_counter()
            out_dir = artifact_root / scale / target
            if out_dir.exists():
                shutil.rmtree(out_dir)
            parent_count, edge_count, category_count = write_reverse_csr_artifact(
                category_parents,
                ids,
                out_dir,
                index_backend,
            )
            build_s = time.perf_counter() - started_at
            rows.append(
                ArtifactOnlyRow(
                    scale=scale,
                    target=target,
                    status="artifact_only",
                    build_s=build_s,
                    category_parent_tsv_bytes=generated_category_parent_tsv_bytes(category_parents),
                    reverse_csr_index_bytes=file_tree_size_bytes(out_dir / "category_child.csr.idx"),
                    reverse_csr_values_bytes=file_tree_size_bytes(out_dir / "category_child.csr.val"),
                    reverse_csr_offsets_lmdb_bytes=file_tree_size_bytes(out_dir / "category_child.csr.offsets.lmdb"),
                    parent_count=parent_count,
                    edge_count=edge_count,
                    category_count=category_count,
                )
            )
    return rows


def print_artifact_only_rows(rows: list[ArtifactOnlyRow]) -> None:
    print(
        "\t".join(
            [
                "scale",
                "target",
                "status",
                "build_s",
                "category_parent_tsv_bytes",
                "reverse_csr_index_bytes",
                "reverse_csr_values_bytes",
                "reverse_csr_offsets_lmdb_bytes",
                "parent_count",
                "edge_count",
                "category_count",
            ]
        )
    )
    for row in rows:
        print(
            "\t".join(
                [
                    row.scale,
                    row.target,
                    row.status,
                    f"{row.build_s:.3f}",
                    str(row.category_parent_tsv_bytes),
                    str(row.reverse_csr_index_bytes),
                    str(row.reverse_csr_values_bytes),
                    str(row.reverse_csr_offsets_lmdb_bytes),
                    str(row.parent_count),
                    str(row.edge_count),
                    str(row.category_count),
                ]
            )
        )


def run_artifact_only(scales: str, artifact_root: Path | None, dry_run: bool) -> int:
    names = scale_names(scales)
    if dry_run:
        root = artifact_root or Path("<temporary artifact root>")
        for scale in names:
            for target, index_backend in ARTIFACT_ONLY_TARGETS:
                print(f"artifact-only {scale} {target} index_backend={index_backend} root={root}")
        return 0

    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        rows = build_artifact_only_rows(names, artifact_root)
        print_artifact_only_rows(rows)
        return 0

    with tempfile.TemporaryDirectory(prefix="wam-c-child-csr-artifacts-") as tmp:
        rows = build_artifact_only_rows(names, Path(tmp))
        print_artifact_only_rows(rows)
    return 0


def main() -> int:
    args = parse_args()
    if args.artifact_only:
        return run_artifact_only(args.scales, args.artifact_root, args.dry_run)

    extra_args = args.matrix_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = matrix_command(args.scales, extra_args)
    if args.dry_run:
        print(" ".join(command))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
