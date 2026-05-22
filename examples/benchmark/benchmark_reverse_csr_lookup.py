#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
benchmark_reverse_csr_lookup.py — compare reverse CSR lookup against
Phase 1 LMDB category_child lookup.

The benchmark is intentionally narrow: it validates CSR/LMDB parity for
sampled parent keys, then times repeated parent -> children lookups for
both backends. It is a measurement hook, not runtime integration.
"""

from __future__ import annotations

import argparse
import random
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("benchmark_reverse_csr_lookup: 'lmdb' Python package required\n")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "examples" / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from read_reverse_csr_artifact import ReverseCsrArtifact  # noqa: E402


BUILDER = BENCHMARK_DIR / "build_reverse_csr_artifact.py"
I32 = struct.Struct("<i")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark reverse CSR lookup vs Phase 1 LMDB category_child.")
    parser.add_argument("phase1_lmdb_dir", type=Path)
    parser.add_argument("--csr-dir", type=Path, default=None, help="existing or output CSR artifact directory")
    parser.add_argument("--refresh-csr", action="store_true", help="rebuild CSR artifact when --csr-dir exists")
    parser.add_argument("--sample-parents", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--format", choices=["tsv"], default="tsv")
    return parser.parse_args(argv)


class LmdbCategoryChildLookup:
    def __init__(self, phase1_lmdb_dir: Path):
        self.env = lmdb.open(str(phase1_lmdb_dir), readonly=True, max_dbs=8, lock=False, subdir=True)

    def close(self) -> None:
        self.env.close()

    def parents(self) -> list[int]:
        result: list[int] = []
        with self.env.begin() as txn:
            db = self.env.open_db(b"category_child", txn=txn, dupsort=True, create=False)
            cursor = txn.cursor(db=db)
            if cursor.first():
                while True:
                    key = cursor.key()
                    if len(key) == 4:
                        result.append(I32.unpack(key)[0])
                    if not cursor.next_nodup():
                        break
        return result

    def lookup(self, parent: int) -> list[int]:
        key = I32.pack(parent)
        children: list[int] = []
        with self.env.begin() as txn:
            db = self.env.open_db(b"category_child", txn=txn, dupsort=True, create=False)
            cursor = txn.cursor(db=db)
            if not cursor.set_key(key):
                return []
            while True:
                value = cursor.value()
                if len(value) == 4:
                    children.append(I32.unpack(value)[0])
                if not cursor.next_dup():
                    break
        return sorted(children)


def ensure_csr_artifact(phase1_lmdb_dir: Path, csr_dir: Path, refresh: bool) -> None:
    if csr_dir.exists() and not refresh:
        return
    command = [sys.executable, str(BUILDER), str(phase1_lmdb_dir), str(csr_dir)]
    if refresh:
        command.append("--refresh")
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CSR builder failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def sample_parent_ids(parent_ids: list[int], count: int, seed: int) -> list[int]:
    if count <= 0 or count >= len(parent_ids):
        return list(parent_ids)
    rng = random.Random(seed)
    return sorted(rng.sample(parent_ids, count))


def time_lookup_loop(label: str, lookup, parent_ids: list[int], iterations: int) -> tuple[str, list[float], int]:
    times_ms: list[float] = []
    total_children = 0
    for _ in range(iterations):
        started = time.perf_counter()
        child_count = 0
        for parent in parent_ids:
            child_count += len(lookup(parent))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        times_ms.append(elapsed_ms)
        total_children = child_count
    return label, times_ms, total_children


def artifact_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def print_tsv(rows: list[dict[str, str]]) -> None:
    headers = [
        "backend",
        "sample_parents",
        "iterations",
        "total_children",
        "median_ms",
        "min_ms",
        "max_ms",
        "artifact_bytes",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(row[h] for h in headers))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    phase1_lmdb_dir: Path = args.phase1_lmdb_dir
    if not (phase1_lmdb_dir / "data.mdb").exists():
        sys.stderr.write(f"missing {phase1_lmdb_dir}/data.mdb\n")
        return 2

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.csr_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        csr_dir = Path(temp_dir.name) / "category_child_csr"
        refresh_csr = False
    else:
        csr_dir = args.csr_dir
        refresh_csr = args.refresh_csr

    try:
        ensure_csr_artifact(phase1_lmdb_dir, csr_dir, refresh_csr)
        csr = ReverseCsrArtifact(csr_dir)
        lmdb_lookup = LmdbCategoryChildLookup(phase1_lmdb_dir)
        try:
            parent_ids = sample_parent_ids(lmdb_lookup.parents(), args.sample_parents, args.seed)
            if not parent_ids:
                sys.stderr.write("no parent ids found in category_child\n")
                return 4

            for parent in parent_ids:
                csr_children = csr.lookup(parent)
                lmdb_children = lmdb_lookup.lookup(parent)
                if csr_children != lmdb_children:
                    sys.stderr.write(
                        f"parity mismatch for parent={parent}: "
                        f"csr={csr_children[:10]} lmdb={lmdb_children[:10]}\n"
                    )
                    return 5

            timed = [
                time_lookup_loop("csr", csr.lookup, parent_ids, args.iterations),
                time_lookup_loop("lmdb", lmdb_lookup.lookup, parent_ids, args.iterations),
            ]
            rows: list[dict[str, str]] = []
            for backend, times_ms, total_children in timed:
                rows.append({
                    "backend": backend,
                    "sample_parents": str(len(parent_ids)),
                    "iterations": str(args.iterations),
                    "total_children": str(total_children),
                    "median_ms": f"{statistics.median(times_ms):.6f}",
                    "min_ms": f"{min(times_ms):.6f}",
                    "max_ms": f"{max(times_ms):.6f}",
                    "artifact_bytes": str(artifact_bytes(csr_dir) if backend == "csr" else (phase1_lmdb_dir / "data.mdb").stat().st_size),
                })
            print_tsv(rows)
            return 0
        finally:
            lmdb_lookup.close()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
