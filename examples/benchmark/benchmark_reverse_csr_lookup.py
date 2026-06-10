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
import json
import random
import shutil
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
    parser.add_argument(
        "--csr-index-backends",
        default="sorted_array",
        help="comma-separated CSR index backends to benchmark: sorted_array,lmdb_offset",
    )
    parser.add_argument(
        "--parent-lmdb-dir",
        type=Path,
        default=None,
        help="existing or output category_parent-only LMDB directory for size comparison",
    )
    parser.add_argument(
        "--refresh-parent-lmdb",
        action="store_true",
        help="rebuild the parent-only LMDB when --parent-lmdb-dir exists",
    )
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


def parse_csr_index_backends(value: str) -> list[str]:
    backends = [part.strip() for part in value.split(",") if part.strip()]
    if not backends:
        raise ValueError("--csr-index-backends must list at least one backend")
    allowed = {"sorted_array", "lmdb_offset"}
    unknown = sorted(set(backends) - allowed)
    if unknown:
        raise ValueError(f"unsupported CSR index backend(s): {', '.join(unknown)}")
    return backends


def ensure_csr_artifact(phase1_lmdb_dir: Path, csr_dir: Path, refresh: bool, index_backend: str) -> None:
    if csr_dir.exists() and not refresh:
        return
    command = [
        sys.executable,
        str(BUILDER),
        str(phase1_lmdb_dir),
        str(csr_dir),
        "--index-backend",
        index_backend,
    ]
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


def ensure_parent_only_lmdb(phase1_lmdb_dir: Path, parent_lmdb_dir: Path, refresh: bool) -> None:
    if parent_lmdb_dir.resolve() == phase1_lmdb_dir.resolve():
        raise ValueError("--parent-lmdb-dir must be different from phase1_lmdb_dir")
    if parent_lmdb_dir.exists() and not refresh:
        return
    if parent_lmdb_dir.exists():
        shutil.rmtree(parent_lmdb_dir)
    parent_lmdb_dir.mkdir(parents=True, exist_ok=True)

    source_size = (phase1_lmdb_dir / "data.mdb").stat().st_size
    dst_env = lmdb.open(
        str(parent_lmdb_dir),
        map_size=max(source_size, 1 << 20),
        max_dbs=8,
        subdir=True,
    )
    src_env = lmdb.open(str(phase1_lmdb_dir), readonly=True, max_dbs=8, lock=False, subdir=True)
    edge_count = 0
    try:
        with src_env.begin() as src_txn:
            src_cp_db = src_env.open_db(b"category_parent", txn=src_txn, dupsort=True, create=False)
            dst_meta_db = dst_env.open_db(b"meta")
            dst_cp_db = dst_env.open_db(b"category_parent", dupsort=True)
            dst_s2i_db = dst_env.open_db(b"s2i")
            dst_i2s_db = dst_env.open_db(b"i2s")
            dst_ac_db = dst_env.open_db(b"article_category", dupsort=True)
            with dst_env.begin(write=True) as dst_txn:
                cursor = src_txn.cursor(db=src_cp_db)
                for key, value in cursor:
                    if len(key) != 4 or len(value) != 4:
                        continue
                    dst_txn.put(key, value, db=dst_cp_db)
                    edge_count += 1
                dst_txn.put(b"schema_version", b"1", db=dst_meta_db)
                dst_txn.put(b"id_encoding", b"int32_le", db=dst_meta_db)
                dst_txn.put(b"category_parent_edge_count", str(edge_count).encode("ascii"), db=dst_meta_db)
                dst_txn.put(b"hot_relation", b"category_parent/2", db=dst_meta_db)
                dst_txn.put(b"storage_scope", b"parent_only_size_probe", db=dst_meta_db)
                _ = (dst_s2i_db, dst_i2s_db, dst_ac_db)
    finally:
        src_env.close()
        dst_env.sync()
        dst_env.close()


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
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def csr_manifest(path: Path) -> dict:
    return json.loads((path / "category_child.csr.meta").read_text(encoding="utf-8"))


def print_tsv(rows: list[dict[str, str]]) -> None:
    headers = [
        "backend",
        "index_backend",
        "sample_parents",
        "iterations",
        "total_children",
        "median_ms",
        "min_ms",
        "max_ms",
        "csr_artifact_bytes",
        "csr_build_seconds",
        "offset_index_bytes",
        "parent_lmdb_env_bytes",
        "phase1_lmdb_env_bytes",
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
    try:
        csr_index_backends = parse_csr_index_backends(args.csr_index_backends)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.csr_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)
        csr_dir = temp_path / "category_child_csr"
        parent_lmdb_dir = args.parent_lmdb_dir or temp_path / "category_parent_only.lmdb"
        refresh_csr = False
        refresh_parent_lmdb = args.refresh_parent_lmdb
    else:
        csr_dir = args.csr_dir
        parent_lmdb_dir = args.parent_lmdb_dir
        refresh_csr = args.refresh_csr
        refresh_parent_lmdb = args.refresh_parent_lmdb
    if parent_lmdb_dir is None:
        if temp_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
        parent_lmdb_dir = Path(temp_dir.name) / "category_parent_only.lmdb"
    csr_dirs = {
        index_backend: csr_dir if len(csr_index_backends) == 1 else csr_dir / index_backend
        for index_backend in csr_index_backends
    }

    try:
        for index_backend, one_csr_dir in csr_dirs.items():
            ensure_csr_artifact(phase1_lmdb_dir, one_csr_dir, refresh_csr, index_backend)
        ensure_parent_only_lmdb(phase1_lmdb_dir, parent_lmdb_dir, refresh_parent_lmdb)
        csrs = {index_backend: ReverseCsrArtifact(one_csr_dir) for index_backend, one_csr_dir in csr_dirs.items()}
        try:
            lmdb_lookup = LmdbCategoryChildLookup(phase1_lmdb_dir)
            try:
                parent_ids = sample_parent_ids(lmdb_lookup.parents(), args.sample_parents, args.seed)
                if not parent_ids:
                    sys.stderr.write("no parent ids found in category_child\n")
                    return 4

                for parent in parent_ids:
                    lmdb_children = lmdb_lookup.lookup(parent)
                    for index_backend, csr in csrs.items():
                        csr_children = csr.lookup(parent)
                        if csr_children != lmdb_children:
                            sys.stderr.write(
                                f"parity mismatch for parent={parent} index_backend={index_backend}: "
                                f"csr={csr_children[:10]} lmdb={lmdb_children[:10]}\n"
                            )
                            return 5

                timed = []
                for index_backend, csr in csrs.items():
                    timed.append((
                        index_backend,
                        *time_lookup_loop(f"csr_{index_backend}", csr.lookup, parent_ids, args.iterations),
                    ))
                timed.append((
                    "n/a",
                    *time_lookup_loop("lmdb", lmdb_lookup.lookup, parent_ids, args.iterations),
                ))
                rows: list[dict[str, str]] = []
                for index_backend, backend, times_ms, total_children in timed:
                    one_csr_dir = csr_dirs[csr_index_backends[0]] if index_backend == "n/a" else csr_dirs[index_backend]
                    manifest = csr_manifest(one_csr_dir)
                    rows.append({
                        "backend": backend,
                        "index_backend": index_backend,
                        "sample_parents": str(len(parent_ids)),
                        "iterations": str(args.iterations),
                        "total_children": str(total_children),
                        "median_ms": f"{statistics.median(times_ms):.6f}",
                        "min_ms": f"{min(times_ms):.6f}",
                        "max_ms": f"{max(times_ms):.6f}",
                        "csr_artifact_bytes": str(artifact_bytes(one_csr_dir)),
                        "csr_build_seconds": f"{manifest['build']['elapsed_seconds']:.6f}",
                        "offset_index_bytes": str(manifest.get("offset_index_bytes", 0)),
                        "parent_lmdb_env_bytes": str((parent_lmdb_dir / "data.mdb").stat().st_size),
                        "phase1_lmdb_env_bytes": str((phase1_lmdb_dir / "data.mdb").stat().st_size),
                    })
                print_tsv(rows)
                return 0
            finally:
                lmdb_lookup.close()
        finally:
            for csr in csrs.values():
                csr.close()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
