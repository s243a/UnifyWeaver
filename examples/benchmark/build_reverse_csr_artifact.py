#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
build_reverse_csr_artifact.py — build a parent-sorted CSR reverse index
from a Phase 1 LMDB category_parent sub-db.

Input:
  category_parent  int32_le child -> int32_le parent (DUPSORT)

Output:
  category_child.csr.idx   records: int32 parent, uint64 offset_edges, uint32 count
  category_child.csr.val   records: int32 child
  category_child.csr.meta  JSON manifest

The output relation is category_child(parent, child). The prototype is
intentionally parent_sort only and uses ordinary buffered reads on the
reader side; direct I/O policy is deferred until the format is stable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("build_reverse_csr_artifact: 'lmdb' Python package required\n")
    sys.exit(1)


IDX_RECORD = struct.Struct("<iQI")
I32 = struct.Struct("<i")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a parent-sorted CSR category_child reverse artifact from Phase 1 LMDB.",
    )
    parser.add_argument("phase1_lmdb_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--refresh", action="store_true", help="replace an existing output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    src_dir: Path = args.phase1_lmdb_dir
    out_dir: Path = args.out_dir
    started_at = time.time()

    if not (src_dir / "data.mdb").exists():
        sys.stderr.write(f"missing {src_dir}/data.mdb\n")
        return 2
    if out_dir.exists():
        if not args.refresh:
            sys.stderr.write(f"output directory exists; pass --refresh to rebuild: {out_dir}\n")
            return 3
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    children_by_parent: dict[int, list[int]] = defaultdict(list)
    edge_count = 0
    max_id = 0

    env = lmdb.open(str(src_dir), readonly=True, max_dbs=8, lock=False, subdir=True)
    try:
        with env.begin() as txn:
            cp_db = env.open_db(b"category_parent", txn=txn, dupsort=True, create=False)
            cursor = txn.cursor(db=cp_db)
            for key, value in cursor:
                if len(key) != 4 or len(value) != 4:
                    continue
                child = I32.unpack(key)[0]
                parent = I32.unpack(value)[0]
                children_by_parent[parent].append(child)
                edge_count += 1
                max_id = max(max_id, child, parent)
    finally:
        env.close()

    idx_path = out_dir / "category_child.csr.idx"
    val_path = out_dir / "category_child.csr.val"
    meta_path = out_dir / "category_child.csr.meta"

    offset_edges = 0
    parent_count = 0
    with idx_path.open("wb") as idx_file, val_path.open("wb") as val_file:
        for parent in sorted(children_by_parent):
            children = sorted(children_by_parent[parent])
            idx_file.write(IDX_RECORD.pack(parent, offset_edges, len(children)))
            for child in children:
                val_file.write(I32.pack(child))
            offset_edges += len(children)
            parent_count += 1

    elapsed_seconds = time.time() - started_at
    manifest = {
        "format": "unifyweaver.reverse_csr.v1",
        "schema_version": 1,
        "relation": "category_child/2",
        "source_relation": "category_parent/2",
        "source_environment_path": str(src_dir),
        "storage_kind": "csr_pread_artifact",
        "id_encoding": "int32_le",
        "ordering": "parent_sort",
        "io_policy": "buffered_pread",
        "index_path": idx_path.name,
        "values_path": val_path.name,
        "index_record_format": "int32_le parent, uint64_le offset_edges, uint32_le count_edges",
        "value_record_format": "int32_le child",
        "index_record_bytes": IDX_RECORD.size,
        "value_record_bytes": I32.size,
        "parent_count": parent_count,
        "edge_count": edge_count,
        "max_id": max_id,
        "build": {
            "tool": "build_reverse_csr_artifact.py",
            "elapsed_seconds": round(elapsed_seconds, 6),
        },
    }
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sys.stderr.write(
        f"build_reverse_csr_artifact: parents={parent_count} edges={edge_count} "
        f"max_id={max_id} -> {out_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
