#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
build_csr_artifact.py — build a CSR artifact from a Phase 1 LMDB
DUPSORT sub-database.

Unlike build_reverse_csr_artifact.py (which always reverses edges),
this builder can produce CSR in either direction:

  --relation category_parent  (default: child -> [parents], forward)
  --relation category_child   (reverse: parent -> [children])

Output:
  <relation>.csr.idx   records: int32 key, uint64 offset_edges, uint32 count
  <relation>.csr.val   records: int32 value
  <relation>.csr.meta  JSON manifest
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
    sys.stderr.write("build_csr_artifact: 'lmdb' Python package required\n")
    sys.exit(1)


IDX_RECORD = struct.Struct("<iQI")
I32 = struct.Struct("<i")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSR artifact from Phase 1 LMDB DUPSORT database.",
    )
    parser.add_argument("phase1_lmdb_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--relation",
        default="category_parent",
        help="LMDB sub-db name to read (default: category_parent)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    src_dir: Path = args.phase1_lmdb_dir
    out_dir: Path = args.out_dir
    relation: str = args.relation
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

    # Read edges directly from the named DUPSORT database.
    # key -> [values] grouping preserves the database's own direction.
    values_by_key: dict[int, list[int]] = defaultdict(list)
    edge_count = 0
    max_id = 0

    env = lmdb.open(str(src_dir), readonly=True, max_dbs=8, lock=False, subdir=True)
    try:
        with env.begin() as txn:
            db = env.open_db(relation.encode("ascii"), txn=txn, dupsort=True, create=False)
            cursor = txn.cursor(db=db)
            for key, value in cursor:
                if len(key) != 4 or len(value) != 4:
                    continue
                k = I32.unpack(key)[0]
                v = I32.unpack(value)[0]
                values_by_key[k].append(v)
                edge_count += 1
                max_id = max(max_id, k, v)
    finally:
        env.close()

    prefix = f"{relation}.csr"
    idx_path = out_dir / f"{prefix}.idx"
    val_path = out_dir / f"{prefix}.val"
    meta_path = out_dir / f"{prefix}.meta"

    offset_edges = 0
    key_count = 0
    sorted_keys = sorted(values_by_key)
    with idx_path.open("wb") as idx_file, val_path.open("wb") as val_file:
        for k in sorted_keys:
            values = sorted(values_by_key[k])
            idx_file.write(IDX_RECORD.pack(k, offset_edges, len(values)))
            for v in values:
                val_file.write(I32.pack(v))
            offset_edges += len(values)
            key_count += 1

    elapsed_seconds = time.time() - started_at
    manifest = {
        "format": "unifyweaver.reverse_csr.v1",
        "schema_version": 1,
        "relation": f"{relation}/2",
        "source_relation": f"{relation}/2",
        "source_environment_path": str(src_dir),
        "storage_kind": "csr_pread_artifact",
        "id_encoding": "int32_le",
        "ordering": "key_sort",
        "index_backend": "sorted_array",
        "io_policy": "buffered_pread",
        "index_path": idx_path.name,
        "values_path": val_path.name,
        "index_record_format": "int32_le key, uint64_le offset_edges, uint32_le count_edges",
        "value_record_format": "int32_le value",
        "index_record_bytes": IDX_RECORD.size,
        "value_record_bytes": I32.size,
        "parent_count": key_count,
        "edge_count": edge_count,
        "max_id": max_id,
        "build": {
            "tool": "build_csr_artifact.py",
            "elapsed_seconds": round(elapsed_seconds, 6),
        },
    }
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sys.stderr.write(
        f"build_csr_artifact: relation={relation} keys={key_count} edges={edge_count} "
        f"max_id={max_id} -> {out_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
