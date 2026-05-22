#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
convert_lmdb_to_phase1_layout.py — convert older streaming-pipeline
LMDBs (single dupsort `main` sub-db with int32 child → int32 parent
edges) into the Phase 1 layout that resident_cursor mode expects:

  s2i              UTF-8 string  -> int32_le      (intern table; empty stub)
  i2s              int32_le      -> UTF-8 string  (intern table; empty stub)
  meta             ASCII keys    -> bytes         (schema_version etc.)
  category_parent  int32_le      -> int32_le      (mirror of `main`)
  category_child   int32_le      -> int32_le      (reverse adjacency)
  article_category int32_le      -> int32_le      (empty stub)

Used to retrofit the simplewiki / enwiki LMDBs that pre-date the
Phase 1 ingester convention. The s2i/i2s/article_category sub-dbs
are present-but-empty: resident_cursor mode loads them at startup
but doesn't exercise them at runtime (kernel uses cpEdgeLookup,
demand BFS uses category_child, output is per-seed int IDs).

Usage:
  python3 convert_lmdb_to_phase1_layout.py <src_lmdb_dir> <dst_lmdb_dir>

  src_lmdb_dir  must contain a `main` dupsort sub-db with int32_le
                child → int32_le parent edges.
  dst_lmdb_dir  is created with the Phase 1 layout above.

Idempotent within a fresh dst_lmdb_dir; will refuse if dst already
has a populated meta sub-db.
"""

import os
import json
import struct
import sys
import time
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("convert_lmdb_to_phase1_layout: 'lmdb' Python package required\n")
    sys.exit(1)


def main() -> int:
    started_at = time.time()
    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: convert_lmdb_to_phase1_layout.py <src_lmdb_dir> <dst_lmdb_dir>\n"
        )
        return 2

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    if not (src_dir / "data.mdb").exists():
        sys.stderr.write(f"missing {src_dir}/data.mdb\n")
        return 2
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Read source: open the named "main" dupsort sub-db.
    src_env = lmdb.open(
        str(src_dir),
        readonly=True,
        max_dbs=4,
        lock=False,
    )
    # Map size: pick 4x source size so dst has room for forward + reverse
    # plus stub sub-dbs.
    src_data_size = (src_dir / "data.mdb").stat().st_size
    dst_map_size = max(4 * src_data_size, 1 << 30)

    dst_env = lmdb.open(
        str(dst_dir),
        map_size=dst_map_size,
        max_dbs=8,
        subdir=True,
    )
    try:
        # Refuse to clobber populated dst.
        meta_db = dst_env.open_db(b"meta")
        with dst_env.begin() as txn:
            if txn.get(b"schema_version", db=meta_db) is not None:
                sys.stderr.write(
                    f"convert_lmdb_to_phase1_layout: dst {dst_dir} already has "
                    f"meta.schema_version; refusing to overwrite\n"
                )
                return 3

        # Open dst sub-dbs.
        s2i_db = dst_env.open_db(b"s2i")
        i2s_db = dst_env.open_db(b"i2s")
        cp_db = dst_env.open_db(b"category_parent", dupsort=True)
        cc_db = dst_env.open_db(b"category_child", dupsort=True)
        ac_db = dst_env.open_db(b"article_category", dupsort=True)

        cp_count = 0
        max_id = 0

        with src_env.begin() as src_txn:
            src_main = src_env.open_db(b"main", txn=src_txn, dupsort=True, create=False)
            with dst_env.begin(write=True) as dst_txn:
                cur = src_txn.cursor(db=src_main)
                for k, v in cur:
                    if len(k) != 4 or len(v) != 4:
                        continue
                    dst_txn.put(k, v, db=cp_db)
                    dst_txn.put(v, k, db=cc_db)
                    cp_count += 1
                    ki = struct.unpack("<i", k)[0]
                    vi = struct.unpack("<i", v)[0]
                    if ki > max_id:
                        max_id = ki
                    if vi > max_id:
                        max_id = vi
                    if cp_count % 500_000 == 0:
                        sys.stderr.write(f"  ...{cp_count} edges copied\n")

                # Meta keys (compatibility with loaders).
                next_id = max_id + 1
                dst_txn.put(b"schema_version", b"1", db=meta_db)
                dst_txn.put(b"next_id", struct.pack("<i", next_id), db=meta_db)
                dst_txn.put(b"compile_time_atoms_count", struct.pack("<i", 0), db=meta_db)
                dst_txn.put(b"cli_args", " ".join(sys.argv).encode("utf-8"), db=meta_db)
                dst_txn.put(b"converted_from", str(src_dir).encode("utf-8"), db=meta_db)
                dst_txn.put(b"id_encoding", b"int32_le", db=meta_db)
                dst_txn.put(b"category_parent_edge_count", str(cp_count).encode("ascii"), db=meta_db)
                dst_txn.put(b"category_child_edge_count", str(cp_count).encode("ascii"), db=meta_db)
                dst_txn.put(b"reverse_index_relation", b"category_child/2", db=meta_db)
                dst_txn.put(b"reverse_index_storage_kind", b"phase1_lmdb_subdb", db=meta_db)

    finally:
        src_env.close()
        dst_env.sync()
        dst_env.close()

    elapsed_seconds = time.time() - started_at
    write_phase1_manifest(
        dst_dir,
        src_dir,
        cp_count,
        max_id,
        elapsed_seconds,
    )

    sys.stderr.write(
        f"convert_lmdb_to_phase1_layout: cp_edges={cp_count} "
        f"cc_edges={cp_count} max_id={max_id} -> {dst_dir}\n"
    )
    return 0


def write_phase1_manifest(
    dst_dir: Path,
    src_dir: Path,
    edge_count: int,
    max_id: int,
    elapsed_seconds: float,
) -> None:
    manifest = {
        "format": "unifyweaver.phase1_lmdb_manifest.v1",
        "schema_version": 1,
        "environment_path": str(dst_dir),
        "source_environment_path": str(src_dir),
        "id_encoding": "int32_le",
        "max_id": max_id,
        "relations": {
            "category_parent/2": {
                "database": "category_parent",
                "dupsort": True,
                "edge_count": edge_count,
                "key": "child",
                "value": "parent",
            },
            "category_child/2": {
                "database": "category_child",
                "dupsort": True,
                "edge_count": edge_count,
                "key": "parent",
                "value": "child",
                "derived_from": "category_parent/2",
                "storage_kind": "phase1_lmdb_subdb",
            },
            "article_category/2": {
                "database": "article_category",
                "dupsort": True,
                "edge_count": 0,
            },
        },
        "reverse_indexes": [
            {
                "relation": "category_child/2",
                "source_relation": "category_parent/2",
                "storage_kind": "phase1_lmdb_subdb",
                "id_encoding": "int32_le",
                "edge_count": edge_count,
            }
        ],
        "build": {
            "tool": "convert_lmdb_to_phase1_layout.py",
            "elapsed_seconds": round(elapsed_seconds, 6),
        },
    }
    manifest_path = dst_dir / "phase1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
