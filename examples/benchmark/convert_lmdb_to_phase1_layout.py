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
import struct
import sys
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("convert_lmdb_to_phase1_layout: 'lmdb' Python package required\n")
    sys.exit(1)


def main() -> int:
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

    finally:
        src_env.close()
        dst_env.sync()
        dst_env.close()

    sys.stderr.write(
        f"convert_lmdb_to_phase1_layout: cp_edges={cp_count} "
        f"cc_edges={cp_count} max_id={max_id} -> {dst_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
