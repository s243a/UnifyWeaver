#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
generate_synthetic_phase1_lmdb.py — create a synthetic Phase 1 LMDB
fixture with category_parent and category_child sub-dbs.

The fixture is intentionally simple and deterministic. It exists to make
reverse CSR lookup measurements reproducible without depending on local
simplewiki/enwiki artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("generate_synthetic_phase1_lmdb: 'lmdb' Python package required\n")
    sys.exit(1)


I32 = struct.Struct("<i")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic Phase 1 LMDB category graph fixture.")
    parser.add_argument("out_lmdb_dir", type=Path)
    parser.add_argument("--parents", type=int, default=1000)
    parser.add_argument("--children-per-parent", type=int, default=8)
    parser.add_argument("--map-size", type=int, default=1 << 30)
    parser.add_argument("--refresh", action="store_true", help="replace an existing output directory")
    return parser.parse_args(argv)


def i32(value: int) -> bytes:
    return I32.pack(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir: Path = args.out_lmdb_dir
    if args.parents <= 0:
        sys.stderr.write("--parents must be positive\n")
        return 2
    if args.children_per_parent <= 0:
        sys.stderr.write("--children-per-parent must be positive\n")
        return 2
    if out_dir.exists():
        if not args.refresh:
            sys.stderr.write(f"output directory exists; pass --refresh to rebuild: {out_dir}\n")
            return 3
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    edge_count = args.parents * args.children_per_parent
    first_parent_id = 1
    first_child_id = args.parents + 1
    max_id = first_child_id + edge_count - 1

    env = lmdb.open(str(out_dir), map_size=args.map_size, max_dbs=8, subdir=True)
    try:
        s2i_db = env.open_db(b"s2i")
        i2s_db = env.open_db(b"i2s")
        meta_db = env.open_db(b"meta")
        cp_db = env.open_db(b"category_parent", dupsort=True)
        cc_db = env.open_db(b"category_child", dupsort=True)
        ac_db = env.open_db(b"article_category", dupsort=True)

        with env.begin(write=True) as txn:
            child_id = first_child_id
            for parent_offset in range(args.parents):
                parent_id = first_parent_id + parent_offset
                for _ in range(args.children_per_parent):
                    txn.put(i32(child_id), i32(parent_id), db=cp_db)
                    txn.put(i32(parent_id), i32(child_id), db=cc_db)
                    child_id += 1

            txn.put(b"schema_version", b"1", db=meta_db)
            txn.put(b"next_id", i32(max_id + 1), db=meta_db)
            txn.put(b"compile_time_atoms_count", i32(0), db=meta_db)
            txn.put(b"id_encoding", b"int32_le", db=meta_db)
            txn.put(b"category_parent_edge_count", str(edge_count).encode("ascii"), db=meta_db)
            txn.put(b"category_child_edge_count", str(edge_count).encode("ascii"), db=meta_db)
            txn.put(b"reverse_index_relation", b"category_child/2", db=meta_db)
            txn.put(b"reverse_index_storage_kind", b"phase1_lmdb_subdb", db=meta_db)

            # Touch stub dbs so the Phase 1 shape is explicit.
            _ = (s2i_db, i2s_db, ac_db)
    finally:
        env.sync()
        env.close()

    manifest = {
        "format": "unifyweaver.synthetic_phase1_lmdb.v1",
        "id_encoding": "int32_le",
        "parents": args.parents,
        "children_per_parent": args.children_per_parent,
        "edge_count": edge_count,
        "max_id": max_id,
        "relations": {
            "category_parent/2": {
                "database": "category_parent",
                "dupsort": True,
                "edge_count": edge_count,
            },
            "category_child/2": {
                "database": "category_child",
                "dupsort": True,
                "edge_count": edge_count,
                "derived_from": "category_parent/2",
            },
        },
        "build": {
            "tool": "generate_synthetic_phase1_lmdb.py",
            "elapsed_seconds": round(time.time() - started_at, 6),
        },
    }
    (out_dir / "phase1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sys.stderr.write(
        f"generate_synthetic_phase1_lmdb: parents={args.parents} "
        f"children_per_parent={args.children_per_parent} edges={edge_count} -> {out_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
