#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
ingest_resident_lmdb_fixture.py — single-pass dual-table ingester for
WAM-Haskell LMDB-resident benchmark fixtures.

The production ingester (src/unifyweaver/runtime/python/lmdb_ingest/
ingest_to_lmdb.py) is single-shot per LMDB path: a second invocation
against the same path bails with the schema_version guard, and
UW_FORCE_REINGEST=1 wipes the prior state — neither matches the
"category_parent + article_category share one intern table" shape
that int_atom_seeds(lmdb) requires.

This helper is scoped to that benchmark layout. It reads a fixture
directory containing `category_parent.tsv` and `article_category.tsv`
(both with header lines + tab-separated `child/parent` and
`article/category` rows) and writes one LMDB containing:

  s2i              UTF-8 string  -> int32_le      (forward intern map)
  i2s              int32_le      -> UTF-8 string  (reverse intern map)
  meta             ASCII keys    -> bytes         (schema_version, next_id)
  category_parent  int32_le      -> int32_le      (dupsort, child -> parent)
  article_category int32_le      -> int32_le      (dupsort, article -> cat)

Atom IDs allocated dense-from-zero in TSV-row order (category_parent
first, then article_category). No compile-time atom reservation: the
WAM-Haskell loader binds `iAtom = id`, so any consistent allocation
scheme works.

Also writes companion files alongside the LMDB:

  seed_ids.txt   newline-separated int32 IDs of distinct seed categories
                 (the set of category column values in article_category.tsv)
  root_ids.txt   newline-separated int32 IDs from root_categories.tsv

Usage:
  python3 ingest_resident_lmdb_fixture.py <fixture_dir> <out_lmdb_dir>
"""

import os
import struct
import sys
from pathlib import Path

try:
    import lmdb
except ImportError:
    sys.stderr.write("ingest_resident_lmdb_fixture: 'lmdb' Python package required\n")
    sys.exit(1)


def le32(i: int) -> bytes:
    return struct.pack("<i", i)


def iter_tsv_pairs(path: Path):
    """Yield (col0, col1) for each non-header row of a 2-column TSV."""
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            yield parts[0], parts[1]


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(
            "Usage: ingest_resident_lmdb_fixture.py <fixture_dir> <out_lmdb_dir>\n"
        )
        return 2

    fixture_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    cp_tsv = fixture_dir / "category_parent.tsv"
    ac_tsv = fixture_dir / "article_category.tsv"
    roots_tsv = fixture_dir / "root_categories.tsv"
    for p in (cp_tsv, ac_tsv):
        if not p.exists():
            sys.stderr.write(f"missing {p}\n")
            return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 GB map suffices for fixtures up to ~enwiki-100k. Bump if needed.
    env = lmdb.open(
        str(out_dir),
        map_size=1 << 30,
        max_dbs=8,
        subdir=True,
    )
    try:
        s2i_db = env.open_db(b"s2i")
        i2s_db = env.open_db(b"i2s")
        meta_db = env.open_db(b"meta")
        cp_db = env.open_db(b"category_parent", dupsort=True)
        ac_db = env.open_db(b"article_category", dupsort=True)

        intern: dict[str, int] = {}

        def intern_atom(s: str, txn) -> int:
            aid = intern.get(s)
            if aid is not None:
                return aid
            aid = len(intern)
            intern[s] = aid
            txn.put(s.encode("utf-8"), le32(aid), db=s2i_db)
            txn.put(le32(aid), s.encode("utf-8"), db=i2s_db)
            return aid

        cp_count = 0
        ac_count = 0
        seed_categories: list[int] = []
        seen_seed: set[int] = set()

        with env.begin(write=True) as txn:
            for child, parent in iter_tsv_pairs(cp_tsv):
                cid = intern_atom(child, txn)
                pid = intern_atom(parent, txn)
                txn.put(le32(cid), le32(pid), db=cp_db)
                cp_count += 1
            for article, category in iter_tsv_pairs(ac_tsv):
                aid = intern_atom(article, txn)
                kid = intern_atom(category, txn)
                txn.put(le32(aid), le32(kid), db=ac_db)
                ac_count += 1
                if kid not in seen_seed:
                    seen_seed.add(kid)
                    seed_categories.append(kid)

            txn.put(b"schema_version", b"1", db=meta_db)
            txn.put(b"next_id", le32(len(intern)), db=meta_db)
            txn.put(b"compile_time_atoms_count", le32(0), db=meta_db)
            txn.put(b"cli_args", " ".join(sys.argv).encode("utf-8"), db=meta_db)

        # Companion files for the int_atom_seeds(lmdb) WAM-Haskell loader.
        # These live next to the LMDB so the binary's factsDir argument
        # can point at the parent of the lmdb/ subdir.
        seed_ids_path = out_dir.parent / "seed_ids.txt"
        root_ids_path = out_dir.parent / "root_ids.txt"
        with open(seed_ids_path, "w", encoding="ascii") as f:
            for sid in seed_categories:
                f.write(f"{sid}\n")
        root_ids: list[int] = []
        if roots_tsv.exists():
            with open(roots_tsv, "r", encoding="utf-8") as f:
                next(f, None)  # skip header
                for raw in f:
                    name = raw.rstrip("\n").rstrip("\r").strip()
                    if not name:
                        continue
                    rid = intern.get(name)
                    if rid is not None:
                        root_ids.append(rid)
                    else:
                        sys.stderr.write(
                            f"warning: root '{name}' not in intern table; skipped\n"
                        )
        with open(root_ids_path, "w", encoding="ascii") as f:
            for rid in root_ids:
                f.write(f"{rid}\n")
    finally:
        env.sync()
        env.close()

    sys.stderr.write(
        f"ingest_resident_lmdb_fixture: cp_edges={cp_count} "
        f"ac_edges={ac_count} interned={len(intern)} "
        f"seeds={len(seed_categories)} roots={len(root_ids)} -> {out_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
