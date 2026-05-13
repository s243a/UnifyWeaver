#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
ingest_to_lmdb — Python consumer for the UnifyWeaver streaming glue.

Reads TSV records from stdin and writes (key, value) pairs to an LMDB
database.  Designed to pair with the `mysql_stream` Rust leaf primitive
(or any other producer that emits compatible TSV).

Row filtering and column selection are configured by environment
variables rather than hard-coded.  The Prolog glue layer sets these
based on the composed pipeline's predicate declarations.

Environment configuration:
  UW_LMDB_PATH       (required) LMDB database directory
  UW_LMDB_MAP_SIZE   (optional) size in bytes, default 1 GB
  UW_LMDB_DBNAME     (optional) named sub-DB; default unnamed (main)
  UW_LMDB_DUPSORT    (optional) "1" to allow multiple values per key
                                (needed for many-to-many relations like
                                 the category→parent edge list); default 0
  UW_FILTER_COL      (optional) column index (0-based) to filter on
  UW_FILTER_VAL      (optional) value the filter column must match
  UW_KEY_COL         (optional) column index to use as the LMDB key
                                (default 0)
  UW_VAL_COL         (optional) column index to use as the LMDB value
                                (default 1)
  UW_KEY_ENCODING    (optional) int32_le | utf8 (default utf8)
  UW_VAL_ENCODING    (optional) int32_le | utf8 (default utf8)
  UW_BATCH_SIZE      (optional) records per LMDB txn, default 10_000

Text-keyed intern mode (optional, default off — backward-compatible):
  UW_INTERN_KEY      (optional) "1" to intern the key column as an Int.
                                Output written as int32_le. Requires
                                UW_LMDB_S2I_DB / UW_LMDB_I2S_DB / UW_LMDB_META_DB.
  UW_INTERN_VAL      (optional) "1" to intern the value column. Same
                                requirements as UW_INTERN_KEY.
  UW_LMDB_S2I_DB     (optional) sub-db for the forward intern map (string → int32_le).
  UW_LMDB_I2S_DB     (optional) sub-db for the reverse intern map (int32_le → string).
  UW_LMDB_META_DB    (optional) sub-db for metadata keys (schema_version,
                                next_id, source_dump_sha256, build_timestamp,
                                cli_args, compile_time_atoms_count).
  UW_LMDB_APPEND     (optional) "1" to use lmdb append=True on edge writes
                                (only safe when input is already sorted by
                                 the chosen key column).
  UW_COMPILE_TIME_ATOMS  (optional) path to a one-atom-per-line file. The
                                pre-listed atoms are interned at IDs 0..N-1
                                so the codegen's compile-time atom table
                                stays aligned with the LMDB.
  UW_SCHEMA_VERSION  (optional) recorded in meta.schema_version (default "1").
  UW_SOURCE_SHA      (optional) recorded in meta.source_dump_sha256.
  UW_FORCE_REINGEST  (optional) "1" to overwrite an existing populated LMDB.

The filter/projection happens in Python because that's where the Prolog
layer places *logic*.  The parser (Rust) stays schema-agnostic.

Usage (integer-keyed enwiki):
  UW_LMDB_PATH=./subcats.lmdb \\
  UW_FILTER_COL=4 UW_FILTER_VAL=subcat \\
  UW_KEY_COL=0 UW_VAL_COL=6 \\
  UW_KEY_ENCODING=int32_le UW_VAL_ENCODING=int32_le \\
    mysql_stream dump.sql.gz | python3 ingest_to_lmdb.py

Usage (text-keyed simplewiki):
  UW_LMDB_PATH=./cats.lmdb UW_LMDB_DBNAME=category_parent UW_LMDB_DUPSORT=1 \\
  UW_FILTER_COL=4 UW_FILTER_VAL=subcat \\
  UW_KEY_COL=0 UW_VAL_COL=6 \\
  UW_INTERN_KEY=1 UW_INTERN_VAL=1 \\
  UW_LMDB_S2I_DB=s2i UW_LMDB_I2S_DB=i2s UW_LMDB_META_DB=meta \\
  UW_LMDB_APPEND=1 \\
    mysql_stream dump.sql.gz | python3 ingest_to_lmdb.py
"""

import datetime
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

try:
    import lmdb
except ImportError:
    sys.stderr.write("ingest_to_lmdb: the 'lmdb' Python package is required\n")
    sys.exit(1)


def tsv_unescape(field: str) -> str:
    """Reverse the TSV escapes emitted by mysql_stream."""
    if "\\" not in field:
        return field
    out = []
    i = 0
    n = len(field)
    while i < n:
        c = field[i]
        if c == "\\" and i + 1 < n:
            nxt = field[i + 1]
            if nxt == "\\":
                out.append("\\")
                i += 2
            elif nxt == "t":
                out.append("\t")
                i += 2
            elif nxt == "n":
                out.append("\n")
                i += 2
            elif nxt == "r":
                out.append("\r")
                i += 2
            elif nxt == "N":
                # NULL — caller has to decide how to interpret; we return
                # the literal "\N" and let upstream detect it.
                out.append("\\N")
                i += 2
            elif nxt == "x" and i + 3 < n:
                try:
                    out.append(chr(int(field[i + 2 : i + 4], 16)))
                    i += 4
                except ValueError:
                    out.append(c)
                    i += 1
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def encode(value: str, encoding: str) -> bytes:
    """Encode a raw TSV field into the bytes written to LMDB."""
    if encoding == "int32_le":
        return struct.pack("<i", int(value))
    if encoding == "utf8":
        return tsv_unescape(value).encode("utf-8")
    raise ValueError(f"unknown encoding: {encoding!r}")


def encode_int32_le(i: int) -> bytes:
    """Encode an Int directly as int32_le (no parse step)."""
    return struct.pack("<i", i)


def load_compile_time_atoms(path: str) -> List[str]:
    """Read a one-atom-per-line file. Blank lines are skipped.

    Order matters: each atom's line index becomes its reserved ID.
    Duplicates are an error (the codegen's compile-time table must be
    a set, not a multiset).
    """
    atoms: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            atom = raw.rstrip("\n")
            if not atom:
                continue
            if atom in seen:
                raise ValueError(
                    f"compile-time atoms file {path!r} has duplicate entry "
                    f"{atom!r} at line {len(atoms) + 1}"
                )
            seen.add(atom)
            atoms.append(atom)
    return atoms


def int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def opt_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else None


def main() -> int:
    lmdb_path = os.environ.get("UW_LMDB_PATH")
    if not lmdb_path:
        sys.stderr.write("ingest_to_lmdb: UW_LMDB_PATH is required\n")
        return 2

    map_size = int_env("UW_LMDB_MAP_SIZE", 1 << 30)
    db_name = os.environ.get("UW_LMDB_DBNAME")
    dupsort = os.environ.get("UW_LMDB_DUPSORT", "0") == "1"
    filter_col = opt_int_env("UW_FILTER_COL")
    filter_val = os.environ.get("UW_FILTER_VAL")
    key_col = int_env("UW_KEY_COL", 0)
    val_col = int_env("UW_VAL_COL", 1)
    key_enc = os.environ.get("UW_KEY_ENCODING", "utf8")
    val_enc = os.environ.get("UW_VAL_ENCODING", "utf8")
    batch_size = int_env("UW_BATCH_SIZE", 10_000)

    # Text-keyed intern mode (optional, default off).
    intern_key = os.environ.get("UW_INTERN_KEY", "0") == "1"
    intern_val = os.environ.get("UW_INTERN_VAL", "0") == "1"
    s2i_db_name = os.environ.get("UW_LMDB_S2I_DB")
    i2s_db_name = os.environ.get("UW_LMDB_I2S_DB")
    meta_db_name = os.environ.get("UW_LMDB_META_DB")
    use_append = os.environ.get("UW_LMDB_APPEND", "0") == "1"
    compile_time_atoms_path = os.environ.get("UW_COMPILE_TIME_ATOMS")
    schema_version = os.environ.get("UW_SCHEMA_VERSION", "1")
    source_sha = os.environ.get("UW_SOURCE_SHA")
    force_reingest = os.environ.get("UW_FORCE_REINGEST", "0") == "1"

    intern_mode = intern_key or intern_val
    if intern_mode:
        if not (s2i_db_name and i2s_db_name and meta_db_name):
            sys.stderr.write(
                "ingest_to_lmdb: UW_INTERN_KEY/UW_INTERN_VAL require "
                "UW_LMDB_S2I_DB, UW_LMDB_I2S_DB, and UW_LMDB_META_DB\n"
            )
            return 2

    # dupsort requires a named sub-DB; create one implicitly if needed.
    if dupsort and not db_name:
        db_name = "main"
    # Intern mode pushes edges into a named sub-db too (so meta/s2i/i2s
    # don't share the unnamed default).
    if intern_mode and not db_name:
        db_name = "category_parent"

    if filter_col is not None and filter_val is None:
        sys.stderr.write(
            "ingest_to_lmdb: UW_FILTER_COL set but UW_FILTER_VAL is not\n"
        )
        return 2

    # Reserve enough sub-db slots: edges + s2i + i2s + meta (4) + headroom.
    max_dbs = 8 if intern_mode else (4 if db_name else 0)

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        readonly=False,
        max_dbs=max_dbs,
    )

    db = env.open_db(db_name.encode(), dupsort=dupsort) if db_name else None
    s2i_db = env.open_db(s2i_db_name.encode()) if intern_mode else None
    i2s_db = env.open_db(i2s_db_name.encode()) if intern_mode else None
    meta_db = env.open_db(meta_db_name.encode()) if intern_mode else None

    # Idempotence guard: if meta has schema_version already, refuse to
    # overwrite without UW_FORCE_REINGEST.
    if intern_mode and meta_db is not None:
        with env.begin(db=meta_db) as ro:
            existing = ro.get(b"schema_version")
        if existing is not None and not force_reingest:
            sys.stderr.write(
                f"ingest_to_lmdb: LMDB at {lmdb_path!r} already has "
                f"meta.schema_version={existing!r}; pass UW_FORCE_REINGEST=1 "
                f"to overwrite\n"
            )
            env.close()
            return 3

    # Initialise intern table from the compile-time atoms sidecar.
    intern_map: Dict[str, int] = {}
    pending_i2s: List[Tuple[int, str]] = []
    next_id = 0
    compile_time_count = 0
    if compile_time_atoms_path:
        for atom in load_compile_time_atoms(compile_time_atoms_path):
            intern_map[atom] = next_id
            pending_i2s.append((next_id, atom))
            next_id += 1
        compile_time_count = next_id

    def intern_str(s: str) -> int:
        nonlocal next_id
        existing = intern_map.get(s)
        if existing is not None:
            return existing
        new_id = next_id
        intern_map[s] = new_id
        pending_i2s.append((new_id, s))
        next_id += 1
        return new_id

    total_in = 0
    total_written = 0
    skipped_bad = 0

    txn = env.begin(write=True)
    try:
        # Drain compile-time pre-population into i2s (monotonic, so append=True).
        for aid, atom in pending_i2s:
            txn.put(
                encode_int32_le(aid),
                atom.encode("utf-8"),
                db=i2s_db,
                append=True,
            )
        pending_i2s.clear()

        for line in sys.stdin:
            total_in += 1
            line = line.rstrip("\n")
            cols = line.split("\t")

            if filter_col is not None:
                if filter_col >= len(cols):
                    skipped_bad += 1
                    continue
                if tsv_unescape(cols[filter_col]) != filter_val:
                    continue

            if key_col >= len(cols) or val_col >= len(cols):
                skipped_bad += 1
                continue

            try:
                if intern_key:
                    key_str = tsv_unescape(cols[key_col])
                    before = next_id
                    kid = intern_str(key_str)
                    if next_id > before:
                        # New atom — write to i2s (monotonic, append=True).
                        txn.put(
                            encode_int32_le(kid),
                            key_str.encode("utf-8"),
                            db=i2s_db,
                            append=True,
                        )
                    key = encode_int32_le(kid)
                else:
                    key = encode(cols[key_col], key_enc)

                if intern_val:
                    val_str = tsv_unescape(cols[val_col])
                    before = next_id
                    vid = intern_str(val_str)
                    if next_id > before:
                        txn.put(
                            encode_int32_le(vid),
                            val_str.encode("utf-8"),
                            db=i2s_db,
                            append=True,
                        )
                    value = encode_int32_le(vid)
                else:
                    value = encode(cols[val_col], val_enc)
            except (ValueError, struct.error):
                skipped_bad += 1
                continue

            try:
                txn.put(key, value, db=db, append=use_append, dupdata=True)
            except lmdb.KeyExistsError:
                # append=True is strict about ordering; fall back to a
                # non-append put for this row only.  This indicates the
                # input is not perfectly sorted, so the caller should
                # consider clearing UW_LMDB_APPEND.
                txn.put(key, value, db=db, append=False, dupdata=True)
            total_written += 1

            if total_written % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)

        # Bulk-load s2i: sort by string then write with append=True.
        if intern_mode:
            for s, aid in sorted(intern_map.items()):
                txn.put(
                    s.encode("utf-8"),
                    encode_int32_le(aid),
                    db=s2i_db,
                    append=True,
                )

            # Meta keys: schema_version, next_id, compile_time_atoms_count,
            # source_dump_sha256, build_timestamp, cli_args.
            meta_kvs = {
                b"schema_version": schema_version.encode("utf-8"),
                b"next_id": encode_int32_le(next_id),
                b"compile_time_atoms_count": encode_int32_le(
                    compile_time_count
                ),
                b"build_timestamp": (
                    datetime.datetime.now(datetime.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .encode("utf-8")
                ),
                b"cli_args": " ".join(sys.argv).encode("utf-8"),
            }
            if source_sha:
                meta_kvs[b"source_dump_sha256"] = source_sha.encode("utf-8")
            for k, v in meta_kvs.items():
                txn.put(k, v, db=meta_db)

        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()

    summary = (
        f"ingest_to_lmdb: in={total_in} written={total_written} "
        f"skipped={skipped_bad}"
    )
    if intern_mode:
        summary += (
            f" interned={len(intern_map)} "
            f"compile_time_atoms={compile_time_count}"
        )
    sys.stderr.write(summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
