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

The filter/projection happens in Python because that's where the Prolog
layer places *logic*.  The parser (Rust) stays schema-agnostic.

Usage:
  UW_LMDB_PATH=./subcats.lmdb \\
  UW_FILTER_COL=4 UW_FILTER_VAL=subcat \\
  UW_KEY_COL=0 UW_VAL_COL=6 \\
  UW_KEY_ENCODING=int32_le UW_VAL_ENCODING=int32_le \\
    mysql_stream dump.sql.gz | python3 ingest_to_lmdb.py
"""

import os
import sys
import struct
from typing import Optional

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

    # dupsort requires a named sub-DB; create one implicitly if needed.
    if dupsort and not db_name:
        db_name = "main"

    if filter_col is not None and filter_val is None:
        sys.stderr.write(
            "ingest_to_lmdb: UW_FILTER_COL set but UW_FILTER_VAL is not\n"
        )
        return 2

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        readonly=False,
        max_dbs=4 if db_name else 0,
    )
    db = env.open_db(db_name.encode(), dupsort=dupsort) if db_name else None

    total_in = 0
    total_written = 0
    skipped_bad = 0

    txn = env.begin(db=db, write=True)
    try:
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
                key = encode(cols[key_col], key_enc)
                value = encode(cols[val_col], val_enc)
            except (ValueError, struct.error):
                skipped_bad += 1
                continue

            txn.put(key, value)
            total_written += 1

            if total_written % batch_size == 0:
                txn.commit()
                txn = env.begin(db=db, write=True)

        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()

    sys.stderr.write(
        f"ingest_to_lmdb: in={total_in} written={total_written} "
        f"skipped={skipped_bad}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
