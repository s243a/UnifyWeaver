#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Export resolved category edges for distribution-cache benchmarks."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data/simplewiki/simplewiki_categories.db")


def category_node(title: str) -> str:
    return title if title.startswith("Category:") else f"Category:{title}"


def iter_category_parent_edges(conn: sqlite3.Connection):
    cursor = conn.execute(
        """
        SELECT p.page_title AS child, cl.cl_to AS parent
        FROM categorylinks cl
        JOIN page p ON cl.cl_from = p.page_id
        WHERE cl.cl_type = 'subcat'
          AND p.page_namespace = 14
          AND cl.cl_to IS NOT NULL
          AND cl.cl_to != ''
        ORDER BY child, parent
        """
    )
    for child, parent in cursor:
        yield category_node(child), category_node(parent)


def write_edges(conn: sqlite3.Connection, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("child\tparent\n")
        for child, parent in iter_category_parent_edges(conn):
            handle.write(f"{child}\t{parent}\n")
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Resolved SimpleWiki category SQLite DB.")
    parser.add_argument("--output", required=True, type=Path, help="Output TSV with child<TAB>parent category edges.")
    args = parser.parse_args(argv)

    if not args.db.exists():
        raise SystemExit(
            f"database not found: {args.db}\n"
            "Build it with examples/benchmark/parse_simplewiki_dump.py or pass --db."
        )

    conn = sqlite3.connect(str(args.db))
    try:
        count = write_edges(conn, args.output)
    finally:
        conn.close()

    print(f"db={args.db}")
    print(f"edges_written={count}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
