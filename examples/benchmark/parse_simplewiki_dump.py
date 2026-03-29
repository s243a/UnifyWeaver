#!/usr/bin/env python3
"""
Parse Simple English Wikipedia SQL dumps into SQLite.

Parses simplewiki-latest-categorylinks.sql.gz and simplewiki-latest-page.sql.gz
into a local SQLite database for fast category hierarchy lookups.

Usage:
    python examples/benchmark/parse_simplewiki_dump.py

Expects dumps in data/simplewiki/:
    simplewiki-latest-categorylinks.sql.gz
    simplewiki-latest-page.sql.gz
"""

import gzip
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

DUMP_DIR = Path("data/simplewiki")
DB_PATH = Path("data/simplewiki/simplewiki_categories.db")


def create_database(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE page (
            page_id INTEGER PRIMARY KEY,
            page_title TEXT,
            page_namespace INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE categorylinks_raw (
            cl_from INTEGER,
            cl_target_id INTEGER,
            cl_type TEXT
        )
    """)

    conn.commit()
    return conn


def parse_mysql_values(line: str, extractor) -> List[Tuple]:
    """Parse VALUES from a MySQL INSERT statement."""
    values_match = re.search(r"VALUES\s*(.+);?\s*$", line, re.IGNORECASE)
    if not values_match:
        return []

    values_str = values_match.group(1)
    results = []

    # State machine parser for MySQL tuples
    i = 0
    while i < len(values_str):
        if values_str[i] == "(":
            # Find matching closing paren, respecting quotes
            j = i + 1
            in_quote = False
            escape_next = False
            while j < len(values_str):
                if escape_next:
                    escape_next = False
                    j += 1
                    continue
                c = values_str[j]
                if c == "\\":
                    escape_next = True
                elif c == "'" and not in_quote:
                    in_quote = True
                elif c == "'" and in_quote:
                    in_quote = False
                elif c == ")" and not in_quote:
                    break
                j += 1

            tuple_str = values_str[i + 1 : j]
            row = extractor(tuple_str)
            if row:
                results.append(row)
            i = j + 1
        else:
            i += 1

    return results


def split_mysql_fields(tuple_str: str) -> List[str]:
    """Split a MySQL tuple string into fields."""
    parts = []
    current = ""
    in_quote = False
    escape_next = False

    for char in tuple_str:
        if escape_next:
            current += char
            escape_next = False
        elif char == "\\":
            escape_next = True
        elif char == "'" and not in_quote:
            in_quote = True
        elif char == "'" and in_quote:
            in_quote = False
        elif char == "," and not in_quote:
            parts.append(current.strip().strip("'"))
            current = ""
        else:
            current += char

    if current:
        parts.append(current.strip().strip("'"))
    return parts


def extract_page(tuple_str: str):
    """Extract (page_id, page_title, page_namespace) from page table row."""
    parts = split_mysql_fields(tuple_str)
    if len(parts) >= 3:
        try:
            page_id = int(parts[0])
            page_namespace = int(parts[1])
            page_title = parts[2]
            # Only keep articles (ns=0) and categories (ns=14)
            if page_namespace in (0, 14):
                return (page_id, page_title, page_namespace)
        except (ValueError, IndexError):
            pass
    return None


def extract_categorylink(tuple_str: str):
    """Extract (cl_from, cl_target_id, cl_type) from categorylinks row.

    New MediaWiki schema (post-2024):
    (cl_from, cl_sortkey, cl_timestamp, cl_sortkey_prefix,
     cl_type, cl_collation_id, cl_target_id)

    cl_target_id is the page_id of the parent category (in page table, ns=14).
    We store cl_target_id and resolve to category name via JOIN later.
    """
    parts = split_mysql_fields(tuple_str)
    if len(parts) >= 7:
        try:
            cl_from = int(parts[0])
            cl_type = parts[4]       # 'page', 'subcat', or 'file'
            cl_target_id = int(parts[6])
            return (cl_from, cl_target_id, cl_type)
        except (ValueError, IndexError):
            pass
    return None


def parse_dump(dump_path: Path, conn: sqlite3.Connection, table: str, extractor, insert_sql: str):
    """Parse a gzipped MySQL dump into SQLite."""
    print(f"Parsing {dump_path.name}...")

    batch = []
    total = 0
    batch_size = 50000

    open_func = gzip.open if str(dump_path).endswith(".gz") else open

    with open_func(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue

            rows = parse_mysql_values(line, extractor)
            batch.extend(rows)

            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                conn.commit()
                total += len(batch)
                print(f"\r  {total:,} rows...", end="", flush=True)
                batch = []

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()
        total += len(batch)

    print(f"\r  {total:,} rows total")
    return total


def extract_linktarget(tuple_str: str):
    """Extract (lt_id, lt_namespace, lt_title) from linktarget row.

    Schema: (lt_id, lt_namespace, lt_title)
    """
    parts = split_mysql_fields(tuple_str)
    if len(parts) >= 3:
        try:
            lt_id = int(parts[0])
            lt_namespace = int(parts[1])
            lt_title = parts[2]
            # Only keep category namespace (14) targets
            if lt_namespace == 14:
                return (lt_id, lt_namespace, lt_title)
        except (ValueError, IndexError):
            pass
    return None


def main():
    catlinks_path = DUMP_DIR / "simplewiki-latest-categorylinks.sql.gz"
    page_path = DUMP_DIR / "simplewiki-latest-page.sql.gz"
    linktarget_path = DUMP_DIR / "simplewiki-latest-linktarget.sql.gz"

    for p in [catlinks_path, page_path, linktarget_path]:
        if not p.exists():
            print(f"Missing: {p}")
            print(f"Download from https://dumps.wikimedia.org/simplewiki/latest/")
            sys.exit(1)

    conn = create_database(DB_PATH)

    # Add linktarget table
    conn.execute("""
        CREATE TABLE linktarget (
            lt_id INTEGER PRIMARY KEY,
            lt_namespace INTEGER,
            lt_title TEXT
        )
    """)
    conn.commit()

    # Parse page table first (needed for article title resolution)
    parse_dump(
        page_path,
        conn,
        "page",
        extract_page,
        "INSERT OR IGNORE INTO page (page_id, page_title, page_namespace) VALUES (?, ?, ?)",
    )

    # Parse linktarget table (maps cl_target_id → category name)
    parse_dump(
        linktarget_path,
        conn,
        "linktarget",
        extract_linktarget,
        "INSERT OR IGNORE INTO linktarget (lt_id, lt_namespace, lt_title) VALUES (?, ?, ?)",
    )

    # Parse categorylinks (raw — uses target_id, not category name)
    parse_dump(
        catlinks_path,
        conn,
        "categorylinks_raw",
        extract_categorylink,
        "INSERT OR IGNORE INTO categorylinks_raw (cl_from, cl_target_id, cl_type) VALUES (?, ?, ?)",
    )

    # Create indexes on raw tables
    print("Creating indexes on raw tables...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_page_title ON page(page_title)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_page_ns ON page(page_namespace)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clr_from ON categorylinks_raw(cl_from)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clr_target ON categorylinks_raw(cl_target_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lt_id ON linktarget(lt_id)")
    conn.commit()

    # Resolve cl_target_id to category name via linktarget table
    # The new MediaWiki schema uses: categorylinks.cl_target_id → linktarget.lt_id
    # linktarget has (lt_id, lt_namespace=14, lt_title=category_name)
    print("Resolving category names (JOIN categorylinks_raw with linktarget)...")
    conn.execute("""
        CREATE TABLE categorylinks AS
        SELECT
            cl.cl_from,
            lt.lt_title AS cl_to,
            cl.cl_type
        FROM categorylinks_raw cl
        JOIN linktarget lt ON cl.cl_target_id = lt.lt_id
    """)
    conn.commit()

    # Create indexes on resolved table
    print("Creating indexes on resolved categorylinks...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cl_from ON categorylinks(cl_from)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cl_to ON categorylinks(cl_to)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cl_type ON categorylinks(cl_type)")
    conn.commit()

    # Print stats
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM page WHERE page_namespace = 0")
    n_articles = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM page WHERE page_namespace = 14")
    n_categories = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM categorylinks_raw")
    n_raw = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM categorylinks")
    n_resolved = cursor.fetchone()[0]
    cursor.execute("SELECT cl_type, COUNT(*) FROM categorylinks GROUP BY cl_type")
    type_counts = dict(cursor.fetchall())

    print(f"\nSimple Wikipedia stats:")
    print(f"  Articles (ns=0):    {n_articles:,}")
    print(f"  Categories (ns=14): {n_categories:,}")
    print(f"  Raw category links: {n_raw:,}")
    print(f"  Resolved links:     {n_resolved:,}")
    for t, c in type_counts.items():
        print(f"    {t}: {c:,}")

    conn.close()
    print(f"\nDatabase: {DB_PATH}")


if __name__ == "__main__":
    main()
