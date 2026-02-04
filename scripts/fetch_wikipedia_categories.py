#!/usr/bin/env python3
"""
Fetch and Parse Wikipedia Category Links

Downloads Wikipedia's categorylinks SQL dump and parses it into SQLite
for fast lookups. Used to bridge Wikipedia articles to Pearltrees hierarchy.

Usage:
    # Download and parse (first time)
    python scripts/fetch_wikipedia_categories.py --download --parse

    # Just parse (if already downloaded)
    python scripts/fetch_wikipedia_categories.py --parse

    # Query categories for an article
    python scripts/fetch_wikipedia_categories.py --query "David Lee (physicist)"

    # Find connection to Pearltrees folder
    python scripts/fetch_wikipedia_categories.py --connect "David Lee (physicist)" --pearltrees reports/pearltrees_targets_full_multi_account.jsonl

Data Source:
    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz
    ~2.4 GB compressed, ~10-15 GB uncompressed
"""

import argparse
import gzip
import re
import sqlite3
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json


# =============================================================================
# Configuration
# =============================================================================

DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz"
DEFAULT_DOWNLOAD_PATH = Path("data/enwiki-categorylinks.sql.gz")
DEFAULT_DB_PATH = Path("data/wikipedia_categories.db")


# =============================================================================
# Download
# =============================================================================

def download_categorylinks(
    url: str = DUMP_URL,
    output_path: Path = DEFAULT_DOWNLOAD_PATH,
    chunk_size: int = 8192 * 1024  # 8MB chunks
) -> Path:
    """
    Download Wikipedia categorylinks SQL dump.

    Args:
        url: URL to download from
        output_path: Where to save the file
        chunk_size: Download chunk size in bytes

    Returns:
        Path to downloaded file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"File already exists: {output_path}")
        print(f"Size: {output_path.stat().st_size / 1e9:.2f} GB")
        return output_path

    print(f"Downloading {url}")
    print(f"This is ~2.4 GB, may take a while...")

    request = urllib.request.Request(url, headers={'User-Agent': 'UnifyWeaver/1.0'})

    with urllib.request.urlopen(request) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\rDownloaded: {downloaded / 1e9:.2f} GB ({pct:.1f}%)", end='', flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / 1e9:.2f} GB", end='', flush=True)

    print(f"\nSaved to {output_path}")
    return output_path


# =============================================================================
# Parse MySQL Dump to SQLite
# =============================================================================

def create_database(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Main table: page_id -> category mappings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS categorylinks (
            cl_from INTEGER,
            cl_to TEXT,
            cl_type TEXT,
            PRIMARY KEY (cl_from, cl_to)
        )
    """)

    # Page titles table (populated separately if needed)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS page (
            page_id INTEGER PRIMARY KEY,
            page_title TEXT,
            page_namespace INTEGER
        )
    """)

    # Create indexes for fast lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cl_to ON categorylinks(cl_to)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_page_title ON page(page_title)")

    conn.commit()
    return conn


def parse_insert_values(line: str) -> List[Tuple]:
    """
    Parse VALUES from a MySQL INSERT statement.

    The categorylinks INSERT format is:
    INSERT INTO `categorylinks` VALUES (page_id,'category_name','sortkey','timestamp','sortkey_prefix','collation','type')

    We extract: (cl_from, cl_to, cl_type)
    """
    # Find VALUES section
    values_match = re.search(r'VALUES\s*(.+);?\s*$', line, re.IGNORECASE)
    if not values_match:
        return []

    values_str = values_match.group(1)
    results = []

    # Parse each tuple: (val1, val2, ...)
    # This is tricky because values can contain commas and quotes
    tuple_pattern = re.compile(r"\(([^)]+)\)")

    for match in tuple_pattern.finditer(values_str):
        tuple_str = match.group(1)

        # Split by comma, but respect quoted strings
        parts = []
        current = ""
        in_quote = False
        escape_next = False

        for char in tuple_str:
            if escape_next:
                current += char
                escape_next = False
            elif char == '\\':
                escape_next = True
                current += char
            elif char == "'" and not in_quote:
                in_quote = True
            elif char == "'" and in_quote:
                in_quote = False
            elif char == ',' and not in_quote:
                parts.append(current.strip().strip("'"))
                current = ""
            else:
                current += char

        if current:
            parts.append(current.strip().strip("'"))

        if len(parts) >= 7:
            try:
                cl_from = int(parts[0])
                # Format: (page_id, sortkey, timestamp, category_name, type, ...)
                # parts[1] is sortkey (binary), parts[3] is the actual category name
                cl_to = parts[3]  # Category name
                cl_type = parts[4] if len(parts) > 4 else 'page'
                results.append((cl_from, cl_to, cl_type))
            except (ValueError, IndexError):
                continue

    return results


def parse_categorylinks_dump(
    dump_path: Path,
    db_path: Path,
    batch_size: int = 10000,
    max_rows: Optional[int] = None
) -> int:
    """
    Parse MySQL dump and insert into SQLite.

    Args:
        dump_path: Path to .sql.gz file
        db_path: Path to SQLite database
        batch_size: Rows to insert per batch
        max_rows: Maximum rows to parse (None for all)

    Returns:
        Number of rows inserted
    """
    conn = create_database(db_path)
    cursor = conn.cursor()

    total_rows = 0
    batch = []

    print(f"Parsing {dump_path}...")

    # Open gzipped file
    open_func = gzip.open if str(dump_path).endswith('.gz') else open

    with open_func(dump_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f):
            # Skip non-INSERT lines
            if not line.startswith('INSERT INTO'):
                continue

            # Parse values from this INSERT statement
            values = parse_insert_values(line)
            batch.extend(values)

            # Insert batch
            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO categorylinks (cl_from, cl_to, cl_type) VALUES (?, ?, ?)",
                    batch
                )
                conn.commit()
                total_rows += len(batch)
                print(f"\rParsed {total_rows:,} rows...", end='', flush=True)
                batch = []

                if max_rows and total_rows >= max_rows:
                    break

    # Insert remaining
    if batch:
        cursor.executemany(
            "INSERT OR IGNORE INTO categorylinks (cl_from, cl_to, cl_type) VALUES (?, ?, ?)",
            batch
        )
        conn.commit()
        total_rows += len(batch)

    print(f"\nInserted {total_rows:,} total rows")

    # Print stats
    cursor.execute("SELECT COUNT(DISTINCT cl_from) FROM categorylinks")
    n_pages = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT cl_to) FROM categorylinks")
    n_categories = cursor.fetchone()[0]

    print(f"Unique pages: {n_pages:,}")
    print(f"Unique categories: {n_categories:,}")

    conn.close()
    return total_rows


# =============================================================================
# Category Lookup
# =============================================================================

class WikipediaCategoryLookup:
    """
    Fast category lookup using SQLite.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

        # Cache for category hierarchy
        self._parent_cache: Dict[str, List[str]] = {}

    def get_categories(self, page_id: int) -> List[str]:
        """Get categories for a page by ID."""
        cursor = self.conn.execute(
            "SELECT cl_to FROM categorylinks WHERE cl_from = ?",
            (page_id,)
        )
        return [row['cl_to'] for row in cursor]

    def get_categories_by_title(self, title: str) -> List[str]:
        """Get categories for a page by title."""
        # First get page_id from title
        cursor = self.conn.execute(
            "SELECT page_id FROM page WHERE page_title = ?",
            (title.replace(' ', '_'),)
        )
        row = cursor.fetchone()
        if row:
            return self.get_categories(row['page_id'])
        return []

    def get_parent_categories(self, category: str) -> List[str]:
        """
        Get parent categories of a category.

        Categories are also pages, so we look up their categories.
        """
        if category in self._parent_cache:
            return self._parent_cache[category]

        # Category pages have a specific page_id
        # We need to look up the category as a page
        cursor = self.conn.execute(
            """SELECT cl_to FROM categorylinks
               WHERE cl_from = (SELECT page_id FROM page WHERE page_title = ? AND page_namespace = 14)""",
            (category,)
        )
        parents = [row['cl_to'] for row in cursor]
        self._parent_cache[category] = parents
        return parents

    def walk_hierarchy(
        self,
        categories: List[str],
        max_depth: int = 10
    ) -> Dict[str, int]:
        """
        Walk up the category hierarchy from starting categories.

        Returns: {category: depth} for all reachable categories
        """
        visited = {}
        queue = [(cat, 0) for cat in categories]

        while queue:
            category, depth = queue.pop(0)

            if category in visited:
                continue
            if depth > max_depth:
                continue

            visited[category] = depth

            # Get parent categories
            parents = self.get_parent_categories(category)
            for parent in parents:
                if parent not in visited:
                    queue.append((parent, depth + 1))

        return visited

    def find_connection_point(
        self,
        article_categories: List[str],
        target_folders: Set[str],
        max_depth: int = 10
    ) -> Tuple[Optional[str], int]:
        """
        Find closest matching folder by walking up category hierarchy.

        Args:
            article_categories: Categories of the Wikipedia article
            target_folders: Set of Pearltrees folder names to match
            max_depth: Maximum depth to search

        Returns:
            (matching_folder, hops) or (None, inf) if no match
        """
        hierarchy = self.walk_hierarchy(article_categories, max_depth)

        best_match = None
        best_depth = float('inf')

        for category, depth in hierarchy.items():
            # Normalize category name for matching
            normalized = category.replace('_', ' ').lower()

            for folder in target_folders:
                folder_lower = folder.lower()
                if normalized == folder_lower or normalized.endswith(folder_lower):
                    if depth < best_depth:
                        best_match = folder
                        best_depth = depth

        return best_match, best_depth


# =============================================================================
# Pearltrees Integration
# =============================================================================

def load_pearltrees_folders(jsonl_path: str) -> Set[str]:
    """Load folder names from Pearltrees JSONL."""
    folders = set()

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('type') == 'Tree':
                title = rec.get('raw_title', rec.get('query', ''))
                if title:
                    folders.add(title)

                # Also extract folder names from target_text hierarchy
                target = rec.get('target_text', '')
                for part in target.split('\n'):
                    part = part.strip().lstrip('- ')
                    if part and not part.startswith('/'):
                        folders.add(part)

    print(f"Loaded {len(folders)} Pearltrees folders")
    return folders


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Wikipedia Category Links Tools')
    parser.add_argument('--download', action='store_true', help='Download categorylinks dump')
    parser.add_argument('--parse', action='store_true', help='Parse dump into SQLite')
    parser.add_argument('--query', type=str, help='Query categories for article title')
    parser.add_argument('--connect', type=str, help='Find connection to Pearltrees for article')
    parser.add_argument('--pearltrees', type=str, help='Path to Pearltrees JSONL')
    parser.add_argument('--dump-path', type=Path, default=DEFAULT_DOWNLOAD_PATH)
    parser.add_argument('--db-path', type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument('--max-rows', type=int, default=None, help='Max rows to parse (for testing)')

    args = parser.parse_args()

    if args.download:
        download_categorylinks(output_path=args.dump_path)

    if args.parse:
        if not args.dump_path.exists():
            print(f"Dump file not found: {args.dump_path}")
            print("Run with --download first")
            return
        parse_categorylinks_dump(args.dump_path, args.db_path, max_rows=args.max_rows)

    if args.query:
        if not args.db_path.exists():
            print(f"Database not found: {args.db_path}")
            print("Run with --parse first")
            return

        lookup = WikipediaCategoryLookup(args.db_path)
        categories = lookup.get_categories_by_title(args.query)

        if categories:
            print(f"Categories for '{args.query}':")
            for cat in categories:
                print(f"  - {cat}")
        else:
            print(f"No categories found for '{args.query}'")
            print("(Page might not be in database, or title format differs)")

    if args.connect:
        if not args.db_path.exists():
            print(f"Database not found: {args.db_path}")
            return
        if not args.pearltrees:
            print("--pearltrees path required for --connect")
            return

        lookup = WikipediaCategoryLookup(args.db_path)
        folders = load_pearltrees_folders(args.pearltrees)

        categories = lookup.get_categories_by_title(args.connect)
        if not categories:
            print(f"No categories found for '{args.connect}'")
            return

        print(f"Article categories: {categories[:5]}...")

        match, depth = lookup.find_connection_point(categories, folders)
        if match:
            print(f"Connection found: '{match}' at depth {depth}")
        else:
            print("No connection to Pearltrees folders found")


if __name__ == '__main__':
    main()
