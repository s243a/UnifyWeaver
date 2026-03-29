#!/usr/bin/env python3
"""
Generate materialized facts from the Simple Wikipedia SQLite database.

Uses the pre-parsed simplewiki_categories.db (from parse_simplewiki_dump.py)
to extract article→category and category→parent relationships, then outputs
Prolog facts and TSV files. No API crawling needed.

Data provenance:
    Source: Simple English Wikipedia dumps from dumps.wikimedia.org/simplewiki/latest/
    Files:  simplewiki-latest-categorylinks.sql.gz (~27 MB)
            simplewiki-latest-page.sql.gz (~32 MB)
            simplewiki-latest-linktarget.sql.gz (~36 MB)
    Parsed into: data/simplewiki/simplewiki_categories.db
    Parser: examples/benchmark/parse_simplewiki_dump.py

Usage:
    # Dev dataset (20 articles from physics JSONL)
    python examples/benchmark/generate_facts_from_db.py \
        --articles /path/to/wikipedia_physics_articles.jsonl \
        --output data/benchmark/dev/ \
        --max-articles 20

    # Larger dataset (all science descendants)
    python examples/benchmark/generate_facts_from_db.py \
        --output data/benchmark/bench/ \
        --root Science \
        --max-articles 50000
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DB_PATH = Path("data/simplewiki/simplewiki_categories.db")


# =============================================================================
# Database queries
# =============================================================================

def get_article_categories(conn: sqlite3.Connection, title: str) -> List[str]:
    """Get categories for an article by title."""
    cursor = conn.execute("""
        SELECT cl.cl_to
        FROM categorylinks cl
        JOIN page p ON cl.cl_from = p.page_id
        WHERE p.page_title = ? AND p.page_namespace = 0 AND cl.cl_type = 'page'
    """, (title.replace(" ", "_"),))
    return [row[0] for row in cursor]


def get_parent_categories(conn: sqlite3.Connection, category: str) -> List[str]:
    """Get parent categories of a category."""
    cursor = conn.execute("""
        SELECT cl.cl_to
        FROM categorylinks cl
        JOIN page p ON cl.cl_from = p.page_id
        WHERE p.page_title = ? AND p.page_namespace = 14 AND cl.cl_type = 'subcat'
    """, (category,))
    return [row[0] for row in cursor]


def load_subcat_graph(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """Load entire subcategory graph into memory. Returns {parent: [child, ...]}."""
    print("  Loading subcategory graph into memory...")
    cursor = conn.execute("""
        SELECT p.page_title AS child, cl.cl_to AS parent
        FROM categorylinks cl
        JOIN page p ON cl.cl_from = p.page_id
        WHERE cl.cl_type = 'subcat' AND p.page_namespace = 14
    """)
    graph = defaultdict(list)
    n = 0
    for child, parent in cursor:
        graph[parent].append(child)
        n += 1
    print(f"  Loaded {n} subcategory edges ({len(graph)} parent categories)")
    return dict(graph)


def get_descendant_categories(
    conn: sqlite3.Connection, root: str, max_depth: int = 20
) -> Dict[str, int]:
    """BFS to find all descendant categories of root. Returns {category: depth}."""
    subcat_graph = load_subcat_graph(conn)

    visited = {root: 0}
    queue = [root]
    depth = 0

    while queue and depth < max_depth:
        next_queue = []
        for cat in queue:
            for child in subcat_graph.get(cat, []):
                if child not in visited:
                    visited[child] = depth + 1
                    next_queue.append(child)
        queue = next_queue
        depth += 1
        if queue:
            print(f"  Depth {depth}: {len(visited)} categories, {len(queue)} in queue")

    return visited


def get_articles_in_categories(
    conn: sqlite3.Connection,
    categories: Set[str],
    max_articles: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Get articles that belong to any of the given categories.
    Returns {article_title: [category, ...]}.
    """
    result = defaultdict(list)
    total = 0

    for cat in categories:
        cursor = conn.execute("""
            SELECT p.page_title
            FROM categorylinks cl
            JOIN page p ON cl.cl_from = p.page_id
            WHERE cl.cl_to = ? AND cl.cl_type = 'page' AND p.page_namespace = 0
        """, (cat,))
        for (title,) in cursor:
            result[title].append(cat)
            if max_articles and len(result) >= max_articles:
                break
        if max_articles and len(result) >= max_articles:
            break

    return dict(result)


def build_category_hierarchy_bulk(
    conn: sqlite3.Connection, categories: Set[str]
) -> Dict[str, List[str]]:
    """Build child→[parent] relationships for a set of categories.
    Uses a single bulk query instead of per-category lookups.
    """
    # Load ALL subcat edges into memory (fast: ~0.3s for all of simplewiki)
    cursor = conn.execute("""
        SELECT p.page_title AS child, cl.cl_to AS parent
        FROM categorylinks cl
        JOIN page p ON cl.cl_from = p.page_id
        WHERE cl.cl_type = 'subcat' AND p.page_namespace = 14
    """)
    hierarchy = defaultdict(list)
    for child, parent in cursor:
        if child in categories:
            hierarchy[child].append(parent)
    return dict(hierarchy)


# =============================================================================
# Filtering
# =============================================================================

SKIP_PREFIXES = (
    "Articles_", "Pages_", "All_", "CS1_", "Webarchive_",
    "Wikipedia_", "Commons_", "Wikidata_", "Use_",
    "Short_description", "Good_articles", "Featured_articles",
)
SKIP_SUFFIXES = (
    "_stubs", "_stub", "_templates", "_navigational_templates",
    "_navigational_boxes", "_disambiguation", "_redirects",
)


def is_maintenance_category(cat: str) -> bool:
    return (
        any(cat.startswith(p) for p in SKIP_PREFIXES)
        or any(cat.endswith(s) for s in SKIP_SUFFIXES)
    )


# =============================================================================
# Output
# =============================================================================

def prolog_atom(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def write_outputs(
    output_dir: Path,
    article_cats: Dict[str, List[str]],
    hierarchy: Dict[str, List[str]],
    root_categories: Set[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # TSV: article_category
    with open(output_dir / "article_category.tsv", "w") as f:
        f.write("article\tcategory\n")
        for art, cats in sorted(article_cats.items()):
            for cat in cats:
                f.write(f"{art}\t{cat}\n")
    n_ac = sum(len(v) for v in article_cats.values())
    print(f"  article_category.tsv: {n_ac} rows")

    # TSV: category_parent
    with open(output_dir / "category_parent.tsv", "w") as f:
        f.write("child\tparent\n")
        for child, parents in sorted(hierarchy.items()):
            for parent in parents:
                f.write(f"{child}\t{parent}\n")
    n_cp = sum(len(v) for v in hierarchy.values())
    print(f"  category_parent.tsv: {n_cp} rows")

    # TSV: root_categories
    with open(output_dir / "root_categories.tsv", "w") as f:
        f.write("category\n")
        for r in sorted(root_categories):
            f.write(f"{r}\n")
    print(f"  root_categories.tsv: {len(root_categories)} rows")

    # Prolog facts
    with open(output_dir / "facts.pl", "w") as f:
        f.write("%% Auto-generated benchmark facts\n")
        f.write("%% Source: Simple English Wikipedia (simplewiki dumps)\n")
        f.write(f"%% Database: data/simplewiki/simplewiki_categories.db\n\n")

        f.write("%% article_category(ArticleTitle, CategoryName).\n")
        for art, cats in sorted(article_cats.items()):
            for cat in cats:
                f.write(f"article_category({prolog_atom(art)}, {prolog_atom(cat)}).\n")

        f.write("\n%% category_parent(ChildCategory, ParentCategory).\n")
        for child, parents in sorted(hierarchy.items()):
            for parent in parents:
                f.write(f"category_parent({prolog_atom(child)}, {prolog_atom(parent)}).\n")

        f.write("\n%% root_category(CategoryName).\n")
        for r in sorted(root_categories):
            f.write(f"root_category({prolog_atom(r)}).\n")

    total = n_ac + n_cp + len(root_categories)
    print(f"  facts.pl: {total} facts")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark facts from Simple Wikipedia SQLite DB"
    )
    parser.add_argument("--articles", help="Path to articles JSONL (optional; if omitted, discovers articles from category descendants)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--root", default="Science", help="Root category (default: Science)")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles")
    parser.add_argument("--max-depth", type=int, default=20, help="Max category depth")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to simplewiki SQLite DB")

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}")
        print("Run: python examples/benchmark/parse_simplewiki_dump.py")
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))

    # 1. Find descendant categories of root
    print(f"Step 1: Finding category descendants of '{args.root}'...")
    desc_cats = get_descendant_categories(conn, args.root, args.max_depth)
    content_cats = {c for c in desc_cats if not is_maintenance_category(c)}
    print(f"  {len(content_cats)} content categories (filtered from {len(desc_cats)})")

    # 2. Get articles
    if args.articles:
        print(f"Step 2: Loading article titles from {args.articles}...")
        titles = []
        with open(args.articles) as f:
            for line in f:
                rec = json.loads(line)
                titles.append(rec["title"].replace(" ", "_"))
                if args.max_articles and len(titles) >= args.max_articles:
                    break
        print(f"  Loaded {len(titles)} titles")

        article_cats = {}
        for t in titles:
            cats = get_article_categories(conn, t)
            cats = [c for c in cats if c in content_cats]
            if cats:
                article_cats[t] = cats
        print(f"  {len(article_cats)} articles have categories under '{args.root}'")
    else:
        print(f"Step 2: Discovering articles in descendant categories...")
        article_cats = get_articles_in_categories(conn, content_cats, args.max_articles)
        print(f"  Found {len(article_cats)} articles")

    # 3. Build category hierarchy — only categories on paths from articles to root
    print("Step 3: Building relevant category hierarchy...")

    # Get full hierarchy for filtering
    full_hierarchy = build_category_hierarchy_bulk(conn, content_cats)
    full_hierarchy = {
        child: [p for p in parents if not is_maintenance_category(p)]
        for child, parents in full_hierarchy.items()
        if not is_maintenance_category(child)
    }

    # Walk UP from article categories to find which categories are on paths to root
    article_direct_cats = set()
    for cats in article_cats.values():
        article_direct_cats.update(cats)

    relevant_cats = set(article_direct_cats)
    frontier = list(article_direct_cats)
    visited_walk = set()
    while frontier:
        cat = frontier.pop()
        if cat in visited_walk:
            continue
        visited_walk.add(cat)
        relevant_cats.add(cat)
        for parent in full_hierarchy.get(cat, []):
            if parent not in visited_walk:
                frontier.append(parent)

    # Keep only edges where both child and parent are relevant
    hierarchy = {
        child: [p for p in parents if p in relevant_cats or p == args.root]
        for child, parents in full_hierarchy.items()
        if child in relevant_cats
    }
    hierarchy = {k: v for k, v in hierarchy.items() if v}

    print(f"  {len(relevant_cats)} relevant categories (trimmed from {len(content_cats)})")
    print(f"  {sum(len(v) for v in hierarchy.values())} hierarchy edges")

    # 4. Write outputs
    print("Step 4: Writing outputs...")
    write_outputs(Path(args.output), article_cats, hierarchy, {args.root})

    # 5. Save metadata
    meta = {
        "source": "Simple English Wikipedia dumps (simplewiki)",
        "database": str(args.db),
        "root": args.root,
        "n_articles": len(article_cats),
        "n_content_categories": len(content_cats),
        "n_hierarchy_edges": sum(len(v) for v in hierarchy.values()),
        "max_depth": args.max_depth,
    }
    meta_path = Path(args.output) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSummary:")
    print(f"  Root: {args.root}")
    print(f"  Articles: {len(article_cats)}")
    print(f"  Content categories: {len(content_cats)}")
    print(f"  Hierarchy edges: {sum(len(v) for v in hierarchy.values())}")

    conn.close()


if __name__ == "__main__":
    main()
