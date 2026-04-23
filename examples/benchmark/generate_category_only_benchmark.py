#!/usr/bin/env python3
"""
Generate a category-only benchmark dataset from simplewiki.

Emits the full category_parent hierarchy (~97k categories, ~297k edges)
and synthesizes article seeds as (C, C) so the existing effective_distance
workload runs against category seeds directly — no Wikipedia articles
involved.

All true root categories (categories with no parent of their own, excluding
maintenance cats) are declared as root_category/1 so seeds can reach any
reachable top-level category.

Usage:
    python examples/benchmark/generate_category_only_benchmark.py \
        --output data/benchmark/100k_cats/ \
        --db context/gemini/UnifyWeaver/data/simplewiki/simplewiki_categories.db
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


SKIP_PREFIXES = (
    "Articles_", "Pages_", "All_", "CS1_", "Webarchive_",
    "Wikipedia_", "Commons_", "Wikidata_", "Use_", "Hidden",
    "Container_", "Navseasoncats", "Category_redirects",
    "Short_description", "Good_articles", "Featured_articles",
    "CatAutoTOC",
)
SKIP_SUFFIXES = (
    "_stubs", "_stub", "_templates", "_navigational_templates",
    "_navigational_boxes", "_disambiguation", "_redirects",
)


def is_maintenance(cat: str) -> bool:
    return (
        any(cat.startswith(p) for p in SKIP_PREFIXES)
        or any(cat.endswith(s) for s in SKIP_SUFFIXES)
    )


def prolog_atom(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--db", default="context/gemini/UnifyWeaver/data/simplewiki/simplewiki_categories.db")
    ap.add_argument("--max-seeds", type=int, default=None,
                    help="Cap number of seed categories (default: all content categories)")
    args = ap.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)

    print("Loading full subcat graph...")
    edges = []
    for child, parent in conn.execute("""
        SELECT p.page_title AS child, cl.cl_to AS parent
        FROM categorylinks cl
        JOIN page p ON cl.cl_from = p.page_id
        WHERE cl.cl_type = 'subcat' AND p.page_namespace = 14
    """):
        if is_maintenance(child) or is_maintenance(parent):
            continue
        edges.append((child, parent))

    children_of = defaultdict(list)
    parents_of = defaultdict(list)
    all_cats = set()
    for child, parent in edges:
        children_of[parent].append(child)
        parents_of[child].append(parent)
        all_cats.add(child)
        all_cats.add(parent)

    print(f"  {len(all_cats):,} content categories")
    print(f"  {len(edges):,} subcat edges after filtering")

    # Roots: appear as parent, have no parent themselves
    roots = sorted(
        c for c in children_of.keys()
        if c not in parents_of and not is_maintenance(c)
    )
    print(f"  {len(roots):,} root categories (top-level)")

    # Seeds: every content category is its own "article"
    seeds = sorted(all_cats)
    if args.max_seeds and len(seeds) > args.max_seeds:
        seeds = seeds[:args.max_seeds]
    print(f"  {len(seeds):,} seed categories")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "category_parent.tsv", "w") as f:
        f.write("child\tparent\n")
        for c, p in sorted(edges):
            f.write(f"{c}\t{p}\n")

    with open(out / "article_category.tsv", "w") as f:
        f.write("article\tcategory\n")
        for c in seeds:
            f.write(f"{c}\t{c}\n")

    with open(out / "root_categories.tsv", "w") as f:
        f.write("category\n")
        for r in roots:
            f.write(f"{r}\n")

    with open(out / "facts.pl", "w") as f:
        f.write("%% Auto-generated category-only benchmark facts\n")
        f.write("%% Source: Simple English Wikipedia (simplewiki dumps)\n")
        f.write(f"%% Seeds: each category appears as its own 'article'\n\n")

        f.write("%% article_category(ArticleTitle, CategoryName).\n")
        for c in seeds:
            f.write(f"article_category({prolog_atom(c)}, {prolog_atom(c)}).\n")

        f.write("\n%% category_parent(ChildCategory, ParentCategory).\n")
        for child, parent in sorted(edges):
            f.write(f"category_parent({prolog_atom(child)}, {prolog_atom(parent)}).\n")

        f.write("\n%% root_category(CategoryName).\n")
        for r in roots:
            f.write(f"root_category({prolog_atom(r)}).\n")

    meta = {
        "source": "Simple English Wikipedia dumps (simplewiki)",
        "database": args.db,
        "shape": "category-only — seeds are categories themselves",
        "n_categories": len(all_cats),
        "n_hierarchy_edges": len(edges),
        "n_seeds": len(seeds),
        "n_roots": len(roots),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    total = len(seeds) + len(edges) + len(roots)
    print(f"\nWrote to {out}/")
    print(f"  facts.pl: {total:,} Prolog facts")
    print(f"  metadata.json: {meta}")

    conn.close()


if __name__ == "__main__":
    main()
