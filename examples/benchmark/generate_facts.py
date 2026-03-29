#!/usr/bin/env python3
"""
Generate materialized facts for the cross-target effective distance benchmark.

Fetches article categories and category hierarchy from the Simple English
Wikipedia API, then outputs Prolog facts and TSV files.

Usage:
    # Dev dataset (first 20 articles, fast)
    python examples/benchmark/generate_facts.py \
        --articles /home/s243a/Projects/UnifyWeaver/reports/wikipedia_physics_articles.jsonl \
        --output data/benchmark/dev/ \
        --max-articles 20

    # Full 300-article dataset
    python examples/benchmark/generate_facts.py \
        --articles /home/s243a/Projects/UnifyWeaver/reports/wikipedia_physics_articles.jsonl \
        --output data/benchmark/dev/

    # Scale-up (requires broader article source)
    python examples/benchmark/generate_facts.py \
        --articles data/benchmark/bench/articles.jsonl \
        --output data/benchmark/bench/ \
        --max-articles 50000

Data source: Simple English Wikipedia API (simple.wikipedia.org)
The Cohere/Supabase dataset uses Simple English Wikipedia, so article titles
match this API.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# =============================================================================
# Wikipedia API
# =============================================================================

API_URL = "https://simple.wikipedia.org/w/api.php"
BATCH_SIZE = 50  # Wikipedia API allows up to 50 titles per request


def api_query(params: dict) -> dict:
    """Make a Wikipedia API request."""
    params["format"] = "json"
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "UnifyWeaver/1.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_article_categories(titles: List[str]) -> Dict[str, List[str]]:
    """
    Fetch categories for a batch of article titles.
    Returns {title: [category_name, ...]} with 'Category:' prefix stripped.
    """
    result = {}
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i : i + BATCH_SIZE]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "categories",
            "cllimit": "max",
            "clshow": "!hidden",  # skip hidden/maintenance categories
        }

        data = api_query(params)
        pages = data.get("query", {}).get("pages", {})

        for page in pages.values():
            title = page.get("title", "")
            cats = page.get("categories", [])
            cat_names = [
                c["title"].replace("Category:", "")
                for c in cats
                if c.get("ns") == 14
            ]
            if cat_names:
                result[title] = cat_names

        # Handle continuation
        while "continue" in data:
            params["clcontinue"] = data["continue"]["clcontinue"]
            data = api_query(params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                title = page.get("title", "")
                cats = page.get("categories", [])
                cat_names = [
                    c["title"].replace("Category:", "")
                    for c in cats
                    if c.get("ns") == 14
                ]
                if title in result:
                    result[title].extend(cat_names)
                elif cat_names:
                    result[title] = cat_names

        if i + BATCH_SIZE < len(titles):
            time.sleep(0.5)  # rate limit

        print(f"\r  Fetched categories for {min(i + BATCH_SIZE, len(titles))}/{len(titles)} articles", end="", flush=True)

    print()
    return result


def fetch_parent_categories(
    categories: Set[str], max_depth: int = 15
) -> Dict[str, List[str]]:
    """
    Walk up the category hierarchy via the API.
    Returns {child_category: [parent_category, ...]} for all reachable categories.
    """
    hierarchy = {}
    queue = list(categories)
    visited = set()
    depth = 0

    while queue and depth < max_depth:
        # Filter already-visited
        to_fetch = [c for c in queue if c not in visited]
        if not to_fetch:
            break

        visited.update(to_fetch)
        next_queue = []

        for i in range(0, len(to_fetch), BATCH_SIZE):
            batch = to_fetch[i : i + BATCH_SIZE]
            # Prepend "Category:" for API lookup
            cat_titles = [f"Category:{c}" for c in batch]

            params = {
                "action": "query",
                "titles": "|".join(cat_titles),
                "prop": "categories",
                "cllimit": "max",
                "clshow": "!hidden",
            }

            data = api_query(params)
            pages = data.get("query", {}).get("pages", {})

            for page in pages.values():
                title = page.get("title", "").replace("Category:", "")
                cats = page.get("categories", [])
                parent_names = [
                    c["title"].replace("Category:", "")
                    for c in cats
                    if c.get("ns") == 14
                ]
                if parent_names:
                    hierarchy[title] = parent_names
                    for p in parent_names:
                        if p not in visited:
                            next_queue.append(p)

            if i + BATCH_SIZE < len(to_fetch):
                time.sleep(0.5)

        queue = next_queue
        depth += 1
        print(f"  Depth {depth}: {len(visited)} categories explored, {len(queue)} in queue")

    return hierarchy


# =============================================================================
# Filtering
# =============================================================================

def filter_maintenance_categories(categories: Set[str]) -> Set[str]:
    """Remove Wikipedia maintenance/meta categories."""
    skip_prefixes = (
        "Articles ", "Pages ", "All ", "CS1 ", "Webarchive ",
        "Wikipedia ", "Commons ", "Wikidata ", "Use ",
        "Short description", "Good articles", "Featured articles",
    )
    skip_suffixes = (
        " stubs", " stub", " articles", " templates",
        " disambiguation", " redirects",
    )
    return {
        c for c in categories
        if not any(c.startswith(p) for p in skip_prefixes)
        and not any(c.endswith(s) for s in skip_suffixes)
    }


# =============================================================================
# Output
# =============================================================================

def write_tsv(path: Path, header: List[str], rows: List[Tuple]):
    """Write a TSV file."""
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"  Wrote {len(rows)} rows to {path}")


def write_prolog_facts(
    path: Path,
    article_categories: Dict[str, List[str]],
    category_hierarchy: Dict[str, List[str]],
    root_categories: Set[str],
):
    """Write Prolog fact file."""
    with open(path, "w") as f:
        f.write("%% Auto-generated benchmark facts\n")
        f.write("%% Source: Simple English Wikipedia API\n\n")

        # article_category/2
        f.write("%% article_category(ArticleTitle, CategoryName).\n")
        for article, cats in sorted(article_categories.items()):
            for cat in cats:
                f.write(f"article_category({prolog_atom(article)}, {prolog_atom(cat)}).\n")

        f.write("\n")

        # category_parent/2
        f.write("%% category_parent(ChildCategory, ParentCategory).\n")
        for child, parents in sorted(category_hierarchy.items()):
            for parent in parents:
                f.write(f"category_parent({prolog_atom(child)}, {prolog_atom(parent)}).\n")

        f.write("\n")

        # root_category/1
        f.write("%% root_category(CategoryName).\n")
        for root in sorted(root_categories):
            f.write(f"root_category({prolog_atom(root)}).\n")

    total = sum(len(v) for v in article_categories.values()) + \
            sum(len(v) for v in category_hierarchy.values()) + \
            len(root_categories)
    print(f"  Wrote {total} facts to {path}")


def prolog_atom(s: str) -> str:
    """Escape a string as a Prolog atom."""
    escaped = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark facts from Wikipedia category hierarchy"
    )
    parser.add_argument(
        "--articles", required=True, help="Path to articles JSONL file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for facts"
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Max articles to process (for testing)"
    )
    parser.add_argument(
        "--root", default="Science",
        help="Root category name (default: Science)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=15,
        help="Max depth for category hierarchy traversal"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load article titles
    print("Step 1: Loading article titles...")
    titles = []
    with open(args.articles) as f:
        for line in f:
            rec = json.loads(line)
            titles.append(rec["title"])
            if args.max_articles and len(titles) >= args.max_articles:
                break
    print(f"  Loaded {len(titles)} article titles")

    # 2. Fetch article categories
    print("Step 2: Fetching article categories from Simple Wikipedia API...")
    article_cats = fetch_article_categories(titles)
    print(f"  Got categories for {len(article_cats)}/{len(titles)} articles")

    # 3. Collect all unique categories
    all_cats = set()
    for cats in article_cats.values():
        all_cats.update(cats)
    all_cats = filter_maintenance_categories(all_cats)
    print(f"  {len(all_cats)} unique content categories")

    # Filter article_cats to only content categories
    article_cats = {
        art: [c for c in cats if c in all_cats]
        for art, cats in article_cats.items()
    }
    article_cats = {art: cats for art, cats in article_cats.items() if cats}

    # 4. Walk category hierarchy up to root
    print("Step 3: Walking category hierarchy...")
    hierarchy = fetch_parent_categories(all_cats, max_depth=args.max_depth)

    # Filter maintenance categories from hierarchy too
    all_hierarchy_cats = set()
    for child, parents in hierarchy.items():
        all_hierarchy_cats.add(child)
        all_hierarchy_cats.update(parents)
    content_cats = filter_maintenance_categories(all_hierarchy_cats)

    hierarchy = {
        child: [p for p in parents if p in content_cats]
        for child, parents in hierarchy.items()
        if child in content_cats
    }
    hierarchy = {k: v for k, v in hierarchy.items() if v}

    # 5. Check root reachability
    root_cats = {args.root}
    reachable_from_root = set()
    queue = [args.root]
    while queue:
        cat = queue.pop(0)
        if cat in reachable_from_root:
            continue
        reachable_from_root.add(cat)
        # Find children of this category
        for child, parents in hierarchy.items():
            if cat in parents and child not in reachable_from_root:
                queue.append(child)

    print(f"  {len(reachable_from_root)} categories reachable from root '{args.root}'")

    # 6. Write outputs
    print("Step 4: Writing output files...")

    # TSV files
    ac_rows = [(art, cat) for art, cats in sorted(article_cats.items()) for cat in cats]
    write_tsv(output_dir / "article_category.tsv", ["article", "category"], ac_rows)

    cp_rows = [(child, parent) for child, parents in sorted(hierarchy.items()) for parent in parents]
    write_tsv(output_dir / "category_parent.tsv", ["child", "parent"], cp_rows)

    write_tsv(output_dir / "root_categories.tsv", ["category"], [(r,) for r in sorted(root_cats)])

    # Prolog facts
    write_prolog_facts(output_dir / "facts.pl", article_cats, hierarchy, root_cats)

    # Summary
    print("\nSummary:")
    print(f"  Articles with categories: {len(article_cats)}")
    print(f"  Unique categories: {len(all_cats)}")
    print(f"  Category hierarchy edges: {sum(len(v) for v in hierarchy.values())}")
    print(f"  Categories reachable from '{args.root}': {len(reachable_from_root)}")
    print(f"  Root: {args.root}")

    # Save metadata
    meta = {
        "source": "simple.wikipedia.org API",
        "root": args.root,
        "n_articles": len(article_cats),
        "n_categories": len(all_cats),
        "n_hierarchy_edges": sum(len(v) for v in hierarchy.values()),
        "n_reachable_from_root": len(reachable_from_root),
        "max_depth": args.max_depth,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata written to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
