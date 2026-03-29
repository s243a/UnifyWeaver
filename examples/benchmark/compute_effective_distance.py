#!/usr/bin/env python3
"""
Compute effective distance from category_ancestor output + facts.

Takes the output of any target's category_ancestor computation (TSV:
source_category, ancestor, hops) and computes d_eff for each article.

This is the aggregation layer that sits on top of the transitive closure
output from any compiled target.

Usage:
    # From AWK output:
    awk -f outputs/category_ancestor.awk /dev/null | \
        python compute_effective_distance.py \
            --facts data/benchmark/dev/facts.pl \
            --root Physics

    # From a pre-computed TSV file:
    python compute_effective_distance.py \
        --ancestor-tsv outputs/category_ancestor_output.tsv \
        --facts data/benchmark/dev/facts.pl \
            --root Physics
"""

import argparse
import sys
from collections import defaultdict


def load_prolog_facts(facts_path):
    """Load article_category and root_category from Prolog facts file."""
    article_cats = defaultdict(list)
    root_cats = set()

    with open(facts_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("article_category("):
                # article_category('Name', 'Category').
                inner = line[len("article_category("):-2]  # strip wrapper + ).
                parts = inner.split("', '")
                if len(parts) == 2:
                    art = parts[0].strip("'")
                    cat = parts[1].strip("'")
                    article_cats[art].append(cat)
            elif line.startswith("root_category("):
                inner = line[len("root_category("):-2]
                root = inner.strip("'")
                root_cats.add(root)

    return dict(article_cats), root_cats


def compute_deff(path_hops, n=5):
    """d_eff = (Σ d^(-n))^(-1/n)"""
    if not path_hops:
        return float('inf')
    weight_sum = sum(h ** (-n) for h in path_hops)
    if weight_sum <= 0:
        return float('inf')
    return weight_sum ** (-1.0 / n)


def main():
    parser = argparse.ArgumentParser(
        description="Compute effective distance from category_ancestor output"
    )
    parser.add_argument("--ancestor-tsv", help="Pre-computed ancestor TSV file")
    parser.add_argument("--facts", required=True, help="Prolog facts file")
    parser.add_argument("--root", default="Physics", help="Root category")
    parser.add_argument("--n", type=float, default=5, help="Dimensionality parameter")
    args = parser.parse_args()

    # Load facts
    article_cats, root_cats = load_prolog_facts(args.facts)
    if args.root:
        root_cats = {args.root}

    # Load ancestor tuples: (source_category, ancestor, hops)
    # From stdin or file
    ancestor_paths = defaultdict(lambda: defaultdict(list))  # cat → ancestor → [hops]

    if args.ancestor_tsv:
        source = open(args.ancestor_tsv)
    else:
        source = sys.stdin

    for line in source:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            src_cat, ancestor, hops = parts[0], parts[1], int(parts[2])
            if ancestor in root_cats:
                ancestor_paths[src_cat][ancestor].append(hops)

    if args.ancestor_tsv:
        source.close()

    # Compute effective distance per (article, root)
    results = []
    for article, cats in sorted(article_cats.items()):
        for root in sorted(root_cats):
            all_hops = []
            for cat in cats:
                # Direct: if cat IS the root
                if cat == root:
                    all_hops.append(1)
                # Via ancestors
                if cat in ancestor_paths and root in ancestor_paths[cat]:
                    for h in ancestor_paths[cat][root]:
                        all_hops.append(h + 1)  # +1 for article→category hop

            if all_hops:
                deff = compute_deff(all_hops, n=args.n)
                results.append((article, root, deff))

    # Output sorted by (root, distance, article)
    results.sort(key=lambda x: (x[1], x[2], x[0]))

    print("article\troot_category\teffective_distance")
    for article, root, deff in results:
        print(f"{article}\t{root}\t{deff:.6f}")


if __name__ == "__main__":
    main()
