#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""
generate_synthetic_category_graph.py — large synthetic category DAG for the
WAM-Rust graph-search cache benchmark, when a real wiki dump is unavailable
(network-restricted environments). Produces the same three TSVs the matrix
bench / ingest_resident_lmdb_fixture.py consume:

    category_parent.tsv   child<TAB>parent   (+ header)
    article_category.tsv  article<TAB>category
    root_categories.tsv   category           (single root)

Structure is chosen to mimic the cache-relevant properties of a wiki category
graph: a single root, a small set of high-in-degree "hub" categories near the
root (so many ancestor paths converge -> reuse -> cache hits), and otherwise
locally-wired parents (so paths have depth). This makes the union of seed
ancestor cones large enough to overflow the runtime's per-thread L1 cache and
exercise the shared L2 under capacity pressure — the regime the small in-repo
fixtures (working set ~1.7k << 65k-slot L1) cannot reach.

Deterministic given --seed. Unlike a real dump there is no golden output, so the
benchmark's correctness invariant is cached-output == lazy-output (caching must
not change the answer), checked by the sweep driver.

Usage:
  generate_synthetic_category_graph.py --out DIR --categories N --articles M \
      [--hubs H] [--local-window W] [--hub-prob P] [--max-parents K] [--seed S]
"""
import argparse
import os
import random


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", type=int, default=200_000)
    ap.add_argument("--articles", type=int, default=200_000)
    ap.add_argument("--hubs", type=int, default=512,
                    help="low-index categories that act as shared hub ancestors")
    ap.add_argument("--local-window", type=int, default=2000,
                    help="parents are drawn from the preceding window (gives depth)")
    ap.add_argument("--hub-prob", type=float, default=0.35,
                    help="probability a parent edge points at a hub (gives reuse)")
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    N, H, W = args.categories, max(1, args.hubs), max(1, args.local_window)

    cp_path = os.path.join(args.out, "category_parent.tsv")
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write("child\tparent\n")
        # C0 is the root (no parents). Every other category points up toward
        # lower indices: with hub-prob to a hub (C1..CH, near the root), else to
        # a local predecessor within the window (creating depth). C1..CH chain to
        # the root so hubs themselves are shallow.
        for i in range(1, N):
            if i <= H:
                f.write(f"C{i}\tC0\n")
                continue
            k = rnd.randint(1, args.max_parents)
            parents = set()
            for _ in range(k):
                if rnd.random() < args.hub_prob:
                    parents.add(rnd.randint(1, H))           # hub
                else:
                    lo = max(1, i - W)
                    parents.add(rnd.randint(lo, i - 1))      # local (depth)
            for p in parents:
                f.write(f"C{i}\tC{p}\n")

    # Articles attach to leaf-ish (higher-index) categories so their ancestor
    # cones are deep and broad; bias toward the upper half of the index range.
    lo_art = max(1, N // 2)
    ac_path = os.path.join(args.out, "article_category.tsv")
    with open(ac_path, "w", encoding="utf-8") as f:
        f.write("article\tcategory\n")
        for j in range(args.articles):
            f.write(f"A{j}\tC{rnd.randint(lo_art, N - 1)}\n")

    with open(os.path.join(args.out, "root_categories.tsv"), "w", encoding="utf-8") as f:
        f.write("category\nC0\n")

    print(f"synthetic graph at {args.out}: {N} categories ({H} hubs), "
          f"{args.articles} articles, root=C0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
