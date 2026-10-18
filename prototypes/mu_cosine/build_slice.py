#!/usr/bin/env python3
"""Reusable streaming-BFS slice extractor for the full enwiki category graph (615MB child->parent TSV;
naive in-memory load OOMs). Builds a local neighbourhood around named seed roots:
  * downward closure (depth-bounded) from each root,
  * one level UP (the roots' parents), for context,
  * apex-capped siblings (the parents' other children, <= --apex-cap each, so a hub parent can't explode
    the slice).
Then writes the induced subgraph (every edge with BOTH endpoints kept) as child<TAB>parent, matching the
wide_enwiki_* slice format. Admin/maintenance categories are filtered throughout.

    python3 build_slice.py --root Network_theory --root Dynamical_systems --root Complex_systems_theory \
        --root Systems_analysis --root Networks --depth 2 \
        --out ../../data/benchmark/wide_enwiki_systheory/category_parent.tsv
"""
import argparse
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(os.path.abspath(os.path.join(ROOT, "..", "..")), "data", "benchmark",
                    "enwiki_named", "category_parent.tsv")
ADMIN = re.compile(r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
                   r"Redirects|Short_desc|Use_|Templates?|Track|_by_|_in_\d|established_in|introductions|"
                   r"disambiguation|_people$|_journals$|_awards$|Wikipedians)")


def stream(path):
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0] and p[1]:
                yield p[0], p[1]   # child, parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[], required=True)
    ap.add_argument("--graph", default=FULL)
    ap.add_argument("--depth", type=int, default=2, help="downward closure depth")
    ap.add_argument("--apex-cap", type=int, default=120, help="max siblings kept per up-parent")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    roots = set(args.root)
    kept = set(roots)

    # --- downward closure (one streaming pass per depth level) ---
    frontier = set(roots)
    for d in range(args.depth):
        nxt = set()
        for c, par in stream(args.graph):
            if par in frontier and c not in kept and not ADMIN.search(c):
                nxt.add(c)
        kept |= nxt
        frontier = nxt
        print(f"  depth {d+1}: +{len(nxt)} nodes (kept {len(kept)})")
        if not frontier:
            break

    # --- one level UP: parents of the roots ---
    ups = set()
    for c, par in stream(args.graph):
        if c in roots and not ADMIN.search(par):
            ups.add(par)
    kept |= ups
    print(f"  up: +{len(ups)} parent nodes (kept {len(kept)})")

    # --- apex-capped siblings: other children of those up-parents ---
    sib_cnt = defaultdict(int)
    sibs = set()
    for c, par in stream(args.graph):
        if par in ups and c not in kept and not ADMIN.search(c) and sib_cnt[par] < args.apex_cap:
            sibs.add(c)
            sib_cnt[par] += 1
    kept |= sibs
    print(f"  siblings (cap {args.apex_cap}): +{len(sibs)} nodes (kept {len(kept)})")

    # --- final pass: induced subgraph (both endpoints kept) ---
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    edges = set()
    for c, par in stream(args.graph):
        if c in kept and par in kept:
            edges.add((c, par))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("child\tparent\n")
        for c, par in sorted(edges):
            f.write(f"{c}\t{par}\n")
    print(f"wrote {len(edges)} edges over {len(kept)} nodes -> {args.out}")


if __name__ == "__main__":
    main()
