#!/usr/bin/env python3
"""Reusable subfield-boundary sampler: DOWNWARD (within-subfield, depth-bounded closure) and
BIDIRECTIONAL-coinflip (the boundary — up to a category's parents and across to sibling subfields) from
named seed categories. e5-math-coherence-filtered so the bidir walk can't drift out of math through apex
hubs (Oceanography/Alchemy). Runs on the math slice (the full enwiki graph OOMs the in-memory loader).

    UW_MU_GRAPH=../../data/benchmark/wide_enwiki_math/category_parent.tsv python3 gen_boundary_pairs.py \
        --down Algebraic_topology --bidir Topological_methods_of_algebraic_geometry --bidir Tensors \
        --out mu_pairs_boundary.tsv
"""
import argparse
import os
import re
import random
from collections import defaultdict

from gen_mu_pairs import load_graph, walk_bidir, GRAPH
from gen_more_sym_pairs import build_children_adj, load_existing_keys
from gen_multidomain_pairs import gen_within, gen_cross, e5_cos_to_roots

ROOT = os.path.dirname(os.path.abspath(__file__))
COH = ["Mathematics", "Physics", "Chemistry", "Computer_science", "Engineering", "Artificial_intelligence"]
ADMIN = re.compile(r"(Wikipedia|Articles?_|All_|Hidden_|CS1|Pages_|Webarchive|Commons|_stubs?$|Stub|"
                   r"Redirects|Short_desc|Use_|Template|Track|_by_|established_in|introductions|"
                   r"disambiguation|Overpopulated|Automatic_category_TOC|_journals$|_people$|_TOC)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", action="append", default=[], help="seed for DOWNWARD (within-subfield)")
    ap.add_argument("--bidir", action="append", default=[], help="seed for BIDIRECTIONAL (boundary)")
    ap.add_argument("--n-down", type=int, default=40)
    ap.add_argument("--n-bidir", type=int, default=40)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--math-floor", type=float, default=0.74)
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--dedup-against", default=os.path.join(ROOT, "mu_pairs_scored_cumulative.tsv"))
    ap.add_argument("--coh-cache", default=os.path.join(ROOT, "e5_mathfields_cos.pt"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "mu_pairs_boundary.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    full = load_graph(GRAPH)
    children = build_children_adj(GRAPH)
    parents = defaultdict(set)
    for par, ch in children.items():
        for c in ch:
            parents[c].add(par)
    deg = {n: max(1, len(full.get(n, ()))) for n in full}
    ns = sorted(full)
    cos = e5_cos_to_roots(ns, COH, cache=args.coh_cache)
    C = {r: {n: cos[r][i] for i, n in enumerate(ns)} for r in COH}
    def mathy(n):
        return max(COH, key=lambda r: C[r][n]) == "Mathematics" and C["Mathematics"][n] >= args.math_floor

    def closure(root):
        seen = {root} if root in full else set()
        fr = list(seen)
        for _ in range(args.depth):
            nx = []
            for n in fr:
                for ch in children.get(n, ()):
                    if ch not in seen and not ADMIN.search(ch):
                        seen.add(ch); nx.append(ch)
            fr = nx
        return {n for n in seen if not ADMIN.search(n)}

    def bidir_reach(root, n_steps=5000):
        out = set()
        for _ in range(n_steps):
            end, _ = walk_bidir(root, children, parents, deg, 0.4, 1.0, rng, mode="coinflip")
            if end != root and not ADMIN.search(end) and mathy(end):
                out.add(end)
        return out - {root}

    existing = load_existing_keys(args.dedup_against) if os.path.exists(args.dedup_against) else set()
    pairs = set(existing)
    rows, dom = [], set()
    for s in args.down:
        pool = closure(s)
        dom |= pool
        st = "pos_" + s.lower()[:14] + "_down"
        for a, b, wl in gen_within(pool, s, children, deg, args.n_down, pairs, rng):
            rows.append((a, b, st, wl))
        print(f"  DOWN {s}: pool {len(pool)} -> within pairs")
    for s in args.bidir:
        if s not in full:
            print(f"  BIDIR {s}: NOT IN GRAPH — skipped"); continue
        seedpool = closure(s) or {s}
        reach = bidir_reach(s)
        dom |= seedpool
        st = "cross_" + s.lower()[:14] + "_bnd"
        for a, b, _ in gen_cross(seedpool, reach, args.n_bidir, pairs, rng):
            rows.append((a, b, st, -1))
        print(f"  BIDIR {s}: seedpool {len(seedpool)}, math reach {len(reach)} -> boundary pairs")

    npos = len(rows)
    doml = sorted(dom)
    nn, tries = 0, 0
    while nn < args.neg_ratio * npos and tries < npos * 400:
        tries += 1
        a, b = rng.choice(doml), rng.choice(ns)
        if a == b or b in full.get(a, ()):
            continue
        k = tuple(sorted((a, b)))
        if k in pairs:
            continue
        pairs.add(k); rows.append((a, b, "neg", -1)); nn += 1
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        f.write("# subfield-boundary candidates (gen_boundary_pairs.py; corpus=enwiki). down=" +
                ",".join(args.down) + " bidir=" + ",".join(args.bidir) +
                ". cols: a\tb\tstratum\twl\tmu\n")
        for a, b, st, wl in rows:
            f.write(f"{a}\t{b}\t{st}\t{wl}\t{'0.0' if st == 'neg' else ''}\n")
    from collections import Counter
    cnt = Counter(r[2] for r in rows)
    print("wrote " + str(len(rows)) + " -> " + args.out + ": " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    print(f"to score (non-neg): {sum(v for k, v in cnt.items() if k != 'neg')}")


if __name__ == "__main__":
    main()
