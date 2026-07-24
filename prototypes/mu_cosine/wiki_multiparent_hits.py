#!/usr/bin/env python3
"""Wikipedia multi-parent hit@k: at what k does e5 capture ALL true parent categories?

Pearltrees/SimpleMind filing is single-principal-folder ranking (one right answer). Wikipedia
categories differ structurally: a category has SEVERAL valid parents (mean ~3.8 on the correct
enwiki ingest), so the natural metrics split:

  any-parent hit@k   — ≥1 true parent in the e5 top-k (the easy direction: more targets)
  all-parents hit@k  — ALL j true parents in the top-k (the owner's question: "at what k do we
                       capture four of the parent categories")

Setup mirrors the Pearltrees harness scale: queries = categories with ≥ --min-parents parents
(sampled), catalog = the union of all sampled queries' parents plus random distractor categories
(--catalog-size total), ranking = e5 query→passage cosine over the catalog, title-equivalence not
needed (titles are unique keys in the named graph). Graph: data/benchmark/enwiki_named (3-dump
Correct-mode ingest, admin categories excluded).

  python3 wiki_multiparent_hits.py                # defaults: 400 queries x >=4 parents, 5k catalog
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import build_e5_tables

EDGES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "data", "benchmark", "enwiki_named", "category_parent.tsv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-parents", type=int, default=4)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--catalog-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ks", type=int, nargs="*", default=(1, 5, 10, 20, 50, 100, 200, 500))
    ap.add_argument("--e5-cache", default="/tmp/mu_data/wiki_multiparent_e5.pt")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    parents = defaultdict(set)
    all_nodes = set()
    with open(EDGES, encoding="utf-8") as f:
        next(f)
        for ln in f:
            c, p = ln.rstrip("\n").split("\t")
            parents[c].add(p)
            all_nodes.add(c)
            all_nodes.add(p)
    print(f"graph: {len(all_nodes):,} nodes; children with >= {a.min_parents} parents: "
          f"{sum(1 for v in parents.values() if len(v) >= a.min_parents):,}")

    eligible = sorted(c for c, v in parents.items() if len(v) >= a.min_parents)
    queries = rng.sample(eligible, a.n_queries)
    true_parents = {q: sorted(parents[q]) for q in queries}

    cat = set()
    for q in queries:
        cat |= parents[q]
    pool = sorted(all_nodes - cat - set(queries))
    n_fill = max(0, a.catalog_size - len(cat))
    cat = sorted(cat) + rng.sample(pool, n_fill)
    ci = {t: i for i, t in enumerate(cat)}
    print(f"queries: {len(queries)} (parents per query: median "
          f"{int(np.median([len(true_parents[q]) for q in queries]))}, "
          f"max {max(len(true_parents[q]) for q in queries)}); catalog: {len(cat)} "
          f"({len(ci) - n_fill} true-parent union + {n_fill} random distractors)")

    names = sorted(set(cat) | set(queries))
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    Q, P = qtbl.numpy(), ptbl.numpy()
    qv = np.stack([Q[idx[t]] for t in queries])
    cv = np.stack([P[idx[t]] for t in cat])
    cos = qv @ cv.T
    order = np.argsort(-cos, axis=1)

    # per-query sorted ranks of the true parents
    j_max = 4
    ranks_of_true = []
    for i, q in enumerate(queries):
        pos = {int(c): r + 1 for r, c in enumerate(order[i])}
        rs = sorted(pos[ci[p]] for p in true_parents[q])
        ranks_of_true.append(rs)

    print(f"\nhit-j@k = fraction of queries with >= j true parents inside the e5 top-k "
          f"(catalog {len(cat)}):")
    hdr = "  k      " + "".join(f"  >= {j}   " for j in range(1, j_max + 1))
    print(hdr)
    for k in a.ks:
        cells = []
        for j in range(1, j_max + 1):
            cells.append(np.mean([len(rs) >= j and rs[j - 1] <= k for rs in ranks_of_true]))
        print(f"  {k:<6d}" + "".join(f"  {c:.3f} " for c in cells))

    # the owner's number: k needed to capture the 4th parent
    k4 = [rs[3] for rs in ranks_of_true if len(rs) >= 4]
    k1 = [rs[0] for rs in ranks_of_true]
    print(f"\nk to capture the 1st parent: median {int(np.median(k1))}, "
          f"p75 {int(np.percentile(k1, 75))}, p90 {int(np.percentile(k1, 90))}")
    print(f"k to capture 4 parents:      median {int(np.median(k4))}, "
          f"p75 {int(np.percentile(k4, 75))}, p90 {int(np.percentile(k4, 90))}")
    print("\n(Contrast: Pearltrees/SimpleMind are single-principal-folder tasks — one target, "
          "harder per-target; Wikipedia's multiple valid parents make ANY-parent easy and "
          "ALL-parents the structurally interesting curve.)")


if __name__ == "__main__":
    main()
