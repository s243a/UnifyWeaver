#!/usr/bin/env python3
"""SimpleMind single-folder hit@k — the third corpus in the cross-corpus ranking table.

Pearltrees and SimpleMind are the SAME task shape: one principal parent per node (single right
answer), unlike Wikipedia's multi-parent facets (wiki_multiparent_hits.py). Two grains:

  parent-level  query = node title, true = immediate parent (materialized path), catalog = all
                internal SM nodes across maps — the direct Pearltrees-filing analog.
  root-level    query = node title, true = its map ROOT, catalog = SM roots + the full Pearltrees
                folder catalog (min_bm=3, 335 titles) as distractors — the owner's framing: map
                roots are the easiest SM↔filing match, so test whether e5 finds the right root
                even when it competes with every real filing folder.

Input: mindmap_lineage.tsv (gen_mindmap_lineage.py; privacy-filtered at parse time).

  python3 sm_filing_hits.py
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_filing import load_filing
from mu_attention import build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))
TREES = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api", "trees")


def metrics(name, ranks, ks=(1, 5, 10)):
    r = np.array(ranks, float)
    out = f"  {name:34s} n={len(r):4d}  MRR {np.mean(1 / r):.3f}"
    for k in ks:
        out += f"  R@{k} {np.mean(r <= k):.3f}"
    print(out + f"  med {int(np.median(r))}")


def rank_of(cos_row, cand_titles, true_title):
    """Best rank over title-equivalent candidates (duplicate titles share the best alias)."""
    tp = [j for j, t in enumerate(cand_titles) if t == true_title]
    best = None
    for j in tp:
        rk = 1 + int(np.sum(cos_row > cos_row[j]))
        best = rk if best is None else min(best, rk)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default=os.path.join(ROOT, "mindmap_lineage.tsv"))
    ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--e5-cache", default="/tmp/mu_data/sm_filing_e5.pt")
    a = ap.parse_args(argv)

    rows = []
    with open(a.lineage, encoding="utf-8") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 5:
                rows.append((p[0], p[2], p[3].split(" / "), int(p[4])))
    internal = sorted({t for _, _, path, _ in rows for t in path[:-1]})
    roots = sorted({path[0] for _, _, path, _ in rows})
    q_parent = [(node, path[-2]) for _, node, path, d in rows if d >= 2 and node != path[-2]]
    q_root = [(node, path[0]) for _, node, path, d in rows if d >= 2 and node != path[0]]
    print(f"lineage rows: {len(rows)}; internal nodes: {len(internal)}; roots: {len(roots)}")

    _, cand = load_filing(TREES, a.min_bm)
    pt_titles = sorted(set(cand.values()))
    root_cat = sorted(set(roots) | set(pt_titles))
    print(f"parent-level catalog: {len(internal)}; root-level catalog: {len(root_cat)} "
          f"({len(roots)} SM roots + {len(pt_titles)} PT folder distractors)")

    names = sorted({n for n, _ in q_parent} | {n for n, _ in q_root} | set(internal) | set(root_cat))
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    Q, P = qtbl.numpy(), ptbl.numpy()

    print("\nSingle-folder ranking (e5), SimpleMind — task-matched to Pearltrees filing:")
    cv = np.stack([P[idx[t]] for t in internal])
    ranks = []
    for node, par in q_parent:
        ranks.append(rank_of(Q[idx[node]] @ cv.T, internal, par))
    metrics("SM parent-level (PT analog)", ranks)

    cv2 = np.stack([P[idx[t]] for t in root_cat])
    ranks2 = []
    for node, rt in q_root:
        ranks2.append(rank_of(Q[idx[node]] @ cv2.T, root_cat, rt))
    metrics("SM root-level (+PT distractors)", ranks2)

    print("\nStanding comparators: PT single-folder R@1 0.203 / MRR 0.291 (catalog 335); "
          "wiki any-parent@1 0.945, all-4 median k=181 (catalog 5,000).")


if __name__ == "__main__":
    main()
