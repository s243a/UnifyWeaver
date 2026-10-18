#!/usr/bin/env python3
"""SVD / μ-coverage page selector — the second sampling method (cf. select_diverse.py's farthest-point).
Per the user's design: take the top-K SVD axes of a category's page-embedding matrix and keep a subset that
COVERS the (e5-principal-axis × μ) joint — along EACH principal direction, keep pages spanning the μ
(centrality) distribution (both ends of the axis × low/mid/high μ), so the ELEM operator learns how
membership varies along each subtopic direction rather than only seeing the densest cluster. Junk (μ <
--min-mu) dropped; negatives passed through (so "enough negatives" is preserved).

PRECONDITIONS (the user's two design points):
  * SUFFICIENCY THRESHOLD — selection is only worthwhile once a category has enough page data. Below
    --min-pages the category is kept WHOLE (no sampling).
  * SUBCATEGORY AUGMENTATION — to push a thin-but-has-subcats category over the threshold, harvest its
    subcategories' pages too first: `fetch_category_pages.py --recurse-subcats N`.

K is chosen by the SVD variance elbow (cumulative explained variance ≥ --var-thresh, capped at --max-k) —
the category's intrinsic dimension, not a fixed fraction.

    python3 select_svd_coverage.py --scored mu_pairs_scored_pages_<ts>.tsv --min-pages 25 \
        --out mu_pairs_scored_pages_svd.tsv
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from mu_attention import build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--min-mu", type=float, default=0.4, help="drop pages below this centrality (junk)")
    ap.add_argument("--min-pages", type=int, default=25, help="SUFFICIENCY: below this, keep the category whole")
    ap.add_argument("--var-thresh", type=float, default=0.9, help="cumulative explained variance to pick K")
    ap.add_argument("--max-k", type=int, default=6, help="cap on SVD axes")
    ap.add_argument("--mu-bins", type=int, default=3, help="μ levels to cover along each axis")
    ap.add_argument("--cache", default=os.path.join(ROOT, "e5_svdsel_cos.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    header, rows = [], []
    cat_rows = defaultdict(list)
    for ln in open(args.scored, encoding="utf-8"):
        if ln.startswith("#"):
            header.append(ln); continue
        c = ln.rstrip("\n").split("\t")
        rows.append(c)
        if len(c) >= 5 and c[2] != "neg" and c[2].startswith("pos_pageof_"):
            cat_rows[c[0]].append(c)

    pages = sorted({c[1] for cs in cat_rows.values() for c in cs})
    q, p, idx = build_e5_tables(pages, cache_path=args.cache)
    P = p.numpy()   # passage embeddings (unit-normed), [N,384]

    keep, report = set(), []
    for cat, cs in sorted(cat_rows.items()):
        cand = [c for c in cs if float(c[4]) >= args.min_mu] or cs
        n = len(cand)
        if n < args.min_pages:                       # sufficiency gate: keep whole
            for c in cand:
                keep.add((c[0], c[1]))
            report.append((cat, len(cs), n, n, "whole<thr"))
            continue
        X = np.stack([P[idx[c[1]]] for c in cand])    # [n,384]
        mu = np.array([float(c[4]) for c in cand])
        Xc = X - X.mean(0, keepdims=True)
        # SVD → principal axes (right singular vectors) + variance elbow for K
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2); cum = np.cumsum(var) / var.sum()
        K = int(min(args.max_k, max(1, np.searchsorted(cum, args.var_thresh) + 1)))
        proj = Xc @ Vt[:K].T                          # [n,K] projection on each axis
        # μ bins (low..high centrality)
        edges = np.linspace(mu.min(), mu.max() + 1e-9, args.mu_bins + 1)
        mubin = np.clip(np.digitize(mu, edges[1:-1]), 0, args.mu_bins - 1)
        picked = set()
        # cover each principal axis at BOTH ends × each μ level
        for k in range(K):
            for b in range(args.mu_bins):
                cell = [i for i in range(n) if mubin[i] == b]
                if not cell:
                    continue
                hi = max(cell, key=lambda i: proj[i, k])   # most +aligned with axis k in this μ band
                lo = min(cell, key=lambda i: proj[i, k])   # most −aligned
                picked.add(hi); picked.add(lo)
        for i in picked:
            keep.add((cand[i][0], cand[i][1]))
        report.append((cat, len(cs), n, len(picked), f"K={K}"))

    # negatives & non-pageof rows pass through verbatim; only pageof rows are filtered
    with open(args.out, "w", encoding="utf-8") as f:
        for h in header:
            f.write(h)
        kept = dropped = 0
        for c in rows:
            if len(c) >= 3 and c[2].startswith("pos_pageof_"):
                if (c[0], c[1]) in keep:
                    f.write("\t".join(c) + "\n"); kept += 1
                else:
                    dropped += 1
            else:
                f.write("\t".join(c) + "\n")
    print(f"{'category':30} {'all':>4} {'≥μ':>4} {'kept':>4}  note")
    for cat, a, b, k, note in report:
        print(f"  {cat[:28]:28} {a:>4} {b:>4} {k:>4}  {note}")
    print(f"\nkept {kept} / dropped {dropped} pos_pageof rows -> {args.out}")


if __name__ == "__main__":
    main()
