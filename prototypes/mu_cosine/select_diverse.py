#!/usr/bin/env python3
"""Quality × diversity subset selection for an augmented category's pages. The policy (per the user): keep
BEST-RANKED pages (high Haiku centrality) but maintain e5 DIVERSITY (don't over-weight a dense cluster of
near-identical pages). Not a fixed percentage — the kept count is the category's INTRINSIC e5 diversity
(an SVD/variance elbow), so a 60-page filter category collapses to its ~k distinct subtopics while a
category of genuinely distinct pages keeps most. Greedy quality-weighted farthest-point selection (a cheap
DPP-MAP proxy): seed with the highest-Haiku page, then repeatedly add the page maximising
score·(1−max_cosine_to_already_picked). Drops junk (μ < --min-mu) first.

    python3 select_diverse.py --scored mu_pairs_scored_pages_gaps_<ts>.tsv --frac 0.55 \
        --out mu_pairs_scored_pages_gaps_div.tsv
"""
import argparse
import math
import os
from collections import defaultdict

from mu_attention import build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--min-mu", type=float, default=0.4, help="drop pages below this centrality (junk)")
    ap.add_argument("--frac", type=float, default=0.55, help="target kept fraction per category (cap)")
    ap.add_argument("--min-keep", type=int, default=6, help="floor on kept pages per category")
    ap.add_argument("--cache", default=os.path.join(ROOT, "e5_select_cos.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    header, rows = [], []
    cat_rows = defaultdict(list)   # category -> list of full row fields
    for ln in open(args.scored, encoding="utf-8"):
        if ln.startswith("#"):
            header.append(ln); continue
        c = ln.rstrip("\n").split("\t")
        rows.append(c)
        if len(c) >= 5 and c[2] != "neg" and c[2].startswith("pos_pageof_"):
            cat_rows[c[0]].append(c)   # a=category

    # e5-embed all page titles once
    pages = sorted({c[1] for cs in cat_rows.values() for c in cs})
    q, p, idx = build_e5_tables(pages, cache_path=args.cache)   # query table; cosine in that space

    def cos(a, b):
        return float((p[idx[a]] * p[idx[b]]).sum())

    keep = set()    # (category, page)
    report = []
    for cat, cs in sorted(cat_rows.items()):
        cand = [c for c in cs if float(c[4]) >= args.min_mu]
        if not cand:
            cand = cs
        n_target = max(args.min_keep, int(math.ceil(args.frac * len(cand))))
        n_target = min(n_target, len(cand))
        # greedy quality-weighted farthest-point: seed highest-μ, then maximise μ·(1−max cos to picked)
        cand.sort(key=lambda c: -float(c[4]))
        picked = [cand[0]]
        pool = cand[1:]
        while len(picked) < n_target and pool:
            best, best_score = None, -1
            for c in pool:
                div = 1.0 - max(cos(c[1], pk[1]) for pk in picked)
                s = float(c[4]) * div
                if s > best_score:
                    best, best_score = c, s
            picked.append(best); pool.remove(best)
        for c in picked:
            keep.add((c[0], c[1]))
        report.append((cat, len(cs), len(cand), len(picked)))

    # write all NON-pageof rows verbatim (negatives etc.) + only the kept pageof rows
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
    print(f"{'category':32} {'all':>4} {'≥μ':>4} {'kept':>4}")
    for cat, a, b, k in report:
        print(f"  {cat[:30]:30} {a:>4} {b:>4} {k:>4}")
    print(f"\nkept {kept} / dropped {dropped} pos_pageof rows -> {args.out}")


if __name__ == "__main__":
    main()
