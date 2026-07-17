#!/usr/bin/env python3
"""Why does e5 beat the μ heads at filing? Two hypotheses, one diagnostic.

H1 (capability): the μ operators genuinely carry less filing signal than raw e5 cosine.
H2 (regime): the filing candidate set is semantically SPREAD OUT, so e5 cosine already separates
    the true folder easily and μ's fine-grained directional/membership discrimination has nothing
    to add — μ would show its strength where folders are semantically CLOSE (the hard cases).

Test: stratify queries by how hard the choice is in e5 space — the gap between the true folder's
e5-cosine and the best DISTRACTOR's e5-cosine. NEAR (small/negative gap = confusable folders) is
where H2 predicts μ should help most; FAR (large gap = e5 already wins) is where e5 dominates.
Report recall@1 for e5 vs the best conditioned μ head per stratum.

  python3 diagnose_filing_e5_vs_mu.py --tuned model_pt_filing.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_filing import load_filing
from eval_pearltrees_filing import ranks_from, score_cond
from fine_tune_pearltrees_filing import load_with_lineage_ops
from mu_attention import Tokenizer, build_e5_tables

ROOT = os.path.dirname(os.path.abspath(__file__))
TREES = os.path.join(ROOT, "..", "..", ".local", "data", "pearltrees_api", "trees")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", default=os.path.join(ROOT, "model_pt_filing.pt"))
    ap.add_argument("--trees", default=TREES)
    ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", default="/tmp/mu_data/pt_filing_eval_e5.pt")
    a = ap.parse_args(argv)
    dev = "cpu"
    torch.set_num_threads(4)

    import random
    queries, cand = load_filing(a.trees, a.min_bm)
    queries = sorted(queries)
    if len(queries) > a.max_queries:
        queries = random.Random(a.seed).sample(queries, a.max_queries)
    f_ids = sorted(cand)
    f_titles = [cand[fid] for fid in f_ids]
    q_titles = [q for q, _ in queries]
    by_title = {}
    for j, t in enumerate(f_titles):
        by_title.setdefault(t, []).append(j)
    truepos = [sorted(by_title[cand[fid]]) for _, fid in queries]

    names = sorted(set(q_titles) | set(f_titles))
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.cache, batch_size=128)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})

    C = (qtbl[[idx[k] for k in q_titles]] @ ptbl[[idx[k] for k in f_titles]].T).cpu()
    # e5 "difficulty" per query: true-folder cosine minus the best distractor cosine
    gaps = []
    for r in range(len(q_titles)):
        tp = set(truepos[r])
        true_c = max(C[r][j] for j in tp)
        distractor = max((C[r][j] for j in range(C.shape[1]) if j not in tp), default=-1.0)
        gaps.append(float(true_c - distractor))
    gaps = np.array(gaps)
    e5_ranks = np.array(ranks_from(C, truepos))

    model, _ = load_with_lineage_ops(a.tuned, dev=dev)
    model.eval()
    S_elem = score_cond(model, tok, q_titles, f_titles, "ELEM", "kalman-fused", dev)
    S_hier = score_cond(model, tok, q_titles, f_titles, "HIER", "kalman-fused", dev)
    S_sym = score_cond(model, tok, q_titles, f_titles, "SYM", "kalman-fused", dev)
    S_max = torch.maximum(torch.maximum(S_elem, S_hier), S_sym)
    mu_ranks = np.array(ranks_from(S_max, truepos))

    print(f"queries {len(q_titles)}; e5 difficulty gap (true − best distractor cosine): "
          f"min {gaps.min():+.3f} median {np.median(gaps):+.3f} max {gaps.max():+.3f}")
    order = np.argsort(gaps)                          # ascending = hardest (most confusable) first
    thirds = np.array_split(order, 3)
    print(f"\n  {'stratum':22s} {'n':>4s} {'gap range':>16s} {'e5 R@1':>8s} {'mu-max R@1':>11s} {'μ−e5':>7s}")
    for name, sel in [("NEAR (confusable)", thirds[0]), ("MID", thirds[1]), ("FAR (e5 easy)", thirds[2])]:
        g = gaps[sel]
        e = float((e5_ranks[sel] <= 1).mean())
        m = float((mu_ranks[sel] <= 1).mean())
        print(f"  {name:22s} {len(sel):4d} [{g.min():+.2f},{g.max():+.2f}]".ljust(46)
              + f" {e:8.3f} {m:11.3f} {m - e:+7.3f}")

    # correlation: does μ rank the true folder better relative to e5 as folders get closer?
    rel = (e5_ranks - mu_ranks)                        # >0 where μ beats e5 on that query
    close = gaps < np.median(gaps)
    print(f"\n  μ-beats-e5 rate on NEAR half: {float((rel[close] > 0).mean()):.3f}  "
          f"vs FAR half: {float((rel[~close] > 0).mean()):.3f}")
    print("  (H2 predicts μ closes the gap on NEAR/confusable folders; H1 predicts no stratum helps)")


if __name__ == "__main__":
    main()
