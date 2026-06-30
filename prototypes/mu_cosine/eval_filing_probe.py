#!/usr/bin/env python3
"""e5-probe baseline for the filing learning curve (PR #3387 review #5).

The filing curve (train_filing.py) showed μ crossing e5-*cos*. The review asked for the *architecture* control —
a TRAINED head on frozen e5 — at each data fraction, so the data-scaling claim isn't μ vs an untrained score.
Here: at each fraction, train a logistic probe on `concat(query: bookmark, passage: folder)` with contrastive
negatives, on the same train bookmarks, eval MRR on the SAME fixed held-out set (mirrors train_filing's split).
Print alongside the known μ curve (0.230/0.317/0.358) and the flat e5-cos bar (0.291).

  python3 eval_filing_probe.py --trees ../../.local/data/pearltrees_api/trees --fracs 0.1,0.3,1.0
"""
import argparse, collections, random
import torch
from mu_attention import build_e5_tables
from eval_filing import load_filing, metrics
from eval_arch_control import train_logistic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", required=True); ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--fracs", default="0.1,0.3,1.0"); ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--max-eval", type=int, default=400); ap.add_argument("--neg", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="/tmp/filingprobe_e5.pt")
    a = ap.parse_args(); dev = torch.device(a.device); rng = random.Random(a.seed)

    queries, cand = load_filing(a.trees, a.min_bm)
    byf = collections.defaultdict(list)
    for q in queries:
        byf[q[1]].append(q)
    eval_q, pool = [], []
    for fid, qs in byf.items():
        qs = qs[:]; rng.shuffle(qs)
        k = max(1, int(a.eval_frac * len(qs))) if len(qs) >= 2 else 0
        eval_q += qs[:k]; pool += qs[k:]
    rng.shuffle(eval_q); eval_q = eval_q[:a.max_eval]

    bm_list = eval_q + pool
    f_ids = list(cand)
    names = [f"F:{t}" for t in f_ids] + [f"B:{i}" for i in range(len(bm_list))]
    texts = {**{f"F:{t}": cand[t] for t in cand}, **{f"B:{i}": bm_list[i][0] for i in range(len(bm_list))}}
    qt, pt, idx = build_e5_tables(names, cache_path=a.cache, texts=texts, device=a.device)
    qF = pt[[idx[f"F:{t}"] for t in f_ids]]                              # folders as passage (the container slot)
    f_pos = {t: i for i, t in enumerate(f_ids)}
    eval_keys = list(range(len(eval_q))); train_pool = list(range(len(eval_q), len(bm_list)))
    eval_true = [f_pos[bm_list[i][1]] for i in eval_keys]

    def feat(bm_i, folder_row):                                          # concat + ELEMENT-WISE PRODUCT so the
        b_, f_ = qt[idx[f"B:{bm_i}"]], qF[folder_row]                    # linear head can compute a bilinear
        return torch.cat([b_, f_, b_ * f_]).tolist()                     # (≈ cosine) match — fair filing baseline

    print(f"[DATA] {len(f_ids)} folders, {len(train_pool)} train-pool, {len(eval_q)} fixed eval")
    print(f"\n  reference: e5-cos bar 0.291 (flat); μ curve 0.10→0.230  0.30→0.317  1.00→0.358\n")
    print(f"  {'frac':>6} {'n_train':>8} {'e5-probe MRR':>13} {'recall@10':>10}")
    for fr in [float(x) for x in a.fracs.split(",")]:
        n = min(max(64, int(fr * len(train_pool))), len(train_pool))
        sub = random.Random(a.seed + 1).sample(train_pool, n)
        X, y = [], []
        for bi in sub:                                                   # 1 positive + K contrastive negatives
            tf = f_pos[bm_list[bi][1]]
            X.append(feat(bi, tf)); y.append(1.0)
            for nf in random.Random(bi).sample(range(len(f_ids)), a.neg + 1):
                if nf != tf:
                    X.append(feat(bi, nf)); y.append(0.0)
        w, b = train_logistic(X, y, dev, steps=500)
        # eval: score every (eval bm, folder), rank the true folder
        ranks = []
        for r, bi in enumerate(eval_keys):
            Xe = torch.tensor([feat(bi, j) for j in range(len(f_ids))], device=dev)
            sc = (Xe @ w + b).cpu()
            ranks.append(1 + int((sc > sc[eval_true[r]]).sum().item()))
        m = metrics(ranks)
        print(f"  {fr:6.2f} {n:8d} {m['MRR']:13.3f} {m['recall@10']:10.3f}")
    print("\n  → compare the e5-probe curve to μ's: does a trained e5 head also cross 0.291, and is μ above it?")


if __name__ == "__main__":
    main()
