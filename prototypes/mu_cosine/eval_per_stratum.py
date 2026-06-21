#!/usr/bin/env python3
"""Per-stratum held-out SYM corr — isolate the 'more Physics' effect from the multi-domain composition.

Reproduces train_mu_attention.py's deterministic held-out split (same seed/order) on the saved
model, then reports held-out μ corr broken down by stratum (pos / pos_phys / pos_chem / cross) so the
overall +0.822 can be read honestly (chemistry pairs are tightly clustered and easier than physics)."""
import os
import random

import torch

from mu_attention import OPS, load_dag, all_names, build_e5_tables, Tokenizer, MuAttention
from train_mu_attention import load_edges, load_mu, pearson, mu_batch
from gen_mu_pairs import GRAPH

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_pairs_strat(path):
    pos, neg = [], []
    for l in open(path):
        if l.lstrip().startswith("#"):
            continue
        p = l.rstrip("\n").split("\t")
        if len(p) < 5 or p[4].strip() == "":
            continue
        (neg if p[2] == "neg" else pos).append((p[0], p[1], float(p[4]), p[2]))
    return pos, neg


def main(pairs=os.path.join(ROOT, "mu_pairs_scored_multidomain.tsv"),
         model_path=os.path.join(ROOT, "model_multidomain.pt"), seed=1):
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    q, p, idx = build_e5_tables(names, cache_path=os.path.join(ROOT, "e5_tables.pt"))
    tok = Tokenizer(q, p, idx, parents, deg, k=1)
    ck = torch.load(model_path, weights_only=False)
    model = MuAttention(d_model=ck["cfg"]["d_model"], n_heads=ck["cfg"]["heads"],
                        n_layers=ck["cfg"]["layers"])
    model.load_state_dict(ck["state"])
    model.eval()

    # reproduce the split EXACTLY (rng consumes the edge shuffle first, then the pos shuffle)
    rng = random.Random(seed)
    edges = [e for e in load_edges() if e[0] in idx and e[1] in idx]
    rng.shuffle(edges)
    pos, neg = load_pairs_strat(pairs)
    pos = [r for r in pos if r[0] in idx and r[1] in idx]
    rng.shuffle(pos)
    n_ph = int(0.2 * len(pos))
    hold = pos[:n_ph]

    ab = mu_batch(model, tok, [(a, b, OPS["SYM"]) for a, b, _, _ in hold])
    ba = mu_batch(model, tok, [(b, a, OPS["SYM"]) for a, b, _, _ in hold])
    pred = ((ab + ba) / 2).tolist()
    tgt = [m for _, _, m, _ in hold]
    strat = [s for _, _, _, s in hold]
    print(f"held-out positives: {len(hold)}  overall corr {pearson(pred, tgt):+.3f}")
    # cross_* strata pooled together too (each alone is small); then every stratum present
    cross_idx = [i for i, st in enumerate(strat) if st.startswith("cross")]
    if len(cross_idx) > 2:
        print(f"  {'cross_ALL':12} n={len(cross_idx):3}  corr "
              f"{pearson([pred[i] for i in cross_idx], [tgt[i] for i in cross_idx]):+.3f}  "
              f"(μ̄ target {sum(tgt[i] for i in cross_idx)/len(cross_idx):.2f})")
    for s in sorted(set(strat)):
        idxs = [i for i, st in enumerate(strat) if st == s]
        if len(idxs) > 2:
            c = pearson([pred[i] for i in idxs], [tgt[i] for i in idxs])
            print(f"  {s:12} n={len(idxs):3}  corr {c:+.3f}  "
                  f"(μ̄ target {sum(tgt[i] for i in idxs)/len(idxs):.2f})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(ROOT, "mu_pairs_scored_multidomain.tsv"))
    ap.add_argument("--model", default=os.path.join(ROOT, "model_multidomain.pt"))
    a = ap.parse_args()
    main(pairs=a.pairs, model_path=a.model)
