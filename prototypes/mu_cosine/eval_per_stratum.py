#!/usr/bin/env python3
"""Per-stratum held-out SYM corr — isolate the 'more Physics' effect from the multi-domain composition.

Reproduces train_mu_attention.py's deterministic held-out split (same seed/order) on the saved
model, then reports held-out μ corr broken down by stratum (pos / pos_phys / pos_chem / cross) so the
overall +0.822 can be read honestly (chemistry pairs are tightly clustered and easier than physics)."""
import os
import random

import torch

from mu_attention import OPS, NODETYPE, load_dag, all_names, build_e5_tables, Tokenizer, MuAttention

_NT = {"category": 0, "page": 1, "mindmap_node": 2, "collection": 3, "pearltrees_collection": 3}
def nt(s):
    return _NT.get(s, 0)
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
        rel = p[5].strip() if len(p) > 5 and p[5].strip() else "subcat_of"
        at = p[6].strip() if len(p) > 6 and p[6].strip() else "category"
        bt = p[7].strip() if len(p) > 7 and p[7].strip() else "category"
        (neg if p[2] == "neg" else pos).append((p[0], p[1], float(p[4]), p[2], rel, at, bt))
    return pos, neg


def main(pairs=os.path.join(ROOT, "mu_pairs_scored_multidomain_260620-235025.tsv"),
         model_path=os.path.join(ROOT, "model_multidomain.pt"), seed=1):
    parents, children, deg = load_dag()
    names = all_names(parents, children)
    # union pair nodes so cold-start (cross-slice / page / pearltrees) endpoints aren't dropped — and so
    # the names list matches the training run's unioned e5 cache (cache hit, no re-encode).
    _pp, _nn = load_pairs_strat(pairs)
    _extra = {x for r in (_pp + _nn) for x in (r[0], r[1])}
    names = list(dict.fromkeys(list(names) + sorted(_extra - set(names))))
    q, p, idx = build_e5_tables(names, cache_path=os.environ.get("UW_E5_CACHE", os.path.join(ROOT, "e5_tables.pt")))
    tok = Tokenizer(q, p, idx, parents, deg, k=1)
    ck = torch.load(model_path, weights_only=False)
    model = MuAttention(d_model=ck["cfg"]["d_model"], n_heads=ck["cfg"]["heads"],
                        n_layers=ck["cfg"]["layers"])
    model.load_state_dict(ck["state"], strict=False)   # tolerate old checkpoints w/o nodetype_emb (zeros)
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

    # route by relation: element_of → directional μ(page|category) on the ELEM operator (a=category,
    # b=page); everything else → order-invariant SYM (avg both orders).
    pred = [0.0] * len(hold)
    sym_i = [i for i, h in enumerate(hold) if h[4] != "element_of"]
    elem_i = [i for i, h in enumerate(hold) if h[4] == "element_of"]
    if sym_i:
        ab = mu_batch(model, tok, [(hold[i][0], hold[i][1], OPS["SYM"]) for i in sym_i])
        ba = mu_batch(model, tok, [(hold[i][1], hold[i][0], OPS["SYM"]) for i in sym_i])
        for i, v in zip(sym_i, ((ab + ba) / 2).tolist()):
            pred[i] = v
    if elem_i and "ELEM" in OPS:
        ef = mu_batch(model, tok, [(hold[i][1], hold[i][0], OPS["ELEM"], None, None,
                                    nt(hold[i][6]), nt(hold[i][5])) for i in elem_i])
        for i, v in zip(elem_i, ef.tolist()):
            pred[i] = v
    tgt = [h[2] for h in hold]
    strat = [h[3] for h in hold]
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
    ap.add_argument("--pairs", default=os.path.join(ROOT, "mu_pairs_scored_multidomain_260620-235025.tsv"))
    ap.add_argument("--model", default=os.path.join(ROOT, "model_multidomain.pt"))
    a = ap.parse_args()
    main(pairs=a.pairs, model_path=a.model)
