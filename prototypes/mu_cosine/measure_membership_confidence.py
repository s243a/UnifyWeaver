#!/usr/bin/env python3
"""ELEM follow-up, step 0 — MEASURE the membership confidences before building.

The precision graph judge weights each membership signal by its reliability c = corr(signal, SYM judge)
(DESIGN_sym_dual_judge.md "Confidence architecture"). v1 used a subcategory-only up_hops proxy (c_mem +0.669).
Here we score the MODEL's own membership readouts on model_prod.pt and correlate each with the SYM judge target:

    c_subcat = corr( HIER-membership , SYM ),    c_elem = corr( ELEM-membership , SYM )

membership(op) = combine( μ_op(a|b), μ_op(b|a) )   (max = "member in either direction"; mean also reported).

Decision (user 2026-07-04): confidence is per-OPERATOR, shared across fwd/bwd. If c_subcat ≈ c_elem ⇒ collapse to
one combined c_mem (simpler, lose nothing); if they diverge ⇒ keep them separate in the precision fusion.

  python3 measure_membership_confidence.py --ckpt model_prod.pt --graph ../../data/benchmark/100k_cats/category_parent.tsv
"""
import argparse, os, sys
from collections import deque
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_relatedness import build_model, score_pairs


def load_pairs_with_target(path, cap):
    rows = []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) < 5:
            continue
        try:
            mu = float(c[4])
        except ValueError:
            continue
        rows.append((c[0], c[1], mu))          # (node, root, SYM target)
    return rows[:cap] if cap else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model_prod.pt")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--pairs", default="mu_pairs_scored_cumulative.tsv")
    ap.add_argument("--cap-pairs", type=int, default=2500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    parents, children, deg = load_dag(a.graph)
    rows = load_pairs_with_target(a.pairs, a.cap_pairs)
    print(f"pairs (before ancestor closure): {len(rows)}")

    # names = pair endpoints + their ancestors (the tokenizer references them), like fit_sym_from_graph
    def anc_of(nm, capd=10):
        out, frontier, steps = set(), {nm}, 0
        while frontier and steps < capd:
            nxt = set()
            for cur in frontier:
                for pp in (parents.get(cur) or []):
                    if pp not in out and pp != nm:
                        out.add(pp); nxt.add(pp)
            frontier = nxt; steps += 1
        return out
    names = {x for r in rows for x in r[:2]}
    for nm in list(names):
        names.update(anc_of(nm))
    names = sorted(names)
    q, p, idx = build_e5_tables(names, cache_path="/tmp/mu_data/elem_conf_e5.pt", device=str(dev))
    # keep only pairs whose endpoints embedded
    rows = [r for r in rows if r[0] in idx and r[1] in idx]
    print(f"pairs scored: {len(rows)}  ({len(names)} names embedded)")
    tok = Tokenizer(q, p, idx, parents, deg, k=8, beta=1.0, max_anc=8)

    model = build_model(a.ckpt, dev)
    n_ops = model.op_emb.num_embeddings
    print(f"model ops: {n_ops}  (HIER={OPS['HIER']}, ELEM={OPS['ELEM']})")

    def readout(op):
        ow = torch.zeros(1, n_ops); ow[0, OPS[op]] = 1.0
        fwd = np.array(score_pairs(model, tok, idx, [(x, y) for x, y, _ in rows], ow, dev))
        bwd = np.array(score_pairs(model, tok, idx, [(y, x) for x, y, _ in rows], ow, dev))
        return fwd, bwd

    tgt = np.array([m for _, _, m in rows])
    print(f"\nSYM target: mean {tgt.mean():.3f}  std {tgt.std():.3f}\n")
    results = {}
    for op, label in [("HIER", "subcat"), ("ELEM", "elem")]:
        fwd, bwd = readout(op)
        for combine, name in [(np.maximum(fwd, bwd), "max"), ((fwd + bwd) / 2, "mean")]:
            c = float(np.corrcoef(combine, tgt)[0, 1])
            results[(label, name)] = c
            print(f"  c_{label:6s} ({name} fwd/bwd) = corr(μ_{op}, SYM) = {c:+.3f}   "
                  f"(signal mean {combine.mean():.3f})")
    # decision
    cs, ce = results[("subcat", "max")], results[("elem", "max")]
    gap = abs(cs - ce)
    print(f"\n  c_subcat {cs:+.3f}  vs  c_elem {ce:+.3f}   |gap| {gap:.3f}")
    print(f"  → {'COLLAPSE to one combined c_mem (close)' if gap < 0.10 else 'KEEP SEPARATE c_subcat / c_elem (diverge)'}")
    print(f"  (v1 up_hops-proxy subcat c_mem was +0.669 — compare the model-HIER readout above)")


if __name__ == "__main__":
    main()
