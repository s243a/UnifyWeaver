#!/usr/bin/env python3
"""Prefix ablation (PR #3387 / architecture study): does directional membership need e5's query:/passage:
prefix, or is the signal in the RAW embedding (the per-node generality axis)?

Trains an order-aware logistic probe on directional edges two ways and compares held-out direction AUC:
  prefixed   : concat(e5[query: a], e5[passage: b])   — the role-prefix geometry (what μ uses today)
  no-prefix  : concat(e5[a],         e5[b])            — same plain embedding for both slots, no role prefix

If no-prefix ≈ prefixed, the directional signal is in the raw embedding (generality), and the prefix is optional;
if no-prefix collapses toward 0.5, the prefix is the carrier. Node-disjoint split (eval_arch_control.build_triples).

  python3 eval_prefix_ablation.py --graph /tmp/merged_category_parent.tsv
"""
import argparse, torch
from sentence_transformers import SentenceTransformer
from mu_attention import load_dag, E5_MODEL
from eval_arch_control import build_triples, train_logistic, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True); ap.add_argument("--n-edges", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    parents, children, deg = load_dag(a.graph)
    tr, te = build_triples(parents, children, a.n_edges, a.seed, node_disjoint=True)
    names = sorted({x for t in tr + te for x in t})
    txt = {n: n.replace('_', ' ') for n in names}
    m = SentenceTransformer(E5_MODEL, device=a.device)
    enc = lambda pre: dict(zip(names, m.encode([f"{pre}{txt[n]}" for n in names], normalize_embeddings=True, show_progress_bar=False)))
    q, p, e = enc("query: "), enc("passage: "), enc("")     # prefixed query/passage, and no-prefix plain
    import numpy as np
    fp = lambda a_, b_: np.concatenate([q[a_], p[b_]]).tolist()    # prefixed ordered feature
    fn = lambda a_, b_: np.concatenate([e[a_], e[b_]]).tolist()    # no-prefix ordered feature
    cosp = lambda prs: [float(np.dot(q[x], p[y])) for x, y in prs]
    cosn = lambda prs: [float(np.dot(e[x], e[y])) for x, y in prs]
    pos, neg = [(c, pp) for c, pp, s in te], [(pp, c) for c, pp, s in te]   # forward vs reverse (DIRECTION)
    print(f"[DATA] node-disjoint: {len(tr)} train / {len(te)} held-out edges\n")
    print(f"{'feature':18} {'DIRECTION AUC (held-out)':>24}")
    for nm, feat, cos in [("e5-cos prefixed", None, cosp), ("e5-cos no-prefix", None, cosn),
                          ("probe prefixed", fp, None), ("probe no-prefix", fn, None)]:
        if cos:
            print(f"{nm:18} {auc(cos(pos), cos(neg)):24.3f}")
        else:
            X = [feat(c, pp) for c, pp, s in tr] + [feat(pp, c) for c, pp, s in tr]
            w, b = train_logistic(X, [1.0] * len(tr) + [0.0] * len(tr), dev)
            sc = lambda prs: (torch.tensor([feat(x, y) for x, y in prs], device=dev) @ w + b).cpu().tolist()
            print(f"{nm:18} {auc(sc(pos), sc(neg)):24.3f}")
    print("\n  read: probe no-prefix ≈ prefixed ⇒ direction lives in the RAW embedding (generality), prefix optional.")


if __name__ == "__main__":
    main()
