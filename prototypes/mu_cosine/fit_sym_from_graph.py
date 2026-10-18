#!/usr/bin/env python3
"""Is SYM derivable from the directionals + graph distance? (the 'graph judge for SYM').

Fit, on the judge-scored SYM positives:
    μ_sym ≈ σ( w0 + w1·decay(dist) + w2·μ(a|b) + w3·μ(b|a) )        (logistic; residual intuition: w1>0, w2,w3<0)
- High R² ⇒ SYM is derivable → teacher/student (distill into the SYM op, e5-only at inference), regression moot.
- Low  R² ⇒ the judge (30% superposition scoring) genuinely earns its keep.

  python3 fit_sym_from_graph.py --ckpt model_prod.pt --graph ../../data/benchmark/100k_cats/category_parent.tsv
"""
import argparse, math, os, sys
from collections import deque
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import build_e5_tables, Tokenizer, OPS, load_dag
from eval_relatedness import build_model, score_pairs


def load_sym_positives(path, cap):
    pos = []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) < 5:
            continue
        a, b, stratum, mu = c[0], c[1], c[2], c[4]
        try:
            mu = float(mu)
        except ValueError:
            continue
        if stratum != "neg" and mu > 0:                 # SYM positives
            pos.append((a, b, mu))
    return pos[:cap] if cap else pos


def undirected_adj(parents, children):
    adj = {}
    for c, ps in parents.items():
        for p in (ps if isinstance(ps, (set, list, tuple)) else [ps]):
            adj.setdefault(c, set()).add(p); adj.setdefault(p, set()).add(c)
    return adj


def bfs_from(adj, src, wanted, cap):
    dist, q = {src: 0}, deque([src])
    need = set(wanted)
    while q and need:
        u = q.popleft()
        if dist[u] >= cap:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v); need.discard(v)
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model_prod.pt")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--pairs", default="mu_pairs_scored_cumulative.tsv")
    ap.add_argument("--cap-pairs", type=int, default=1500)
    ap.add_argument("--bfs-cap", type=int, default=10)
    ap.add_argument("--struct-emb", default=None, help="learned structural embedding (structural_embedding.py .pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)

    parents, children, deg = load_dag(a.graph)
    adj = undirected_adj(parents, children)
    present = set(parents) | set(children)
    for v in children.values():
        present.update(v if isinstance(v, (set, list, tuple)) else [v])

    sym = load_sym_positives(a.pairs, a.cap_pairs)
    sym = [(x, y, m) for x, y, m in sym if x in present and y in present]
    print(f"SYM positives with both nodes in graph: {len(sym)}")
    if len(sym) < 30:
        print("too few in-graph pairs — try --graph enwiki_named/category_parent.tsv"); return

    # model directionals μ(a|b), μ(b|a) via the HIER operator
    model = build_model(a.ckpt, dev)

    def anc_of(nm, capd=12):                              # ALL ancestors (multi-parent BFS-up) the tokenizer refs
        out, frontier, steps = set(), {nm}, 0
        while frontier and steps < capd:
            nxt = set()
            for cur in frontier:
                ps = parents.get(cur)
                if not ps:
                    continue
                for pp in (ps if isinstance(ps, (set, list, tuple)) else [ps]):
                    if pp not in out and pp != nm:
                        out.add(pp); nxt.add(pp)
            frontier = nxt; steps += 1
        return out
    allnames = {x for pr in sym for x in pr[:2]}
    for nm in list(allnames):
        allnames.update(anc_of(nm))
    names = sorted(allnames)
    q, p, idx = build_e5_tables(names, cache_path="/tmp/mu_data/sym_fit_e5.pt", device=str(dev))
    tok = Tokenizer(q, p, idx, parents, deg, k=8, beta=1.0, max_anc=8)
    n_ops = model.op_emb.num_embeddings
    ow = torch.zeros(1, n_ops); ow[0, OPS["HIER"]] = 1.0
    mu_ab = score_pairs(model, tok, idx, [(x, y) for x, y, _ in sym], ow, dev)
    mu_ba = score_pairs(model, tok, idx, [(y, x) for x, y, _ in sym], ow, dev)
    ow_sym = torch.zeros(1, n_ops); ow_sym[0, OPS["SYM"]] = 1.0          # the e5-semantic SYM judge
    mu_sym_model = np.array(score_pairs(model, tok, idx, [(x, y) for x, y, _ in sym], ow_sym, dev))

    # graph distances (BFS per unique source)
    by_src = {}
    for x, y, _ in sym:
        by_src.setdefault(x, []).append(y)
    dist_of = {}
    for s, ys in by_src.items():
        d = bfs_from(adj, s, ys, a.bfs_cap)
        for y in ys:
            dist_of[(s, y)] = d.get(y, a.bfs_cap + 1)

    dists = np.array([dist_of[(x, y)] for x, y, _ in sym], float)
    reach = dists <= a.bfs_cap
    recip = 3.0 / np.maximum(dists, 1.0)                  # 3/distance — inverse-radial (1/r); the graph-judge input
    mab, mba = np.array(mu_ab), np.array(mu_ba)
    X = np.column_stack([np.ones(len(sym)), recip, mab, mba])
    y = np.clip(np.array([m for _, _, m in sym], float), 1e-4, 1 - 1e-4)
    z = np.log(y / (1 - y))                               # logit target — linear least squares in logit space

    w, *_ = np.linalg.lstsq(X, z, rcond=None)
    zp = X @ w
    yp = 1 / (1 + np.exp(-zp))
    ss_res = float(((y - yp) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    corr = float(np.corrcoef(yp, y)[0, 1])
    print(f"\nlogistic fit  μ_sym ≈ σ(w0 + w1·(3/dist) + w2·μ(a|b) + w3·μ(b|a)):")
    print(f"  weights: w0={w[0]:+.3f}  w1·(3/dist)={w[1]:+.3f}  w2·μ(a|b)={w[2]:+.3f}  w3·μ(b|a)={w[3]:+.3f}")
    print(f"  R² {r2:+.3f}   pred-vs-target corr {corr:+.3f}   ({len(sym)} pairs, {reach.mean()*100:.0f}% reachable)")
    print(f"  mean μ_sym target {y.mean():.3f}  |  mean μ(a|b) {np.mean(mab):.3f}  μ(b|a) {np.mean(mba):.3f}")
    # FIXED residual form (no fitted weights): μ_sym ≈ σ(3/dist − μab − μba)
    fixed = 1 / (1 + np.exp(-(recip - mab - mba)))
    print(f"  [FIXED σ(3/dist − μab − μba)]  corr {np.corrcoef(fixed, y)[0,1]:+.3f}  "
          f"R² {1 - ((y-fixed)**2).sum()/ss_tot:+.3f}")
    # STRUCTURAL EMBEDDING (learned metric emb) — the cheap O(1)-inference proxy: 3/‖emb(a)−emb(b)‖
    se = torch.load(a.struct_emb)
    svmap = {n: v for n, v in zip(se["nodes"], se["emb"].numpy())}
    cap2 = se.get("cap", 6) + 2
    def sdist(x, y):
        return float(np.linalg.norm(svmap[x] - svmap[y])) if (x in svmap and y in svmap) else cap2
    struct_feat = 3.0 / (1.0 + np.array([sdist(x, y) for x, y, _ in sym]))
    print(f"  struct-embed (learned {se['dim']}d) vs true 3/dist corr: {np.corrcoef(struct_feat, recip)[0, 1]:+.3f}")

    print(f"  --- dual judge (superposition test) ---")
    for name, feats in [("graph only  (3/dist true)", [recip]),
                        ("e5-SYM only (operator)", [mu_sym_model]),
                        ("DUAL  (3/dist true + e5-SYM)", [recip, mu_sym_model]),
                        ("struct-embed only", [struct_feat]),
                        ("DUAL  (struct-embed + e5-SYM)", [struct_feat, mu_sym_model])]:
        Xd = np.column_stack([np.ones(len(sym))] + feats)
        wd, *_ = np.linalg.lstsq(Xd, z, rcond=None)
        ypd = 1 / (1 + np.exp(-(Xd @ wd)))
        print(f"    [{name:29s}] R² {1 - ((y-ypd)**2).sum()/ss_tot:+.3f}  corr {np.corrcoef(ypd, y)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
