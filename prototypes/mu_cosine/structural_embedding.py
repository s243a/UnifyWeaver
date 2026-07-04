#!/usr/bin/env python3
"""Learned STRUCTURAL (metric) embedding — the cheap O(1)-inference proxy for the graph-judge `3/dist`
(DESIGN_sym_dual_judge.md step 2). node2vec/gensim unavailable, so train per-node vectors in torch so that
    ‖emb(a) − emb(b)‖  ≈  graph_dist(a, b)
on BFS-sampled distance-labeled pairs. At inference, `3/‖emb(a)−emb(b)‖` replaces the BFS distance (embedding
lookup + a subtraction — no traversal). Saves {node: vec} for the fit diagnostic / the SYM blend.

  python3 structural_embedding.py --graph ../../data/benchmark/100k_cats/category_parent.tsv --dim 32 --out /tmp/mu_data/struct_emb.pt
"""
import argparse, os, random, sys
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import load_dag


def undirected_adj(parents, children):
    adj = defaultdict(set)
    for c, ps in parents.items():
        for p in (ps if isinstance(ps, (set, list, tuple)) else [ps]):
            adj[c].add(p); adj[p].add(c)
    for p, cs in children.items():
        for c in (cs if isinstance(cs, (set, list, tuple)) else [cs]):
            adj[c].add(p); adj[p].add(c)
    return adj


def sample_pairs(adj, sources, cap, rng):
    """BFS from each source → (src, node, dist) for dist 1..cap (all reachable within cap)."""
    out = []
    for s in sources:
        dist, q = {s: 0}, deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= cap:
                continue
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        for n, d in dist.items():
            if d > 0:
                out.append((s, n, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--cap", type=int, default=6, help="max BFS distance used as a training label")
    ap.add_argument("--sources", type=int, default=4000, help="BFS source nodes (sampled)")
    ap.add_argument("--far-frac", type=float, default=0.3, help="fraction of far random negatives (dist>cap ⇒ cap+2)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/tmp/mu_data/struct_emb.pt")
    a = ap.parse_args()
    dev = torch.device(a.device); rng = random.Random(0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    parents, children, deg = load_dag(a.graph)
    adj = undirected_adj(parents, children)
    nodes = sorted(adj)
    nid = {n: i for i, n in enumerate(nodes)}
    print(f"graph: {len(nodes)} nodes, {sum(len(v) for v in adj.values())//2} undirected edges")

    srcs = rng.sample(nodes, min(a.sources, len(nodes)))
    pairs = sample_pairs(adj, srcs, a.cap, rng)
    # far negatives: random pairs (mostly disconnected/far) labelled dist = cap+2
    nfar = int(len(pairs) * a.far_frac)
    for _ in range(nfar):
        pairs.append((rng.choice(nodes), rng.choice(nodes), a.cap + 2))
    rng.shuffle(pairs)
    A = torch.tensor([nid[s] for s, _, _ in pairs], device=dev)
    B = torch.tensor([nid[n] for _, n, _ in pairs], device=dev)
    D = torch.tensor([float(d) for _, _, d in pairs], device=dev)
    print(f"training pairs: {len(pairs)} ({nfar} far negatives)  dist range 1..{a.cap+2}")

    emb = nn.Embedding(len(nodes), a.dim).to(dev)
    nn.init.normal_(emb.weight, std=0.1)
    opt = torch.optim.Adam(emb.parameters(), lr=a.lr)
    for step in range(a.steps):
        idx = torch.randint(0, len(pairs), (a.bs,), device=dev)
        ea, eb, d = emb(A[idx]), emb(B[idx]), D[idx]
        pred = 3.0 / (1.0 + (ea - eb).norm(dim=1))          # closeness 3/(1+‖Δ‖) — matches inference use
        target = 3.0 / d                                     # RECIPROCAL target: far pairs (small 3/d) weigh less
        loss = ((pred - target) ** 2).mean()                 # so the embedding spends capacity on the near scale
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                corr = float(np.corrcoef(pred.cpu().numpy(), target.cpu().numpy())[0, 1])
            print(f"  step {step+1:5d}  MSE {float(loss):.4f}  pred-vs-(3/d) corr {corr:+.3f}")

    W = emb.weight.detach().cpu()
    torch.save({"nodes": nodes, "emb": W, "dim": a.dim, "cap": a.cap}, a.out)
    print(f"saved {len(nodes)}×{a.dim} structural embedding → {a.out}")


if __name__ == "__main__":
    main()
