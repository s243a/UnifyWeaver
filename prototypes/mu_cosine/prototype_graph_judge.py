#!/usr/bin/env python3
"""Prototype the GRAPH judge (DESIGN_mindmap_lineage.md §3b) on one mindmap.

For each node (the item to file) with a true structural parent p, score candidate parents by
    mu_graph = max(floor, gamma^hops(c,p) * lca_frac(c,p))
where hops = undirected graph distance, lca_frac = shared-prefix depth of lineage(c) vs lineage(p) over
len(lineage(p)) -- i.e. "how much of the TRUTH's ancestry the candidate shares" (whole-lineage structural
factor). Truth (c==p) -> hops 0, lca_frac 1 -> mu 1.

Candidate pool = NEGATIVES only: excludes n, its descendants, AND its ancestors (ancestors are graded
POSITIVES, not negatives -- review round 2). Unreachable candidates get hops = D_max+1 (graceful fallback
for a single-map parse, not a sentinel) so lca_frac still differentiates them by shared ancestry.

Eyeball: true parent should be #1 at mu=1; ancestors excluded; near subtrees moderate; far -> floor.

  python3 prototype_graph_judge.py --map "context/Chaos theory.smmx"
"""
import argparse, os, random, sys
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_mindmap_lineage import parse_map, lineage


def undirected_adj(parent):
    adj = defaultdict(set)
    for c, p in parent.items():
        adj[c].add(p); adj[p].add(c)
    return adj


def bfs_dist(adj, src):
    dist, q = {src: 0}, deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def descendants(node, children):
    seen, stack = set(), [node]
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c); stack.append(c)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=os.path.join(os.path.dirname(__file__), "context", "Chaos theory.smmx"))
    ap.add_argument("--gamma", type=float, default=0.6, help="decay base gamma^hops")
    ap.add_argument("--floor", type=float, default=0.02, help="mu floor (shares-only-root, never 0)")
    ap.add_argument("--k", type=int, default=6, help="sampled candidate parents per node (besides the true one)")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--tmp", default="/tmp/mu_data/mm_parse")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    rng = random.Random(0)

    title, parent, fallback, err = parse_map(a.map, a.tmp)
    if not title:
        print("parse failed:", err[:120]); return
    children = defaultdict(set)
    for c, p in parent.items():
        children[p].add(c)

    adj = undirected_adj(parent)                              # build ONCE (review: was per-node O(N^2))
    D_max = max((len(lineage(n, parent)) for n in title), default=1)
    dist_cache = {}                                           # bfs from each true parent, cached by key

    filed = [n for n in title if n in parent]
    true_is_top, true_mu1, shown, all_mu, unreachable = 0, 0, 0, [], 0
    for n in filed:
        p = parent[n]
        lin_p = lineage(p, parent)
        if p not in dist_cache:
            dist_cache[p] = bfs_dist(adj, p)
        dist = dist_cache[p]
        ancestors = set(lineage(n, parent)[:-1])              # graded POSITIVES, not negatives -> exclude
        forbidden = descendants(n, children) | ancestors | {n}
        pool = [m for m in title if m not in forbidden]
        cands = [p] + rng.sample(pool, min(a.k, len(pool)))
        scored = []
        for c in cands:
            hops = dist.get(c, D_max + 1)                     # graceful fallback, not a sentinel
            if c not in dist:
                unreachable += 1
            frac = common_prefix_len(lineage(c, parent), lin_p) / max(1, len(lin_p))
            mu = max(a.floor, (a.gamma ** hops) * frac)
            scored.append((mu, hops, round(frac, 2), c))
        scored.sort(key=lambda x: -x[0])
        true_mu = next(m for m, h, f, c in scored if c == p)
        all_mu.append(true_mu)
        if scored[0][3] == p:
            true_is_top += 1
        if abs(true_mu - 1.0) < 1e-9:
            true_mu1 += 1
        if shown < a.show:
            print(f"\nfile '{title[n]}'  (true parent: '{title[p]}')")
            for mu, hops, frac, c in scored:
                print(f"    mu={mu:.3f}  hops={hops:>2}  lcaFrac={frac:<4}  {title.get(c, c)}"
                      + ("   <- TRUE" if c == p else ""))
            shown += 1

    N = len(filed)
    print(f"\n=== summary ({N} filable nodes, gamma={a.gamma}, floor={a.floor}, D_max={D_max}) ===")
    print(f"  true parent ranked #1: {true_is_top}/{N} ({100*true_is_top/max(1,N):.0f}%)")
    print(f"  true parent mu == 1.0: {true_mu1}/{N}")
    print(f"  mean/min true-parent mu: {sum(all_mu)/max(1,N):.3f} / {min(all_mu):.3f}")
    print(f"  unreachable candidates (single-map fragmentation, hops=D_max+1): {unreachable}")


if __name__ == "__main__":
    main()
