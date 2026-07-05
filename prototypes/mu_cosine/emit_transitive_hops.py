#!/usr/bin/env python3
"""Transitive discrimination target (user 2026-07-05): make the HIER discrimination operator CONTINUOUS across
hops instead of direct-only. For an ancestor-descendant pair at h hops, target

    μ_fwd(desc | anc) = p^h          (forward membership decays with distance)
    μ_rev(anc | desc) = 1 − p^h      (direction confidence degrades toward ambiguous as h grows)

p = per-source semantic-leakage base (≈0.9 on Wikipedia; LOWER leakage ⇒ HIGHER p, cleaner structural decay).
Emitted as HIER rows tagged judge=graph (purely graph-structural). Sampled ancestor-descendant chains at exact
hop distances via BFS-up; node-disjoint train/held split so the eval measures the LEARNED decay curve.

  python3 emit_transitive_hops.py --graph .../100k_cats/category_parent.tsv --p 0.9 --hmax 5 --chains 1500 \
      --out /tmp/mu_data/transitive_train.tsv --held-out /tmp/mu_data/transitive_held.json
"""
import argparse, json, os, random, sys
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mu_attention import load_dag


def anc_by_hop(parents, start, hmax):
    seen, q, byh = {start: 0}, deque([start]), {}
    while q:
        x = q.popleft(); h = seen[x]
        if h >= hmax:
            continue
        for p in (parents.get(x) or []):
            if p not in seen:
                seen[p] = h + 1; byh.setdefault(h + 1, []).append(p); q.append(p)
    return byh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--p", type=float, default=0.9, help="per-source leakage base for μ_fwd = p^h")
    ap.add_argument("--hmax", type=int, default=5)
    ap.add_argument("--chains", type=int, default=1500)
    ap.add_argument("--held-frac", type=float, default=0.15)
    ap.add_argument("--corpus", default="enwiki")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", required=True, help="JSON of held chains for the eval")
    a = ap.parse_args()

    parents, _, _ = load_dag(a.graph)
    rng = random.Random(a.seed)
    nodes = list(parents.keys()); rng.shuffle(nodes)
    held_nodes = set(nodes[:int(a.held_frac * len(nodes))])

    train, held = {h: [] for h in range(1, a.hmax + 1)}, {h: [] for h in range(1, a.hmax + 1)}
    for s in nodes:
        byh = anc_by_hop(parents, s, a.hmax)
        if not all(h in byh for h in range(1, a.hmax + 1)):
            continue
        chain = {h: rng.choice(byh[h]) for h in range(1, a.hmax + 1)}
        bucket = held if (s in held_nodes) else train
        for h in range(1, a.hmax + 1):
            bucket[h].append((s, chain[h]))
        if sum(len(v) for v in train.values()) >= a.chains * a.hmax:
            break

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# node\troot\tmu\top\trelation\tnode_type\troot_type\tcorpus\tjudge\tconf\n")
        n = 0
        for h in range(1, a.hmax + 1):
            mf, mr = a.p ** h, 1.0 - a.p ** h
            for desc, anc in train[h]:
                f.write(f"{desc}\t{anc}\t{mf:.3f}\tHIER\tsubcategory\tcategory\tcategory\t{a.corpus}\tgraph\t1.0\n")
                f.write(f"{anc}\t{desc}\t{mr:.3f}\tHIER\tsubcategory\tcategory\tcategory\t{a.corpus}\tgraph\t1.0\n")
                n += 2
    json.dump({str(h): held[h] for h in held}, open(a.held_out, "w"))
    print(f"wrote {n} transitive HIER rows (p={a.p}, h=1..{a.hmax}, {sum(len(v) for v in train.values())} train "
          f"chains) → {a.out}; held {sum(len(v) for v in held.values())} chains → {a.held_out}")
    print("  targets μ_fwd=p^h:", {h: round(a.p ** h, 3) for h in range(1, a.hmax + 1)})


if __name__ == "__main__":
    main()
