#!/usr/bin/env python3
"""Generate wiki_lineage_v2 — lever 1 (decision scale). Source: 100k_cats
category_parent.tsv (196,901 named child->parent edges; v1 sampled 30k nodes from this
graph family). Same targets.tsv schema as v1: pos rows at hops 1..4 with target 0.85^(h-1),
2 easy negatives per node. All children with >=1 parent are used (no sampling cap) —
decisions scale ~5x. Seed-fixed walks."""
import os, sys
import numpy as np

SRC = "/home/s243a/Projects/UnifyWeaver/data/benchmark/100k_cats/category_parent.tsv"
OUT = os.path.expanduser("~/mu_data/wiki_lineage_v2")
SEED, DECAY, MAX_HOP, N_EASY = 3997001, 0.85, 4, 2

os.makedirs(OUT, mode=0o700, exist_ok=True)
parents = {}
with open(SRC, encoding="utf-8") as fh:
    next(fh)
    for ln in fh:
        c, _, p = ln.rstrip("\n").partition("\t")
        if c and p and c != p:
            parents.setdefault(c, []).append(p)
children = sorted(parents)
all_names = sorted(set(children) | {p for ps in parents.values() for p in ps})
rng = np.random.default_rng(SEED)
rows = 0
with open(os.path.join(OUT, "targets.tsv"), "w", encoding="utf-8") as out:
    # v0.4 spelling per REGISTRY_V04_MIGRATION_MANIFEST.json: 100k_cats is the
    # SimpleWiki category graph, and the structural mu-source is stated as mu=graph.
    out.write('# process_expression\tlineage(simplewiki,mu=graph,estimand="ancestry")'
              " [wiki v2, 100k_cats full, approx negatives]\n")
    out.write("# node\tancestor\ttarget\tkind\n")
    for c in children:
        node, seen = c, {c}
        for h in range(1, MAX_HOP + 1):
            ps = [p for p in parents.get(node, []) if p not in seen]
            if not ps:
                break
            node = ps[int(rng.integers(len(ps)))]
            seen.add(node)
            out.write(f"{c}\t{node}\t{round(DECAY ** (h - 1), 6)}\tpos\n")
            rows += 1
        for _ in range(N_EASY):
            e = all_names[int(rng.integers(len(all_names)))]
            if e not in seen:
                out.write(f"{c}\t{e}\t0.0\teasy\n")
                rows += 1
print(f"children {len(children)} names {len(all_names)} rows {rows} -> {OUT}/targets.tsv")
