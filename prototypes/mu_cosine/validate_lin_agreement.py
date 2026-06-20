#!/usr/bin/env python3
"""Validate that the encoder's **semantic** μ agrees with the **graph-side** `lin_from_ic` (README
step 4 — the unification `μ(X | root) == Lin(X, root)`, similarity in vector space vs the category DAG).

Faithful pure-python port of the Rust `gated_ic` / `resnik_from_ic` / `lin_from_ic`
(`boundary_cache.rs.mustache`), matching the parameters of the existing Rust test
`wikipedia_gated_similarity_tracks_physics_relatedness`:

  * sparse fixture μ (`wikipedia_physics_fuzzy_nodes.tsv`) as the membership map,
  * gate `threshold = 0.3`, denominator `total_mu = Σ μ`,
  * `gated_ic` = `−log2( min(gated_desc_mass / Σμ, 1) )`  (μ-gated descendant cone),
  * `lin(a,b) = 2·IC(MICA) / (IC(a)+IC(b))`, MICA = max finite-IC common reflexive ancestor.

Then it compares, over the scored physics nodes, the graph-side gated **Lin(a,b)** with the encoder's
**cosine(a,b)** (its pairwise semantic μ). High rank/linear correlation ⇒ the two similarity spaces
converge — the agreement the README asks for. (Pairwise is the right comparison: node-vs-root Lin is
degenerate because the root is the most-general node, IC≈0.)

    python3 validate_lin_agreement.py --encoder enc.pt          # trained encoder
    python3 validate_lin_agreement.py                            # untrained MiniLM-cosine baseline
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque
from itertools import combinations

import torch

from mu_encoder_torch import Config, MuEncoder, build_minilm_init

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")


def load_graph(path):
    """parents: child -> set(parents); children: parent -> set(children)."""
    parents, children = {}, {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            c, p = parts[0], parts[1]
            parents.setdefault(c, set()).add(p)
            children.setdefault(p, set()).add(c)
            parents.setdefault(p, parents.get(p, set()))
            children.setdefault(c, children.get(c, set()))
    return parents, children


def load_mu(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def reflexive_ancestors(u, parents):
    seen = {u}
    q = deque([u])
    while q:
        x = q.popleft()
        for p in parents.get(x, ()):
            if p not in seen:
                seen.add(p)
                q.append(p)
    return seen


def gated_desc_mass(t, children, mu, threshold):
    """μ-gated descendant cone mass for node t (faithful to descendant_mu_mass_gated): descend into a
    child only if μ(child) ≥ threshold; root always kept; diamonds counted once (seen set)."""
    seen = {t}
    q = deque([t])
    mass = mu.get(t, 0.0)
    while q:
        x = q.popleft()
        for c in children.get(x, ()):
            if c not in seen and mu.get(c, 0.0) >= threshold:
                seen.add(c)
                mass += mu.get(c, 0.0)
                q.append(c)
    return mass


def gated_ic_for(node, children, mu, threshold, total_mu, cache):
    if node in cache:
        return cache[node]
    mass = gated_desc_mass(node, children, mu, threshold)
    ic = math.inf if mass <= 0.0 else -math.log2(min(mass / total_mu, 1.0))
    cache[node] = ic
    return ic


def lin_from_ic(a, b, parents, children, mu, threshold, total_mu, cache):
    au = reflexive_ancestors(a, parents)
    av = reflexive_ancestors(b, parents)
    common = au & av
    mica = None
    for n in common:
        ic = gated_ic_for(n, children, mu, threshold, total_mu, cache)
        if math.isfinite(ic) and (mica is None or ic > mica):
            mica = ic
    if mica is None:
        return None
    ica = gated_ic_for(a, children, mu, threshold, total_mu, cache)
    icb = gated_ic_for(b, children, mu, threshold, total_mu, cache)
    denom = ica + icb
    if denom <= 0.0:
        return None
    return min(2.0 * mica / denom, 1.0)


def pearson(xs, ys):
    xs, ys = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    xs, ys = xs - xs.mean(), ys - ys.mean()
    return float((xs @ ys) / (xs.norm() * ys.norm()).clamp_min(1e-12))


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="trained shared encoder (omit = MiniLM baseline)")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max-pairs", type=int, default=4000, help="cap scored-node pairs evaluated")
    ap.add_argument("--root", default="Physics", help="anchor for the node-vs-root claim check")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    parents, children = load_graph(GRAPH)
    mu = load_mu(FIXTURE)
    total_mu = sum(mu.values())
    nodes = [n for n in mu if n in parents and mu[n] >= args.threshold]
    print(f"{len(nodes)} scored nodes at μ≥{args.threshold}; Σμ={total_mu:.2f}")

    # --- README's stated claim, directly: graph-side Lin(X, root) vs the LLM μ(X) target ---
    cache0 = {}
    xs_lin, xs_mu = [], []
    for n in nodes:
        if n == args.root:
            continue
        lin = lin_from_ic(n, args.root, parents, children, mu, args.threshold, total_mu, cache0)
        if lin is not None:
            xs_lin.append(lin)
            xs_mu.append(mu[n])
    if xs_lin:
        print(f"\nclaim check  μ(X)  vs  graph Lin(X, {args.root})  over {len(xs_lin)} scored nodes:")
        print(f"  Pearson r = {pearson(xs_mu, xs_lin):+.3f}  Spearman ρ = {spearman(xs_mu, xs_lin):+.3f}")
        sat = sum(1 for v in xs_lin if v >= 0.999)
        print(f"  ({sat}/{len(xs_lin)} node-vs-root Lin saturate at 1.0 — root is the most-general "
              f"node, so node-vs-root Lin is largely degenerate; pairwise below is the real test)")

    # graph-side gated Lin for scored-node pairs that share a finite-IC ancestor
    cache = {}
    pairs, lin_vals = [], []
    for a, b in combinations(nodes, 2):
        lin = lin_from_ic(a, b, parents, children, mu, args.threshold, total_mu, cache)
        if lin is not None:
            pairs.append((a, b))
            lin_vals.append(lin)
    print(f"{len(pairs)} pairs have a finite-IC common ancestor (graph-side Lin defined)")
    if len(pairs) > args.max_pairs:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(pairs), generator=g)[: args.max_pairs].tolist()
        pairs = [pairs[i] for i in idx]
        lin_vals = [lin_vals[i] for i in idx]
        print(f"sampled {len(pairs)} pairs")

    # semantic side: encoder cosine for the same pairs
    names = sorted({n for p in pairs for n in p})
    print("MiniLM-encoding the scored nodes ...")
    init = build_minilm_init(names)
    if args.encoder:
        ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
        cfg = Config(**ckpt["cfg"])
        model = MuEncoder(cfg, init_embeddings=init, names=names, freeze_init=True)
        model.blocks.load_state_dict(ckpt["blocks"])
        tag = f"trained encoder ({args.encoder})"
    else:
        model = MuEncoder(Config(n_layers=0, d_model=init.shape[1]), init_embeddings=init,
                          names=names, freeze_init=True)
        tag = "untrained MiniLM-cosine baseline"
    model.eval()
    a_ids = model._ids([a for a, _ in pairs])
    b_ids = model._ids([b for _, b in pairs])
    with torch.no_grad():
        cos, _ = model.mu(a_ids, b_ids)
    cos = cos.clamp(0.0, 1.0).tolist()

    print(f"\nagreement (graph-side gated Lin  vs  semantic cosine, {tag}):")
    print(f"  Pearson  r = {pearson(lin_vals, cos):+.3f}")
    print(f"  Spearman ρ = {spearman(lin_vals, cos):+.3f}   over {len(pairs)} pairs")
    ex = sorted(range(len(pairs)), key=lambda i: -lin_vals[i])[:6]
    print("  highest-Lin pairs (Lin / cos):")
    for i in ex:
        a, b = pairs[i]
        print(f"    Lin {lin_vals[i]:.2f}  cos {cos[i]:.2f}   {a} ~ {b}")


if __name__ == "__main__":
    main()
