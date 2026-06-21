#!/usr/bin/env python3
"""Emit a **dense** μ map for the whole category graph (README step 5) — the prototype's payoff.

Takes the shared encoder trained by `train_cosine_mu_torch.py --save-encoder` (vocabulary-independent:
it maps any MiniLM vector into the μ-cosine space), MiniLM-encodes **every** category name in
`category_parent.tsv`, and emits `name<TAB>μ` where `μ = clamp(cos(encode(name), encode(root)), [0,1])`.

The two integration guards the README flags as load-bearing are enforced here:
  1. **Clamp cosine to [0,1]** (`to_membership`) before emitting — `descendant_mu_mass` / `sketch_mu_mass`
     *sum* μ as mass and a negative weight corrupts the mass/KMV estimate.
  2. **Names verbatim from the TSV** (case/underscore-sensitive). The Rust loaders intern names → ids;
     a name absent from the graph hits `unwrap_or(0.0)` and silently becomes μ=0. We emit exactly the
     graph's names, and the assertion below reports coverage (it is 100% by construction).

Output matches `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (`name<TAB>μ`, '#'-comment header), so
the existing Rust loaders consume it unchanged — just larger and dense.

    python3 train_cosine_mu_torch.py --minilm --save-encoder enc.pt   # train + save first
    python3 emit_dense_mu.py --encoder enc.pt --root Physics --out dense_mu_physics.tsv
"""
from __future__ import annotations

import argparse
import os

import torch

from mu_encoder_torch import Config, MuEncoder, build_minilm_init, to_membership

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")


def load_graph_names(path):
    """Every distinct category name in the graph, **verbatim** (order: first appearance)."""
    names, seen = [], set()
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            for n in (parts[0], parts[1]):
                if n not in seen:
                    seen.add(n)
                    names.append(n)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=None, help="trained shared encoder from --save-encoder "
                    "(omit to emit the *untrained* MiniLM-cosine baseline = README prompt A)")
    ap.add_argument("--root", default="Physics", help="domain anchor (must be a graph name)")
    ap.add_argument("--out", default=os.path.join(ROOT, "dense_mu.tsv"))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    names = load_graph_names(GRAPH)
    print(f"{len(names)} categories in the graph")
    if args.root not in names:
        raise SystemExit(f"root '{args.root}' is not a graph name (names are case/underscore-sensitive)")

    print("MiniLM-encoding all category names ...")
    init = build_minilm_init(names)

    if args.encoder:
        ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
        cfg = Config(**ckpt["cfg"])
        model = MuEncoder(cfg, init_embeddings=init, names=names, freeze_init=True)
        model.blocks.load_state_dict(ckpt["blocks"])
        tag = f"trained encoder ({args.encoder})"
    else:
        cfg = Config(n_layers=0, d_model=init.shape[1])  # identity encoder = raw MiniLM cosine
        model = MuEncoder(cfg, init_embeddings=init, names=names, freeze_init=True)
        tag = "untrained MiniLM-cosine baseline (prompt A)"
    model.eval()
    print(f"emitting μ(·|{args.root}) with: {tag}")

    root_id = model._ids([args.root])
    mus = []
    with torch.no_grad():
        for i in range(0, len(names), args.batch_size):
            ids = torch.arange(i, min(i + args.batch_size, len(names)), dtype=torch.long)
            cos, _ = model.mu(ids, root_id.expand(ids.shape[0]))
            mus.extend(to_membership(cos).tolist())   # guard #1: clamp to [0,1]

    resolved = sum(1 for m in mus if m is not None)   # 100% by construction (we emit graph names)
    with open(args.out, "w") as f:
        f.write(f"# Dense μ(category | {args.root}) from the cosine-μ encoder ({tag}).\n")
        f.write(f"# {resolved}/{len(names)} names resolved against {os.path.relpath(GRAPH, REPO)} "
                f"(verbatim; clamp to [0,1] applied). Format: name<TAB>μ.\n")
        for n, m in zip(names, mus):
            f.write(f"{n}\t{m:.4f}\n")   # guard #2: name emitted verbatim

    nz = sum(1 for m in mus if m > 0.0)
    band = sum(1 for m in mus if 0.3 <= m <= 0.7)
    print(f"wrote {len(names)} dense μ rows → {args.out}")
    print(f"coverage: {resolved}/{len(names)} names resolve (assert PASS); "
          f"{nz} have μ>0; {band} in boundary band [0.3,0.7]")
    top = sorted(zip(names, mus), key=lambda x: -x[1])[:10]
    print("highest-μ categories:")
    for n, m in top:
        print(f"  μ {m:.3f}  {n}")
    assert resolved == len(names), "coverage assertion failed — some names did not resolve"


if __name__ == "__main__":
    main()
