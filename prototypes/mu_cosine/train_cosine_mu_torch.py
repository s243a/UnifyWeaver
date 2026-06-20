#!/usr/bin/env python3
"""Train the cosine-μ objective in torch — the real-environment counterpart of `train_cosine_mu.py`.

Same objective the stdlib proof validates: minimise `(cos(encode(X), encode(root)) − μ)²`. Two modes:

  * `--mode fixture` (default): single-anchor regression on the Haiku-scored
    `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (anchor `Physics`). Spends **no** LLM budget —
    the fixture is already scored. Two sub-flavours:
      - `--free-vectors`: learnable per-category vectors, identity encoder (`n_layers=0`), random init —
        the torch re-creation of `train_cosine_mu.py`. With enough dim it reaches `corr→1.00`,
        confirming the torch objective + Adam match the stdlib proof (capacity, not generalisation).
      - default: **frozen MiniLM init + a trained shared encoder** (`n_layers=1`). This is the
        generalisation setup — `--holdout` measures μ-prediction on categories whose label the encoder
        never saw, which is the real test the proof can't make.

  * `--mode pairs`: pairwise regression on `gen_mu_pairs.py` output once its `mu` column is **scored**
    (the budget step — confirm with the user first). Varied anchors; `(a, b, μ)` per row.
    NOTE: do **not** add `--dist-bias` here — those pairs already encode the distance preference via
    explicit negatives (README), so a distance bias double-penalises far pairs. (`--dist-bias` is
    fixture-only and ignored in pairs mode.)

Optimiser: **Adam** (per-row adaptive steps suit an embedding table — README), `SparseAdam` when the
embedding table is trainable and sparse. MoE (`--n-experts>1`) adds the load-balancing aux loss.

    python3 train_cosine_mu_torch.py --free-vectors          # reproduce the proof (corr→1.00)
    python3 train_cosine_mu_torch.py --minilm --holdout 0.2  # generalisation: shared encoder on MiniLM
    python3 train_cosine_mu_torch.py --mode pairs --pairs mu_pairs.tsv --minilm
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque

import torch
import torch.nn.functional as F

from mu_encoder_torch import Config, MuEncoder, build_minilm_init

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
GRAPH = os.path.join(REPO, "data", "benchmark", "10k", "category_parent.tsv")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "wikipedia_physics_fuzzy_nodes.tsv")
ANCHOR = "Physics"


# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------
def load_graph(path):
    adj = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("child"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            c, p = parts[0], parts[1]
            adj.setdefault(c, set()).add(p)
            adj.setdefault(p, set()).add(c)
    return adj


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


def load_pairs(path):
    """Read scored `gen_mu_pairs.py` output: name_a, name_b, stratum, walk_len, mu (mu filled)."""
    rows = []
    skipped = 0
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[4].strip() == "":
                skipped += 1
                continue
            try:
                mu = float(parts[4])
            except ValueError:
                skipped += 1
                continue
            rows.append((parts[0], parts[1], mu, parts[2]))
    if skipped:
        print(f"WARNING: {skipped} pair rows had a blank/non-numeric μ (unscored) — skipped. "
              f"Score them first (gen_mu_pairs.py score_stub; spends LLM budget).")
    return rows


def bfs_dist(adj, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj.get(x, ()):
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist


def pearson(xs, ys):
    xs, ys = torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(ys, dtype=torch.float64)
    xs, ys = xs - xs.mean(), ys - ys.mean()
    denom = (xs.norm() * ys.norm()).clamp_min(1e-12)
    return float((xs @ ys) / denom)


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------
def make_optimizer(model, lr, weight_decay=0.0):
    """SparseAdam for a trainable sparse embedding table; AdamW for everything else (README: Adam).

    `weight_decay` regularises the *shared encoder* blocks toward zero — i.e. toward the identity
    residual `x + FFN(LN(x))` ≈ raw MiniLM — which is the strongest generaliser when labels are few
    (a small pairwise set easily overfits a 1.2M-param encoder). SparseAdam has no weight_decay arg."""
    sparse_params, dense_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "init_emb.weight" and model.init_emb.sparse:
            sparse_params.append(p)
        else:
            dense_params.append(p)
    opts = []
    if dense_params:
        opts.append(torch.optim.AdamW(dense_params, lr=lr, weight_decay=weight_decay))
    if sparse_params:
        opts.append(torch.optim.SparseAdam(sparse_params, lr=lr))
    return opts


def train_fixture(args):
    adj = load_graph(GRAPH)
    mu = load_mu(FIXTURE)
    dist = bfs_dist(adj, ANCHOR)
    nodes = [n for n in mu if n in adj and n != ANCHOR]
    print(f"loaded {len(nodes)} scored nodes connected to '{ANCHOR}' "
          f"(μ in [{min(mu[n] for n in nodes):.1f},{max(mu[n] for n in nodes):.1f}])")

    # held-out split (generalisation test). Stratify-ish by shuffling with the seed.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(nodes), generator=g).tolist()
    nodes = [nodes[i] for i in perm]
    n_hold = int(round(args.holdout * len(nodes)))
    hold = nodes[:n_hold]
    train = nodes[n_hold:]
    print(f"train {len(train)} / held-out {len(hold)} (holdout={args.holdout})")

    names = [ANCHOR] + nodes
    if args.minilm and not args.free_vectors:
        print("building MiniLM init for fixture nodes ...")
        init = build_minilm_init(names)
        cfg = Config(n_layers=args.n_layers, n_experts=args.n_experts, d_model=init.shape[1])
        model = MuEncoder(cfg, init_embeddings=init, names=names, freeze_init=not args.unfreeze)
    else:
        # free-vector mode: random init, identity encoder, trainable table (= train_cosine_mu.py)
        cfg = Config(n_layers=0 if args.free_vectors else args.n_layers,
                     n_experts=args.n_experts, d_model=args.dim)
        torch.manual_seed(args.seed)
        init = torch.randn(len(names), cfg.d_model) / math.sqrt(cfg.d_model)
        model = MuEncoder(cfg, init_embeddings=init, names=names,
                          freeze_init=False if args.free_vectors else not args.unfreeze)

    # distance-bias sampling weights (fixture only). README: do NOT use in pairs mode.
    def w(n):
        return (1.0 / (1.0 + dist.get(n, 99))) ** args.dist_bias
    train_w = torch.tensor([w(n) for n in train])

    root_ids = model._ids([ANCHOR] * len(train))
    train_ids = model._ids(train)
    train_mu = torch.tensor([mu[n] for n in train])

    opts = make_optimizer(model, args.lr, args.weight_decay)

    def report(tag):
        with torch.no_grad():
            cos_tr = model.mu_by_name(train, ANCHOR)
            line = f"{tag} train MSE {F.mse_loss(cos_tr, train_mu):.4f} corr {pearson(cos_tr, train_mu):+.3f}"
            if hold:
                cos_h = model.mu_by_name(hold, ANCHOR)
                hmu = torch.tensor([mu[n] for n in hold])
                line += f" | HELD-OUT MSE {F.mse_loss(cos_h, hmu):.4f} corr {pearson(cos_h, hmu):+.3f}"
            print(line)

    report("initial")
    for ep in range(args.epochs):
        for o in opts:
            o.zero_grad()
        cos, aux = model.mu(train_ids, root_ids)
        err2 = (cos - train_mu) ** 2
        loss = (err2 * train_w).sum() / train_w.sum() + args.aux_weight * aux
        loss.backward()
        for o in opts:
            o.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            report(f"epoch {ep+1:5d}")
    print()
    report("FINAL")

    if hold:
        print("\nheld-out examples (μ vs predicted cosine):")
        with torch.no_grad():
            cos_h = model.mu_by_name(hold, ANCHOR)
        ex = sorted(range(len(hold)), key=lambda i: mu[hold[i]])
        for i in [ex[0], ex[len(ex)//2], ex[-1]]:
            n = hold[i]
            print(f"  μ {mu[n]:.2f}  cos {float(cos_h[i]):+.2f}  dist {dist.get(n,'?')}  {n}")
    return model


def train_pairs(args):
    rows = load_pairs(args.pairs)
    if not rows:
        raise SystemExit(f"no scored pairs in {args.pairs} — fill the μ column first (budget step).")
    pos = [r for r in rows if r[3] == "pos"]
    neg = [r for r in rows if r[3] != "pos"]
    print(f"loaded {len(rows)} scored pairs: {len(pos)} pos (μ in "
          f"[{min((r[2] for r in pos), default=0):.2f},{max((r[2] for r in pos), default=0):.2f}]) / "
          f"{len(neg)} neg (μ=0 SGNS negatives)")
    names = sorted({n for a, b, _, _ in rows for n in (a, b)})

    if args.minilm:
        print("building MiniLM init for paired categories ...")
        init = build_minilm_init(names)
        cfg = Config(n_layers=args.n_layers, n_experts=args.n_experts, d_model=init.shape[1])
        model = MuEncoder(cfg, init_embeddings=init, names=names, freeze_init=not args.unfreeze)
    else:
        cfg = Config(n_layers=args.n_layers, n_experts=args.n_experts, d_model=args.dim)
        torch.manual_seed(args.seed)
        init = torch.randn(len(names), cfg.d_model) / math.sqrt(cfg.d_model)
        model = MuEncoder(cfg, init_embeddings=init, names=names, freeze_init=False)

    # Hold out a fraction of the *positives* (the graded signal). Negatives are μ=0 by construction,
    # so a held-out metric on them is trivial (predict 0) and would dilute corr — keep them all in
    # training as SGNS negatives, and measure generalisation on held-out positives only.
    g = torch.Generator().manual_seed(args.seed)
    pperm = torch.randperm(len(pos), generator=g).tolist()
    pos = [pos[i] for i in pperm]
    n_hold = int(round(args.holdout * len(pos)))
    hold_pos = pos[:n_hold]
    train_rows = pos[n_hold:] + neg
    gt = torch.Generator().manual_seed(args.seed + 1)
    train_rows = [train_rows[i] for i in torch.randperm(len(train_rows), generator=gt).tolist()]
    print(f"train {len(train_rows)} pairs ({len(pos)-n_hold} pos + {len(neg)} neg) / "
          f"held-out {len(hold_pos)} positives")

    def ids(split):
        a = model._ids([r[0] for r in split])
        b = model._ids([r[1] for r in split])
        m = torch.tensor([r[2] for r in split])
        return a, b, m

    a_tr, b_tr, m_tr = ids(train_rows)
    pos_tr = [r for r in train_rows if r[3] == "pos"]
    a_p, b_p, m_p = ids(pos_tr)
    opts = make_optimizer(model, args.lr, args.weight_decay)

    def report(tag):
        with torch.no_grad():
            cos, _ = model.mu(a_tr, b_tr)
            cosp, _ = model.mu(a_p, b_p)
            line = (f"{tag} train MSE {F.mse_loss(cos, m_tr):.4f} (pos-only corr {pearson(cosp, m_p):+.3f})")
            if hold_pos:
                a_h, b_h, m_h = ids(hold_pos)
                cosh, _ = model.mu(a_h, b_h)
                line += f" | HELD-OUT pos MSE {F.mse_loss(cosh, m_h):.4f} corr {pearson(cosh, m_h):+.3f}"
            print(line)

    report("initial")
    for ep in range(args.epochs):
        for o in opts:
            o.zero_grad()
        cos, aux = model.mu(a_tr, b_tr)
        loss = F.mse_loss(cos, m_tr) + args.aux_weight * aux
        loss.backward()
        for o in opts:
            o.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            report(f"epoch {ep+1:5d}")
    print()
    report("FINAL")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixture", "pairs"], default="fixture")
    ap.add_argument("--pairs", default=os.path.join(ROOT, "mu_pairs.tsv"))
    ap.add_argument("--free-vectors", action="store_true",
                    help="fixture: learnable vectors + identity encoder (reproduce train_cosine_mu.py)")
    ap.add_argument("--minilm", action="store_true", help="init the embedding table from MiniLM")
    ap.add_argument("--unfreeze", action="store_true", help="also fine-tune the MiniLM init table")
    ap.add_argument("--dim", type=int, default=16, help="dim for random-init (free-vector) mode")
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--n-experts", type=int, default=1)
    ap.add_argument("--aux-weight", type=float, default=0.01, help="MoE load-balancing aux weight")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay on the shared encoder (regularises toward raw MiniLM)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction held out for generalisation")
    ap.add_argument("--dist-bias", type=float, default=1.0,
                    help="fixture only: exponent on 1/(1+dist) sampling weight (0 = uniform)")
    ap.add_argument("--save-encoder", default=None,
                    help="save the trained shared encoder (blocks + cfg) for emit_dense_mu.py")
    args = ap.parse_args()

    if args.mode == "fixture":
        model = train_fixture(args)
    else:
        if args.dist_bias != 1.0:
            print("note: --dist-bias is ignored in pairs mode (pairs already encode distance via negatives)")
        model = train_pairs(args)

    if args.save_encoder:
        # Save ONLY the shared encoder (blocks) + config — vocabulary-independent, so it can be
        # applied to ANY MiniLM-init vector (the whole graph) to emit dense μ (emit_dense_mu.py).
        torch.save({"cfg": vars(model.cfg), "blocks": model.blocks.state_dict()}, args.save_encoder)
        print(f"saved shared encoder (config + {sum(p.numel() for p in model.blocks.parameters())} "
              f"block params) → {args.save_encoder}")


if __name__ == "__main__":
    main()
