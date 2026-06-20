#!/usr/bin/env python3
"""Torch port of the μ-cosine encoder (`mu_encoder.py`), with MiniLM init wired into `embed()`.

Faithful to the stdlib scaffold: an **MLP/MoE encoder, not a transformer** (see `mu_encoder.py`'s
docstring for why attention is the wrong tool for a single per-category vector). The only additions a
real ML environment unlocks are the ones the handoff calls out:

  * **MiniLM init** (`build_minilm_init`): each category's name is encoded with
    `sentence-transformers/all-MiniLM-L6-v2` (384-d) and used as the *frozen* init embedding. The
    shared encoder (the trained part) maps these into the μ-cosine space, so it produces dense μ for
    *unlabelled* categories too — the whole point of the project (README "Resolved design decisions").
  * **Adam-friendly torch params** with **sparse embedding gradients** (the embedding table is the
    real memory lever at Wikipedia scale — README "Compute & batching").
  * **MoE load-balancing aux loss** (`aux = n_experts · Σ_i f_i·P_i`) so a soft gate can't collapse
    onto one expert (README "If you turn on MoE").

`encode(ids) -> [B, d]`; `mu(a, root) = cosine(encode(a), encode(root))`. `to_membership` clamps the
cosine to [0,1] before emitting to the Rust core (a negative weight corrupts the mass/KMV estimate).

    python3 mu_encoder_torch.py              # forward parity smoke test (MLP and MoE), random init
    python3 mu_encoder_torch.py --minilm     # same, but init the table from MiniLM (needs HF egress)
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 384          # MiniLM-sized
    n_layers: int = 1           # start at 1; few params ⇒ generalise from few labels
    d_ff: int = 384 * 4         # FFN inner width (4x, standard)
    n_experts: int = 1          # 1 = plain MLP; >1 = soft gated MoE (region routing, interpretable)
    seed: int = 0


class MLP(nn.Module):
    """Linear → GELU → Linear (one FFN / expert). GELU(tanh) matches the stdlib scaffold."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


class FeedForward(nn.Module):
    """Plain MLP (n_experts=1), or a soft gated mixture of region experts (n_experts>1).

    The soft gate (dot-product gating → softmax) is interpretable but can collapse all weight onto one
    expert; `forward` returns the load-balancing aux term alongside the output so the trainer can add
    `aux_weight · aux` to the loss (README). For n_experts=1 the aux term is 0."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.experts = nn.ModuleList([MLP(cfg) for _ in range(cfg.n_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts) if cfg.n_experts > 1 else None

    def forward(self, x: torch.Tensor):
        if self.gate is None:
            return self.experts[0](x), x.new_zeros(())
        g = F.softmax(self.gate(x), dim=-1)                 # [B, E] region routing weights
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # [B, E, d]
        y = (g.unsqueeze(-1) * outs).sum(dim=-2)            # [B, d]
        # load balancing: fraction routed (argmax) × mean gate, summed, ×E  (Switch-Transformer form)
        with torch.no_grad():
            routed = F.one_hot(g.argmax(dim=-1), self.n_experts).float().mean(dim=0)  # f_i
        importance = g.mean(dim=0)                          # P_i
        aux = self.n_experts * (routed * importance).sum()
        return y, aux


class Block(nn.Module):
    """Pre-norm residual MLP/MoE block: x + FFN(LN(x)). (No attention — see module docstring.)"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor):
        y, aux = self.ff(self.ln(x))
        return x + y, aux


class MuEncoder(nn.Module):
    """Shared encoder over a (mostly-frozen) MiniLM-init embedding table.

    `init_embeddings`: a [n_cats, d_model] tensor (e.g. from `build_minilm_init`). `freeze_init=True`
    (default) trains only the encoder blocks — required so unlabelled categories don't stay at random
    init (README). `names` maps category name → row id, so `mu_by_name` / emission can use names that
    match `category_parent.tsv` verbatim."""

    def __init__(self, cfg: Config = Config(), init_embeddings=None, names=None,
                 freeze_init: bool = True, sparse: bool = False):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        n_cats = init_embeddings.shape[0] if init_embeddings is not None else 1
        self.init_emb = nn.Embedding(n_cats, cfg.d_model, sparse=sparse)
        if init_embeddings is not None:
            with torch.no_grad():
                self.init_emb.weight.copy_(init_embeddings)
        else:
            nn.init.normal_(self.init_emb.weight, std=1.0 / math.sqrt(cfg.d_model))
        self.init_emb.weight.requires_grad = not freeze_init
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # name <-> id maps (verbatim names — load-bearing for the Rust integration)
        self.names = list(names) if names is not None else None
        self.id_of = {n: i for i, n in enumerate(self.names)} if self.names else None

    # ---- core forward ----
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """MiniLM-init embedding lookup (random fallback if no init was provided)."""
        return self.init_emb(ids)

    def encode(self, ids: torch.Tensor, neighbor_ids=None) -> torch.Tensor:
        """ids: [B] long. Optional neighbor_ids: [B, K] long for cheap fixed mean-pool context."""
        x = self.embed(ids)
        if neighbor_ids is not None:
            nb = self.embed(neighbor_ids).mean(dim=1)       # fixed-weight 'sum of vectors'
            x = (x + nb * neighbor_ids.shape[1]) / (1 + neighbor_ids.shape[1])
        aux_total = x.new_zeros(())
        for blk in self.blocks:
            x, aux = blk(x)
            aux_total = aux_total + aux
        return x, aux_total

    def mu(self, a_ids: torch.Tensor, root_ids: torch.Tensor):
        """Cosine μ(a | root) for batched ids. Returns (cos [B], aux scalar)."""
        a, aux_a = self.encode(a_ids)
        b, aux_b = self.encode(root_ids)
        cos = F.cosine_similarity(a, b, dim=-1)
        return cos, aux_a + aux_b

    # ---- name-keyed convenience (verbatim names) ----
    def _ids(self, names):
        if self.id_of is None:
            raise ValueError("encoder built without names; pass names= to use name-keyed methods")
        return torch.tensor([self.id_of[n] for n in names], dtype=torch.long,
                            device=self.init_emb.weight.device)

    @torch.no_grad()
    def mu_by_name(self, a_names, root_name):
        a = self._ids(list(a_names))
        r = self._ids([root_name] * len(a_names))
        cos, _ = self.mu(a, r)
        return cos


def to_membership(cos):
    """Clamp a cosine (∈[-1,1]) to a membership μ (∈[0,1]) for the Rust core (README guard #1).

    `descendant_mu_mass` / `sketch_mu_mass` *sum* μ as mass and assume μ ≥ 0; a negative weight
    corrupts the mass/KMV estimate. Works on python floats or torch tensors."""
    if isinstance(cos, torch.Tensor):
        return cos.clamp(0.0, 1.0)
    return max(0.0, min(1.0, cos))


# --------------------------------------------------------------------------------------------------
# MiniLM init
# --------------------------------------------------------------------------------------------------
def build_minilm_init(names, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=256,
                      device=None, normalize=False, humanize=True):
    """Encode each category *name* with MiniLM → a [len(names), 384] init tensor (one row per name).

    Wikipedia category names are underscore-joined (`Quantum_mechanics`); `humanize` turns them into
    natural text (`quantum mechanics`) for the sentence encoder, which is what MiniLM was trained on.
    `normalize=False` keeps raw embeddings (the encoder + cosine handle scale); set True to unit-norm.
    Requires `sentence-transformers` + HuggingFace egress."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def humanized(n):
        return n.replace("_", " ") if humanize else n

    texts = [humanized(n) for n in names]
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=normalize, show_progress_bar=False)
    return torch.tensor(emb, dtype=torch.float32)


def _smoke(use_minilm: bool):
    names = ["Physics", "Electromagnetism", "Optics", "Electricity", "Quantum_mechanics",
             "Religious_buildings", "Cooking"]
    if use_minilm:
        init = build_minilm_init(names)
        print(f"MiniLM init: {tuple(init.shape)} (rows = categories)")
    else:
        init = None
    for n_experts in (1, 4):
        cfg = Config(n_layers=1, n_experts=n_experts)
        if init is None:
            torch.manual_seed(0)
            init_t = torch.randn(len(names), cfg.d_model) / math.sqrt(cfg.d_model)
        else:
            init_t = init
        enc = MuEncoder(cfg, init_embeddings=init_t, names=names)
        em = enc.mu_by_name(["Electromagnetism"], "Physics").item()
        rel = enc.mu_by_name(["Religious_buildings"], "Physics").item()
        kind = "MLP" if n_experts == 1 else f"MoE({n_experts} experts)"
        print(f"{kind:14} forward OK  μ(EM|Physics)={em:+.3f}  μ(Religious_buildings|Physics)={rel:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minilm", action="store_true", help="init the table from MiniLM (needs HF egress)")
    args = ap.parse_args()
    _smoke(args.minilm)
    print("forward OK. Training: train_cosine_mu_torch.py (objective proven in train_cosine_mu.py).")
