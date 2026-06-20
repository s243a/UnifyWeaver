#!/usr/bin/env python3
"""Encoder scaffold for the μ-cosine model (the separate project) — an honest MLP, not a transformer.

`encode(category) -> vector`; `μ(a | root) = cosine(encode(a), encode(root))`. Faithful **forward
pass**, pure stdlib (no numpy/torch here), so the architecture is concrete and inspectable. Training
is the separate project; the *objective* is proven on real data by `train_cosine_mu.py`.

WHY AN MLP, NOT MULTIHEAD ATTENTION. Each category is a single (MiniLM-pooled) vector. Self-attention
over one token has softmax ≡ 1, so its query/key projections are dead weight and the value/output
projections collapse to one linear layer — a redundant Linear in an attention costume. The
region-dependent computation you actually want (different units firing in different parts of semantic
space) is the **MLP nonlinearity**, optionally made explicit/interpretable by a **gated MoE** (route
to a region expert). True attention only earns its keep with a *sequence* input — see the neighbour
context below, which here is folded in cheaply by a fixed mean-pool; learning those pooling weights
(real attention over neighbours) is the documented future upgrade.

Design sizing (from the discussion): d_model = 384 (MiniLM); n_layers configurable (1 default — "a lot
with one layer", and few params generalise from few labels); MoE `n_experts` (1 = plain MLP); init
from MiniLM by Wikipedia id (random fallback here, HF egress blocked).
"""
import math, random
from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 384          # MiniLM-sized
    n_layers: int = 1           # start at 1; few params ⇒ generalise from few labels
    d_ff: int = 384 * 4         # FFN inner width (4x, standard)
    n_experts: int = 1          # 1 = plain MLP; >1 = soft gated MoE (region routing, interpretable)
    seed: int = 0


# ---- primitive ops on python lists ----
def gelu(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layernorm(x, g, b, eps=1e-5):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [g[i] * (x[i] - mu) * inv + b[i] for i in range(n)]


def matvec(W, x):
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def add(*vs):
    return [sum(col) for col in zip(*vs)]


def softmax(xs):
    m = max(xs)
    e = [math.exp(v - m) for v in xs]
    s = sum(e)
    return [v / s for v in e]


def rand_mat(out_d, in_d, rng, scale):
    return [[rng.gauss(0, scale) for _ in range(in_d)] for _ in range(out_d)]


class MLP:
    """Linear → GELU → Linear (one FFN/expert)."""
    def __init__(self, cfg, rng):
        s = 1.0 / math.sqrt(cfg.d_model)
        self.W1 = rand_mat(cfg.d_ff, cfg.d_model, rng, s)
        self.b1 = [0.0] * cfg.d_ff
        self.W2 = rand_mat(cfg.d_model, cfg.d_ff, rng, s)
        self.b2 = [0.0] * cfg.d_model

    def __call__(self, x):
        h = [gelu(v + self.b1[i]) for i, v in enumerate(matvec(self.W1, x))]
        return [v + self.b2[i] for i, v in enumerate(matvec(self.W2, h))]


class FeedForward:
    """Plain MLP (n_experts=1), or a soft gated mixture of region experts (n_experts>1).

    The gate is the MoE-style 'dot product gating' (a learned linear scoring each expert by where x is
    in semantic space, then softmax) — this is the region-activation behaviour, made explicit and
    inspectable (the gate weights say which expert/region fired)."""
    def __init__(self, cfg, rng):
        self.experts = [MLP(cfg, rng) for _ in range(cfg.n_experts)]
        self.gate = rand_mat(cfg.n_experts, cfg.d_model, rng, 1.0 / math.sqrt(cfg.d_model)) \
            if cfg.n_experts > 1 else None

    def __call__(self, x):
        if self.gate is None:
            return self.experts[0](x)
        g = softmax(matvec(self.gate, x))                 # region routing weights
        outs = [e(x) for e in self.experts]
        return [sum(g[k] * outs[k][i] for k in range(len(outs))) for i in range(len(x))]


class Block:
    """Pre-norm residual MLP/MoE block: x + FFN(LN(x)). (No attention — see module docstring.)"""
    def __init__(self, cfg, rng):
        self.ff = FeedForward(cfg, rng)
        self.ln_g = [1.0] * cfg.d_model
        self.ln_b = [0.0] * cfg.d_model

    def __call__(self, x):
        return add(x, self.ff(layernorm(x, self.ln_g, self.ln_b)))


class MuEncoder:
    def __init__(self, cfg=Config(), init_embeddings=None):
        self.cfg = cfg
        rng = random.Random(cfg.seed)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layers)]
        self.init = init_embeddings or {}     # id -> MiniLM vector (random fallback)
        self._rng = rng

    def embed(self, cid):
        e = self.init.get(cid)
        if e is None:
            e = [self._rng.gauss(0, 1.0 / math.sqrt(self.cfg.d_model)) for _ in range(self.cfg.d_model)]
            self.init[cid] = e
        return list(e)

    def encode(self, cid, neighbors=None):
        # Cheap context: mean-pool the category with a few neighbour vectors (a fixed-weight 'sum of
        # vectors'). Learning these weights = attention over neighbours = the future upgrade.
        if neighbors:
            x = add(self.embed(cid), *[self.embed(n) for n in neighbors])
            x = [v / (1 + len(neighbors)) for v in x]
        else:
            x = self.embed(cid)
        for blk in self.blocks:
            x = blk(x)
        return x

    def mu(self, cid, root, neighbors=None, root_neighbors=None):
        a, b = self.encode(cid, neighbors), self.encode(root, root_neighbors)
        na = math.sqrt(sum(v * v for v in a)) or 1e-12
        nb = math.sqrt(sum(v * v for v in b)) or 1e-12
        return sum(x * y for x, y in zip(a, b)) / (na * nb)


def to_membership(cos):
    """Map a cosine (∈ [-1, 1]) to a membership μ (∈ [0, 1]) for emitting to the Rust core.

    **Required before feeding the dense μ to `descendant_mu_mass` / `sketch_mu_mass`:** those sum μ as
    *mass* and assume μ ≥ 0 — a negative weight corrupts the mass / KMV estimate. `descendant_mu_mass_
    gated` tolerates a negative (it just fails the `≥ threshold` gate), but the others do not, so clamp
    at the emission step. (Training targets the LLM μ ∈ [0,1], so the model is only ever *asked* for
    [0,1]; but an unlabelled, very-dissimilar pair can produce a negative cosine at inference.)
    """
    return max(0.0, min(1.0, cos))


if __name__ == "__main__":
    for n_experts in (1, 4):
        enc = MuEncoder(Config(n_layers=1, n_experts=n_experts))
        plain = enc.mu("Electromagnetism", "Physics")
        ctx = enc.mu("Electromagnetism", "Physics", neighbors=["Optics", "Electricity"])
        kind = "MLP" if n_experts == 1 else f"MoE({n_experts} experts)"
        print(f"{kind:14} forward OK  μ(EM|Physics)={plain:+.3f}  μ with neighbour-context={ctx:+.3f}")
    print("NOTE: forward scaffold; training is the separate project (numpy/torch + MiniLM + labels). "
          "Objective proven in train_cosine_mu.py; attention-over-neighbours is the documented upgrade.")
