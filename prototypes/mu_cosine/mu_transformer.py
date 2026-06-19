#!/usr/bin/env python3
"""Architecture scaffold for the μ-cosine transformer (the separate project).

Faithful **forward pass** of the design from the discussion, pure stdlib so the architecture is
concrete and inspectable without numpy/torch (which are not installed here). Training the full encoder
wants numpy/torch for speed and is the separate project; the *objective* it optimises is already
proven runnable on real data by `train_cosine_mu.py` (cosine ≈ μ, corr ≈ 1.0).

Design rationale (from the discussion):
  - A Wikipedia category id needs ≈ log2(1e6) ≈ 19-20 bits to address ⇒ ~19 "components" per concept.
  - MiniLM / nanoGPT use d_model ≈ 384; 384 / 19 ≈ 20 ⇒ ~20 attention heads.
  - One layer likely does a lot (the input is a single per-category vector, so attention is light);
    n_layers is configurable.
  - Each category embedding is INITIALISED from MiniLM (looked up by Wikipedia id), then refined.
  - μ(A | root) = cosine(encode(A), encode(root)) — the TWO-vector form: the same node encoding is
    reused for any root, so swapping the root vector swaps the domain.
  - Train so cosine matches the LLM-provided μ; sample category pairs that are *close* in a Wikipedia
    distance measure with higher probability (see `train_cosine_mu.py` for the runnable sampler).

NOTE on the single-token input: with one vector per category, self-attention's softmax is over a
single position (≡ 1), so MHA reduces to a linear map `W_O · concat_h(W_V·x)`. The head structure is
implemented faithfully anyway, so the *same* code accepts a multi-token input later (e.g. the
category's name tokens, or the category plus its graph neighbours) where attention becomes non-trivial.
"""
import math, random
from dataclasses import dataclass, field


@dataclass
class Config:
    d_model: int = 384          # MiniLM-sized
    # ~20 heads was the target (d_model / log2(|categories|) ≈ 384/19), but 384 is not divisible by
    # 20; its divisors near there are 16 (24 dims/head) and 24 (16 dims/head). 16 is the closest to
    # the intended head count. (d_model must be divisible by n_heads.)
    n_heads: int = 16
    n_layers: int = 1           # "a lot with one layer"; configurable
    d_ff: int = 384 * 4         # FFN inner width (4x, standard)
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
    """W: list[out][in], x: list[in] -> list[out]."""
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def rand_mat(out_d, in_d, rng, scale):
    return [[rng.gauss(0, scale) for _ in range(in_d)] for _ in range(out_d)]


class MultiHeadSelfAttention:
    """Single-position self-attention (softmax over 1 token ⇒ identity weighting); the head structure
    is kept so a multi-token input later gives real attention."""
    def __init__(self, cfg, rng):
        d = cfg.d_model
        s = 1.0 / math.sqrt(d)
        self.h = cfg.n_heads
        self.dh = d // cfg.n_heads
        assert d % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.Wq = rand_mat(d, d, rng, s)
        self.Wk = rand_mat(d, d, rng, s)
        self.Wv = rand_mat(d, d, rng, s)
        self.Wo = rand_mat(d, d, rng, s)

    def __call__(self, x):
        # x is a single token (list[d_model]). q·k/√dh over one key ⇒ softmax = [1.0] ⇒ context = v.
        v = matvec(self.Wv, x)
        _ = matvec(self.Wq, x)  # computed for fidelity; unused with a single key
        _ = matvec(self.Wk, x)
        return matvec(self.Wo, v)  # concat of per-head v == v here; O projects it back


class FeedForward:
    def __init__(self, cfg, rng):
        s = 1.0 / math.sqrt(cfg.d_model)
        self.W1 = rand_mat(cfg.d_ff, cfg.d_model, rng, s)
        self.b1 = [0.0] * cfg.d_ff
        self.W2 = rand_mat(cfg.d_model, cfg.d_ff, rng, s)
        self.b2 = [0.0] * cfg.d_model

    def __call__(self, x):
        h = [gelu(v + self.b1[i]) for i, v in enumerate(matvec(self.W1, x))]
        return [v + self.b2[i] for i, v in enumerate(matvec(self.W2, h))]


class Block:
    """Pre-norm transformer block: x + MHA(LN(x)); h + FFN(LN(h))."""
    def __init__(self, cfg, rng):
        self.attn = MultiHeadSelfAttention(cfg, rng)
        self.ff = FeedForward(cfg, rng)
        self.ln1_g = [1.0] * cfg.d_model; self.ln1_b = [0.0] * cfg.d_model
        self.ln2_g = [1.0] * cfg.d_model; self.ln2_b = [0.0] * cfg.d_model

    def __call__(self, x):
        h = add(x, self.attn(layernorm(x, self.ln1_g, self.ln1_b)))
        return add(h, self.ff(layernorm(h, self.ln2_g, self.ln2_b)))


class MuEncoder:
    """encode(category_id) -> vector; μ(a|root) = cosine(encode(a), encode(root))."""
    def __init__(self, cfg=Config(), init_embeddings=None):
        self.cfg = cfg
        rng = random.Random(cfg.seed)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layers)]
        # init_embeddings: dict id -> list[d_model] (from MiniLM in the real project). Absent ⇒ random.
        self.init = init_embeddings or {}
        self._rng = rng

    def embed(self, cid):
        e = self.init.get(cid)
        if e is None:  # stand-in for a missing MiniLM lookup
            e = [self._rng.gauss(0, 1.0 / math.sqrt(self.cfg.d_model)) for _ in range(self.cfg.d_model)]
            self.init[cid] = e
        return list(e)

    def encode(self, cid):
        x = self.embed(cid)
        for blk in self.blocks:
            x = blk(x)
        return x

    def mu(self, cid, root):
        a, b = self.encode(cid), self.encode(root)
        na = math.sqrt(sum(v * v for v in a)) or 1e-12
        nb = math.sqrt(sum(v * v for v in b)) or 1e-12
        return sum(x * y for x, y in zip(a, b)) / (na * nb)


if __name__ == "__main__":
    # smoke: forward pass runs end-to-end at the design scale and produces a μ in [-1,1].
    cfg = Config(n_layers=1)
    enc = MuEncoder(cfg)
    m = enc.mu("Electromagnetism", "Physics")
    print(f"forward OK: d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} "
          f"params/block≈{cfg.d_model*cfg.d_model*4 + cfg.d_model*cfg.d_ff*2:,}  μ(EM|Physics)={m:+.3f}")
    print("NOTE: this is the forward scaffold; training the encoder is the separate project "
          "(needs numpy/torch + MiniLM weights + more LLM μ labels). The objective is proven in "
          "train_cosine_mu.py.")
