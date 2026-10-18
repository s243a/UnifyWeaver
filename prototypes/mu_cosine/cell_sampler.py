#!/usr/bin/env python3
"""Hard-cell sampler + E[μ] reducers for the operator-superposition (DESIGN §12/§13).

Haiku supplies the distributional **parameters** `P(cell)` per row (cached, §9); the TRAINING LOOP draws the
cell HERE with an **isolated** rng (§12(4)) — reproducible from cache+seed, no train-time Haiku call, and an
on/off A/B shares the batch trajectory. Pure + module-level (like `switched_op`) so it is unit-testable.

A *cell* is just an INDEX into a probability vector; the caller maps index → meaning (a relation, an overlap
set / atom, or the `none` cell, §9/§10). This module never decides WHICH cells exist — that is the §12(5)/(7)
**construction** step (`threshold_to_cell` below, deterministic, run once). It only (a) SAMPLES from a given
`P` and (b) REDUCES per-cell μ to the expectation `E[μ]` (§12(1) cell-level).

Two reducers, by design equal in expectation (§11/§12(6)):
  * `expected`     — analytic `E[μ] = Σ_i p_i·μ_i` over the per-cell HARD readouts (exact; R forwards).
  * `mc_expected`  — Monte-Carlo mean ± SE over N sampled cells (for when you can't enumerate the partition).
Feeding the *blended* cell `μ(Σ p·cell)` is the Jensen-biased shortcut and is NOT provided here (§11)."""
import math


def sample_index(p, rng):
    """Draw one index from the finite categorical `p` (list, sums to ~1). Pure; `rng` is an isolated
    `random.Random` (§12(4)). Robust to tiny normalisation drift."""
    total = sum(p)
    if total <= 0:
        raise ValueError("sample_index: non-positive total mass")
    x = rng.random() * total
    acc = 0.0
    for i, pi in enumerate(p):
        acc += pi
        if x < acc:
            return i
    return len(p) - 1                          # float-rounding fallback → last cell


def threshold_to_cell(p_rel, tau=0.25):
    """§12(5)/(7) CONSTRUCTION (deterministic, run once): the cell = the set of relation indices whose
    parameter clears `tau`. Returns a tuple of indices (sorted); the EMPTY tuple `()` means the `none`
    cell (no relation clears tau, §9). A singleton tuple is an ordinary anchor; 2+ is an overlap cell."""
    cell = tuple(i for i, pi in enumerate(p_rel) if pi >= tau)
    return cell                                # () == none


def expected(p, mu):
    """Analytic cell-level expectation `E[μ] = Σ_i p_i·μ_i` (§12(1)). `mu[i]` is the model's HARD-cell μ for
    cell i (the `none` cell carries μ≈0). Normalises `p` defensively."""
    if len(p) != len(mu):
        raise ValueError("expected: len(p) != len(mu)")
    total = sum(p) or 1.0
    return sum((pi / total) * mi for pi, mi in zip(p, mu))


def mc_expected(p, mu_fn, rng, n=32):
    """Monte-Carlo `E[μ]` over N hard draws (§12(6)): sample a cell, evaluate `mu_fn(cell_index)` (a possibly
    NON-linear model readout for that hard cell), average. Returns `(mean, se)` with `se = std/sqrt(N)`.
    Converges to `expected(p, [mu_fn(i) for i])` as N→∞ — the test asserts this."""
    if n < 1:
        raise ValueError("mc_expected: n must be >= 1")
    vals = [mu_fn(sample_index(p, rng)) for _ in range(n)]
    mean = sum(vals) / n
    if n == 1:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var) / math.sqrt(n)


def n_for_se(sigma, target_se=0.02):
    """§12(6) sample-count rule: smallest N with `sigma/sqrt(N) <= target_se`. With sigma~0.1 ⇒ N≈25 (default
    N=32 in the trainer leaves headroom)."""
    if target_se <= 0:
        raise ValueError("n_for_se: target_se must be > 0")
    return max(1, math.ceil((sigma / target_se) ** 2))
