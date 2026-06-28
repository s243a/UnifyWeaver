#!/usr/bin/env python3
"""Tests for the hard-cell sampler + E[μ] reducers (DESIGN §12/§13). Pure python, no torch.
Run: `python3 test_cell_sampler.py`."""
import math
import random

from cell_sampler import sample_index, threshold_to_cell, expected, mc_expected, n_for_se


def test_sample_index_matches_distribution():
    rng = random.Random(0)
    p = [0.1, 0.6, 0.3]
    counts = [0, 0, 0]
    N = 40000
    for _ in range(N):
        counts[sample_index(p, rng)] += 1
    for i in range(3):
        assert abs(counts[i] / N - p[i]) < 0.02, (i, counts[i] / N, p[i])   # empirical ≈ P


def test_sample_index_isolated_and_reproducible():
    p = [0.25, 0.25, 0.5]
    a = [sample_index(p, random.Random(7)) for _ in range(50)]
    b = [sample_index(p, random.Random(7)) for _ in range(50)]
    assert a == b                                                           # same seed → same draws (§12(4))


def test_sample_index_unnormalised_ok():
    rng = random.Random(1)
    p = [2.0, 6.0, 2.0]                                                     # sums to 10, not 1
    counts = [0, 0, 0]
    for _ in range(20000):
        counts[sample_index(p, rng)] += 1
    assert abs(counts[1] / 20000 - 0.6) < 0.02


def test_threshold_construction_singleton_overlap_none():
    assert threshold_to_cell([0.7, 0.2, 0.1], tau=0.25) == (0,)            # singleton anchor
    assert threshold_to_cell([0.4, 0.4, 0.2], tau=0.25) == (0, 1)          # overlap cell (2+ rels)
    assert threshold_to_cell([0.2, 0.2, 0.2], tau=0.25) == ()             # empty → none (§9)
    assert threshold_to_cell([0.25, 0.1, 0.1], tau=0.25) == (0,)           # boundary inclusive (>= tau)


def test_expected_is_convex_combination():
    p = [0.2, 0.3, 0.5]
    mu = [0.9, 0.4, 0.0]                                                    # last = none cell, μ≈0
    e = expected(p, mu)
    assert abs(e - (0.2 * 0.9 + 0.3 * 0.4 + 0.5 * 0.0)) < 1e-9
    assert min(mu) <= e <= max(mu)                                          # bounded (§10)


def test_expected_normalises_defensively():
    assert abs(expected([2, 3, 5], [0.9, 0.4, 0.0]) - expected([0.2, 0.3, 0.5], [0.9, 0.4, 0.0])) < 1e-9


def test_mc_converges_to_analytic():
    p = [0.2, 0.3, 0.5]
    mu = [0.9, 0.4, 0.0]
    mu_fn = lambda i: mu[i]                                                 # hard-cell readout
    mean, se = mc_expected(p, mu_fn, random.Random(3), n=20000)
    assert abs(mean - expected(p, mu)) < 0.01                              # MC → analytic (§12(6))
    assert se < 0.01 and math.isfinite(se)


def test_mc_se_shrinks_with_n():
    p, mu = [0.5, 0.5], [1.0, 0.0]
    _, se_small = mc_expected(p, lambda i: mu[i], random.Random(4), n=64)
    _, se_big = mc_expected(p, lambda i: mu[i], random.Random(4), n=4096)
    assert se_big < se_small                                                # SE = std/sqrt(N) ↓


def test_n_for_se_rule():
    assert n_for_se(0.1, 0.02) == 25                                        # (0.1/0.02)^2 = 25
    assert n_for_se(0.1, 0.02) <= 32                                        # trainer default N=32 has headroom


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} cell_sampler tests passed")
