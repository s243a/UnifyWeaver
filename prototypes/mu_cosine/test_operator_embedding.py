#!/usr/bin/env python3
"""Tests for the RANDOM operator embedding (consume the fitted posterior → a superposition op token).
Pure-Python + numpy (torch-free). Run: `python3 test_operator_embedding.py`."""
import numpy as np

from mu_posterior import sample_operator_weights, random_operator_embedding


def test_weights_on_simplex():
    w = sample_operator_weights([0.7, 0.2, 0.1], alpha=20, rng=np.random.default_rng(0))
    assert abs(w.sum() - 1.0) < 1e-9 and (w >= 0).all()


def test_mean_weights_track_probs():
    rng = np.random.default_rng(0)
    p = [0.6, 0.3, 0.1]
    W = np.mean([sample_operator_weights(p, 20, rng) for _ in range(8000)], 0)
    assert np.allclose(W, p, atol=0.02)                    # Dirichlet mean = probs


def test_alpha_controls_noise():
    rng = np.random.default_rng(0)
    p = [0.5, 0.3, 0.2]
    var_hi_alpha = np.var([sample_operator_weights(p, 200, rng) for _ in range(3000)], 0).sum()
    var_lo_alpha = np.var([sample_operator_weights(p, 5, rng) for _ in range(3000)], 0).sum()
    assert var_lo_alpha > var_hi_alpha                     # lower alpha ⇒ more operator noise


def test_confident_posterior_recovers_its_operator():
    op_emb = np.eye(3)                                     # 3 operators, orthonormal embeddings
    e = random_operator_embedding([1.0, 0.0, 0.0], op_emb, alpha=1000, rng=np.random.default_rng(0))
    assert np.allclose(e, [1, 0, 0], atol=0.05)            # p on op0 ⇒ ≈ op0's embedding


def test_flat_posterior_blends():
    op_emb = np.eye(3)
    e = random_operator_embedding([1 / 3, 1 / 3, 1 / 3], op_emb, alpha=1000, rng=np.random.default_rng(0))
    assert np.allclose(e, [1 / 3, 1 / 3, 1 / 3], atol=0.05)  # flat p ⇒ even superposition


def test_out_of_set_noise_adds_spread():
    op_emb = np.eye(4)
    rng = np.random.default_rng(0)
    base = [random_operator_embedding([.4, .3, .2, .1], op_emb, 50, 0.0, rng) for _ in range(800)]
    noisy = [random_operator_embedding([.4, .3, .2, .1], op_emb, 50, 0.5, rng) for _ in range(800)]
    assert np.var(noisy, 0).sum() > np.var(base, 0).sum()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} operator-embedding tests passed (torch-free)")
