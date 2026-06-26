#!/usr/bin/env python3
"""Tests for JointPosterior — the joint conditional P(relation | μ_vector) captures INTERACTIONS that a
product of per-source marginals structurally cannot (the whole point). Needs torch.
Run: `python3 test_joint_posterior.py`."""
import random

from mu_posterior import JointPosterior


def _asym(n, seed):
    """relation = sign(f − r): A if f>r else B. The MARGINALS of f and r are identical uniform[0,1] for BOTH
    A and B ⇒ a product of 1-D marginals has ZERO information; only the JOINT (the f−r interaction) separates."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        f, r = rng.uniform(0, 1), rng.uniform(0, 1)
        X.append([f, r]); y.append("A" if f > r else "B")
    return X, y


def test_joint_learns_asymmetry_a_product_cannot():
    Xtr, ytr = _asym(800, 0)
    jp = JointPosterior(["A", "B"], n_features=2, hidden=0).fit(Xtr, ytr, epochs=400)
    Xh, yh = _asym(400, 1)
    pr = jp.proba(Xh)
    acc = sum(pr[i].argmax() == jp.ri[yh[i]] for i in range(len(yh))) / len(yh)
    assert acc > 0.85, acc                                 # LR over [f,r] learns the f−r boundary (asymmetry)


def test_proba_normalised():
    Xtr, ytr = _asym(200, 2)
    jp = JointPosterior(["A", "B"], n_features=2).fit(Xtr, ytr, epochs=100)
    pr = jp.proba([[0.5, 0.5], [0.9, 0.1]])
    assert all(abs(row.sum() - 1.0) < 1e-5 for row in pr)


def test_nan_feature_imputed():
    Xtr, ytr = _asym(200, 3)
    jp = JointPosterior(["A", "B"], n_features=2).fit(Xtr, ytr, epochs=50)
    pr = jp.proba([[float("nan"), 0.5]])                   # NaN ⇒ imputed to feature mean, no crash
    assert abs(pr[0].sum() - 1.0) < 1e-5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} joint-posterior tests passed")
