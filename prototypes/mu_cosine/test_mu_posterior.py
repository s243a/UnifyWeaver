#!/usr/bin/env python3
"""Tests for MuPosterior (label-data μ→relation estimator). Pure-Python + numpy — genuinely torch-free.
Run: `python3 test_mu_posterior.py`."""
import math
import random

from mu_posterior import MuPosterior


def _data(seed=0, n=300):
    rng = random.Random(seed)
    a = [("A", rng.uniform(0.85, 0.95)) for _ in range(n)]   # relation A lives high
    b = [("B", rng.uniform(0.35, 0.45)) for _ in range(n)]   # relation B lives low
    return a + b


def test_posterior_separates_by_mu():
    post = MuPosterior(nbins=20)
    post.fit_source("s", _data())
    assert post.posterior({"s": 0.90})["A"] > 0.9
    assert post.posterior({"s": 0.40})["B"] > 0.9


def test_posterior_normalised():
    post = MuPosterior(nbins=20)
    post.fit_source("s", _data())
    for mu in (0.2, 0.4, 0.6, 0.9):
        assert abs(sum(post.posterior({"s": mu}).values()) - 1.0) < 1e-6


def test_anomaly_band():
    post = MuPosterior(nbins=20)
    post.fit_source("s", _data())
    assert post.is_anomalous("s", "A", 0.40)            # far below A's band ⇒ review
    assert not post.is_anomalous("s", "A", 0.90)
    lo, hi = post.band("s", "A", 0.05)
    assert 0.84 < lo and hi < 0.96


def test_nan_falls_back_to_prior():
    post = MuPosterior(nbins=20)
    post.fit_source("s", _data())
    p = post.posterior({"s": float("nan")})             # no evidence ⇒ posterior ≈ prior, still normalised
    assert abs(sum(p.values()) - 1.0) < 1e-6
    assert not post.is_anomalous("s", "A", float("nan"))


def test_two_sources_product_and_weight():
    post = MuPosterior(nbins=20)
    post.fit_source("s1", _data(seed=1), weight=1.0)
    post.fit_source("s2", _data(seed=2), weight=1.0)
    assert post.posterior({"s1": 0.90, "s2": 0.90})["A"] > 0.9
    # a zero-weighted source contributes nothing
    post.weights["s2"] = 0.0
    one = post.posterior({"s1": 0.90, "s2": 0.40})       # s2 says B, but weight 0 ⇒ s1 (A) wins
    assert one["A"] > one["B"]


def test_separability_reports_spread():
    post = MuPosterior(nbins=20)
    post.fit_source("s", _data())
    spread, means = post.separability("s")
    assert means["A"] > means["B"] and spread > 0.1     # well-separated synthetic data


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} mu_posterior tests passed (torch-free)")
