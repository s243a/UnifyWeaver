#!/usr/bin/env python3
"""Tests for finite product-space transforms.

Run: `python3 test_product_space.py`.
"""

import math

import numpy as np

from product_space import (
    clip_mu,
    link_jacobian,
    log_mu,
    logit_mu,
    normalized_geometric_weights,
    product_interval,
    product_interval_jacobian,
    product_lower,
    product_lower_jacobian,
    product_upper,
    product_upper_jacobian,
)


def test_clip_mu_keeps_log_links_finite():
    clipped = clip_mu([0.0, 0.5, 1.0], eps=1e-4)
    assert np.allclose(clipped, [1e-4, 0.5, 1.0 - 1e-4])
    assert np.isfinite(log_mu([0.0, 1.0], eps=1e-4)).all()
    assert np.isfinite(logit_mu([0.0, 1.0], eps=1e-4)).all()


def test_product_lower_and_upper_match_manual_formulas():
    mu = np.array([0.2, 0.5, 0.8])
    assert abs(product_lower(mu) - (0.2 * 0.5 * 0.8)) < 1e-12
    assert abs(product_upper(mu) - (1.0 - 0.8 * 0.5 * 0.2)) < 1e-12
    lo, hi, width = product_interval(mu)
    assert lo < hi
    assert abs(width - (hi - lo)) < 1e-12


def test_weighted_product_proxies_support_geometric_mean():
    mu = np.array([0.25, 1.0])
    weights = normalized_geometric_weights(len(mu))
    assert np.allclose(weights, [0.5, 0.5])
    assert abs(product_lower(mu, weights, eps=1e-9) - 0.5) < 1e-9
    assert abs(product_upper(mu, weights, eps=1e-9) - 1.0) < 5e-5


def test_link_jacobians_match_closed_forms_after_clipping():
    mu = np.array([0.25, 0.5])
    assert np.allclose(link_jacobian(mu, "log"), [4.0, 2.0])
    assert np.allclose(link_jacobian(mu, "logit"), [1.0 / (0.25 * 0.75), 4.0])
    edge = link_jacobian([0.0], "logit", eps=1e-3)[0]
    assert math.isfinite(edge)
    assert abs(edge - 1.0 / (1e-3 * (1.0 - 1e-3))) < 1e-9


def test_product_jacobians_match_closed_forms():
    mu = np.array([0.2, 0.5, 0.8])
    lower = 0.2 * 0.5 * 0.8
    upper_complement = 0.8 * 0.5 * 0.2
    assert np.allclose(product_lower_jacobian(mu), lower / mu)
    assert np.allclose(product_upper_jacobian(mu), upper_complement / (1.0 - mu))

    J = product_interval_jacobian(mu)
    assert J.shape == (3, 3)
    assert np.allclose(J[0], product_lower_jacobian(mu))
    assert np.allclose(J[1], product_upper_jacobian(mu))
    assert np.allclose(J[2], J[1] - J[0])


def test_invalid_weights_and_links_fail_fast():
    try:
        product_lower([0.2, 0.3], weights=[1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("shape-mismatched weights should fail")

    try:
        product_upper([0.2, 0.3], weights=[1.0, -1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("negative weights should fail")

    try:
        link_jacobian([0.5], "odds")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown link should fail")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-space tests passed")
