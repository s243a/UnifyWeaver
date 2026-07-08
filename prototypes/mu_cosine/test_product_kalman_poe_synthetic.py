#!/usr/bin/env python3
"""Synthetic checks for Product-Kalman / correlated-PoE fusion.

Run: `python3 test_product_kalman_poe_synthetic.py`.

These tests anchor the design-note warning: independent Gaussian PoE is calibrated only
when expert errors are conditionally independent. Shared evidence makes the fused mean
look reasonable, but the reported variance is overconfident unless the covariance is
modeled.
"""

import numpy as np


def independent_gaussian_poe(means, variances):
    """Gaussian PoE / information fusion under independent expert errors."""
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    precisions = 1.0 / variances
    fused_var = 1.0 / precisions.sum()
    fused_mean = fused_var * float(precisions @ means)
    return fused_mean, fused_var


def correlated_gaussian_fusion(means, covariance):
    """Best linear unbiased fusion when expert-error covariance is known."""
    means = np.asarray(means, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    precision = np.linalg.inv(covariance)
    ones = np.ones(len(means))
    denom = float(ones @ precision @ ones)
    weights = (ones @ precision) / denom
    fused_mean = float(weights @ means)
    fused_var = 1.0 / denom
    return fused_mean, fused_var, weights


def two_expert_covariance(total_var=1.0, rho=0.0):
    return total_var * np.array([[1.0, rho], [rho, 1.0]])


def test_independent_poe_matches_full_covariance_when_errors_are_independent():
    means = [0.25, 0.75]
    variances = [1.0, 1.0]
    _, poe_var = independent_gaussian_poe(means, variances)
    _, joint_var, weights = correlated_gaussian_fusion(means, two_expert_covariance(rho=0.0))
    assert np.allclose(weights, [0.5, 0.5])
    assert abs(poe_var - joint_var) < 1e-12


def test_independent_poe_understates_variance_for_correlated_experts():
    rho = 0.8
    means = [0.0, 0.0]
    _, naive_var = independent_gaussian_poe(means, [1.0, 1.0])
    _, joint_var, weights = correlated_gaussian_fusion(means, two_expert_covariance(rho=rho))
    assert np.allclose(weights, [0.5, 0.5])
    assert abs(joint_var - ((1.0 + rho) / 2.0)) < 1e-12
    assert joint_var / naive_var > 1.7


def test_shared_evidence_double_counting_is_empirically_overconfident():
    rng = np.random.default_rng(7)
    n = 60000
    shared_var = 0.8
    unique_var = 0.2
    shared = rng.normal(scale=shared_var**0.5, size=n)
    expert_a = shared + rng.normal(scale=unique_var**0.5, size=n)
    expert_b = shared + rng.normal(scale=unique_var**0.5, size=n)

    total_var = shared_var + unique_var
    rho = shared_var / total_var
    covariance = two_expert_covariance(total_var=total_var, rho=rho)
    _, naive_var = independent_gaussian_poe([0.0, 0.0], [total_var, total_var])
    _, joint_var, weights = correlated_gaussian_fusion([0.0, 0.0], covariance)

    fused_error = weights[0] * expert_a + weights[1] * expert_b
    empirical_var = float(np.var(fused_error))

    assert np.corrcoef(expert_a, expert_b)[0, 1] > 0.75
    assert abs(joint_var - (shared_var + unique_var / 2.0)) < 1e-12
    assert abs(empirical_var - joint_var) < 0.02
    assert empirical_var > naive_var * 1.7


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} product-kalman synthetic tests passed")
