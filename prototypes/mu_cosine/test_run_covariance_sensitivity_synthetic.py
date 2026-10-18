#!/usr/bin/env python3
"""Focused regressions for the cached synthetic covariance controls."""
import numpy as np
import pytest

from run_covariance_sensitivity_synthetic import (
    ALPHAS,
    CHANNELS,
    KRRCache,
    NLLScorer,
    SCENARIOS,
    SyntheticGeometry,
    _candidate_covariance,
    _fit_block_covariance,
    run_replicate,
)
from structured_residual_covariance import (
    fit_block_model,
    gaussian_joint_nll,
    rbf_kernel,
    select_kernel_ridge_mean,
)


def test_cached_krr_refit_exactly_matches_project_reference():
    x = np.linspace(-1.2, 1.4, 9)[:, None]
    train, held = np.arange(6), np.arange(6, 9)
    semantic = rbf_kernel(x[train], length_scale=0.7)
    graph = rbf_kernel(1.3 * x[train], length_scale=1.1)
    train_kernels = {
        "semantic": semantic,
        "graph": graph,
        "equal_mixture": 0.5 * (semantic + graph),
    }
    semantic_cross = rbf_kernel(x[held], x[train], length_scale=0.7)
    graph_cross = rbf_kernel(1.3 * x[held], 1.3 * x[train], length_scale=1.1)
    cross_kernels = {
        "semantic": semantic_cross,
        "graph": graph_cross,
        "equal_mixture": 0.5 * (semantic_cross + graph_cross),
    }
    residuals = np.column_stack((np.sin(x[train, 0]), np.cos(0.8 * x[train, 0])))
    ridges = (1e-2, 0.3, 2.0)
    actual = KRRCache(train_kernels, cross_kernels, ridges).fit_predict(residuals)
    expected = select_kernel_ridge_mean(residuals, train_kernels, ridge_grid=ridges)
    assert actual.kernel_name == expected.kernel_name
    assert actual.ridge == expected.ridge
    assert actual.loo_mse == pytest.approx(expected.loo_mse, abs=2e-15)
    np.testing.assert_allclose(actual.intercept, expected.intercept, atol=2e-15)
    np.testing.assert_allclose(actual.alpha, expected.alpha, atol=2e-15)
    np.testing.assert_allclose(actual.loo_residuals, expected.loo_residuals, atol=2e-15)
    np.testing.assert_allclose(
        actual.prediction,
        expected.predict(cross_kernels[expected.kernel_name]),
        atol=2e-15,
    )


def test_fast_block_refit_matches_project_reference_covariance():
    rng = np.random.default_rng(81)
    residuals = rng.normal(size=(17, CHANNELS))
    expected = fit_block_model(residuals, shrinkage=0.07).independent_covariance
    actual = _fit_block_covariance(residuals, 0.07)
    np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_true_and_candidate_endpoints_are_psd_and_preserve_item_marginals():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    scenario = next(value for value in SCENARIOS if value.name == "in_family_coupling_0.20")
    correlation = geometry.true_correlation(scenario, np.arange(geometry.item_count))
    np.linalg.cholesky(correlation)
    for item in range(geometry.item_count):
        section = slice(CHANNELS * item, CHANNELS * (item + 1))
        np.testing.assert_allclose(correlation[section, section], np.eye(CHANNELS), atol=0.0)
    maximum = 0.0
    for left in range(geometry.item_count):
        rows = slice(CHANNELS * left, CHANNELS * (left + 1))
        for right in range(left + 1, geometry.item_count):
            cols = slice(CHANNELS * right, CHANNELS * (right + 1))
            maximum = max(maximum, np.linalg.norm(correlation[rows, cols], ord=2))
    assert maximum == pytest.approx(0.20, abs=2e-15)

    endpoint = geometry.candidate_endpoint(scenario, "outer", 0.5, 2.0, 0.5)
    for alpha in ALPHAS:
        np.linalg.cholesky(endpoint.mixture(alpha))
    covariance = _candidate_covariance(
        geometry, geometry.block_covariance, endpoint, alpha=0.35
    )
    for item in range(len(geometry.outer_held)):
        section = slice(CHANNELS * item, CHANNELS * (item + 1))
        np.testing.assert_allclose(
            covariance[section, section], geometry.block_covariance, atol=3e-15
        )


def test_wrong_geometry_is_permutation_congruence_with_identical_energy():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    canonical = geometry.canonical_item_kernel
    wrong = geometry.wrong_item_kernel
    assert np.all(geometry.wrong_permutation != np.arange(geometry.item_count))
    np.testing.assert_allclose(
        wrong,
        canonical[np.ix_(geometry.wrong_permutation, geometry.wrong_permutation)],
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(wrong), np.linalg.eigvalsh(canonical), atol=3e-15
    )
    mask = ~np.eye(len(canonical), dtype=bool)
    assert np.sqrt(np.mean(wrong[mask] ** 2)) == pytest.approx(
        np.sqrt(np.mean(canonical[mask] ** 2)), abs=2e-15
    )


def test_cached_grid_nll_matches_independent_cholesky_reference():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    scenario = next(value for value in SCENARIOS if value.name == "in_family_coupling_0.10")
    rng = np.random.default_rng(92)
    residuals = rng.normal(size=(len(geometry.outer_held), CHANNELS))
    block = _fit_block_covariance(rng.normal(size=(25, CHANNELS)), 0.05)
    endpoint = geometry.candidate_endpoint(scenario, "outer", 2.0, 0.5, 1.0)
    scorer = NLLScorer(residuals, block)
    expected_block = gaussian_joint_nll(
        residuals, np.kron(np.eye(len(residuals)), block)
    ).per_scalar
    assert scorer.block_nll == pytest.approx(expected_block, abs=2e-15)
    scores = scorer.score_path(endpoint)
    for alpha in ALPHAS[1:]:
        expected = gaussian_joint_nll(
            residuals, _candidate_covariance(geometry, block, endpoint, alpha)
        ).per_scalar
        assert scores[alpha] == pytest.approx(expected, abs=3e-15)


def test_one_replicate_runs_three_fold_selector_and_outer_oracle():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    scenario = next(value for value in SCENARIOS if value.name == "block_null")
    record = run_replicate(geometry, scenario, replicate=0, seed=12300, shrinkage=0.05)
    assert len(record["inner_regional_means"]) == 3
    assert record["selection"]["alpha"] in ALPHAS
    assert record["outer_grid_oracle"]["nll_per_scalar"] <= (
        record["residual_nll_per_scalar"]["block"] + 1e-15
    )
    assert record["measurement_dimension"] == len(geometry.outer_held) * CHANNELS
