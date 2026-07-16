#!/usr/bin/env python3
"""Focused mathematical regressions for covariance-sensitivity primitives."""
from types import SimpleNamespace

import numpy as np
import pytest

from covariance_sensitivity import (
    build_correlation_path,
    centered_linear_kernel,
    directional_posterior_sensitivity,
    expected_rbf_kernel_gaussian,
    gaussian_input_std_ratio_for_length_multiplier,
    gaussian_kl,
    materialize_channel_shrunk,
    mean_marginal_symmetric_kl,
    posterior_mean_dense,
    select_nested_candidate,
    shrink_channel_covariance,
    whitened_covariance_error,
)
from structured_residual_covariance import gaussian_joint_nll, rbf_kernel


def _assert_psd(matrix, tolerance=2e-12):
    assert np.linalg.eigvalsh(matrix)[0] >= -tolerance


def test_expected_rbf_zero_uncertainty_matches_ordinary_rbf():
    features = np.array([
        [-1.0, 0.2],
        [0.1, 0.7],
        [1.4, -0.3],
        [2.0, 1.1],
    ])
    expected = rbf_kernel(features, length_scale=0.8)
    actual = expected_rbf_kernel_gaussian(
        features, length_scale=0.8, input_std=0.0
    )
    np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_common_gaussian_uncertainty_broadens_normalized_rbf_exactly():
    features = np.array([[-1.0], [-0.2], [0.4], [1.7]])
    length_scale, input_std = 0.7, 0.3
    effective = np.sqrt(length_scale**2 + 2.0 * input_std**2)
    actual = expected_rbf_kernel_gaussian(
        features,
        length_scale=length_scale,
        input_std=input_std,
        normalize=True,
    )
    expected = rbf_kernel(features, length_scale=effective)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    assert gaussian_input_std_ratio_for_length_multiplier(
        effective / length_scale
    ) == pytest.approx(input_std / length_scale)
    assert gaussian_input_std_ratio_for_length_multiplier(0.9) is None


def test_heterogeneous_uncertain_rbf_is_psd_before_and_after_normalization():
    rng = np.random.default_rng(15)
    features = rng.normal(size=(12, 3))
    input_std = np.linspace(0.0, 0.8, len(features))
    raw = expected_rbf_kernel_gaussian(
        features,
        length_scale=1.1,
        input_std=input_std,
        normalize=False,
    )
    normalized = expected_rbf_kernel_gaussian(
        features,
        length_scale=1.1,
        input_std=input_std,
        normalize=True,
    )
    _assert_psd(raw)
    _assert_psd(normalized)
    np.testing.assert_allclose(np.diag(normalized), 1.0, atol=0.0)


def test_centered_linear_kernel_encodes_midpoint_sign_and_raw_amplitude():
    mu = np.array([0.1, 0.4, 0.6, 0.9])
    centered = mu - 0.5
    kernel = centered_linear_kernel(mu)
    np.testing.assert_allclose(kernel, np.outer(centered, centered), atol=0.0)
    assert kernel[0, 1] > 0.0
    assert kernel[0, 2] < 0.0
    assert kernel[0, 0] > kernel[1, 1]
    _assert_psd(kernel)
    judge_covariance = np.array([[1.0, 0.25], [0.25, 0.7]])
    _assert_psd(np.kron(kernel, judge_covariance))


def test_centered_linear_cross_kernel_and_normalization_are_consistent():
    left = np.array([[0.2, 0.9], [0.8, 0.7], [0.9, 0.2]])
    right = np.array([[0.1, 0.4], [0.7, 0.8]])
    expected = (left - 0.5) @ (right - 0.5).T
    np.testing.assert_allclose(centered_linear_kernel(left, right), expected, atol=0.0)

    normalized = centered_linear_kernel(left, normalize=True)
    _assert_psd(normalized)
    np.testing.assert_allclose(np.diag(normalized), 1.0, atol=0.0)
    expected_normalized_cross = (
        expected
        / np.linalg.norm(left - 0.5, axis=1)[:, None]
        / np.linalg.norm(right - 0.5, axis=1)[None, :]
    )
    np.testing.assert_allclose(
        centered_linear_kernel(left, right, normalize=True),
        expected_normalized_cross,
        atol=2e-15,
    )
    with pytest.raises(ValueError, match="zero norm"):
        centered_linear_kernel([0.2, 0.5, 0.8], normalize=True)


def test_centering_by_half_does_not_change_rbf_distances():
    mu = np.array([[0.05], [0.3], [0.8], [0.95]])
    original = rbf_kernel(mu, length_scale=0.25)
    centered = rbf_kernel(mu - 0.5, length_scale=0.25)
    np.testing.assert_allclose(centered, original, atol=2e-15)


def _structured_example():
    features = np.array([[-0.4], [0.3], [1.2]])
    item_kernel = rbf_kernel(features, length_scale=0.9)
    independent = np.array([[1.1, 0.12], [0.12, 0.8]])
    shared = np.array([[0.35, -0.04], [-0.04, 0.22]])
    structured = np.kron(np.eye(len(features)), independent) + np.kron(
        item_kernel, shared
    )
    block = np.array([[0.9, 0.18], [0.18, 0.65]])
    return block, structured, independent + shared


def test_correlation_path_endpoints_and_all_item_marginals_are_exact():
    block, structured, structured_item = _structured_example()
    path = build_correlation_path(block, structured, block_size=2)
    expected_block = np.kron(np.eye(3), block)
    np.testing.assert_allclose(path.covariance(0.0), expected_block, atol=2e-15)

    structured_factor = np.linalg.cholesky(structured_item)
    inverse_factor = np.linalg.solve(structured_factor, np.eye(2))
    whitener = np.kron(np.eye(3), inverse_factor)
    normalized = whitener @ structured @ whitener.T
    block_factor = np.kron(np.eye(3), np.linalg.cholesky(block))
    expected_matched = block_factor @ normalized @ block_factor.T
    np.testing.assert_allclose(path.covariance(1.0), expected_matched, atol=3e-15)

    for alpha in (0.0, 0.07, 0.5, 1.0):
        covariance = path.covariance(alpha)
        _assert_psd(covariance)
        for item in range(3):
            section = slice(2 * item, 2 * item + 2)
            np.testing.assert_allclose(covariance[section, section], block, atol=3e-15)


def test_correlation_path_fast_nll_matches_independent_cholesky_nll():
    block, structured, _ = _structured_example()
    path = build_correlation_path(block, structured, block_size=2)
    residuals = np.array([[0.5, -0.3], [-1.2, 0.7], [0.1, 0.2]])
    for alpha in (0.0, 0.025, 0.35, 1.0):
        expected = gaussian_joint_nll(residuals, path.covariance(alpha)).per_scalar
        assert path.nll_per_scalar(residuals, alpha) == pytest.approx(
            expected, abs=2e-15
        )


def test_channel_covariance_shrinkage_and_materialization_preserve_psd():
    semantic_covariance = np.array([[0.5, 0.25], [0.25, 0.3]])
    graph_covariance = np.array([[0.2, -0.08], [-0.08, 0.15]])
    model = SimpleNamespace(
        independent_covariance=np.array([[0.4, 0.03], [0.03, 0.35]]),
        semantic_covariance=semantic_covariance,
        graph_covariance=graph_covariance,
    )
    semantic = rbf_kernel(np.array([[0.0], [0.8], [1.5]]), length_scale=0.7)
    graph = rbf_kernel(np.array([[0.1], [0.4], [1.8]]), length_scale=1.0)
    np.testing.assert_allclose(
        shrink_channel_covariance(semantic_covariance, 0.0), semantic_covariance
    )
    np.testing.assert_allclose(
        shrink_channel_covariance(semantic_covariance, 1.0),
        np.diag(np.diag(semantic_covariance)),
    )
    for beta in (0.0, 0.5, 1.0):
        _assert_psd(materialize_channel_shrunk(model, semantic, graph, beta))


def test_analytic_posterior_covariance_sensitivity_matches_finite_difference():
    rng = np.random.default_rng(72)
    prior = np.array([[1.4, 0.2], [0.2, 0.9]])
    design = np.array([[1.0, 0.2], [0.1, 1.0], [0.7, -0.4]])
    innovation = rng.normal(size=(2, 3))
    factor = rng.normal(size=(6, 6))
    covariance = factor @ factor.T + 2.0 * np.eye(6)
    raw_direction = rng.normal(size=(6, 6))
    direction = 0.5 * (raw_direction + raw_direction.T)
    direction /= np.linalg.norm(direction, ord=2)

    analytic = directional_posterior_sensitivity(
        prior, design, innovation, covariance, direction
    )
    epsilon = 2e-6
    plus, _ = posterior_mean_dense(
        prior, design, innovation, covariance + epsilon * direction
    )
    minus, _ = posterior_mean_dense(
        prior, design, innovation, covariance - epsilon * direction
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        analytic.directional_derivative,
        finite_difference,
        rtol=2e-8,
        atol=2e-10,
    )


def _candidate_records(candidate_rows):
    records = [
        {
            "fold": fold,
            "alpha": 0.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 1.0,
            "nll_per_scalar": 1.0,
        }
        for fold in range(3)
    ]
    for alpha, beta, values in candidate_rows:
        records.extend({
            "fold": fold,
            "alpha": alpha,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": beta,
            "nll_per_scalar": nll,
        } for fold, nll in enumerate(values))
    return records


def test_nested_selector_prefers_smallest_stable_alpha_within_tolerance():
    records = _candidate_records([
        (0.05, 1.0, [0.80, 1.10, 1.10]),  # Only one positive fold: ineligible.
        (0.10, 0.5, [0.9000, 0.9005, 0.9010]),
        (0.20, 0.0, [0.9000, 0.9000, 0.9000]),
    ])
    selected, summaries = select_nested_candidate(records, tolerance=1e-3)
    assert selected["alpha"] == pytest.approx(0.10)
    assert selected["positive_folds"] == 3
    assert len(summaries) == 4


def test_nested_selector_falls_back_to_block_without_stable_gain():
    records = _candidate_records([
        (0.10, 0.5, [0.80, 1.10, 1.10]),
        (0.30, 0.0, [1.01, 1.02, 0.99]),
    ])
    selected, _ = select_nested_candidate(records)
    assert selected["alpha"] == 0.0
    assert selected["macro_gain_vs_block"] == 0.0


def test_gaussian_kl_and_marginal_symmetric_kl_have_expected_invariants():
    mean = np.array([0.2, -0.3])
    covariance = np.array([[1.3, 0.2], [0.2, 0.7]])
    assert gaussian_kl(mean, covariance, mean, covariance) == pytest.approx(
        0.0, abs=2e-16
    )
    shifted = mean + np.array([0.4, -0.1])
    other = np.array([[0.8, -0.05], [-0.05, 1.1]])
    forward = gaussian_kl(mean, covariance, shifted, other)
    reverse = gaussian_kl(shifted, other, mean, covariance)
    assert forward > 0.0
    assert reverse > 0.0

    means_a = np.stack([mean, mean + 0.1])
    means_b = np.stack([shifted, shifted - 0.1])
    covariances_a = np.stack([covariance, 1.2 * covariance])
    covariances_b = np.stack([other, 0.9 * other])
    expected = np.mean([
        0.5 * (
            gaussian_kl(ma, Pa, mb, Pb) + gaussian_kl(mb, Pb, ma, Pa)
        )
        for ma, Pa, mb, Pb in zip(
            means_a, covariances_a, means_b, covariances_b
        )
    ])
    assert mean_marginal_symmetric_kl(
        means_a, covariances_a, means_b, covariances_b
    ) == pytest.approx(expected)
    with pytest.raises(np.linalg.LinAlgError):
        gaussian_kl(mean, -np.eye(2), mean, covariance)


def test_whitened_covariance_error_uses_reference_relative_spectral_radius():
    reference = np.diag([4.0, 1.0])
    candidate = np.diag([6.0, 1.5])
    assert whitened_covariance_error(reference, candidate) == pytest.approx(0.5)
    assert whitened_covariance_error(reference, reference) == 0.0
