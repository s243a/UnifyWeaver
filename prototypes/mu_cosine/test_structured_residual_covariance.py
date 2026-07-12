#!/usr/bin/env python3
"""Correctness tests for structured cross-item conditional residual covariance."""
import numpy as np
import pytest
import torch

from product_kalman import gaussian_condition_update
from structured_residual_covariance import (
    condition_item_batch,
    conditional_residuals,
    fit_block_model,
    fit_lmc_model,
    gaussian_joint_nll,
    median_rbf_bandwidth,
    off_block_diagnostics,
    rbf_kernel,
    select_kernel_ridge_mean,
)


def test_conditional_residual_sign_and_design_match_schur_reduction():
    P = np.array([[2.0, 0.3], [0.3, 1.0]])
    C = np.array([[0.2, -0.1], [0.05, 0.3]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    e = np.array([[0.4, -0.2], [-0.1, 0.8]])
    q_true = np.array([[0.3, -0.4], [0.1, 0.2]])
    solved = np.linalg.solve(P, C)
    v = e @ solved + q_true
    J, q = conditional_residuals(e, v, P, C, H)
    np.testing.assert_allclose(J, H + C.T @ np.linalg.inv(P))
    np.testing.assert_allclose(q, q_true)


def test_rbf_kernel_is_psd_unit_diagonal_and_cross_aligned():
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    length = median_rbf_bandwidth(features)
    kernel = rbf_kernel(features, length_scale=length)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    assert np.linalg.eigvalsh(kernel).min() > -1e-12
    cross = rbf_kernel(features[:2], features[2:], length_scale=length)
    np.testing.assert_allclose(cross, kernel[:2, 2:])
    with pytest.raises(ValueError, match="identical"):
        median_rbf_bandwidth(np.ones((3, 2)))


def test_kernel_ridge_regional_mean_uses_exact_train_loo_and_predicts():
    x = np.linspace(0.0, 3.0, 30)[:, None]
    target = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 0])])
    kernel = rbf_kernel(x, length_scale=0.5)
    model = select_kernel_ridge_mean(
        target,
        {"semantic": kernel, "identity": np.eye(len(x))},
        ridge_grid=[0.01, 0.1, 1.0],
    )
    assert model.kernel_name == "semantic"
    assert model.loo_mse < np.mean((target - target.mean(axis=0)) ** 2)
    assert model.loo_residuals.shape == target.shape
    prediction = model.predict(rbf_kernel(x[:4], x, length_scale=0.5))
    assert prediction.shape == (4, 2)
    assert np.isfinite(prediction).all()


def test_block_model_materialization_is_item_major_and_exact():
    residuals = np.array([[1.0, 0.2], [-0.5, 0.7], [0.3, -0.4]])
    model = fit_block_model(residuals, shrinkage=0.0)
    got = model.materialize(np.eye(3), np.eye(3))
    expected_block = residuals.T @ residuals / len(residuals)
    # The only difference is the declared tiny SPD floor.
    np.testing.assert_allclose(got[:2, :2], expected_block, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got[:2, 2:], 0.0)
    np.testing.assert_allclose(got[2:4, 2:4], got[:2, :2])
    metadata = model.to_dict()
    assert metadata["train_objective_type"] == "full_joint_gaussian_nll"
    assert metadata["train_objective_units"] == "original_channel_units"
    assert metadata["statistical_floor_units"] == "original_channel_variance"
    np.testing.assert_allclose(
        metadata["statistical_floor_original_channel_diagonal"],
        np.repeat(model.statistical_floor, 2),
    )


def _synthetic_lmc(seed=8, n=28):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n)[:, None]
    g = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 0])])
    Ks = rbf_kernel(x, length_scale=0.7)
    Kg = rbf_kernel(g, length_scale=1.0)
    B0 = np.array([[0.12, 0.01], [0.01, 0.10]])
    Bs = np.array([[0.20, 0.08], [0.08, 0.08]])
    Bg = np.array([[0.04, -0.02], [-0.02, 0.12]])
    covariance = np.kron(np.eye(n), B0) + np.kron(Ks, Bs) + np.kron(Kg, Bg)
    residuals = rng.multivariate_normal(np.zeros(2 * n), covariance).reshape(n, 2)
    return residuals, Ks, Kg


@pytest.mark.parametrize("kind", ["separable", "dense_lmc"])
def test_lmc_fit_is_deterministic_psd_and_does_not_lose_its_train_objective(kind):
    residuals, Ks, Kg = _synthetic_lmc()
    kwargs = dict(kind=kind, steps=50, learning_rate=0.03, max_pairs=256, seed=4)
    first = fit_lmc_model(residuals, Ks, Kg, **kwargs)
    second = fit_lmc_model(residuals, Ks, Kg, **kwargs)
    assert first.train_objective <= first.initial_objective + 1e-10
    assert first.train_objective <= first.block_reference_objective + 1e-10
    np.testing.assert_array_equal(first.independent_covariance, second.independent_covariance)
    np.testing.assert_array_equal(first.semantic_covariance, second.semantic_covariance)
    np.testing.assert_array_equal(first.graph_covariance, second.graph_covariance)
    covariance = first.materialize(Ks, Kg)
    assert np.linalg.eigvalsh(covariance).min() > 0.0
    if kind == "separable":
        assert np.isclose(first.semantic_weight + first.graph_weight, 1.0)
    metadata = first.to_dict()
    assert metadata["train_objective_type"] == "pairwise_composite_gaussian_nll"
    assert metadata["train_objective_units"] == "train_rms_standardized_channels"
    assert metadata["statistical_floor_units"] == "train_rms_standardized_variance"
    np.testing.assert_allclose(
        metadata["statistical_floor_original_channel_diagonal"],
        first.statistical_floor * first.channel_rms_scale ** 2,
    )


def test_dense_lmc_can_start_from_and_not_lose_separable_fit():
    residuals, Ks, Kg = _synthetic_lmc()
    separable = fit_lmc_model(
        residuals, Ks, Kg, kind="separable", steps=20, max_pairs=256, seed=4
    )
    dense = fit_lmc_model(
        residuals,
        Ks,
        Kg,
        kind="dense_lmc",
        steps=20,
        learning_rate=0.5,
        max_pairs=256,
        seed=4,
        initial_model=separable,
    )
    assert dense.train_objective <= dense.initial_objective + 1e-12
    assert dense.train_objective <= dense.block_reference_objective + 1e-12
    assert dense.train_objective <= dense.initial_model_reference_objective + 1e-12
    np.testing.assert_allclose(
        dense.initial_model_reference_objective,
        separable.train_objective,
        rtol=0.0,
        atol=1e-10,
    )


def test_dense_lmc_escapes_zero_structured_block_warm_start():
    residuals, Ks, Kg = _synthetic_lmc(n=40)
    block = fit_block_model(residuals, shrinkage=0.0)
    dense = fit_lmc_model(
        residuals,
        Ks,
        Kg,
        kind="dense_lmc",
        steps=100,
        learning_rate=0.03,
        max_pairs=512,
        seed=9,
        initial_model=block,
    )
    structured_mass = (
        np.linalg.norm(dense.semantic_covariance, ord="fro")
        + np.linalg.norm(dense.graph_covariance, ord="fro")
    )
    assert structured_mass > 1e-4
    assert dense.train_objective < dense.block_reference_objective - 1e-4


def test_dense_lmc_beats_restricted_separable_model_on_planted_nonseparable_field():
    residuals, Ks, Kg = _synthetic_lmc(n=60)
    separable = fit_lmc_model(
        residuals, Ks, Kg, kind="separable", steps=120, max_pairs=1024, seed=9
    )
    dense = fit_lmc_model(
        residuals,
        Ks,
        Kg,
        kind="dense_lmc",
        steps=120,
        max_pairs=1024,
        seed=9,
        initial_model=separable,
    )
    assert dense.train_objective < separable.train_objective - 1e-3


def test_gaussian_joint_nll_matches_torch_distribution():
    residuals = np.array([[0.2, -0.3], [0.1, 0.7]])
    covariance = np.array([
        [1.2, 0.1, 0.2, 0.0],
        [0.1, 0.8, 0.0, 0.1],
        [0.2, 0.0, 1.0, -0.1],
        [0.0, 0.1, -0.1, 0.9],
    ])
    got = gaussian_joint_nll(residuals, covariance)
    distribution = torch.distributions.MultivariateNormal(
        torch.zeros(4, dtype=torch.float64),
        covariance_matrix=torch.tensor(covariance, dtype=torch.float64),
    )
    expected = -float(distribution.log_prob(torch.tensor(residuals.reshape(-1), dtype=torch.float64)))
    assert np.isclose(got.total, expected)
    assert np.isclose(got.per_scalar, expected / 4.0)


def test_off_block_diagnostics_pin_mass_coupling_and_independence():
    B0 = np.array([[1.0, 0.2], [0.2, 0.8]])
    cross = np.array([[0.3, 0.1], [0.1, 0.2]])
    covariance = np.block([[B0, cross], [cross, B0]])
    got = off_block_diagnostics(covariance, 2)
    assert got.relative_off_item_frobenius_mass > 0.0
    assert got.maximum_whitened_off_block_spectral_norm > 0.0
    assert got.maximum_coupling_item_pair == (0, 1)
    independent = off_block_diagnostics(np.kron(np.eye(3), B0), 2)
    assert independent.relative_off_item_frobenius_mass == 0.0
    assert independent.maximum_whitened_off_block_spectral_norm == 0.0
    assert independent.maximum_coupling_item_pair is None


def test_joint_information_qr_batch_matches_dense_gaussian_conditioner():
    n_items = 3
    P0 = np.array([[1.2, 0.15], [0.15, 0.9]])
    # Match the campaign's overdetermined four-judge/two-state geometry.
    J0 = np.array([[1.0, 0.2], [-0.1, 0.8], [0.9, -0.2], [0.15, 1.1]])
    innovation = np.array([
        [0.2, -0.4, 0.1, 0.3],
        [0.7, 0.1, -0.2, 0.5],
        [-0.3, 0.5, 0.4, -0.1],
    ])
    K = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.25], [0.1, 0.25, 1.0]])
    factor = np.array([
        [0.6, 0.0, 0.0, 0.0],
        [0.1, 0.7, 0.0, 0.0],
        [-0.05, 0.08, 0.5, 0.0],
        [0.02, -0.03, 0.07, 0.55],
    ])
    B0 = factor @ factor.T
    Rc = np.kron(np.eye(n_items), np.eye(4) * 0.1) + np.kron(K, B0)
    actual = condition_item_batch(P0, J0, innovation, Rc)
    P = np.kron(np.eye(n_items), P0)
    H = np.kron(np.eye(n_items), J0)
    dense = gaussian_condition_update(
        np.zeros(2 * n_items), P, innovation.reshape(-1), Rc, H=H,
        cross_covariance=np.zeros((2 * n_items, 4 * n_items)), jitter=0.0,
    )
    np.testing.assert_allclose(actual.state_mean.reshape(-1), dense.mean, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual.full_covariance, dense.covariance, rtol=1e-9, atol=1e-10)
    assert not actual.loading_diagnostics["was_loaded"]
    assert not actual.prior_loading_diagnostics["was_loaded"]


def test_item_permutation_equivariance_of_materialization_nll_and_posterior():
    residuals, Ks, Kg = _synthetic_lmc(n=8)
    model = fit_lmc_model(
        residuals, Ks, Kg, kind="separable", steps=20, learning_rate=0.03, max_pairs=28, seed=2
    )
    covariance = model.materialize(Ks, Kg)
    permutation = np.array([5, 0, 7, 2, 1, 6, 3, 4])
    permuted_covariance = model.materialize(Ks[np.ix_(permutation, permutation)], Kg[np.ix_(permutation, permutation)])
    assert np.isclose(
        gaussian_joint_nll(residuals, covariance).total,
        gaussian_joint_nll(residuals[permutation], permuted_covariance).total,
    )
    P0 = np.array([[1.1, 0.2], [0.2, 0.8]])
    J0 = np.array([[1.0, 0.1], [-0.2, 0.9]])
    original = condition_item_batch(P0, J0, residuals, covariance)
    permuted = condition_item_batch(P0, J0, residuals[permutation], permuted_covariance)
    state_indices = np.concatenate([
        np.arange(2 * item, 2 * item + 2) for item in permutation
    ])
    np.testing.assert_allclose(permuted.state_mean, original.state_mean[permutation])
    np.testing.assert_allclose(
        permuted.full_covariance,
        original.full_covariance[np.ix_(state_indices, state_indices)],
    )
