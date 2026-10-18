import numpy as np
import pytest
from types import SimpleNamespace

from covariance_sensitivity import CorrelationPath, build_correlation_path
from run_covariance_sensitivity import (
    ALPHAS,
    PRIMARY,
    SECONDARY,
    _family_scale_grid,
    _induced_node_subsample,
    _null_maximum_gains,
    _preflight_stability_subsamples,
    _validate_args,
    aggregate_results,
)
from structured_residual_covariance import gaussian_joint_nll


def test_null_maximum_is_exactly_zero_for_identity_paths():
    dimension = 12
    identity = np.eye(dimension)
    path = CorrelationPath(
        np.eye(3), np.eye(3), identity, np.ones(dimension), identity, block_size=3
    )
    gains = _null_maximum_gains([path], draws=50, seed=7)
    np.testing.assert_allclose(gains, 0.0, atol=2e-15)


def test_null_maximum_is_deterministic_and_includes_block_fallback():
    eigenvalues = np.array([0.8, 0.9, 1.1, 1.2])
    identity = np.eye(4)
    path = CorrelationPath(
        np.eye(2), np.eye(2), np.diag(eigenvalues), eigenvalues, identity, block_size=2
    )
    first = _null_maximum_gains([path], draws=25, seed=11)
    second = _null_maximum_gains([path], draws=25, seed=11)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.0)


def test_null_maximum_matches_brute_force_full_gaussian_nll():
    block = np.array([[0.7, 0.12], [0.12, 0.4]])
    item_kernel = np.array([[1.0, 0.35], [0.35, 1.0]])
    structured = np.kron(item_kernel, block)
    path = build_correlation_path(block, structured, block_size=2)
    draws, seed = 9, 23
    fast = _null_maximum_gains([path], draws=draws, seed=seed)

    z = np.random.default_rng(seed).standard_normal((4, draws))
    factor = np.kron(np.eye(2), np.linalg.cholesky(block))
    residual_fields = (factor @ z).T.reshape(draws, 2, 2)
    block_covariance = np.kron(np.eye(2), block)
    brute = []
    for residuals in residual_fields:
        baseline = gaussian_joint_nll(residuals, block_covariance).per_scalar
        gains = [0.0] + [
            baseline - gaussian_joint_nll(residuals, path.covariance(alpha)).per_scalar
            for alpha in ALPHAS[1:]
        ]
        brute.append(max(gains))
    np.testing.assert_allclose(fast, brute, atol=2e-15, rtol=0.0)


def test_geometry_grids_keep_secondary_family_separate():
    assert len(_family_scale_grid(PRIMARY)) == 9
    assert len(_family_scale_grid(SECONDARY)) == 3
    assert {graph for _, graph in _family_scale_grid(SECONDARY)} == {1.0}


def test_node_subsample_is_deterministic_and_induced():
    materialized = {
        "pairs": [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("a", "d")]
    }
    outer_train = np.arange(5)
    first, node_total, node_kept = _induced_node_subsample(
        materialized, outer_train, 0.75, seed=19
    )
    second, _, _ = _induced_node_subsample(materialized, outer_train, 0.75, seed=19)
    np.testing.assert_array_equal(first, second)
    assert node_total == 4
    assert node_kept == 3
    retained_nodes = {node for index in first for node in materialized["pairs"][index]}
    assert len(retained_nodes) <= node_kept


def test_stability_preflight_rejects_too_small_induced_schedule():
    materialized = {
        "pairs": [(f"n{i}", f"n{i + 1}") for i in range(12)],
        "semantic": np.arange(24, dtype=float).reshape(12, 2),
        "graph_item_raw": np.arange(36, dtype=float).reshape(12, 3),
    }
    args = SimpleNamespace(
        stability_subsamples=2,
        stability_node_fraction=0.8,
    )
    with pytest.raises(ValueError, match="minimum is 20"):
        _preflight_stability_subsamples(
            materialized, np.arange(12), args, seed=17
        )


def test_multi_seed_v1_run_requires_explicit_failed_gate_acknowledgement():
    args = SimpleNamespace(
        seeds=10,
        allow_failed_v1_selector=False,
        null_draws=200,
        stability_subsamples=100,
        stability_node_fraction=0.8,
    )
    with pytest.raises(ValueError, match="synthetic selector gate failed"):
        _validate_args(args)
    args.allow_failed_v1_selector = True
    _validate_args(args)


def test_invalidated_v1_aggregate_can_never_emit_a_passing_gate():
    def posterior(nll, log_loss, aurc):
        return {
            "state": {"mean_bivariate_nll": nll},
            "decision": {"log_loss": log_loss, "aurc_margin": aurc},
            "conditioner": {
                "loading": {"relative_diagonal_loading": 0.0},
                "prior_loading": {"relative_diagonal_loading": 0.0},
            },
        }

    rows = []
    for corpus in ("exploratory", "fresh"):
        for seed in range(10, 20):
            family = {
                "selection": {"alpha": 0.5},
                "nested_selected": {
                    "held_joint_residual_nll_per_scalar": 0.9,
                    "posterior": posterior(0.9, 0.9, 0.9),
                },
                "outer_oracle": {"gain_vs_block": 0.2, "above_null_max95": True},
            }
            rows.append({
                "corpus": corpus,
                "outer_seed": seed,
                "models": {
                    "block_regional": {
                        "held_joint_residual_nll_per_scalar": 1.0,
                        "posterior": posterior(1.0, 1.0, 1.0),
                    }
                },
                "families": {PRIMARY: family, SECONDARY: family},
            })
    aggregate = aggregate_results(rows)
    assert aggregate["complete_preregistered_outer_run"]
    assert not aggregate["inferential_gate_evaluable"]
    for key in ("primary_gate", "secondary_endogenous_geometry_gate"):
        gate = aggregate[key]
        assert gate["uncalibrated_v1_raw_direction_and_stability_would_pass"]
        assert gate["conditional_fixed_path_oracle_headroom_diagnostic"]
        assert not gate["nested_covariance_gate_passes"]
        assert not gate["full_procedure_null_calibrated_oracle_headroom_passes"]
