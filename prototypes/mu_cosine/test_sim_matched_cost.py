#!/usr/bin/env python3
"""Focused invariants for the post-#3648 matched-cost simulation hardening."""
import argparse

import numpy as np
import pytest

from sim_matched_cost import (
    H_FREE,
    budget_plan,
    fused_targets,
    nested_ridge_fit_predict,
    paired_delta_summary,
    positive_integer,
    replicate_permutation,
    ridge_fit_predict,
    stable_seed,
)


def test_ridge_selection_is_row_permutation_invariant():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 8))
    y = X @ rng.normal(size=8) + rng.normal(scale=0.4, size=60)
    Xq = rng.normal(size=(11, 8))
    ids = np.arange(1000, 1060)
    pred, info = ridge_fit_predict(
        X, y, Xq, sample_ids=ids, split_seed=91, return_info=True,
    )
    perm = rng.permutation(len(X))
    pred_perm, info_perm = ridge_fit_predict(
        X[perm], y[perm], Xq, sample_ids=ids[perm], split_seed=91, return_info=True,
    )
    assert info["lambda"] == info_perm["lambda"]
    assert np.allclose(pred, pred_perm, rtol=0.0, atol=1e-12)
    # The inner scaler must be fit on inner-train, not all rows (the means differ for this fixture).
    assert not np.allclose(info["inner_mu"], X.mean(axis=0))


def test_nested_target_fit_excludes_paid_validation_then_refits_all_paid():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(90, 7))
    y = X @ rng.normal(size=7) + rng.normal(scale=0.2, size=90)
    Xq = rng.normal(size=(9, 7))
    ids = np.arange(5000, 5090)
    paid = np.arange(30)
    bulk = np.arange(30, 80)
    fit_calls = []

    def pseudo_builder(fit_paid_idx):
        fit_calls.append(np.asarray(fit_paid_idx).copy())
        # A supervised target generator: its output changes with the paid labels it was allowed to fit.
        return 0.25 * X[:, 0] + y[fit_paid_idx].mean()

    _, info = nested_ridge_fit_predict(
        X, y, Xq, paid, bulk, pseudo_target_builder=pseudo_builder,
        sample_ids=ids, split_seed=41, return_info=True,
    )
    selection_fit_ids = set(info["selection_target_fit_ids"].tolist())
    validation_ids = set(info["paid_valid_ids"].tolist())
    assert selection_fit_ids.isdisjoint(validation_ids)
    assert set(ids[fit_calls[0]].tolist()) == selection_fit_ids
    assert set(ids[fit_calls[0]].tolist()).isdisjoint(validation_ids)
    # The second builder call is the post-selection full-budget refit and must recover every paid row.
    assert set(ids[fit_calls[1]].tolist()) == set(ids[paid].tolist())
    assert set(info["final_target_fit_ids"].tolist()) == set(ids[paid].tolist())


def test_nested_pseudo_pipeline_is_row_permutation_invariant():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(75, 6))
    y = X @ rng.normal(size=6) + rng.normal(scale=0.3, size=75)
    Xq = rng.normal(size=(10, 6))
    ids = np.arange(8000, 8075)
    paid_ids = set(ids[:24].tolist())
    bulk_ids = set(ids[24:65].tolist())

    def run(X_, y_, ids_):
        paid = np.array([i for i, value in enumerate(ids_) if value in paid_ids])
        bulk = np.array([i for i, value in enumerate(ids_) if value in bulk_ids])

        def pseudo_builder(fit_paid_idx):
            return 0.4 * X_[:, 0] - 0.2 * X_[:, 1] + y_[fit_paid_idx].mean()

        return nested_ridge_fit_predict(
            X_, y_, Xq, paid, bulk, pseudo_target_builder=pseudo_builder,
            sample_ids=ids_, split_seed=67, return_info=True,
        )

    pred, info = run(X, y, ids)
    perm = rng.permutation(len(X))
    pred_perm, info_perm = run(X[perm], y[perm], ids[perm])
    assert info["lambda"] == info_perm["lambda"]
    assert set(info["paid_valid_ids"].tolist()) == set(info_perm["paid_valid_ids"].tolist())
    assert np.allclose(pred, pred_perm, rtol=0.0, atol=1e-12)


def test_replicate_sampling_is_independent_of_n_grid_order():
    pool = np.arange(900)
    # A single rep order is reused: every n cell is a prefix, regardless of requested traversal order.
    order = replicate_permutation(pool, seed=123, corpus="fresh", rep=4)
    ascending = {n: order[:n].copy() for n in (80, 160, 320)}
    descending = {n: replicate_permutation(pool, 123, "fresh", 4)[:n] for n in (320, 80, 160)}
    assert all(np.array_equal(ascending[n], descending[n]) for n in ascending)
    assert np.array_equal(ascending[80], ascending[320][:80])
    assert not np.array_equal(order, replicate_permutation(pool, 123, "fresh", 5))


def test_stable_seed_uses_cell_identity_not_call_order():
    expected = stable_seed(9, "fresh", 160, 2, "D")
    _ = [stable_seed(9, "expl", n, rep) for n in (640, 80) for rep in range(3)]
    assert stable_seed(9, "fresh", 160, 2, "D") == expected


def test_budget_and_sample_integration_for_reusable_order():
    order = replicate_permutation(np.arange(1000), seed=3, corpus="expl", rep=0)
    n, n_ov, k = 80, 30, 8
    want, used, truncated, spend, feasible = budget_plan(n, k, n_ov, avail=len(order) - n_ov)
    overlap = order[:n_ov]
    bulk = order[n_ov:n_ov + used]
    assert feasible and not truncated and want == used == 370
    assert len(np.intersect1d(overlap, bulk)) == 0
    assert len(overlap) * (1 + 1 / k) + len(bulk) / k == pytest.approx(spend)
    assert spend == pytest.approx(n)


def test_noninteger_k_is_rejected_in_cli_and_budget_api():
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        positive_integer("2.5")
    with pytest.raises(ValueError, match="positive integer"):
        budget_plan(80, 2.5, 30, 1000)


def test_free_tier_conditioner_accepts_two_measurement_channels():
    rng = np.random.default_rng(11)
    y = rng.uniform(0.1, 0.9, size=(48, 2))
    prior = np.clip(y + rng.normal(scale=0.16, size=y.shape), 0.0, 1.0)
    free_meas = np.clip(y + rng.normal(scale=0.10, size=y.shape), 0.0, 1.0)
    post = fused_targets(prior, free_meas, y, np.arange(32), H=H_FREE)
    assert post.shape == y.shape
    assert np.isfinite(post).all()
    assert ((0.0 <= post) & (post <= 1.0)).all()


def test_paired_delta_summary_uses_within_replicate_differences():
    # Large shared replicate effects cancel: every paired candidate gain is exactly +0.02.
    baseline = np.array([-0.7, 0.8, -0.2, 0.5, 0.1])
    candidate = baseline + 0.02
    out = paired_delta_summary(baseline, candidate, n_boot=1000, confidence=0.95, seed=4)
    assert out["n_pairs"] == 5
    assert out["mean"] == pytest.approx(0.02)
    assert out["ci_low"] == pytest.approx(0.02)
    assert out["ci_high"] == pytest.approx(0.02)


def test_paired_delta_summary_validates_pairing():
    with pytest.raises(ValueError, match="paired"):
        paired_delta_summary([0.1, 0.2], [0.3], n_boot=10)
