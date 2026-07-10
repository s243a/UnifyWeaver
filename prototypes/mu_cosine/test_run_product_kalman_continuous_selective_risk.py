#!/usr/bin/env python3
"""Focused tests for continuous Product-Kalman selective risk."""

import numpy as np

from run_product_kalman_continuous_selective_risk import (
    SelectiveRiskError,
    average_ranks,
    decision,
    permutation_test,
    selective_metrics,
    tie_averaged_risk_curve,
)


def assert_raises(fn, *args):
    try:
        fn(*args)
    except SelectiveRiskError:
        return
    raise AssertionError("expected SelectiveRiskError")


def test_average_ranks_are_tie_aware():
    np.testing.assert_allclose(average_ranks([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])


def test_tie_averaged_curve_is_invariant_to_order_inside_blocks():
    risk = np.array([0.1, 0.1, 0.2, 0.2])
    loss_a = np.array([1.0, 3.0, 10.0, 6.0])
    loss_b = np.array([3.0, 1.0, 6.0, 10.0])
    np.testing.assert_allclose(
        tie_averaged_risk_curve(risk, loss_a),
        tie_averaged_risk_curve(risk, loss_b),
    )
    np.testing.assert_allclose(tie_averaged_risk_curve(risk, loss_a), [2.0, 2.0, 4.0, 5.0])


def test_selective_metrics_reward_correct_risk_ordering():
    loss = np.array([0.1, 0.2, 1.0, 2.0])
    good = selective_metrics(loss, loss)
    bad = selective_metrics(loss[::-1], loss)
    assert abs(good["spearman"] - 1.0) < 1e-12
    assert abs(bad["spearman"] + 1.0) < 1e-12
    assert good["normalized_aurc"] < 1.0 < bad["normalized_aurc"]


def test_permutation_p_values_use_plus_one_correction():
    loss = np.arange(1.0, 13.0)
    result = permutation_test(loss, loss, permutations=99, seed=0)
    assert 0.01 <= result["spearman_p_one_sided"] <= 1.0
    assert 0.01 <= result["aurc_p_one_sided"] <= 1.0


def test_decision_requires_every_frozen_axis():
    primary = {
        "trace": {"spearman": 0.4},
        "permutation": {"spearman_p_one_sided": 0.001, "aurc_p_one_sided": 0.001},
        "bootstrap": {"normalized_aurc_ci_high": 0.9},
    }
    stability = {"positive_spearman_fraction": 0.8, "normalized_aurc_below_one_fraction": 0.8}
    assert decision(primary, stability)["eligible"]
    primary["bootstrap"]["normalized_aurc_ci_high"] = 1.0
    assert not decision(primary, stability)["eligible"]


def test_invalid_vectors_fail_fast():
    assert_raises(selective_metrics, [1.0], [1.0, 2.0])
    assert_raises(selective_metrics, [-1.0], [1.0])
    assert_raises(selective_metrics, [1.0], [0.0])


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} continuous selective-risk tests passed")


if __name__ == "__main__":
    main()
