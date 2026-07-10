#!/usr/bin/env python3
"""Tests for the frozen public Product-Kalman holdout runner."""

import numpy as np

from run_product_kalman_public_holdout import (
    affine_calibrate_graph,
    promotion_decision,
    source_diagnostics,
    strict_branch_identity_split,
    valid_split,
)


def row(index, branch, descendant, ancestor, family="directional", hop=1):
    return {
        "pair_id": f"p{index}",
        "branch_unit": branch,
        "descendant_identity": descendant,
        "ancestor_identity": ancestor,
        "operator_family": family,
        "hop": str(hop),
    }


def test_branch_split_is_deterministic_and_identity_disjoint():
    rows = [
        row(0, "a", "i0", "shared"),
        row(1, "b", "shared", "i1"),
        row(2, "c", "i2", "i3"),
        row(3, "d", "i4", "i5"),
    ]
    cal1, eval1, manifest1 = strict_branch_identity_split(rows, seed=3)
    cal2, eval2, manifest2 = strict_branch_identity_split(rows, seed=3)
    assert np.array_equal(cal1, cal2)
    assert np.array_equal(eval1, eval2)
    assert manifest1 == manifest2
    cal_ids = {value for i in cal1 for value in (rows[i]["descendant_identity"], rows[i]["ancestor_identity"])}
    eval_ids = {value for i in eval1 for value in (rows[i]["descendant_identity"], rows[i]["ancestor_identity"])}
    assert not cal_ids & eval_ids
    assert manifest1["omitted_identity_overlap_components"] in (0, 1)


def test_constant_graph_affine_calibration_reduces_to_calibration_mean():
    calibrated, fit = affine_calibrate_graph([1, 1, 1], [0.2, 0.5, 0.8], [1, 1])
    assert np.allclose(calibrated, 0.5)
    assert fit == {"slope": 0.0, "intercept": 0.5}


def test_valid_split_requires_all_hops_and_family_coverage():
    rows = []
    for index in range(120):
        rows.append(row(
            index,
            f"b{index}",
            f"d{index}",
            f"a{index}",
            family=("directional", "symmetric", "open_world")[index % 3],
            hop=index % 5 + 1,
        ))
    cal = np.arange(80)
    evaluation = np.arange(80, 120)
    assert valid_split(rows, cal, evaluation)[0]
    for index in evaluation:
        rows[index]["hop"] = "1"
    valid, reasons = valid_split(rows, cal, evaluation)
    assert not valid
    assert "evaluation_missing_hop" in reasons


def test_source_diagnostics_marks_constant_source_correlation_unknown():
    data = {
        "raw_sources": np.array([
            [0.1, 0.2, 0.3, 0.4, 1.0, 1.0],
            [0.2, 0.3, 0.4, 0.5, 1.0, 2.0],
            [0.3, 0.4, 0.5, 0.6, 1.0, 3.0],
        ]),
        "family": np.array(["directional", "symmetric", "open_world"]),
    }
    result = source_diagnostics(data)
    assert result["correlation_matrix"][4] == [None] * 6
    assert result["separability"]["graph_measurement"] == 0.0


def score(nll, pit=0.1, coverage=0.1, mahal=1.0):
    return {
        "mean_nll": nll,
        "pit_mean_ks": pit,
        "coverage_mean_absolute_error": coverage,
        "mahalanobis_per_dim": mahal,
    }


def category(log_loss, ece, aurc, high):
    return {
        "log_loss": log_loss,
        "ece_10_equal_width": ece,
        "aurc_margin": aurc,
        "aurc_ci_high": high,
    }


def test_promotion_decision_requires_every_frozen_axis():
    result = {
        "continuous": {
            "prior": score(1.0),
            "independent_kalman": score(0.9),
            "hop_product_kalman": score(0.7),
        },
        "continuous_gains": {
            "prior_to_hop_product_kalman": {"ci_low": 0.1},
            "independent_kalman_to_hop_product_kalman": {"ci_low": 0.05},
        },
        "categorical": {
            "joint_baseline": category(1.0, 0.2, 0.3, 0.4),
            "joint_plus_hop": category(0.9, 0.1, 0.2, 0.25),
        },
    }
    assert promotion_decision(result)["promote"]
    result["categorical"]["joint_plus_hop"]["aurc_ci_high"] = 0.31
    decision = promotion_decision(result)
    assert not decision["promote"]
    assert any("AURC" in reason for reason in decision["reasons"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} public holdout tests passed")
