#!/usr/bin/env python3
"""Regressions for versioned family-wise synthetic selection."""
import json

import numpy as np
import pytest

from covariance_sensitivity import (
    finite_null_maximum_threshold,
    maximum_eligible_macro_gain,
    select_nested_candidate,
    select_nested_candidate_null_calibrated,
)
from run_covariance_sensitivity_synthetic import SCENARIOS, SyntheticGeometry
from run_covariance_sensitivity_synthetic_v2 import (
    SEARCH_CANONICAL,
    SEARCH_FULL,
    SEARCH_SINGLE,
    _complete_control_gate,
    _filter_search,
    _full_records,
    _procedure_scorers,
    _threshold_ratio,
    _write_scientific_payload,
    calibrate_nulls,
    run_mechanism_replicate_pair,
    run_replicate_pair,
)


def _records(gain):
    rows = []
    for fold in range(3):
        rows.append({
            "fold": fold,
            "alpha": 0.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 1.0,
            "nll_per_scalar": 1.0,
        })
        rows.append({
            "fold": fold,
            "alpha": 0.1,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 0.0,
            "nll_per_scalar": 1.0 - gain,
        })
    return rows


def test_finite_order_statistic_and_strict_familywise_gate():
    threshold, rank = finite_null_maximum_threshold([0.0, 0.1, 0.2, 0.3], confidence=0.5)
    assert rank == 3
    assert threshold == pytest.approx(0.2)

    records = _records(0.15)
    v1, summaries = select_nested_candidate(records)
    assert v1["alpha"] == pytest.approx(0.1)
    assert maximum_eligible_macro_gain(summaries) == pytest.approx(0.15)

    selected, _, calibration = select_nested_candidate_null_calibrated(
        records, [0.15] * 20
    )
    assert selected["alpha"] == 0.0  # equality does not reject
    assert not calibration["strictly_exceeds_threshold"]
    selected, _, calibration = select_nested_candidate_null_calibrated(
        records, [0.10] * 20
    )
    assert selected["alpha"] == pytest.approx(0.1)
    assert calibration["strictly_exceeds_threshold"]


def test_partial_or_empty_control_sets_cannot_pass_vacuously():
    required = {"block_null", "regional_mean_only"}
    assert _complete_control_gate(required, {"block_null"}, [True]) == (False, False)
    assert _complete_control_gate(required, required, []) == (True, False)
    assert _complete_control_gate(required, required, [True, True]) == (True, True)


def test_zero_fixed_path_threshold_has_undefined_safe_ratio():
    assert _threshold_ratio(0.0, 0.0) is None
    assert _threshold_ratio(0.25, 0.0) is None
    assert _threshold_ratio(0.25, 0.10) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="nonnegative"):
        _threshold_ratio(0.25, -0.10)


def test_scientific_artifact_is_deterministic_and_excludes_runtime(tmp_path):
    output = tmp_path / "artifact.json"
    _write_scientific_payload(output, {"z": 1.0, "a": [2, 3]})
    first = output.read_bytes()
    _write_scientific_payload(output, {"a": [2, 3], "z": 1.0})

    assert output.read_bytes() == first
    assert json.loads(first) == {"a": [2, 3], "z": 1.0}
    with pytest.raises(ValueError, match="stdout"):
        _write_scientific_payload(output, {"wall_seconds": 1.23})
    with pytest.raises(ValueError, match="Out of range float"):
        _write_scientific_payload(output, {"not_finite": float("nan")})
    assert output.read_bytes() == first


def test_search_filters_have_frozen_candidate_capacities():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    scenario = next(value for value in SCENARIOS if value.name == "block_null")
    field = np.random.default_rng(3).normal(size=(geometry.item_count, 4))
    full = _full_records(
        geometry,
        scenario,
        _procedure_scorers(geometry, field, 0.05, "regional_krr"),
    )
    # Three block rows plus candidate count times three folds.
    assert len(_filter_search(full, SEARCH_FULL)) == 3 + 216 * 3
    assert len(_filter_search(full, SEARCH_CANONICAL)) == 3 + 8 * 3
    assert len(_filter_search(full, SEARCH_SINGLE)) == 3 + 1 * 3


def test_paired_fixed_and_full_procedure_calibrations_are_deterministic():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    first_arrays, first_report = calibrate_nulls(
        geometry, draws=4, seed=5100, shrinkage=0.05
    )
    second_arrays, second_report = calibrate_nulls(
        geometry, draws=4, seed=5100, shrinkage=0.05
    )
    assert first_report == second_report
    for null_name in ("fixed_path_shared_z", "full_procedure_krr"):
        for search in (SEARCH_FULL, SEARCH_CANONICAL, SEARCH_SINGLE):
            np.testing.assert_array_equal(
                first_arrays[null_name][search], second_arrays[null_name][search]
            )


def test_end_to_end_omits_invalid_pre_krr_truth_but_mechanism_truth_is_exact():
    geometry = SyntheticGeometry(32, outer_held_count=8, inner_held_count=8)
    scenario = next(value for value in SCENARIOS if value.name == "block_null")
    never_reject = {
        SEARCH_FULL: np.ones(20),
        SEARCH_CANONICAL: np.ones(20),
        SEARCH_SINGLE: np.ones(20),
    }
    end = run_replicate_pair(
        geometry, scenario, replicate=0, seed=6200, shrinkage=0.05,
        full_procedure_null_maxima=never_reject,
    )
    for record in end.values():
        assert "known_true_covariance" not in record["residual_nll_per_scalar"]
        assert "known_true_covariance" not in record["posterior_state_mse"]
        assert "e_H-W e_T" in record["truth_scope"]

    mechanism = run_mechanism_replicate_pair(
        geometry, scenario, replicate=0, seed=6200,
        fixed_path_null_maxima=never_reject,
    )
    for record in mechanism.values():
        assert record["selection"]["alpha"] == 0.0
        assert record["residual_nll_per_scalar"]["known_true_covariance"] == pytest.approx(
            record["residual_nll_per_scalar"]["block"], abs=2e-15
        )
        assert record["posterior_state_mse"]["known_true_covariance"] == pytest.approx(
            record["posterior_state_mse"]["block"], abs=2e-15
        )
