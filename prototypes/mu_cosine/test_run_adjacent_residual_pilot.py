#!/usr/bin/env python3
"""Runner-level regressions for the descriptive adjacent residual pilot."""
from types import SimpleNamespace

import pytest

from run_adjacent_residual_pilot import (
    ADJACENCY_ALPHAS,
    _validate_args,
    _write_payload,
    aggregate_results,
)


def _row(corpus, value, evaluable):
    curve = [{
        "alpha": alpha,
        "adjacency_gain_vs_block_component_macro": alpha / 100.0,
        "deranged_gain_vs_block_component_macro": alpha / 200.0,
    } for alpha in ADJACENCY_ALPHAS]
    return {
        "corpus": corpus,
        "primary_trace_over_4": value,
        "stability": {
            "gate_evaluable": evaluable,
            "leave_one_component_out_positive_fraction": 1.0,
            "by_stat": {
                "trace_over_4": {"pointwise_low": value - 0.01, "simultaneous_low": value - 0.02}
            },
        },
        "nondeployable_held_alpha_grid": {"component_macro_curve": curve},
    }


def test_frozen_runner_requires_exactly_five_folds():
    base = dict(
        assignments=1,
        maximum_controls=3,
        multiplier_draws=99,
        stability_confidence=0.95,
    )
    _validate_args(SimpleNamespace(folds=5, **base))
    with pytest.raises(ValueError, match="exactly five"):
        _validate_args(SimpleNamespace(folds=4, **base))


def test_aggregate_keeps_held_alpha_curve_nondeployable():
    payload = aggregate_results([
        _row("exploratory", 0.20, False),
        _row("fresh", 0.10, True),
    ], expected_assignments=1)

    assert payload["complete"]
    assert not payload["advance_to_repeated_judge_confirmation"]
    assert not payload["qr_covariance_deployment"]
    assert not payload["by_corpus"]["exploratory"]["all_stability_gates_evaluable"]
    assert payload["by_corpus"]["fresh"]["ci_like_positive_assignment_counts"]["pointwise"] == 1
    assert payload["by_corpus"]["exploratory"]["ci_like_positive_assignment_counts"] is None
    assert len(payload["by_corpus"]["fresh"]["nondeployable_held_alpha_curve"]) == len(
        ADJACENCY_ALPHAS
    )


def test_payload_writer_is_byte_deterministic_and_rejects_nan(tmp_path):
    output = tmp_path / "pilot.json"
    _write_payload(output, {"z": 1, "a": [2]})
    first = output.read_bytes()
    _write_payload(output, {"a": [2], "z": 1})
    assert output.read_bytes() == first
    with pytest.raises(ValueError, match="Out of range float"):
        _write_payload(output, {"bad": float("nan")})
    assert output.read_bytes() == first
