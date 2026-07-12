#!/usr/bin/env python3
"""Runner-level regressions for the descriptive adjacent residual pilot."""
from types import SimpleNamespace

import pytest

from run_adjacent_residual_pilot import (
    ADJACENCY_ALPHAS,
    _content_provenance,
    _portable_artifact_provenance,
    _scientific_configuration,
    _validate_args,
    _write_payload,
    aggregate_results,
)
from run_adjacent_residual_synthetic import (
    _file_record as synthetic_file_record,
    _scientific_configuration as synthetic_scientific_configuration,
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


def test_content_provenance_is_portable_across_identical_files(tmp_path):
    first = tmp_path / "first" / "input.bin"
    second = tmp_path / "second" / "renamed.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"portable-input")
    second.write_bytes(b"portable-input")

    expected = {"size_bytes": 14, "sha256": synthetic_file_record(first)["sha256"]}
    assert synthetic_file_record(first) == expected
    assert synthetic_file_record(second) == expected
    assert _content_provenance(first) == expected
    assert "path" not in expected


def test_scientific_configuration_excludes_runtime_locators():
    scientific = dict(
        assignments=10,
        folds=5,
        ridge_grid=[0.1, 1.0],
        stability_confidence=0.95,
    )
    first = SimpleNamespace(
        **scientific,
        artifact_repo="/checkout/a",
        ckpt="/models/a.pt",
        campaign="/data/a.tsv",
        luna="/data/luna-a.tsv",
        out="/tmp/a.json",
        resume=False,
    )
    second = SimpleNamespace(
        **scientific,
        artifact_repo="/checkout/b",
        ckpt="/models/b.pt",
        campaign="/data/b.tsv",
        luna="/data/luna-b.tsv",
        out="/other/b.json",
        resume=True,
    )
    assert (
        _scientific_configuration(first)
        == _scientific_configuration(second)
        == scientific
    )

    synthetic_first = SimpleNamespace(replicates=2, seed=7, out="/tmp/a.json")
    synthetic_second = SimpleNamespace(replicates=2, seed=7, out="/other/b.json")
    assert (
        synthetic_scientific_configuration(synthetic_first)
        == synthetic_scientific_configuration(synthetic_second)
        == {"replicates": 2, "seed": 7}
    )


def test_artifact_provenance_drops_repository_and_directory_locators():
    record = {
        "repository": "/checkout/private",
        "exploratory_graph": {
            "path": "/checkout/private/graph.tsv",
            "size_bytes": 10,
            "sha256": "a",
        },
        "fresh_lmdb_directory": "/checkout/private/lmdb",
        "fresh_lmdb_data": {
            "path": "/checkout/private/lmdb/data.mdb",
            "size_bytes": 20,
            "sha256": "b",
        },
        "fresh_lmdb_lock_excluded": "lock.mdb is process state",
    }
    portable = _portable_artifact_provenance(record)
    assert portable == {
        "exploratory_graph": {"size_bytes": 10, "sha256": "a"},
        "fresh_lmdb_data": {"size_bytes": 20, "sha256": "b"},
        "fresh_lmdb_lock_excluded": "lock.mdb is process state",
    }
    assert "/checkout/private" not in repr(portable)
