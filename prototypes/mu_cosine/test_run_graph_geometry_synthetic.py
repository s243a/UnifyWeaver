#!/usr/bin/env python3
"""Focused tests for the graph-geometry mechanism runner."""
from types import SimpleNamespace

import numpy as np
import pytest

from run_graph_geometry_synthetic import (
    _validate_args,
    benchmark_graph,
    calibrate_familywise_threshold,
    candidate_kernels,
    correlation_path,
    draw_fields,
    mean_nll_per_scalar,
    prepare_candidates,
    prepare_gaussian,
    run_benchmark,
    select_with_threshold,
)


def _args(**overrides):
    values = dict(
        replicates=8,
        calibration_draws=30,
        train_fields=6,
        held_fields=8,
        confidence=0.90,
        seed=1234,
        out="/tmp/not-scientific.json",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_benchmark_graph_and_candidate_set_are_deterministic_psd():
    nodes, neighbors = benchmark_graph()
    second_nodes, second = benchmark_graph()
    assert nodes == second_nodes and neighbors == second
    returned, kernels = candidate_kernels()
    assert returned == nodes
    assert set(kernels) == {"closed", "walk_decay", "heat", "resolvent", "deranged_walk"}
    for kernel in kernels.values():
        assert np.allclose(kernel, kernel.T)
        assert np.allclose(np.diag(kernel), 1.0)
        assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-10


def test_draws_and_nll_prefer_the_generating_covariance():
    _nodes, kernels = candidate_kernels()
    truth = correlation_path(kernels["walk_decay"], 0.35)
    fields = draw_fields(truth, 4000, np.random.default_rng(7))
    assert np.allclose(np.cov(fields, rowvar=False), truth, atol=0.08)
    true_nll = mean_nll_per_scalar(fields, prepare_gaussian(truth))
    block_nll = mean_nll_per_scalar(fields, prepare_gaussian(np.eye(len(truth))))
    assert true_nll < block_nll


def test_familywise_threshold_is_deterministic_and_can_force_block_fallback():
    _nodes, kernels = candidate_kernels()
    prepared = prepare_candidates(kernels)
    first, maxima = calibrate_familywise_threshold(
        prepared, train_fields=5, draws=30, seed=99, confidence=0.9
    )
    second, _ = calibrate_familywise_threshold(
        prepared, train_fields=5, draws=30, seed=99, confidence=0.9
    )
    assert first == second == pytest.approx(np.quantile(maxima, 0.9))
    fields = np.zeros((5, len(next(iter(kernels.values())))))
    selected, _scores = select_with_threshold(fields, prepared, threshold=float("inf"))
    assert selected == ("block", 0.0)


def test_small_benchmark_is_deterministic_and_never_unlocks_real_or_qr():
    first = run_benchmark(_args())
    second = run_benchmark(_args(out="/different/runtime/path.json"))
    assert first == second
    assert first["status"].startswith("KNOWN-MEAN/KNOWN-B")
    assert not first["real_covariance_gate_unlocked"]
    assert not first["qr_deployment_unlocked"]
    assert len(first["scenarios"]) == 11


def test_argument_validation():
    _validate_args(_args())
    with pytest.raises(ValueError, match="positive"):
        _validate_args(_args(replicates=0))
    with pytest.raises(ValueError, match="at least 10"):
        _validate_args(_args(calibration_draws=9))
