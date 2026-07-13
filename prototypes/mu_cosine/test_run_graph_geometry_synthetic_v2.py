#!/usr/bin/env python3
"""Focused tests for matched-coupling graph-geometry v2."""
from types import SimpleNamespace

import numpy as np
import pytest

from run_graph_geometry_synthetic import candidate_kernels
from run_graph_geometry_synthetic_v2 import (
    _validate_args,
    matched_correlation,
    maximum_off_diagonal,
    prepare_matched_candidates,
    run_benchmark,
)


def _args(**overrides):
    values = dict(
        replicates=6,
        calibration_draws=20,
        train_fields=8,
        held_fields=8,
        confidence=0.9,
        seed=22,
        out="/tmp/runtime-only.json",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_matched_correlation_equalizes_maximum_off_diagonal_coupling():
    _nodes, kernels = candidate_kernels()
    amplitudes = []
    for kernel in kernels.values():
        covariance, amplitude = matched_correlation(kernel, 0.1)
        amplitudes.append(amplitude)
        assert maximum_off_diagonal(covariance) == pytest.approx(0.1)
        assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert max(amplitudes) / min(amplitudes) > 3.0


def test_matched_candidate_keys_use_rho_not_common_alpha():
    _nodes, kernels = candidate_kernels()
    prepared, amplitudes = prepare_matched_candidates(kernels)
    assert ("block", 0.0) in prepared
    assert ("walk_decay", 0.2) in prepared
    assert amplitudes["walk_decay@0.2"] > amplitudes["closed@0.2"]


def test_small_v2_benchmark_is_deterministic_and_hard_gated():
    first = run_benchmark(_args())
    second = run_benchmark(_args(out="/different/path.json"))
    assert first == second
    assert not first["real_covariance_gate_unlocked"]
    assert not first["batching_gate_unlocked"]
    assert not first["qr_deployment_unlocked"]
    assert len(first["scenarios"]) == 11


def test_v2_argument_validation():
    _validate_args(_args())
    with pytest.raises(ValueError, match="positive"):
        _validate_args(_args(held_fields=0))
