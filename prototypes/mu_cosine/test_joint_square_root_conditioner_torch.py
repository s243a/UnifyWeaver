#!/usr/bin/env python3
"""Correctness tests for the batched CPU/CUDA Householder-QR backend."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from joint_square_root_conditioner import (  # noqa: E402
    condition_correlated_gaussian_qr,
    householder_information_update,
    precision_root_from_covariance,
)
from joint_square_root_conditioner_torch import (  # noqa: E402
    CompiledCorrelatedConditionerTorch,
    CompiledDenseGainConditionerTorch,
    SquareRootInformationStateTorch,
    compile_information_update_torch,
    condition_correlated_gaussian_qr_torch,
    conditional_measurement_model_torch,
    householder_information_update_torch,
    precision_root_from_covariance_torch,
    prepare_noise_whitener_torch,
)


def _joint_spd(rng, n, m, floor=0.75):
    factor = rng.standard_normal((n + m, n + m))
    joint = factor @ factor.T + floor * np.eye(n + m)
    return joint[:n, :n], joint[:n, n:], joint[n:, n:]


def _problem(seed=7, n=4, m=5):
    rng = np.random.default_rng(seed)
    P, C, R = _joint_spd(rng, n, m)
    H = rng.standard_normal((m, n))
    mean = rng.standard_normal(n)
    observation = rng.standard_normal(m)
    return P, C, R, H, mean, observation


def _as_torch(*arrays, device="cpu", dtype=torch.float64):
    return tuple(torch.as_tensor(a, dtype=dtype, device=device) for a in arrays)


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 2e-5, 2e-6),
    (torch.float64, 2e-12, 2e-13),
])
def test_noise_whitener_correlation_and_covariance_are_equivalent(dtype, rtol, atol):
    correlation = torch.tensor(
        [[1.0, 0.25, -0.1], [0.25, 1.0, 0.2], [-0.1, 0.2, 1.0]],
        dtype=dtype,
    )
    stddev = torch.tensor([2.0, 0.5, 1.5], dtype=dtype)
    covariance = stddev[:, None] * correlation * stddev[None, :]
    from_correlation = prepare_noise_whitener_torch(
        correlation=correlation, stddev=stddev
    )
    from_covariance = prepare_noise_whitener_torch(covariance)
    torch.testing.assert_close(
        from_correlation.covariance, from_covariance.covariance, rtol=rtol, atol=atol
    )

    rhs = torch.arange(12, dtype=dtype).reshape(4, 3) / 7.0
    torch.testing.assert_close(
        from_correlation.apply_vectors(rhs),
        from_covariance.apply_vectors(rhs),
        rtol=rtol,
        atol=atol,
    )
    precision_root = from_correlation.materialize_row_root()
    identity = torch.eye(3, dtype=dtype)
    torch.testing.assert_close(
        precision_root.mT @ precision_root @ from_correlation.covariance,
        identity,
        rtol=10 * rtol,
        atol=10 * atol,
    )
    assert from_correlation.diagnostics.source == "correlation+stddev"
    assert from_covariance.diagnostics.source == "covariance"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noise_whitener_nearly_singular_psd_loading_is_scale_relative(dtype):
    scales = torch.tensor([1e-6, 1.0, 1e6], dtype=dtype)
    singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=dtype)
    covariance = scales[:, None, None] * singular
    whitener = prepare_noise_whitener_torch(covariance)
    diagnostics = whitener.diagnostics
    assert diagnostics.was_loaded.shape == (3,)
    assert bool(torch.all(diagnostics.was_loaded))
    assert bool(torch.all(diagnostics.diagonal_loading > 0))
    expected = torch.full_like(
        diagnostics.relative_diagonal_loading,
        diagnostics.relative_eigenvalue_floor,
    )
    assert diagnostics.relative_eigenvalue_floor == pytest.approx(
        max(
            np.sqrt(torch.finfo(dtype).eps),
            8.0 * 2 * torch.finfo(dtype).eps,
        )
    )
    torch.testing.assert_close(
        diagnostics.relative_diagonal_loading,
        expected,
        rtol=0.08,
        atol=8 * torch.finfo(dtype).eps,
    )
    root = whitener.materialize_row_root()
    identity = torch.eye(2, dtype=dtype).expand(3, 2, 2)
    torch.testing.assert_close(
        root.transpose(-2, -1) @ root @ whitener.covariance,
        identity,
        rtol=3e-3 if dtype == torch.float32 else 2e-9,
        atol=3e-3 if dtype == torch.float32 else 2e-9,
    )


def test_noise_whitener_rejects_indefinite_and_excessive_loading():
    indefinite = torch.tensor([[1.0, 0.0], [0.0, -0.05]], dtype=torch.float64)
    with pytest.raises(ValueError, match="genuinely indefinite"):
        prepare_noise_whitener_torch(indefinite)

    singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="exceeding maximum_relative_loading"):
        prepare_noise_whitener_torch(
            singular,
            relative_eigenvalue_floor=1e-3,
            maximum_relative_loading=1e-4,
        )


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noise_whitener_batched_dtype_device_and_solve(device, dtype):
    correlation = torch.tensor(
        [
            [[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]],
            [[1.0, -0.3, 0.05], [-0.3, 1.0, 0.15], [0.05, 0.15, 1.0]],
        ],
        dtype=dtype,
        device=device,
    )
    stddev = torch.tensor(
        [[1.0, 0.5, 2.0], [0.7, 1.4, 0.9]], dtype=dtype, device=device
    )
    whitener = prepare_noise_whitener_torch(
        correlation=correlation, stddev=stddev
    )
    values = torch.arange(24, dtype=dtype, device=device).reshape(4, 2, 3) / 11.0
    actual = whitener.apply_vectors(values)
    expected = torch.stack([
        torch.stack([
            torch.linalg.solve_triangular(
                whitener.cholesky_factor[batch],
                values[sample, batch, :, None],
                upper=False,
            ).squeeze(-1)
            for batch in range(2)
        ])
        for sample in range(4)
    ])
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    assert actual.device.type == device
    assert whitener.diagnostics.diagonal_loading.shape == (2,)


def test_noise_whitener_validates_correlation_and_positive_stddev():
    correlation = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_noise_whitener_torch(
            correlation=correlation,
            stddev=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="diagonal must be one"):
        prepare_noise_whitener_torch(
            correlation=torch.tensor([[0.9, 0.0], [0.0, 1.0]]),
            stddev=torch.ones(2),
        )
    with pytest.raises(ValueError, match=r"\|rho\| <= 1"):
        prepare_noise_whitener_torch(
            correlation=torch.tensor([[1.0, 1.01], [1.01, 1.0]]),
            stddev=torch.ones(2),
        )


def test_noise_whitener_never_calls_explicit_inverse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit inverse called")

    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    monkeypatch.setattr(torch, "inverse", forbidden)
    covariance = torch.tensor([[1.2, 0.2], [0.2, 0.8]], dtype=torch.float64)
    whitener = prepare_noise_whitener_torch(covariance)
    whitener.apply_vectors(torch.tensor([0.3, -0.4], dtype=torch.float64))
    whitener.apply_columns(torch.eye(2, dtype=torch.float64))
    whitener.materialize_row_root()


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_information_state_update_noise_block_matches_normal_equations(device, dtype):
    correlation = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.2, 1.0, 0.15], [-0.1, 0.15, 1.0]],
            [[1.0, -0.25, 0.0], [-0.25, 1.0, 0.1], [0.0, 0.1, 1.0]],
        ],
        dtype=dtype,
        device=device,
    )
    stddev = torch.tensor(
        [[0.8, 1.2, 0.6], [1.1, 0.7, 1.4]], dtype=dtype, device=device
    )
    whitener = prepare_noise_whitener_torch(
        correlation=correlation, stddev=stddev
    )
    matrix = torch.tensor(
        [
            [[1.0, 0.2], [0.1, -0.7], [0.4, 0.9]],
            [[0.5, -0.1], [0.3, 0.8], [-0.6, 0.2]],
        ],
        dtype=dtype,
        device=device,
    )
    rhs = torch.tensor(
        [[0.4, -0.2, 0.9], [-0.1, 0.7, 0.3]], dtype=dtype, device=device
    )
    root = torch.eye(2, dtype=dtype, device=device)
    state = SquareRootInformationStateTorch(
        root, torch.zeros(2, dtype=dtype, device=device)
    )
    _, actual = state.update_noise_block(matrix, rhs, whitener)

    whitened_matrix = whitener.apply_columns(matrix)
    whitened_rhs = whitener.apply_vectors(rhs)
    precision = (
        torch.eye(2, dtype=dtype, device=device).expand(2, 2, 2)
        + whitened_matrix.transpose(-2, -1) @ whitened_matrix
    )
    information = (
        whitened_matrix.transpose(-2, -1) @ whitened_rhs.unsqueeze(-1)
    )
    expected = torch.linalg.solve(precision, information).squeeze(-1)
    torch.testing.assert_close(actual.solution, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("dtype,scales", [
    (torch.float32, (1e-20, 1.0, 1e20)),
    (torch.float64, (1e-150, 1.0, 1e150)),
])
def test_information_qr_rank_check_is_scale_homogeneous(dtype, scales):
    for scale in scales:
        root = torch.eye(3, dtype=dtype) * scale
        compiled = compile_information_update_torch(
            root, torch.zeros(0, 3, dtype=dtype)
        )
        torch.testing.assert_close(
            compiled.precision_root / scale,
            torch.eye(3, dtype=dtype),
            rtol=20 * torch.finfo(dtype).eps,
            atol=20 * torch.finfo(dtype).eps,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_information_qr_rank_boundary_preserves_frobenius_sensitivity(dtype):
    n, m = 128, 32
    eps = torch.finfo(dtype).eps
    expected_frobenius_scale = math.sqrt(m * n)
    boundary = eps * (n + m) * expected_frobenius_scale
    measurement = torch.ones(m, n, dtype=dtype)

    with pytest.raises(torch.linalg.LinAlgError, match="rank deficient"):
        compile_information_update_torch(
            torch.eye(n, dtype=dtype) * (0.5 * boundary), measurement
        )

    accepted = compile_information_update_torch(
        torch.eye(n, dtype=dtype) * (2.0 * boundary), measurement
    )
    assert torch.isfinite(accepted.precision_root).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_information_qr_rank_scale_does_not_overflow_near_dtype_max(dtype):
    n = 16
    scale = torch.finfo(dtype).max / 2.0
    compiled = compile_information_update_torch(
        torch.eye(n, dtype=dtype) * scale,
        torch.zeros(0, n, dtype=dtype),
    )
    torch.testing.assert_close(
        compiled.precision_root / scale,
        torch.eye(n, dtype=dtype),
        rtol=20 * torch.finfo(dtype).eps,
        atol=20 * torch.finfo(dtype).eps,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_information_qr_rejects_subnormal_scale_with_rescale_message(dtype):
    scale = torch.finfo(dtype).tiny / 2.0
    with pytest.raises(torch.linalg.LinAlgError, match="subnormal; rescale"):
        compile_information_update_torch(
            torch.eye(2, dtype=dtype) * scale,
            torch.zeros(0, 2, dtype=dtype),
        )


def test_conditioner_exposes_loading_without_absolute_float32_eps_inflation():
    dtype = torch.float32
    root = torch.eye(2, dtype=dtype)
    H = torch.zeros(2, 2, dtype=dtype)
    R = torch.eye(2, dtype=dtype) * 1e-9
    C = torch.zeros(2, 2, dtype=dtype)
    mean = torch.zeros(2, dtype=dtype)
    observation = torch.ones(2, dtype=dtype) * 1e-4
    update = condition_correlated_gaussian_qr_torch(
        mean, root, observation, R, H, C
    )
    compiled = CompiledCorrelatedConditionerTorch.compile(root, H, R, C)
    torch.testing.assert_close(
        update.conditional_observation_covariance,
        R,
        rtol=2e-6,
        atol=1e-15,
    )
    assert not bool(update.conditional_loading_diagnostics.was_loaded)
    assert not bool(compiled.conditional_loading_diagnostics.was_loaded)


@pytest.mark.parametrize("dtype,scales", [
    (torch.float32, (1e-20, 1e-12, 1.0, 1e20)),
    (torch.float64, (1e-150, 1e-30, 1.0, 1e150)),
])
def test_default_covariance_roots_and_conditioner_are_scale_relative(dtype, scales):
    for scale in scales:
        covariance = torch.eye(2, dtype=dtype) * scale
        root = precision_root_from_covariance_torch(covariance)
        torch.testing.assert_close(
            root.mT @ root @ covariance,
            torch.eye(2, dtype=dtype),
            rtol=5e-5 if dtype == torch.float32 else 2e-12,
            atol=5e-6 if dtype == torch.float32 else 2e-13,
        )

        update = condition_correlated_gaussian_qr_torch(
            torch.zeros(2, dtype=dtype),
            torch.eye(2, dtype=dtype),
            torch.zeros(2, dtype=dtype),
            covariance,
            torch.zeros(2, 2, dtype=dtype),
            torch.zeros(2, 2, dtype=dtype),
        )
        torch.testing.assert_close(
            update.conditional_observation_covariance, covariance
        )
        assert not bool(update.conditional_loading_diagnostics.was_loaded)


def test_float32_large_n_negative_tolerance_fits_loading_budget():
    n = 128
    dtype = torch.float32
    maximum_loading = 1e-3
    eps = torch.finfo(dtype).eps
    relative_floor = max(math.sqrt(eps), 8.0 * n * eps)
    expected_negative_tolerance = min(
        64.0 * n * eps,
        0.5 * (maximum_loading - relative_floor),
    )
    covariance = torch.eye(n, dtype=dtype)
    covariance[-1, -1] = -0.9 * expected_negative_tolerance

    whitener = prepare_noise_whitener_torch(
        covariance, maximum_relative_loading=maximum_loading
    )
    assert whitener.diagnostics.negative_eigenvalue_tolerance == pytest.approx(
        expected_negative_tolerance
    )
    assert float(whitener.diagnostics.relative_diagonal_loading) < maximum_loading

    too_negative = covariance.clone()
    too_negative[-1, -1] = -1.1 * expected_negative_tolerance
    with pytest.raises(ValueError, match="genuinely indefinite"):
        prepare_noise_whitener_torch(
            too_negative, maximum_relative_loading=maximum_loading
        )


def test_zero_covariance_full_scale_repair_requires_explicit_budget():
    whitener = prepare_noise_whitener_torch(
        torch.zeros(2, 2, dtype=torch.float64),
        relative_eigenvalue_floor=0.0,
        absolute_eigenvalue_floor=1e-4,
        maximum_relative_loading=1.0,
    )
    torch.testing.assert_close(
        whitener.covariance, torch.eye(2, dtype=torch.float64) * 1e-4
    )
    assert float(whitener.diagnostics.relative_diagonal_loading) == pytest.approx(1.0)


def test_information_state_update_noise_block_rejects_wrong_whitener_type():
    state = SquareRootInformationStateTorch(torch.eye(2), torch.zeros(2))
    with pytest.raises(TypeError, match="NoiseWhitenerTorch"):
        state.update_noise_block(torch.eye(2), torch.zeros(2), object())


@pytest.mark.parametrize("n,m", [(1, 1), (2, 4), (4, 5)])
def test_cpu_float64_matches_numpy_correlated_conditioner(n, m):
    P, C, R, H, mean, observation = _problem(10 + n + m, n, m)
    root_np = precision_root_from_covariance(P, jitter=0.0)
    expected = condition_correlated_gaussian_qr(
        mean, root_np, observation, R, H, C, jitter=0.0
    )
    root, Ht, Rt, Ct, mt, yt = _as_torch(root_np, H, R, C, mean, observation)
    actual = condition_correlated_gaussian_qr_torch(
        mt, root, yt, Rt, Ht, Ct, jitter=0.0
    )

    np.testing.assert_allclose(actual.mean.numpy(), expected.mean, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(
        actual.covariance.numpy(), expected.covariance, rtol=3e-9, atol=3e-10
    )
    np.testing.assert_allclose(
        (actual.precision_root.mT @ actual.precision_root).numpy(),
        np.linalg.inv(expected.covariance),
        rtol=2e-8,
        atol=2e-9,
    )
    assert actual.mean.device.type == "cpu"
    assert actual.mean.dtype == torch.float64


def test_compiled_batch_rhs_matches_individual_numpy_updates():
    rng = np.random.default_rng(31)
    n, m, rhs_count = 4, 7, 9
    P = np.diag([0.5, 0.9, 1.7, 3.0])
    root = precision_root_from_covariance(P, jitter=0.0)
    A = rng.standard_normal((m, n))
    z = rng.standard_normal((n, rhs_count))
    b = rng.standard_normal((m, rhs_count))

    compiled = compile_information_update_torch(*_as_torch(root, A))
    actual = compiled.apply_columns(*_as_torch(z, b))
    for column in range(rhs_count):
        expected = householder_information_update(
            root, z[:, column], A, b[:, column]
        )
        np.testing.assert_allclose(
            actual.solution[:, column].numpy(), expected.solution, atol=2e-10
        )
        np.testing.assert_allclose(
            actual.information_rhs[:, column].numpy(),
            expected.information_rhs,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            actual.residual_sum_squares[column].numpy(),
            expected.residual_sum_squares,
            atol=2e-10,
        )


def test_fixed_design_compiles_once_and_conditions_arbitrary_batch_shape():
    P, C, R, H, mean, observation = _problem(41, 3, 5)
    root_np = precision_root_from_covariance(P, jitter=0.0)
    root, Ht, Rt, Ct = _as_torch(root_np, H, R, C)
    compiled = CompiledCorrelatedConditionerTorch.compile(
        root, Ht, Rt, Ct, jitter=0.0
    )

    rng = np.random.default_rng(42)
    means = rng.standard_normal((2, 4, 3))
    observations = rng.standard_normal((2, 4, 5))
    actual = compiled.condition(*_as_torch(means, observations))
    assert actual.mean.shape == (2, 4, 3)
    assert actual.covariance.shape == (2, 4, 3, 3)
    for index in np.ndindex(2, 4):
        expected = condition_correlated_gaussian_qr(
            means[index],
            root_np,
            observations[index],
            R,
            H,
            C,
            jitter=0.0,
        )
        np.testing.assert_allclose(actual.mean[index].numpy(), expected.mean, atol=3e-10)
        np.testing.assert_allclose(
            actual.covariance[index].numpy(), expected.covariance, atol=3e-10
        )


@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 2e-4, 2e-5),
    (torch.float64, 3e-9, 3e-10),
])
def test_compiled_dense_gain_matches_compiled_qr(dtype, rtol, atol):
    P, C, R, H, _, _ = _problem(47, 5, 7)
    rng = np.random.default_rng(48)
    means = rng.standard_normal((3, 11, 5))
    observations = rng.standard_normal((3, 11, 7))
    Pt, Ct, Rt, Ht, mt, yt = _as_torch(
        P, C, R, H, means, observations, dtype=dtype
    )
    root = precision_root_from_covariance_torch(Pt)
    qr = CompiledCorrelatedConditionerTorch.compile(
        root, Ht, Rt, Ct
    ).condition(mt, yt)
    dense = CompiledDenseGainConditionerTorch.compile(
        root, Ht, Rt, Ct
    ).condition(mt, yt)
    torch.testing.assert_close(dense.mean, qr.mean, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        dense.covariance, qr.covariance, rtol=rtol, atol=atol
    )


def test_compiled_conditioners_snapshot_coefficients_against_caller_mutation():
    P, C, R, H, mean, observation = _problem(48, 3, 5)
    Pt, Ct, Rt, Ht, mt, yt = _as_torch(P, C, R, H, mean, observation)
    root = precision_root_from_covariance_torch(Pt)
    qr = CompiledCorrelatedConditionerTorch.compile(root, Ht, Rt, Ct)
    dense = CompiledDenseGainConditionerTorch.compile(root, Ht, Rt, Ct)
    qr_before = qr.condition(mt, yt).mean.clone()
    dense_before = dense.condition(mt, yt).mean.clone()
    root.add_(100.0)
    Ht.add_(100.0)
    Rt.add_(100.0)
    Ct.add_(100.0)
    torch.testing.assert_close(qr.condition(mt, yt).mean, qr_before)
    torch.testing.assert_close(dense.condition(mt, yt).mean, dense_before)


@pytest.mark.parametrize("noise_scale", [1e-12, 1e-9])
def test_dense_gain_joseph_covariance_matches_qr_below_float32_epsilon(noise_scale):
    dtype = torch.float32
    P = torch.eye(2, dtype=dtype)
    H = torch.eye(2, dtype=dtype)
    R = torch.eye(2, dtype=dtype) * noise_scale
    C = torch.zeros(2, 2, dtype=dtype)
    root = precision_root_from_covariance_torch(P, jitter=0.0)
    dense = CompiledDenseGainConditionerTorch.compile(
        root, H, R, C, jitter=0.0
    )
    qr = CompiledCorrelatedConditionerTorch.compile(
        root, H, R, C, jitter=0.0
    )
    assert torch.all(torch.linalg.eigvalsh(dense.posterior_covariance) > 0)
    torch.testing.assert_close(
        dense.posterior_covariance,
        qr.posterior_covariance,
        rtol=5e-4,
        atol=noise_scale * 5e-5,
    )


def test_dense_gain_accepts_equivalent_nontriangular_precision_factor():
    P, C, R, H, mean, observation = _problem(49, 4, 6)
    root_np = precision_root_from_covariance(P, jitter=0.0)
    rng = np.random.default_rng(50)
    rotation, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    rotated_root = rotation @ root_np
    root, rotated, Ht, Rt, Ct, mt, yt = _as_torch(
        root_np, rotated_root, H, R, C, mean, observation
    )
    reference = CompiledDenseGainConditionerTorch.compile(
        root, Ht, Rt, Ct, jitter=0.0
    ).condition(mt, yt)
    actual = CompiledDenseGainConditionerTorch.compile(
        rotated, Ht, Rt, Ct, jitter=0.0
    ).condition(mt, yt)
    torch.testing.assert_close(actual.mean, reference.mean)
    torch.testing.assert_close(actual.covariance, reference.covariance)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ),
])
def test_batched_distinct_designs_match_unbatched_calls(device):
    rng = np.random.default_rng(53)
    batch, n, m = 4, 3, 5
    roots, matrices, prior_rhs, measurement_rhs = [], [], [], []
    for _ in range(batch):
        diagonal = np.exp(rng.standard_normal(n)) + 0.2
        roots.append(precision_root_from_covariance(np.diag(diagonal), jitter=0.0))
        matrices.append(rng.standard_normal((m, n)))
        prior_rhs.append(rng.standard_normal(n))
        measurement_rhs.append(rng.standard_normal(m))
    roots, matrices, prior_rhs, measurement_rhs = _as_torch(
        np.stack(roots),
        np.stack(matrices),
        np.stack(prior_rhs),
        np.stack(measurement_rhs),
        device=device,
    )
    actual = householder_information_update_torch(
        roots, prior_rhs, matrices, measurement_rhs
    )
    for index in range(batch):
        expected = householder_information_update_torch(
            roots[index], prior_rhs[index], matrices[index], measurement_rhs[index]
        )
        torch.testing.assert_close(actual.precision_root[index], expected.precision_root)
        torch.testing.assert_close(actual.information_rhs[index], expected.information_rhs)
        torch.testing.assert_close(actual.solution[index], expected.solution)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ),
])
def test_full_correlated_conditioner_batches_distinct_designs(device):
    batch, n, m = 4, 3, 5
    problems = [_problem(55 + index, n, m) for index in range(batch)]
    roots = [precision_root_from_covariance(item[0], jitter=0.0) for item in problems]
    root, H, R, C, mean, observation = _as_torch(
        np.stack(roots),
        np.stack([item[3] for item in problems]),
        np.stack([item[2] for item in problems]),
        np.stack([item[1] for item in problems]),
        np.stack([item[4] for item in problems]),
        np.stack([item[5] for item in problems]),
        device=device,
    )
    actual = condition_correlated_gaussian_qr_torch(
        mean, root, observation, R, H, C, jitter=0.0
    )
    for index in range(batch):
        expected = condition_correlated_gaussian_qr_torch(
            mean[index],
            root[index],
            observation[index],
            R[index],
            H[index],
            C[index],
            jitter=0.0,
        )
        torch.testing.assert_close(actual.mean[index], expected.mean)
        torch.testing.assert_close(actual.covariance[index], expected.covariance)


def test_batched_rank_failure_identifies_offending_design():
    roots = torch.stack([torch.eye(2), torch.zeros(2, 2)])
    matrices = torch.zeros(2, 1, 2)
    with pytest.raises(torch.linalg.LinAlgError, match=r"batch indices \[\[1\]\]"):
        householder_information_update_torch(
            roots,
            torch.zeros(2, 2),
            matrices,
            torch.zeros(2, 1),
        )


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
    ),
])
def test_sequential_state_threads_updated_root_and_matches_batch_update(device):
    rng = np.random.default_rng(61)
    n = 4
    root_np = precision_root_from_covariance(
        np.diag([0.6, 1.1, 1.9, 2.7]), jitter=0.0
    )
    A1, b1 = rng.standard_normal((3, n)), rng.standard_normal(3)
    A2, b2 = rng.standard_normal((6, n)), rng.standard_normal(6)
    root, A1t, b1t, A2t, b2t = _as_torch(root_np, A1, b1, A2, b2, device=device)
    initial = SquareRootInformationStateTorch(
        root, torch.zeros(n, dtype=root.dtype, device=root.device)
    )
    after_first, _ = initial.update(A1t, b1t)
    after_second, streamed = after_first.update(A2t, b2t)
    batch = householder_information_update_torch(
        root,
        torch.zeros(n, dtype=root.dtype, device=root.device),
        torch.cat([A1t, A2t]),
        torch.cat([b1t, b2t]),
    )
    torch.testing.assert_close(after_second.precision_root, batch.precision_root)
    torch.testing.assert_close(after_second.information_rhs, batch.information_rhs)
    torch.testing.assert_close(streamed.solution, batch.solution)


def test_sequential_recentring_resets_rhs_and_shifts_likelihood():
    rng = np.random.default_rng(63)
    n = 3
    root_np = precision_root_from_covariance(
        np.diag([0.7, 1.4, 2.2]), jitter=0.0
    )
    A1, b1 = rng.standard_normal((4, n)), rng.standard_normal(4)
    A2, b2 = rng.standard_normal((5, n)), rng.standard_normal(5)
    root, A1t, b1t, A2t, b2t = _as_torch(root_np, A1, b1, A2, b2)

    initial = SquareRootInformationStateTorch(root, torch.zeros(n, dtype=root.dtype))
    after_first, first = initial.update(A1t, b1t)
    _, global_coordinates = after_first.update(A2t, b2t)

    # In coordinates delta = e - e_hat_1, the posterior prior has zero RHS
    # and the second likelihood RHS is b2 - A2 e_hat_1.  Carrying z1 while
    # also making this shift would double-count the first update.
    recentered = SquareRootInformationStateTorch(
        after_first.precision_root,
        torch.zeros_like(after_first.information_rhs),
    )
    _, local_coordinates = recentered.update(
        A2t, b2t - A2t @ first.solution
    )
    torch.testing.assert_close(
        first.solution + local_coordinates.solution,
        global_coordinates.solution,
    )


def test_nonzero_cross_covariance_block_diagonal_rc_streams_exactly():
    rng = np.random.default_rng(65)
    n, split, m = 3, 2, 5
    P = np.diag([0.8, 1.3, 2.1])
    C = 0.08 * rng.standard_normal((n, m))
    Rc1 = np.array([[0.9, 0.12], [0.12, 1.2]])
    Rc2 = np.array([[1.1, 0.08, 0.0], [0.08, 0.8, 0.1], [0.0, 0.1, 1.4]])
    Rc = np.zeros((m, m))
    Rc[:split, :split] = Rc1
    Rc[split:, split:] = Rc2
    R = Rc + C.T @ np.linalg.solve(P, C)
    H = rng.standard_normal((m, n))
    mean = rng.standard_normal(n)
    observation = rng.standard_normal(m)
    root_np = precision_root_from_covariance(P, jitter=0.0)
    root, Ht, Rt, Ct, mt, yt = _as_torch(
        root_np, H, R, C, mean, observation
    )

    full = condition_correlated_gaussian_qr_torch(
        mt, root, yt, Rt, Ht, Ct, jitter=0.0
    )
    J, conditional_R = conditional_measurement_model_torch(
        root, Ht, Rt, Ct, jitter=0.0
    )
    innovation = yt - Ht @ mt
    blocks = []
    for block in (slice(0, split), slice(split, m)):
        chol = torch.linalg.cholesky(conditional_R[block, block])
        blocks.append((
            torch.linalg.solve_triangular(chol, J[block], upper=False),
            torch.linalg.solve_triangular(
                chol, innovation[block, None], upper=False
            ).squeeze(-1),
        ))
    state = SquareRootInformationStateTorch(root, torch.zeros(n, dtype=root.dtype))
    for A, b in blocks:
        state, streamed = state.update(A, b)
    torch.testing.assert_close(mt + streamed.solution, full.mean)
    torch.testing.assert_close(state.precision_root, full.precision_root)


def test_float32_device_and_dtype_are_preserved():
    P, C, R, H, mean, observation = _problem(71, 2, 4)
    Pt, Ct, Rt, Ht, mt, yt = _as_torch(
        P, C, R, H, mean, observation, dtype=torch.float32
    )
    root = precision_root_from_covariance_torch(Pt)
    compiled = CompiledCorrelatedConditionerTorch.compile(root, Ht, Rt, Ct)
    actual = compiled.condition(mt, yt)
    for value in (
        actual.mean,
        actual.covariance,
        actual.precision_root,
        actual.information_rhs,
    ):
        assert value.dtype == torch.float32
        assert value.device == Pt.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype,rtol,atol", [
    (torch.float32, 3e-4, 3e-5),
    (torch.float64, 3e-9, 3e-10),
])
def test_cuda_geqrf_ormqr_matches_cpu(dtype, rtol, atol):
    P, C, R, H, _, _ = _problem(83, 5, 8)
    rng = np.random.default_rng(84)
    means = rng.standard_normal((17, 5))
    observations = rng.standard_normal((17, 8))
    cpu_values = _as_torch(P, C, R, H, means, observations, dtype=dtype)
    Pt, Ct, Rt, Ht, mt, yt = cpu_values
    cpu_root = precision_root_from_covariance_torch(Pt)
    cpu = CompiledCorrelatedConditionerTorch.compile(
        cpu_root, Ht, Rt, Ct
    ).condition(mt, yt)
    cpu_dense = CompiledDenseGainConditionerTorch.compile(
        cpu_root, Ht, Rt, Ct
    ).condition(mt, yt)

    gpu_values = tuple(value.cuda() for value in cpu_values)
    Pg, Cg, Rg, Hg, mg, yg = gpu_values
    gpu_root = precision_root_from_covariance_torch(Pg)
    gpu = CompiledCorrelatedConditionerTorch.compile(
        gpu_root, Hg, Rg, Cg
    ).condition(mg, yg)
    gpu_dense = CompiledDenseGainConditionerTorch.compile(
        gpu_root, Hg, Rg, Cg
    ).condition(mg, yg)
    torch.cuda.synchronize()
    torch.testing.assert_close(gpu.mean.cpu(), cpu.mean, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        gpu.covariance.cpu(), cpu.covariance, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        gpu_dense.mean.cpu(), cpu_dense.mean, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        gpu_dense.covariance.cpu(), cpu_dense.covariance, rtol=rtol, atol=atol
    )
    assert gpu.mean.device.type == "cuda"
    assert gpu.mean.dtype == dtype
