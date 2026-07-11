#!/usr/bin/env python3
"""Correctness tests for the batched CPU/CUDA Householder-QR backend."""

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


def test_compiled_conditioners_snapshot_H_against_caller_mutation():
    P, C, R, H, mean, observation = _problem(48, 3, 5)
    Pt, Ct, Rt, Ht, mt, yt = _as_torch(P, C, R, H, mean, observation)
    root = precision_root_from_covariance_torch(Pt)
    qr = CompiledCorrelatedConditionerTorch.compile(root, Ht, Rt, Ct)
    dense = CompiledDenseGainConditionerTorch.compile(root, Ht, Rt, Ct)
    qr_before = qr.condition(mt, yt).mean.clone()
    dense_before = dense.condition(mt, yt).mean.clone()
    Ht.add_(100.0)
    torch.testing.assert_close(qr.condition(mt, yt).mean, qr_before)
    torch.testing.assert_close(dense.condition(mt, yt).mean, dense_before)


def test_dense_gain_keeps_positive_covariance_under_float32_cancellation():
    dtype = torch.float32
    P = torch.eye(2, dtype=dtype)
    H = torch.eye(2, dtype=dtype)
    R = torch.eye(2, dtype=dtype) * 1e-9
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
        rtol=2e-6,
        atol=torch.finfo(dtype).eps,
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


def test_batched_distinct_designs_match_unbatched_calls():
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


def test_full_correlated_conditioner_batches_distinct_designs():
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


def test_sequential_state_threads_updated_root_and_matches_batch_update():
    rng = np.random.default_rng(61)
    n = 4
    root_np = precision_root_from_covariance(
        np.diag([0.6, 1.1, 1.9, 2.7]), jitter=0.0
    )
    A1, b1 = rng.standard_normal((3, n)), rng.standard_normal(3)
    A2, b2 = rng.standard_normal((6, n)), rng.standard_normal(6)
    root, A1t, b1t, A2t, b2t = _as_torch(root_np, A1, b1, A2, b2)
    initial = SquareRootInformationStateTorch(root, torch.zeros(n, dtype=root.dtype))
    after_first, _ = initial.update(A1t, b1t)
    after_second, streamed = after_first.update(A2t, b2t)
    batch = householder_information_update_torch(
        root,
        torch.zeros(n, dtype=root.dtype),
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
