#!/usr/bin/env python3
"""Focused tests for the cheap-judge-pipeline blockers (PR #3648).

Blocker 7: correlated_update_H now delegates to product_kalman.gaussian_condition_update (Cholesky solves),
so we pin its algebra against hand-computed analytic cases (scalar + 2x2, both with NONZERO cross-covariance
C) and against an independent explicit-inverse reference for the realistic non-square-H shape.
Blocker 2: the matched-cost budget_plan is checked for the realized-spend invariant, the 30-row floor,
truncation, and infeasibility.

Run: `python3 test_cheap_judge_blockers.py`  (also collected by pytest).
"""
import numpy as np

from run_judge_channel import correlated_update_H
from sim_matched_cost import budget_plan


def _ref_inv(x, P, y, R, C, H):
    """Independent explicit-inverse reference for the correlated update (the pre-blocker-7 formula)."""
    S = H @ P @ H.T + R + H @ C + (H @ C).T
    K = (P @ H.T + C) @ np.linalg.inv(S)
    xp = x + K @ (y - H @ x)
    Pp = P - K @ S @ K.T
    return xp, Pp


# ----------------------------------------------------------------------------- blocker 7

def test_scalar_nonzero_C_analytic():
    # n=k=1, H=[[1]]: S=P+R+2C, K=(P+C)/S, xp=x+K(y-x), Pp=P-K^2 S.
    x = np.array([0.0]); P = np.array([[2.0]]); R = np.array([[1.0]]); C = np.array([[0.5]])
    y = np.array([1.0]); H = np.array([[1.0]])
    xp, Pp = correlated_update_H(x, P, y, R, C, H)
    # S=4, K=0.625, xp=0.625, Pp=2-0.625^2*4=0.4375
    assert np.allclose(xp, [0.625], atol=1e-9), xp
    assert np.allclose(Pp, [[0.4375]], atol=1e-9), Pp


def test_2x2_nonzero_C_analytic():
    # P=I, R=I, C=0.5I, H=I => S=3I, K=0.5I, xp=0.5*y, Pp=0.25I.
    P = np.eye(2); R = np.eye(2); C = 0.5 * np.eye(2); H = np.eye(2)
    x = np.array([0.0, 0.0]); y = np.array([1.0, 2.0])
    xp, Pp = correlated_update_H(x, P, y, R, C, H)
    assert np.allclose(xp, [0.5, 1.0], atol=1e-9), xp
    assert np.allclose(Pp, 0.25 * np.eye(2), atol=1e-9), Pp


def test_matches_inv_reference_mapped_H():
    # Realistic shape: 2-D state, 3 measurements mapping [graph->D, judgeD->D, judgeS->S], nonzero C.
    rng = np.random.default_rng(3)
    A = rng.standard_normal((2, 2)); P = A @ A.T + np.eye(2)          # SPD prior
    B = rng.standard_normal((3, 3)); R = B @ B.T + np.eye(3)          # SPD meas noise
    C = 0.1 * rng.standard_normal((2, 3))                             # state<->meas cross-cov
    H = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = rng.standard_normal(2); y = rng.standard_normal(3)
    xp, Pp = correlated_update_H(x, P, y, R, C, H)
    xr, Pr = _ref_inv(x, P, y, R, C, H)
    assert np.allclose(xp, xr, atol=1e-8), (xp, xr)
    assert np.allclose(Pp, Pr, atol=1e-8), (Pp, Pr)
    assert np.allclose(Pp, Pp.T, atol=1e-10)                         # symmetric posterior cov
    assert np.linalg.eigvalsh(Pp)[0] > -1e-9                         # PSD


def test_zero_C_reduces_to_standard_kalman():
    P = np.array([[2.0, 0.3], [0.3, 1.0]]); R = np.array([[0.5, 0.0], [0.0, 0.5]])
    C = np.zeros((2, 2)); H = np.eye(2)
    x = np.array([0.1, -0.2]); y = np.array([0.7, 0.4])
    xp, Pp = correlated_update_H(x, P, y, R, C, H)
    S = P + R
    K = P @ np.linalg.inv(S)
    assert np.allclose(xp, x + K @ (y - x), atol=1e-9)
    assert np.allclose(Pp, P - K @ S @ K.T, atol=1e-9)


def test_update_returns_writable_copies():
    xp, Pp = correlated_update_H(np.array([0.0]), np.array([[1.0]]), np.array([1.0]),
                                 np.array([[1.0]]), np.array([[0.0]]), np.array([[1.0]]))
    xp[0] = 5.0; Pp[0, 0] = 9.0                                      # must not raise (read-only would)


# ----------------------------------------------------------------------------- blocker 2

def test_budget_untruncated_spends_exactly_n():
    # Floor binds (n_ov=30 > 0.3n=24 at n=80); realized spend must equal n exactly (integer k,n,n_ov).
    for n, n_ov in [(80, 30), (160, 48), (320, 96)]:
        for k in (2, 4, 8):
            n_bulk, used, trunc, spend, feasible = budget_plan(n, k, n_ov, avail=100000)
            assert feasible and not trunc
            assert n_bulk == k * n - n_ov * (k + 1)
            assert used == n_bulk
            assert abs(spend - n) < 1e-9, (n, k, spend)


def test_budget_floor_uses_n_ov_not_0_3n():
    # The old bug sized the bulk with 0.3n=24 (giving n_bulk=int(0.7*8*80-24)=424); the fix uses n_ov=30.
    n_bulk, *_ = budget_plan(80, 8, n_ov=30, avail=100000)
    assert n_bulk == 8 * 80 - 30 * 9 == 370
    assert n_bulk != int(0.7 * 8 * 80 - 0.3 * 80)                    # != the old (overspending) 424


def test_budget_truncation_underspends():
    # Pool too small: cell is flagged truncated and realized spend drops below n.
    n_bulk, used, trunc, spend, feasible = budget_plan(320, 8, n_ov=96, avail=615)
    assert feasible and trunc
    assert used == 615 < n_bulk
    assert spend < 320


def test_budget_infeasible_overlap():
    # At n=40 the 30-row dual overlap alone costs 30*1.5=45 > 40 => infeasible, no bulk.
    n_bulk, used, trunc, spend, feasible = budget_plan(40, 2, n_ov=30, avail=100000)
    assert not feasible and used == 0 and not trunc


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
