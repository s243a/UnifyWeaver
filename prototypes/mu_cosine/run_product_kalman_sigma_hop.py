#!/usr/bin/env python3
"""Rung (a): hop-conditioned covariance blocks INSIDE the Kalman gain (the designed Sigma(hop) x fusion synthesis).

Yesterday's real-data run (`run_product_kalman_realdata.py`) fit CONSTANT covariance blocks and left a measured
residual overconfidence (Mahal/dim 1.39 on fresh). The confirmed Sigma(hop) result says the error covariance is
predictable from hop — so here the JOINT trivariate error covariance (prior-error D, prior-error S,
measurement-error) is fit as a SMOOTH function of hop and its blocks P(h), R(h), C(h) drive a per-row correlated
update. Parametrization: hop-dependent Cholesky factor L(h) (diag exp(a+b·h), off-diag c+d·h; 12 params) so
Sigma(h) = L(h)L(h)^T is SPD by construction; MLE via Nelder-Mead on the calibration split.

Ladder (per descendant-disjoint split):
  prior/const     — model readouts, constant P            (yesterday's baseline)
  prior/hop       — model readouts, P(h)                  (the confirmatory result restated in this harness)
  kalman/const    — correlated update, constant P,R,C     (yesterday's winner)
  kalman/hop      — correlated update, P(h),R(h),C(h)     (the new rung)

Target: kalman/hop drops Mahal/dim toward 1 (esp. fresh) and beats kalman/const on held-out NLL.
NLL includes the 2*pi constant — compare within this run, not against yesterday's table.

  python3 run_product_kalman_sigma_hop.py --dataset exploratory
  python3 run_product_kalman_sigma_hop.py --dataset fresh
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_product_kalman_realdata import DATASETS, affine_calibrate
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
H = np.array([[1.0, 0.0]])                                  # graph channel observes the D component
LOG2PI = np.log(2 * np.pi)


def chol_of_hop(params, h):
    """Hop-dependent lower-triangular factor: diag exp(a+b·h), off-diag c+d·h. Sigma(h)=L L^T is SPD always."""
    a = params[:3]; b = params[3:6]; c = params[6:9]; d = params[9:12]
    L = np.zeros((3, 3))
    L[0, 0], L[1, 1], L[2, 2] = np.exp(a + b * h)
    L[1, 0] = c[0] + d[0] * h; L[2, 0] = c[1] + d[1] * h; L[2, 1] = c[2] + d[2] * h
    return L


def fit_joint_sigma_of_hop(E, hop):
    """MLE of the smooth trivariate Sigma(h) on calibration joint errors E (n x 3)."""
    C0 = np.cov(E.T) + 1e-6 * np.eye(3)
    L0 = np.linalg.cholesky(C0)
    p0 = np.concatenate([np.log(np.diag(L0)), np.zeros(3), [L0[1, 0], L0[2, 0], L0[2, 1]], np.zeros(3)])
    hs = sorted(set(hop.tolist()))

    def nll(p):
        tot = 0.0
        for h in hs:
            m = hop == h
            L = chol_of_hop(p, h)
            z = np.linalg.solve(L, E[m].T)                   # whiten
            tot += 0.5 * np.sum(z * z) + m.sum() * (np.log(np.abs(np.diag(L))).sum() + 1.5 * LOG2PI)
        return tot / len(E)

    r = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-7})
    return r.x


def biv_nll_mahal(r, V):
    """Per-row bivariate Gaussian NLL and squared Mahalanobis."""
    Vi = np.linalg.inv(V)
    m2 = float(r @ Vi @ r)
    return 0.5 * m2 + 0.5 * np.log(np.linalg.det(V)) + LOG2PI, m2


def correlated_update(x, P, y, R, C):
    """Kalman update with correlated prior/measurement errors: y = H z + v, C = Cov(prior_err, v)."""
    S = H @ P @ H.T + R + H @ C + (H @ C).T
    K = (P @ H.T + C) @ np.linalg.inv(S)
    x_post = x + (K @ (y - H @ x)).ravel()
    P_post = P - K @ S @ K.T
    return x_post, P_post


def run_dataset(name, seeds, held_frac=0.30, min_train=30, min_held=12):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    prior = data.X[:, :2]; target = np.column_stack([data.D, data.S]); d = data.X[:, 2]
    print(f"\n=== {name}: n={len(data.pairs)} pairs ===")

    rungs = ["prior/const", "prior/hop", "kalman/const", "kalman/hop"]
    acc = {r: {"nll": [], "m2": []} for r in rungs}
    used = 0
    for seed in range(seeds):
        tr, he = descendant_disjoint_split(data.pairs, seed, held_frac=held_frac)
        if len(tr) < min_train or len(he) < min_held:
            continue
        used += 1
        m = affine_calibrate(d[tr], data.D[tr], d)
        eP = target - prior                                  # prior error (z - x_hat), rows
        eM = m - target[:, 0]                                # measurement error (v = y - H z)
        E_tr = np.column_stack([eP[:, 0], eP[:, 1], eM])[tr]
        Cc = np.cov(E_tr.T) + 1e-9 * np.eye(3)               # constant joint covariance
        pj = fit_joint_sigma_of_hop(E_tr, data.hop[tr])      # smooth Sigma(h)

        for i in he:
            h = float(data.hop[i]); x = prior[i]; y = np.array([m[i]]); t = target[i]
            Lh = chol_of_hop(pj, h); Ch = Lh @ Lh.T
            blocks = {"const": (Cc[:2, :2], Cc[2:, 2:], Cc[:2, 2:]),
                      "hop": (Ch[:2, :2], Ch[2:, 2:], Ch[:2, 2:])}
            for kind, (P, R, C) in blocks.items():
                nll, m2 = biv_nll_mahal(t - x, P)
                acc[f"prior/{kind}"]["nll"].append(nll); acc[f"prior/{kind}"]["m2"].append(m2)
                xp, Pp = correlated_update(x, P, y, R, C)
                nll, m2 = biv_nll_mahal(t - xp, Pp)
                acc[f"kalman/{kind}"]["nll"].append(nll); acc[f"kalman/{kind}"]["m2"].append(m2)

    print(f"splits used: {used}/{seeds}; per-row pooled over splits (NLL lower better; Mahal/dim ≈ 1 calibrated,")
    print(f">1 overconfident; q95 vs chi2_2 ref 5.99):")
    print(f"    {'rung':14s} {'NLL':>8s} {'Mahal/dim':>10s} {'q95':>7s}")
    out = {}
    for r in rungs:
        nll = np.array(acc[r]["nll"]); m2 = np.array(acc[r]["m2"])
        out[r] = nll
        print(f"    {r:14s} {nll.mean():+8.4f} {m2.mean()/2:10.2f} {np.quantile(m2, 0.95):7.2f}")
    for base, cand in [("prior/const", "prior/hop"), ("kalman/const", "kalman/hop"), ("prior/hop", "kalman/hop")]:
        g = out[base] - out[cand]
        print(f"    gain {base}→{cand}: {g.mean():+.4f} (row-SE {g.std()/np.sqrt(len(g)):.4f} — stability only)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args()
    for n in (list(DATASETS) if a.dataset == "all" else [a.dataset]):
        run_dataset(n, a.seeds)


if __name__ == "__main__":
    main()
