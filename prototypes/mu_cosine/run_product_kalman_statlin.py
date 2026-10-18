#!/usr/bin/env python3
"""Statistically-linearized transport for the logit expert (user design, REPORT_product_kalman_gated future work).

A boundary label is a NARROW DISTRIBUTION, not a point: quantized LLM reports carry a half-step width. The three
boundary treatments used so far (de-quantization = move the point; Jacobian weighting = zero the weight; point
change-of-variables) are partial renderings of that one fact. This build replaces them with STATISTICAL
LINEARIZATION: each label y is a uniform distribution on [y−q/2, y+q/2] ∩ (0,1), transported to logit space by
3-point Gauss–Legendre quadrature, giving per row and channel:

  ell_mean = E[logit(u)]                      (transported label)
  ell_var  = Var[logit(u)]                    (transport-induced noise, KNOWN per row)
  L*       = Cov(u, logit(u)) / Var(u)        (distribution-averaged Jacobian — finite at boundaries,
                                               so endpoints are no longer zero-weight)

The structural logit covariance Sigma(h) is fit by heteroscedastic MLE treating ell_var as known per-row noise
(no double-count); scoring adds ell_var back per row; the mu-space density uses the statistically-linearized
change-of-variables (log L* instead of the point Jacobian).

Ladder: F mu/hop | E_pt logit/hop point-transport (merged reference) | E_sl logit/hop stat-lin |
        G_pt mix(F,E_pt) (merged champion) | G_sl mix(F,E_sl).
Target: E_sl/G_sl improve PIT SHAPE (KS ↓ toward the ~0.025 critical value) without losing NLL.

  python3 run_product_kalman_statlin.py
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_product_kalman_logit import NAIVE_EPS, Q_HALF, dequant, joint_errors, log_jac, logit
from run_product_kalman_sigma_hop import H, LOG2PI, chol_of_hop, correlated_update, fit_joint_sigma_of_hop
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-4
GL_T = np.array([0.1127016654, 0.5, 0.8873983346])           # 3-pt Gauss–Legendre nodes on [0,1]
GL_W = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])


def statlin_transport(y):
    """Per-value statistical linearization of logit over uniform[y−q/2, y+q/2] ∩ (EPS, 1−EPS).
    Returns (ell_mean, ell_var, L_star)."""
    a = max(EPS, y - Q_HALF); b = min(1.0 - EPS, y + Q_HALF)
    u = a + (b - a) * GL_T
    lu = logit(u)
    l_mean = float(GL_W @ lu)
    l_var = float(GL_W @ (lu - l_mean) ** 2)
    u_mean = float(GL_W @ u)
    u_var = float(GL_W @ (u - u_mean) ** 2)
    L = float(GL_W @ ((u - u_mean) * (lu - l_mean)) / max(u_var, 1e-12))
    return l_mean, l_var, L


def fit_sigma_hop_hetero(E, hop, tv, p0_fit):
    """Heteroscedastic MLE of the smooth trivariate structural Sigma(h): residual_i ~ N(0, Sigma(h_i) + D_i),
    D_i = diag(tv_i_D, tv_i_S, 0) the KNOWN per-row transport noise (so it is not double-counted)."""
    hs = sorted(set(hop.tolist()))

    def nll(p):
        tot = 0.0
        for h in hs:
            m = hop == h
            L = chol_of_hop(p, h); S = L @ L.T
            for e, d in zip(E[m], tv[m]):
                V = S + np.diag([d[0], d[1], 0.0])
                sign, logdet = np.linalg.slogdet(V)
                if sign <= 0:
                    return 1e12
                tot += 0.5 * float(e @ np.linalg.solve(V, e)) + 0.5 * logdet
        return tot / len(E)

    r = minimize(nll, p0_fit, method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-4, "fatol": 1e-6})
    return r.x


def run_dataset(name, seeds, held_frac=0.30, min_train=30, min_held=12):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    prior = data.X[:, :2]; target = np.column_stack([data.D, data.S]); d = data.X[:, 2]
    y_deq = dequant(target); yl_pt = logit(y_deq)
    xl = logit(np.clip(prior, NAIVE_EPS, 1 - NAIVE_EPS))
    sl = np.array([[statlin_transport(target[i, c]) for c in (0, 1)] for i in range(len(target))])
    yl_sl = sl[:, :, 0]; tv = sl[:, :, 1]; Lstar = sl[:, :, 2]  # (n,2) each
    print(f"\n=== {name}: n={len(data.pairs)} pairs; mean transport sd (logit) "
          f"D {np.sqrt(tv[:,0]).mean():.2f} S {np.sqrt(tv[:,1]).mean():.2f} ===")

    rungs = ["F mu/hop", "E_pt logit point", "E_sl logit statlin", "G_pt mix(F,E_pt)", "G_sl mix(F,E_sl)"]
    acc = {r: [] for r in rungs}
    pits = {r: {"D": [], "S": []} for r in rungs}
    used = 0
    for seed in range(seeds):
        tr, he = descendant_disjoint_split(data.pairs, seed, held_frac=held_frac)
        if len(tr) < min_train or len(he) < min_held:
            continue
        used += 1
        m = affine_calibrate(d[tr], data.D[tr], d)
        ml = logit(np.clip(m, NAIVE_EPS, 1 - NAIVE_EPS))
        eMu = np.column_stack([(y_deq - prior)[:, 0], (y_deq - prior)[:, 1], m - y_deq[:, 0]])[tr]
        pj_mu = fit_joint_sigma_of_hop(eMu, data.hop[tr])
        E_pt = joint_errors(xl, yl_pt, ml)[tr]
        pj_pt = fit_joint_sigma_of_hop(E_pt, data.hop[tr])
        E_sl = joint_errors(xl, yl_sl, ml)[tr]
        pj_sl = fit_sigma_hop_hetero(E_sl, data.hop[tr], tv[tr], pj_pt)  # warm-start from the point fit

        def row(i):
            h = float(data.hop[i])
            # F: mu expert
            Lm = chol_of_hop(pj_mu, h); Cm = Lm @ Lm.T
            xpF, PpF = correlated_update(prior[i], Cm[:2, :2], np.array([m[i]]), Cm[2:, 2:], Cm[:2, 2:])
            rF = y_deq[i] - xpF; ViF = np.linalg.inv(PpF)
            nF = 0.5 * float(rF @ ViF @ rF) + 0.5 * np.log(np.linalg.det(PpF)) + LOG2PI
            pF = stats.norm.cdf(rF / np.sqrt(np.diag(PpF)))
            # E_pt: point transport, point change-of-variables
            Lp = chol_of_hop(pj_pt, h); Cp = Lp @ Lp.T
            xpP, PpP = correlated_update(xl[i], Cp[:2, :2], np.array([ml[i]]), Cp[2:, 2:], Cp[:2, 2:])
            rP = yl_pt[i] - xpP; ViP = np.linalg.inv(PpP)
            nP = 0.5 * float(rP @ ViP @ rP) + 0.5 * np.log(np.linalg.det(PpP)) + LOG2PI - log_jac(y_deq[i])
            pP = stats.norm.cdf(rP / np.sqrt(np.diag(PpP)))
            # E_sl: statlin transport — predictive adds per-row transport noise; change-of-vars uses L*
            Ls = chol_of_hop(pj_sl, h); Cs = Ls @ Ls.T
            xpS, PpS = correlated_update(xl[i], Cs[:2, :2], np.array([ml[i]]), Cs[2:, 2:], Cs[:2, 2:])
            Vs = PpS + np.diag(tv[i])
            rS = yl_sl[i] - xpS; ViS = np.linalg.inv(Vs)
            nS = (0.5 * float(rS @ ViS @ rS) + 0.5 * np.log(np.linalg.det(Vs)) + LOG2PI
                  - float(np.sum(np.log(np.maximum(Lstar[i], 1e-12)))))
            pS = stats.norm.cdf(rS / np.sqrt(np.diag(Vs)))
            return (nF, pF), (nP, pP), (nS, pS)

        cal = [row(i) for i in tr]
        def fit_w(idx_a, idx_b):
            na = np.array([c[idx_a][0] for c in cal]); nb = np.array([c[idx_b][0] for c in cal])
            ws = np.linspace(0, 1, 101)
            return float(ws[int(np.argmin([-np.mean(np.log(np.clip(
                w * np.exp(-na) + (1 - w) * np.exp(-nb), 1e-300, None))) for w in ws]))])
        w_pt = fit_w(0, 1); w_sl = fit_w(0, 2)

        for i in he:
            (nF, pF), (nP, pP), (nS, pS) = row(i)
            for rung, nll, p in [("F mu/hop", nF, pF), ("E_pt logit point", nP, pP), ("E_sl logit statlin", nS, pS)]:
                acc[rung].append(nll)
                pits[rung]["D"].append(p[0]); pits[rung]["S"].append(p[1])
            for rung, w, (na, pa), (nb, pb) in [("G_pt mix(F,E_pt)", w_pt, (nF, pF), (nP, pP)),
                                                ("G_sl mix(F,E_sl)", w_sl, (nF, pF), (nS, pS))]:
                acc[rung].append(-np.log(max(w * np.exp(-na) + (1 - w) * np.exp(-nb), 1e-300)))
                pits[rung]["D"].append(w * pa[0] + (1 - w) * pb[0])
                pits[rung]["S"].append(w * pa[1] + (1 - w) * pb[1])

    print(f"splits used: {used}/{seeds} (NLL mu-density comparable; PIT-KS vs uniform, ~0.025 = calibrated shape):")
    print(f"    {'rung':20s} {'NLL':>8s} {'KS_D':>6s} {'KS_S':>6s}")
    out = {}
    for r in rungs:
        nll = np.array(acc[r]); out[r] = nll
        ksd = stats.kstest(np.array(pits[r]["D"]), "uniform").statistic
        kss = stats.kstest(np.array(pits[r]["S"]), "uniform").statistic
        print(f"    {r:20s} {nll.mean():+8.4f} {ksd:6.3f} {kss:6.3f}")
    for base, cand in [("E_pt logit point", "E_sl logit statlin"), ("G_pt mix(F,E_pt)", "G_sl mix(F,E_sl)")]:
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
