#!/usr/bin/env python3
"""Gated dual-objective mixture: w(context) instead of constant w — the MoE gate, on observable context only.

The constant-w mixture over the mu-space and logit-space hop-conditioned Kalman experts won on both corpora
(REPORT_product_kalman_logit.md), with a 100%-boundary complementarity signal measured on LABEL position. A gate
cannot see the label at inference — this run tests whether OBSERVABLE context (the PRIOR's boundary proximity,
hop) predicts the regime well enough to beat constant w:

  w_i = sigmoid(a + b·prox_i + c·hop_i/5),  prox_i = max_c |2·prior_ic − 1|   (1 = prior at a boundary)

fit on calibration rows by mixture NLL (3 params, Nelder-Mead; component NLLs precomputed, so the gate fit is
just a reweighting). Also adds the missing calibration check for mixtures: per-channel PIT (probability integral
transform — CDF of the predictive marginal at the observed label; TRANSFORM-INVARIANT, so the logit expert's PIT
needs no Jacobian) with a KS-vs-uniform statistic.

Ladder: F mu/hop and E logit/hop (components, reference) | G mix const-w | H mix gated-w.

  python3 run_product_kalman_gated.py
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


def posterior_score(x, P, y_obs, R, C, t):
    """Correlated update → per-row NLL, and per-channel marginal (mean, sd) for PIT."""
    xp, Pp = correlated_update(x, P, y_obs, R, C)
    r = t - xp
    Vi = np.linalg.inv(Pp)
    nll = 0.5 * float(r @ Vi @ r) + 0.5 * np.log(np.linalg.det(Pp)) + LOG2PI
    sd = np.sqrt(np.diag(Pp))
    return nll, xp, sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_dataset(name, seeds, held_frac=0.30, min_train=30, min_held=12):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    prior = data.X[:, :2]; target = np.column_stack([data.D, data.S]); d = data.X[:, 2]
    y_deq = dequant(target); yl_d = logit(y_deq)
    xl = logit(np.clip(prior, NAIVE_EPS, 1 - NAIVE_EPS))
    prox = np.max(np.abs(2 * prior - 1), axis=1)             # observable: prior boundary proximity in [0,1]
    print(f"\n=== {name}: n={len(data.pairs)} pairs ===")

    rungs = ["F mu/hop", "E logit/hop", "G mix const-w", "H1 gated-NLL", "H2 gated-error", "H3 computed-BG"]
    acc = {r: [] for r in rungs}
    pits = {r: {"D": [], "S": []} for r in rungs}
    gate_params, w_bgs = [], []
    comp_prox = {"prior-near-boundary": [0, 0], "prior-interior": [0, 0]}
    comp_err = {"label-boundary": [0, 0], "label-interior": [0, 0]}   # complementarity in projected-ERROR terms
    # user prediction: the LOGIT expert degrades MORE when the graph misclassifies (unbounded log-odds influence
    # vs mu-space's bounded innovations). Buckets by |m − D_label|; accumulate per-expert squared projected error.
    comp_mis = {"graph-consistent": [0.0, 0.0, 0], "graph-misclassified": [0.0, 0.0, 0]}
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
        pj_l = fit_joint_sigma_of_hop(joint_errors(xl, yl_d, ml)[tr], data.hop[tr])

        def components(i):
            """Per-row (nll, PIT, direct-space error) for each expert. The logit expert's mean projects to
            direct space EXACTLY via sigmoid (median of the pushforward) — user's error-projection move."""
            h = float(data.hop[i]); lj = log_jac(y_deq[i])
            Lm = chol_of_hop(pj_mu, h); Cm = Lm @ Lm.T
            nF, xpF, sdF = posterior_score(prior[i], Cm[:2, :2], np.array([m[i]]), Cm[2:, 2:], Cm[:2, 2:], y_deq[i])
            pF = stats.norm.cdf((y_deq[i] - xpF) / sdF)
            eF = y_deq[i] - xpF                              # direct-space error, mu expert
            Ll = chol_of_hop(pj_l, h); Cl = Ll @ Ll.T
            nE, xpE, sdE = posterior_score(xl[i], Cl[:2, :2], np.array([ml[i]]), Cl[2:, 2:], Cl[:2, 2:], yl_d[i])
            pE = stats.norm.cdf((yl_d[i] - xpE) / sdE)       # PIT is transform-invariant: CDF at logit(y)
            eE = y_deq[i] - sigmoid(xpE)                     # logit expert's error PROJECTED to direct space
            return (nF, pF, eF), (nE - lj, pE, eE)

        cal = [components(i) for i in tr]
        nf = np.array([c[0][0] for c in cal]); ne = np.array([c[1][0] for c in cal])
        eF_c = np.array([c[0][2] for c in cal]); eE_c = np.array([c[1][2] for c in cal])
        px = prox[tr]; hn = data.hop[tr] / 5.0

        def mix_nll_vec(w, a=nf, b=ne):
            return -np.log(np.clip(w * np.exp(-a) + (1 - w) * np.exp(-b), 1e-300, None))
        ws = np.linspace(0, 1, 101)
        w_const = float(ws[int(np.argmin([mix_nll_vec(w).mean() for w in ws]))])

        # H1: NLL-trained logistic gate, now RIDGE-regularized (the unregularized run saturated to a hard step)
        def gate_obj(p):
            return mix_nll_vec(sigmoid(p[0] + p[1] * px + p[2] * hn)).mean() + 1e-2 * (p[1] ** 2 + p[2] ** 2)
        p0 = [np.log(w_const / max(1 - w_const, 1e-6)), 0.0, 0.0]
        gp = minimize(gate_obj, p0, method="Nelder-Mead", options={"maxiter": 4000}).x
        gate_params.append(gp)

        # H2 (user): responsibilities from PROJECTED direct-space errors → regularized logistic gate
        from sklearn.linear_model import LogisticRegression
        z = ((eF_c ** 2).sum(axis=1) < (eE_c ** 2).sum(axis=1)).astype(int)   # 1 = mu expert closer
        Xg = np.column_stack([px, hn])
        if 0 < z.sum() < len(z):
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xg, z)
            h2_w = lambda pxi, hni: float(lr.predict_proba([[pxi, hni]])[0, 1])
        else:
            zc = float(z.mean()); h2_w = lambda pxi, hni: zc

        # H3 (user): total error = weighted sum of projected errors → Bates–Granger/Kalman w, COMPUTED not fitted
        a = eF_c.ravel(); b = eE_c.ravel()                   # stack channels: common direct space makes this legal
        vF, vE, cFE = a.var(), b.var(), np.cov(a, b)[0, 1]
        w_bg = float(np.clip((vE - cFE) / max(vF + vE - 2 * cFE, 1e-12), 0.0, 1.0))

        w_bgs.append(w_bg)
        for i in he:
            (nF, pF, eF), (nE, pE, eE) = components(i)
            acc["F mu/hop"].append(nF); acc["E logit/hop"].append(nE)
            for ch, k in [(0, "D"), (1, "S")]:
                pits["F mu/hop"][k].append(pF[ch]); pits["E logit/hop"][k].append(pE[ch])
            hni = data.hop[i] / 5.0
            for rung, w in [("G mix const-w", w_const),
                            ("H1 gated-NLL", float(sigmoid(gp[0] + gp[1] * prox[i] + gp[2] * hni))),
                            ("H2 gated-error", h2_w(prox[i], hni)),
                            ("H3 computed-BG", w_bg)]:
                acc[rung].append(-np.log(max(w * np.exp(-nF) + (1 - w) * np.exp(-nE), 1e-300)))
                for ch, k in [(0, "D"), (1, "S")]:
                    pits[rung][k].append(w * pF[ch] + (1 - w) * pE[ch])
            kind = "prior-near-boundary" if prox[i] > 0.9 else "prior-interior"
            comp_prox[kind][0] += int(nE < nF); comp_prox[kind][1] += 1
            kind = "label-boundary" if (target[i].min() <= Q_HALF or target[i].max() >= 1 - Q_HALF) else "label-interior"
            comp_err[kind][0] += int((eE ** 2).sum() < (eF ** 2).sum()); comp_err[kind][1] += 1
            kind = "graph-misclassified" if abs(m[i] - target[i, 0]) > 0.25 else "graph-consistent"
            comp_mis[kind][0] += float((eF ** 2).sum()); comp_mis[kind][1] += float((eE ** 2).sum())
            comp_mis[kind][2] += 1

    print(f"splits used: {used}/{seeds} (NLL in mu-density terms, comparable; PIT-KS vs uniform, lower = better")
    print(f"calibrated; per-channel):")
    print(f"    {'rung':16s} {'NLL':>8s} {'KS_D':>6s} {'KS_S':>6s}")
    out = {}
    for r in rungs:
        nll = np.array(acc[r]); out[r] = nll
        ksd = stats.kstest(np.array(pits[r]["D"]), "uniform").statistic
        kss = stats.kstest(np.array(pits[r]["S"]), "uniform").statistic
        print(f"    {r:16s} {nll.mean():+8.4f} {ksd:6.3f} {kss:6.3f}")
    for cand in ["H1 gated-NLL", "H2 gated-error", "H3 computed-BG"]:
        g = out["G mix const-w"] - out[cand]
        print(f"    gain const-w→{cand}: {g.mean():+.4f} (row-SE {g.std()/np.sqrt(len(g)):.4f} — stability only)")
    gp = np.array(gate_params)
    print(f"    H1 gate coefficients (ridge; mean over splits): bias {gp[:,0].mean():+.2f}, "
          f"prox {gp[:,1].mean():+.2f}, hop {gp[:,2].mean():+.2f}   (w = P(mu expert))")
    print(f"    H3 Bates–Granger w (computed from projected error covariances): {np.mean(w_bgs):.2f} ± {np.std(w_bgs):.2f}")
    for kind, (nw, nt) in comp_prox.items():
        if nt:
            print(f"    complementarity (DENSITY terms) by prior position — logit wins on {kind}: {nw}/{nt} ({nw/nt:.0%})")
    for kind, (nw, nt) in comp_err.items():
        if nt:
            print(f"    complementarity (projected-ERROR terms) by label position — logit closer on {kind}: "
                  f"{nw}/{nt} ({nw/nt:.0%})")
    for kind, (sF, sE, nt) in comp_mis.items():
        if nt:
            print(f"    graph-misclassification (|m−D|>0.25) — mean sq projected error on {kind} (n={nt}): "
                  f"mu {sF/nt:.4f}  logit {sE/nt:.4f}  ratio logit/mu {sE/max(sF,1e-12):.2f}")
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
