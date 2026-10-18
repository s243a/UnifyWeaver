#!/usr/bin/env python3
"""Rung (b): logit-space fusion with boundary de-quantization + Jacobian-weighted per-row R (user design).

The interior-Gaussianity diagnostic showed the residuals' natural Gaussian home is LOGIT space — once the
boundary atoms (quantized LLM labels at/near exact 0/1) are handled. The user's treatment:
  - DE-QUANTIZE: a reported 0.0 is "≤ half the quantization step", not certainty → clip labels to
    [q/2, 1−q/2] (q=0.05 → [0.025, 0.975]) before the logit transform.
  - JACOBIAN WEIGHTING: the measurement's noise is defined in mu-space (sigma_mu from calibration residuals)
    and propagated per-row through the link: R_i = (sigma_mu / (m_i(1−m_i)))² — near-boundary measurements
    self-downweight (Fisher information mu(1−mu) does the censoring inside the plain Kalman update).

FAIR SCORING across spaces: every rung is scored as a density over mu at the SAME de-quantized label values —
logit-space rungs via the change of variables NLL_mu = NLL_logit − log|J(y)|, J = 1/(y_D(1−y_D)) · 1/(y_S(1−y_S)).
Mahal/dim is computed in each model's own fusion space (a space-internal calibration check).

Ladder (per descendant-disjoint split; all correlated updates):
  A mu/const        — yesterday's winner, rescored at de-quantized labels (reference)
  B logit/naive     — logit with a hard 1e-3 clip, no de-quantization (the atom-poisoned strawman)
  C logit/dequant   — logit with de-quantized labels
  D logit/dequant+w — C + per-row Jacobian-weighted measurement R
  E logit/hop+w     — D + hop-conditioned blocks (full stack)
  F mu/hop          — rung-(a) result at de-quantized labels (the mu-space full stack, for the head-to-head)

  python3 run_product_kalman_logit.py --dataset exploratory
  python3 run_product_kalman_logit.py --dataset fresh
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_product_kalman_sigma_hop import (
    H,
    LOG2PI,
    biv_nll_mahal,
    chol_of_hop,
    correlated_update,
    fit_joint_sigma_of_hop,
)
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
Q_HALF = 0.025                                              # LLM label quantization half-step (q=0.05)
NAIVE_EPS = 1e-3


def logit(m):
    return np.log(m / (1.0 - m))


def dequant(y):
    """A reported boundary label means 'beyond the half-step', not certainty."""
    return np.clip(y, Q_HALF, 1.0 - Q_HALF)


def log_jac(y2):
    """log |d logit / d mu| for a 2-dim label row (change-of-variables term for fair cross-space NLL)."""
    return float(-np.sum(np.log(y2 * (1.0 - y2))))


def joint_errors(prior_l, target_l, meas_l):
    eP = target_l - prior_l
    eM = meas_l - target_l[:, 0]
    return np.column_stack([eP[:, 0], eP[:, 1], eM])


def run_dataset(name, seeds, held_frac=0.30, min_train=30, min_held=12):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    prior = data.X[:, :2]; target = np.column_stack([data.D, data.S]); d = data.X[:, 2]
    y_deq = dequant(target)                                  # scoring point for EVERY rung
    print(f"\n=== {name}: n={len(data.pairs)} pairs (labels de-quantized to [{Q_HALF}, {1-Q_HALF}]) ===")

    rungs = ["A mu/const", "B logit/naive", "C logit/dequant", "D logit/dequant+w", "E logit/hop", "F mu/hop",
             "G mix(F,E)"]
    acc = {r: {"nll": [], "m2": []} for r in rungs}
    comp = {"boundary": [0, 0], "interior": [0, 0]}          # complementarity: rows where logit component wins
    w_fits = []
    used = 0
    for seed in range(seeds):
        tr, he = descendant_disjoint_split(data.pairs, seed, held_frac=held_frac)
        if len(tr) < min_train or len(he) < min_held:
            continue
        used += 1
        m = affine_calibrate(d[tr], data.D[tr], d)

        # --- mu-space statistics (rungs A, F) at de-quantized labels ---
        eMu = np.column_stack([(y_deq - prior)[:, 0], (y_deq - prior)[:, 1], m - y_deq[:, 0]])[tr]
        Cmu = np.cov(eMu.T) + 1e-9 * np.eye(3)
        pj_mu = fit_joint_sigma_of_hop(eMu, data.hop[tr])
        sig_m_mu = np.sqrt(Cmu[2, 2])                        # measurement noise scale, mu-space

        # --- logit-space statistics ---
        def to_logit(clip_lo):
            yl = logit(np.clip(target, clip_lo, 1 - clip_lo))
            xl = logit(np.clip(prior, NAIVE_EPS, 1 - NAIVE_EPS))
            ml = logit(np.clip(m, NAIVE_EPS, 1 - NAIVE_EPS))
            return yl, xl, ml
        yl_n, xl, ml = to_logit(NAIVE_EPS)                   # naive: hard clip only
        yl_d = logit(y_deq)                                  # de-quantized labels
        E_n = joint_errors(xl, yl_n, ml)[tr]; C_n = np.cov(E_n.T) + 1e-9 * np.eye(3)
        E_d = joint_errors(xl, yl_d, ml)[tr]; C_d = np.cov(E_d.T) + 1e-9 * np.eye(3)
        pj_l = fit_joint_sigma_of_hop(E_d, data.hop[tr])

        def score_row(x, P, y_obs, R, C, t, extra_nll=0.0):
            xp, Pp = correlated_update(x, P, y_obs, R, C)
            nll, m2 = biv_nll_mahal(t - xp, Pp)
            return nll + extra_nll, m2

        def fe_row(i):
            """Rungs F (mu/hop) and E (logit/hop) per-row mu-space NLLs — the mixture components."""
            h = float(data.hop[i]); lj = log_jac(y_deq[i])
            Lh = chol_of_hop(pj_mu, h); Ch = Lh @ Lh.T
            f = score_row(prior[i], Ch[:2, :2], np.array([m[i]]), Ch[2:, 2:], Ch[:2, 2:], y_deq[i])
            Ll = chol_of_hop(pj_l, h); Cl = Ll @ Ll.T
            e = score_row(xl[i], Cl[:2, :2], np.array([ml[i]]), Cl[2:, 2:], Cl[:2, 2:], yl_d[i], extra_nll=-lj)
            return f, e

        # fit the DUAL-OBJECTIVE mixture weight on the CALIBRATION rows: p = w·p_F + (1−w)·p_E (both mu-densities)
        nf = np.array([fe_row(i)[0][0] for i in tr]); ne = np.array([fe_row(i)[1][0] for i in tr])
        ws = np.linspace(0.0, 1.0, 101)
        mix_nll = [-np.mean(np.log(np.clip(w * np.exp(-nf) + (1 - w) * np.exp(-ne), 1e-300, None))) for w in ws]
        w_star = float(ws[int(np.argmin(mix_nll))]); w_fits.append(w_star)

        for i in he:
            h = float(data.hop[i]); yq = y_deq[i]; lj = log_jac(yq)
            mi = float(np.clip(m[i], NAIVE_EPS, 1 - NAIVE_EPS))
            jac_R = (sig_m_mu / (mi * (1.0 - mi))) ** 2      # user's per-row Jacobian-weighted measurement noise

            def score(rung, *args, **kw):
                nll, m2 = score_row(*args, **kw)
                acc[rung]["nll"].append(nll); acc[rung]["m2"].append(m2)

            # A: mu/const
            score("A mu/const", prior[i], Cmu[:2, :2], np.array([m[i]]), Cmu[2:, 2:], Cmu[:2, 2:], yq)
            # B: logit/naive (atom-poisoned covariance; scored at de-quantized labels for comparability)
            score("B logit/naive", xl[i], C_n[:2, :2], np.array([ml[i]]), C_n[2:, 2:], C_n[:2, 2:],
                  yl_d[i], extra_nll=-lj)
            # C: logit/dequant
            score("C logit/dequant", xl[i], C_d[:2, :2], np.array([ml[i]]), C_d[2:, 2:], C_d[:2, 2:],
                  yl_d[i], extra_nll=-lj)
            # D: C + Jacobian-weighted per-row R (replaces the fitted measurement variance)
            score("D logit/dequant+w", xl[i], C_d[:2, :2], np.array([ml[i]]),
                  np.array([[jac_R]]), C_d[:2, 2:], yl_d[i], extra_nll=-lj)
            # E, F: the two mixture components; G: the dual-objective mixture (user)
            (nllF, m2F), (nllE, m2E) = fe_row(i)
            acc["F mu/hop"]["nll"].append(nllF); acc["F mu/hop"]["m2"].append(m2F)
            acc["E logit/hop"]["nll"].append(nllE); acc["E logit/hop"]["m2"].append(m2E)
            nllG = -np.log(max(w_star * np.exp(-nllF) + (1 - w_star) * np.exp(-nllE), 1e-300))
            acc["G mix(F,E)"]["nll"].append(nllG); acc["G mix(F,E)"]["m2"].append(np.nan)
            # complementarity: which component wins, split by label position
            kind = "boundary" if (target[i].min() <= Q_HALF or target[i].max() >= 1 - Q_HALF) else "interior"
            comp[kind][0] += int(nllE < nllF); comp[kind][1] += 1

    print(f"splits used: {used}/{seeds}; per-row pooled (NLL in MU-space density via change-of-variables —")
    print(f"comparable ACROSS rungs; Mahal/dim in each rung's own space, ≈1 calibrated; q95 ref 5.99):")
    print(f"    {'rung':18s} {'NLL_mu':>8s} {'Mahal/dim':>10s} {'q95':>7s}")
    out = {}
    for r in rungs:
        nll = np.array(acc[r]["nll"]); m2 = np.array(acc[r]["m2"])
        out[r] = nll
        print(f"    {r:18s} {nll.mean():+8.4f} {np.nanmean(m2)/2:10.2f} {np.nanquantile(m2, 0.95):7.2f}")
    print(f"    mixture weight w (mu component): mean {np.mean(w_fits):.2f} ± {np.std(w_fits):.2f} over splits")
    for kind, (nw, nt) in comp.items():
        if nt:
            print(f"    complementarity — logit component wins on {kind} rows: {nw}/{nt} ({nw/nt:.0%})")
    for base, cand in [("B logit/naive", "C logit/dequant"), ("A mu/const", "D logit/dequant+w"),
                       ("F mu/hop", "E logit/hop"), ("F mu/hop", "G mix(F,E)")]:
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
