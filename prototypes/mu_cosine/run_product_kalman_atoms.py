#!/usr/bin/env python3
"""D-channel atoms: the discrete/continuous (atom-inflated) likelihood — completeness check on the last defect.

The champion (G_sl) is a continuous density, but ~7% of D labels are ATOMS (the judge reports exactly 0.0 / 1.0).
A continuous model implicitly prices the atom event at whatever mass its tail assigns. This run puts everything
on the correct MIXED footing — atoms scored as probability MASS, interiors as densities — and asks two questions:

  1. Is the continuous champion's IMPLIED atom mass badly wrong, and does a LEARNED atom head fix it?
     (3-class multinomial: lo-atom / interior / hi-atom on observable features [fused D-mean, hop].)
  2. Does the inflated model finally pass D-channel shape? (RANDOMIZED PIT — the standard uniformity
     construction for mixed discrete/continuous distributions.)

Models on the mixed footing (identical continuous core = the merged G_sl mixture):
  IMPLIED — atom masses = the mixture's own D-marginal CDF mass on the atom regions (D < 0.0375 / > 0.9625);
  LEARNED — atom masses = the multinomial head; continuous part unchanged.
(Both leave the interior density unrenormalized over the interior region — same small approximation both sides.)

  python3 run_product_kalman_atoms.py
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_product_kalman_logit import NAIVE_EPS, Q_HALF, dequant, joint_errors, logit
from run_product_kalman_sigma_hop import LOG2PI, chol_of_hop, correlated_update, fit_joint_sigma_of_hop
from run_product_kalman_statlin import fit_sigma_hop_hetero, statlin_transport
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CUT_LO, CUT_HI = 0.0375, 0.9625                              # atom regions: below/above the first quantized interior value


def run_dataset(name, seeds, held_frac=0.30, min_train=30, min_held=12, rng_pit=None):
    cfg = DATASETS[name]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    prior = data.X[:, :2]; target = np.column_stack([data.D, data.S]); d = data.X[:, 2]
    y_deq = dequant(target)
    xl = logit(np.clip(prior, NAIVE_EPS, 1 - NAIVE_EPS))
    sl = np.array([[statlin_transport(target[i, c]) for c in (0, 1)] for i in range(len(target))])
    yl_sl = sl[:, :, 0]; tv = sl[:, :, 1]; Lstar = sl[:, :, 2]
    klass = np.where(target[:, 0] < CUT_LO, 0, np.where(target[:, 0] > CUT_HI, 2, 1))   # lo/int/hi on ORIGINAL D
    print(f"\n=== {name}: n={len(data.pairs)}; D-atom rates lo {np.mean(klass==0):.1%} hi {np.mean(klass==2):.1%} ===")

    rng_pit = rng_pit or np.random.default_rng(0)
    acc = {"IMPLIED": [], "LEARNED": []}
    pit_d = {"IMPLIED": [], "LEARNED": []}
    pit_bin = {"D": [], "S": []}   # bin-mass randomized PIT over the FULL quantization lattice (champion model)
    atom_cal = []                                            # (mean predicted pi_lo, empirical lo rate) per split
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
        pj_pt = fit_joint_sigma_of_hop(joint_errors(xl, yl_sl, ml)[tr], data.hop[tr])
        pj_sl = fit_sigma_hop_hetero(joint_errors(xl, yl_sl, ml)[tr], data.hop[tr], tv[tr], pj_pt)

        def posts(i):
            """Both experts' posteriors + per-row mu-space NLL pieces (as in the statlin champion)."""
            h = float(data.hop[i])
            Lm = chol_of_hop(pj_mu, h); Cm = Lm @ Lm.T
            xpF, PpF = correlated_update(prior[i], Cm[:2, :2], np.array([m[i]]), Cm[2:, 2:], Cm[:2, 2:])
            rF = y_deq[i] - xpF
            nF = 0.5 * float(rF @ np.linalg.solve(PpF, rF)) + 0.5 * np.log(np.linalg.det(PpF)) + LOG2PI
            Ls = chol_of_hop(pj_sl, h); Cs = Ls @ Ls.T
            xpS, PpS = correlated_update(xl[i], Cs[:2, :2], np.array([ml[i]]), Cs[2:, 2:], Cs[:2, 2:])
            Vs = PpS + np.diag(tv[i])
            rS = yl_sl[i] - xpS
            nS = (0.5 * float(rS @ np.linalg.solve(Vs, rS)) + 0.5 * np.log(np.linalg.det(Vs)) + LOG2PI
                  - float(np.sum(np.log(np.maximum(Lstar[i], 1e-12)))))
            return (nF, xpF, np.sqrt(np.diag(PpF))), (nS, xpS, np.sqrt(np.diag(Vs)))

        cal = [posts(i) for i in tr]
        nf = np.array([c[0][0] for c in cal]); ns = np.array([c[1][0] for c in cal])
        ws = np.linspace(0, 1, 101)
        w = float(ws[int(np.argmin([-np.mean(np.log(np.clip(u * np.exp(-nf) + (1 - u) * np.exp(-ns),
                                                            1e-300, None))) for u in ws]))])

        def d_cdf(i, c, F, E):
            """Mixture D-marginal CDF at cut c (CDF is transform-invariant for the logit expert)."""
            return (w * stats.norm.cdf((c - F[1][0]) / F[2][0])
                    + (1 - w) * stats.norm.cdf((logit(c) - E[1][0]) / E[2][0]))

        # LEARNED atom head: multinomial on observable [fused D-mean, hop/5], calibration rows only
        featc = np.array([[w * c[0][1][0] + (1 - w) * float(1 / (1 + np.exp(-c[1][1][0]))), hh / 5.0]
                          for c, hh in zip(cal, data.hop[tr])])
        kc = klass[tr]
        classes = sorted(set(kc.tolist()))
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(featc, kc) if len(classes) > 1 else None

        def pis(feat):
            if lr is None:
                p = np.zeros(3); p[classes[0]] = 1.0; return np.clip(p, 1e-4, None)
            pr = lr.predict_proba([feat])[0]
            p = np.zeros(3)
            for j, cl in enumerate(lr.classes_):
                p[cl] = pr[j]
            return np.clip(p, 1e-4, None)

        pred_lo, emp_lo = [], []
        for i in he:
            F, E = posts(i)
            nll_mix = -np.log(max(w * np.exp(-F[0]) + (1 - w) * np.exp(-E[0]), 1e-300))
            # S-marginal density of the mixture (needed on atom rows, where D is scored as mass)
            pS = (w * stats.norm.pdf(y_deq[i][1], F[1][1], F[2][1])
                  + (1 - w) * stats.norm.pdf(yl_sl[i][1], E[1][1], E[2][1]) * Lstar[i][1])
            nS_marg = -np.log(max(pS, 1e-300))
            F_lo = d_cdf(i, CUT_LO, F, E); F_hi = d_cdf(i, CUT_HI, F, E)
            pi_imp = np.clip([F_lo, F_hi - F_lo, 1 - F_hi], 1e-4, None)
            fdm = w * F[1][0] + (1 - w) * float(1 / (1 + np.exp(-E[1][0])))
            pi_lrn = pis([fdm, data.hop[i] / 5.0])
            k = klass[i]
            for tag, pi in [("IMPLIED", pi_imp), ("LEARNED", pi_lrn)]:
                if k == 1:
                    acc[tag].append(-np.log(pi[1]) + nll_mix)
                else:
                    acc[tag].append(-np.log(pi[0] if k == 0 else pi[2]) + nS_marg)
                # randomized PIT for the mixed D-marginal
                v = rng_pit.uniform()
                if k == 0:
                    u = pi[0] * v
                elif k == 2:
                    u = 1 - pi[2] + pi[2] * v
                else:
                    Fc = (d_cdf(i, float(y_deq[i][0]), F, E) - F_lo) / max(F_hi - F_lo, 1e-9)
                    u = pi[0] + pi[1] * np.clip(Fc, 0, 1)
                pit_d[tag].append(float(np.clip(u, 0, 1)))
            pred_lo.append(pi_lrn[0]); emp_lo.append(int(k == 0))
            # bin-mass randomized PIT over the FULL 0.05 lattice: every label is quantized, not just boundaries —
            # score its BIN's mass under the champion; if this passes uniformity, the model was shape-calibrated
            # all along and the earlier continuous-PIT failures were the lattice artifact.
            def marg_cdf(ch, c):
                cc = float(np.clip(c, 1e-6, 1 - 1e-6))
                return (w * stats.norm.cdf((cc - F[1][ch]) / F[2][ch])
                        + (1 - w) * stats.norm.cdf((logit(cc) - E[1][ch]) / E[2][ch]))
            for ch, kch in [(0, "D"), (1, "S")]:
                y0 = float(target[i][ch])
                flo = marg_cdf(ch, y0 - Q_HALF); fhi = marg_cdf(ch, y0 + Q_HALF)
                pit_bin[kch].append(float(np.clip(flo + rng_pit.uniform() * max(fhi - flo, 1e-12), 0, 1)))
        atom_cal.append((float(np.mean(pred_lo)), float(np.mean(emp_lo))))

    print(f"splits used: {used}/{seeds} (mixed-footing NLL: atoms as MASS + S-density, interior as joint density;")
    print(f"randomized-PIT KS for the mixed D-marginal, ~0.025 = calibrated shape):")
    print(f"    {'model':10s} {'NLL':>8s} {'KS_D(rand)':>10s}")
    out = {}
    for tag in ("IMPLIED", "LEARNED"):
        nll = np.array(acc[tag]); out[tag] = nll
        ks = stats.kstest(np.array(pit_d[tag]), "uniform").statistic
        print(f"    {tag:10s} {nll.mean():+8.4f} {ks:10.3f}")
    g = out["IMPLIED"] - out["LEARNED"]
    print(f"    gain IMPLIED→LEARNED: {g.mean():+.4f} (row-SE {g.std()/np.sqrt(len(g)):.4f} — stability only)")
    ac = np.array(atom_cal)
    print(f"    learned lo-atom head calibration: predicted {ac[:,0].mean():.3f} vs empirical {ac[:,1].mean():.3f}")
    ksd = stats.kstest(np.array(pit_bin["D"]), "uniform").statistic
    kss = stats.kstest(np.array(pit_bin["S"]), "uniform").statistic
    print(f"    BIN-MASS randomized PIT over the full 0.05 lattice (champion): KS_D {ksd:.3f}  KS_S {kss:.3f} "
          f"(~0.025 = calibrated)")
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
