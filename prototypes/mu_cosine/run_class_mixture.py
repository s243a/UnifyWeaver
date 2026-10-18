#!/usr/bin/env python3
"""B2 step 4: class-mixture predictive for the D-channel bimodality defect, judged against the
THEORY_evidence_fusion §11.5 acceptance gate.

The atoms run diagnosed the residual predictive defect as D's bimodality (directional-or-not) vs an
effectively-unimodal Gaussian predictive — the JointPosterior result #1 resurfacing. The candidate fix:

    p(D | context) = Σ_c P(c | context) · N_c(D)        c ∈ {directional, lateral}

with the classes ANCHORED to observable relation classes (gate 3): c = which relation family the judge's
own labels put the pair in (max directional-μ vs max symmetric/none — i.e. is the pair a hierarchy pair at
all). P(c | context) is predicted from JUDGE-FREE observables only (prior readouts, graph walk, stratum
one-hots — logistic regression), so the mixture is deployable before any judge call.

Gates (§11.5): (1) held-out PROPER score must improve — bin-mass NLL (labels live on the 0.05 lattice, so
the score is log ∫_bin p, not a density); (2) the class posterior must CALIBRATE (reliability table / ECE);
(3) modes anchored (by construction, reported); (4) shape evidence (label histogram) as SUPPORT only.

  python3 run_class_mixture.py --ckpt model_channel_heads_namecond_r0.pt
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fine_tune_channel_heads import load_campaign_datasets, load_expanded
from fine_tune_fused_head import agnostic_readouts
from run_product_kalman_realdata import affine_calibrate

ROOT = os.path.dirname(os.path.abspath(__file__))
Q = 0.05                                                     # label lattice step


def bin_mass_nll(y, mu, sd, w=None):
    """Proper score for lattice labels: −log Σ_c w_c ∫_{y±Q/2} N_c. mu/sd/w: [N] or [N,K]."""
    mu, sd = np.atleast_2d(mu.T).T, np.atleast_2d(sd.T).T
    if mu.ndim == 1:
        mu, sd = mu[:, None], sd[:, None]
    w = np.ones_like(mu) if w is None else w
    lo = np.clip((y[:, None] - Q / 2 - mu) / sd, -30, 30)
    hi = np.clip((y[:, None] + Q / 2 - mu) / sd, -30, 30)
    mass = np.clip((norm.cdf(hi) - norm.cdf(lo)) * w, 1e-12, None).sum(1)
    return -np.log(mass)


def fit_sigmoid_platt(X, c, ridge=1.0, iters=200, lr=0.5):
    """Tiny logistic regression (no sklearn dependency): returns w, b."""
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    Z = (X - Xm) / Xs
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - c) / len(c) + ridge * w / len(c)
        gb = (p - c).mean()
        w -= lr * g; b -= lr * gb
    return (Xm, Xs, w, b)


def predict_sigmoid(model, X):
    Xm, Xs, w, b = model
    return 1 / (1 + np.exp(-(((X - Xm) / Xs) @ w + b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond_r0.pt"))
    a = ap.parse_args()
    model, _ = load_expanded(a.ckpt, dev="cpu")
    dss = load_campaign_datasets()

    for n, ds in dss.items():
        ro = agnostic_readouts(model, ds, "cpu")
        D = ds["D"]
        tr, he = ds["tr"], ds["he"]
        strata = sorted({t if not t.startswith("campaign_h") else "trans" for t in ds["tags"]})
        tagg = [t if not t.startswith("campaign_h") else "trans" for t in ds["tags"]]
        onehot = np.array([[1.0 if t == s else 0.0 for s in strata] for t in tagg])
        X = np.column_stack([ro["prior_D"], ro["prior_S"], ds["d"], onehot])   # judge-free context only

        # anchored classes (gate 3): is the pair a hierarchy pair at all, per the judge's relation family
        c = (D >= 0.5).astype(float)
        print(f"\n=== {n}: {len(D)} rows, P(directional) = {c.mean():.2f} ===")
        # gate 4 (support only): the label histogram's two masses
        hist, _ = np.histogram(D, bins=[0, 0.25, 0.75, 1.001])
        print(f"D-label mass [<0.25 / mid / >0.75]: {hist / len(D)} (bimodal shape — supporting evidence only)")

        # baseline: single Gaussian around the calibrated prior readout
        mu1 = affine_calibrate(ro["prior_D"][tr], D[tr], ro["prior_D"])
        sd1 = np.full(len(D), (D[tr] - mu1[tr]).std())
        nll1 = bin_mass_nll(D[he], mu1[he], sd1[he])

        # class mixture: logistic P(c|X) on train; per-class calibrated means + sds on train
        lr_model = fit_sigmoid_platt(X[tr], c[tr])
        pc = predict_sigmoid(lr_model, X)
        mus, sds = np.zeros((len(D), 2)), np.zeros((len(D), 2))
        for k, mask in [(0, c[tr] < 0.5), (1, c[tr] >= 0.5)]:
            sub = np.array(tr)[mask]
            mk = affine_calibrate(ro["prior_D"][sub], D[sub], ro["prior_D"])
            mus[:, k] = mk
            sds[:, k] = max((D[sub] - mk[sub]).std(), 0.02)
        w2 = np.column_stack([1 - pc, pc])
        nll2 = bin_mass_nll(D[he], mus[he], sds[he], w2[he])

        d = nll1 - nll2
        print(f"GATE 1 — held bin-mass NLL: unimodal {nll1.mean():+.4f}  mixture {nll2.mean():+.4f}  "
              f"Δ {d.mean():+.4f} (row-SE {d.std() / np.sqrt(len(d)):.4f}) "
              f"{'← IMPROVES' if d.mean() > 2 * d.std() / np.sqrt(len(d)) else '← no clear gain'}")
        # honesty slice: the defect was diagnosed on TRANSITIVE data — does the gain survive inside the
        # stratum (where the stratum one-hot can't carry it)?
        tm = np.array([tagg[i] == "trans" for i in he])
        dt = d[tm]
        print(f"    trans-only held (n={int(tm.sum())}): unimodal {nll1[tm].mean():+.4f}  "
              f"mixture {nll2[tm].mean():+.4f}  Δ {dt.mean():+.4f} (row-SE {dt.std() / np.sqrt(len(dt)):.4f})")

        # gate 2: class-posterior calibration on held
        bins = np.linspace(0, 1, 6)
        print("GATE 2 — P(directional|context) calibration (held):")
        ece, tot = 0.0, 0
        for lo, hi_ in zip(bins[:-1], bins[1:]):
            m = (pc[he] >= lo) & (pc[he] < hi_)
            if m.sum() < 5:
                continue
            emp = c[he][m].mean()
            ece += m.sum() * abs(pc[he][m].mean() - emp); tot += m.sum()
            print(f"    pred [{lo:.1f},{hi_:.1f}): n {int(m.sum()):3d}  mean pred {pc[he][m].mean():.2f}  empirical {emp:.2f}")
        print(f"    ECE ≈ {ece / max(tot, 1):.3f}")
        print(f"GATE 3 — classes anchored to the relation family (directional μ ≥ 0.5), predicted from "
              f"judge-free context (prior readouts, graph walk, stratum)")


if __name__ == "__main__":
    main()
