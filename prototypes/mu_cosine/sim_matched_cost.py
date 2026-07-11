#!/usr/bin/env python3
"""Matched-cost simulation (DESIGN_cheap_judge_pipeline §5.2): at equal scoring budget, do n pure-5.5
labels or the cheap-judge scheme (0.3n overlap + luna bulk with fused targets) train a better predictor?

Budget accounting (in 5.5-call units, price ratio k): arm A spends n on n 5.5-labeled pairs. Arm B spends
0.3n·(1+1/k) on a dual-scored overlap and the remaining budget on luna-only bulk at 1/k per pair:
n_bulk = k·(0.7n − 0.3n/k) = 0.7kn − 0.3n. Fusion blocks (prior ⊕ graph_D ⊕ graph_S ⊕ luna, correlated)
fit on the overlap; the bulk trains on fused posteriors; the overlap trains on its 5.5 labels.

Downstream estimator: ridge regression from frozen e5 pair-features (p_x⊙q_y ++ |p_x−q_y|) — a fast proxy
for the head fine-tune; both arms use the SAME estimator so the comparison is about label quality×quantity,
not the estimator (caveat: absolute numbers are proxy-level; the transformer head sees more). Eval: held
corr vs 5.5 labels (D and S), 10 resamples per grid point.

  python3 sim_matched_cost.py --k 2 4 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from fine_tune_channel_heads import load_campaign_datasets, load_expanded
from fine_tune_fused_head import agnostic_readouts
from product_kalman import fit_residual_covariance
from run_judge_channel import correlated_update_H
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS, affine_calibrate
from run_sym_channel_fusion import H4, sym_graph_features
from sigma_hop_confirmatory import FeatureGraphConfig, descendant_disjoint_split, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
LUNA_CAMPAIGN = "/tmp/mu_data/campaign_scored_luna.tsv"


def ridge_fit_predict(X, y, Xq, lams=(3.0, 30.0, 300.0, 3000.0)):
    """Ridge with λ selected on an inner 80/20 holdout — a fixed λ sits near the interpolation
    threshold (n ≈ 768 features) at the large-n grid points and double-descent wrecks the curve."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    n = len(Z); n_in = max(20, int(0.8 * n))
    G = Z.T @ Z; b = Z.T @ (y - y.mean())
    Gi = Z[:n_in].T @ Z[:n_in]; bi = Z[:n_in].T @ (y[:n_in] - y[:n_in].mean())
    best, w_best = -2.0, None
    for lam in lams:
        wi = np.linalg.solve(Gi + lam * np.eye(Z.shape[1]), bi)
        pv = Z[n_in:] @ wi
        c = np.corrcoef(pv, y[n_in:])[0, 1] if pv.std() > 1e-9 else -1.0
        if c > best:
            best, w_best = c, np.linalg.solve(G + lam * np.eye(Z.shape[1]), b)
    return ((Xq - mu) / sd) @ w_best + y.mean()


def fused_targets(prior, meas, y, fit_idx, shrink=0.05):
    """Blocks fit on fit_idx (the overlap); posteriors for ALL rows."""
    E = np.column_stack([y - prior, meas - y[:, [0, 1, 0, 1]]])[fit_idx]
    C6 = fit_residual_covariance(E, shrinkage=shrink)
    P0, C_pm, R0 = C6[:2, :2], C6[:2, 2:], C6[2:, 2:]
    post = np.zeros_like(prior)
    for i in range(len(prior)):
        xp, _ = correlated_update_H(prior[i], P0, meas[i], R0, C_pm, H4)
        post[i] = np.clip(xp, 0.0, 1.0)
    return post


def main():
    ap = argparse.ArgumentParser()
    # Campaign-INDEPENDENT prior (blocker 1) — see run_sym_channel_fusion.py.
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    ap.add_argument("--k", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--n", type=int, nargs="+", default=[80, 160, 320, 640])
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()

    ref, _ = load_expanded(a.ckpt, dev="cpu")
    ref.eval()
    lp, lD, lS = load_scored_mu_tsv(LUNA_CAMPAIGN)
    luna_by = {p: (lD[i], lS[i]) for i, p in enumerate(lp)}
    dss = load_campaign_datasets()

    for n_name, ds in dss.items():
        corpus = n_name.replace("-campaign", "")
        parents, _, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
        keep = [i for i, p in enumerate(ds["pairs"]) if p in luna_by]
        pairs = [ds["pairs"][i] for i in keep]
        y = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
        ro = agnostic_readouts(ref, ds, "cpu")
        prior = np.column_stack([ro["prior_D"][keep], ro["prior_S"][keep]])
        d = ds["d"][keep]
        luna = np.array([luna_by[p] for p in pairs])
        F = sym_graph_features(parents, pairs)
        tok = ds["tok"]
        ii = [tok.idx[x] for x, _ in pairs]; jj = [tok.idx[yy] for _, yy in pairs]
        px = tok.p[ii].numpy(); qy = tok.q[jj].numpy()
        feat = np.column_stack([px * qy, np.abs(px - qy)])

        tr, he = descendant_disjoint_split(pairs, 0, held_frac=0.30)
        tr, he = np.array(tr), np.array(he)
        print(f"\n=== {n_name}: {len(pairs)} rows (pool {len(tr)}, held {len(he)}) ===")
        print(f"{'n':>5s} {'arm':>14s}" + "".join(f" {c:>8s}" for c in ("D corr", "S corr")))
        rngs = np.random.default_rng(0)
        for n in a.n:
            if n > len(tr):
                continue
            res = {}
            for rep in range(a.reps):
                sel = rngs.permutation(tr)
                # arm A: n pure 5.5 labels
                A_idx = sel[:n]
                for ch, col in (("D", 0), ("S", 1)):
                    pred = ridge_fit_predict(feat[A_idx], y[A_idx, col], feat[he])
                    res.setdefault(("A: 5.5 only", ch), []).append(np.corrcoef(pred, y[he, col])[0, 1])
                # arm B per k: overlap 0.3n (5.5 labels) + bulk fused targets
                m_cal = affine_calibrate(d[sel[:max(30, int(0.3 * n))]], y[sel[:max(30, int(0.3 * n))], 0], d)
                X = np.column_stack([F, np.ones(len(F))])
                ov = sel[:max(30, int(0.3 * n))]
                beta, *_ = np.linalg.lstsq(X[ov], y[ov, 1], rcond=None)
                meas = np.column_stack([m_cal, X @ beta, luna[:, 0], luna[:, 1]])
                post = fused_targets(prior, meas, y, ov)
                for k in a.k:
                    n_bulk = int(0.7 * k * n - 0.3 * n)
                    B_bulk = sel[len(ov):len(ov) + min(n_bulk, len(sel) - len(ov))]
                    idx = np.concatenate([ov, B_bulk])
                    for ch, col in (("D", 0), ("S", 1)):
                        tgt = np.concatenate([y[ov, col], post[B_bulk, col]])
                        pred = ridge_fit_predict(feat[idx], tgt, feat[he])
                        res.setdefault((f"B: scheme k={k:g}", ch), []).append(
                            np.corrcoef(pred, y[he, col])[0, 1])
            arms = sorted({arm for arm, _ in res})
            for arm in arms:
                cd = np.array(res[(arm, "D")]); cs = np.array(res[(arm, "S")])
                print(f"{n:>5d} {arm:>14s} {cd.mean():+8.3f} {cs.mean():+8.3f}"
                      f"   (±{cd.std():.3f}/±{cs.std():.3f})")


if __name__ == "__main__":
    main()
