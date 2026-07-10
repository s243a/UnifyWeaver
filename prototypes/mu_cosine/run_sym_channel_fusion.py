#!/usr/bin/env python3
"""The symmetric-graph S channel (user, DESIGN_cheap_judge_pipeline §5.1) — does S fusion become
non-trivial the way D's did?

Every fusion so far had NO graph measurement of S ("the graph doesn't observe S"), so the S posterior was
prior⊕judge only. But the graph DOES carry symmetric structure: common-ancestor lateral distance,
shared-parent/grandparent, ancestor flags. This builds a graph_S measurement — a small linear model on
those features, calibrated to S on the train split (the multivariate analog of affine_calibrate for d→D) —
and adds it as an S measurement row (H=[0,1]).

Ladder (40 descendant-disjoint splits, joint 6×6 fit per split, correlated updates):
  prior | +graph_D | +graph_D+graph_S (FREE-ONLY fusion) | +graph_D+luna | ALL
Reported: joint NLL and the S-MARGINAL NLL (the channel under test). Two questions:
  (a) free-only: does graph_S improve S with no judge at all (the deployment cheap path)?
  (b) marginal: does graph_S still add after luna?

  python3 run_sym_channel_fusion.py
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
from run_judge_channel import correlated_update_H, nll_mahal
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS, affine_calibrate
from sample_channel_campaign import ancestors
from sigma_hop_confirmatory import FeatureGraphConfig, descendant_disjoint_split, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
LUNA_CAMPAIGN = "/tmp/mu_data/campaign_scored_luna.tsv"
LOG2PI = float(np.log(2 * np.pi))
H4 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)   # gD→D, gS→S, lunaD→D, lunaS→S
RUNGS = {"prior": [], "+graph_D": [0], "+graph_D+graph_S": [0, 1],
         "+graph_D+luna": [0, 2, 3], "ALL": [0, 1, 2, 3]}


def sym_graph_features(parents, pairs, hmax=6, cap=13):
    """[N, 4]: 1/(1+d_sym) with d_sym = min common-ancestor hop sum (self counts at 0);
    shared-parent, shared-grandparent, is-ancestor indicators."""
    feats = np.zeros((len(pairs), 4))
    anc_cache = {}

    def anc(n):
        if n not in anc_cache:
            a = ancestors(parents, n, hmax)
            a[n] = 0
            anc_cache[n] = a
        return anc_cache[n]

    for i, (x, y) in enumerate(pairs):
        ax, ay = anc(x), anc(y)
        common = set(ax) & set(ay)
        d_sym = min((ax[c] + ay[c] for c in common), default=cap)
        px, py = set(parents.get(x, ())), set(parents.get(y, ()))
        gx = {g for p in px for g in parents.get(p, ())}
        gy = {g for p in py for g in parents.get(p, ())}
        feats[i] = (1.0 / (1.0 + d_sym), float(bool(px & py)), float(bool(gx & gy)),
                    float(y in ax or x in ay))
    return feats


def s_marginal_nll(r, v):
    return 0.5 * (r * r / v + np.log(v) + LOG2PI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_channel_heads_namecond_r0.pt"))
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--shrink", type=float, default=0.05)
    a = ap.parse_args()

    ref, _ = load_expanded(a.ckpt, dev="cpu")
    ref.eval()
    lp, lD, lS = load_scored_mu_tsv(LUNA_CAMPAIGN)
    luna_by = {p: (lD[i], lS[i]) for i, p in enumerate(lp)}
    dss = load_campaign_datasets()

    for n, ds in dss.items():
        corpus = n.replace("-campaign", "")
        parents, _, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
        keep = [i for i, p in enumerate(ds["pairs"]) if p in luna_by]
        pairs = [ds["pairs"][i] for i in keep]
        y = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
        ro = agnostic_readouts(ref, ds, "cpu")
        prior = np.column_stack([ro["prior_D"][keep], ro["prior_S"][keep]])
        d = ds["d"][keep]
        luna = np.array([luna_by[p] for p in pairs])
        F = sym_graph_features(parents, pairs)
        r = lambda x_, y_: float(np.corrcoef(x_, y_)[0, 1])
        print(f"\n=== {n}: {len(pairs)} rows; raw corr(1/(1+d_sym), S) = {r(F[:, 0], y[:, 1]):+.3f}, "
              f"corr(shared_parent, S) = {r(F[:, 1], y[:, 1]):+.3f} ===")

        acc = {rn: {"j": [], "s": []} for rn in RUNGS}
        used = 0
        for seed in range(a.seeds):
            tr, he = descendant_disjoint_split(pairs, seed, held_frac=0.30)
            if len(tr) < 30 or len(he) < 12:
                continue
            used += 1
            m = affine_calibrate(d[tr], y[tr, 0], d)
            X = np.column_stack([F, np.ones(len(F))])
            beta, *_ = np.linalg.lstsq(X[tr], y[tr, 1], rcond=None)   # graph_S: linear fit to S on train
            gs = X @ beta
            meas = np.column_stack([m, gs, luna[:, 0], luna[:, 1]])
            E = np.column_stack([y - prior, meas - y[:, [0, 1, 0, 1]]])[tr]
            C6 = fit_residual_covariance(E, shrinkage=a.shrink)
            P0, C_pm, R0 = C6[:2, :2], C6[:2, 2:], C6[2:, 2:]
            for i in he:
                x = prior[i]
                for rn, sel in RUNGS.items():
                    if not sel:
                        xp, Pp = x, P0
                    else:
                        xp, Pp = correlated_update_H(x, P0, meas[i][sel], R0[np.ix_(sel, sel)],
                                                     C_pm[:, sel], H4[sel])
                    acc[rn]["j"].append(nll_mahal(y[i] - xp, Pp)[0])
                    acc[rn]["s"].append(s_marginal_nll(y[i, 1] - xp[1], Pp[1, 1]))

        print(f"ladder ({used} splits; NLL ↓):")
        print(f"    {'rung':22s} {'joint':>8s} {'S-marginal':>11s}")
        for rn in RUNGS:
            print(f"    {rn:22s} {np.mean(acc[rn]['j']):+8.4f} {np.mean(acc[rn]['s']):+11.4f}")
        for nm, base, plus in [("graph_S free-only (S)", "+graph_D", "+graph_D+graph_S"),
                               ("graph_S after luna (S)", "+graph_D+luna", "ALL")]:
            g = np.array(acc[base]["s"]) - np.array(acc[plus]["s"])
            print(f"    value of {nm:24s}: {g.mean():+.4f} (row-SE {g.std()/np.sqrt(len(g)):.4f})")


if __name__ == "__main__":
    main()
