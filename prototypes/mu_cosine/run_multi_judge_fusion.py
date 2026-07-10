#!/usr/bin/env python3
"""Multi-judge fusion — the fused-head null's boundary case #1, tested on existing data (zero scoring cost).

REPORT_fused_head.md: with a single reliable judge (R≈0.004) the Kalman posterior collapses onto the label
and fusion machinery adds nothing. Boundary claim: a NOISY judge (luna: R ~5-10× 5.5's, opposite-signed
biases) makes the fusion non-degenerate. The fresh 250 already carries everything: gpt-5.5 run 1 (target),
gpt-5.5 run 2 + gpt-5.6-luna (measurement channels), the graph walk, the model prior.

Setup mirrors run_judge_channel.py (constant blocks, correlated updates, descendant-disjoint splits): the
7×7 joint residual covariance [prior_D, prior_S, graph, j2_D, j2_S, luna_D, luna_S] is fit on each train
split — so luna's cross-correlations (with 5.5, with the prior) are PRICED, not assumed.

Target honesty: y = judge1 (gpt-5.5 run 1) = the campaign judge's labels. This measures "predict the
operating judge", not ground truth — j2 is same-family and thus privileged (its fitted R absorbs shared
5.5 bias); luna's fitted R absorbs its true tilt. That is the operationally-correct frame (5.5 IS the
campaign judge) but it lower-bounds luna's value against an independent truth.

Economics ladder (cost ↑): prior | +graph | +luna (cheap judge) | +j2 (full-price judge) | +j2+luna.

  python3 run_multi_judge_fusion.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from product_kalman import fit_residual_covariance
from run_judge_channel import correlated_update_H, nll_mahal
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS, affine_calibrate
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
J2_TSV = "/tmp/mu_data/sigma_hop_fresh_scored_j2.tsv"
LUNA_TSV = "/tmp/mu_data/sigma_hop_fresh_scored_luna.tsv"
# measurement rows: graph→D, j2D→D, j2S→S, lunaD→D, lunaS→S
H5 = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
RUNGS = {  # name → measurement row indices (cost order)
    "prior": [],
    "prior+graph": [0],
    "prior+graph+luna": [0, 3, 4],
    "prior+graph+j2": [0, 1, 2],
    "prior+graph+j2+luna": [0, 1, 2, 3, 4],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--shrink", type=float, default=0.05)
    a = ap.parse_args()

    cfg = DATASETS["fresh"]
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"], FeatureGraphConfig(**cfg["graph"]),
        os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    by = {}
    for path, key in [(J2_TSV, "j2"), (LUNA_TSV, "luna")]:
        p, d_, s_ = load_scored_mu_tsv(path)
        by[key] = {pp: (d_[i], s_[i]) for i, pp in enumerate(p)}
    keep = [i for i, p in enumerate(data.pairs) if p in by["j2"] and p in by["luna"]]
    print(f"fresh pairs with prior+graph+j2+luna: {len(keep)}")
    prior = data.X[keep, :2]; d = data.X[keep, 2]
    y = dequant(np.column_stack([data.D[keep], data.S[keep]]))          # judge1 = the operating target
    j2 = np.array([by["j2"][data.pairs[i]] for i in keep])
    luna = np.array([by["luna"][data.pairs[i]] for i in keep])
    pk = [data.pairs[i] for i in keep]

    acc = {r: {"nll": [], "m2": []} for r in RUNGS}
    lg = {"+luna_after_j2": [], "+j2_after_luna": []}
    used = 0
    for seed in range(a.seeds):
        tr, he = descendant_disjoint_split(pk, seed, held_frac=0.30)
        if len(tr) < 30 or len(he) < 12:
            continue
        used += 1
        m = affine_calibrate(d[tr], y[tr, 0], d)
        meas = np.column_stack([m, j2[:, 0], j2[:, 1], luna[:, 0], luna[:, 1]])
        E = np.column_stack([y - prior, meas - y[:, [0, 0, 1, 0, 1]]])[tr]   # 2 prior + 5 meas errors
        C7 = fit_residual_covariance(E, shrinkage=a.shrink)
        P0, C_pm, R0 = C7[:2, :2], C7[:2, 2:], C7[2:, 2:]
        for i in he:
            x = prior[i]
            row = {}
            for rung, sel in RUNGS.items():
                if not sel:
                    xp, Pp = x, P0
                else:
                    xp, Pp = correlated_update_H(x, P0, meas[i][sel], R0[np.ix_(sel, sel)],
                                                 C_pm[:, sel], H5[sel])
                nll, m2 = nll_mahal(y[i] - xp, Pp)
                acc[rung]["nll"].append(nll); acc[rung]["m2"].append(m2)
                row[rung] = nll
            lg["+luna_after_j2"].append(row["prior+graph+j2"] - row["prior+graph+j2+luna"])
            lg["+j2_after_luna"].append(row["prior+graph+luna"] - row["prior+graph+j2+luna"])

    print(f"\nfusion ladder ({used} splits; NLL ↓, Mahal/dim ≈ 1):")
    print(f"    {'rung':24s} {'NLL':>8s} {'Mahal/dim':>10s}")
    base = None
    for r in RUNGS:
        nll = np.array(acc[r]["nll"]); m2 = np.array(acc[r]["m2"])
        print(f"    {r:24s} {nll.mean():+8.4f} {m2.mean()/2:10.2f}")
        if r == "prior+graph":
            base = nll
    for nm, ref in [("luna (cheap judge)", "prior+graph+luna"), ("j2 (full-price judge)", "prior+graph+j2")]:
        g = base - np.array(acc[ref]["nll"])
        print(f"    value of {nm:22s} over prior+graph: {g.mean():+.4f} (row-SE {g.std()/np.sqrt(len(g)):.4f})")
    for nm, g in lg.items():
        g = np.array(g)
        print(f"    marginal {nm:18s}: {g.mean():+.4f} (row-SE {g.std()/np.sqrt(len(g)):.4f})")

    # how non-degenerate is the luna fusion? posterior pull off the raw luna measurement
    print("\nfusion non-degeneracy (mean |posterior − measurement|, D channel, last split):")
    for rung, mcol in [("prior+graph+luna", 3), ("prior+graph+j2", 1)]:
        sel = RUNGS[rung]
        pull = []
        for i in range(len(y)):
            xp, _ = correlated_update_H(prior[i], P0, meas[i][sel], R0[np.ix_(sel, sel)], C_pm[:, sel], H5[sel])
            pull.append(abs(xp[0] - meas[i][mcol]))
        print(f"    {rung:24s} |post_D − judge_D| = {np.mean(pull):.3f}")


if __name__ == "__main__":
    main()
