#!/usr/bin/env python3
"""First REAL-DATA run of the Product-Kalman holdout harness (DESIGN_amortized_fusion_heads build step 1).

Wires the already-scored corpora into `evaluate_product_kalman_holdout`, which scores the fusion ladder per
descendant-disjoint split:
  prior              — model readouts (mu_D, mu_S) with fitted prior-error covariance P
  independent_kalman — Kalman update with the cross-covariance C zeroed (independent-experts fusion)
  product_kalman     — the correlated fusion (learned C)

State/target = continuous LLM labels (D, S). Prior = model readouts. Measurement channels:
  config "graph"     — the walk hit-prob, affine-calibrated to D on the calibration split, H=[1,0]
  config "graph+poe" — graph channel + PoE lower / noisy-OR upper channels (product_space) built from
                       (mu_D, graph) — deliberately correlated with the prior; the learned C must handle it
                       (DESIGN_product_kalman_poe variant 6-lite vs variant 1/3).

EXPLORATORY comparison (not preregistered): mean NLL over splits + split-level stability. Datasets:
  exploratory — 250 multihop pairs, 100k_cats TSV graph
  fresh       — 250 Behavior-slice pairs, enwiki_cats_correct scoped LMDB (same retained slice as the
                confirmatory run; features rebuilt with the confirmatory runner's own loaders)

  python3 run_product_kalman_realdata.py --dataset exploratory
  python3 run_product_kalman_realdata.py --dataset fresh
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from product_kalman_evaluation import evaluate_product_kalman_holdout
from product_space import product_lower, product_upper
from sigma_hop_confirmatory import (
    FeatureGraphConfig,
    build_confirmatory_data_from_labels,
    descendant_disjoint_split,
    load_scored_pairs,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))

DATASETS = {
    "exploratory": dict(
        score_in="/tmp/mu_data/multihop_score_in.tsv",
        responses="/tmp/mu_data/multihop_resp.txt",
        e5_cache="/tmp/mu_data/multihop_e5.pt",
        graph=dict(graph=os.path.join(REPO, "data", "benchmark", "100k_cats", "category_parent.tsv")),
    ),
    "fresh": dict(
        score_in="/tmp/mu_data/sigma_hop_fresh_pairs.tsv",
        responses="/tmp/mu_data/sigma_hop_fresh_responses_gpt55low.txt",
        e5_cache="/tmp/mu_data/sigma_hop_behavior_slice_e5.pt",
        graph=dict(
            graph=None,
            candidate_lmdb=os.path.join(REPO, "data", "benchmark", "enwiki_cats_correct", "lmdb_scoped"),
            lmdb_root="Behavior",
            exploratory_graph=os.path.join(REPO, "data", "benchmark", "100k_cats", "category_parent.tsv"),
        ),
    ),
}


def affine_calibrate(d_cal, target_cal, d_all):
    """1-dim affine map d -> E[D] fit on the calibration split only (removes the walk's scale/floor bias so the
    harness's zero-mean measurement-error model applies; without it the bias skews every update)."""
    A = np.column_stack([d_cal, np.ones(len(d_cal))])
    beta, *_ = np.linalg.lstsq(A, target_cal, rcond=None)
    return np.clip(beta[0] * d_all + beta[1], 0.0, 1.0)


def poe_channels(muD, m):
    """PoE lower / noisy-OR upper channels over the two D-estimates (model readout, calibrated graph).
    Uses codex's product_space definitions row-wise."""
    lo = np.array([product_lower([a, b]) for a, b in zip(muD, m)])
    hi = np.array([product_upper([a, b]) for a, b in zip(muD, m)])
    return lo, hi


def run_dataset(name, seeds, shrinkage, held_frac=0.30, min_train=30, min_held=12):
    cfg = DATASETS[name]
    # load_scored_pairs + from_labels (not build_confirmatory_data): this is an EXPLORATORY comparison, the
    # preregistered no-overlap gate does not apply (the "exploratory" dataset IS the exploratory graph).
    pairs, hop, D, S = load_scored_pairs(cfg["score_in"], cfg["responses"], prefix="transitive_h")
    data = build_confirmatory_data_from_labels(
        pairs, hop, D, S, cfg["e5_cache"],
        FeatureGraphConfig(**cfg["graph"]), os.path.join(ROOT, "model_prod.pt"), "cpu",
    )
    prior = data.X[:, :2]                                   # (mu_D, mu_S) model readouts
    target = np.column_stack([data.D, data.S])              # continuous LLM labels
    d = data.X[:, 2]                                        # walk hit-prob
    print(f"\n=== {name}: n={len(data.pairs)} pairs, hops {sorted(set(data.hop.tolist()))} ===")

    configs = {
        "graph": dict(H=np.array([[1.0, 0.0]])),
        "graph+poe": dict(H=np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])),
    }
    results = {c: {} for c in configs}
    used = 0
    for seed in range(seeds):
        tr, he = descendant_disjoint_split(data.pairs, seed, held_frac=held_frac)
        if len(tr) < min_train or len(he) < min_held:
            continue
        used += 1
        m = affine_calibrate(d[tr], data.D[tr], d)          # calibrated graph channel (fit on train only)
        lo, hi = poe_channels(prior[:, 0], m)
        meas = {"graph": m[:, None], "graph+poe": np.column_stack([m, lo, hi])}
        for cname, ccfg in configs.items():
            ev = evaluate_product_kalman_holdout(
                prior[tr], meas[cname][tr], target[tr],
                prior[he], meas[cname][he], target[he],
                H=ccfg["H"], shrinkage=shrinkage,
            )
            for s in ev.scores:
                results[cname].setdefault(s.name, []).append(
                    (s.mean_nll, s.mse, s.mahalanobis_per_dim, s.squared_mahalanobis_q95))
    print(f"splits used: {used}/{seeds} (descendant-disjoint, held_frac={held_frac}, shrinkage={shrinkage})")

    summary = {}
    for cname, per_variant in results.items():
        print(f"\n  config [{cname}] — over {used} splits (NLL lower better; Mahal/dim ≈ 1 = calibrated error bars,")
        print(f"  >1 overconfident; msM q95 vs chi2_2 ref 5.99):")
        print(f"    {'variant':22s} {'NLL':>8s} {'MSE':>8s} {'Mahal/dim':>10s} {'msM q95':>8s}")
        for vname, vals in per_variant.items():
            nll = np.array([v[0] for v in vals]); mse = np.array([v[1] for v in vals])
            mpd = np.array([v[2] for v in vals]); q95 = np.array([v[3] for v in vals])
            print(f"    {vname:22s} {nll.mean():+8.4f} {mse.mean():8.4f} {mpd.mean():10.2f} {q95.mean():8.2f}")
            summary[(cname, vname)] = nll
        pr = summary[(cname, "prior")]
        for cand in ("independent_kalman", "product_kalman"):
            g = pr - summary[(cname, cand)]
            print(f"    NLL gain prior→{cand}: {g.mean():+.4f} (split-SE {g.std()/np.sqrt(len(g)):.4f} — stability "
                  f"only)  positive splits {int((g > 0).sum())}/{len(g)}")
        gi = summary[(cname, "independent_kalman")] - summary[(cname, "product_kalman")]
        print(f"    NLL gain independent(=Gaussian PoE)→product(correlated): {gi.mean():+.4f} "
              f"(split-SE {gi.std()/np.sqrt(len(gi)):.4f})  positive splits {int((gi > 0).sum())}/{len(gi)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--shrinkage", type=float, default=0.05)
    a = ap.parse_args()
    names = list(DATASETS) if a.dataset == "all" else [a.dataset]
    for n in names:                                          # sequential — consumer hardware
        run_dataset(n, a.seeds, a.shrinkage)


if __name__ == "__main__":
    main()
