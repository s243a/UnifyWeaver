#!/usr/bin/env python3
"""Confirm the symmetric-graph S channel with strict node-disjoint evaluation.

Every fusion before PR #3648 had no graph measurement of S ("the graph doesn't observe S"), so the S
posterior was prior⊕judge only.  The graph does carry symmetric structure: common-ancestor lateral distance,
shared-parent/grandparent, and ancestor flags.  This script builds a graph_S measurement — a small linear model
on those features, calibrated to S on the train split — and adds it as an S measurement row (H=[0,1]).

Ladder (40 node-disjoint splits, joint 6×6 fit per split, correlated updates):
  prior | +graph_D | +graph_D+graph_S (FREE-ONLY fusion) | +graph_D+luna | ALL

Two uncertainty summaries are intentionally separate:
  * SD across seeded node partitions is descriptive Monte Carlo split stability, not an SE or CI.
  * A paired two-endpoint node-block bootstrap on one fixed held partition gives a 95% population interval
    conditional on that fitted split.

--debias affine+bins (DESIGN_bias_state_augmentation.md §5.1) additionally evaluates the treatment
ladder where every measurement channel is corrected by shrunk per-(judge, bin, channel) offset states
(fit_bias_states, train-split-only, ON TOP of the retained global affine — never replacing it) and
reports the PAIRED control-vs-treatment gains.  Frozen primary metric: held-out S-marginal NLL at the
ALL rung; D-marginal and joint NLL are secondary.

  python3 run_sym_channel_fusion.py [--debias affine+bins]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from fine_tune_channel_heads import load_campaign_datasets, load_expanded
from fine_tune_fused_head import agnostic_readouts
from fit_bias_states import DEFAULT_TAUS, fit_bias_states, pair_distance_features
from node_disjoint_eval import (
    format_split_diagnostics,
    node_disjoint_pair_split,
    paired_node_bootstrap_ci,
)
from product_kalman import fit_residual_covariance
from run_judge_channel import correlated_update_H, nll_mahal
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS, affine_calibrate
from sample_channel_campaign import ancestors
from sigma_hop_confirmatory import FeatureGraphConfig, load_feature_graph

ROOT = os.path.dirname(os.path.abspath(__file__))
LUNA_CAMPAIGN = "/tmp/mu_data/campaign_scored_luna.tsv"
LOG2PI = float(np.log(2 * np.pi))
H4 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)  # gD→D, gS→S, lunaD→D, lunaS→S
RUNGS = {
    "prior": [],
    "+graph_D": [0],
    "+graph_D+graph_S": [0, 1],
    "+graph_D+luna": [0, 2, 3],
    "ALL": [0, 1, 2, 3],
}
EFFECTS = [
    ("graph_S free-only (S)", "+graph_D", "+graph_D+graph_S"),
    ("graph_S after luna (S)", "+graph_D+luna", "ALL"),
]
# meas column order ↔ (judge, channel) bias-state keys and their observed y component
BIAS_CHANNELS = ((("graph", "D"), 0), (("graph", "S"), 1), (("luna", "D"), 0), (("luna", "S"), 1))
# paired control-vs-treatment gains at the ALL rung; S-marginal is the FROZEN primary metric
DEBIAS_EFFECTS = [
    ("debias bins (S, ALL) [primary]", "s"),
    ("debias bins (D, ALL)", "d"),
    ("debias bins (joint, ALL)", "j"),
]


def sym_graph_features(parents, pairs, hmax=6, cap=13):
    """[N, 4]: inverse common-ancestor distance plus three symmetric graph indicators."""
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
        feats[i] = (
            1.0 / (1.0 + d_sym),
            float(bool(px & py)),
            float(bool(gx & gy)),
            float(y in ax or x in ay),
        )
    return feats


def calibrate_luna(luna, y, fit_idx):
    """Fit global per-channel affine Luna calibration on train/overlap rows, then apply to all rows."""
    cal_D = affine_calibrate(luna[fit_idx, 0], y[fit_idx, 0], luna[:, 0])
    cal_S = affine_calibrate(luna[fit_idx, 1], y[fit_idx, 1], luna[:, 1])
    return np.column_stack([cal_D, cal_S])


def s_marginal_nll(r, v):
    return 0.5 * (r * r / v + np.log(v) + LOG2PI)


def build_arg_parser():
    ap = argparse.ArgumentParser()
    # Campaign-independent prior: model_prod_namecond.pt was not fine-tuned on the evaluated campaign.
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    ap.add_argument(
        "--seeds",
        type=int,
        default=40,
        help="node-partition seeds used for descriptive Monte Carlo split stability",
    )
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument(
        "--held-node-frac",
        type=float,
        default=0.40,
        help=("fraction of unique nodes assigned held; 0.40 gives about 30% held among retained pairs; "
              "cross-partition pairs are discarded"),
    )
    ap.add_argument(
        "--split-candidates",
        type=int,
        default=64,
        help="outcome-blind node assignments tried per seed to balance pair-stratum coverage",
    )
    ap.add_argument(
        "--minimum-per-stratum",
        type=int,
        default=1,
        help="minimum train and held pairs required in every campaign stratum",
    )
    ap.add_argument("--min-train", type=int, default=30)
    ap.add_argument("--min-held", type=int, default=12)
    ap.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="paired two-endpoint node-block draws on the fixed held split; 0 disables",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=1729)
    ap.add_argument(
        "--bootstrap-split-seed",
        type=int,
        default=0,
        help="node-partition seed held fixed for the population-uncertainty interval",
    )
    ap.add_argument("--bootstrap-confidence", type=float, default=0.95)
    ap.add_argument(
        "--debias",
        choices=["affine", "affine+bins"],
        default="affine",
        help=("affine = the existing global per-channel calibration only (control); affine+bins ALSO "
              "evaluates shrunk per-(judge,bin,channel) offset states on top of the affine "
              "(fit_bias_states, train-only) and reports paired control-vs-treatment gains"),
    )
    ap.add_argument("--bias-prior-sd", type=float, default=0.10,
                    help="zero-prior scale of each bias state (ridge shrinkage strength)")
    ap.add_argument("--bias-info-floor", type=float, default=0.10,
                    help="minimum conditional information ratio; states below fall back to the prior")
    ap.add_argument("--bias-taus", type=float, nargs="+", default=list(DEFAULT_TAUS),
                    help="candidate kernel bandwidths for the train-tuned soft bin assignment")
    return ap


def _mean_sd(values):
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else float("nan")
    return float(values.mean()), float(sd)


def _validate_args(a):
    if a.seeds < 1:
        raise ValueError("--seeds must be positive")
    if a.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be non-negative")
    if a.bootstrap_seed < 0 or a.bootstrap_split_seed < 0:
        raise ValueError("bootstrap seeds must be non-negative")
    if not 0.0 < a.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must be strictly between 0 and 1")
    if a.min_train < 1 or a.min_held < 1:
        raise ValueError("--min-train and --min-held must be positive")


def main(argv=None):
    a = build_arg_parser().parse_args(argv)
    _validate_args(a)

    ref, _ = load_expanded(a.ckpt, dev="cpu")
    ref.eval()
    lp, lD, lS = load_scored_mu_tsv(LUNA_CAMPAIGN)
    luna_by = {p: (lD[i], lS[i]) for i, p in enumerate(lp)}
    dss = load_campaign_datasets()

    for corpus_index, (name, ds) in enumerate(dss.items()):
        corpus = name.replace("-campaign", "")
        parents, children, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
        keep = [i for i, pair in enumerate(ds["pairs"]) if pair in luna_by]
        pairs = [ds["pairs"][i] for i in keep]
        all_tags = ds.get("tags", ["all"] * len(ds["pairs"]))
        tags = [all_tags[i] for i in keep]
        y = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
        ro = agnostic_readouts(ref, ds, "cpu")
        prior = np.column_stack([ro["prior_D"][keep], ro["prior_S"][keep]])
        d = ds["d"][keep]
        luna = np.array([luna_by[pair] for pair in pairs])
        F = sym_graph_features(parents, pairs)

        def corr(x, target):
            return float(np.corrcoef(x, target)[0, 1])

        print(
            f"\n=== {name}: {len(pairs)} rows; raw corr(1/(1+d_sym), S) = {corr(F[:, 0], y[:, 1]):+.3f}, "
            f"corr(shared_parent, S) = {corr(F[:, 1], y[:, 1]):+.3f} ==="
        )

        variants = ["affine"]
        if a.debias == "affine+bins":
            variants.append("affine+bins")
            in_graph = set(parents) | {c for kids in children.values() for c in kids}
            # outcome-blind distance features, computed once per corpus (graph only, never labels)
            feats = pair_distance_features(parents, pairs, in_graph=in_graph)

        # Split-seed values answer only whether the result is stable to the node partition. They are not
        # independent replications, so their SD is descriptive and is never divided by sqrt(n).
        split_scores = {v: {rung: {"j": [], "s": [], "d": []} for rung in RUNGS} for v in variants}
        valid_splits = []
        skipped = []
        fixed = None
        fixed_states = None
        bias_fits = []
        mc_seeds = list(range(a.seeds))
        eval_seeds = list(mc_seeds)
        if a.bootstrap_resamples and a.bootstrap_split_seed not in eval_seeds:
            eval_seeds.append(a.bootstrap_split_seed)

        for seed in eval_seeds:
            split = node_disjoint_pair_split(
                pairs,
                seed,
                held_node_fraction=a.held_node_frac,
                strata=tags,
                candidates=a.split_candidates,
                minimum_per_stratum=a.minimum_per_stratum,
            )
            tr, he = split.train, split.held
            missing_train, missing_held = split.missing_strata(a.minimum_per_stratum)
            reasons = []
            if len(tr) < a.min_train:
                reasons.append(f"train {len(tr)} < {a.min_train}")
            if len(he) < a.min_held:
                reasons.append(f"held {len(he)} < {a.min_held}")
            if missing_train:
                reasons.append(f"train strata below minimum: {','.join(map(str, missing_train))}")
            if missing_held:
                reasons.append(f"held strata below minimum: {','.join(map(str, missing_held))}")
            if reasons:
                if seed in mc_seeds:
                    skipped.append((seed, "; ".join(reasons)))
                if a.bootstrap_resamples and seed == a.bootstrap_split_seed:
                    raise RuntimeError(
                        f"fixed bootstrap split seed {seed} is invalid: {'; '.join(reasons)}; "
                        "choose another --bootstrap-split-seed or increase --split-candidates"
                    )
                continue

            graph_D = affine_calibrate(d[tr], y[tr, 0], d)
            X = np.column_stack([F, np.ones(len(F))])
            beta, *_ = np.linalg.lstsq(X[tr], y[tr, 1], rcond=None)
            graph_S = X @ beta
            luna_calibrated = calibrate_luna(luna, y, tr)
            meas = np.column_stack([graph_D, graph_S, luna_calibrated[:, 0], luna_calibrated[:, 1]])
            meas_by = {"affine": meas}
            if "affine+bins" in variants:
                # residual offsets fit ON TOP of the affine calibration above (never replacing it),
                # train rows only; applied to every row through the outcome-blind kernel basis
                states = fit_bias_states(
                    feats,
                    tr,
                    {key: meas[:, j] - y[:, y_col] for j, (key, y_col) in enumerate(BIAS_CHANNELS)},
                    prior_sd=a.bias_prior_sd,
                    info_floor=a.bias_info_floor,
                    taus=a.bias_taus,
                    cv_groups=[min(pair) for pair in pairs],
                    verbose=False,
                )
                meas_by["affine+bins"] = meas - np.column_stack(
                    [states.corrections(key) for key, _ in BIAS_CHANNELS]
                )
                if seed in mc_seeds:
                    bias_fits.append(states)
                if a.bootstrap_resamples and seed == a.bootstrap_split_seed:
                    fixed_states = states

            row_scores = {v: {rung: {"j": [], "s": [], "d": []} for rung in RUNGS} for v in variants}
            for v in variants:
                mv = meas_by[v]
                errors = np.column_stack([y - prior, mv - y[:, [0, 1, 0, 1]]])[tr]
                covariance = fit_residual_covariance(errors, shrinkage=a.shrink)
                P0, C_pm, R0 = covariance[:2, :2], covariance[:2, 2:], covariance[2:, 2:]
                for i in he:
                    x = prior[i]
                    for rung, selected in RUNGS.items():
                        if not selected:
                            xp, Pp = x, P0
                        else:
                            xp, Pp = correlated_update_H(
                                x,
                                P0,
                                mv[i][selected],
                                R0[np.ix_(selected, selected)],
                                C_pm[:, selected],
                                H4[selected],
                            )
                        row_scores[v][rung]["j"].append(nll_mahal(y[i] - xp, Pp)[0])
                        row_scores[v][rung]["s"].append(s_marginal_nll(y[i, 1] - xp[1], Pp[1, 1]))
                        row_scores[v][rung]["d"].append(s_marginal_nll(y[i, 0] - xp[0], Pp[0, 0]))

            if seed in mc_seeds:
                valid_splits.append(split)
                for v in variants:
                    for rung in RUNGS:
                        for m in ("j", "s", "d"):
                            split_scores[v][rung][m].append(float(np.mean(row_scores[v][rung][m])))
            if a.bootstrap_resamples and seed == a.bootstrap_split_seed:
                gains = {
                    effect: np.asarray(row_scores["affine"][base]["s"])
                    - np.asarray(row_scores["affine"][plus]["s"])
                    for effect, base, plus in EFFECTS
                }
                if "affine+bins" in variants:
                    for effect, m in DEBIAS_EFFECTS:
                        gains[effect] = np.asarray(row_scores["affine"]["ALL"][m]) - np.asarray(
                            row_scores["affine+bins"]["ALL"][m]
                        )
                fixed = {
                    "split": split,
                    "pairs": [pairs[i] for i in he],
                    "gains": gains,
                }

        if not valid_splits:
            raise RuntimeError("no valid node-disjoint Monte Carlo splits; inspect coverage settings")

        print(
            f"node-disjoint coverage: {len(valid_splits)}/{a.seeds} valid split seeds; "
            f"{len(skipped)} skipped (cross-partition pairs always discarded)"
        )
        train_sizes = np.array([len(split.train) for split in valid_splits])
        held_sizes = np.array([len(split.held) for split in valid_splits])
        cross_sizes = np.array([len(split.cross) for split in valid_splits])
        print(
            "    pair counts across valid seeds (mean [min,max]): "
            f"train {train_sizes.mean():.1f} [{train_sizes.min()},{train_sizes.max()}], "
            f"held {held_sizes.mean():.1f} [{held_sizes.min()},{held_sizes.max()}], "
            f"cross {cross_sizes.mean():.1f} [{cross_sizes.min()},{cross_sizes.max()}]"
        )
        for tag in sorted(valid_splits[0].strata, key=str):
            train_cov = np.array([split.strata[tag].train for split in valid_splits])
            held_cov = np.array([split.strata[tag].held for split in valid_splits])
            print(
                f"    stratum {str(tag):18.18s}: train min/median {train_cov.min()}/{np.median(train_cov):.0f}; "
                f"held min/median {held_cov.min()}/{np.median(held_cov):.0f}"
            )
        if skipped:
            sample = "; ".join(f"{seed} ({why})" for seed, why in skipped[:5])
            print(f"    skipped seeds: {sample}" + ("; ..." if len(skipped) > 5 else ""))

        # control-only runs keep the exact pre---debias output format (archived logs stay diffable);
        # the treatment run prints both variants with the added D-marginal column
        treatment = "affine+bins" in variants
        for v in variants:
            header = f"ladder [{v}]" if treatment else "ladder"
            print(f"{header} (mean across {len(valid_splits)} node-disjoint split means; NLL ↓):")
            print("    split SD is descriptive Monte Carlo partition stability, not an SE or confidence interval")
            cols = f"    {'rung':22s} {'joint mean ± SD':>20s} {'S-marginal mean ± SD':>23s}"
            if treatment:
                cols += f" {'D-marginal mean ± SD':>23s}"
            print(cols)
            for rung in RUNGS:
                joint_mean, joint_sd = _mean_sd(split_scores[v][rung]["j"])
                s_mean, s_sd = _mean_sd(split_scores[v][rung]["s"])
                line = f"    {rung:22s} {joint_mean:+8.4f} ± {joint_sd:7.4f} {s_mean:+11.4f} ± {s_sd:7.4f}"
                if treatment:
                    d_mean, d_sd = _mean_sd(split_scores[v][rung]["d"])
                    line += f" {d_mean:+11.4f} ± {d_sd:7.4f}"
                print(line)
        effect_width = 31 if treatment else 24
        for effect, base, plus in EFFECTS:
            gains = np.asarray(split_scores["affine"][base]["s"]) - np.asarray(split_scores["affine"][plus]["s"])
            mean, sd = _mean_sd(gains)
            print(
                f"    value of {effect:{effect_width}s}: {mean:+.4f} ± {sd:.4f} split SD; "
                f"{int((gains > 0).sum())}/{len(gains)} split seeds +"
            )
        if treatment:
            for effect, m in DEBIAS_EFFECTS:
                gains = np.asarray(split_scores["affine"]["ALL"][m]) - np.asarray(
                    split_scores["affine+bins"]["ALL"][m]
                )
                mean, sd = _mean_sd(gains)
                print(
                    f"    value of {effect:{effect_width}s}: {mean:+.4f} ± {sd:.4f} split SD; "
                    f"{int((gains > 0).sum())}/{len(gains)} split seeds +"
                )
            taus = [st.tau for st in bias_fits]
            print(
                f"bias-state fits across {len(bias_fits)} MC splits: "
                f"tau chosen {sorted(set(taus))} (mode {max(set(taus), key=taus.count)})"
            )
            for key, _ in BIAS_CHANNELS:
                fits = [st.fits[key] for st in bias_fits]
                print(
                    f"    {key[0]}.{key[1]:1s}: rank min/max {min(f.rank for f in fits)}/{max(f.rank for f in fits)}, "
                    f"cond max {max(f.cond for f in fits):.1f}, "
                    f"fallbacks/split mean {np.mean([f.fallback.sum() for f in fits]):.1f} "
                    f"(fail-closed states reverted to prior)"
                )

        if a.bootstrap_resamples:
            if fixed is None:
                raise RuntimeError(f"fixed bootstrap split seed {a.bootstrap_split_seed} was not evaluated")
            print(
                f"fixed node-disjoint held evaluation (seed {a.bootstrap_split_seed}, "
                f"selected candidate {fixed['split'].selected_candidate + 1}/{a.split_candidates}):"
            )
            for line in format_split_diagnostics(fixed["split"]).splitlines():
                print(f"    {line}")
            if fixed_states is not None:
                print("    bias-state diagnostics on this fitted split (per fit, fail-closed):")
                for line in fixed_states.diagnostics_lines():
                    print(f"      {line}")
            print(
                f"    paired two-endpoint node-block bootstrap ({a.bootstrap_confidence:.0%} percentile CI; "
                "conditional on this fitted split, not split-seed variability):"
            )
            effect_names = [effect for effect, _, _ in EFFECTS]
            if treatment:
                effect_names += [effect for effect, _ in DEBIAS_EFFECTS]
            boot_width = 31 if treatment else 27
            for effect_index, effect in enumerate(effect_names):
                interval = paired_node_bootstrap_ci(
                    fixed["pairs"],
                    fixed["gains"][effect],
                    n_resamples=a.bootstrap_resamples,
                    seed=a.bootstrap_seed + 1009 * corpus_index + effect_index,
                    confidence=a.bootstrap_confidence,
                )
                print(
                    f"      {effect:{boot_width}s}: {interval.estimate:+.4f} "
                    f"[{interval.low:+.4f}, {interval.high:+.4f}] "
                    f"(B={interval.n_resamples}, held-node resampling uncertainty)"
                )


if __name__ == "__main__":
    main()
