#!/usr/bin/env python3
"""Run the frozen public Product-Kalman and JointPosterior holdout protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from mu_posterior import JointPosterior, aurc, aurc_boot, margin_conf
from product_kalman_calibration import apply_product_kalman_calibration
from product_kalman_evaluation import (
    evaluate_product_kalman_holdout,
    gaussian_marginal_pit_diagnostics,
    score_gaussian_prediction_vectors_rowwise,
    score_gaussian_predictions_rowwise,
)
from run_product_kalman_sigma_hop import chol_of_hop, correlated_update, fit_joint_sigma_of_hop


SCHEMA_VERSION = 1
FAMILY_ORDER = ("directional", "symmetric", "open_world")
RAW_SOURCE_NAMES = ("e5_fwd", "e5_rev", "model_D", "model_S", "graph_measurement", "hop")
CONTINUOUS_VARIANTS = ("prior", "independent_kalman", "product_kalman", "hop_product_kalman")
H = np.array([[1.0, 0.0]])
CALIBRATION_TOL = {"pit_ks": 0.02, "coverage_error": 0.02, "mahalanobis_distance": 0.10}


class PublicHoldoutError(ValueError):
    pass


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_feature_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {
        "pair_id", "corpus", "branch_unit", "descendant_identity", "ancestor_identity",
        "descendant_id", "ancestor_id", "hop", "operator_family",
        *RAW_SOURCE_NAMES[:-1], "target_D", "target_S",
    }
    missing = sorted(required - (set(rows[0]) if rows else set()))
    if missing:
        raise PublicHoldoutError(f"feature table is missing fields: {', '.join(missing)}")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise PublicHoldoutError("feature table pair IDs must be unique")
    if any(row["operator_family"] not in FAMILY_ORDER for row in rows):
        raise PublicHoldoutError("feature table contains an unknown operator family")
    return rows


def strict_branch_identity_split(rows, seed, evaluation_branch_frac=0.5):
    branches = sorted({row["branch_unit"] for row in rows})
    if len(branches) < 2:
        raise PublicHoldoutError("at least two branch units are required")
    random.Random(seed).shuffle(branches)
    n_eval = min(len(branches) - 1, max(1, int(evaluation_branch_frac * len(branches))))
    evaluation_branches = set(branches[:n_eval])
    calibration = [i for i, row in enumerate(rows) if row["branch_unit"] not in evaluation_branches]
    evaluation = [i for i, row in enumerate(rows) if row["branch_unit"] in evaluation_branches]
    calibration_identities = {
        identity
        for i in calibration
        for identity in (rows[i]["descendant_identity"], rows[i]["ancestor_identity"])
    }
    evaluation_identities = {
        identity
        for i in evaluation
        for identity in (rows[i]["descendant_identity"], rows[i]["ancestor_identity"])
    }
    overlap = calibration_identities & evaluation_identities
    calibration = [
        i for i in calibration
        if rows[i]["descendant_identity"] not in overlap and rows[i]["ancestor_identity"] not in overlap
    ]
    evaluation = [
        i for i in evaluation
        if rows[i]["descendant_identity"] not in overlap and rows[i]["ancestor_identity"] not in overlap
    ]
    retained_cal_ids = {
        identity
        for i in calibration
        for identity in (rows[i]["descendant_identity"], rows[i]["ancestor_identity"])
    }
    retained_eval_ids = {
        identity
        for i in evaluation
        for identity in (rows[i]["descendant_identity"], rows[i]["ancestor_identity"])
    }
    if retained_cal_ids & retained_eval_ids:
        raise PublicHoldoutError("internal error: identity closure overlaps after filtering")
    return np.asarray(calibration, dtype=int), np.asarray(evaluation, dtype=int), {
        "seed": int(seed),
        "evaluation_branch_fraction": float(evaluation_branch_frac),
        "branch_count": len(branches),
        "evaluation_branch_count": len(evaluation_branches),
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "omitted_identity_overlap_components": len(overlap),
        "omitted_rows": len(rows) - len(calibration) - len(evaluation),
        "identity_overlap_after_filter": 0,
    }


def valid_split(rows, calibration, evaluation, min_calibration=80, min_evaluation=30):
    cal_rows = [rows[i] for i in calibration]
    eval_rows = [rows[i] for i in evaluation]
    reasons = []
    if len(cal_rows) < min_calibration:
        reasons.append("too_few_calibration_rows")
    if len(eval_rows) < min_evaluation:
        reasons.append("too_few_evaluation_rows")
    if {row["operator_family"] for row in cal_rows} != set(FAMILY_ORDER):
        reasons.append("calibration_missing_operator_family")
    if len({row["operator_family"] for row in eval_rows}) < 2:
        reasons.append("evaluation_has_fewer_than_two_operator_families")
    if {int(row["hop"]) for row in cal_rows} != {1, 2, 3, 4, 5}:
        reasons.append("calibration_missing_hop")
    if {int(row["hop"]) for row in eval_rows} != {1, 2, 3, 4, 5}:
        reasons.append("evaluation_missing_hop")
    return not reasons, reasons


def arrays_from_rows(rows):
    def matrix(columns):
        return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)

    return {
        "pair_id": np.asarray([row["pair_id"] for row in rows]),
        "prior": matrix(("model_D", "model_S")),
        "graph": matrix(("graph_measurement",)).ravel(),
        "target": matrix(("target_D", "target_S")),
        "hop": matrix(("hop",)).ravel().astype(int),
        "raw_sources": matrix(RAW_SOURCE_NAMES),
        "family": np.asarray([row["operator_family"] for row in rows]),
    }


def affine_calibrate_graph(graph_calibration, target_calibration, graph_all):
    graph_calibration = np.asarray(graph_calibration, dtype=float)
    target_calibration = np.asarray(target_calibration, dtype=float)
    graph_all = np.asarray(graph_all, dtype=float)
    if np.std(graph_calibration) < 1e-12:
        slope, intercept = 0.0, float(np.mean(target_calibration))
    else:
        design = np.column_stack([graph_calibration, np.ones(len(graph_calibration))])
        (slope, intercept), *_ = np.linalg.lstsq(design, target_calibration, rcond=None)
    calibrated = np.clip(slope * graph_all + intercept, 0.0, 1.0)
    return calibrated, {"slope": float(slope), "intercept": float(intercept)}


def source_diagnostics(data):
    X, labels = data["raw_sources"], data["family"]
    correlations = []
    for i in range(X.shape[1]):
        row = []
        for j in range(X.shape[1]):
            if np.std(X[:, i]) < 1e-12 or np.std(X[:, j]) < 1e-12:
                row.append(None)
            else:
                row.append(float(np.corrcoef(X[:, i], X[:, j])[0, 1]))
        correlations.append(row)
    separability = {}
    class_means = {}
    for column, name in enumerate(RAW_SOURCE_NAMES):
        means = {
            family: float(np.mean(X[labels == family, column]))
            for family in FAMILY_ORDER
            if np.any(labels == family)
        }
        class_means[name] = means
        separability[name] = float(np.std(list(means.values()))) if len(means) > 1 else 0.0
    return {
        "source_names": list(RAW_SOURCE_NAMES),
        "correlation_matrix": correlations,
        "class_means": class_means,
        "separability": separability,
    }


def score_summary(score, pit):
    return {
        "n": score.n,
        "mean_nll": score.mean_nll,
        "mse": score.mse,
        "mahalanobis_per_dim": score.mahalanobis_per_dim,
        "squared_mahalanobis_q95": score.squared_mahalanobis_q95,
        "pit_channel_ks": pit.channel_ks.tolist(),
        "pit_mean_ks": float(np.mean(pit.channel_ks)),
        "coverage_levels": list(pit.coverage_levels),
        "central_interval_coverage": pit.central_interval_coverage.tolist(),
        "coverage_mean_absolute_error": float(np.mean(np.abs(pit.central_interval_error))),
    }


def fit_hop_product_parameters(prior, measurement, target, hop, calibration_indices):
    prior_error = target - prior
    measurement_error = measurement - target[:, 0]
    joint_error = np.column_stack([prior_error[:, 0], prior_error[:, 1], measurement_error])
    return fit_joint_sigma_of_hop(joint_error[calibration_indices], hop[calibration_indices])


def apply_hop_product_parameters(prior, measurement, hop, indices, params):
    means, covariances = [], []
    for index in indices:
        factor = chol_of_hop(params, float(hop[index]))
        covariance = factor @ factor.T
        P, R, C = covariance[:2, :2], covariance[2:, 2:], covariance[:2, 2:]
        mean, posterior_covariance = correlated_update(
            prior[index], P, np.asarray([measurement[index]]), R, C
        )
        means.append(mean)
        covariances.append(posterior_covariance)
    return np.asarray(means), np.asarray(covariances)


def categorical_metrics(probabilities, labels, boot=0, seed=0):
    relation_index = {family: index for index, family in enumerate(FAMILY_ORDER)}
    truth = np.asarray([relation_index[label] for label in labels])
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (prediction == truth).astype(float)
    log_loss = float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-9, 1.0)).mean())
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for index in range(10):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(mask):
            ece += float(np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(correct[mask])))
    margin = margin_conf(probabilities)
    if boot:
        aurc_point, low, high = aurc_boot(margin, correct, B=boot, seed=seed)
    else:
        aurc_point, low, high = aurc(margin, correct), None, None
    return {
        "n": len(labels),
        "accuracy": float(np.mean(correct)),
        "log_loss": log_loss,
        "ece_10_equal_width": ece,
        "aurc_margin": aurc_point,
        "aurc_ci_low": low,
        "aurc_ci_high": high,
    }


def bootstrap_gain(baseline, candidate, boot, seed=0):
    difference = np.asarray(baseline, dtype=float) - np.asarray(candidate, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(boot, dtype=float)
    for index in range(boot):
        sample = rng.integers(0, len(difference), len(difference))
        values[index] = float(np.mean(difference[sample]))
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "observed_mean_gain": float(np.mean(difference)),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_mean_gain": float(np.mean(values)),
        "n_boot": int(boot),
        "seed": int(seed),
        "n": len(difference),
        "method": "paired_row_resample",
    }


def evaluate_split(data, calibration, evaluation, shrinkage, jitter, boot=0):
    graph_calibrated, graph_fit = affine_calibrate_graph(
        data["graph"][calibration], data["target"][calibration, 0], data["graph"]
    )
    measurement = graph_calibrated[:, None]
    result = evaluate_product_kalman_holdout(
        data["prior"][calibration],
        measurement[calibration],
        data["target"][calibration],
        data["prior"][evaluation],
        measurement[evaluation],
        data["target"][evaluation],
        H=H,
        shrinkage=shrinkage,
        jitter=jitter,
        calibration_ids=data["pair_id"][calibration],
        evaluation_ids=data["pair_id"][evaluation],
    )
    constant_scores = {score.name: score for score in result.scores}
    constant_vectors = {vector.name: vector for vector in result.score_vectors}
    constant_pit = {diagnostic.name: diagnostic for diagnostic in result.pit_diagnostics}

    hop_params = fit_hop_product_parameters(
        data["prior"], graph_calibrated, data["target"], data["hop"], calibration
    )
    hop_eval_mean, hop_eval_covariance = apply_hop_product_parameters(
        data["prior"], graph_calibrated, data["hop"], evaluation, hop_params
    )
    hop_cal_mean, _hop_cal_covariance = apply_hop_product_parameters(
        data["prior"], graph_calibrated, data["hop"], calibration, hop_params
    )
    hop_vectors = score_gaussian_prediction_vectors_rowwise(
        "hop_product_kalman",
        data["target"][evaluation],
        hop_eval_mean,
        hop_eval_covariance,
        jitter=jitter,
    )
    hop_score = score_gaussian_predictions_rowwise(
        "hop_product_kalman",
        data["target"][evaluation],
        hop_eval_mean,
        hop_eval_covariance,
        jitter=jitter,
    )
    hop_pit = gaussian_marginal_pit_diagnostics(
        "hop_product_kalman",
        data["target"][evaluation],
        hop_eval_mean,
        hop_eval_covariance,
        jitter=jitter,
        channel_names=("D", "S"),
    )
    scores = {
        name: score_summary(constant_scores[name], constant_pit[name])
        for name in ("prior", "independent_kalman", "product_kalman")
    }
    scores["hop_product_kalman"] = score_summary(hop_score, hop_pit)

    constant_cal_mean = apply_product_kalman_calibration(
        result.calibration,
        data["prior"][calibration],
        measurement[calibration],
        jitter=jitter,
    ).mean
    categorical_sources = {
        "joint_baseline": (data["raw_sources"][calibration], data["raw_sources"][evaluation]),
        "joint_plus_constant": (
            np.column_stack([data["raw_sources"][calibration], constant_cal_mean]),
            np.column_stack([data["raw_sources"][evaluation], result.correlated_update.mean]),
        ),
        "joint_plus_hop": (
            np.column_stack([data["raw_sources"][calibration], hop_cal_mean]),
            np.column_stack([data["raw_sources"][evaluation], hop_eval_mean]),
        ),
    }
    categorical = {}
    probabilities = {}
    for name, (train_sources, eval_sources) in categorical_sources.items():
        posterior = JointPosterior(FAMILY_ORDER, train_sources.shape[1], hidden=0, seed=0).fit(
            train_sources,
            data["family"][calibration].tolist(),
        )
        probability = posterior.proba(eval_sources)
        probabilities[name] = probability
        categorical[name] = categorical_metrics(
            probability,
            data["family"][evaluation],
            boot=boot,
            seed=0,
        )

    gains = {}
    if boot:
        vectors = dict(constant_vectors)
        vectors["hop_product_kalman"] = hop_vectors
        for baseline in ("prior", "independent_kalman", "product_kalman"):
            for candidate in ("product_kalman", "hop_product_kalman"):
                if baseline == candidate:
                    continue
                key = f"{baseline}_to_{candidate}"
                gains[key] = bootstrap_gain(vectors[baseline].nll, vectors[candidate].nll, boot=boot)

    return {
        "graph_affine_calibration": graph_fit,
        "continuous": scores,
        "continuous_gains": gains,
        "categorical": categorical,
        "hop_sigma_parameters": hop_params.tolist(),
    }, {
        "calibration_indices": calibration,
        "evaluation_indices": evaluation,
        "evaluation_pair_ids": data["pair_id"][evaluation],
        "evaluation_target": data["target"][evaluation],
        "evaluation_prior": data["prior"][evaluation],
        "evaluation_measurement": measurement[evaluation],
        "constant_product_mean": result.correlated_update.mean,
        "constant_product_covariance": result.correlated_update.covariance,
        "hop_product_mean": hop_eval_mean,
        "hop_product_covariance": hop_eval_covariance,
        "family_labels": data["family"][evaluation],
        **{f"{name}_probabilities": value for name, value in probabilities.items()},
    }


def stability_summary(split_results):
    continuous = {}
    for baseline, candidate in (
        ("prior", "product_kalman"),
        ("independent_kalman", "product_kalman"),
        ("prior", "hop_product_kalman"),
        ("independent_kalman", "hop_product_kalman"),
        ("product_kalman", "hop_product_kalman"),
    ):
        gains = np.asarray([
            result["continuous"][baseline]["mean_nll"] - result["continuous"][candidate]["mean_nll"]
            for result in split_results
        ])
        continuous[f"{baseline}_to_{candidate}"] = {
            "mean_gain": float(np.mean(gains)),
            "split_standard_error": float(np.std(gains) / math.sqrt(len(gains))),
            "positive_splits": int(np.sum(gains > 0.0)),
            "split_count": len(gains),
        }
    categorical = {}
    for candidate in ("joint_plus_constant", "joint_plus_hop"):
        categorical[candidate] = {}
        for metric in ("log_loss", "ece_10_equal_width", "aurc_margin"):
            deltas = np.asarray([
                result["categorical"]["joint_baseline"][metric]
                - result["categorical"][candidate][metric]
                for result in split_results
            ])
            categorical[candidate][metric] = {
                "mean_improvement": float(np.mean(deltas)),
                "positive_splits": int(np.sum(deltas > 0.0)),
                "split_count": len(deltas),
            }
    return {"continuous": continuous, "categorical": categorical}


def promotion_decision(primary):
    candidate = primary["continuous"]["hop_product_kalman"]
    controls = [primary["continuous"][name] for name in ("prior", "independent_kalman")]
    reasons = []
    for baseline in ("prior", "independent_kalman"):
        gain = primary["continuous_gains"][f"{baseline}_to_hop_product_kalman"]
        if gain["ci_low"] <= 0.0:
            reasons.append(f"NLL interval versus {baseline} crosses zero")
    best_pit = min(control["pit_mean_ks"] for control in controls)
    if candidate["pit_mean_ks"] > best_pit + CALIBRATION_TOL["pit_ks"]:
        reasons.append("mean PIT KS exceeds calibration tolerance")
    best_coverage = min(control["coverage_mean_absolute_error"] for control in controls)
    if candidate["coverage_mean_absolute_error"] > best_coverage + CALIBRATION_TOL["coverage_error"]:
        reasons.append("central coverage error exceeds calibration tolerance")
    best_mahal = min(abs(control["mahalanobis_per_dim"] - 1.0) for control in controls)
    if abs(candidate["mahalanobis_per_dim"] - 1.0) > best_mahal + CALIBRATION_TOL["mahalanobis_distance"]:
        reasons.append("Mahalanobis calibration exceeds tolerance")
    baseline = primary["categorical"]["joint_baseline"]
    fused = primary["categorical"]["joint_plus_hop"]
    if fused["log_loss"] >= baseline["log_loss"]:
        reasons.append("fused JointPosterior log-loss does not improve")
    if fused["ece_10_equal_width"] >= baseline["ece_10_equal_width"]:
        reasons.append("fused JointPosterior ECE does not improve")
    if fused["aurc_ci_high"] is None or fused["aurc_ci_high"] >= baseline["aurc_margin"]:
        reasons.append("fused AURC interval is not below baseline point estimate")
    return {"promote": not reasons, "decision": "promote" if not reasons else "do_not_promote", "reasons": reasons}


def render_markdown(result):
    primary = result["primary_result"]
    lines = [
        f"# Public Product-Kalman holdout: {result['corpus']}",
        "",
        "Status: exploratory frozen-protocol evaluation.",
        "",
        f"Primary split seed: `{result['primary_seed']}`; calibration/evaluation rows: "
        f"`{result['primary_split']['calibration_rows']}/{result['primary_split']['evaluation_rows']}`.",
        "",
        "## Continuous evaluation",
        "",
        "| variant | NLL | MSE | Mahal/dim | q95 | mean PIT KS | coverage MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONTINUOUS_VARIANTS:
        score = primary["continuous"][name]
        lines.append(
            f"| `{name}` | {score['mean_nll']:+.4f} | {score['mse']:.4f} | "
            f"{score['mahalanobis_per_dim']:.3f} | {score['squared_mahalanobis_q95']:.3f} | "
            f"{score['pit_mean_ks']:.3f} | {score['coverage_mean_absolute_error']:.3f} |"
        )
    lines.extend(["", "Paired primary-split NLL gains (positive favors candidate):", ""])
    for name, gain in sorted(primary["continuous_gains"].items()):
        lines.append(f"- `{name}`: {gain['observed_mean_gain']:+.4f} [{gain['ci_low']:+.4f}, {gain['ci_high']:+.4f}]")
    lines.extend([
        "",
        "## JointPosterior evaluation",
        "",
        "| variant | accuracy | log-loss | ECE-10 | margin AURC (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("joint_baseline", "joint_plus_constant", "joint_plus_hop"):
        score = primary["categorical"][name]
        lines.append(
            f"| `{name}` | {score['accuracy']:.3f} | {score['log_loss']:.3f} | "
            f"{score['ece_10_equal_width']:.3f} | {score['aurc_margin']:.3f} "
            f"[{score['aurc_ci_low']:.3f}, {score['aurc_ci_high']:.3f}] |"
        )
    decision = result["promotion_decision"]
    lines.extend(["", "## Decision", "", f"**{decision['decision']}**"])
    for reason in decision["reasons"]:
        lines.append(f"- {reason}")
    lines.extend([
        "",
        "The targets come from one non-deterministic LLM judge. Split stability is secondary because seeds reuse",
        "the same 250-pair corpus; it is not an independent replication.",
        "",
    ])
    return "\n".join(lines)


def run(args):
    rows = load_feature_table(args.features)
    corpora = {row["corpus"] for row in rows}
    if len(corpora) != 1:
        raise PublicHoldoutError("feature table must contain one corpus")
    corpus = next(iter(corpora))
    data = arrays_from_rows(rows)
    split_records = []
    valid = []
    for seed in range(args.seeds):
        calibration, evaluation, manifest = strict_branch_identity_split(
            rows, seed, evaluation_branch_frac=args.evaluation_branch_frac
        )
        is_valid, reasons = valid_split(rows, calibration, evaluation)
        manifest["valid"] = is_valid
        manifest["invalid_reasons"] = reasons
        manifest["calibration_family_counts"] = dict(sorted(Counter(data["family"][calibration]).items()))
        manifest["evaluation_family_counts"] = dict(sorted(Counter(data["family"][evaluation]).items()))
        manifest["calibration_hop_counts"] = dict(sorted(Counter(data["hop"][calibration].tolist()).items()))
        manifest["evaluation_hop_counts"] = dict(sorted(Counter(data["hop"][evaluation].tolist()).items()))
        split_records.append(manifest)
        if is_valid:
            valid.append((seed, calibration, evaluation, manifest))
    if not valid:
        raise PublicHoldoutError("no split satisfies the frozen composition rules")

    split_results = []
    primary_arrays = None
    for position, (seed, calibration, evaluation, manifest) in enumerate(valid):
        print(
            f"{corpus}: split {position + 1}/{len(valid)} seed={seed} "
            f"cal={len(calibration)} eval={len(evaluation)}",
            flush=True,
        )
        result, arrays = evaluate_split(
            data,
            calibration,
            evaluation,
            shrinkage=args.shrinkage,
            jitter=args.jitter,
            boot=args.boot if position == 0 else 0,
        )
        result["seed"] = seed
        split_results.append(result)
        if position == 0:
            primary_arrays = arrays

    primary_seed, _calibration, _evaluation, primary_manifest = valid[0]
    output = {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus,
        "protocol": "PROTOCOL_product_kalman_public_holdout.md",
        "feature_table": str(args.features),
        "row_count": len(rows),
        "source_diagnostics": source_diagnostics(data),
        "split_settings": {
            "seeds": args.seeds,
            "evaluation_branch_fraction": args.evaluation_branch_frac,
            "minimum_calibration_rows": 80,
            "minimum_evaluation_rows": 30,
            "identity_closure": "endpoint ID joined to frozen canonical title",
        },
        "split_records": split_records,
        "valid_split_count": len(valid),
        "primary_seed": primary_seed,
        "primary_split": primary_manifest,
        "primary_result": split_results[0],
        "stability": stability_summary(split_results),
        "calibration_tolerances": CALIBRATION_TOL,
    }
    output["promotion_decision"] = promotion_decision(output["primary_result"])
    atomic_json(args.json_out, output)
    atomic_text(args.md_out, render_markdown(output))
    atomic_npz(args.npz_out, **primary_arrays)
    print(f"{corpus}: {output['promotion_decision']['decision']} -> {args.json_out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--evaluation-branch-frac", type=float, default=0.5)
    ap.add_argument("--shrinkage", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--json-out", required=True, type=Path)
    ap.add_argument("--md-out", required=True, type=Path)
    ap.add_argument("--npz-out", required=True, type=Path)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seeds < 1 or args.boot < 1:
        raise SystemExit("--seeds and --boot must be positive")
    if not 0.0 < args.evaluation_branch_frac < 1.0:
        raise SystemExit("--evaluation-branch-frac must be in (0, 1)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
