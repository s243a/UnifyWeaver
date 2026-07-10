#!/usr/bin/env python3
"""Evaluate hop-conditioned covariance as a continuous selective-risk gate."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from run_product_kalman_public_holdout import (
    affine_calibrate_graph,
    apply_hop_product_parameters,
    arrays_from_rows,
    atomic_json,
    atomic_npz,
    atomic_text,
    fit_hop_product_parameters,
    load_feature_table,
    strict_branch_identity_split,
    valid_split,
)


SCHEMA_VERSION = 1
COVERAGES = (0.25, 0.50, 0.75, 1.00)


class SelectiveRiskError(ValueError):
    pass


def _vectors(predicted_risk, realized_loss):
    risk = np.asarray(predicted_risk, dtype=float)
    loss = np.asarray(realized_loss, dtype=float)
    if risk.ndim != 1 or loss.ndim != 1 or risk.shape != loss.shape or not len(risk):
        raise SelectiveRiskError("predicted risk and realized loss must be non-empty matching vectors")
    if not np.all(np.isfinite(risk)) or not np.all(np.isfinite(loss)):
        raise SelectiveRiskError("predicted risk and realized loss must be finite")
    if np.any(risk < 0.0) or np.any(loss < 0.0):
        raise SelectiveRiskError("predicted risk and realized loss must be non-negative")
    return risk, loss


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x, y):
    x, y = _vectors(x, y)
    x_rank, y_rank = average_ranks(x), average_ranks(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def tie_averaged_risk_curve(predicted_risk, realized_loss):
    """Expected prefix risk under a uniform order within equal-risk blocks."""
    risk, loss = _vectors(predicted_risk, realized_loss)
    order = np.argsort(risk, kind="mergesort")
    sorted_risk, sorted_loss = risk[order], loss[order]
    curve = np.empty(len(loss), dtype=float)
    complete_count = 0
    complete_loss = 0.0
    start = 0
    while start < len(loss):
        end = start + 1
        while end < len(loss) and sorted_risk[end] == sorted_risk[start]:
            end += 1
        block_mean = float(np.mean(sorted_loss[start:end]))
        for selected in range(1, end - start + 1):
            k = complete_count + selected
            curve[k - 1] = (complete_loss + selected * block_mean) / k
        complete_count = end
        complete_loss += float(np.sum(sorted_loss[start:end]))
        start = end
    return curve


def selective_metrics(predicted_risk, realized_loss):
    risk, loss = _vectors(predicted_risk, realized_loss)
    curve = tie_averaged_risk_curve(risk, loss)
    full_risk = float(np.mean(loss))
    if full_risk <= 0.0:
        raise SelectiveRiskError("full-coverage loss must be positive")
    return {
        "n": len(loss),
        "spearman": spearman_correlation(risk, loss),
        "aurc": float(np.mean(curve)),
        "normalized_aurc": float(np.mean(curve) / full_risk),
        "full_coverage_risk": full_risk,
        "coverage_risk": {
            f"{coverage:.2f}": float(curve[math.ceil(len(curve) * coverage) - 1])
            for coverage in COVERAGES
        },
        "unique_risk_values": int(len(np.unique(risk))),
    }


def permutation_test(predicted_risk, realized_loss, permutations=1000, seed=0):
    risk, loss = _vectors(predicted_risk, realized_loss)
    observed = selective_metrics(risk, loss)
    rng = np.random.default_rng(seed)
    rho_null = np.empty(permutations, dtype=float)
    aurc_null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        permuted = rng.permutation(risk)
        metrics = selective_metrics(permuted, loss)
        rho_null[index] = metrics["spearman"]
        aurc_null[index] = metrics["aurc"]
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "spearman_p_one_sided": float((1 + np.sum(rho_null >= observed["spearman"])) / (permutations + 1)),
        "aurc_p_one_sided": float((1 + np.sum(aurc_null <= observed["aurc"])) / (permutations + 1)),
        "spearman_null_mean": float(np.mean(rho_null)),
        "aurc_null_mean": float(np.mean(aurc_null)),
        "aurc_null_q025": float(np.quantile(aurc_null, 0.025)),
        "aurc_null_q975": float(np.quantile(aurc_null, 0.975)),
    }


def bootstrap_metrics(predicted_risk, realized_loss, boot=1000, seed=0):
    risk, loss = _vectors(predicted_risk, realized_loss)
    rng = np.random.default_rng(seed)
    rho_values = []
    normalized_aurc = np.empty(boot, dtype=float)
    for index in range(boot):
        sample = rng.integers(0, len(risk), len(risk))
        metrics = selective_metrics(risk[sample], loss[sample])
        if np.isfinite(metrics["spearman"]):
            rho_values.append(metrics["spearman"])
        normalized_aurc[index] = metrics["normalized_aurc"]
    if not rho_values:
        raise SelectiveRiskError("all bootstrap Spearman samples were degenerate")
    return {
        "boot": int(boot),
        "seed": int(seed),
        "degenerate_spearman_samples": int(boot - len(rho_values)),
        "spearman_ci_low": float(np.quantile(rho_values, 0.025)),
        "spearman_ci_high": float(np.quantile(rho_values, 0.975)),
        "normalized_aurc_ci_low": float(np.quantile(normalized_aurc, 0.025)),
        "normalized_aurc_ci_high": float(np.quantile(normalized_aurc, 0.975)),
    }


def hop_calibration(hop, predicted_risk, realized_loss):
    return {
        str(level): {
            "n": int(np.sum(hop == level)),
            "mean_predicted_risk": float(np.mean(predicted_risk[hop == level])),
            "mean_realized_loss": float(np.mean(realized_loss[hop == level])),
        }
        for level in sorted(np.unique(hop))
    }


def evaluate_split(data, calibration, evaluation, permutations=0, boot=0):
    graph, graph_fit = affine_calibrate_graph(
        data["graph"][calibration], data["target"][calibration, 0], data["graph"]
    )
    params = fit_hop_product_parameters(
        data["prior"], graph, data["target"], data["hop"], calibration
    )
    means, covariances = apply_hop_product_parameters(
        data["prior"], graph, data["hop"], evaluation, params
    )
    target = data["target"][evaluation]
    loss = np.sum(np.square(target - means), axis=1)
    trace_risk = np.trace(covariances, axis1=1, axis2=2)
    signs, logdet_risk = np.linalg.slogdet(covariances)
    if np.any(signs <= 0.0):
        raise SelectiveRiskError("hop posterior covariance must be positive definite")
    # Shift log determinant to satisfy the generic non-negative score contract; rankings are unchanged.
    logdet_shifted = logdet_risk - np.min(logdet_risk)
    result = {
        "graph_affine_calibration": graph_fit,
        "trace": selective_metrics(trace_risk, loss),
        "logdet_secondary": selective_metrics(logdet_shifted, loss),
        "hop_calibration": hop_calibration(data["hop"][evaluation], trace_risk, loss),
        "hop_sigma_parameters": params.tolist(),
    }
    if permutations:
        result["permutation"] = permutation_test(trace_risk, loss, permutations=permutations, seed=0)
    if boot:
        result["bootstrap"] = bootstrap_metrics(trace_risk, loss, boot=boot, seed=0)
    arrays = {
        "evaluation_indices": evaluation,
        "evaluation_pair_ids": data["pair_id"][evaluation],
        "hop": data["hop"][evaluation],
        "target": target,
        "posterior_mean": means,
        "posterior_covariance": covariances,
        "predicted_trace_risk": trace_risk,
        "predicted_logdet_risk": logdet_risk,
        "realized_squared_error": loss,
    }
    return result, arrays


def stability_summary(results):
    rho = np.asarray([result["trace"]["spearman"] for result in results])
    aurc = np.asarray([result["trace"]["normalized_aurc"] for result in results])
    return {
        "valid_split_count": len(results),
        "positive_spearman_splits": int(np.sum(rho > 0.0)),
        "positive_spearman_fraction": float(np.mean(rho > 0.0)),
        "normalized_aurc_below_one_splits": int(np.sum(aurc < 1.0)),
        "normalized_aurc_below_one_fraction": float(np.mean(aurc < 1.0)),
        "mean_spearman": float(np.mean(rho)),
        "mean_normalized_aurc": float(np.mean(aurc)),
    }


def decision(primary, stability):
    reasons = []
    if not primary["trace"]["spearman"] > 0.0 or primary["permutation"]["spearman_p_one_sided"] >= 0.01:
        reasons.append("primary trace-risk Spearman does not pass the one-sided permutation rule")
    if primary["permutation"]["aurc_p_one_sided"] >= 0.01:
        reasons.append("primary trace-risk AURC does not pass the one-sided permutation rule")
    if primary["bootstrap"]["normalized_aurc_ci_high"] >= 1.0:
        reasons.append("normalized AURC bootstrap upper bound is not below one")
    if stability["positive_spearman_fraction"] < 0.75:
        reasons.append("fewer than 75% of valid splits have positive Spearman")
    if stability["normalized_aurc_below_one_fraction"] < 0.75:
        reasons.append("fewer than 75% of valid splits have normalized AURC below one")
    return {
        "eligible": not reasons,
        "decision": "eligible_continuous_gate" if not reasons else "do_not_use_as_gate",
        "reasons": reasons,
    }


def render_markdown(output):
    primary = output["primary_result"]
    trace, secondary = primary["trace"], primary["logdet_secondary"]
    permutation, bootstrap = primary["permutation"], primary["bootstrap"]
    lines = [
        f"# Product-Kalman continuous selective risk: {output['corpus']}",
        "",
        "Status: exploratory frozen-protocol evaluation.",
        "",
        f"Primary seed `{output['primary_seed']}`; `{output['valid_split_count']}` valid splits.",
        "",
        "## Primary trace-covariance gate",
        "",
        f"- Spearman: `{trace['spearman']:+.4f}`; permutation `p={permutation['spearman_p_one_sided']:.6f}`; "
        f"bootstrap 95% CI `[{bootstrap['spearman_ci_low']:+.4f}, {bootstrap['spearman_ci_high']:+.4f}]`.",
        f"- Normalized AURC: `{trace['normalized_aurc']:.4f}`; permutation "
        f"`p={permutation['aurc_p_one_sided']:.6f}`; bootstrap 95% CI "
        f"`[{bootstrap['normalized_aurc_ci_low']:.4f}, {bootstrap['normalized_aurc_ci_high']:.4f}]`.",
        f"- Secondary log-determinant normalized AURC: `{secondary['normalized_aurc']:.4f}`.",
        "",
        "| coverage | selective squared error |",
        "|---:|---:|",
    ]
    for coverage, risk in trace["coverage_risk"].items():
        lines.append(f"| {float(coverage):.0%} | {risk:.5f} |")
    stability = output["stability"]
    lines.extend([
        "",
        "## Split stability",
        "",
        f"- Positive Spearman: `{stability['positive_spearman_splits']}/{stability['valid_split_count']}`.",
        f"- Normalized AURC below one: `{stability['normalized_aurc_below_one_splits']}/"
        f"{stability['valid_split_count']}`.",
        "",
        "## Decision",
        "",
        f"**{output['decision']['decision']}**",
    ])
    for reason in output["decision"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def run(args):
    rows = load_feature_table(args.features)
    corpora = {row["corpus"] for row in rows}
    if len(corpora) != 1:
        raise SelectiveRiskError("feature table must contain one corpus")
    corpus = next(iter(corpora))
    data = arrays_from_rows(rows)
    split_records, valid = [], []
    for seed in range(args.seeds):
        calibration, evaluation, manifest = strict_branch_identity_split(rows, seed)
        is_valid, reasons = valid_split(rows, calibration, evaluation)
        manifest["valid"] = is_valid
        manifest["invalid_reasons"] = reasons
        manifest["calibration_hop_counts"] = dict(sorted(Counter(data["hop"][calibration].tolist()).items()))
        manifest["evaluation_hop_counts"] = dict(sorted(Counter(data["hop"][evaluation].tolist()).items()))
        split_records.append(manifest)
        if is_valid:
            valid.append((seed, calibration, evaluation, manifest))
    if not valid:
        raise SelectiveRiskError("no split satisfies the frozen composition rules")

    results, primary_arrays = [], None
    for position, (seed, calibration, evaluation, _manifest) in enumerate(valid):
        print(f"{corpus}: split {position + 1}/{len(valid)} seed={seed}", flush=True)
        result, arrays = evaluate_split(
            data,
            calibration,
            evaluation,
            permutations=args.permutations if position == 0 else 0,
            boot=args.boot if position == 0 else 0,
        )
        result["seed"] = seed
        results.append(result)
        if position == 0:
            primary_arrays = arrays

    stability = stability_summary(results)
    primary_seed, _calibration, _evaluation, primary_manifest = valid[0]
    output = {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus,
        "protocol": "PROTOCOL_product_kalman_continuous_selective_risk.md",
        "feature_table": str(args.features),
        "row_count": len(rows),
        "primary_seed": primary_seed,
        "primary_split": primary_manifest,
        "valid_split_count": len(valid),
        "split_records": split_records,
        "primary_result": results[0],
        "stability": stability,
    }
    output["decision"] = decision(output["primary_result"], stability)
    atomic_json(args.json_out, output)
    atomic_text(args.md_out, render_markdown(output))
    atomic_npz(args.npz_out, **primary_arrays)
    print(f"{corpus}: {output['decision']['decision']} -> {args.json_out}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.seeds, args.permutations, args.boot) < 1:
        raise SystemExit("--seeds, --permutations, and --boot must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
