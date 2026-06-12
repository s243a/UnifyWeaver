#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 John William Creighton (s243a)
"""Compare path-length histograms with fits and depth-conditioned priors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distribution_cache_support_bounds import (  # noqa: E402
    SIMPLEWIKI_ROOT,
    load_targets_file,
    parse_int_list,
    parse_targets,
    safe_graph_name,
)
from tools.distribution_cache_support import (  # noqa: E402
    EXPONENT,
    FIXTURES,
    ROOT,
    exact_histogram,
    histogram_bytes,
    load_parent_edges_tsv,
    reachable_nodes_by_parent_distance,
)


DEFAULT_DEPTHS = [2, 4, 8, 16]
DEFAULT_PRUNE_THRESHOLDS = [0.01, 0.001, 0.0001]


def clamp_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0.0:
        if not weights:
            return []
        out = [0.0 for _ in weights]
        out[0] = 1.0
        return out
    return [weight / total for weight in weights]


def exact_excess_distribution(hist: dict[int, int]) -> tuple[list[float], int | None]:
    """Return P(L - L_min = k) and the shifted origin."""
    if not hist:
        return [], None
    origin = min(hist)
    width = max(hist) - origin
    weights = [float(hist.get(origin + k, 0)) for k in range(width + 1)]
    return normalize(weights), origin


def distribution_moments(probabilities: list[float]) -> tuple[float, float]:
    mean = sum(index * probability for index, probability in enumerate(probabilities))
    variance = sum(((index - mean) ** 2) * probability for index, probability in enumerate(probabilities))
    return mean, variance


def distribution_skewness(probabilities: list[float]) -> float | None:
    mean, variance = distribution_moments(probabilities)
    if variance <= 0.0:
        return None
    stddev = math.sqrt(variance)
    third = sum(((index - mean) ** 3) * probability for index, probability in enumerate(probabilities))
    return third / (stddev**3)


def ancestor_cone_size(node, parents) -> tuple[int, int]:
    """Count the parent-reachable cone for a single target node."""
    seen = set()
    stack = [node]
    edges = 0
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for parent in parents.get(current, []):
            edges += 1
            if parent not in seen:
                stack.append(parent)
    return len(seen), edges


def parametric_state_bytes_estimate(params: dict[str, object]) -> int:
    """Estimate stored family id, support, parameters, and error metadata."""
    scalar_count = 4
    scalar_count += sum(1 for value in params.values() if isinstance(value, (int, float)))
    scalar_count += 2
    return scalar_count * 8


def dense_sample_bytes(sample_points: int) -> int:
    return max(0, sample_points) * 8


def parse_float_list(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def tail_pruning_summary(probabilities: list[float], thresholds: list[float], origin: int = 0) -> dict[str, dict[str, float | int | None]]:
    summaries = {}
    total_weighted_power = sum(((origin + index + 1) ** (-EXPONENT)) * probability for index, probability in enumerate(probabilities))
    total_first_moment = sum((origin + index) * probability for index, probability in enumerate(probabilities))
    for threshold in thresholds:
        allowed_tail = max(0.0, threshold)
        dropped_mass = 0.0
        dropped_first_moment = 0.0
        dropped_weighted_power = 0.0
        dropped_bins = 0
        for index in range(len(probabilities) - 1, -1, -1):
            probability = probabilities[index]
            if dropped_bins >= len(probabilities) - 1:
                break
            if dropped_mass + probability > allowed_tail:
                break
            dropped_mass += probability
            dropped_first_moment += (origin + index) * probability
            dropped_weighted_power += ((origin + index + 1) ** (-EXPONENT)) * probability
            dropped_bins += 1
        kept_bins = len(probabilities) - dropped_bins
        summaries[str(threshold)] = {
            "kept_bins": kept_bins,
            "dropped_bins": dropped_bins,
            "dropped_mass": dropped_mass,
            "dropped_first_moment": dropped_first_moment,
            "relative_first_moment_error": 0.0 if total_first_moment == 0.0 else dropped_first_moment / total_first_moment,
            "dropped_weighted_power": dropped_weighted_power,
            "relative_weighted_power_error": 0.0 if total_weighted_power == 0.0 else dropped_weighted_power / total_weighted_power,
            "first_dropped_index": None if dropped_bins == 0 else origin + kept_bins,
        }
    return summaries


def binomial_pmf(trials: int, probability: float) -> list[float]:
    if trials <= 0:
        return [1.0]
    probability = clamp_probability(probability)
    if probability == 0.0:
        return [1.0] + [0.0 for _ in range(trials)]
    if probability == 1.0:
        return [0.0 for _ in range(trials)] + [1.0]

    q = 1.0 - probability
    values = [0.0 for _ in range(trials + 1)]
    values[0] = q**trials
    for k in range(trials):
        values[k + 1] = values[k] * (trials - k) / (k + 1) * probability / q
    return normalize(values)


def fitted_binomial_pmf(empirical: list[float]) -> tuple[list[float], dict[str, float | int]]:
    trials = max(0, len(empirical) - 1)
    mean, variance = distribution_moments(empirical)
    probability = 0.0 if trials == 0 else clamp_probability(mean / trials)
    return binomial_pmf(trials, probability), {
        "trials": trials,
        "probability": probability,
        "mean": mean,
        "variance": variance,
    }


def degenerate_pmf(index: int, bins: int) -> list[float]:
    bins = max(1, bins)
    index = min(max(0, index), bins - 1)
    out = [0.0 for _ in range(bins)]
    out[index] = 1.0
    return out


def gamma_midpoint_pmf(mean: float, variance: float, bins: int) -> tuple[list[float], dict[str, float | int | str]]:
    """Discretise a Gamma fit over integer excess bins by midpoint sampling."""
    bins = max(1, bins)
    if mean <= 0.0:
        return degenerate_pmf(0, bins), {
            "family": "degenerate",
            "shape": 0.0,
            "scale": 0.0,
            "mean": mean,
            "variance": variance,
        }
    if variance <= 1e-12:
        return degenerate_pmf(round(mean), bins), {
            "family": "degenerate",
            "shape": "inf",
            "scale": 0.0,
            "mean": mean,
            "variance": variance,
        }

    shape = (mean * mean) / variance
    scale = variance / mean
    log_norm = math.lgamma(shape) + shape * math.log(scale)
    weights = []
    for k in range(bins):
        x = k + 0.5
        log_density = (shape - 1.0) * math.log(x) - (x / scale) - log_norm
        weights.append(math.exp(log_density))
    return normalize(weights), {
        "family": "gamma_midpoint",
        "shape": shape,
        "scale": scale,
        "mean": mean,
        "variance": variance,
    }


def fitted_gamma_pmf(empirical: list[float]) -> tuple[list[float], dict[str, float | int | str]]:
    mean, variance = distribution_moments(empirical)
    return gamma_midpoint_pmf(mean, variance, len(empirical))


def convolve(left: list[float], right: list[float]) -> list[float]:
    if not left or not right:
        return []
    out = [0.0 for _ in range(len(left) + len(right) - 1)]
    for i, left_value in enumerate(left):
        if left_value == 0.0:
            continue
        for j, right_value in enumerate(right):
            if right_value:
                out[i + j] += left_value * right_value
    return normalize(out)


def nfold_convolution(base: list[float], depth: int) -> list[float]:
    if depth <= 0:
        return [1.0]
    out = [1.0]
    for _ in range(depth):
        out = convolve(out, base)
    return out


def size_biased_excess_pmf(parent_degrees: list[int]) -> list[float]:
    """Distribution of Y=p-1 seen after arriving through a parent edge."""
    weights: dict[int, float] = {}
    total_weight = 0.0
    for degree in parent_degrees:
        if degree <= 0:
            continue
        excess = degree - 1
        weights[excess] = weights.get(excess, 0.0) + degree
        total_weight += degree
    if total_weight <= 0.0:
        return [1.0]
    max_excess = max(weights)
    return [weights.get(excess, 0.0) / total_weight for excess in range(max_excess + 1)]


def pad_to(left: list[float], right: list[float]) -> tuple[list[float], list[float]]:
    size = max(len(left), len(right))
    return left + [0.0] * (size - len(left)), right + [0.0] * (size - len(right))


def l1_error(empirical: list[float], model: list[float]) -> float:
    empirical, model = pad_to(empirical, model)
    return sum(abs(a - b) for a, b in zip(empirical, model))


def max_cdf_error(empirical: list[float], model: list[float]) -> float:
    empirical, model = pad_to(empirical, model)
    empirical_total = 0.0
    model_total = 0.0
    worst = 0.0
    for empirical_value, model_value in zip(empirical, model):
        empirical_total += empirical_value
        model_total += model_value
        worst = max(worst, abs(empirical_total - model_total))
    return worst


def mean_absolute_error(empirical: list[float], model: list[float]) -> float:
    empirical, model = pad_to(empirical, model)
    if not empirical:
        return 0.0
    return sum(abs(a - b) for a, b in zip(empirical, model)) / len(empirical)


def effective_support_bins(probabilities: list[float], tail_epsilon: float) -> int:
    tail_epsilon = max(0.0, tail_epsilon)
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if 1.0 - cumulative <= tail_epsilon:
            return index + 1
    return len(probabilities)


def compare_models(empirical: list[float], model_builders) -> list[dict[str, object]]:
    records = []
    for model_name, builder in model_builders:
        started = time.perf_counter_ns()
        model, params = builder(empirical)
        build_time_ns = time.perf_counter_ns() - started
        model_mean, model_variance = distribution_moments(model)
        records.append({
            "model": model_name,
            "l1_error": l1_error(empirical, model),
            "max_cdf_error": max_cdf_error(empirical, model),
            "mean_absolute_error": mean_absolute_error(empirical, model),
            "model_mean_excess": model_mean,
            "model_variance_excess": model_variance,
            "model_support_bins": len(model),
            "build_time_ns": build_time_ns,
            "fit_params": params,
        })
    return records


def realized_model_builders():
    return [
        ("binomial_fit", fitted_binomial_pmf),
        ("shifted_gamma_fit", fitted_gamma_pmf),
    ]


def prior_model_builders(depth: int):
    def build_binomial(empirical: list[float]):
        mean, variance = distribution_moments(empirical)
        probability = 0.0 if depth <= 0 else clamp_probability(mean / depth)
        return binomial_pmf(depth, probability), {
            "trials": depth,
            "probability": probability,
            "mean": mean,
            "variance": variance,
        }

    def build_gamma(empirical: list[float]):
        mean, variance = distribution_moments(empirical)
        return gamma_midpoint_pmf(mean, variance, len(empirical))

    return [
        ("binomial_prior", build_binomial),
        ("shifted_gamma_prior", build_gamma),
    ]


def target_fit_records(
    graph_name,
    graph_label,
    parents,
    root,
    target,
    hist_memo,
    tail_epsilon,
    continuous_sample_points,
    prune_thresholds,
):
    started = time.perf_counter_ns()
    hist = exact_histogram(target, parents, root, hist_memo)
    exact_time_ns = time.perf_counter_ns() - started
    empirical, origin = exact_excess_distribution(hist)
    if not empirical or origin is None:
        return []

    mean, variance = distribution_moments(empirical)
    exact_bytes = histogram_bytes(hist)
    dense_sample_estimate = dense_sample_bytes(continuous_sample_points)
    cone_nodes, cone_edges = ancestor_cone_size(target, parents)
    records = []
    for model_record in compare_models(empirical, realized_model_builders()):
        parametric_bytes = parametric_state_bytes_estimate(model_record["fit_params"])
        records.append({
            "record_type": "distribution_fit",
            "distribution_role": "realized_fit",
            "graph": graph_name,
            "fixture": graph_label,
            "root": root,
            "target_node": target,
            "L_min": origin,
            "L_max": origin + len(empirical) - 1,
            "support_width": len(empirical) - 1,
            "support_bins": len(empirical),
            "effective_support_bins": effective_support_bins(empirical, tail_epsilon),
            "tail_pruning": tail_pruning_summary(empirical, prune_thresholds, origin),
            "path_count": sum(hist.values()),
            "mean_excess": mean,
            "variance_excess": variance,
            "skewness_excess": distribution_skewness(empirical),
            "exact_histogram_time_ns": exact_time_ns,
            "ancestor_cone_nodes": cone_nodes,
            "ancestor_cone_edges": cone_edges,
            "exact_histogram_bytes": exact_bytes,
            "parametric_state_bytes_estimate": parametric_bytes,
            "compression_ratio_estimate": 0.0 if parametric_bytes == 0 else exact_bytes / parametric_bytes,
            "continuous_sample_points": continuous_sample_points,
            "continuous_sample_bytes_estimate": dense_sample_estimate,
            "sample_storage_ratio_estimate": 0.0 if dense_sample_estimate == 0 else exact_bytes / dense_sample_estimate,
            "parent_degree": len(parents.get(target, [])),
            **model_record,
        })
    return records


def depth_prior_records(graph_name, graph_label, parent_degrees, depths, tail_epsilon, continuous_sample_points, prune_thresholds):
    base = size_biased_excess_pmf(parent_degrees)
    base_mean, base_variance = distribution_moments(base)
    records = []
    for depth in depths:
        started = time.perf_counter_ns()
        empirical = nfold_convolution(base, depth)
        exact_time_ns = time.perf_counter_ns() - started
        mean, variance = distribution_moments(empirical)
        dense_sample_estimate = dense_sample_bytes(continuous_sample_points)
        dense_prior_estimate = dense_sample_bytes(len(empirical))
        for model_record in compare_models(empirical, prior_model_builders(depth)):
            parametric_bytes = parametric_state_bytes_estimate(model_record["fit_params"])
            records.append({
                "record_type": "distribution_fit",
                "distribution_role": "depth_prior",
                "graph": graph_name,
                "fixture": graph_label,
                "root_distance_horizon": depth,
                "support_width": len(empirical) - 1,
                "support_bins": len(empirical),
                "effective_support_bins": effective_support_bins(empirical, tail_epsilon),
                "tail_pruning": tail_pruning_summary(empirical, prune_thresholds, 0),
                "mean_excess": mean,
                "variance_excess": variance,
                "skewness_excess": distribution_skewness(empirical),
                "base_mean_excess": base_mean,
                "base_variance_excess": base_variance,
                "base_support_bins": len(base),
                "exact_histogram_time_ns": exact_time_ns,
                "dense_prior_bytes_estimate": dense_prior_estimate,
                "parametric_state_bytes_estimate": parametric_bytes,
                "compression_ratio_estimate": 0.0 if parametric_bytes == 0 else dense_prior_estimate / parametric_bytes,
                "continuous_sample_points": continuous_sample_points,
                "continuous_sample_bytes_estimate": dense_sample_estimate,
                "sample_storage_ratio_estimate": 0.0 if dense_sample_estimate == 0 else dense_prior_estimate / dense_sample_estimate,
                **model_record,
            })
    return records


def run_graph_fit_comparison(
    graph_name,
    graph_label,
    parents,
    targets,
    root,
    depths=None,
    tail_epsilon=0.001,
    continuous_sample_points=100,
    prune_thresholds=None,
):
    prune_thresholds = prune_thresholds or DEFAULT_PRUNE_THRESHOLDS
    records = []
    hist_memo = {}
    for target in targets:
        try:
            records.extend(
                target_fit_records(
                    graph_name,
                    graph_label,
                    parents,
                    root,
                    target,
                    hist_memo,
                    tail_epsilon,
                    continuous_sample_points,
                    prune_thresholds,
                )
            )
        except ValueError as exc:
            records.append({
                "record_type": "distribution_fit_error",
                "graph": graph_name,
                "fixture": graph_label,
                "root": root,
                "target_node": target,
                "error": str(exc),
            })
    parent_degrees = [len(parents.get(target, [])) for target in targets]
    records.extend(
        depth_prior_records(
            graph_name,
            graph_label,
            parent_degrees,
            depths or DEFAULT_DEPTHS,
            tail_epsilon,
            continuous_sample_points,
            prune_thresholds,
        )
    )
    return records


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def summarize(records: list[dict[str, object]]) -> str:
    fit_records = [record for record in records if record.get("record_type") == "distribution_fit"]
    error_records = [record for record in records if record.get("record_type") == "distribution_fit_error"]
    realized_records = [record for record in fit_records if record.get("distribution_role") == "realized_fit"]
    prior_records = [record for record in fit_records if record.get("distribution_role") == "depth_prior"]
    lines = [
        "# Distribution Fit Comparison Summary",
        "",
        "| realized_targets | prior_depth_rows | model_rows | errors |",
        "|------------------|---------------------|------------|--------|",
        "| {targets} | {depth_rows} | {rows} | {errors} |".format(
            targets=len({record["target_node"] for record in realized_records}),
            depth_rows=len(prior_records),
            rows=len(fit_records),
            errors=len(error_records),
        ),
        "",
        "## Realized Histogram Fits",
        "",
        "| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns | mean_compression |",
        "|-------|------|---------|--------|--------|----------------|---------------|--------------------|------------------|",
    ]
    append_model_table(lines, realized_records)
    if realized_records:
        lines.extend([
            "",
            "## Realized Support By Root Distance",
            "",
            "| L_min | targets | mean_bins | p95_bins | max_bins | mean_effective_bins | mean_path_count | mean_parent_degree |",
            "|-------|---------|-----------|----------|----------|---------------------|-----------------|--------------------|",
        ])
        append_realized_support_table(lines, realized_records)
        lines.extend([
            "",
            "## Realized Tail-Pruned Support",
            "",
        ])
        append_tail_pruning_table(lines, realized_records)
    lines.extend([
        "",
        "## Depth-Conditioned Prior Distributions",
        "",
        "| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error | mean_build_ns | mean_exact_hist_ns | mean_compression |",
        "|-------|------|---------|--------|--------|----------------|---------------|--------------------|------------------|",
    ])
    append_model_table(lines, prior_records)

    if prior_records:
        lines.extend([
            "",
            "## Prior Support By Depth",
            "",
            "| depth | rows | mean_bins | mean_effective_bins | max_effective_bins | mean_excess |",
            "|-------|------|-----------|---------------------|--------------------|-------------|",
        ])
        by_depth = {}
        for record in prior_records:
            by_depth.setdefault(record["root_distance_horizon"], []).append(record)
        for depth in sorted(by_depth):
            rows = by_depth[depth]
            lines.append(
                "| {depth} | {rows} | {mean_bins:.3f} | {mean_eff:.3f} | {max_eff} | {mean_excess:.6f} |".format(
                    depth=depth,
                    rows=len(rows),
                    mean_bins=statistics.mean(int(row["support_bins"]) for row in rows),
                    mean_eff=statistics.mean(int(row["effective_support_bins"]) for row in rows),
                    max_eff=max(int(row["effective_support_bins"]) for row in rows),
                    mean_excess=statistics.mean(float(row["mean_excess"]) for row in rows),
                )
            )
        lines.extend([
            "",
            "## Prior Tail-Pruned Support",
            "",
        ])
        append_tail_pruning_table(lines, prior_records)

    if error_records:
        lines.extend(["", "## Errors", ""])
        for record in error_records[:10]:
            lines.append(f"- {record['target_node']}: {record['error']}")
    return "\n".join(lines) + "\n"


def unique_distribution_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {}
    for record in rows:
        if record.get("distribution_role") == "realized_fit":
            key = (record.get("distribution_role"), record.get("fixture"), record.get("target_node"))
        else:
            key = (record.get("distribution_role"), record.get("fixture"), record.get("root_distance_horizon"))
        if key not in by_key:
            by_key[key] = record
    return list(by_key.values())


def append_tail_pruning_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    unique_rows = unique_distribution_records(rows)
    thresholds = sorted({float(threshold) for row in unique_rows for threshold in row.get("tail_pruning", {})})
    lines.extend([
        "| tail_threshold | distributions | mean_original_bins | mean_kept_bins | mean_dropped_bins | mean_dropped_mass | mean_weighted_power_error |",
        "|----------------|---------------|--------------------|----------------|-------------------|-------------------|---------------------------|",
    ])
    for threshold in thresholds:
        key = str(threshold)
        threshold_rows = [row for row in unique_rows if key in row.get("tail_pruning", {})]
        if not threshold_rows:
            continue
        pruning_rows = [row["tail_pruning"][key] for row in threshold_rows]
        lines.append(
            "| {threshold:g} | {rows} | {mean_original:.3f} | {mean_kept:.3f} | {mean_dropped:.3f} | {mean_mass:.6f} | {mean_weighted:.6f} |".format(
                threshold=threshold,
                rows=len(threshold_rows),
                mean_original=statistics.mean(int(row["support_bins"]) for row in threshold_rows),
                mean_kept=statistics.mean(int(row["kept_bins"]) for row in pruning_rows),
                mean_dropped=statistics.mean(int(row["dropped_bins"]) for row in pruning_rows),
                mean_mass=statistics.mean(float(row["dropped_mass"]) for row in pruning_rows),
                mean_weighted=statistics.mean(float(row["relative_weighted_power_error"]) for row in pruning_rows),
            )
        )


def unique_realized_target_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_target = {}
    for record in rows:
        target = record.get("target_node")
        if target is not None and target not in by_target:
            by_target[target] = record
    return list(by_target.values())


def append_realized_support_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    by_distance = {}
    for record in unique_realized_target_records(rows):
        l_min = record.get("L_min")
        if l_min is not None:
            by_distance.setdefault(l_min, []).append(record)
    for l_min in sorted(by_distance):
        distance_rows = by_distance[l_min]
        support_bins = [int(row["support_bins"]) for row in distance_rows]
        effective_bins = [int(row["effective_support_bins"]) for row in distance_rows]
        path_counts = [int(row["path_count"]) for row in distance_rows]
        parent_degrees = [int(row["parent_degree"]) for row in distance_rows]
        lines.append(
            "| {l_min} | {targets} | {mean_bins:.3f} | {p95_bins:.3f} | {max_bins} | {mean_effective:.3f} | {mean_paths:.3f} | {mean_parent:.6f} |".format(
                l_min=l_min,
                targets=len(distance_rows),
                mean_bins=statistics.mean(support_bins) if support_bins else 0.0,
                p95_bins=percentile([float(value) for value in support_bins], 95),
                max_bins=max(support_bins, default=0),
                mean_effective=statistics.mean(effective_bins) if effective_bins else 0.0,
                mean_paths=statistics.mean(path_counts) if path_counts else 0.0,
                mean_parent=statistics.mean(parent_degrees) if parent_degrees else 0.0,
            )
        )


def append_model_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    by_model = {}
    for record in rows:
        by_model.setdefault(record["model"], []).append(record)
    for model in sorted(by_model):
        model_rows = by_model[model]
        l1_values = [float(row["l1_error"]) for row in model_rows]
        cdf_values = [float(row["max_cdf_error"]) for row in model_rows]
        build_times = [int(row["build_time_ns"]) for row in model_rows]
        exact_times = [int(row["exact_histogram_time_ns"]) for row in model_rows]
        lines.append(
            "| {model} | {rows} | {mean_l1:.6f} | {p95_l1:.6f} | {max_l1:.6f} | {mean_cdf:.6f} | {mean_build:.1f} | {mean_exact:.1f} | {mean_compression:.3f} |".format(
                model=model,
                rows=len(model_rows),
                mean_l1=statistics.mean(l1_values) if l1_values else 0.0,
                p95_l1=percentile(l1_values, 95),
                max_l1=max(l1_values, default=0.0),
                mean_cdf=statistics.mean(cdf_values) if cdf_values else 0.0,
                mean_build=statistics.mean(build_times) if build_times else 0.0,
                mean_exact=statistics.mean(exact_times) if exact_times else 0.0,
                mean_compression=statistics.mean(float(row.get("compression_ratio_estimate", 0.0)) for row in model_rows),
            )
        )


def select_file_targets(parents, root, explicit_targets, targets_file, max_target_depth, target_limit):
    if explicit_targets:
        targets = parse_targets(explicit_targets)
    elif targets_file:
        targets = load_targets_file(targets_file)
    else:
        targets = reachable_nodes_by_parent_distance(parents, root, max_target_depth)
    if target_limit is not None:
        targets = targets[:target_limit]
    if not targets:
        raise SystemExit("no distribution-fit targets selected")
    return targets


def write_outputs(records, summary, output_dir, graph_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = safe_graph_name(graph_name)
    jsonl_path = output_dir / f"distribution_fit_comparison_{safe_name}_{timestamp}.jsonl"
    summary_path = output_dir / f"distribution_fit_comparison_summary_{safe_name}_{timestamp}.md"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path.write_text(summary, encoding="utf-8")
    return jsonl_path, summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="all", help="Comma-separated fixture names or all. Ignored when --edge-file is set.")
    parser.add_argument("--edge-file", type=Path, help="Optional TSV edge list with child<TAB>parent rows.")
    parser.add_argument("--graph-name", help="Graph label used in records and output filenames.")
    parser.add_argument("--root", help="Root node. Defaults to R for fixtures and Category:Articles for edge files.")
    parser.add_argument("--targets", help="Comma-separated target nodes for edge-file mode.")
    parser.add_argument("--targets-file", type=Path, help="Optional newline-delimited target list for edge-file mode.")
    parser.add_argument("--max-target-depth", type=int, help="Select reachable edge-file targets within this parent distance from root.")
    parser.add_argument("--target-limit", type=int, help="Limit selected edge-file targets after sorting/filtering.")
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)), help="Depth horizons for prior distributions.")
    parser.add_argument("--tail-epsilon", type=float, default=0.001, help="Tail mass allowed outside effective support.")
    parser.add_argument(
        "--continuous-sample-points",
        type=int,
        default=100,
        help="Point count for estimating sampled-continuous representation cost.",
    )
    parser.add_argument(
        "--prune-thresholds",
        default=",".join(map(str, DEFAULT_PRUNE_THRESHOLDS)),
        help="Comma-separated tail-mass thresholds for suffix-pruning estimates.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional directory for JSONL and markdown output.")
    args = parser.parse_args(argv)

    depths = parse_int_list(args.depths)
    prune_thresholds = parse_float_list(args.prune_thresholds)
    records = []
    if args.edge_file:
        root = args.root or SIMPLEWIKI_ROOT
        parents = load_parent_edges_tsv(args.edge_file)
        graph_name = args.graph_name or args.edge_file.stem
        targets = select_file_targets(
            parents,
            root,
            args.targets,
            args.targets_file,
            args.max_target_depth,
            args.target_limit,
        )
        records.extend(
            run_graph_fit_comparison(
                graph_name,
                graph_name,
                parents,
                targets,
                root,
                depths,
                args.tail_epsilon,
                args.continuous_sample_points,
                prune_thresholds,
            )
        )
    else:
        root = args.root or ROOT
        if root != ROOT:
            raise SystemExit("--root is only supported with --edge-file; fixtures use root R")
        fixture_names = sorted(FIXTURES) if args.fixtures == "all" else [name.strip() for name in args.fixtures.split(",")]
        graph_name = args.graph_name or "tiny_fixture"
        for fixture_name in fixture_names:
            if fixture_name not in FIXTURES:
                raise SystemExit(f"unknown fixture: {fixture_name}")
            fixture = FIXTURES[fixture_name]
            records.extend(
                run_graph_fit_comparison(
                    graph_name,
                    fixture_name,
                    fixture["parents"],
                    fixture["targets"],
                    root,
                    depths,
                    args.tail_epsilon,
                    args.continuous_sample_points,
                    prune_thresholds,
                )
            )

    summary = summarize(records)
    print(summary, end="")
    if args.output_dir:
        jsonl_path, summary_path = write_outputs(records, summary, args.output_dir, graph_name)
        print(f"\nwrote {jsonl_path}")
        print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
