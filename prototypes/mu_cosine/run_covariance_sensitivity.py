#!/usr/bin/env python3
"""Descriptive legacy-v1 covariance-misspecification and posterior-sensitivity runner.

This is the executable companion to ``DESIGN_covariance_sensitivity.md``.  It deliberately refits every
outcome-dependent object inside each node-disjoint partition.  The primary graph/semantic family and the
secondary neutral-centered mu family are selected and null-calibrated separately.

The v1 selector failed family-wise synthetic controls.  Multi-seed execution requires an explicit descriptive
override, every aggregate inferential gate is disabled, and real-data full-procedure v2 calibration is not
implemented here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from covariance_sensitivity import (
    build_correlation_path,
    centered_linear_kernel,
    direct_covariance_blend,
    directional_posterior_sensitivity,
    materialize_channel_shrunk,
    mean_marginal_symmetric_kl,
    select_nested_candidate,
    whitened_covariance_error,
)
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from fine_tune_channel_heads import CAMPAIGN_E5_100K, load_campaign_datasets, load_expanded
from mu_posterior import JointPosterior
from node_disjoint_eval import format_split_diagnostics, node_disjoint_pair_split
from run_cheap_judge_joint_posterior import CLASSES, calibrate_sources, gaussian_bridge_proba, load_decision_targets
from run_product_kalman_realdata import DATASETS
from run_structured_residual_covariance import (
    configure_artifact_repo,
    decision_metrics,
    file_provenance,
    materialize_corpus,
    state_metrics,
    train_standardize,
)
from run_sym_channel_fusion import H4
from structured_residual_covariance import (
    condition_item_batch,
    conditional_residuals,
    fit_block_model,
    fit_lmc_model,
    gaussian_joint_nll,
    median_rbf_bandwidth,
    rbf_kernel,
    select_kernel_ridge_mean,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAMPAIGN = "/tmp/mu_data/campaign_scored.tsv"
DEFAULT_LUNA = "/tmp/mu_data/campaign_scored_luna.tsv"
ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
MULTIPLIERS = (0.5, 1.0, 2.0)
BETAS = (0.0, 0.5, 1.0)
PRIMARY = "semantic_graph_rbf"
SECONDARY = "semantic_centered_mu_linear"


@dataclass
class FitContext:
    fit: np.ndarray
    evaluate: np.ndarray
    calibrated: object
    conditional_design: np.ndarray
    q: np.ndarray
    centered_evaluate: np.ndarray
    evaluate_mean: np.ndarray
    block_model: object
    semantic: np.ndarray
    graph: np.ndarray
    mu_features: np.ndarray
    semantic_length: float
    graph_length: float
    regional: object
    graph_mean: np.ndarray
    graph_scale: np.ndarray


@dataclass
class Endpoint:
    family: str
    semantic_multiplier: float
    graph_multiplier: float
    beta: float
    model: object
    path: object
    structured_covariance: np.ndarray


def _jsonable(value):
    """Convert NumPy scalars/arrays recursively for stable JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _node_split(materialized, indices, seed, held_fraction, args):
    """Node-disjoint split of an induced row subset, mapped back to global indices."""
    indices = np.asarray(indices, dtype=int)
    split = node_disjoint_pair_split(
        [materialized["pairs"][i] for i in indices],
        seed,
        held_node_fraction=held_fraction,
        strata=materialized["tags"][indices],
        candidates=args.split_candidates,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    return split, indices[split.train], indices[split.held], indices[split.cross]


def _canonical_mean_kernels(semantic, graph, fit, evaluate, sem_length, graph_length):
    train = {
        "semantic": rbf_kernel(semantic[fit], length_scale=sem_length),
        "graph_feature": rbf_kernel(graph[fit], length_scale=graph_length),
    }
    train["equal_mixture"] = 0.5 * (train["semantic"] + train["graph_feature"])
    cross = {
        "semantic": rbf_kernel(semantic[evaluate], semantic[fit], length_scale=sem_length),
        "graph_feature": rbf_kernel(graph[evaluate], graph[fit], length_scale=graph_length),
    }
    cross["equal_mixture"] = 0.5 * (cross["semantic"] + cross["graph_feature"])
    return train, cross


def fit_context(materialized, fit, evaluate, args, *, fixed_regional=None):
    """Refit calibration, conditional residuals, mean, bandwidths, and block covariance."""
    fit = np.asarray(fit, dtype=int)
    evaluate = np.asarray(evaluate, dtype=int)
    if len(fit) < 2 or not len(evaluate):
        raise ValueError("fit/evaluate partitions must be non-empty and fit needs two rows")
    calibrated = calibrate_sources(
        materialized["prior"],
        materialized["graph_d_raw"],
        materialized["graph_s_features"],
        materialized["luna"],
        materialized["y_ds"],
        fit,
    )
    observed_measurement_state = materialized["y_ds"][:, [0, 1, 0, 1]]
    prior_error = materialized["y_ds"] - calibrated.prior
    measurement_error = calibrated.meas - observed_measurement_state
    conditional_design, q = conditional_residuals(
        prior_error, measurement_error, calibrated.P0, calibrated.C_pm, H4
    )
    graph, graph_mean, graph_scale = train_standardize(materialized["graph_item_raw"], fit)
    semantic = materialized["semantic"]
    sem_length = median_rbf_bandwidth(semantic[fit])
    graph_length = median_rbf_bandwidth(graph[fit])
    mean_train, mean_cross = _canonical_mean_kernels(
        semantic, graph, fit, evaluate, sem_length, graph_length
    )
    if fixed_regional is None:
        regional = select_kernel_ridge_mean(q[fit], mean_train, ridge_grid=args.ridge_grid)
    else:
        kernel_name, ridge = fixed_regional
        regional = select_kernel_ridge_mean(
            q[fit], {kernel_name: mean_train[kernel_name]}, ridge_grid=[ridge]
        )
    evaluate_mean = regional.predict(mean_cross[regional.kernel_name])
    block = fit_block_model(regional.loo_residuals, shrinkage=args.shrinkage)
    mu_features = calibrated.X - 0.5
    return FitContext(
        fit,
        evaluate,
        calibrated,
        conditional_design,
        q,
        q[evaluate] - evaluate_mean,
        evaluate_mean,
        block,
        semantic,
        graph,
        mu_features,
        sem_length,
        graph_length,
        regional,
        graph_mean,
        graph_scale,
    )


def _family_scale_grid(family):
    if family == PRIMARY:
        return [(semantic, graph) for semantic in MULTIPLIERS for graph in MULTIPLIERS]
    if family == SECONDARY:
        return [(semantic, 1.0) for semantic in MULTIPLIERS]
    raise ValueError(f"unknown geometry family: {family}")


def fit_endpoint(
    context,
    family,
    semantic_multiplier,
    graph_multiplier,
    beta,
    args,
    seed,
    *,
    fitted_model=None,
):
    """Fit one structured endpoint on context.fit and materialize it on context.evaluate."""
    sem_scale = context.semantic_length * float(semantic_multiplier)
    sem_train = rbf_kernel(context.semantic[context.fit], length_scale=sem_scale)
    sem_evaluate = rbf_kernel(context.semantic[context.evaluate], length_scale=sem_scale)
    if family == PRIMARY:
        graph_scale = context.graph_length * float(graph_multiplier)
        second_train = rbf_kernel(context.graph[context.fit], length_scale=graph_scale)
        second_evaluate = rbf_kernel(context.graph[context.evaluate], length_scale=graph_scale)
    elif family == SECONDARY:
        if float(graph_multiplier) != 1.0:
            raise ValueError("centered-mu family has no graph bandwidth")
        second_train = centered_linear_kernel(context.mu_features[context.fit], center=0.0, normalize=True)
        second_evaluate = centered_linear_kernel(
            context.mu_features[context.evaluate], center=0.0, normalize=True
        )
    else:
        raise ValueError(f"unknown geometry family: {family}")
    model = fitted_model
    if model is None:
        model = fit_lmc_model(
            context.regional.loo_residuals,
            sem_train,
            second_train,
            kind="separable",
            steps=args.fit_steps,
            learning_rate=args.learning_rate,
            max_pairs=args.max_pairs,
            seed=seed,
        )
    structured = materialize_channel_shrunk(model, sem_evaluate, second_evaluate, beta)
    path = build_correlation_path(
        context.block_model.independent_covariance, structured, H4.shape[0]
    )
    return Endpoint(
        family,
        float(semantic_multiplier),
        float(graph_multiplier),
        float(beta),
        model,
        path,
        structured,
    )


def score_inner_fold(context, fold, args, seed):
    """Fit every structured scale once and score all PSD trust/channel perturbations."""
    block_nll = gaussian_joint_nll(
        context.centered_evaluate,
        np.kron(np.eye(len(context.evaluate)), context.block_model.independent_covariance),
    ).per_scalar
    by_family = {}
    for family in (PRIMARY, SECONDARY):
        records = [{
            "fold": fold,
            "geometry_family": family,
            "alpha": 0.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 1.0,
            "nll_per_scalar": float(block_nll),
        }]
        for semantic_multiplier, graph_multiplier in _family_scale_grid(family):
            fitted_model = None
            for beta in BETAS:
                endpoint = fit_endpoint(
                    context,
                    family,
                    semantic_multiplier,
                    graph_multiplier,
                    beta,
                    args,
                    # Keep the composite-likelihood row-pair sample identical across geometries.
                    seed=seed,
                    fitted_model=fitted_model,
                )
                fitted_model = endpoint.model
                for alpha in ALPHAS[1:]:
                    records.append({
                        "fold": fold,
                        "geometry_family": family,
                        "alpha": alpha,
                        "semantic_multiplier": semantic_multiplier,
                        "graph_multiplier": graph_multiplier,
                        "beta": beta,
                        "nll_per_scalar": endpoint.path.nll_per_scalar(
                            context.centered_evaluate, alpha
                        ),
                    })
        by_family[family] = records
    return by_family


def nested_select(materialized, outer_train, outer_seed, args):
    """Three repeated inner node-disjoint partitions with complete nuisance refitting."""
    records = {PRIMARY: [], SECONDARY: []}
    diagnostics = []
    for fold in range(3):
        split, fit, held, cross = _node_split(
            materialized,
            outer_train,
            10000 + 3 * outer_seed + fold,
            args.inner_held_node_fraction,
            args,
        )
        context = fit_context(materialized, fit, held, args)
        scored = score_inner_fold(context, fold, args, seed=200000 + 1000 * outer_seed + 100 * fold)
        for family in records:
            records[family].extend(scored[family])
        diagnostics.append({
            "fold": fold,
            "seed": split.seed,
            "diagnostics": format_split_diagnostics(split),
            "fit_rows": len(fit),
            "held_rows": len(held),
            "cross_rows_dropped": len(cross),
            "regional_mean": context.regional.to_dict(),
        })
    selections, summaries = {}, {}
    for family in records:
        selections[family], summaries[family] = select_nested_candidate(records[family])
        selections[family]["geometry_family"] = family
        for row in summaries[family]:
            row["geometry_family"] = family
    return selections, summaries, diagnostics


def _null_maximum_gains(paths, draws, seed):
    """Block-null distribution of the exact same alpha/path oracle maximum."""
    if not paths:
        raise ValueError("paths must not be empty")
    dimension = len(paths[0].correlation)
    if any(len(path.correlation) != dimension for path in paths):
        raise ValueError("all null-calibration paths must have the same geometry")
    z = np.random.default_rng(seed).standard_normal((dimension, draws))
    block_quadratic = np.sum(z * z, axis=0)
    maximum = np.zeros(draws)
    for path in paths:
        coordinates = path.eigenvectors.T @ z
        squared = coordinates * coordinates
        for alpha in ALPHAS[1:]:
            values = path.mixture_eigenvalues(alpha)
            gain = 0.5 / dimension * (
                block_quadratic
                - np.sum(squared / values[:, None], axis=0)
                - np.sum(np.log(values))
            )
            maximum = np.maximum(maximum, gain)
    return maximum


def outer_grid(context, args, seed):
    """Fit/score the complete outer diagnostic grid and null-calibrate per family."""
    block_nll = gaussian_joint_nll(
        context.centered_evaluate,
        np.kron(np.eye(len(context.evaluate)), context.block_model.independent_covariance),
    ).per_scalar
    records, endpoints, null = {}, {}, {}
    for family_index, family in enumerate((PRIMARY, SECONDARY)):
        family_records = [{
            "geometry_family": family,
            "alpha": 0.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 1.0,
            "nll_per_scalar": float(block_nll),
            "gain_vs_block": 0.0,
        }]
        family_paths = []
        for semantic_multiplier, graph_multiplier in _family_scale_grid(family):
            fitted_model = None
            for beta in BETAS:
                endpoint = fit_endpoint(
                    context,
                    family,
                    semantic_multiplier,
                    graph_multiplier,
                    beta,
                    args,
                    # Keep the composite-likelihood row-pair sample identical across geometries.
                    seed=seed,
                    fitted_model=fitted_model,
                )
                fitted_model = endpoint.model
                key = (semantic_multiplier, graph_multiplier, beta)
                endpoints[(family,) + key] = endpoint
                family_paths.append(endpoint.path)
                for alpha in ALPHAS[1:]:
                    nll = endpoint.path.nll_per_scalar(context.centered_evaluate, alpha)
                    family_records.append({
                        "geometry_family": family,
                        "alpha": alpha,
                        "semantic_multiplier": semantic_multiplier,
                        "graph_multiplier": graph_multiplier,
                        "beta": beta,
                        "nll_per_scalar": float(nll),
                        "gain_vs_block": float(block_nll - nll),
                    })
        records[family] = family_records
        maxima = _null_maximum_gains(
            family_paths, args.null_draws, seed + 50000 + 1000 * family_index
        )
        null[family] = {
            "scope": "conditional_fixed_path_shared_z; KRR, block covariance, and LMC endpoints held fixed",
            "inferential": False,
            "warning": "not full-procedure v2; endpoint-estimation uncertainty is omitted",
            "draws": int(args.null_draws),
            "mean_maximum_gain": float(np.mean(maxima)),
            "maximum_gain_95th_percentile": float(np.quantile(maxima, 0.95)),
            "maximum_observed_null_gain": float(np.max(maxima)),
        }
    return float(block_nll), records, endpoints, null


def _endpoint_for_selection(endpoints, selection):
    return endpoints[(
        selection["geometry_family"],
        float(selection["semantic_multiplier"]),
        float(selection["graph_multiplier"]),
        float(selection["beta"]),
    )]


def _posterior_distribution(context, covariance, materialized, bridge, hard_target, args):
    prior = context.calibrated.prior[context.evaluate]
    measurement = context.calibrated.meas[context.evaluate]
    innovation = measurement - prior @ H4.T - context.evaluate_mean
    started = time.perf_counter()
    conditioned = condition_item_batch(
        context.calibrated.P0,
        context.conditional_design,
        innovation,
        covariance,
        maximum_relative_loading=args.maximum_relative_loading,
    )
    elapsed = time.perf_counter() - started
    mean = prior + conditioned.state_mean
    probability = gaussian_bridge_proba(
        bridge, mean, conditioned.marginal_covariances, order=args.quad_order
    )
    posterior = {
        "state": state_metrics(
            materialized["y_ds"][context.evaluate], mean, conditioned.marginal_covariances
        ),
        "decision": decision_metrics(probability, hard_target[context.evaluate]),
        "conditioner": {
            "wall_seconds": elapsed,
            "state_dimension": int(2 * len(context.evaluate)),
            "measurement_dimension": int(H4.shape[0] * len(context.evaluate)),
            "loading": conditioned.loading_diagnostics,
            "prior_loading": conditioned.prior_loading_diagnostics,
        },
    }
    return posterior, mean, conditioned.marginal_covariances, probability


def _covariance_record(
    label,
    context,
    covariance,
    materialized,
    bridge,
    hard_target,
    args,
    *,
    candidate,
):
    nll = gaussian_joint_nll(context.centered_evaluate, covariance)
    posterior, mean, marginals, probability = _posterior_distribution(
        context, covariance, materialized, bridge, hard_target, args
    )
    return {
        "label": label,
        "candidate": _jsonable(candidate),
        "held_joint_residual_nll": float(nll.total),
        "held_joint_residual_nll_per_scalar": float(nll.per_scalar),
        "posterior": posterior,
    }, mean, marginals, probability


def _selected_path_range(
    context,
    endpoint,
    materialized,
    bridge,
    hard_target,
    args,
):
    means, nlls, decisions = [], [], []
    for alpha in ALPHAS:
        covariance = endpoint.path.covariance(alpha)
        nlls.append(endpoint.path.nll_per_scalar(context.centered_evaluate, alpha))
        _, mean, _, probability = _posterior_distribution(
            context, covariance, materialized, bridge, hard_target, args
        )
        means.append(mean)
        decisions.append(np.argmax(probability, axis=1))
    means = np.asarray(means)
    coordinate_range = np.ptp(means, axis=0)
    reference = means[0]
    displacement = np.linalg.norm(means - reference[None, :, :], axis=2)
    decision_reference = decisions[0]
    return {
        "alphas": list(ALPHAS),
        "nll_per_scalar": [float(value) for value in nlls],
        "nll_range": float(max(nlls) - min(nlls)),
        "mean_coordinate_range_rms": float(np.sqrt(np.mean(coordinate_range ** 2))),
        "mean_item_max_displacement_mean": float(np.mean(np.max(displacement, axis=0))),
        "mean_item_max_displacement_max": float(np.max(displacement)),
        "maximum_decision_flip_rate_vs_block": float(max(
            np.mean(value != decision_reference) for value in decisions
        )),
    }


def _direct_blend_diagnostic(context, endpoint):
    block = np.kron(
        np.eye(len(context.evaluate)), context.block_model.independent_covariance
    )
    nlls = []
    for alpha in ALPHAS:
        covariance = direct_covariance_blend(
            block, endpoint.structured_covariance, alpha
        )
        nlls.append(
            gaussian_joint_nll(context.centered_evaluate, covariance).per_scalar
        )
    return {
        "warning": "raw covariance blend changes within-item marginals; not a correlation-only effect",
        "alphas": list(ALPHAS),
        "nll_per_scalar": [float(value) for value in nlls],
        "best_gain_vs_block": float(nlls[0] - min(nlls)),
    }


def _summarize(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _posterior_displacement(reference_mean, reference_covariance, candidate_mean):
    values = []
    for reference, covariance, candidate in zip(
        reference_mean, reference_covariance, candidate_mean
    ):
        delta = candidate - reference
        values.append(math.sqrt(max(float(delta @ np.linalg.solve(covariance, delta)), 0.0)))
    return float(np.mean(values))


def _mean_abs_log_sd_change(reference_covariance, candidate_covariance):
    reference_sd = np.sqrt(np.diagonal(reference_covariance, axis1=1, axis2=2))
    candidate_sd = np.sqrt(np.diagonal(candidate_covariance, axis1=1, axis2=2))
    return float(np.mean(np.abs(np.log(candidate_sd / reference_sd))))


def _induced_node_subsample(materialized, outer_train, fraction, seed):
    outer_train = np.asarray(outer_train, dtype=int)
    nodes = sorted(
        {node for i in outer_train for node in materialized["pairs"][i]},
        key=lambda value: (type(value).__qualname__, repr(value)),
    )
    count = min(len(nodes), max(2, int(math.floor(fraction * len(nodes) + 0.5))))
    chosen_index = np.random.default_rng(seed).choice(len(nodes), size=count, replace=False)
    chosen = {nodes[i] for i in chosen_index}
    kept = [
        i for i in outer_train
        if materialized["pairs"][i][0] in chosen and materialized["pairs"][i][1] in chosen
    ]
    return np.asarray(kept, dtype=int), len(nodes), len(chosen)


def _preflight_stability_subsamples(materialized, outer_train, args, seed):
    """Reject deterministic induced-node schedules that cannot support the frozen refits."""
    if args.stability_subsamples == 0:
        return {"subsamples": 0, "status": "disabled by explicit CLI override"}
    row_counts = []
    for replicate in range(args.stability_subsamples):
        fit, _, _ = _induced_node_subsample(
            materialized,
            outer_train,
            args.stability_node_fraction,
            seed + replicate,
        )
        if len(fit) < 20:
            raise ValueError(
                f"stability replicate {replicate} retains only {len(fit)} rows; minimum is 20"
            )
        semantic = materialized["semantic"][fit]
        graph = materialized["graph_item_raw"][fit]
        if not np.any(np.abs(semantic - semantic[0]) > 1e-12):
            raise ValueError(f"stability replicate {replicate} has degenerate semantic geometry")
        if not np.any(np.abs(graph - graph[0]) > 1e-12):
            raise ValueError(f"stability replicate {replicate} has degenerate graph geometry")
        row_counts.append(len(fit))
    return {
        "subsamples": int(args.stability_subsamples),
        "minimum_fit_rows": int(min(row_counts)),
        "maximum_fit_rows": int(max(row_counts)),
        "minimum_required_rows": 20,
        "geometry_non_degenerate": True,
    }


def covariance_for_fixed_selection(context, selection, args, seed):
    alpha = float(selection["alpha"])
    if alpha == 0.0:
        covariance = np.kron(
            np.eye(len(context.evaluate)), context.block_model.independent_covariance
        )
        return covariance, None
    endpoint = fit_endpoint(
        context,
        selection["geometry_family"],
        selection["semantic_multiplier"],
        selection["graph_multiplier"],
        selection["beta"],
        args,
        seed,
    )
    return endpoint.path.covariance(alpha), endpoint


def stability_study(
    materialized,
    outer_train,
    outer_held,
    selection,
    outer_context,
    reference_covariance,
    reference_mean,
    reference_marginals,
    reference_probability,
    bridge,
    hard_target,
    args,
    seed,
):
    """Deterministic 80%-node induced-subsample covariance/posterior stability."""
    if args.stability_subsamples == 0:
        return {"subsamples": 0, "status": "disabled by explicit CLI override"}
    metrics = {
        "whitened_covariance_error": [],
        "posterior_sigma_mean_displacement": [],
        "mean_marginal_symmetric_gaussian_kl": [],
        "mean_abs_log_standard_deviation_change": [],
        "decision_flip_rate": [],
        "held_nll_change_vs_outer_full_selected": [],
        "fit_rows": [],
    }
    samples = []
    fixed_regional = (outer_context.regional.kernel_name, outer_context.regional.ridge)
    reference_decision = np.argmax(reference_probability, axis=1)
    reference_nll = gaussian_joint_nll(
        outer_context.centered_evaluate, reference_covariance
    ).per_scalar
    for replicate in range(args.stability_subsamples):
        replicate_seed = seed + replicate
        fit, node_total, node_kept = _induced_node_subsample(
            materialized, outer_train, args.stability_node_fraction, replicate_seed
        )
        context = fit_context(
            materialized,
            fit,
            outer_held,
            args,
            fixed_regional=fixed_regional,
        )
        covariance, _ = covariance_for_fixed_selection(
            # Fixed pair-sampling RNG isolates node/data sensitivity from optimizer sampling noise.
            context, selection, args, seed=seed + 100000
        )
        _, mean, marginals, probability = _posterior_distribution(
            context, covariance, materialized, bridge, hard_target, args
        )
        values = {
            "whitened_covariance_error": whitened_covariance_error(
                reference_covariance, covariance
            ),
            "posterior_sigma_mean_displacement": _posterior_displacement(
                reference_mean, reference_marginals, mean
            ),
            "mean_marginal_symmetric_gaussian_kl": mean_marginal_symmetric_kl(
                reference_mean, reference_marginals, mean, marginals
            ),
            "mean_abs_log_standard_deviation_change": _mean_abs_log_sd_change(
                reference_marginals, marginals
            ),
            "decision_flip_rate": float(np.mean(
                np.argmax(probability, axis=1) != reference_decision
            )),
            "held_nll_change_vs_outer_full_selected": float(
                gaussian_joint_nll(context.centered_evaluate, covariance).per_scalar
                - reference_nll
            ),
            "fit_rows": int(len(fit)),
        }
        for key in metrics:
            metrics[key].append(values[key])
        samples.append({
            "replicate": replicate,
            "seed": replicate_seed,
            "outer_train_nodes": node_total,
            "selected_nodes": node_kept,
            **values,
        })
    return {
        "subsamples": int(args.stability_subsamples),
        "node_fraction": float(args.stability_node_fraction),
        "interpretation": "induced-node fit stability; not a bootstrap confidence interval",
        "summary": {key: _summarize(value) for key, value in metrics.items()},
        "samples": samples if args.keep_stability_samples else None,
    }


def run_outer_split(corpus, materialized, outer_seed, args):
    all_indices = np.arange(len(materialized["pairs"]), dtype=int)
    split, outer_train, outer_held, outer_cross = _node_split(
        materialized, all_indices, outer_seed, args.outer_held_node_fraction, args
    )
    preflight = _preflight_stability_subsamples(
        materialized, outer_train, args, seed=700000 + 10000 * outer_seed
    )
    started = time.perf_counter()
    selections, inner_summaries, inner_diagnostics = nested_select(
        materialized, outer_train, outer_seed, args
    )
    context = fit_context(materialized, outer_train, outer_held, args)
    block_nll, grid, endpoints, null = outer_grid(
        context, args, seed=400000 + 10000 * outer_seed
    )
    bridge = JointPosterior(CLASSES, n_features=2, hidden=0, seed=3000 + outer_seed).fit(
        materialized["y_ds"][outer_train],
        materialized["hard_target"][outer_train],
        epochs=args.bridge_epochs,
    )
    block_covariance = np.kron(
        np.eye(len(outer_held)), context.block_model.independent_covariance
    )
    models = {}
    block_record, _, _, _ = _covariance_record(
        "regional block",
        context,
        block_covariance,
        materialized,
        bridge,
        materialized["hard_target"],
        args,
        candidate={"alpha": 0.0},
    )
    models["block_regional"] = block_record
    family_records = {}
    for family in (PRIMARY, SECONDARY):
        selection = selections[family]
        endpoint = _endpoint_for_selection(endpoints, selection)
        selected_covariance = endpoint.path.covariance(selection["alpha"])
        selected_record, selected_mean, selected_marginals, selected_probability = _covariance_record(
            f"nested-selected {family}",
            context,
            selected_covariance,
            materialized,
            bridge,
            materialized["hard_target"],
            args,
            candidate=selection,
        )
        full_candidate = dict(selection, alpha=1.0)
        full_record, _, _, _ = _covariance_record(
            f"full-trust marginal-matched endpoint at nested-selected {family} geometry",
            context,
            endpoint.path.covariance(1.0),
            materialized,
            bridge,
            materialized["hard_target"],
            args,
            candidate=full_candidate,
        )
        oracle = min(grid[family], key=lambda row: row["nll_per_scalar"])
        oracle_endpoint = _endpoint_for_selection(endpoints, oracle) if oracle["alpha"] else None
        oracle_covariance = (
            block_covariance if oracle_endpoint is None
            else oracle_endpoint.path.covariance(oracle["alpha"])
        )
        oracle_record, _, _, _ = _covariance_record(
            f"non-deployable outer oracle {family}",
            context,
            oracle_covariance,
            materialized,
            bridge,
            materialized["hard_target"],
            args,
            candidate=oracle,
        )
        oracle_gain = float(block_nll - oracle["nll_per_scalar"])
        oracle_record["gain_vs_block"] = oracle_gain
        oracle_record["null_maximum_gain_95th_percentile"] = null[family][
            "maximum_gain_95th_percentile"
        ]
        oracle_record["above_null_max95"] = bool(
            oracle_gain > null[family]["maximum_gain_95th_percentile"]
        )
        oracle_record["null_calibration_scope"] = null[family]["scope"]
        oracle_record["inferential_oracle_headroom"] = False
        innovation = (
            context.calibrated.meas[outer_held]
            - context.calibrated.prior[outer_held] @ H4.T
            - context.evaluate_mean
        )
        sensitivity = directional_posterior_sensitivity(
            context.calibrated.P0,
            context.conditional_design,
            innovation,
            selected_covariance,
            endpoint.path.covariance(1.0) - endpoint.path.covariance(0.0),
        )
        stability = (
            stability_study(
                materialized,
                outer_train,
                outer_held,
                selection,
                context,
                selected_covariance,
                selected_mean,
                selected_marginals,
                selected_probability,
                bridge,
                materialized["hard_target"],
                args,
                seed=700000 + 10000 * outer_seed,
            )
            if family == PRIMARY else
            {
                "status": "not run for the secondary endogenous geometry",
                "reason": "the preregistered 100-subsample stability budget applies to the primary family",
            }
        )
        family_records[family] = {
            "selection": selection,
            "selected_endpoint_component_fit": endpoint.model.to_dict(),
            "outer_grid_component_fits": [
                {
                    "semantic_multiplier": semantic_multiplier,
                    "graph_multiplier": graph_multiplier,
                    "fit": endpoints[(
                        family, semantic_multiplier, graph_multiplier, 0.0
                    )].model.to_dict(),
                }
                for semantic_multiplier, graph_multiplier in _family_scale_grid(family)
            ],
            "inner_candidate_summaries": inner_summaries[family],
            "nested_selected": selected_record,
            "full_at_selected_geometry": full_record,
            "outer_oracle": oracle_record,
            "null_calibration": null[family],
            "outer_held_nll_grid": grid[family],
            "analytic_sensitivity": sensitivity.to_dict(),
            "selected_path_range": _selected_path_range(
                context,
                endpoint,
                materialized,
                bridge,
                materialized["hard_target"],
                args,
            ),
            "direct_raw_covariance_blend": _direct_blend_diagnostic(context, endpoint),
            "covariance_estimation_stability": stability,
        }
    canonical_endpoint = endpoints[(PRIMARY, 1.0, 1.0, 0.0)]
    canonical_record, _, _, _ = _covariance_record(
        "original canonical primary raw plug-in endpoint",
        context,
        canonical_endpoint.structured_covariance,
        materialized,
        bridge,
        materialized["hard_target"],
        args,
        candidate={
            "geometry_family": PRIMARY,
            "alpha": 1.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 0.0,
        },
    )
    models["canonical_primary_raw_plugin"] = canonical_record
    canonical_matched_record, _, _, _ = _covariance_record(
        "canonical primary marginal-matched full-trust endpoint",
        context,
        canonical_endpoint.path.covariance(1.0),
        materialized,
        bridge,
        materialized["hard_target"],
        args,
        candidate={
            "geometry_family": PRIMARY,
            "alpha": 1.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 0.0,
        },
    )
    models["canonical_primary_marginal_matched"] = canonical_matched_record
    return {
        "corpus": corpus,
        "outer_seed": outer_seed,
        "target_frame": "GPT-5.5 operating-judge fidelity; not independent truth",
        "split": {
            "diagnostics": format_split_diagnostics(split),
            "train_rows": len(outer_train),
            "held_rows": len(outer_held),
            "cross_rows_dropped": len(outer_cross),
            "train_nodes": len(split.train_nodes),
            "held_nodes": len(split.held_nodes),
            "selected_candidate": split.selected_candidate,
            "train_tags": dict(Counter(map(str, materialized["tags"][outer_train]))),
            "held_tags": dict(Counter(map(str, materialized["tags"][outer_held]))),
        },
        "inner_partitions": inner_diagnostics,
        "stability_preflight": preflight,
        "outer_fit": {
            "semantic_rbf_bandwidth": context.semantic_length,
            "graph_rbf_bandwidth": context.graph_length,
            "graph_standardizer_mean": context.graph_mean.tolist(),
            "graph_standardizer_scale": context.graph_scale.tolist(),
            "regional_mean": context.regional.to_dict(),
            "block_model": context.block_model.to_dict(),
        },
        "models": models,
        "families": family_records,
        "wall_seconds": time.perf_counter() - started,
    }


def aggregate_results(results):
    """Apply the frozen direction/stability/guardrail gates without row-level pseudo-SEs."""
    by_family = {}
    for family in (PRIMARY, SECONDARY):
        by_corpus = {}
        for corpus in sorted({row["corpus"] for row in results}):
            rows = sorted(
                [row for row in results if row["corpus"] == corpus],
                key=lambda row: row["outer_seed"],
            )
            block = np.asarray([
                row["models"]["block_regional"]["held_joint_residual_nll_per_scalar"]
                for row in rows
            ])
            selected = np.asarray([
                row["families"][family]["nested_selected"]["held_joint_residual_nll_per_scalar"]
                for row in rows
            ])
            posterior_block = np.asarray([
                row["models"]["block_regional"]["posterior"]["state"]["mean_bivariate_nll"]
                for row in rows
            ])
            posterior_selected = np.asarray([
                row["families"][family]["nested_selected"]["posterior"]["state"]["mean_bivariate_nll"]
                for row in rows
            ])
            logloss_block = np.asarray([
                row["models"]["block_regional"]["posterior"]["decision"]["log_loss"]
                for row in rows
            ])
            logloss_selected = np.asarray([
                row["families"][family]["nested_selected"]["posterior"]["decision"]["log_loss"]
                for row in rows
            ])
            aurc_block = np.asarray([
                row["models"]["block_regional"]["posterior"]["decision"]["aurc_margin"]
                for row in rows
            ])
            aurc_selected = np.asarray([
                row["families"][family]["nested_selected"]["posterior"]["decision"]["aurc_margin"]
                for row in rows
            ])
            gain = block - selected
            posterior_gain = posterior_block - posterior_selected
            logloss_gain = logloss_block - logloss_selected
            aurc_gain = aurc_block - aurc_selected
            oracle_gain = np.asarray([
                row["families"][family]["outer_oracle"]["gain_vs_block"] for row in rows
            ])
            oracle_above = np.asarray([
                row["families"][family]["outer_oracle"]["above_null_max95"] for row in rows
            ], dtype=bool)
            alphas = np.asarray([
                row["families"][family]["selection"]["alpha"] for row in rows
            ])
            noise_loading = np.asarray([
                row["families"][family]["nested_selected"]["posterior"]["conditioner"]["loading"]
                ["relative_diagonal_loading"]
                for row in rows
            ])
            prior_loading = np.asarray([
                row["families"][family]["nested_selected"]["posterior"]["conditioner"]
                ["prior_loading"]["relative_diagonal_loading"]
                for row in rows
            ])
            by_corpus[corpus] = {
                "outer_seeds": [row["outer_seed"] for row in rows],
                "nested_gain_mean_sd": [
                    float(np.mean(gain)),
                    float(np.std(gain, ddof=1)) if len(gain) > 1 else 0.0,
                ],
                "nested_positive_seeds": int(np.sum(gain > 0.0)),
                "nonzero_alpha_seeds": int(np.sum(alphas > 0.0)),
                "posterior_nll_gain_mean": float(np.mean(posterior_gain)),
                "decision_log_loss_gain_mean": float(np.mean(logloss_gain)),
                "decision_aurc_gain_mean": float(np.mean(aurc_gain)),
                "oracle_gain_mean_sd": [
                    float(np.mean(oracle_gain)),
                    float(np.std(oracle_gain, ddof=1)) if len(oracle_gain) > 1 else 0.0,
                ],
                "oracle_positive_seeds": int(np.sum(oracle_gain > 0.0)),
                "oracle_above_null_max95_seeds": int(np.sum(oracle_above)),
                "maximum_noise_relative_diagonal_loading": float(np.max(noise_loading)),
                "maximum_prior_relative_diagonal_loading": float(np.max(prior_loading)),
            }
        by_family[family] = by_corpus

    complete = (
        {row["corpus"] for row in results} == {"exploratory", "fresh"}
        and all(
            len([row for row in results if row["corpus"] == corpus]) == 10
            and {row["outer_seed"] for row in results if row["corpus"] == corpus}
                == set(range(10, 20))
            for corpus in ("exploratory", "fresh")
        )
    )

    def gate(family):
        summaries = by_family[family]
        raw_nested_direction = complete and all(
            value["nested_gain_mean_sd"][0] > 0.0 and value["nested_positive_seeds"] >= 8
            for value in summaries.values()
        )
        posterior = all(
            value["posterior_nll_gain_mean"] >= 0.0 for value in summaries.values()
        )
        decision = all(
            value["decision_log_loss_gain_mean"] >= -0.01
            and value["decision_aurc_gain_mean"] >= -0.01
            for value in summaries.values()
        )
        loading = all(
            value["maximum_noise_relative_diagonal_loading"] <= 1e-3
            and value["maximum_prior_relative_diagonal_loading"] <= 1e-3
            for value in summaries.values()
        )
        conditional_oracle = complete and all(
            value["oracle_positive_seeds"] >= 8
            and value["oracle_above_null_max95_seeds"] >= 8
            for value in summaries.values()
        )
        return {
            "gate_evaluable": False,
            "selector_status": (
                "INVALIDATED v1: family-wise synthetic controls failed; multi-seed execution is descriptive only"
            ),
            "uncalibrated_v1_raw_direction_and_stability_would_pass": raw_nested_direction,
            "posterior_guardrail_passes": posterior,
            "decision_guardrail_passes": decision,
            "loading_guardrail_passes": loading,
            "nested_covariance_gate_passes": False,
            "conditional_fixed_path_oracle_headroom_diagnostic": conditional_oracle,
            "full_procedure_null_calibrated_oracle_headroom_passes": False,
            "interpretation": (
                "incomplete smoke/descriptive run; no inferential gate"
                if not complete else
                "complete descriptive v1 grid only; selector and fixed-path null are undercalibrated"
            ),
        }

    return {
        "complete_preregistered_outer_run": complete,
        "inferential_gate_evaluable": False,
        "blocking_reason": (
            "v1 selector failed family-wise synthetic controls and the real outer null is conditional fixed-path only"
        ),
        "by_family": by_family,
        "primary_gate": gate(PRIMARY),
        "secondary_endogenous_geometry_gate": gate(SECONDARY),
        "variation_note": "mean/SD across node partitions is descriptive stability, not a population CI",
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo", default=os.path.abspath(os.path.join(ROOT, "..", "..")))
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--luna", default=DEFAULT_LUNA)
    parser.add_argument("--outer-seed-start", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--outer-held-node-fraction", type=float, default=0.40)
    parser.add_argument("--inner-held-node-fraction", type=float, default=0.35)
    parser.add_argument("--split-candidates", type=int, default=64)
    parser.add_argument("--minimum-per-stratum", type=int, default=1)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--fit-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--bridge-epochs", type=int, default=300)
    parser.add_argument("--quad-order", type=int, default=5)
    parser.add_argument("--maximum-relative-loading", type=float, default=1e-3)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--stability-subsamples", type=int, default=100)
    parser.add_argument("--stability-node-fraction", type=float, default=0.80)
    parser.add_argument("--keep-stability-samples", action="store_true")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Torch intra-op threads only; set BLAS thread environment variables externally if required",
    )
    parser.add_argument("--out", default="/tmp/covariance_sensitivity.json")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an exactly matching atomically checkpointed output",
    )
    parser.add_argument(
        "--allow-failed-v1-selector",
        action="store_true",
        help=(
            "explicitly allow a multi-seed v1 run even though its preregistered synthetic selector gate failed"
        ),
    )
    return parser


def _validate_args(args):
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.seeds > 1 and not args.allow_failed_v1_selector:
        raise ValueError(
            "multi-seed v1 execution is blocked because its synthetic selector gate failed; "
            "pass --allow-failed-v1-selector only for an explicitly descriptive run"
        )
    if args.null_draws < 1:
        raise ValueError("--null-draws must be positive")
    if args.stability_subsamples < 0:
        raise ValueError("--stability-subsamples must be nonnegative")
    if not 0.0 < args.stability_node_fraction <= 1.0:
        raise ValueError("--stability-node-fraction must be in (0,1]")


def _write_payload(path, payload):
    path = os.path.abspath(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main():
    args = build_arg_parser().parse_args()
    _validate_args(args)
    torch.set_num_threads(args.cpu_threads)
    np.random.seed(0)
    artifacts = configure_artifact_repo(args.artifact_repo)
    target_by_pair, cur_rel = load_decision_targets(args.campaign)
    luna_pairs, luna_d, luna_s = load_scored_mu_tsv(args.luna)
    luna_by_pair = {pair: (luna_d[i], luna_s[i]) for i, pair in enumerate(luna_pairs)}
    checkpoint = load_expanded(args.ckpt, dev="cpu")
    checkpoint[0].eval()
    datasets = load_campaign_datasets(campaign_scored=args.campaign)
    materialized = {
        name.replace("-campaign", ""): materialize_corpus(
            name, dataset, target_by_pair, luna_by_pair, checkpoint
        )
        for name, dataset in datasets.items()
    }
    configuration = vars(args).copy()
    configuration.pop("resume")
    payload = {
        "status": (
            "DESCRIPTIVE LEGACY V1; selector failed family-wise synthetic calibration; no inferential gate"
        ),
        "real_data_v2_selector_implemented": False,
        "protocol": (
            "legacy-v1 nested covariance sensitivity; node-disjoint; train-only refitting; "
            "multi-seed inference disabled"
        ),
        "target_scope": "GPT-5.5 operating-judge fidelity; not independent ground truth",
        "inputs": {
            "checkpoint": file_provenance(args.ckpt),
            "campaign": file_provenance(args.campaign),
            "luna": file_provenance(args.luna),
            "artifact_paths": artifacts,
            "e5_caches": {
                "exploratory": file_provenance(CAMPAIGN_E5_100K),
                "fresh": file_provenance(DATASETS["fresh"]["e5_cache"]),
            },
            "campaign_cur_rel_counts": dict(cur_rel),
        },
        "configuration": configuration,
        "frozen_grid": {
            "alphas": list(ALPHAS),
            "multipliers": list(MULTIPLIERS),
            "betas": list(BETAS),
            "families": [PRIMARY, SECONDARY],
        },
        "compute_note": (
            "CPU float64; --cpu-threads controls Torch only, while NumPy linear algebra follows the "
            "process BLAS environment; no CUDA claim"
        ),
        "results": [],
        "aggregate": None,
        "resume_events": [],
    }
    if args.resume and os.path.isfile(args.out):
        with open(args.out, "r", encoding="utf-8") as stream:
            previous = json.load(stream)
        checks = (
            ("protocol", previous.get("protocol"), payload["protocol"]),
            ("configuration", previous.get("configuration"), payload["configuration"]),
            ("frozen_grid", previous.get("frozen_grid"), payload["frozen_grid"]),
            ("inputs", previous.get("inputs"), payload["inputs"]),
        )
        mismatched = [name for name, old, new in checks if old != new]
        if mismatched:
            raise ValueError(
                "refusing to resume output with mismatched " + ", ".join(mismatched)
            )
        payload = previous
        payload.setdefault("resume_events", []).append({
            "completed_records_at_resume": len(payload.get("results", [])),
            "note": "exact configuration, grid, and input provenance matched",
        })
    completed = {
        (row["corpus"], int(row["outer_seed"])) for row in payload["results"]
    }
    if len(completed) != len(payload["results"]):
        raise ValueError("checkpoint contains duplicate corpus/outer-seed records")
    for outer_seed in range(args.outer_seed_start, args.outer_seed_start + args.seeds):
        for corpus in ("exploratory", "fresh"):
            if (corpus, outer_seed) in completed:
                print(f"skipping completed {corpus} outer seed {outer_seed}", flush=True)
                continue
            print(f"\n=== covariance sensitivity: {corpus}, outer seed {outer_seed} ===", flush=True)
            row = run_outer_split(corpus, materialized[corpus], outer_seed, args)
            payload["results"].append(row)
            completed.add((corpus, outer_seed))
            for family in (PRIMARY, SECONDARY):
                nested = row["families"][family]["nested_selected"]
                oracle = row["families"][family]["outer_oracle"]
                print(
                    f"  {family:34s} alpha={row['families'][family]['selection']['alpha']:.3f} "
                    f"nested={nested['held_joint_residual_nll_per_scalar']:+.6f} "
                    f"oracle_gain={oracle['gain_vs_block']:+.6f} "
                    f"above_null95={oracle['above_null_max95']}",
                    flush=True,
                )
            payload["aggregate"] = aggregate_results(payload["results"])
            _write_payload(args.out, payload)
            print(f"  checkpointed {args.out}; split wall={row['wall_seconds']:.1f}s", flush=True)
    payload["aggregate"] = aggregate_results(payload["results"])
    _write_payload(args.out, payload)
    print(f"\nwrote {args.out}")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
