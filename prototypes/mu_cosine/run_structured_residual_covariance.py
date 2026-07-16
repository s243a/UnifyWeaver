#!/usr/bin/env python3
"""Held node-disjoint evidence gate for cross-item conditional residual covariance.

This runner reuses the post-#3648 campaign calibration and residual convention.  It first removes the
within-item prior/measurement cross term, then asks whether the remaining conditional residual ``q`` is
predictable/correlated across campaign rows in frozen semantic or graph-feature geometry.

The target is GPT-5.5 operating-judge fidelity, not ground truth.  No new judge calls are made.  See
``DESIGN_structured_residual_covariance.md`` for the preregistered models, metrics, and go/no-go rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_luna_transfer import load_luna as load_scored_mu_tsv
from emit_transitive_hops import hit_prob
from fine_tune_channel_heads import CAMPAIGN_E5_100K, load_campaign_datasets, load_expanded
from fine_tune_fused_head import agnostic_readouts
from mu_posterior import JointPosterior, _eval, aurc, margin_conf
from node_disjoint_eval import format_split_diagnostics, node_disjoint_pair_split
from run_cheap_judge_joint_posterior import (
    CLASSES,
    calibrate_sources,
    gaussian_bridge_proba,
    load_decision_targets,
)
from run_product_kalman_logit import dequant
from run_product_kalman_realdata import DATASETS
from run_sym_channel_fusion import H4, sym_graph_features
from sigma_hop_confirmatory import FeatureGraphConfig, load_feature_graph
from structured_residual_covariance import (
    condition_item_batch,
    conditional_residuals,
    fit_block_model,
    fit_lmc_model,
    gaussian_joint_nll,
    median_rbf_bandwidth,
    off_block_diagnostics,
    rbf_kernel,
    select_kernel_ridge_mean,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAMPAIGN = "/tmp/mu_data/campaign_scored.tsv"
DEFAULT_LUNA = "/tmp/mu_data/campaign_scored_luna.tsv"
MODEL_NAMES = (
    "block_global",
    "block_regional",
    "separable_regional",
    "dense_lmc_regional",
)
CHI2_2_95 = 5.991464547107979


def sha256_file(path):
    """Hash one immutable input without loading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def file_provenance(path):
    """Absolute path, byte size, and content hash for an outcome-determining regular file."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing provenance file: {path}")
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
    }


def configure_artifact_repo(repo):
    """Point ignored graph artifacts at a checkout that owns them.

    Git worktrees intentionally do not duplicate ignored checkpoints/LMDB stores.  Mutating the shared
    ``DATASETS`` dictionary before calling the canonical loader keeps the loader itself unchanged and makes
    the external-artifact dependency explicit in result provenance.
    """
    repo = os.path.abspath(repo)
    exploratory = os.path.join(repo, "data", "benchmark", "100k_cats", "category_parent.tsv")
    fresh = os.path.join(repo, "data", "benchmark", "enwiki_cats_correct", "lmdb_scoped")
    if not os.path.isfile(exploratory):
        raise FileNotFoundError(f"missing exploratory graph: {exploratory}")
    if not os.path.isdir(fresh):
        raise FileNotFoundError(f"missing fresh LMDB graph: {fresh}")
    DATASETS["exploratory"]["graph"]["graph"] = exploratory
    DATASETS["fresh"]["graph"]["candidate_lmdb"] = fresh
    DATASETS["fresh"]["graph"]["exploratory_graph"] = exploratory
    fresh_data = os.path.join(fresh, "data.mdb")
    return {
        "repository": repo,
        "exploratory_graph": file_provenance(exploratory),
        "fresh_lmdb_directory": fresh,
        "fresh_lmdb_data": file_provenance(fresh_data),
        "fresh_lmdb_lock_excluded": "lock.mdb is process state, not graph content",
    }


def semantic_pair_features(ds, pairs):
    """Normalized outcome-blind row features: ``[passage(node), query(root)]``."""
    rows = []
    for node, root in pairs:
        try:
            left = ds["tok"].p[ds["tok"].idx[node]].detach().cpu().numpy()
            right = ds["tok"].q[ds["tok"].idx[root]].detach().cpu().numpy()
        except KeyError as exc:
            raise ValueError(f"pair endpoint is absent from the frozen e5 cache: {exc.args[0]!r}") from exc
        rows.append(np.concatenate([left, right]))
    features = np.asarray(rows, dtype=float)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norm == 0) or not np.isfinite(features).all():
        raise ValueError("semantic pair features must be finite and nonzero")
    return features / norm


def train_standardize(features, train):
    """Fit a feature standardizer on train rows and apply it to every row."""
    features = np.asarray(features, dtype=float)
    train = np.asarray(train, dtype=int)
    if features.ndim != 2 or len(train) == 0 or not np.isfinite(features).all():
        raise ValueError("features must be a finite matrix with non-empty train indices")
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (features - mean) / scale, mean, scale


def decision_metrics(proba, labels):
    """Operating-judge decision metrics; AURC uses margin confidence and no row bootstrap."""
    labels = np.asarray(labels)
    relation_index = {name: i for i, name in enumerate(CLASSES)}
    accuracy, log_loss, ece = _eval(proba, labels, relation_index, bins=10)
    target = np.asarray([relation_index[value] for value in labels], dtype=int)
    correct = proba.argmax(axis=1) == target
    return {
        "accuracy": accuracy,
        "log_loss": log_loss,
        "ece_10_equal_width": ece,
        "aurc_margin": aurc(margin_conf(proba), correct),
    }


def state_metrics(observed, mean, marginal_covariances):
    """Per-item bivariate posterior diagnostics from marginal 2x2 covariance blocks."""
    observed = np.asarray(observed, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(marginal_covariances, dtype=float)
    if observed.shape != mean.shape or observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("observed and mean must both be [N,2]")
    if covariance.shape != (len(mean), 2, 2):
        raise ValueError("marginal_covariances must be [N,2,2]")
    error = observed - mean
    nll, mahal = [], []
    for value, cov in zip(error, covariance):
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError("posterior marginal covariance is not positive definite")
        m = float(value @ np.linalg.solve(cov, value))
        mahal.append(m)
        nll.append(0.5 * (m + logdet + 2.0 * np.log(2.0 * np.pi)))
    mahal = np.asarray(mahal)
    return {
        "mean_bivariate_nll": float(np.mean(nll)),
        "mse_per_scalar": float(np.mean(error * error)),
        "mahalanobis_per_dimension": float(np.mean(mahal) / 2.0),
        "coverage_95_bivariate_ellipse": float(np.mean(mahal <= CHI2_2_95)),
    }


def _kernel_geometry(semantic, graph, train, held):
    semantic_length = median_rbf_bandwidth(semantic[train])
    graph_length = median_rbf_bandwidth(graph[train])
    kernels_train = {
        "semantic": rbf_kernel(semantic[train], length_scale=semantic_length),
        "graph_feature": rbf_kernel(graph[train], length_scale=graph_length),
    }
    kernels_train["equal_mixture"] = 0.5 * (
        kernels_train["semantic"] + kernels_train["graph_feature"]
    )
    kernels_cross = {
        "semantic": rbf_kernel(semantic[held], semantic[train], length_scale=semantic_length),
        "graph_feature": rbf_kernel(graph[held], graph[train], length_scale=graph_length),
    }
    kernels_cross["equal_mixture"] = 0.5 * (
        kernels_cross["semantic"] + kernels_cross["graph_feature"]
    )
    held_semantic = rbf_kernel(semantic[held], length_scale=semantic_length)
    held_graph = rbf_kernel(graph[held], length_scale=graph_length)
    return kernels_train, kernels_cross, held_semantic, held_graph, semantic_length, graph_length


def _model_record(model, held_semantic, held_graph, centered_held, block_size):
    covariance = model.materialize(held_semantic, held_graph)
    nll = gaussian_joint_nll(centered_held, covariance)
    diagnostics = off_block_diagnostics(covariance, block_size)
    return covariance, {
        "held_joint_residual_nll": float(nll.total),
        "held_joint_residual_nll_per_scalar": float(nll.per_scalar),
        "fit": model.to_dict(),
        "diagnostics": diagnostics.to_dict(),
    }


def _posterior_record(split, held, conditional_design, residual_mean, covariance, bridge, hard_target, args):
    prior = split.prior[held]
    measurement = split.meas[held]
    innovation = measurement - prior @ H4.T - residual_mean
    started = time.perf_counter()
    conditioned = condition_item_batch(
        split.P0,
        conditional_design,
        innovation,
        covariance,
        maximum_relative_loading=args.maximum_relative_loading,
    )
    elapsed = time.perf_counter() - started
    posterior_mean = prior + conditioned.state_mean
    state = state_metrics(split.y_ds[held], posterior_mean, conditioned.marginal_covariances)
    probability = gaussian_bridge_proba(
        bridge, posterior_mean, conditioned.marginal_covariances, order=args.quad_order
    )
    return {
        "state": state,
        "decision": decision_metrics(probability, hard_target[held]),
        "conditioner": {
            "wall_seconds": elapsed,
            "state_dimension": int(2 * len(held)),
            "measurement_dimension": int(H4.shape[0] * len(held)),
            "loading": conditioned.loading_diagnostics,
            "prior_loading": conditioned.prior_loading_diagnostics,
        },
    }


def materialize_corpus(name, ds, target_by_pair, luna_by_pair, checkpoint):
    """Compute seed-independent campaign sources and geometry once per corpus."""
    corpus = name.replace("-campaign", "")
    parents, _, _, _ = load_feature_graph(FeatureGraphConfig(**DATASETS[corpus]["graph"]))
    parents = {node: tuple(sorted(values)) for node, values in parents.items()}
    ds["tok"].parents = parents
    model, _ = checkpoint
    readout = agnostic_readouts(model, ds, "cpu")
    keep = [
        i for i, pair in enumerate(ds["pairs"])
        if pair in target_by_pair and pair in luna_by_pair
    ]
    pairs = [ds["pairs"][i] for i in keep]
    if len(pairs) != len(set(pairs)):
        raise ValueError("structured residual comparison requires unique pairs")
    tags = np.asarray([ds["tags"][i] for i in keep])
    y_ds = dequant(np.column_stack([ds["D"][keep], ds["S"][keep]]))
    prior = np.column_stack([readout["prior_D"][keep], readout["prior_S"][keep]])
    graph_d_raw = np.asarray([hit_prob(parents, node, root) for node, root in pairs])
    graph_s_features = sym_graph_features(parents, pairs)
    graph_item_raw = np.column_stack([graph_d_raw, graph_s_features])
    luna = np.asarray([luna_by_pair[pair] for pair in pairs], dtype=float)
    hard_target = np.asarray([target_by_pair[pair][1] for pair in pairs])
    semantic = semantic_pair_features(ds, pairs)
    return {
        "pairs": pairs,
        "tags": tags,
        "y_ds": y_ds,
        "prior": prior,
        "graph_d_raw": graph_d_raw,
        "graph_s_features": graph_s_features,
        "graph_item_raw": graph_item_raw,
        "luna": luna,
        "hard_target": hard_target,
        "semantic": semantic,
    }


def run_corpus(name, ds, target_by_pair, luna_by_pair, checkpoint, seed, args):
    corpus = name.replace("-campaign", "")
    cache_key = "_structured_residual_covariance_cache"
    if cache_key not in ds:
        ds[cache_key] = materialize_corpus(name, ds, target_by_pair, luna_by_pair, checkpoint)
    materialized = ds[cache_key]
    pairs = materialized["pairs"]
    tags = materialized["tags"]
    y_ds = materialized["y_ds"]
    prior = materialized["prior"]
    graph_d_raw = materialized["graph_d_raw"]
    graph_s_features = materialized["graph_s_features"]
    graph_item_raw = materialized["graph_item_raw"]
    luna = materialized["luna"]
    hard_target = materialized["hard_target"]
    semantic = materialized["semantic"]

    split_spec = node_disjoint_pair_split(
        pairs,
        seed,
        held_node_fraction=args.held_node_fraction,
        strata=tags,
        candidates=args.split_candidates,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    train, held = split_spec.train, split_spec.held
    if not len(train) or not len(held):
        raise ValueError(f"{name} seed {seed} produced an empty retained partition")
    calibrated = calibrate_sources(prior, graph_d_raw, graph_s_features, luna, y_ds, train)
    observed_measurement_state = y_ds[:, [0, 1, 0, 1]]
    prior_error = y_ds - calibrated.prior
    measurement_error = calibrated.meas - observed_measurement_state
    conditional_design, q = conditional_residuals(
        prior_error, measurement_error, calibrated.P0, calibrated.C_pm, H4
    )

    graph_item, graph_mean, graph_scale = train_standardize(graph_item_raw, train)
    geometry = _kernel_geometry(semantic, graph_item, train, held)
    kernels_train, kernels_cross, held_semantic, held_graph, sem_length, graph_length = geometry
    regional = select_kernel_ridge_mean(
        q[train], kernels_train, ridge_grid=args.ridge_grid
    )
    held_regional_mean = regional.predict(kernels_cross[regional.kernel_name])
    global_mean = np.mean(q[train], axis=0)

    block_global = fit_block_model(q[train] - global_mean, shrinkage=args.shrinkage)
    block_regional = fit_block_model(regional.loo_residuals, shrinkage=args.shrinkage)
    separable = fit_lmc_model(
        regional.loo_residuals,
        kernels_train["semantic"],
        kernels_train["graph_feature"],
        kind="separable",
        steps=args.fit_steps,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
        seed=4000 + seed,
    )
    dense_lmc = fit_lmc_model(
        regional.loo_residuals,
        kernels_train["semantic"],
        kernels_train["graph_feature"],
        kind="dense_lmc",
        steps=args.fit_steps,
        learning_rate=args.learning_rate,
        max_pairs=args.max_pairs,
        seed=4000 + seed,
        initial_model=separable,
    )
    fits = {
        "block_global": block_global,
        "block_regional": block_regional,
        "separable_regional": separable,
        "dense_lmc_regional": dense_lmc,
    }
    held_means = {
        "block_global": np.broadcast_to(global_mean, (len(held), len(global_mean))),
        "block_regional": held_regional_mean,
        "separable_regional": held_regional_mean,
        "dense_lmc_regional": held_regional_mean,
    }
    bridge = JointPosterior(CLASSES, n_features=2, hidden=0, seed=3000 + seed).fit(
        y_ds[train], hard_target[train], epochs=args.bridge_epochs
    )

    records = {}
    for model_name in MODEL_NAMES:
        centered = q[held] - held_means[model_name]
        covariance, record = _model_record(
            fits[model_name], held_semantic, held_graph, centered, H4.shape[0]
        )
        record["posterior"] = _posterior_record(
            calibrated,
            held,
            conditional_design,
            held_means[model_name],
            covariance,
            bridge,
            hard_target,
            args,
        )
        records[model_name] = record

    return {
        "corpus": corpus,
        "seed": seed,
        "target_frame": "GPT-5.5 operating-judge D/S and macro-decision fidelity; not ground truth",
        "split": {
            "diagnostics": format_split_diagnostics(split_spec),
            "train_rows": len(train),
            "held_rows": len(held),
            "cross_rows_dropped": len(split_spec.cross),
            "train_nodes": len(split_spec.train_nodes),
            "held_nodes": len(split_spec.held_nodes),
            "selected_candidate": split_spec.selected_candidate,
            "candidates": split_spec.candidates,
            "stratification": "campaign neighborhood tag; outcome-blind",
            "train_tags": dict(Counter(map(str, tags[train]))),
            "held_tags": dict(Counter(map(str, tags[held]))),
        },
        "geometry": {
            "semantic_feature": "normalized concat(passage(node), query(root))",
            "graph_feature": "train-standardized [hit_prob, *sym_graph_features]; not shortest-path",
            "semantic_rbf_bandwidth": sem_length,
            "graph_feature_rbf_bandwidth": graph_length,
            "graph_standardizer_mean": graph_mean.tolist(),
            "graph_standardizer_scale": graph_scale.tolist(),
            "regional_mean": regional.to_dict(),
        },
        "conditional_model": {
            "signs": "e=truth-prior; v=measurement-H truth; q=v-C.T P^-1 e",
            "prior_covariance": calibrated.P0.tolist(),
            "prior_measurement_cross_covariance": calibrated.C_pm.tolist(),
            "conditional_design": conditional_design.tolist(),
            "train_global_q_mean": global_mean.tolist(),
        },
        "models": records,
    }


def aggregate_gate(results):
    """Apply the predeclared engineering gate to completed split records."""
    summary = {}
    for corpus in sorted({row["corpus"] for row in results}):
        rows = [row for row in results if row["corpus"] == corpus]
        block = np.asarray([
            row["models"]["block_regional"]["held_joint_residual_nll_per_scalar"]
            for row in rows
        ])
        separable = np.asarray([
            row["models"]["separable_regional"]["held_joint_residual_nll_per_scalar"]
            for row in rows
        ])
        dense = np.asarray([
            row["models"]["dense_lmc_regional"]["held_joint_residual_nll_per_scalar"]
            for row in rows
        ])
        dense_gain, separable_gain = block - dense, block - separable
        global_block = np.asarray([
            row["models"]["block_global"]["held_joint_residual_nll_per_scalar"]
            for row in rows
        ])
        regional_gain = global_block - block

        def secondary_gain(model, path):
            values = []
            for row in rows:
                baseline = row["models"]["block_regional"]
                candidate = row["models"][model]
                for key in path:
                    baseline, candidate = baseline[key], candidate[key]
                values.append(float(baseline) - float(candidate))
            return np.asarray(values)

        separable_posterior = secondary_gain(
            "separable_regional", ("posterior", "state", "mean_bivariate_nll")
        )
        separable_log_loss = secondary_gain(
            "separable_regional", ("posterior", "decision", "log_loss")
        )
        separable_aurc = secondary_gain(
            "separable_regional", ("posterior", "decision", "aurc_margin")
        )
        loading = np.asarray([
            value
            for row in rows
            for model_name in MODEL_NAMES
            for value in (
                row["models"][model_name]["posterior"]["conditioner"]["loading"]
                ["relative_diagonal_loading"],
                row["models"][model_name]["posterior"]["conditioner"].get(
                    "prior_loading", {"relative_diagonal_loading": 0.0}
                )["relative_diagonal_loading"],
            )
        ])
        summary[corpus] = {
            "split_seeds": [row["seed"] for row in rows],
            "block_nll_mean_sd": [float(block.mean()), float(block.std(ddof=1)) if len(block) > 1 else 0.0],
            "dense_gain_mean_sd": [
                float(dense_gain.mean()), float(dense_gain.std(ddof=1)) if len(rows) > 1 else 0.0,
            ],
            "dense_positive_seeds": int(np.sum(dense_gain > 0)),
            "separable_gain_mean_sd": [
                float(separable_gain.mean()),
                float(separable_gain.std(ddof=1)) if len(rows) > 1 else 0.0,
            ],
            "separable_positive_seeds": int(np.sum(separable_gain > 0)),
            "regional_mean_gain_vs_global_block_mean_sd": [
                float(regional_gain.mean()),
                float(regional_gain.std(ddof=1)) if len(rows) > 1 else 0.0,
            ],
            "regional_mean_positive_seeds": int(np.sum(regional_gain > 0)),
            "separable_posterior_nll_gain_mean": float(separable_posterior.mean()),
            "separable_posterior_nll_positive_seeds": int(np.sum(separable_posterior > 0)),
            "separable_decision_log_loss_gain_mean": float(separable_log_loss.mean()),
            "separable_decision_aurc_gain_mean": float(separable_aurc.mean()),
            "maximum_relative_diagonal_loading": float(np.max(loading)),
        }
    expected_corpora = {"exploratory", "fresh"}
    expected_seeds = set(range(10))
    complete_primary_run = set(summary) == expected_corpora and all(
        len(value["split_seeds"]) == 10
        and set(value["split_seeds"]) == expected_seeds
        for value in summary.values()
    )
    required = 8 if complete_primary_run else None
    dense_ok = required is not None and all(
        value["dense_gain_mean_sd"][0] > 0 and value["dense_positive_seeds"] >= required
        for value in summary.values()
    )
    separable_ok = required is not None and all(
        value["separable_gain_mean_sd"][0] > 0 and value["separable_positive_seeds"] >= required
        for value in summary.values()
    )
    dense_gain = sum(value["dense_gain_mean_sd"][0] for value in summary.values())
    separable_gain = sum(value["separable_gain_mean_sd"][0] for value in summary.values())
    macro_split_recovery = separable_gain / dense_gain if dense_gain > 0 else None
    posterior_guardrail = all(
        value["separable_posterior_nll_gain_mean"] >= 0.0 for value in summary.values()
    )
    decision_guardrail = all(
        value["separable_decision_log_loss_gain_mean"] >= -0.01
        and value["separable_decision_aurc_gain_mean"] >= -0.01
        for value in summary.values()
    )
    loading_guardrail = all(
        value["maximum_relative_diagonal_loading"] <= 1e-3 for value in summary.values()
    )
    statistical_gate = (
        dense_ok
        and separable_ok
        and macro_split_recovery is not None
        and macro_split_recovery >= 0.80
        and posterior_guardrail
        and decision_guardrail
        and loading_guardrail
    )
    return {
        "by_corpus": summary,
        "gate_evaluable": required is not None,
        "dense_lmc_passes_direction_and_stability": dense_ok,
        "separable_passes_direction_and_stability": separable_ok,
        "macro_split_separable_fraction_of_dense_gain": macro_split_recovery,
        "passes_80_percent_recovery": (
            macro_split_recovery is not None and macro_split_recovery >= 0.80
        ),
        "separable_posterior_nll_guardrail_passes": posterior_guardrail,
        "separable_decision_guardrail_passes": decision_guardrail,
        "loading_budget_guardrail_passes": loading_guardrail,
        "structured_covariance_gate_passes": statistical_gate,
        "recommendation": (
            "proceed to optimized inverse-root/CUDA work"
            if statistical_gate else
            "retain block covariance; do not optimize structured inverse-root/CUDA from this evidence"
        ),
        "note": (
            "10-seed direction, stability, posterior, decision, and loading guardrails evaluated"
            if required is not None else
            "fewer than 10 seeds: smoke/descriptive result only; predeclared 8/10 gate not evaluated"
        ),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo", default=os.path.abspath(os.path.join(ROOT, "..", "..")))
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "model_prod_namecond.pt"))
    parser.add_argument("--campaign", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--luna", default=DEFAULT_LUNA)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--held-node-fraction", type=float, default=0.40)
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
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--out", default="/tmp/structured_residual_covariance.json")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    torch.set_num_threads(args.cpu_threads)
    np.random.seed(0)
    artifacts = configure_artifact_repo(args.artifact_repo)
    target_by_pair, cur_rel = load_decision_targets(args.campaign)
    luna_pairs, luna_d, luna_s = load_scored_mu_tsv(args.luna)
    luna_by_pair = {pair: (luna_d[i], luna_s[i]) for i, pair in enumerate(luna_pairs)}
    checkpoint = load_expanded(args.ckpt, dev="cpu")
    checkpoint[0].eval()
    datasets = load_campaign_datasets(campaign_scored=args.campaign)
    results = []
    for seed in range(args.seeds):
        for name, dataset in datasets.items():
            print(f"\n=== structured residual covariance: {name}, split seed {seed} ===", flush=True)
            row = run_corpus(name, dataset, target_by_pair, luna_by_pair, checkpoint, seed, args)
            results.append(row)
            for model_name in MODEL_NAMES:
                model = row["models"][model_name]
                print(
                    f"  {model_name:22s} q-NLL/scalar "
                    f"{model['held_joint_residual_nll_per_scalar']:+.6f}; "
                    f"posterior NLL {model['posterior']['state']['mean_bivariate_nll']:+.6f}; "
                    f"decision logloss {model['posterior']['decision']['log_loss']:.6f}",
                    flush=True,
                )
    payload = {
        "protocol": "structured conditional residual covariance evidence gate; strict node-disjoint; train-only fitting",
        "target_scope": "GPT-5.5 operating-judge fidelity; not independent ground truth",
        "inputs": {
            "checkpoint": os.path.abspath(args.ckpt),
            "checkpoint_sha256": sha256_file(args.ckpt),
            "campaign": os.path.abspath(args.campaign),
            "campaign_sha256": sha256_file(args.campaign),
            "luna": os.path.abspath(args.luna),
            "luna_sha256": sha256_file(args.luna),
            "artifact_paths": artifacts,
            "e5_caches": {
                "exploratory": file_provenance(CAMPAIGN_E5_100K),
                "fresh": file_provenance(DATASETS["fresh"]["e5_cache"]),
            },
            "campaign_cur_rel_counts": dict(cur_rel),
        },
        "configuration": vars(args),
        "results": results,
        "engineering_gate": aggregate_gate(results),
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"\nwrote {args.out}")
    print(json.dumps(payload["engineering_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
