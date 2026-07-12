#!/usr/bin/env python3
"""Versioned v2 family-wise calibration for the synthetic covariance selector.

This executable preserves the invalidated v1 output record, not an exact v1 runner: the shared synthetic
runner was later amended to use the corrected deranged wrong geometry.  V2 compares a conditional fixed-path
null against a full block-null procedure that reruns regional KRR and block fitting, audits mean/search
multiplicity, and evaluates two exploratory capacities: v2A's complete grid and v2B's canonical-geometry
alpha grid.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import time

import numpy as np

from covariance_sensitivity import (
    finite_null_maximum_threshold,
    maximum_eligible_macro_gain,
    select_nested_candidate,
    select_nested_candidate_null_calibrated,
)
from run_covariance_sensitivity_synthetic import (
    ALPHAS,
    BETAS,
    CHANNELS,
    MULTIPLIERS,
    NLLScorer,
    RIDGES,
    SCENARIOS,
    SCENARIO_ORDER,
    SyntheticGeometry,
    _candidate_covariance,
    _fit_block_covariance,
    _fit_partition,
    _jsonable,
    _posterior_mse,
    _summary,
)
from structured_residual_covariance import gaussian_joint_nll


SEARCH_FULL = "full_216"
SEARCH_CANONICAL = "canonical_alpha_8"
SEARCH_SINGLE = "canonical_alpha1_single"
SEARCHES = (SEARCH_FULL, SEARCH_CANONICAL, SEARCH_SINGLE)
SELECTOR_V2A = "v2A_full_grid_familywise"
SELECTOR_V2B = "v2B_canonical_alpha_familywise"
SELECTOR_SEARCH = {
    SELECTOR_V2A: SEARCH_FULL,
    SELECTOR_V2B: SEARCH_CANONICAL,
}


def _block_scenario():
    return next(value for value in SCENARIOS if value.name == "block_null")


def _fixed_path_scorers(geometry, standardized_field):
    scorers = []
    for fold in range(3):
        partition_name = f"inner_{fold}"
        _, held = geometry.partition_map[partition_name]
        scorers.append((
            fold,
            partition_name,
            NLLScorer(standardized_field[held], np.eye(CHANNELS)),
            {
                "fold": fold,
                "mean_mode": "fixed_identity_held",
                "fit_items": None,
                "held_items": len(held),
            },
        ))
    return scorers


def _intercept_fit(field, fit, held):
    train = field[fit]
    intercept = np.mean(train, axis=0)
    # Exact residual from predicting each train row with the mean of the other rows.
    loo_residuals = len(train) / (len(train) - 1.0) * (train - intercept)
    prediction = np.repeat(intercept[None, :], len(held), axis=0)
    return loo_residuals, field[held] - prediction, {
        "kernel_name": "intercept_only",
        "ridge": None,
        "loo_mse": float(np.mean(loo_residuals * loo_residuals)),
    }


def _procedure_scorers(geometry, field, shrinkage, mean_mode):
    scorers = []
    for fold in range(3):
        partition_name = f"inner_{fold}"
        fit, held = geometry.partition_map[partition_name]
        if mean_mode == "regional_krr":
            _, _, regional, block, centered = _fit_partition(
                geometry, partition_name, field, shrinkage
            )
            mean_record = {
                "kernel_name": regional.kernel_name,
                "ridge": regional.ridge,
                "loo_mse": regional.loo_mse,
            }
        elif mean_mode == "intercept_only":
            loo, centered, mean_record = _intercept_fit(field, fit, held)
            block = _fit_block_covariance(loo, shrinkage)
        elif mean_mode == "oracle_zero":
            block = _fit_block_covariance(field[fit], shrinkage)
            centered = field[held]
            mean_record = {
                "kernel_name": "oracle_zero",
                "ridge": None,
                "loo_mse": float(np.mean(field[fit] * field[fit])),
            }
        else:
            raise ValueError(f"unknown mean mode: {mean_mode}")
        scorers.append((
            fold,
            partition_name,
            NLLScorer(centered, block),
            {
                "fold": fold,
                "mean_mode": mean_mode,
                "fit_items": len(fit),
                "held_items": len(held),
                **mean_record,
            },
        ))
    return scorers


def _full_records(geometry, scenario, scorers):
    records = []
    for fold, partition_name, scorer, _ in scorers:
        records.append({
            "fold": fold,
            "alpha": 0.0,
            "semantic_multiplier": 1.0,
            "graph_multiplier": 1.0,
            "beta": 1.0,
            "nll_per_scalar": float(scorer.block_nll),
        })
        for semantic_multiplier in MULTIPLIERS:
            for graph_multiplier in MULTIPLIERS:
                for beta in BETAS:
                    endpoint = geometry.candidate_endpoint(
                        scenario,
                        partition_name,
                        semantic_multiplier,
                        graph_multiplier,
                        beta,
                    )
                    path_scores = scorer.score_path(endpoint)
                    for alpha in ALPHAS[1:]:
                        records.append({
                            "fold": fold,
                            "alpha": float(alpha),
                            "semantic_multiplier": float(semantic_multiplier),
                            "graph_multiplier": float(graph_multiplier),
                            "beta": float(beta),
                            "nll_per_scalar": path_scores[alpha],
                        })
    return records


def _filter_search(records, search):
    if search == SEARCH_FULL:
        return records
    output = []
    for row in records:
        if row["alpha"] == 0.0:
            output.append(row)
            continue
        canonical = (
            row["semantic_multiplier"] == 1.0
            and row["graph_multiplier"] == 1.0
            and row["beta"] == 0.0
        )
        if canonical and (search == SEARCH_CANONICAL or row["alpha"] == 1.0):
            output.append(row)
    if search not in {SEARCH_CANONICAL, SEARCH_SINGLE}:
        raise ValueError(f"unknown search: {search}")
    return output


def _v1_statistics(records):
    selected, summaries = select_nested_candidate(records)
    maximum = maximum_eligible_macro_gain(summaries)
    return selected, summaries, maximum


def _threshold_ratio(numerator, denominator):
    """Return a finite diagnostic ratio, or ``None`` for a zero baseline.

    A very small smoke calibration can legitimately produce an all-zero
    fixed-path null maximum.  The full/fixed threshold ratio is undefined in
    that case; it should not abort the run or be emitted as non-standard JSON
    infinity.
    """
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError("null thresholds must be finite")
    if numerator < 0.0 or denominator < 0.0:
        raise ValueError("null thresholds must be nonnegative")
    return None if denominator == 0.0 else numerator / denominator


def calibrate_nulls(geometry, *, draws, seed, shrinkage):
    """Paired conditional and full-procedure null maxima on shared global fields."""
    scenario = _block_scenario()
    maximum = {
        "fixed_path_shared_z": {search: [] for search in SEARCHES},
        "full_procedure_krr": {search: [] for search in SEARCHES},
    }
    selected_nonzero = {
        null: {search: 0 for search in SEARCHES} for null in maximum
    }
    block_factor = np.linalg.cholesky(geometry.block_covariance)
    for draw in range(draws):
        z = np.random.default_rng(seed + draw).standard_normal(
            (geometry.item_count, CHANNELS)
        )
        scorer_sets = {
            "fixed_path_shared_z": _fixed_path_scorers(geometry, z),
            "full_procedure_krr": _procedure_scorers(
                geometry, z @ block_factor.T, shrinkage, "regional_krr"
            ),
        }
        for null_name, scorers in scorer_sets.items():
            full = _full_records(geometry, scenario, scorers)
            for search in SEARCHES:
                selected, _, value = _v1_statistics(_filter_search(full, search))
                maximum[null_name][search].append(value)
                selected_nonzero[null_name][search] += int(selected["alpha"] > 0.0)
    arrays = {
        null: {search: np.asarray(values, dtype=float) for search, values in by_search.items()}
        for null, by_search in maximum.items()
    }
    report = {}
    for null_name, by_search in arrays.items():
        report[null_name] = {}
        for search, values in by_search.items():
            threshold, rank = finite_null_maximum_threshold(values)
            report[null_name][search] = {
                "draws": draws,
                "seed_start": seed,
                "v1_nonzero_selection_rate": selected_nonzero[null_name][search] / draws,
                "maximum_eligible_macro_gain": _summary(values),
                "upper_95_order_statistic_threshold": threshold,
                "order_statistic_rank_one_based": rank,
            }
    report["full_vs_fixed_threshold_ratio"] = {
        search: _threshold_ratio(
            report["full_procedure_krr"][search]["upper_95_order_statistic_threshold"],
            report["fixed_path_shared_z"][search]["upper_95_order_statistic_threshold"],
        )
        for search in SEARCHES
    }
    return arrays, report


def attribution_audit(geometry, *, draws, seed, shrinkage):
    """Cross mean capacity with search size using common block-null fields."""
    scenario = _block_scenario()
    modes = ("regional_krr", "intercept_only", "oracle_zero")
    maxima = {mode: {search: [] for search in SEARCHES} for mode in modes}
    nonzero = {mode: {search: 0 for search in SEARCHES} for mode in modes}
    block_factor = np.linalg.cholesky(geometry.block_covariance)
    for draw in range(draws):
        z = np.random.default_rng(seed + draw).standard_normal(
            (geometry.item_count, CHANNELS)
        )
        field = z @ block_factor.T
        for mode in modes:
            full = _full_records(
                geometry,
                scenario,
                _procedure_scorers(geometry, field, shrinkage, mode),
            )
            for search in SEARCHES:
                selected, _, value = _v1_statistics(_filter_search(full, search))
                maxima[mode][search].append(value)
                nonzero[mode][search] += int(selected["alpha"] > 0.0)
    report = {}
    for mode in modes:
        report[mode] = {}
        for search in SEARCHES:
            values = np.asarray(maxima[mode][search], dtype=float)
            report[mode][search] = {
                "draws": draws,
                "seed_start": seed,
                "v1_nonzero_selection_rate": nonzero[mode][search] / draws,
                "maximum_eligible_macro_gain": _summary(values),
            }
    report["contrasts"] = {
        "full_grid_krr_minus_oracle_zero_nonzero_rate": (
            report["regional_krr"][SEARCH_FULL]["v1_nonzero_selection_rate"]
            - report["oracle_zero"][SEARCH_FULL]["v1_nonzero_selection_rate"]
        ),
        "krr_full_grid_minus_canonical_grid_nonzero_rate": (
            report["regional_krr"][SEARCH_FULL]["v1_nonzero_selection_rate"]
            - report["regional_krr"][SEARCH_CANONICAL]["v1_nonzero_selection_rate"]
        ),
        "krr_canonical_grid_minus_single_nonzero_rate": (
            report["regional_krr"][SEARCH_CANONICAL]["v1_nonzero_selection_rate"]
            - report["regional_krr"][SEARCH_SINGLE]["v1_nonzero_selection_rate"]
        ),
    }
    return report


def _outer_common(geometry, scenario, conditional_residual, innovation, state, shrinkage):
    _, evaluate, regional, block, centered = _fit_partition(
        geometry, "outer", conditional_residual, shrinkage
    )
    scorer = NLLScorer(centered, block)
    block_covariance = np.kron(np.eye(len(evaluate)), block)
    grid = [{
        "alpha": 0.0,
        "semantic_multiplier": 1.0,
        "graph_multiplier": 1.0,
        "beta": 1.0,
        "nll_per_scalar": float(scorer.block_nll),
    }]
    for semantic_multiplier in MULTIPLIERS:
        for graph_multiplier in MULTIPLIERS:
            for beta in BETAS:
                endpoint = geometry.candidate_endpoint(
                    scenario, "outer", semantic_multiplier, graph_multiplier, beta
                )
                path_scores = scorer.score_path(endpoint)
                for alpha in ALPHAS[1:]:
                    grid.append({
                        "alpha": float(alpha),
                        "semantic_multiplier": float(semantic_multiplier),
                        "graph_multiplier": float(graph_multiplier),
                        "beta": float(beta),
                        "nll_per_scalar": path_scores[alpha],
                    })
    oracle = min(grid, key=lambda row: row["nll_per_scalar"])
    oracle_endpoint = geometry.candidate_endpoint(
        scenario,
        "outer",
        oracle["semantic_multiplier"],
        oracle["graph_multiplier"],
        oracle["beta"],
    )
    oracle_covariance = _candidate_covariance(
        geometry, block, oracle_endpoint, oracle["alpha"]
    )
    common_posterior = {
        "block": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, block_covariance
        ),
        "outer_nll_oracle": _posterior_mse(
            geometry, innovation, state, evaluate, regional.prediction, oracle_covariance
        ),
    }
    return {
        "evaluate": evaluate,
        "regional": regional,
        "block": block,
        "centered": centered,
        "scorer": scorer,
        "block_covariance": block_covariance,
        "block_nll": scorer.block_nll,
        "oracle": oracle,
        "common_posterior": common_posterior,
    }


def run_replicate_pair(
    geometry,
    scenario,
    replicate,
    seed,
    shrinkage,
    full_procedure_null_maxima,
):
    conditional_residual, innovation, state = geometry.draw(scenario, seed + replicate)
    inner_scorers = _procedure_scorers(
        geometry, conditional_residual, shrinkage, "regional_krr"
    )
    full_records = _full_records(geometry, scenario, inner_scorers)
    selections = {}
    summaries = {}
    calibrations = {}
    for selector, search in SELECTOR_SEARCH.items():
        records = _filter_search(full_records, search)
        selections[selector], summaries[selector], calibrations[selector] = (
            select_nested_candidate_null_calibrated(
                records, full_procedure_null_maxima[search]
            )
        )
        selections[selector]["selector_capacity"] = selector
        calibrations[selector]["null_type"] = "full_procedure_krr"
        calibrations[selector]["candidate_search"] = search
    outer = _outer_common(
        geometry, scenario, conditional_residual, innovation, state, shrinkage
    )
    records_by_selector = {}
    for selector, selection in selections.items():
        endpoint = geometry.candidate_endpoint(
            scenario,
            "outer",
            selection["semantic_multiplier"],
            selection["graph_multiplier"],
            selection["beta"],
        )
        covariance = _candidate_covariance(
            geometry, outer["block"], endpoint, selection["alpha"]
        )
        selected_nll = (
            outer["block_nll"]
            if selection["alpha"] == 0.0
            else outer["scorer"].score_path(endpoint)[selection["alpha"]]
        )
        posterior = dict(outer["common_posterior"])
        posterior["selected"] = _posterior_mse(
            geometry,
            innovation,
            state,
            outer["evaluate"],
            outer["regional"].prediction,
            covariance,
        )
        records_by_selector[selector] = {
            "replicate": int(replicate),
            "seed": int(seed + replicate),
            "selection": selection,
            "selection_calibration": calibrations[selector],
            "eligible_nonzero_candidates": int(sum(
                row["alpha"] > 0.0
                and row["macro_gain_vs_block"] > 0.0
                and row["positive_folds"] >= 2
                for row in summaries[selector]
            )),
            "inner_regional_means": [row[3] for row in inner_scorers],
            "outer_regional_mean": {
                "kernel_name": outer["regional"].kernel_name,
                "ridge": outer["regional"].ridge,
                "loo_mse": outer["regional"].loo_mse,
            },
            "outer_items": int(len(outer["evaluate"])),
            "measurement_dimension": int(CHANNELS * len(outer["evaluate"])),
            "residual_nll_per_scalar": {
                "block": float(outer["block_nll"]),
                "selected": float(selected_nll),
                "outer_grid_oracle": float(outer["oracle"]["nll_per_scalar"]),
            },
            "outer_grid_oracle": {
                **outer["oracle"],
                "scope": "full length/channel/alpha sensitivity grid; never deploys v2B",
            },
            "posterior_state_mse": posterior,
            "truth_scope": (
                "end-to-end fitted KRR: no known-truth recovery denominator; centered residual is "
                "e_H-W e_T and W is outcome-selected"
            ),
        }
    return records_by_selector


def _mean_gain_ratio(numerator, denominator):
    denominator = float(np.mean(denominator))
    return None if denominator <= 0.0 else float(np.mean(numerator)) / denominator


def aggregate_end_to_end(scenario, records):
    """Valid fitted-KRR comparisons; deliberately contains no pre-KRR truth denominator."""
    alphas = np.asarray([row["selection"]["alpha"] for row in records])
    block_nll = np.asarray([row["residual_nll_per_scalar"]["block"] for row in records])
    selected_nll = np.asarray([
        row["residual_nll_per_scalar"]["selected"] for row in records
    ])
    oracle_nll = np.asarray([
        row["residual_nll_per_scalar"]["outer_grid_oracle"] for row in records
    ])
    block_mse = np.asarray([row["posterior_state_mse"]["block"] for row in records])
    selected_mse = np.asarray([
        row["posterior_state_mse"]["selected"] for row in records
    ])
    oracle_mse = np.asarray([
        row["posterior_state_mse"]["outer_nll_oracle"] for row in records
    ])
    block_rate = float(np.mean(alphas == 0.0))
    gain = block_nll - selected_nll
    criterion = {
        "type": "end_to_end_power_descriptive",
        "pass": None,
        "reason": (
            "known-covariance recovery is evaluated only in the oracle-mean/known-B mechanism track"
        ),
    }
    if scenario.name == "wrong_item_geometry":
        criterion = {
            "type": "common_mode_contaminated_wrong_geometry_diagnostic",
            "pass": None,
            "reason": (
                "permutation preserves the dense positive RBF common mode, so this is not an orthogonal "
                "block-selection negative control"
            ),
        }
    elif scenario.name in {"block_null", "regional_mean_only"}:
        harm = float(np.mean(selected_nll - block_nll))
        criterion = {
            "type": "end_to_end_null_or_misspecified_geometry_control",
            "requirements": [
                "mean held harm <= 0.001 NLL/scalar",
                "block selected in at least 80% of replicates",
            ],
            "mean_held_harm_nll_per_scalar": harm,
            "block_selection_rate": block_rate,
            "harm_requirement_pass": bool(harm <= 0.001),
            "selection_requirement_pass": bool(block_rate >= 0.80),
            "pass": bool(harm <= 0.001 and block_rate >= 0.80),
        }
    return {
        "scenario": scenario.name,
        "replicates": len(records),
        "selection": {
            "block_rate": block_rate,
            "nonzero_alpha_rate": float(np.mean(alphas > 0.0)),
            "alpha_counts": dict(Counter(map(str, alphas.tolist()))),
        },
        "residual_nll_per_scalar": {
            "block": _summary(block_nll),
            "selected": _summary(selected_nll),
            "outer_grid_oracle": _summary(oracle_nll),
            "selected_gain_vs_block": _summary(gain),
            "outer_grid_oracle_gain_vs_block": _summary(block_nll - oracle_nll),
        },
        "posterior_state_mse": {
            "block": _summary(block_mse),
            "selected": _summary(selected_mse),
            "outer_nll_grid_oracle": _summary(oracle_mse),
            "selected_gain_vs_block": _summary(block_mse - selected_mse),
        },
        "preregistered_criterion": criterion,
        "truth_scope": records[0]["truth_scope"],
    }


def _known_block_scorers(geometry, field):
    scorers = []
    for fold in range(3):
        partition_name = f"inner_{fold}"
        _, held = geometry.partition_map[partition_name]
        scorers.append((
            fold,
            partition_name,
            NLLScorer(field[held], geometry.block_covariance),
            {
                "fold": fold,
                "mean_mode": "oracle_zero",
                "block_mode": "known_true_B",
                "fit_items": None,
                "held_items": len(held),
            },
        ))
    return scorers


def run_mechanism_replicate_pair(
    geometry,
    scenario,
    replicate,
    seed,
    fixed_path_null_maxima,
):
    """Oracle-zero-mean, known-B power mechanism where R_HH is the scored truth."""
    field, innovation, state = geometry.draw(scenario, seed + replicate)
    if scenario.regional_mean:
        raise ValueError("regional-mean-only is not an oracle-zero-mean mechanism scenario")
    inner = _known_block_scorers(geometry, field)
    full_records = _full_records(geometry, scenario, inner)
    selections, summaries, calibrations = {}, {}, {}
    for selector, search in SELECTOR_SEARCH.items():
        selections[selector], summaries[selector], calibrations[selector] = (
            select_nested_candidate_null_calibrated(
                _filter_search(full_records, search), fixed_path_null_maxima[search]
            )
        )
        selections[selector]["selector_capacity"] = selector
        calibrations[selector]["null_type"] = "fixed_path_shared_z_known_B"
        calibrations[selector]["candidate_search"] = search

    evaluate = geometry.outer_held
    held = field[evaluate]
    scorer = NLLScorer(held, geometry.block_covariance)
    block_covariance = np.kron(np.eye(len(evaluate)), geometry.block_covariance)
    grid = [{
        "alpha": 0.0,
        "semantic_multiplier": 1.0,
        "graph_multiplier": 1.0,
        "beta": 1.0,
        "nll_per_scalar": float(scorer.block_nll),
    }]
    for semantic_multiplier in MULTIPLIERS:
        for graph_multiplier in MULTIPLIERS:
            for beta in BETAS:
                endpoint = geometry.candidate_endpoint(
                    scenario, "outer", semantic_multiplier, graph_multiplier, beta
                )
                scores = scorer.score_path(endpoint)
                for alpha in ALPHAS[1:]:
                    grid.append({
                        "alpha": float(alpha),
                        "semantic_multiplier": float(semantic_multiplier),
                        "graph_multiplier": float(graph_multiplier),
                        "beta": float(beta),
                        "nll_per_scalar": scores[alpha],
                    })
    oracle = min(grid, key=lambda row: row["nll_per_scalar"])
    oracle_endpoint = geometry.candidate_endpoint(
        scenario,
        "outer",
        oracle["semantic_multiplier"],
        oracle["graph_multiplier"],
        oracle["beta"],
    )
    oracle_covariance = _candidate_covariance(
        geometry, geometry.block_covariance, oracle_endpoint, oracle["alpha"]
    )
    true_covariance = geometry.true_covariance(scenario, evaluate)
    true_nll = gaussian_joint_nll(held, true_covariance).per_scalar
    zeros = np.zeros_like(held)
    common_posterior = {
        "block": _posterior_mse(
            geometry, innovation, state, evaluate, zeros, block_covariance
        ),
        "outer_nll_oracle": _posterior_mse(
            geometry, innovation, state, evaluate, zeros, oracle_covariance
        ),
        "known_true_covariance": _posterior_mse(
            geometry, innovation, state, evaluate, zeros, true_covariance
        ),
    }
    output = {}
    for selector, selection in selections.items():
        endpoint = geometry.candidate_endpoint(
            scenario,
            "outer",
            selection["semantic_multiplier"],
            selection["graph_multiplier"],
            selection["beta"],
        )
        covariance = _candidate_covariance(
            geometry, geometry.block_covariance, endpoint, selection["alpha"]
        )
        selected_nll = (
            scorer.block_nll
            if selection["alpha"] == 0.0
            else scorer.score_path(endpoint)[selection["alpha"]]
        )
        posterior = dict(common_posterior)
        posterior["selected"] = _posterior_mse(
            geometry, innovation, state, evaluate, zeros, covariance
        )
        output[selector] = {
            "replicate": int(replicate),
            "seed": int(seed + replicate),
            "selection": selection,
            "selection_calibration": calibrations[selector],
            "eligible_nonzero_candidates": int(sum(
                row["alpha"] > 0.0
                and row["macro_gain_vs_block"] > 0.0
                and row["positive_folds"] >= 2
                for row in summaries[selector]
            )),
            "residual_nll_per_scalar": {
                "block": float(scorer.block_nll),
                "selected": float(selected_nll),
                "outer_grid_oracle": float(oracle["nll_per_scalar"]),
                "known_true_covariance": float(true_nll),
            },
            "outer_grid_oracle": oracle,
            "posterior_state_mse": posterior,
            "truth_scope": "oracle zero mean and known B; R_HH is the exact scored covariance",
        }
    return output


def aggregate_mechanism(scenario, records):
    alphas = np.asarray([row["selection"]["alpha"] for row in records])
    block_nll = np.asarray([row["residual_nll_per_scalar"]["block"] for row in records])
    selected_nll = np.asarray([
        row["residual_nll_per_scalar"]["selected"] for row in records
    ])
    oracle_nll = np.asarray([
        row["residual_nll_per_scalar"]["outer_grid_oracle"] for row in records
    ])
    true_nll = np.asarray([
        row["residual_nll_per_scalar"]["known_true_covariance"] for row in records
    ])
    block_mse = np.asarray([row["posterior_state_mse"]["block"] for row in records])
    selected_mse = np.asarray([
        row["posterior_state_mse"]["selected"] for row in records
    ])
    true_mse = np.asarray([
        row["posterior_state_mse"]["known_true_covariance"] for row in records
    ])
    selected_nll_gain = block_nll - selected_nll
    true_nll_gain = block_nll - true_nll
    selected_mse_gain = block_mse - selected_mse
    true_mse_gain = block_mse - true_mse
    nll_recovery = _mean_gain_ratio(selected_nll_gain, true_nll_gain)
    mse_recovery = _mean_gain_ratio(selected_mse_gain, true_mse_gain)
    block_rate = float(np.mean(alphas == 0.0))
    criterion = {"type": "measured_power_only", "pass": None}
    if scenario.name == "wrong_item_geometry":
        criterion = {
            "type": "common_mode_contaminated_wrong_geometry_diagnostic",
            "pass": None,
            "reason": (
                "P K P.T preserves the dense positive RBF common mode; candidate and truth remain "
                "covariance-aligned despite wrong pair identities"
            ),
        }
    elif scenario.name == "block_null":
        harm = float(np.mean(selected_nll - block_nll))
        criterion = {
            "type": "mechanism_null_or_wrong_geometry_control",
            "mean_held_harm_nll_per_scalar": harm,
            "block_selection_rate": block_rate,
            "harm_requirement_pass": bool(harm <= 0.001),
            "selection_requirement_pass": bool(block_rate >= 0.80),
            "pass": bool(harm <= 0.001 and block_rate >= 0.80),
        }
    elif scenario.true_coupling >= 0.10:
        criterion = {
            "type": "oracle_mean_known_B_power_control",
            "nonzero_alpha_selection_rate": float(np.mean(alphas > 0.0)),
            "residual_nll_recovery_fraction": nll_recovery,
            "posterior_risk_recovery_fraction": mse_recovery,
            "selection_requirement_pass": bool(np.mean(alphas > 0.0) >= 0.80),
            "residual_recovery_requirement_pass": bool(
                nll_recovery is not None and nll_recovery >= 0.50
            ),
            "posterior_recovery_requirement_pass": bool(
                mse_recovery is not None and mse_recovery >= 0.50
            ),
        }
        criterion["pass"] = bool(
            criterion["selection_requirement_pass"]
            and criterion["residual_recovery_requirement_pass"]
            and criterion["posterior_recovery_requirement_pass"]
        )
    return {
        "scenario": scenario.name,
        "replicates": len(records),
        "selection": {
            "block_rate": block_rate,
            "nonzero_alpha_rate": float(np.mean(alphas > 0.0)),
            "alpha_counts": dict(Counter(map(str, alphas.tolist()))),
        },
        "residual_nll_per_scalar": {
            "block": _summary(block_nll),
            "selected": _summary(selected_nll),
            "outer_grid_oracle": _summary(oracle_nll),
            "known_true_covariance": _summary(true_nll),
            "selected_gain_vs_block": _summary(selected_nll_gain),
            "known_true_covariance_gain_vs_block": _summary(true_nll_gain),
            "selected_recovery_fraction_of_known_truth_mean_gain": nll_recovery,
        },
        "posterior_state_mse": {
            "block": _summary(block_mse),
            "selected": _summary(selected_mse),
            "known_true_covariance": _summary(true_mse),
            "selected_gain_vs_block": _summary(selected_mse_gain),
            "known_true_covariance_gain_vs_block": _summary(true_mse_gain),
            "selected_recovery_fraction_of_known_truth_mean_gain": mse_recovery,
        },
        "preregistered_criterion": criterion,
        "truth_scope": records[0]["truth_scope"],
    }


def _calibration_rejection_summary(records):
    observed = np.asarray([
        row["selection_calibration"]["observed_maximum_eligible_macro_gain"]
        for row in records
    ])
    rejected = np.asarray([
        row["selection_calibration"]["strictly_exceeds_threshold"] for row in records
    ], dtype=bool)
    return {
        "familywise_rejection_rate": float(np.mean(rejected)),
        "observed_maximum_eligible_macro_gain": _summary(observed),
        "threshold": records[0]["selection_calibration"][
            "familywise_macro_gain_threshold"
        ],
    }


def _complete_control_gate(required_scenarios, selected_scenarios, decisions):
    """Never let an empty or partial scenario run pass by vacuous truth."""
    evaluable = set(required_scenarios).issubset(set(selected_scenarios))
    return evaluable, bool(evaluable and decisions and all(decisions))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--calibration-draws", type=int, default=2000)
    parser.add_argument("--audit-draws", type=int, default=1000)
    parser.add_argument("--items", type=int, default=96)
    parser.add_argument("--outer-held-items", type=int, default=32)
    parser.add_argument("--inner-held-items", type=int, default=22)
    parser.add_argument("--evaluation-seed", type=int, default=881000)
    parser.add_argument("--calibration-seed", type=int, default=991000)
    parser.add_argument("--audit-seed", type=int, default=994000)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIO_ORDER, default=list(SCENARIO_ORDER)
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--out", default="/tmp/covariance_sensitivity_synthetic_v2.json")
    return parser


def _validate_args(args):
    if args.replicates < 1 or args.calibration_draws < 1 or args.audit_draws < 1:
        raise ValueError("replicate, calibration, and audit counts must be positive")
    if not 0.0 <= args.shrinkage <= 1.0:
        raise ValueError("--shrinkage must be in [0,1]")


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
    started = time.perf_counter()
    geometry = SyntheticGeometry(
        args.items, args.outer_held_items, args.inner_held_items
    )
    null_arrays, null_report = calibrate_nulls(
        geometry,
        draws=args.calibration_draws,
        seed=args.calibration_seed,
        shrinkage=args.shrinkage,
    )
    audit = attribution_audit(
        geometry,
        draws=args.audit_draws,
        seed=args.audit_seed,
        shrinkage=args.shrinkage,
    )
    scenario_by_name = {scenario.name: scenario for scenario in SCENARIOS}
    end_records = {selector: {} for selector in SELECTOR_SEARCH}
    end_aggregates = {selector: {} for selector in SELECTOR_SEARCH}
    mechanism_records = {selector: {} for selector in SELECTOR_SEARCH}
    mechanism_aggregates = {selector: {} for selector in SELECTOR_SEARCH}
    full_null = null_arrays["full_procedure_krr"]
    fixed_null = null_arrays["fixed_path_shared_z"]
    for name in args.scenarios:
        scenario = scenario_by_name[name]
        scenario_index = SCENARIO_ORDER.index(name)
        paired = [
            run_replicate_pair(
                geometry,
                scenario,
                replicate,
                args.evaluation_seed + 100000 * scenario_index,
                args.shrinkage,
                full_null,
            )
            for replicate in range(args.replicates)
        ]
        for selector in SELECTOR_SEARCH:
            selector_records = [row[selector] for row in paired]
            end_records[selector][name] = selector_records
            end_aggregates[selector][name] = aggregate_end_to_end(
                scenario, selector_records
            )
            end_aggregates[selector][name]["familywise_calibration"] = (
                _calibration_rejection_summary(selector_records)
            )
        if not scenario.regional_mean:
            mechanism_paired = [
                run_mechanism_replicate_pair(
                    geometry,
                    scenario,
                    replicate,
                    args.evaluation_seed + 100000 * scenario_index,
                    fixed_null,
                )
                for replicate in range(args.replicates)
            ]
            for selector in SELECTOR_SEARCH:
                selector_records = [row[selector] for row in mechanism_paired]
                mechanism_records[selector][name] = selector_records
                mechanism_aggregates[selector][name] = aggregate_mechanism(
                    scenario, selector_records
                )
                mechanism_aggregates[selector][name]["familywise_calibration"] = (
                    _calibration_rejection_summary(selector_records)
                )
    end_required, mechanism_required = {}, {}
    end_gate_scenarios = {"block_null", "regional_mean_only"}
    mechanism_gate_scenarios = {
        "block_null", "in_family_coupling_0.10", "in_family_coupling_0.20",
    }
    selected_scenarios = set(args.scenarios)
    end_gate_evaluable = end_gate_scenarios.issubset(selected_scenarios)
    mechanism_gate_evaluable = mechanism_gate_scenarios.issubset(selected_scenarios)
    for selector in SELECTOR_SEARCH:
        end_decisions = [
            row["preregistered_criterion"]["pass"]
            for row in end_aggregates[selector].values()
            if row["preregistered_criterion"]["pass"] is not None
        ]
        mechanism_decisions = [
            row["preregistered_criterion"]["pass"]
            for row in mechanism_aggregates[selector].values()
            if row["preregistered_criterion"]["pass"] is not None
        ]
        _, end_required[selector] = _complete_control_gate(
            end_gate_scenarios, selected_scenarios, end_decisions
        )
        _, mechanism_required[selector] = _complete_control_gate(
            mechanism_gate_scenarios, selected_scenarios, mechanism_decisions
        )
    payload = {
        "protocol": (
            "post-v1 v2 family-wise null calibration; v2A full-grid and v2B canonical-alpha; "
            "full block-null procedure is the deployment threshold"
        ),
        "design_addendum": "DESIGN_covariance_sensitivity_v2_null_calibration.md",
        "configuration": {
            "evaluation_replicates_per_scenario": args.replicates,
            "calibration_draws": args.calibration_draws,
            "attribution_audit_draws": args.audit_draws,
            "evaluation_seed": args.evaluation_seed,
            "calibration_seed": args.calibration_seed,
            "audit_seed": args.audit_seed,
            "requested_scenarios": list(args.scenarios),
            "block_covariance_shrinkage": args.shrinkage,
            "regional_mean_ridges": list(RIDGES),
            "semantic_base_length": geometry.semantic_length,
            "graph_base_length": geometry.graph_length,
            "geometry_feature_seed": 24051,
            "outer_partition_seed": 907,
            "inner_partition_seeds": [10000, 10001, 10002],
            "wrong_geometry": {
                "construction": "fixed derangement congruence P K_canonical P.T",
                "permutation_seed": 33517,
            },
            "items": args.items,
            "outer_train_items": len(geometry.outer_train),
            "outer_held_items": len(geometry.outer_held),
            "inner_fit_items": len(geometry.inner_partitions[0][0]),
            "inner_held_items": len(geometry.inner_partitions[0][1]),
            "alphas": list(ALPHAS),
            "length_multipliers": list(MULTIPLIERS),
            "channel_shrinkage_betas": list(BETAS),
            "familywise_confidence": 0.95,
        },
        "null_calibrations": null_report,
        "v1_failure_attribution_audit": audit,
        "end_to_end_krr": {
            "scope": (
                "selection and harm only; no pre-KRR true-covariance recovery denominator"
            ),
            "selector_aggregates": end_aggregates,
            "required_gate_scenarios": sorted(end_gate_scenarios),
            "gate_evaluable_by_selector": {
                selector: end_gate_evaluable for selector in SELECTOR_SEARCH
            },
            "all_applicable_controls_pass_by_selector": end_required,
        },
        "oracle_mean_known_B_mechanism": {
            "scope": (
                "oracle zero mean and known block marginal; exact R_HH recovery/power gate"
            ),
            "selector_aggregates": mechanism_aggregates,
            "required_gate_scenarios": sorted(mechanism_gate_scenarios),
            "gate_evaluable_by_selector": {
                selector: mechanism_gate_evaluable for selector in SELECTOR_SEARCH
            },
            "all_applicable_controls_pass_by_selector": mechanism_required,
        },
        "invalidated_outputs": [
            {
                "path": (
                    "/tmp/covariance_sensitivity_synthetic_v1_"
                    "INVALID_pre_krr_denominator_and_weak_wrong_geometry.json"
                ),
                "reason": (
                    "pre-KRR truth denominator plus weak narrow-RBF wrong geometry; exact old geometry "
                    "is not reproduced by the amended shared runner"
                ),
            },
            {
                "path": (
                    "/tmp/covariance_sensitivity_synthetic_v2_"
                    "INVALID_pre_krr_denominator.json"
                ),
                "reason": "pre-KRR R_HH was not the covariance of e_H-W e_T",
            },
        ],
        "replicate_records": None if args.summary_only else {
            "end_to_end_krr": end_records,
            "oracle_mean_known_B_mechanism": mechanism_records,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write_payload(args.out, payload)
    print(json.dumps({
        "output": os.path.abspath(args.out),
        "null_calibrations": null_report,
        "v1_failure_attribution_audit": audit,
        "end_to_end_krr": payload["end_to_end_krr"],
        "oracle_mean_known_B_mechanism": payload["oracle_mean_known_B_mechanism"],
        "wall_seconds": payload["wall_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
