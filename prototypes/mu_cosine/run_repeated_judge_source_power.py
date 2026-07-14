#!/usr/bin/env python3
"""Run the no-history Stage-A source-dependent repeated-judge power study.

The default/custom mode is diagnostic and always fails closed.  ``--full-prereg``
freezes the complete discovery grid and, only after a provisional smallest-G /
coarsest-K choice, a seed-disjoint fixed-pair confirmation.  Full runs are
expensive and are not part of the unit-test suite.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from repeated_judge_power import (  # noqa: E402
    Candidate,
    DEFAULT_GAMMAS,
    DEFAULT_MEAN_RIDGES,
    DEFAULT_RHOS,
    MAX_PROMPT_ROWS,
    PRIMARY_ENDPOINTS,
    SCENARIOS,
    SCENARIO_BY_NAME,
    derive_seed,
)
from repeated_judge_source_power import (  # noqa: E402
    REQUIRED_SOURCE_CORPORA,
    SOURCE_ETA_GRID,
    SourcePowerReplicate,
    aggregate_source_power_records,
    build_source_design,
    combine_source_corpus_null_maxima,
    combine_source_power_corpus_replicates,
    prepare_graph_aware_source_corpus_multiplier,
    run_source_corpus_power_replicate,
    source_atomic_component_splits,
    source_corpus_null_maximum,
    source_split_diagnostics,
)
from repeated_judge_source_power_bundle import (  # noqa: E402
    DEFAULT_SUMMARY_PATH,
    canonical_value_record,
    load_source_design_bundle,
    source_design_bundle_identity,
)


ALGORITHM = "repeated-judge-source-stage-a-power-v1"
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1

FULL_REGION_COUNTS = (64, 96, 128)
FULL_COMPONENT_COUNTS = (160, 320, 512, 800)
FULL_PRIMARY_REPEATS = 3
FULL_REPEAT4_COMPONENT_COUNTS = (320, 800)
FULL_CONFIGURATIONS = tuple(
    (region_count, components, FULL_PRIMARY_REPEATS)
    for components in FULL_COMPONENT_COUNTS
    for region_count in FULL_REGION_COUNTS
) + tuple(
    (region_count, components, 4)
    for components in FULL_REPEAT4_COMPONENT_COUNTS
    for region_count in FULL_REGION_COUNTS
)
FULL_SCENARIOS = tuple(scenario.name for scenario in SCENARIOS)
FULL_NULL_TYPES = ("block_null", "source_smooth_mean_null")
NULL_SCENARIO_BY_TYPE = {
    "block_null": "block_null",
    "source_smooth_mean_null": "mean_only",
}
PRIMARY_TRUTH_SCENARIOS = tuple(
    f"{family}_rho_{rho:.2f}"
    for family in ("cumulative", "nomic", "mixture")
    for rho in (0.10, 0.20)
)
DERANGED_SCENARIOS = tuple(
    f"deranged_rho_{rho:.2f}" for rho in (0.04, 0.10, 0.20)
)

FULL_NULL_DRAWS = 1999
FULL_POWER_REPLICATES = 200
FULL_CONFIRMATION_REPLICATES = 200
FULL_MULTIPLIER_DRAWS = 999
FULL_CONFIDENCE = 0.95
FULL_SHRINKAGE = 0.05
FULL_MISSING_RATE = 0.02
FULL_SEED = 7132026
FULL_OUTER_FOLDS = 5
FULL_INNER_FOLDS = 3

# Operational checkpoint granularity only; these values do not alter the
# scientific seed, simulation, or decision contracts.
NULL_DRAW_CHUNK_SIZE = 40
POWER_REPLICATE_CHUNK_SIZE = 20

DEFAULT_SOURCE_DESIGN = (
    HERE / "repro" / "repeated_judge_source_power" / "source_design.json"
)

_BLAS_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_WORKER_THREADPOOL_LIMIT = None
_WORKER_CONTEXT = None
_MAX_PENDING_TASKS_PER_WORKER = 2

SCIENCE_ARGUMENTS = (
    "config",
    "source_eta_grid",
    "null_types",
    "scenarios",
    "null_draws",
    "power_replicates",
    "confirmation_replicates",
    "multiplier_draws",
    "confidence",
    "gammas",
    "item_rhos",
    "mean_ridges",
    "shrinkage",
    "missing_rate",
    "prompt_batch_rows",
    "seed",
)


def parse_configuration(value: str) -> tuple[int, int, int]:
    try:
        region_text, component_text, repeat_text = value.split(":")
        region_count = int(region_text)
        components = int(component_text)
        repeats = int(repeat_text)
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("configuration must have form K:G:R") from exc
    if region_count not in FULL_REGION_COUNTS:
        raise argparse.ArgumentTypeError(f"K must belong to {FULL_REGION_COUNTS}")
    if components not in FULL_COMPONENT_COUNTS:
        raise argparse.ArgumentTypeError(f"G must belong to {FULL_COMPONENT_COUNTS}")
    if repeats not in (3, 4):
        raise argparse.ArgumentTypeError("R must be 3 or 4")
    return region_count, components, repeats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-prereg", action="store_true")
    parser.add_argument("--config", action="append", type=parse_configuration)
    parser.add_argument("--source-eta-grid", nargs="+", type=float)
    parser.add_argument("--null-types", nargs="+", choices=FULL_NULL_TYPES)
    parser.add_argument("--scenarios", nargs="+", choices=FULL_SCENARIOS)
    parser.add_argument("--null-draws", type=int)
    parser.add_argument("--power-replicates", type=int)
    parser.add_argument("--confirmation-replicates", type=int)
    parser.add_argument("--multiplier-draws", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--gammas", nargs="+", type=float)
    parser.add_argument("--item-rhos", nargs="+", type=float)
    parser.add_argument("--mean-ridges", nargs="+", type=float)
    parser.add_argument("--shrinkage", type=float)
    parser.add_argument("--missing-rate", type=float)
    parser.add_argument("--prompt-batch-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-design", default=str(DEFAULT_SOURCE_DESIGN))
    parser.add_argument("--source-summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--out", default="/tmp/repeated_judge_source_power.json")
    return parser


def _unique_sorted_floats(values, label, *, include_zero=False):
    output = tuple(map(float, values))
    if (
        not output
        or len(output) != len(set(output))
        or tuple(sorted(output)) != output
        or any(not math.isfinite(value) for value in output)
    ):
        raise ValueError(f"{label} must be finite, unique, nonempty, and increasing")
    if include_zero and 0.0 not in output:
        raise ValueError(f"{label} must include zero")
    return output


def resolve_configuration(args: argparse.Namespace) -> dict:
    if args.full_prereg:
        overridden = [name for name in SCIENCE_ARGUMENTS if getattr(args, name) is not None]
        if overridden:
            rendered = ", ".join("--" + name.replace("_", "-") for name in overridden)
            raise ValueError(f"--full-prereg rejects scientific overrides: {rendered}")
        return {
            "mode": "exact-full-preregistration",
            "configurations": FULL_CONFIGURATIONS,
            "source_eta_grid": SOURCE_ETA_GRID,
            "null_types": FULL_NULL_TYPES,
            "scenarios": FULL_SCENARIOS,
            "null_draws": FULL_NULL_DRAWS,
            "power_replicates": FULL_POWER_REPLICATES,
            "confirmation_replicates": FULL_CONFIRMATION_REPLICATES,
            "multiplier_draws": FULL_MULTIPLIER_DRAWS,
            "confidence": FULL_CONFIDENCE,
            "gammas": DEFAULT_GAMMAS,
            "item_rhos": DEFAULT_RHOS,
            "mean_ridges": DEFAULT_MEAN_RIDGES,
            "shrinkage": FULL_SHRINKAGE,
            "missing_rate": FULL_MISSING_RATE,
            "prompt_batch_rows": MAX_PROMPT_ROWS,
            "outer_folds": FULL_OUTER_FOLDS,
            "inner_folds": FULL_INNER_FOLDS,
            "seed": FULL_SEED,
        }

    configurations = tuple(args.config or ((64, 160, 3),))
    if len(configurations) != len(set(configurations)):
        raise ValueError("custom configurations must be unique")
    source_eta_grid = _unique_sorted_floats(
        args.source_eta_grid or (0.0,), "source eta grid"
    )
    if any(value not in SOURCE_ETA_GRID for value in source_eta_grid):
        raise ValueError(f"source eta grid must be drawn from {SOURCE_ETA_GRID}")
    null_types = tuple(args.null_types or ("block_null",))
    scenarios = tuple(args.scenarios or ("block_null",))
    if len(null_types) != len(set(null_types)) or len(scenarios) != len(set(scenarios)):
        raise ValueError("custom null types and scenarios must be unique")
    positive = {
        "null_draws": 1 if args.null_draws is None else args.null_draws,
        "power_replicates": 1 if args.power_replicates is None else args.power_replicates,
        "confirmation_replicates": (
            0 if args.confirmation_replicates is None else args.confirmation_replicates
        ),
        "multiplier_draws": 19 if args.multiplier_draws is None else args.multiplier_draws,
    }
    if positive["confirmation_replicates"] < 0 or any(
        int(value) < 1 for key, value in positive.items() if key != "confirmation_replicates"
    ):
        raise ValueError("draw and replicate counts must be positive")
    confidence = FULL_CONFIDENCE if args.confidence is None else float(args.confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    gammas = _unique_sorted_floats(args.gammas or DEFAULT_GAMMAS, "gammas")
    if any(not 0.0 <= value <= 1.0 for value in gammas):
        raise ValueError("gammas must lie in [0,1]")
    item_rhos = _unique_sorted_floats(
        args.item_rhos or DEFAULT_RHOS, "item rhos", include_zero=True
    )
    if any(not 0.0 <= value < 1.0 for value in item_rhos):
        raise ValueError("item rhos must lie in [0,1)")
    mean_ridges = _unique_sorted_floats(
        args.mean_ridges or DEFAULT_MEAN_RIDGES, "mean ridges"
    )
    if any(value < 0.0 for value in mean_ridges):
        raise ValueError("mean ridges must be nonnegative")
    shrinkage = FULL_SHRINKAGE if args.shrinkage is None else float(args.shrinkage)
    missing_rate = FULL_MISSING_RATE if args.missing_rate is None else float(args.missing_rate)
    prompt_rows = MAX_PROMPT_ROWS if args.prompt_batch_rows is None else int(
        args.prompt_batch_rows
    )
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must lie in [0,1)")
    if not 0.0 <= missing_rate < 0.25:
        raise ValueError("missing rate must lie in [0,.25)")
    if not 1 <= prompt_rows <= MAX_PROMPT_ROWS:
        raise ValueError("prompt batch rows must lie in [1,10]")
    return {
        "mode": "custom-diagnostic",
        "configurations": configurations,
        "source_eta_grid": source_eta_grid,
        "null_types": null_types,
        "scenarios": scenarios,
        **{key: int(value) for key, value in positive.items()},
        "confidence": confidence,
        "gammas": gammas,
        "item_rhos": item_rhos,
        "mean_ridges": mean_ridges,
        "shrinkage": shrinkage,
        "missing_rate": missing_rate,
        "prompt_batch_rows": prompt_rows,
        "outer_folds": FULL_OUTER_FOLDS,
        "inner_folds": FULL_INNER_FOLDS,
        "seed": FULL_SEED if args.seed is None else int(args.seed),
    }


def _is_exact_full_configuration(resolved) -> bool:
    return bool(
        resolved.get("mode") == "exact-full-preregistration"
        and tuple(resolved["configurations"]) == FULL_CONFIGURATIONS
        and tuple(resolved["source_eta_grid"]) == SOURCE_ETA_GRID
        and tuple(resolved["null_types"]) == FULL_NULL_TYPES
        and tuple(resolved["scenarios"]) == FULL_SCENARIOS
        and resolved["null_draws"] == FULL_NULL_DRAWS
        and resolved["power_replicates"] == FULL_POWER_REPLICATES
        and resolved["confirmation_replicates"] == FULL_CONFIRMATION_REPLICATES
        and resolved["multiplier_draws"] == FULL_MULTIPLIER_DRAWS
        and resolved["confidence"] == FULL_CONFIDENCE
        and tuple(resolved["gammas"]) == DEFAULT_GAMMAS
        and tuple(resolved["item_rhos"]) == DEFAULT_RHOS
        and tuple(resolved["mean_ridges"]) == DEFAULT_MEAN_RIDGES
        and resolved["shrinkage"] == FULL_SHRINKAGE
        and resolved["missing_rate"] == FULL_MISSING_RATE
        and resolved["prompt_batch_rows"] == MAX_PROMPT_ROWS
        and resolved["outer_folds"] == FULL_OUTER_FOLDS
        and resolved["inner_folds"] == FULL_INNER_FOLDS
        and resolved["seed"] == FULL_SEED
    )


def build_designs(bundle, region_count, components):
    _validate_bundle_source_eta_grid(bundle)
    output = {}
    for corpus in REQUIRED_SOURCE_CORPORA:
        row = bundle["designs"][corpus][str(int(region_count))]
        allocation = row["allocations"][str(int(components))]
        region_ids = tuple(row["region_ids"])
        assignments = tuple(
            region_ids[int(index)]
            for index in allocation["assignment_region_indices"]
        )
        output[corpus] = build_source_design(
            region_ids, row["exposure_matrix"], assignments
        )
    return output


def _validate_bundle_source_eta_grid(bundle):
    try:
        observed = tuple(map(float, bundle["configuration"]["source_eta_grid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source-design bundle has no valid source_eta_grid") from exc
    if observed != SOURCE_ETA_GRID:
        raise ValueError(
            "source-design bundle source_eta_grid does not equal the scientific grid"
        )
    return observed


def finite_null_threshold(values, confidence=FULL_CONFIDENCE):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("null maxima must be a nonempty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    rank = min(int(math.ceil(confidence * (len(values) + 1))), len(values))
    return float(np.sort(values)[rank - 1]), rank


def _binomial_upper_tail(successes, trials, probability):
    return math.fsum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def _binomial_cdf(successes, trials, probability):
    return math.fsum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes + 1)
    )


def clopper_pearson_lower(successes, trials, alpha=0.05):
    if trials < 1 or not 0 <= successes <= trials or not 0.0 < alpha < 1.0:
        raise ValueError("invalid Clopper-Pearson arguments")
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _binomial_upper_tail(successes, trials, middle) < alpha:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def clopper_pearson_upper(successes, trials, alpha=0.05):
    if trials < 1 or not 0 <= successes <= trials or not 0.0 < alpha < 1.0:
        raise ValueError("invalid Clopper-Pearson arguments")
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def _summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("summary needs finite values")
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _topology_successes(row):
    rate = row.get("topology_truth_beats_derangement_rate")
    if rate is None:
        return None
    count = int(round(float(rate) * int(row["replicates"])))
    if not math.isclose(
        float(rate), count / int(row["replicates"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("topology success rate is not an exact replicate fraction")
    return count


def confirmation_binomial_family_size():
    per_source_eta_pair = (
        len(FULL_NULL_TYPES) * len(SOURCE_ETA_GRID)
        + 2 * len(PRIMARY_TRUTH_SCENARIOS) * len(SOURCE_ETA_GRID)
        + len(DERANGED_SCENARIOS) * len(SOURCE_ETA_GRID)
    )
    # The expression above was the former shared-eta diagonal.  The complete
    # contract crosses the two corpus-specific source-dependence axes, adding
    # one further factor of ``len(SOURCE_ETA_GRID)``.
    return per_source_eta_pair * len(SOURCE_ETA_GRID)


def _source_eta_pairs(source_eta_grid):
    return tuple(
        (float(exploratory_source_eta), float(fresh_source_eta))
        for exploratory_source_eta in source_eta_grid
        for fresh_source_eta in source_eta_grid
    )


def _row_source_eta_pair(row):
    value = row.get("generator_source_eta_by_corpus")
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_SOURCE_CORPORA):
        raise ValueError(
            "scenario row must identify both corpus-specific generator source etas"
        )
    pair = tuple(float(value[corpus]) for corpus in REQUIRED_SOURCE_CORPORA)
    if any(item not in SOURCE_ETA_GRID for item in pair):
        raise ValueError("scenario row generator source eta is outside the frozen grid")
    return pair


def _source_eta_pair_record(pair):
    return {
        corpus: float(value)
        for corpus, value in zip(REQUIRED_SOURCE_CORPORA, pair)
    }


def evaluate_configuration(
    scenario_rows: Sequence[dict],
    resolved: Mapping,
    *,
    phase: str,
    repeats: int,
    null_complete: bool,
):
    exact = _is_exact_full_configuration(resolved)
    expected_keys = {
        (scenario, exploratory_source_eta, fresh_source_eta)
        for scenario in FULL_SCENARIOS
        for exploratory_source_eta, fresh_source_eta in _source_eta_pairs(
            SOURCE_ETA_GRID
        )
    }
    by_key = {
        (str(row["scenario"]), *_row_source_eta_pair(row)): row
        for row in scenario_rows
    }
    complete = len(scenario_rows) == len(expected_keys) and set(by_key) == expected_keys
    if (
        not exact
        or phase not in ("discovery", "confirmation")
        or repeats != FULL_PRIMARY_REPEATS
        or not null_complete
        or not complete
    ):
        return {
            "evaluable": False,
            "pass": False,
            "reason": (
                "custom, incomplete, R=4, or non-barrier runs are diagnostic only"
            ),
        }
    expected_replicates = (
        FULL_POWER_REPLICATES
        if phase == "discovery"
        else FULL_CONFIRMATION_REPLICATES
    )
    if any(int(row["replicates"]) != expected_replicates for row in scenario_rows):
        return {
            "evaluable": False,
            "pass": False,
            "reason": "replicate count does not match the frozen phase",
        }
    for row in scenario_rows:
        events = row.get("joint_source_primary_events")
        nonidentified = row.get("inference_nonidentified_replicates")
        selector_rejections = row.get("both_corpus_selector_rejections")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (events, nonidentified, selector_rejections)
        ):
            raise ValueError("scenario event counts must be integer replicate counts")
        if (
            not 0 <= events <= expected_replicates
            or not 0 <= nonidentified <= expected_replicates
            or not 0 <= selector_rejections <= expected_replicates
            or events > expected_replicates - nonidentified
            or row.get("all_replicates_inference_identified")
            is not (nonidentified == 0)
        ):
            raise ValueError("scenario event counts are internally inconsistent")
        exact_rates = (
            ("joint_source_primary_event_rate", events),
            ("both_corpus_selector_rejection_rate", selector_rejections),
        )
        if any(
            not math.isclose(
                float(row.get(rate_name)),
                count / expected_replicates,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for rate_name, count in exact_rates
        ):
            raise ValueError("scenario event rate is not an exact replicate fraction")
    all_inference_identified = bool(all(
        row.get("all_replicates_inference_identified") is True
        for row in scenario_rows
    ))
    family_size = confirmation_binomial_family_size() if phase == "confirmation" else 1
    alpha = (1.0 - FULL_CONFIDENCE) / family_size
    controls = {}
    control_pass = True
    for null_type, scenario in NULL_SCENARIO_BY_TYPE.items():
        controls[null_type] = []
        for source_eta_pair in _source_eta_pairs(SOURCE_ETA_GRID):
            row = by_key[(scenario, *source_eta_pair)]
            successes = int(row["joint_source_primary_events"])
            upper = clopper_pearson_upper(successes, expected_replicates, alpha)
            residual_gains = {
                corpus: float(
                    row["endpoint_mean_gain_per_scalar"][corpus]["residual_nll"]["mean"]
                )
                for corpus in REQUIRED_SOURCE_CORPORA
            }
            passed = bool(upper <= 0.10 and all(value >= 0.0 for value in residual_gains.values()))
            controls[null_type].append({
                "generator_source_eta_by_corpus": _source_eta_pair_record(
                    source_eta_pair
                ),
                "successes": successes,
                "replicates": expected_replicates,
                "one_sided_cp_upper": upper,
                "mean_residual_gain_by_corpus": residual_gains,
                "pass": passed,
            })
            control_pass &= passed
    truths = {}
    truth_pass = True
    for scenario in PRIMARY_TRUTH_SCENARIOS:
        truths[scenario] = []
        for source_eta_pair in _source_eta_pairs(SOURCE_ETA_GRID):
            row = by_key[(scenario, *source_eta_pair)]
            events = int(row["joint_source_primary_events"])
            topology = _topology_successes(row)
            event_lower = clopper_pearson_lower(events, expected_replicates, alpha)
            topology_lower = clopper_pearson_lower(
                int(topology), expected_replicates, alpha
            )
            gains = {
                corpus: {
                    endpoint: float(
                        row["endpoint_mean_gain_per_scalar"][corpus][endpoint]["mean"]
                    )
                    for endpoint in PRIMARY_ENDPOINTS
                }
                for corpus in REQUIRED_SOURCE_CORPORA
            }
            positive = all(
                value > 0.0
                for corpus in gains.values()
                for value in corpus.values()
            )
            passed = bool(event_lower >= 0.80 and topology_lower >= 0.80 and positive)
            truths[scenario].append({
                "generator_source_eta_by_corpus": _source_eta_pair_record(
                    source_eta_pair
                ),
                "events": events,
                "topology_successes": topology,
                "replicates": expected_replicates,
                "primary_event_cp_lower": event_lower,
                "topology_cp_lower": topology_lower,
                "mean_endpoint_gains": gains,
                "pass": passed,
            })
            truth_pass &= passed
    deranged = {}
    deranged_pass = True
    for scenario in DERANGED_SCENARIOS:
        deranged[scenario] = []
        for source_eta_pair in _source_eta_pairs(SOURCE_ETA_GRID):
            row = by_key[(scenario, *source_eta_pair)]
            successes = int(row["joint_source_primary_events"])
            upper = clopper_pearson_upper(successes, expected_replicates, alpha)
            passed = upper <= 0.10
            deranged[scenario].append({
                "generator_source_eta_by_corpus": _source_eta_pair_record(
                    source_eta_pair
                ),
                "successes": successes,
                "replicates": expected_replicates,
                "one_sided_cp_upper": upper,
                "pass": passed,
            })
            deranged_pass &= passed
    passed = bool(control_pass and truth_pass and deranged_pass)
    return {
        "evaluable": True,
        "phase": phase,
        "binomial_family_size": family_size,
        "one_sided_alpha_per_gate": alpha,
        "raw_mean_sign_guards_are_descriptive_conjunction_diagnostics": True,
        "all_scenario_replicates_inference_identified_diagnostic": (
            all_inference_identified
        ),
        "controls": controls,
        "primary_truths": truths,
        "deranged_controls": deranged,
        "pass": passed,
    }


def _null_worker(task):
    (
        phase,
        region_count,
        components,
        repeats,
        null_type,
        corpus,
        source_eta,
        start,
        stop,
    ) = task
    context = _require_worker_context((region_count, components, repeats))
    resolved = context["resolved"]
    scenario = SCENARIO_BY_NAME[NULL_SCENARIO_BY_TYPE[null_type]]
    values = [
        source_corpus_null_maximum(
            context["designs"][corpus],
            scenario,
            corpus_name=corpus,
            source_eta=source_eta,
            repeats=repeats,
            seed=_null_corpus_seed(
                resolved, phase, region_count, components, repeats,
                null_type, corpus, source_eta, draw,
            ),
            gammas=resolved["gammas"],
            rhos=resolved["item_rhos"],
            mean_ridges=resolved["mean_ridges"],
            shrinkage=resolved["shrinkage"],
            missing_rate=resolved["missing_rate"],
            max_prompt_rows=resolved["prompt_batch_rows"],
            splits=context["splits"][corpus],
        )
        for draw in range(int(start), int(stop))
    ]
    return (
        str(null_type), str(corpus), float(source_eta), int(start),
        tuple(map(float, values)),
    )


def _power_corpus_worker(task):
    (
        phase,
        region_count,
        components,
        repeats,
        scenario_name,
        corpus,
        source_eta,
        start,
        stop,
        threshold,
    ) = task
    context = _require_worker_context((region_count, components, repeats))
    resolved = context["resolved"]
    records = tuple(
        run_source_corpus_power_replicate(
            context["designs"][corpus],
            SCENARIO_BY_NAME[scenario_name],
            source_eta,
            corpus_name=corpus,
            repeats=repeats,
            seed=_power_corpus_seed(
                resolved, phase, region_count, components, repeats,
                scenario_name, corpus, source_eta, replicate,
            ),
            null_threshold=threshold,
            gammas=resolved["gammas"],
            rhos=resolved["item_rhos"],
            mean_ridges=resolved["mean_ridges"],
            shrinkage=resolved["shrinkage"],
            missing_rate=resolved["missing_rate"],
            max_prompt_rows=resolved["prompt_batch_rows"],
            splits=context["splits"][corpus],
        )
        for replicate in range(int(start), int(stop))
    )
    return (
        str(scenario_name), str(corpus), float(source_eta), int(start), records
    )


def _portable_blas_runtime(records):
    fields = (
        "user_api", "internal_api", "prefix", "version",
        "threading_layer", "architecture",
    )
    return sorted(
        ({field: record.get(field) for field in fields} for record in records),
        key=lambda row: tuple(str(row[field]) for field in fields),
    )


def _blas_runtime_identity():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return _portable_blas_runtime(threadpool_info())


def _file_record(path):
    data = Path(path).read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _provenance(bundle):
    files = {
        "design": HERE / "DESIGN_repeated_judge_source_power.md",
        "source_science": HERE / "repeated_judge_source_power.py",
        "source_bundle_loader": HERE / "repeated_judge_source_power_bundle.py",
        "baseline_science": HERE / "repeated_judge_power.py",
        "runner": Path(__file__).resolve(),
    }
    return {
        "files": {name: _file_record(path) for name, path in files.items()},
        "source_design_bundle": source_design_bundle_identity(bundle),
        "source_dependence_full_payload": copy.deepcopy(
            bundle["parent"]["full_payload_record"]
        ),
        "reviewed_source_summary": copy.deepcopy(
            bundle["parent"]["reviewed_summary_record"]
        ),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "blas_runtime": _blas_runtime_identity(),
        "blas_thread_limit_policy": (
            "child workers request one BLAS thread; serial work uses threadpoolctl when available"
        ),
        "seed_namespaces": "discovery and confirmation are disjoint sha256-derived namespaces",
    }


def _assert_provenance_unchanged(expected, bundle=None, bundle_path=None, summary_path=None):
    if bundle_path is not None:
        bundle = load_source_design_bundle(bundle_path, summary_path)
    if bundle is None or _provenance(bundle) != expected:
        raise RuntimeError(
            "scientific files or exact source-design content changed during the run"
        )


def _install_worker_context(configuration, designs, resolved):
    global _WORKER_CONTEXT
    k, g, r = map(int, configuration)
    if tuple(designs) != REQUIRED_SOURCE_CORPORA:
        raise ValueError("worker designs have the wrong corpus order")
    splits = {
        corpus: source_atomic_component_splits(
            designs[corpus],
            seed=derive_seed(resolved["seed"], "worker-split-cache", k, g, corpus),
            max_prompt_rows=resolved["prompt_batch_rows"],
        )
        for corpus in REQUIRED_SOURCE_CORPORA
    }
    _WORKER_CONTEXT = {
        "configuration": (k, g, r),
        "designs": designs,
        "splits": splits,
        "resolved": resolved,
    }


def _require_worker_context(configuration):
    expected = tuple(map(int, configuration))
    if _WORKER_CONTEXT is None or _WORKER_CONTEXT["configuration"] != expected:
        raise RuntimeError("worker scientific context is not initialized")
    return _WORKER_CONTEXT


def _initialize_worker_blas(
    bundle_path=None,
    summary_path=None,
    expected=None,
    configuration=None,
    resolved=None,
):
    global _WORKER_THREADPOOL_LIMIT
    for name in _BLAS_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"
    bundle = None
    if expected is not None or configuration is not None:
        if bundle_path is None:
            raise ValueError("worker scientific context requires a source-design path")
        bundle = load_source_design_bundle(bundle_path, summary_path)
    if expected is not None and _provenance(bundle) != expected:
        raise RuntimeError(
            "scientific files or exact source-design content changed during the run"
        )
    if configuration is not None:
        if resolved is None:
            raise ValueError("worker scientific context requires resolved configuration")
        k, g, _r = map(int, configuration)
        _install_worker_context(
            configuration, build_designs(bundle, k, g), resolved
        )
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    _WORKER_THREADPOOL_LIMIT = threadpool_limits(limits=1, user_api="blas")
    _WORKER_THREADPOOL_LIMIT.__enter__()


@contextmanager
def _single_thread_current_blas():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api="blas"):
        yield


@contextmanager
def _single_thread_child_environment():
    previous = {name: os.environ.get(name) for name in _BLAS_ENVIRONMENT_VARIABLES}
    try:
        for name in _BLAS_ENVIRONMENT_VARIABLES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _process_pool(
    workers,
    bundle_path=None,
    summary_path=None,
    provenance=None,
    configuration=None,
    resolved=None,
):
    if int(workers) == 1:
        yield None
        return
    with _single_thread_child_environment():
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker_blas,
            initargs=(
                bundle_path, summary_path, provenance, configuration, resolved
            ),
        ) as executor:
            yield executor


def _execute_tasks(tasks, worker, executor, *, max_pending=None):
    if executor is None:
        with _single_thread_current_blas():
            for task in tasks:
                yield worker(task)
        return
    task_iterator = iter(tasks)
    worker_count = int(getattr(executor, "_max_workers", 1))
    pending_limit = (
        max(1, int(max_pending))
        if max_pending is not None
        else max(1, _MAX_PENDING_TASKS_PER_WORKER * worker_count)
    )
    pending = set()
    exhausted = False
    while pending or not exhausted:
        while len(pending) < pending_limit and not exhausted:
            try:
                task = next(task_iterator)
            except StopIteration:
                exhausted = True
            else:
                pending.add(executor.submit(worker, task))
        if not pending:
            continue
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            yield future.result()


def _canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical_digest(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON value {value}")


def _checkpoint_envelope(payload):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "payload": payload,
        "payload_sha256": _canonical_digest(payload),
    }


def _read_checkpoint(path):
    try:
        envelope = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint file: {path}") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version", "payload", "payload_sha256"
    }:
        raise ValueError(f"invalid checkpoint envelope: {path}")
    if envelope["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    if _canonical_digest(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"checkpoint integrity hash mismatch: {path}")
    return envelope["payload"]


def _write_checkpoint(path, payload):
    data = (json.dumps(
        _checkpoint_envelope(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing conflicting checkpoint overwrite: {path}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"refusing conflicting checkpoint overwrite: {path}")
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _configuration_record(resolved):
    record = copy.deepcopy(dict(resolved))
    record["configurations"] = [
        {"region_count": int(k), "components_per_corpus": int(g), "repeats": int(r)}
        for k, g, r in resolved["configurations"]
    ]
    return json.loads(json.dumps(record, allow_nan=False))


def _run_fingerprint(resolved, provenance):
    return _canonical_digest({
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "configuration": _configuration_record(resolved),
        "provenance": provenance,
    })


def _candidate_json(candidate):
    return {"gamma": float(candidate.gamma), "rho_item": float(candidate.rho)}


def _candidate_from_json(value):
    _require_keys(value, {"gamma", "rho_item"}, "checkpoint candidate")
    gamma = float(value["gamma"])
    rho_item = float(value["rho_item"])
    if (
        not math.isfinite(gamma)
        or not math.isfinite(rho_item)
        or not 0.0 <= gamma <= 1.0
        or not 0.0 <= rho_item < 1.0
        or (rho_item == 0.0 and gamma != 0.5)
    ):
        raise ValueError("checkpoint candidate is outside the admissible grid")
    return Candidate(gamma, rho_item)


def _power_record_to_json(record):
    return {
        "record_schema": "source-power-aggregate-input-v2",
        "scenario": str(record.scenario),
        "generator_source_eta_by_corpus": {
            corpus: float(source_eta)
            for corpus, source_eta in zip(
                record.corpus_names, record.generator_source_eta_by_corpus
            )
        },
        "corpus_names": list(record.corpus_names),
        "selected": [
            [_candidate_json(candidate) for candidate in corpus]
            for corpus in record.selected
        ],
        "corpus_selector_rejected": list(map(bool, record.corpus_selector_rejected)),
        "familywise_rejected": bool(record.familywise_rejected),
        "maximum_inner_gain": float(record.maximum_inner_gain),
        "endpoint_mean_gains": np.asarray(record.endpoint_mean_gains).tolist(),
        "inference_source_eta_grid": list(record.inference_source_eta_grid),
        "endpoint_worst_source_eta_lower_bounds": np.asarray(
            record.endpoint_worst_source_eta_lower_bounds
        ).tolist(),
        "inference_identified": bool(record.inference_identified),
        "multiplier_critical_value": float(record.multiplier_critical_value),
        "multiplier_order_statistic_rank_one_based": int(
            record.multiplier_order_statistic_rank_one_based
        ),
        "inference_prompt_blocks": list(map(int, record.inference_prompt_blocks)),
        "inference_source_regions": list(map(int, record.inference_source_regions)),
        "topology_truth_beats_derangement": record.topology_truth_beats_derangement,
        "promoted": bool(record.promoted),
        "call_loading": float(record.call_loading),
        "persistent_loading": float(record.persistent_loading),
        "request_loading": float(record.request_loading),
    }


def _require_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} has an invalid schema")


def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} has invalid shape or values")
    return array


def _power_record_from_json(value):
    keys = {
        "record_schema", "scenario", "generator_source_eta_by_corpus",
        "corpus_names", "selected",
        "corpus_selector_rejected", "familywise_rejected", "maximum_inner_gain",
        "endpoint_mean_gains", "inference_source_eta_grid",
        "endpoint_worst_source_eta_lower_bounds", "inference_identified",
        "multiplier_critical_value", "multiplier_order_statistic_rank_one_based",
        "inference_prompt_blocks", "inference_source_regions",
        "topology_truth_beats_derangement", "promoted", "call_loading",
        "persistent_loading", "request_loading",
    }
    _require_keys(value, keys, "power checkpoint record")
    if value["record_schema"] != "source-power-aggregate-input-v2":
        raise ValueError("unsupported power checkpoint record schema")
    if tuple(value["corpus_names"]) != REQUIRED_SOURCE_CORPORA:
        raise ValueError("power checkpoint corpus order mismatch")
    if value["scenario"] not in FULL_SCENARIOS:
        raise ValueError("power checkpoint scenario mismatch")
    source_eta_by_corpus = value["generator_source_eta_by_corpus"]
    _require_keys(
        source_eta_by_corpus,
        set(REQUIRED_SOURCE_CORPORA),
        "power checkpoint generator source eta mapping",
    )
    generator_source_eta_by_corpus = tuple(
        float(source_eta_by_corpus[corpus]) for corpus in REQUIRED_SOURCE_CORPORA
    )
    if any(source_eta not in SOURCE_ETA_GRID for source_eta in generator_source_eta_by_corpus):
        raise ValueError("power checkpoint source eta mismatch")
    for key in (
        "familywise_rejected", "inference_identified", "promoted",
    ):
        if not isinstance(value[key], bool):
            raise ValueError(f"power checkpoint {key} must be boolean")
    if (
        not isinstance(value["corpus_selector_rejected"], list)
        or len(value["corpus_selector_rejected"]) != len(REQUIRED_SOURCE_CORPORA)
        or any(not isinstance(item, bool) for item in value["corpus_selector_rejected"])
    ):
        raise ValueError("power checkpoint selector decisions mismatch")
    if value["topology_truth_beats_derangement"] not in (None, True, False):
        raise ValueError("power checkpoint topology decision mismatch")
    if not isinstance(value["selected"], list):
        raise ValueError("power checkpoint selected candidates mismatch")
    for corpus in value["selected"]:
        if not isinstance(corpus, list):
            raise ValueError("power checkpoint selected candidates mismatch")
    selected = tuple(tuple(
        _candidate_from_json(candidate)
        for candidate in corpus
    ) for corpus in value["selected"])
    if (
        len(selected) != len(REQUIRED_SOURCE_CORPORA)
        or any(len(corpus) != FULL_OUTER_FOLDS for corpus in selected)
    ):
        raise ValueError("power checkpoint selected-candidate shape mismatch")
    endpoint_means = _finite_array(value["endpoint_mean_gains"], (2, 2), "endpoint means")
    worst = _finite_array(
        value["endpoint_worst_source_eta_lower_bounds"],
        (2, 2),
        "endpoint lower bounds",
    )
    eta_grid = tuple(map(float, value["inference_source_eta_grid"]))
    if (
        not eta_grid
        or tuple(sorted(set(eta_grid))) != eta_grid
        or any(eta not in SOURCE_ETA_GRID for eta in eta_grid)
    ):
        raise ValueError("power checkpoint inference eta grid mismatch")
    for key in ("inference_prompt_blocks", "inference_source_regions"):
        if (
            not isinstance(value[key], list)
            or len(value[key]) != len(REQUIRED_SOURCE_CORPORA)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 2 for item in value[key])
        ):
            raise ValueError(f"power checkpoint {key} mismatch")
    finite_scalars = [
        value["maximum_inner_gain"], value["multiplier_critical_value"],
        value["call_loading"], value["persistent_loading"], value["request_loading"],
    ]
    if not all(math.isfinite(float(item)) for item in finite_scalars):
        raise ValueError("power checkpoint contains nonfinite scalars")
    order_rank = value["multiplier_order_statistic_rank_one_based"]
    if isinstance(order_rank, bool) or not isinstance(order_rank, int) or order_rank < 1:
        raise ValueError("power checkpoint multiplier order rank mismatch")
    return SourcePowerReplicate(
        scenario=str(value["scenario"]),
        generator_source_eta_by_corpus=generator_source_eta_by_corpus,
        corpus_names=REQUIRED_SOURCE_CORPORA,
        selected=selected,
        corpus_selector_rejected=tuple(map(bool, value["corpus_selector_rejected"])),
        familywise_rejected=bool(value["familywise_rejected"]),
        maximum_inner_gain=float(value["maximum_inner_gain"]),
        endpoint_component_gains=np.empty((0, 0, 0)),
        endpoint_mean_gains=endpoint_means,
        inference_source_eta_grid=eta_grid,
        endpoint_lower_bounds_by_source_eta=np.empty((0, 0, 0)),
        endpoint_worst_source_eta_lower_bounds=worst,
        inference_identified=bool(value["inference_identified"]),
        multiplier_critical_value=float(value["multiplier_critical_value"]),
        multiplier_order_statistic_rank_one_based=int(order_rank),
        inference_prompt_blocks=tuple(map(int, value["inference_prompt_blocks"])),
        inference_source_regions=tuple(map(int, value["inference_source_regions"])),
        topology_component_advantage=None,
        topology_truth_beats_derangement=value["topology_truth_beats_derangement"],
        promoted=bool(value["promoted"]),
        call_loading=float(value["call_loading"]),
        persistent_loading=float(value["persistent_loading"]),
        request_loading=float(value["request_loading"]),
    )


class CheckpointStore:
    def __init__(self, root, resolved, provenance, fingerprint):
        self.root = Path(root).resolve()
        self.fingerprint = fingerprint
        manifest = {
            "kind": "manifest",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "configuration": _configuration_record(resolved),
            "provenance": provenance,
        }
        path = self.root / "manifest.json"
        if path.exists():
            if _read_checkpoint(path) != manifest:
                raise ValueError("checkpoint fingerprint/configuration mismatch")
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("nonempty checkpoint directory has no valid manifest")
            _write_checkpoint(path, manifest)

    def _base(self, phase, k, g, r):
        return self.root / phase / f"K{k}_G{g}_R{r}"

    def _metadata(self, kind, phase, k, g, r):
        return {
            "kind": kind,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "phase": phase,
            "region_count": int(k),
            "components": int(g),
            "repeats": int(r),
        }

    def _load(self, path, expected):
        if not path.exists():
            return None
        payload = _read_checkpoint(path)
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint payload must be an object: {path}")
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                raise ValueError(f"checkpoint hash/configuration mismatch: {path}")
        return payload

    def _null_chunk_path(self, phase, k, g, r, null_type, start, stop):
        return (
            self._base(phase, k, g, r)
            / "null_chunks"
            / null_type
            / f"draws_{start:06d}_{stop - 1:06d}.json"
        )

    def _null_chunk_metadata(
        self, phase, k, g, r, null_type, start, stop, base_seeds, source_eta_grid
    ):
        metadata = self._metadata("null_corpus_pair_chunk", phase, k, g, r)
        metadata.update({
            "null_type": str(null_type),
            "index_start": int(start),
            "index_stop_exclusive": int(stop),
            "indices": list(range(int(start), int(stop))),
            "base_seeds": list(map(int, base_seeds)),
            "source_eta_grid": list(map(float, source_eta_grid)),
        })
        return metadata

    def load_null_chunk(
        self, phase, k, g, r, null_type, start, stop, base_seeds, source_eta_grid
    ):
        path = self._null_chunk_path(phase, k, g, r, null_type, start, stop)
        expected = self._null_chunk_metadata(
            phase, k, g, r, null_type, start, stop, base_seeds, source_eta_grid
        )
        payload = self._load(path, expected)
        if payload is None:
            return None
        _require_keys(
            payload,
            set(expected) | {"complete", "corpus_maxima", "pair_maxima"},
            "null-chunk checkpoint",
        )
        if payload["complete"] is not True:
            raise ValueError(f"checkpoint null chunk is incomplete: {path}")
        count = int(stop) - int(start)
        corpus_values = {}
        if not isinstance(payload["corpus_maxima"], list):
            raise ValueError(f"checkpoint corpus maxima are invalid: {path}")
        for row in payload["corpus_maxima"]:
            _require_keys(
                row,
                {"corpus", "generator_source_eta", "maxima"},
                "null corpus-maxima row",
            )
            key = (str(row["corpus"]), float(row["generator_source_eta"]))
            values = np.asarray(row["maxima"], dtype=float)
            if key in corpus_values or values.shape != (count,):
                raise ValueError(f"checkpoint corpus maxima grid is invalid: {path}")
            if not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError(f"checkpoint corpus maxima values are invalid: {path}")
            corpus_values[key] = values.tolist()
        expected_corpus_keys = {
            (corpus, float(source_eta))
            for corpus in REQUIRED_SOURCE_CORPORA
            for source_eta in source_eta_grid
        }
        if set(corpus_values) != expected_corpus_keys:
            raise ValueError(f"checkpoint corpus maxima grid mismatch: {path}")
        pair_values = {}
        if not isinstance(payload["pair_maxima"], list):
            raise ValueError(f"checkpoint pair maxima are invalid: {path}")
        for row in payload["pair_maxima"]:
            _require_keys(
                row,
                {"generator_source_eta_by_corpus", "maxima"},
                "null pair-maxima row",
            )
            pair = _row_source_eta_pair(row)
            values = np.asarray(row["maxima"], dtype=float)
            if pair in pair_values or values.shape != (count,):
                raise ValueError(f"checkpoint pair maxima grid is invalid: {path}")
            expected_values = np.asarray([
                combine_source_corpus_null_maxima({
                    corpus: corpus_values[(corpus, pair[corpus_index])][offset]
                    for corpus_index, corpus in enumerate(REQUIRED_SOURCE_CORPORA)
                })
                for offset in range(count)
            ])
            if not np.array_equal(values, expected_values):
                raise ValueError(f"checkpoint pair maxima are inconsistent: {path}")
            pair_values[pair] = values.tolist()
        if set(pair_values) != set(_source_eta_pairs(source_eta_grid)):
            raise ValueError(f"checkpoint pair maxima grid mismatch: {path}")
        return {"corpus_maxima": corpus_values, "pair_maxima": pair_values}

    def save_null_chunk(
        self, phase, k, g, r, null_type, start, stop, base_seeds,
        source_eta_grid, corpus_values, pair_values,
    ):
        path = self._null_chunk_path(phase, k, g, r, null_type, start, stop)
        payload = self._null_chunk_metadata(
            phase, k, g, r, null_type, start, stop, base_seeds, source_eta_grid
        )
        payload.update({
            "complete": True,
            "corpus_maxima": [
                {
                    "corpus": corpus,
                    "generator_source_eta": float(source_eta),
                    "maxima": list(map(float, corpus_values[(corpus, source_eta)])),
                }
                for corpus in REQUIRED_SOURCE_CORPORA
                for source_eta in source_eta_grid
            ],
            "pair_maxima": [
                {
                    "generator_source_eta_by_corpus": _source_eta_pair_record(pair),
                    "maxima": list(map(float, pair_values[pair])),
                }
                for pair in _source_eta_pairs(source_eta_grid)
            ],
        })
        _write_checkpoint(path, payload)

    def load_barrier(self, phase, k, g, r, expected):
        path = self._base(phase, k, g, r) / "null_barrier.json"
        metadata = self._metadata("null_barrier", phase, k, g, r)
        metadata.update(expected)
        payload = self._load(path, metadata)
        if payload is None:
            return None
        _require_keys(
            payload,
            set(metadata) | {"complete", "cells", "operational_threshold"},
            "null-barrier checkpoint",
        )
        if payload["complete"] is not True or not isinstance(payload["cells"], list):
            raise ValueError(f"checkpoint null barrier is incomplete: {path}")
        expected_cells = {
            (null_type, exploratory_source_eta, fresh_source_eta)
            for null_type in expected["null_types"]
            for exploratory_source_eta, fresh_source_eta in _source_eta_pairs(
                expected["source_eta_grid"]
            )
        }
        observed_cells = set()
        for cell in payload["cells"]:
            _require_keys(
                cell,
                {
                    "null_type", "generator_source_eta_by_corpus", "draws",
                    "maxima", "rank", "threshold", "maximum_summary",
                },
                "null-barrier cell",
            )
            key = (str(cell["null_type"]), *_row_source_eta_pair(cell))
            if key in observed_cells:
                raise ValueError(f"checkpoint null barrier has a duplicate cell: {path}")
            observed_cells.add(key)
            maxima = np.asarray(cell["maxima"], dtype=float)
            if maxima.shape != (int(expected["draws_per_cell"]),):
                raise ValueError(f"checkpoint null barrier has wrong draw count: {path}")
            threshold, rank = finite_null_threshold(maxima, expected["confidence"])
            if (
                int(cell["draws"]) != len(maxima)
                or int(cell["rank"]) != rank
                or float(cell["threshold"]) != threshold
                or cell["maximum_summary"] != _summary(maxima)
            ):
                raise ValueError(f"checkpoint null barrier is internally inconsistent: {path}")
        if observed_cells != expected_cells:
            raise ValueError(f"checkpoint null barrier cell grid mismatch: {path}")
        expected_operational = max(float(cell["threshold"]) for cell in payload["cells"])
        if float(payload["operational_threshold"]) != expected_operational:
            raise ValueError(f"checkpoint operational threshold is inconsistent: {path}")
        return payload

    def save_barrier(self, phase, k, g, r, payload):
        path = self._base(phase, k, g, r) / "null_barrier.json"
        complete = self._metadata("null_barrier", phase, k, g, r)
        complete.update(payload)
        _write_checkpoint(path, complete)

    def _power_chunk_path(self, phase, k, g, r, scenario, start, stop):
        return (
            self._base(phase, k, g, r)
            / "power_chunks"
            / scenario
            / f"replicates_{start:06d}_{stop - 1:06d}.json"
        )

    def _power_chunk_metadata(
        self, phase, k, g, r, scenario, start, stop, threshold,
        base_seeds, source_eta_grid,
    ):
        metadata = self._metadata("power_corpus_pair_chunk", phase, k, g, r)
        metadata.update({
            "scenario": str(scenario),
            "index_start": int(start),
            "index_stop_exclusive": int(stop),
            "indices": list(range(int(start), int(stop))),
            "base_seeds": list(map(int, base_seeds)),
            "threshold": float(threshold),
            "source_eta_grid": list(map(float, source_eta_grid)),
        })
        return metadata

    def load_power_chunk(
        self, phase, k, g, r, scenario, start, stop, threshold,
        base_seeds, source_eta_grid,
    ):
        path = self._power_chunk_path(phase, k, g, r, scenario, start, stop)
        expected = self._power_chunk_metadata(
            phase, k, g, r, scenario, start, stop, threshold,
            base_seeds, source_eta_grid,
        )
        payload = self._load(path, expected)
        if payload is None:
            return None
        _require_keys(
            payload,
            set(expected) | {"complete", "pair_records"},
            "power-chunk checkpoint",
        )
        if payload["complete"] is not True:
            raise ValueError(f"checkpoint power chunk is incomplete: {path}")
        if not isinstance(payload["pair_records"], list):
            raise ValueError(f"checkpoint power pair rows are invalid: {path}")
        pair_records = {}
        for row in payload["pair_records"]:
            _require_keys(row, {"replicate", "record"}, "power pair row")
            if isinstance(row["replicate"], bool) or not isinstance(
                row["replicate"], int
            ):
                raise ValueError(f"checkpoint replicate index is invalid: {path}")
            replicate = row["replicate"]
            record = _power_record_from_json(row["record"])
            if (
                record.scenario != scenario
                or tuple(record.inference_source_eta_grid)
                != tuple(source_eta_grid)
            ):
                raise ValueError(f"checkpoint power record contract mismatch: {path}")
            pair = tuple(map(float, record.generator_source_eta_by_corpus))
            key = (replicate, *pair)
            if key in pair_records:
                raise ValueError(f"checkpoint power chunk has duplicate pair row: {path}")
            pair_records[key] = record
        expected_pair_keys = {
            (replicate, *pair)
            for replicate in range(int(start), int(stop))
            for pair in _source_eta_pairs(source_eta_grid)
        }
        if set(pair_records) != expected_pair_keys:
            raise ValueError(f"checkpoint power pair grid mismatch: {path}")
        return {"pair_records": pair_records}

    def save_power_chunk(
        self, phase, k, g, r, scenario, start, stop, threshold,
        base_seeds, source_eta_grid, pair_records,
    ):
        path = self._power_chunk_path(phase, k, g, r, scenario, start, stop)
        payload = self._power_chunk_metadata(
            phase, k, g, r, scenario, start, stop, threshold,
            base_seeds, source_eta_grid,
        )
        payload.update({
            "complete": True,
            "pair_records": [
                {
                    "replicate": replicate,
                    "record": _power_record_to_json(
                        pair_records[(replicate, *pair)]
                    ),
                }
                for replicate in range(int(start), int(stop))
                for pair in _source_eta_pairs(source_eta_grid)
            ],
        })
        _write_checkpoint(path, payload)


def _null_base_seed(resolved, phase, k, g, r, null_type, draw):
    return derive_seed(
        resolved["seed"], phase, k, g, r, "null", null_type, draw
    )


def _null_corpus_seed(
    resolved, phase, k, g, r, null_type, corpus, _source_eta, draw
):
    return derive_seed(
        _null_base_seed(resolved, phase, k, g, r, null_type, draw), corpus
    )


def _power_base_seed(resolved, phase, k, g, r, scenario, replicate):
    return derive_seed(
        resolved["seed"], phase, k, g, r, "power", scenario, replicate
    )


def _power_corpus_seed(
    resolved, phase, k, g, r, scenario, corpus, _source_eta, replicate
):
    return derive_seed(
        _power_base_seed(resolved, phase, k, g, r, scenario, replicate), corpus
    )


def _power_multiplier_seed(resolved, phase, k, g, r, scenario, replicate):
    return derive_seed(
        _power_base_seed(resolved, phase, k, g, r, scenario, replicate),
        "joint-prompt-source-multiplier",
    )


def _public_barrier(barrier):
    return {
        "complete": bool(barrier["complete"]),
        "draws_per_cell": int(barrier["draws_per_cell"]),
        "confidence": float(barrier["confidence"]),
        "operational_threshold": float(barrier["operational_threshold"]),
        "cells": [
            {
                key: copy.deepcopy(cell[key])
                for key in (
                    "null_type", "generator_source_eta_by_corpus", "draws",
                    "rank", "threshold", "maximum_summary",
                )
            }
            for cell in barrier["cells"]
        ],
    }


def _chunk_ranges(count, chunk_size):
    if int(count) < 0 or int(chunk_size) < 1:
        raise ValueError("chunk count must be nonnegative and chunk size positive")
    return tuple(
        (start, min(start + int(chunk_size), int(count)))
        for start in range(0, int(count), int(chunk_size))
    )


def run_configuration(
    configuration,
    resolved,
    bundle,
    *,
    phase="discovery",
    workers=1,
    checkpoint_store=None,
    provenance_snapshot=None,
    bundle_path=None,
    summary_path=None,
):
    k, g, r = map(int, configuration)
    designs = build_designs(bundle, k, g)
    _install_worker_context((k, g, r), designs, resolved)
    source_eta_grid = tuple(map(float, resolved["source_eta_grid"]))
    source_eta_pairs = _source_eta_pairs(source_eta_grid)
    expected_barrier = {
        "draws_per_cell": int(resolved["null_draws"]),
        "confidence": float(resolved["confidence"]),
        "null_types": list(resolved["null_types"]),
        "source_eta_grid": list(source_eta_grid),
        "null_draw_chunk_size": NULL_DRAW_CHUNK_SIZE,
    }
    barrier = (
        checkpoint_store.load_barrier(phase, k, g, r, expected_barrier)
        if checkpoint_store is not None else None
    )
    scenario_rows = []
    with _process_pool(
        workers,
        bundle_path,
        summary_path,
        provenance_snapshot,
        (k, g, r),
        resolved,
    ) as executor:
        if barrier is None:
            pair_maxima = {
                (null_type, *pair): []
                for null_type in resolved["null_types"]
                for pair in source_eta_pairs
            }
            for null_type in resolved["null_types"]:
                for start, stop in _chunk_ranges(
                    resolved["null_draws"], NULL_DRAW_CHUNK_SIZE
                ):
                    base_seeds = [
                        _null_base_seed(
                            resolved, phase, k, g, r, null_type, draw
                        )
                        for draw in range(start, stop)
                    ]
                    chunk = (
                        checkpoint_store.load_null_chunk(
                            phase, k, g, r, null_type, start, stop,
                            base_seeds, source_eta_grid,
                        )
                        if checkpoint_store is not None else None
                    )
                    if chunk is None:
                        tasks = (
                            (
                                phase, k, g, r, null_type, corpus,
                                source_eta, start, stop,
                            )
                            for corpus in REQUIRED_SOURCE_CORPORA
                            for source_eta in source_eta_grid
                        )
                        corpus_values = {}
                        for (
                            observed_null_type, corpus, source_eta,
                            observed_start, maxima,
                        ) in _execute_tasks(tasks, _null_worker, executor):
                            if (
                                observed_null_type != null_type
                                or observed_start != start
                            ):
                                raise RuntimeError("null worker returned the wrong chunk")
                            corpus_values[(corpus, source_eta)] = list(maxima)
                        expected_corpus_keys = {
                            (corpus, source_eta)
                            for corpus in REQUIRED_SOURCE_CORPORA
                            for source_eta in source_eta_grid
                        }
                        if set(corpus_values) != expected_corpus_keys:
                            raise RuntimeError("null worker corpus grid is incomplete")
                        chunk_pair_values = {
                            pair: [
                                combine_source_corpus_null_maxima({
                                    corpus: corpus_values[
                                        (corpus, pair[corpus_index])
                                    ][offset]
                                    for corpus_index, corpus in enumerate(
                                        REQUIRED_SOURCE_CORPORA
                                    )
                                })
                                for offset in range(stop - start)
                            ]
                            for pair in source_eta_pairs
                        }
                        chunk = {
                            "corpus_maxima": corpus_values,
                            "pair_maxima": chunk_pair_values,
                        }
                        if checkpoint_store is not None:
                            checkpoint_store.save_null_chunk(
                                phase, k, g, r, null_type, start, stop,
                                base_seeds, source_eta_grid, corpus_values,
                                chunk_pair_values,
                            )
                    for pair in source_eta_pairs:
                        pair_maxima[(null_type, *pair)].extend(
                            chunk["pair_maxima"][pair]
                        )
            cells = []
            for null_type in resolved["null_types"]:
                for pair in source_eta_pairs:
                    maxima = pair_maxima[(null_type, *pair)]
                    threshold, rank = finite_null_threshold(
                        maxima, resolved["confidence"]
                    )
                    cells.append({
                        "null_type": null_type,
                        "generator_source_eta_by_corpus": (
                            _source_eta_pair_record(pair)
                        ),
                        "draws": len(maxima),
                        "maxima": maxima,
                        "rank": rank,
                        "threshold": threshold,
                        "maximum_summary": _summary(maxima),
                    })
            barrier = {
                **expected_barrier,
                "complete": True,
                "cells": cells,
                "operational_threshold": float(
                    max(cell["threshold"] for cell in cells)
                ),
            }
            if checkpoint_store is not None:
                checkpoint_store.save_barrier(phase, k, g, r, barrier)
        if not barrier.get("complete"):
            raise ValueError("power cannot start before the complete null barrier")
        threshold = float(barrier["operational_threshold"])
        replicates = (
            resolved["power_replicates"] if phase == "discovery"
            else resolved["confirmation_replicates"]
        )
        for scenario_name in resolved["scenarios"]:
            records_by_pair = {pair: [] for pair in source_eta_pairs}
            for start, stop in _chunk_ranges(
                replicates, POWER_REPLICATE_CHUNK_SIZE
            ):
                base_seeds = [
                    _power_base_seed(
                        resolved, phase, k, g, r, scenario_name, replicate
                    )
                    for replicate in range(start, stop)
                ]
                chunk = (
                    checkpoint_store.load_power_chunk(
                        phase, k, g, r, scenario_name, start, stop,
                        threshold, base_seeds, source_eta_grid,
                    )
                    if checkpoint_store is not None else None
                )
                if chunk is None:
                    tasks = (
                        (
                            phase, k, g, r, scenario_name, corpus,
                            source_eta, start, stop, threshold,
                        )
                        for corpus in REQUIRED_SOURCE_CORPORA
                        for source_eta in source_eta_grid
                    )
                    corpus_records = {}
                    for (
                        observed_scenario, corpus, source_eta,
                        observed_start, chunk_records,
                    ) in _execute_tasks(tasks, _power_corpus_worker, executor):
                        if (
                            observed_scenario != scenario_name
                            or observed_start != start
                            or len(chunk_records) != stop - start
                        ):
                            raise RuntimeError("power worker returned the wrong chunk")
                        for offset, record in enumerate(chunk_records):
                            corpus_records[
                                (start + offset, corpus, source_eta)
                            ] = record
                    expected_corpus_keys = {
                        (replicate, corpus, source_eta)
                        for replicate in range(start, stop)
                        for corpus in REQUIRED_SOURCE_CORPORA
                        for source_eta in source_eta_grid
                    }
                    if set(corpus_records) != expected_corpus_keys:
                        raise RuntimeError("power worker corpus grid is incomplete")
                    pair_records = {}
                    with _single_thread_current_blas():
                        for replicate in range(start, stop):
                            multiplier_seed = _power_multiplier_seed(
                                resolved, phase, k, g, r,
                                scenario_name, replicate,
                            )
                            prepared_components = {
                                (corpus, source_eta): (
                                    prepare_graph_aware_source_corpus_multiplier(
                                        corpus_records[
                                            (replicate, corpus, source_eta)
                                        ].endpoint_component_gains,
                                        corpus_records[
                                            (replicate, corpus, source_eta)
                                        ].prompt_block_ids,
                                        designs[corpus],
                                        corpus_name=corpus,
                                        inference_source_eta_grid=(
                                            source_eta_grid
                                        ),
                                        draws=resolved["multiplier_draws"],
                                        multiplier_seed=multiplier_seed,
                                    )
                                )
                                for corpus in REQUIRED_SOURCE_CORPORA
                                for source_eta in source_eta_grid
                            }
                            for pair in source_eta_pairs:
                                combined = combine_source_power_corpus_replicates(
                                    {
                                        corpus: corpus_records[
                                            (
                                                replicate,
                                                corpus,
                                                pair[corpus_index],
                                            )
                                        ]
                                        for corpus_index, corpus in enumerate(
                                            REQUIRED_SOURCE_CORPORA
                                        )
                                    },
                                    designs,
                                    SCENARIO_BY_NAME[scenario_name],
                                    multiplier_seed=multiplier_seed,
                                    confidence=resolved["confidence"],
                                    multiplier_draws=resolved[
                                        "multiplier_draws"
                                    ],
                                    inference_source_eta_grid=source_eta_grid,
                                    prepared_multiplier_components_by_corpus={
                                        corpus: prepared_components[
                                            (corpus, pair[corpus_index])
                                        ]
                                        for corpus_index, corpus in enumerate(
                                            REQUIRED_SOURCE_CORPORA
                                        )
                                    },
                                )
                                pair_records[(replicate, *pair)] = (
                                    _power_record_from_json(
                                        _power_record_to_json(combined)
                                    )
                                )
                    chunk = {
                        "pair_records": pair_records,
                    }
                    if checkpoint_store is not None:
                        checkpoint_store.save_power_chunk(
                            phase, k, g, r, scenario_name, start, stop,
                            threshold, base_seeds, source_eta_grid,
                            pair_records,
                        )
                for pair in source_eta_pairs:
                    records_by_pair[pair].extend(
                        chunk["pair_records"][(replicate, *pair)]
                        for replicate in range(start, stop)
                    )
            for pair in source_eta_pairs:
                scenario_rows.append(
                    aggregate_source_power_records(records_by_pair[pair])
                )
    if provenance_snapshot is not None:
        _assert_provenance_unchanged(
            provenance_snapshot,
            bundle=bundle,
            bundle_path=bundle_path,
            summary_path=summary_path,
        )
    exact = _is_exact_full_configuration(resolved)
    decision = evaluate_configuration(
        scenario_rows,
        resolved,
        phase=phase,
        repeats=r,
        null_complete=bool(barrier.get("complete") and exact),
    )
    split_context = _require_worker_context((k, g, r))["splits"]
    design_identity = {
        corpus: {
            "exposure_matrix_record": copy.deepcopy(
                bundle["designs"][corpus][str(k)]["exposure_matrix_record"]
            ),
            "assignment_record": copy.deepcopy(
                bundle["designs"][corpus][str(k)]["allocations"][str(g)]["source_records"]["assignment_region_ids"]
            ),
            "used_source_regions": int(
                bundle["designs"][corpus][str(k)]["allocations"][str(g)]["used_region_count"]
            ),
            "source_split_diagnostics": source_split_diagnostics(
                designs[corpus], split_context[corpus]
            ),
        }
        for corpus in REQUIRED_SOURCE_CORPORA
    }
    return {
        "phase": phase,
        "region_count": k,
        "components_per_required_corpus": g,
        "repeats": r,
        "source_design_identity": design_identity,
        "operational_checkpoint_chunking": {
            "null_draws_per_shard": NULL_DRAW_CHUNK_SIZE,
            "power_replicates_per_shard": POWER_REPLICATE_CHUNK_SIZE,
        },
        "null_calibration": _public_barrier(barrier),
        "scenarios": scenario_rows,
        "decision": decision,
    }


def build_scientific_payload(
    resolved,
    bundle,
    *,
    workers=1,
    checkpoint_dir=None,
    provenance_snapshot=None,
    run_fingerprint=None,
    bundle_path=None,
    summary_path=None,
):
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    _validate_bundle_source_eta_grid(bundle)
    provenance_snapshot = _provenance(bundle) if provenance_snapshot is None else provenance_snapshot
    expected_fingerprint = _run_fingerprint(resolved, provenance_snapshot)
    if run_fingerprint is not None and run_fingerprint != expected_fingerprint:
        raise ValueError("run fingerprint does not match configuration/provenance")
    checkpoint_store = None if checkpoint_dir is None else CheckpointStore(
        checkpoint_dir, resolved, provenance_snapshot, expected_fingerprint
    )
    discovery = []
    for configuration in resolved["configurations"]:
        _assert_provenance_unchanged(
            provenance_snapshot, bundle=bundle,
            bundle_path=bundle_path, summary_path=summary_path,
        )
        discovery.append(run_configuration(
            configuration,
            resolved,
            bundle,
            phase="discovery",
            workers=workers,
            checkpoint_store=checkpoint_store,
            provenance_snapshot=provenance_snapshot,
            bundle_path=bundle_path,
            summary_path=summary_path,
        ))
    exact = _is_exact_full_configuration(resolved)
    passing = [
        row for row in discovery
        if row["repeats"] == FULL_PRIMARY_REPEATS
        and row["decision"].get("evaluable")
        and row["decision"].get("pass")
    ] if exact else []
    provisional = None
    if passing:
        winner = min(
            passing,
            key=lambda row: (
                row["components_per_required_corpus"], row["region_count"]
            ),
        )
        provisional = {
            "region_count": winner["region_count"],
            "components_per_required_corpus": winner["components_per_required_corpus"],
            "repeats": FULL_PRIMARY_REPEATS,
        }
    confirmation = None
    if provisional is not None:
        confirmation = run_configuration(
            (
                provisional["region_count"],
                provisional["components_per_required_corpus"],
                provisional["repeats"],
            ),
            resolved,
            bundle,
            phase="confirmation",
            workers=workers,
            checkpoint_store=checkpoint_store,
            provenance_snapshot=provenance_snapshot,
            bundle_path=bundle_path,
            summary_path=summary_path,
        )
    confirmed = bool(
        exact
        and confirmation is not None
        and confirmation["decision"].get("evaluable")
        and confirmation["decision"].get("pass")
    )
    authorization = {
        "attempted_input_identity_inventory_unlocked": confirmed,
        "candidate_enumeration_unlocked": False,
        "nomic_embedding_unlocked": False,
        "judge_calls_authorized": False,
        "live_campaign_authorized": False,
        "covariance_deployment_unlocked": False,
        "independent_batching_unlocked": False,
        "qr_specialization_unlocked": False,
        "cuda_claim_unlocked": False,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "status": (
            "STAGE A CONFIRMED; ATTEMPTED-INPUT IDENTITY INVENTORY ONLY"
            if confirmed else "STAGE A FAIL-CLOSED; NO DOWNSTREAM AUTHORIZATION"
        ),
        "configuration": _configuration_record(resolved),
        "provenance": provenance_snapshot,
        "source_design_bundle_identity": source_design_bundle_identity(bundle),
        "discovery_results": discovery,
        "selection": {
            "exact_full_discovery_evaluable": exact,
            "selection_order": "smallest G, then coarsest/smallest K",
            "provisional_design": provisional,
            "provisional_design_unlocks_nothing": True,
            "confirmation_seed_namespace_disjoint": True,
            "confirmation_pass": confirmed,
        },
        "confirmation_result": confirmation,
        "authorization": authorization,
        "stage_b_requirement": (
            "After the immutable identity inventory, topology-only universe, "
            "revision-pinned Nomic cache, and exact immutable packing are frozen, "
            "the realized H/E/prompt incidence and cross-corpus shared-nuisance "
            "audit must pass a fixed-design 1,999-null/200-evaluation run and a "
            "second seed-disjoint fixed-design 1,999-null/200 confirmation with "
            "no K/G reselection; scalar ESS or row sums are not substitutes."
        ),
    }
    _assert_provenance_unchanged(
        provenance_snapshot, bundle=bundle,
        bundle_path=bundle_path, summary_path=summary_path,
    )
    return payload


def serialize_payload(payload):
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _content_record(data):
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def project_summary(payload):
    projection = copy.deepcopy(payload)
    full_bytes = serialize_payload(payload).encode("utf-8")
    projection["artifact_projection"] = "tracked-summary-v1"
    projection["full_payload_record"] = _content_record(full_bytes)
    for row in projection["discovery_results"]:
        scenarios = row.pop("scenarios")
        row["scenario_results_record"] = canonical_value_record(scenarios)
    if projection["confirmation_result"] is not None:
        scenarios = projection["confirmation_result"].pop("scenarios")
        projection["confirmation_result"]["scenario_results_record"] = canonical_value_record(scenarios)
    return projection


def write_payload(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = serialize_payload(payload).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main(argv: Iterable[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    resolved = resolve_configuration(args)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    bundle = load_source_design_bundle(args.source_design, args.source_summary)
    _validate_bundle_source_eta_grid(bundle)
    provenance = _provenance(bundle)
    fingerprint = _run_fingerprint(resolved, provenance)
    started = time.perf_counter()
    payload = build_scientific_payload(
        resolved,
        bundle,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        provenance_snapshot=provenance,
        run_fingerprint=fingerprint,
        bundle_path=args.source_design,
        summary_path=args.source_summary,
    )
    _assert_provenance_unchanged(
        provenance, bundle_path=args.source_design, summary_path=args.source_summary
    )
    output_payload = project_summary(payload) if args.summary_only else payload
    write_payload(args.out, output_payload)
    full_record = _content_record(serialize_payload(payload).encode("utf-8"))
    print(json.dumps({
        "full_payload_record": full_record,
        "written_projection": "tracked-summary-v1" if args.summary_only else "complete",
        "wall_seconds": time.perf_counter() - started,
        "authorization_pass": payload["authorization"]["attempted_input_identity_inventory_unlocked"],
    }, indent=2, sort_keys=True))
    return payload


def cli(argv: Iterable[str] | None = None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    payload = main(arguments)
    audit_only = "--audit-only" in arguments
    passed = payload["authorization"]["attempted_input_identity_inventory_unlocked"]
    return 0 if audit_only or passed else 2


if __name__ == "__main__":
    raise SystemExit(cli())
