#!/usr/bin/env python3
"""Run the frozen repeated-judge synthetic sizing mechanism.

The ordinary CLI defaults are deliberately small smoke settings.  Passing
``--full-prereg`` selects, without scientific overrides, the exact component,
repeat, scenario, candidate-grid, resampling, and confidence settings frozen in
``PREREG_graph_geometry_repeated_judge.md``.  Full runs are expensive and are
not part of the unit-test suite.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from typing import Iterable, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repeated_judge_power import (  # noqa: E402
    Candidate,
    DEFAULT_GAMMAS,
    DEFAULT_MEAN_RIDGES,
    DEFAULT_RHOS,
    MAX_PROMPT_ROWS,
    PRIMARY_ENDPOINTS,
    PowerReplicate,
    SCENARIOS,
    SCENARIO_BY_NAME,
    SYNTHETIC_CORPORA,
    _replicate_geometry_and_splits,
    aggregate_power_records,
    calibrate_synthetic_selector_null,
    derive_seed,
    draw_repeated_field,
    finite_null_maximum_threshold,
    inner_candidate_search,
    run_power_replicate,
    summary,
)


SMOKE_CONFIGURATIONS = ((32, 3),)
FULL_CONFIGURATIONS = (
    (160, 3),
    (320, 3),
    (512, 3),
    (800, 3),
    (320, 4),
    (800, 4),
)
SMOKE_SCENARIOS = (
    "block_null",
    "mean_only",
    "cumulative_rho_0.10",
    "cumulative_rho_0.20",
)
FULL_SCENARIOS = tuple(scenario.name for scenario in SCENARIOS)

FULL_NULL_DRAWS = 1999
FULL_POWER_REPLICATES = 200
FULL_MULTIPLIER_DRAWS = 999
FULL_CONFIDENCE = 0.95
FULL_OUTER_FOLDS = 5
FULL_INNER_FOLDS = 3
FULL_SHRINKAGE = 0.05
FULL_MISSING_RATE = 0.02
FULL_SEED = 1207000

CHECKPOINT_SCHEMA_VERSION = 1
_BLAS_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_WORKER_THREADPOOL_LIMIT = None

SCIENCE_ARGUMENTS = (
    "config",
    "null_draws",
    "power_replicates",
    "multiplier_draws",
    "confidence",
    "outer_folds",
    "inner_folds",
    "mean_ridges",
    "shrinkage",
    "gammas",
    "rhos",
    "scenarios",
    "missing_rate",
    "prompt_batch_rows",
    "seed",
)


def parse_configuration(value: str) -> Tuple[int, int]:
    try:
        component_text, repeat_text = value.split(":", 1)
        components, repeats = int(component_text), int(repeat_text)
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError(
            "configuration must have form COMPONENTS:REPEATS"
        ) from error
    if components < 12 or repeats < 3:
        raise argparse.ArgumentTypeError(
            "configuration needs at least 12 components and 3 repeats"
        )
    return components, repeats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-prereg", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        type=parse_configuration,
        default=None,
        help="COMPONENTS:REPEATS; may be supplied more than once",
    )
    parser.add_argument("--null-draws", type=int, default=None)
    parser.add_argument("--power-replicates", type=int, default=None)
    parser.add_argument("--multiplier-draws", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--outer-folds", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--mean-ridges", nargs="+", type=float, default=None)
    parser.add_argument("--shrinkage", type=float, default=None)
    parser.add_argument("--gammas", nargs="+", type=float, default=None)
    parser.add_argument("--rhos", nargs="+", type=float, default=None)
    parser.add_argument(
        "--scenarios", nargs="+", choices=FULL_SCENARIOS, default=None
    )
    parser.add_argument("--missing-rate", type=float, default=None)
    parser.add_argument("--prompt-batch-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="operational worker-process count; does not change the scientific design",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="operational atomic checkpoint directory; an exact matching run resumes",
    )
    parser.add_argument("--out", default="/tmp/repeated_judge_power_smoke.json")
    return parser


def resolve_configuration(args: argparse.Namespace) -> dict:
    if args.full_prereg:
        overridden = [name for name in SCIENCE_ARGUMENTS if getattr(args, name) is not None]
        if overridden:
            rendered = ", ".join("--" + name.replace("_", "-") for name in overridden)
            raise ValueError(f"--full-prereg rejects scientific overrides: {rendered}")
        return {
            "full_preregistration": True,
            "configurations": FULL_CONFIGURATIONS,
            "scenarios": FULL_SCENARIOS,
            "gammas": DEFAULT_GAMMAS,
            "rhos": DEFAULT_RHOS,
            "mean_ridges": DEFAULT_MEAN_RIDGES,
            "null_draws": FULL_NULL_DRAWS,
            "power_replicates": FULL_POWER_REPLICATES,
            "multiplier_draws": FULL_MULTIPLIER_DRAWS,
            "confidence": FULL_CONFIDENCE,
            "outer_folds": FULL_OUTER_FOLDS,
            "inner_folds": FULL_INNER_FOLDS,
            "shrinkage": FULL_SHRINKAGE,
            "missing_rate": FULL_MISSING_RATE,
            "prompt_batch_rows": MAX_PROMPT_ROWS,
            "seed": FULL_SEED,
        }

    configurations = tuple(args.config or SMOKE_CONFIGURATIONS)
    values = {
        "null_draws": 19 if args.null_draws is None else args.null_draws,
        "power_replicates": 12 if args.power_replicates is None else args.power_replicates,
        "multiplier_draws": 199 if args.multiplier_draws is None else args.multiplier_draws,
    }
    if any(int(value) < 1 for value in values.values()):
        raise ValueError("draw and replicate counts must be positive")
    confidence = FULL_CONFIDENCE if args.confidence is None else float(args.confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    outer_folds = FULL_OUTER_FOLDS if args.outer_folds is None else int(args.outer_folds)
    inner_folds = FULL_INNER_FOLDS if args.inner_folds is None else int(args.inner_folds)
    if outer_folds != FULL_OUTER_FOLDS or inner_folds != FULL_INNER_FOLDS:
        raise ValueError("the implemented procedure requires exactly five outer and three inner folds")
    shrinkage = FULL_SHRINKAGE if args.shrinkage is None else float(args.shrinkage)
    missing_rate = FULL_MISSING_RATE if args.missing_rate is None else float(args.missing_rate)
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must lie in [0,1)")
    if not 0.0 <= missing_rate < 0.25:
        raise ValueError("missing rate must lie in [0,.25)")
    prompt_batch_rows = (
        MAX_PROMPT_ROWS if args.prompt_batch_rows is None else int(args.prompt_batch_rows)
    )
    if not 1 <= prompt_batch_rows <= MAX_PROMPT_ROWS:
        raise ValueError("prompt batch rows must lie in [1,10]")
    gammas = tuple(DEFAULT_GAMMAS if args.gammas is None else map(float, args.gammas))
    rhos = tuple(DEFAULT_RHOS if args.rhos is None else map(float, args.rhos))
    if not gammas or len(set(gammas)) != len(gammas) or any(
        not 0.0 <= value <= 1.0 for value in gammas
    ):
        raise ValueError("gammas must be unique and lie in [0,1]")
    if not rhos or len(set(rhos)) != len(rhos) or 0.0 not in rhos or any(
        not 0.0 <= value < 1.0 for value in rhos
    ):
        raise ValueError("rhos must be unique, lie in [0,1), and include zero")
    mean_ridges = tuple(
        DEFAULT_MEAN_RIDGES if args.mean_ridges is None else map(float, args.mean_ridges)
    )
    if not mean_ridges or len(set(mean_ridges)) != len(mean_ridges) or any(
        value < 0.0 for value in mean_ridges
    ):
        raise ValueError("mean ridges must be unique and nonnegative")
    scenarios = tuple(args.scenarios or SMOKE_SCENARIOS)
    return {
        "full_preregistration": False,
        "configurations": configurations,
        "scenarios": scenarios,
        "gammas": gammas,
        "rhos": rhos,
        "mean_ridges": mean_ridges,
        "null_draws": int(values["null_draws"]),
        "power_replicates": int(values["power_replicates"]),
        "multiplier_draws": int(values["multiplier_draws"]),
        "confidence": confidence,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "shrinkage": shrinkage,
        "missing_rate": missing_rate,
        "prompt_batch_rows": prompt_batch_rows,
        "seed": FULL_SEED if args.seed is None else int(args.seed),
    }


def _one_sided_exact_binomial_pvalue(successes, trials, null_probability=0.05):
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < null_probability < 1.0:
        raise ValueError("null probability must lie in (0,1)")
    if successes == 0:
        return 1.0
    tail = math.fsum(
        math.comb(trials, count)
        * null_probability**count
        * (1.0 - null_probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )
    return float(min(1.0, max(0.0, tail)))


def _mean_endpoint_gains(row, endpoint):
    return {
        corpus: float(
            row["endpoint_mean_gain_per_scalar"][corpus][endpoint]["mean"]
        )
        for corpus in SYNTHETIC_CORPORA
    }


def _configuration_decision(
    results: Sequence[dict], selected_scenarios: Sequence[str], *, repeats=3
) -> dict:
    by_name = {row["scenario"]: row for row in results}
    complete = set(FULL_SCENARIOS).issubset(selected_scenarios) and set(
        FULL_SCENARIOS
    ).issubset(by_name)
    if not complete:
        return {
            "evaluable": False,
            "pass": False,
            "reason": "the complete frozen scenario grid is required",
        }

    control_gates = {}
    for name in ("block_null", "mean_only"):
        row = by_name[name]
        trials = int(row["replicates"])
        event_rate = float(row["joint_synthetic_primary_event_rate"])
        promotions = int(row.get(
            "joint_synthetic_primary_events", round(event_rate * trials)
        ))
        pvalue = _one_sided_exact_binomial_pvalue(promotions, trials)
        false_promotion_gate = bool(event_rate <= 0.10 and pvalue >= 0.05)
        gains = _mean_endpoint_gains(row, "residual_nll")
        control_gates[name] = {
            "promotions": promotions,
            "replicates": trials,
            "joint_synthetic_primary_event_rate": event_rate,
            "one_sided_exact_binomial_excess_pvalue_at_p_0_05": pvalue,
            "does_not_reject_p_at_most_0_05_and_observed_at_most_0_10": (
                false_promotion_gate
            ),
            "mean_selected_residual_nll_gain_by_synthetic_corpus": gains,
            "mean_harm_nonpositive_in_both_synthetic_corpora": bool(
                all(gain >= 0.0 for gain in gains.values())
            ),
        }

    truth_gates = {}
    for family in ("cumulative", "nomic", "mixture"):
        for rho in (0.10, 0.20):
            name = f"{family}_rho_{rho:.2f}"
            row = by_name[name]
            endpoint_gains = {
                corpus: {
                    endpoint: float(
                        row["endpoint_mean_gain_per_scalar"][corpus][endpoint]["mean"]
                    )
                    for endpoint in PRIMARY_ENDPOINTS
                }
                for corpus in SYNTHETIC_CORPORA
            }
            all_endpoint_gains = [
                endpoint_gains[corpus][endpoint]
                for corpus in SYNTHETIC_CORPORA
                for endpoint in PRIMARY_ENDPOINTS
            ]
            topology_rate = row["topology_truth_beats_derangement_rate"]
            truth_gates[name] = {
                "joint_two_corpus_synthetic_primary_event_power": float(
                    row["joint_synthetic_primary_event_rate"]
                ),
                "power_at_least_80pct": bool(
                    float(row["joint_synthetic_primary_event_rate"]) >= 0.80
                ),
                "mean_endpoint_gains_by_synthetic_corpus": endpoint_gains,
                "all_four_corpus_by_endpoint_mean_gains_positive": bool(
                    all(value > 0.0 for value in all_endpoint_gains)
                ),
                "topology_truth_beats_derangement_rate": topology_rate,
                "topology_rate_at_least_80pct": bool(
                    topology_rate is not None and float(topology_rate) >= 0.80
                ),
            }

    passed = bool(
        all(
            row["does_not_reject_p_at_most_0_05_and_observed_at_most_0_10"]
            and row["mean_harm_nonpositive_in_both_synthetic_corpora"]
            for row in control_gates.values()
        )
        and all(
            row["power_at_least_80pct"]
            and row["all_four_corpus_by_endpoint_mean_gains_positive"]
            and row["topology_rate_at_least_80pct"]
            for row in truth_gates.values()
        )
    )
    return {
        "evaluable": True,
        "repeat_design": int(repeats),
        "block_null": {
            key: control_gates["block_null"][key]
            for key in (
                "promotions",
                "replicates",
                "joint_synthetic_primary_event_rate",
                "one_sided_exact_binomial_excess_pvalue_at_p_0_05",
                "does_not_reject_p_at_most_0_05_and_observed_at_most_0_10",
            )
        },
        "controls": control_gates,
        "required_truths": truth_gates,
        "pass": passed,
    }


def _compute_null_draw(components, repeats, draw, null_seed, resolved):
    """Compute one indexed draw exactly as the monolithic calibrator does."""
    replicate_seed = derive_seed(null_seed, "null", int(draw))
    joint_search_maxima = []
    for corpus in SYNTHETIC_CORPORA:
        corpus_seed = derive_seed(replicate_seed, corpus)
        geometry, splits = _replicate_geometry_and_splits(
            components, corpus_seed, resolved["prompt_batch_rows"]
        )
        field = draw_repeated_field(
            geometry,
            SCENARIO_BY_NAME["block_null"],
            repeats,
            derive_seed(corpus_seed, "field"),
            missing_rate=resolved["missing_rate"],
        )
        searches = [
            inner_candidate_search(
                field,
                geometry,
                fold,
                gammas=resolved["gammas"],
                rhos=resolved["rhos"],
                mean_ridges=resolved["mean_ridges"],
                shrinkage=resolved["shrinkage"],
            )
            for fold in splits.outer
        ]
        joint_search_maxima.extend(
            search.maximum_eligible_gain for search in searches
        )
    return float(max(joint_search_maxima))


def _null_worker(task):
    components, repeats, draw, null_seed, resolved = task
    return int(draw), _compute_null_draw(
        components, repeats, draw, null_seed, resolved
    )


def _power_worker(task):
    components, repeats, scenario_index, replicate, threshold, resolved = task
    scenario_name = resolved["scenarios"][scenario_index]
    replicate_seed = derive_seed(
        resolved["seed"], components, repeats, scenario_name, replicate
    )
    record = run_power_replicate(
        components,
        SCENARIO_BY_NAME[scenario_name],
        repeats=repeats,
        seed=replicate_seed,
        null_threshold=threshold,
        gammas=resolved["gammas"],
        rhos=resolved["rhos"],
        mean_ridges=resolved["mean_ridges"],
        shrinkage=resolved["shrinkage"],
        confidence=resolved["confidence"],
        multiplier_draws=resolved["multiplier_draws"],
        missing_rate=resolved["missing_rate"],
        max_prompt_rows=resolved["prompt_batch_rows"],
    )
    return int(replicate), record


def _initialize_worker_blas(expected_provenance=None):
    global _WORKER_THREADPOOL_LIMIT
    for name in _BLAS_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"
    if expected_provenance is not None:
        _assert_provenance_unchanged(expected_provenance)
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
def _process_pool(workers, provenance_snapshot=None):
    if workers == 1:
        yield None
        return
    with _single_thread_child_environment():
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker_blas,
            initargs=(provenance_snapshot,),
        ) as executor:
            yield executor


def _execute_tasks(tasks, worker, executor):
    if executor is None:
        with _single_thread_current_blas():
            for task in tasks:
                yield worker(task)
        return
    futures = [executor.submit(worker, task) for task in tasks]
    for future in as_completed(futures):
        yield future.result()


def run_configuration(
    configuration: Tuple[int, int],
    resolved: dict,
    *,
    configuration_index=0,
    workers=1,
    checkpoint_store=None,
    provenance_snapshot=None,
) -> dict:
    components, repeats = map(int, configuration)
    null_seed = derive_seed(
        resolved["seed"], components, repeats, "block_null_calibration"
    )

    barrier = (
        checkpoint_store.load_null_barrier(
            configuration_index,
            components,
            repeats,
            null_seed,
            resolved["confidence"],
            resolved["null_draws"],
        )
        if checkpoint_store is not None
        else None
    )
    if barrier is None:
        maxima_by_draw = {}
        if checkpoint_store is not None:
            for draw in range(resolved["null_draws"]):
                value = checkpoint_store.load_null_draw(
                    configuration_index, components, repeats, draw, null_seed
                )
                if value is not None:
                    maxima_by_draw[draw] = value
        missing_draws = [
            draw for draw in range(resolved["null_draws"])
            if draw not in maxima_by_draw
        ]
        null_tasks = [
            (components, repeats, draw, null_seed, resolved)
            for draw in missing_draws
        ]
        if provenance_snapshot is not None:
            _assert_provenance_unchanged(provenance_snapshot)
        with _process_pool(workers, provenance_snapshot) as executor:
            for draw, value in _execute_tasks(null_tasks, _null_worker, executor):
                maxima_by_draw[draw] = value
                if checkpoint_store is not None:
                    checkpoint_store.save_null_draw(
                        configuration_index,
                        components,
                        repeats,
                        draw,
                        null_seed,
                        value,
                    )
        null_maxima = np.asarray([
            maxima_by_draw[draw] for draw in range(resolved["null_draws"])
        ])
        threshold, rank = finite_null_maximum_threshold(
            null_maxima, resolved["confidence"]
        )
        if checkpoint_store is not None:
            checkpoint_store.save_null_barrier(
                configuration_index,
                components,
                repeats,
                null_seed,
                null_maxima,
                threshold,
                rank,
                resolved["confidence"],
            )
    else:
        null_maxima, threshold, rank = barrier

    scenario_results = []
    if provenance_snapshot is not None:
        _assert_provenance_unchanged(provenance_snapshot)
    with _process_pool(workers, provenance_snapshot) as executor:
        for scenario_index, scenario_name in enumerate(resolved["scenarios"]):
            records_by_replicate = {}
            if checkpoint_store is not None:
                for replicate in range(resolved["power_replicates"]):
                    replicate_seed = derive_seed(
                        resolved["seed"],
                        components,
                        repeats,
                        scenario_name,
                        replicate,
                    )
                    record = checkpoint_store.load_power_record(
                        configuration_index,
                        components,
                        repeats,
                        scenario_index,
                        scenario_name,
                        replicate,
                        threshold,
                        replicate_seed,
                    )
                    if record is not None:
                        records_by_replicate[replicate] = record
            missing_replicates = [
                replicate for replicate in range(resolved["power_replicates"])
                if replicate not in records_by_replicate
            ]
            tasks = [
                (
                    components,
                    repeats,
                    scenario_index,
                    replicate,
                    threshold,
                    resolved,
                )
                for replicate in missing_replicates
            ]
            for replicate, record in _execute_tasks(tasks, _power_worker, executor):
                records_by_replicate[replicate] = record
                if checkpoint_store is not None:
                    replicate_seed = derive_seed(
                        resolved["seed"],
                        components,
                        repeats,
                        scenario_name,
                        replicate,
                    )
                    checkpoint_store.save_power_record(
                        configuration_index,
                        components,
                        repeats,
                        scenario_index,
                        scenario_name,
                        replicate,
                        threshold,
                        replicate_seed,
                        record,
                    )
            records = [
                records_by_replicate[replicate]
                for replicate in range(resolved["power_replicates"])
            ]
            scenario_results.append(aggregate_power_records(records))
    if provenance_snapshot is not None:
        _assert_provenance_unchanged(provenance_snapshot)
    return {
        "components_per_required_corpus": components,
        "repeats": repeats,
        "outer_component_folds": resolved["outer_folds"],
        "inner_component_folds": resolved["inner_folds"],
        "maximum_rows_per_prompt_request": resolved["prompt_batch_rows"],
        "synthetic_joint_selector_null": {
            "draws": resolved["null_draws"],
            "seed": int(null_seed),
            "maximum_eligible_inner_gain": summary(null_maxima),
            "upper_order_statistic_threshold": float(threshold),
            "order_statistic_rank_one_based": int(rank),
            "strict_rejection": "synthetic observed joint maximum must be strictly greater than threshold",
            "real_campaign_reuse_prohibited": True,
        },
        "scenarios": scenario_results,
        "decision": _configuration_decision(
            scenario_results, resolved["scenarios"], repeats=repeats
        ),
    }


def _file_digest(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _blas_runtime_identity():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    fields = (
        "user_api", "internal_api", "prefix", "filepath", "version",
        "threading_layer", "architecture",
    )
    return sorted(
        ({field: record.get(field) for field in fields} for record in threadpool_info()),
        key=lambda record: tuple(str(record[field]) for field in fields),
    )


def _provenance():
    files = {
        "preregistration": os.path.join(HERE, "PREREG_graph_geometry_repeated_judge.md"),
        "power_primitives": os.path.join(HERE, "repeated_judge_power.py"),
        "power_runner": os.path.abspath(__file__),
    }
    return {
        "files": {name: _file_digest(path) for name, path in files.items()},
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "blas_runtime": _blas_runtime_identity(),
        "worker_blas_threads": 1,
        "seed_derivation": "sha256(base,G,R,scenario,replicate); null namespace is disjoint",
    }


def _configuration_record(resolved):
    record = {key: value for key, value in resolved.items() if key != "configurations"}
    record["configurations"] = [
        {"components_per_required_corpus": int(value[0]), "repeats": int(value[1])}
        for value in resolved["configurations"]
    ]
    return json.loads(json.dumps(record, allow_nan=False))


def _canonical_digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_fingerprint(resolved, provenance):
    return _canonical_digest({
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "configuration": _configuration_record(resolved),
        "provenance": provenance,
    })


def _assert_provenance_unchanged(expected):
    observed = _provenance()
    if observed != expected:
        raise RuntimeError(
            "preregistration or implementation changed during the run; refusing output"
        )


def _checkpoint_envelope(payload):
    return {"payload": payload, "payload_sha256": _canonical_digest(payload)}


def _read_checkpoint(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            envelope = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint file: {path}") from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "payload", "payload_sha256"
    }:
        raise ValueError(f"invalid checkpoint envelope: {path}")
    if _canonical_digest(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"checkpoint integrity hash mismatch: {path}")
    return envelope["payload"]


def _candidate_to_json(candidate):
    return {"gamma": float(candidate.gamma), "rho": float(candidate.rho)}


def _power_record_to_json(record):
    # Persist only fields consumed by aggregate_power_records.  The full
    # component arrays would make the preregistered checkpoint set enormous.
    return {
        "record_schema": "aggregate_inputs_v1",
        "scenario": record.scenario,
        "selected": [
            [_candidate_to_json(candidate) for candidate in corpus]
            for corpus in record.selected
        ],
        "corpus_selector_rejected": list(record.corpus_selector_rejected),
        "familywise_rejected": bool(record.familywise_rejected),
        "maximum_inner_gain": float(record.maximum_inner_gain),
        "endpoint_mean_gains": record.endpoint_mean_gains.tolist(),
        "endpoint_lower_bounds": record.endpoint_lower_bounds.tolist(),
        "multiplier_critical_value": float(record.multiplier_critical_value),
        "inference_prompt_blocks": list(record.inference_prompt_blocks),
        "topology_truth_beats_derangement": record.topology_truth_beats_derangement,
        "promoted": bool(record.promoted),
        "call_loading": float(record.call_loading),
        "persistent_loading": float(record.persistent_loading),
        "request_loading": float(record.request_loading),
    }


def _power_record_from_json(value):
    if value.get("record_schema") != "aggregate_inputs_v1":
        raise ValueError("unsupported power checkpoint record schema")
    return PowerReplicate(
        str(value["scenario"]),
        tuple(tuple(
            Candidate(float(candidate["gamma"]), float(candidate["rho"]))
            for candidate in corpus
        ) for corpus in value["selected"]),
        tuple(map(bool, value["corpus_selector_rejected"])),
        bool(value["familywise_rejected"]),
        float(value["maximum_inner_gain"]),
        np.empty((0, 0, 0), dtype=float),
        np.asarray(value["endpoint_mean_gains"], dtype=float),
        np.asarray(value["endpoint_lower_bounds"], dtype=float),
        float(value["multiplier_critical_value"]),
        tuple(map(int, value["inference_prompt_blocks"])),
        None,
        value["topology_truth_beats_derangement"],
        bool(value["promoted"]),
        float(value["call_loading"]),
        float(value["persistent_loading"]),
        float(value["request_loading"]),
    )


class CheckpointStore:
    def __init__(self, root, resolved, provenance, fingerprint):
        self.root = os.path.abspath(root)
        self.fingerprint = fingerprint
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "configuration": _configuration_record(resolved),
            "provenance": provenance,
        }
        manifest_path = os.path.join(self.root, "manifest.json")
        if os.path.exists(manifest_path):
            if _read_checkpoint(manifest_path) != manifest:
                raise ValueError(
                    "checkpoint fingerprint/configuration mismatch; use a new directory"
                )
        else:
            if os.path.isdir(self.root) and os.listdir(self.root):
                raise ValueError("nonempty checkpoint directory has no valid manifest")
            write_payload(manifest_path, _checkpoint_envelope(manifest))

    def _configuration_dir(self, index, components, repeats):
        return os.path.join(
            self.root, f"configuration_{index:03d}_G{components}_R{repeats}"
        )

    def _metadata(self, kind, index, components, repeats):
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "kind": kind,
            "configuration_index": int(index),
            "components": int(components),
            "repeats": int(repeats),
        }

    def _load_optional(self, path, expected):
        if not os.path.exists(path):
            return None
        payload = _read_checkpoint(path)
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint hash/config mismatch: {path}")
        return payload

    def load_null_draw(self, index, components, repeats, draw, null_seed):
        path = os.path.join(
            self._configuration_dir(index, components, repeats),
            "null",
            f"draw_{draw:06d}.json",
        )
        expected = self._metadata("null_draw", index, components, repeats)
        expected.update({"draw": int(draw), "null_seed": int(null_seed)})
        payload = self._load_optional(path, expected)
        return None if payload is None else float(payload["maximum"])

    def save_null_draw(
        self, index, components, repeats, draw, null_seed, maximum
    ):
        path = os.path.join(
            self._configuration_dir(index, components, repeats),
            "null",
            f"draw_{draw:06d}.json",
        )
        payload = self._metadata("null_draw", index, components, repeats)
        payload.update({
            "draw": int(draw),
            "null_seed": int(null_seed),
            "maximum": float(maximum),
        })
        write_payload(path, _checkpoint_envelope(payload))

    def load_null_barrier(
        self, index, components, repeats, null_seed, confidence, draws
    ):
        path = os.path.join(
            self._configuration_dir(index, components, repeats),
            "null_barrier.json",
        )
        expected = self._metadata("null_barrier", index, components, repeats)
        expected.update({
            "null_seed": int(null_seed),
            "confidence": float(confidence),
            "draws": int(draws),
        })
        payload = self._load_optional(path, expected)
        if payload is None:
            return None
        maxima = np.asarray(payload["maxima"], dtype=float)
        if maxima.shape != (int(draws),):
            raise ValueError(f"checkpoint null barrier has wrong draw count: {path}")
        threshold, rank = finite_null_maximum_threshold(
            maxima, float(confidence)
        )
        if threshold != float(payload["threshold"]) or rank != int(payload["rank"]):
            raise ValueError(f"checkpoint null barrier is inconsistent: {path}")
        return maxima, threshold, rank

    def save_null_barrier(
        self, index, components, repeats, null_seed, maxima, threshold, rank,
        confidence
    ):
        path = os.path.join(
            self._configuration_dir(index, components, repeats),
            "null_barrier.json",
        )
        payload = self._metadata("null_barrier", index, components, repeats)
        payload.update({
            "null_seed": int(null_seed),
            "confidence": float(confidence),
            "draws": int(len(maxima)),
            "maxima": np.asarray(maxima, dtype=float).tolist(),
            "threshold": float(threshold),
            "rank": int(rank),
        })
        write_payload(path, _checkpoint_envelope(payload))

    def _power_path(self, index, components, repeats, scenario_index, replicate):
        return os.path.join(
            self._configuration_dir(index, components, repeats),
            "power",
            f"scenario_{scenario_index:03d}",
            f"replicate_{replicate:06d}.json",
        )

    def load_power_record(
        self, index, components, repeats, scenario_index, scenario_name,
        replicate, threshold, replicate_seed
    ):
        path = self._power_path(
            index, components, repeats, scenario_index, replicate
        )
        expected = self._metadata("power_replicate", index, components, repeats)
        expected.update({
            "scenario_index": int(scenario_index),
            "scenario": scenario_name,
            "replicate": int(replicate),
            "threshold": float(threshold),
            "replicate_seed": int(replicate_seed),
        })
        payload = self._load_optional(path, expected)
        return None if payload is None else _power_record_from_json(payload["record"])

    def save_power_record(
        self, index, components, repeats, scenario_index, scenario_name,
        replicate, threshold, replicate_seed, record
    ):
        path = self._power_path(
            index, components, repeats, scenario_index, replicate
        )
        payload = self._metadata("power_replicate", index, components, repeats)
        payload.update({
            "scenario_index": int(scenario_index),
            "scenario": scenario_name,
            "replicate": int(replicate),
            "threshold": float(threshold),
            "replicate_seed": int(replicate_seed),
            "record": _power_record_to_json(record),
        })
        write_payload(path, _checkpoint_envelope(payload))


def _is_exact_full_preregistration(resolved):
    return bool(
        resolved.get("full_preregistration") is True
        and tuple(resolved["configurations"]) == FULL_CONFIGURATIONS
        and tuple(resolved["scenarios"]) == FULL_SCENARIOS
        and tuple(resolved["gammas"]) == DEFAULT_GAMMAS
        and tuple(resolved["rhos"]) == DEFAULT_RHOS
        and tuple(resolved["mean_ridges"]) == DEFAULT_MEAN_RIDGES
        and resolved["null_draws"] == FULL_NULL_DRAWS
        and resolved["power_replicates"] == FULL_POWER_REPLICATES
        and resolved["multiplier_draws"] == FULL_MULTIPLIER_DRAWS
        and resolved["confidence"] == FULL_CONFIDENCE
        and resolved["outer_folds"] == FULL_OUTER_FOLDS
        and resolved["inner_folds"] == FULL_INNER_FOLDS
        and resolved["shrinkage"] == FULL_SHRINKAGE
        and resolved["missing_rate"] == FULL_MISSING_RATE
        and resolved["prompt_batch_rows"] == MAX_PROMPT_ROWS
        and resolved["seed"] == FULL_SEED
    )


def build_scientific_payload(
    resolved: dict,
    *,
    workers=1,
    checkpoint_dir=None,
    provenance_snapshot=None,
    run_fingerprint=None,
) -> dict:
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    provenance_snapshot = _provenance() if provenance_snapshot is None else provenance_snapshot
    expected_fingerprint = _run_fingerprint(resolved, provenance_snapshot)
    if run_fingerprint is not None and run_fingerprint != expected_fingerprint:
        raise ValueError("run fingerprint does not match configuration/provenance")
    run_fingerprint = expected_fingerprint
    checkpoint_store = (
        None if checkpoint_dir is None
        else CheckpointStore(
            checkpoint_dir, resolved, provenance_snapshot, run_fingerprint
        )
    )
    configurations = []
    for index, configuration in enumerate(resolved["configurations"]):
        _assert_provenance_unchanged(provenance_snapshot)
        if workers == 1 and checkpoint_store is None:
            row = run_configuration(configuration, resolved)
        else:
            row = run_configuration(
                configuration,
                resolved,
                configuration_index=index,
                workers=int(workers),
                checkpoint_store=checkpoint_store,
                provenance_snapshot=provenance_snapshot,
            )
        configurations.append(row)
    primary_r3 = [row for row in configurations if row["repeats"] == 3]
    repeat4 = [row for row in configurations if row["repeats"] == 4]
    exact_full_preregistration = _is_exact_full_preregistration(resolved)
    passing_r3 = sorted((
        row["components_per_required_corpus"]
        for row in primary_r3
        if row["decision"]["evaluable"] and row["decision"]["pass"]
    )) if exact_full_preregistration else []
    synthetic_smallest_passing_G = passing_r3[0] if passing_r3 else None
    sizing_evaluable = exact_full_preregistration and bool(primary_r3) and all(
        row["decision"]["evaluable"] for row in primary_r3
    )
    gate_reason = (
        "R=3 can identify only a smallest passing two-corpus synthetic-primary-event "
        "grid point; R=4 is reported separately. No final campaign G or deployment "
        "recommendation is emitted until the unsimulated real-data gates pass."
        if exact_full_preregistration
        else "Smoke or customized runs are diagnostic only. Synthetic sizing is evaluable "
        "only for --full-prereg with the exact frozen grid, counts, and seed."
    )

    payload = {
        "schema_version": 2,
        "status": "SYNTHETIC SIZING ONLY; NO REAL COVARIANCE DEPLOYMENT",
        "scope": (
            "generic per-corpus synthetic sizing mechanism with three-row endpoint-disjoint "
            "components, stable split-contained prompt blocks of at most ten rows, shared "
            "request effects, repeated four-channel measurements, five-fold outer "
            "cross-fitting, and distinct residual and latent-state posterior NLL endpoints"
        ),
        "configuration": _configuration_record(resolved),
        "provenance": provenance_snapshot,
        "primary_r3_results": primary_r3,
        "repeat4_sensitivity_results": repeat4,
        "gate": {
            "synthetic_primary_event_sizing_evaluable": sizing_evaluable,
            "synthetic_primary_event_pass": bool(
                sizing_evaluable and synthetic_smallest_passing_G is not None
            ),
            "smallest_passing_synthetic_primary_event_G_per_corpus": (
                synthetic_smallest_passing_G
            ),
            "final_campaign_G_recommendation": None,
            "deployment_pass": False,
            "real_covariance_deployment": False,
            "reason": gate_reason,
        },
        "real_campaign_selector_requirement": (
            "Each outer-training prompt-block set must recalibrate its own block-null "
            "familywise selector. The synthetic sizing threshold is never reusable."
        ),
        "unsimulated_required_gates": [
            "blinded third-family, human, graph-structural, or downstream truth/generalization",
            "real repeated-judge repeat-specific stochastic-correlation identification",
            "real posterior calibration, coverage, and Mahalanobis non-worsening",
            "real decision log-loss and margin-gated AURC degradation at most .01",
            "ten-bin ECE and prompt-block-bootstrap AURC intervals",
            "real graph/Nomic source-correlation and same-split ablations",
            "real topology benefit over derangement and hard negatives",
            "list-position engineering pilot",
            "frozen train-only position×role×judge adjustment",
            "position-effect power sensitivity",
            "s_safe spectral adapter, full call/prompt/wave incidence covariance, and delta_95",
            "statistical/numerical loading limits and cross-batch independence bound",
        ],
    }
    _assert_provenance_unchanged(provenance_snapshot)
    return payload


def serialize_payload(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_payload(path: str, payload: dict) -> None:
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(serialize_payload(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main(argv: Iterable[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    resolved = resolve_configuration(args)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    provenance_snapshot = _provenance()
    run_fingerprint = _run_fingerprint(resolved, provenance_snapshot)
    started = time.perf_counter()
    payload = build_scientific_payload(
        resolved,
        workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        provenance_snapshot=provenance_snapshot,
        run_fingerprint=run_fingerprint,
    )
    _assert_provenance_unchanged(provenance_snapshot)
    serialized = serialize_payload(payload).encode("utf-8")
    _assert_provenance_unchanged(provenance_snapshot)
    write_payload(args.out, payload)
    print(json.dumps({
        "content_sha256": hashlib.sha256(serialized).hexdigest(),
        "output": os.path.abspath(args.out),
        "wall_seconds": time.perf_counter() - started,
        "full_preregistration": resolved["full_preregistration"],
        "configurations": len(resolved["configurations"]),
    }, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
