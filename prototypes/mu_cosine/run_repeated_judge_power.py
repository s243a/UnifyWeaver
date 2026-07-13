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
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from typing import Iterable, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repeated_judge_power import (  # noqa: E402
    DEFAULT_GAMMAS,
    DEFAULT_MEAN_RIDGES,
    DEFAULT_RHOS,
    MAX_PROMPT_ROWS,
    PRIMARY_ENDPOINTS,
    SCENARIOS,
    SCENARIO_BY_NAME,
    SYNTHETIC_CORPORA,
    aggregate_power_records,
    calibrate_synthetic_selector_null,
    derive_seed,
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

    null = by_name["block_null"]
    null_trials = int(null["replicates"])
    null_promotions = int(null.get(
        "joint_synthetic_primary_events",
        round(float(null["joint_synthetic_primary_event_rate"]) * null_trials),
    ))
    null_pvalue = _one_sided_exact_binomial_pvalue(null_promotions, null_trials)
    null_gate = bool(
        float(null["joint_synthetic_primary_event_rate"]) <= 0.10
        and null_pvalue >= 0.05
    )

    control_gates = {}
    for name in ("block_null", "mean_only"):
        gains = _mean_endpoint_gains(by_name[name], "residual_nll")
        control_gates[name] = {
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
        null_gate
        and all(
            row["mean_harm_nonpositive_in_both_synthetic_corpora"]
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
            "promotions": null_promotions,
            "replicates": null_trials,
            "joint_synthetic_primary_event_rate": float(
                null["joint_synthetic_primary_event_rate"]
            ),
            "one_sided_exact_binomial_excess_pvalue_at_p_0_05": null_pvalue,
            "does_not_reject_p_at_most_0_05_and_observed_at_most_0_10": null_gate,
        },
        "controls": control_gates,
        "required_truths": truth_gates,
        "pass": passed,
    }


def run_configuration(configuration: Tuple[int, int], resolved: dict) -> dict:
    components, repeats = map(int, configuration)
    null_seed = derive_seed(
        resolved["seed"], components, repeats, "block_null_calibration"
    )
    null_maxima, threshold, rank = calibrate_synthetic_selector_null(
        components,
        repeats=repeats,
        draws=resolved["null_draws"],
        seed=null_seed,
        gammas=resolved["gammas"],
        rhos=resolved["rhos"],
        mean_ridges=resolved["mean_ridges"],
        shrinkage=resolved["shrinkage"],
        confidence=resolved["confidence"],
        missing_rate=resolved["missing_rate"],
        max_prompt_rows=resolved["prompt_batch_rows"],
    )
    scenario_results = []
    for scenario_name in resolved["scenarios"]:
        scenario = SCENARIO_BY_NAME[scenario_name]
        records = []
        for replicate in range(resolved["power_replicates"]):
            replicate_seed = derive_seed(
                resolved["seed"], components, repeats, scenario_name, replicate
            )
            records.append(run_power_replicate(
                components,
                scenario,
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
            ))
        scenario_results.append(aggregate_power_records(records))
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


def _provenance():
    files = {
        "preregistration": os.path.join(HERE, "PREREG_graph_geometry_repeated_judge.md"),
        "power_primitives": os.path.join(HERE, "repeated_judge_power.py"),
        "power_runner": os.path.abspath(__file__),
    }
    return {
        "files": {name: _file_digest(path) for name, path in files.items()},
        "numpy_version": np.__version__,
        "seed_derivation": "sha256(base,G,R,scenario,replicate); null namespace is disjoint",
    }


def build_scientific_payload(resolved: dict) -> dict:
    configurations = [
        run_configuration(configuration, resolved)
        for configuration in resolved["configurations"]
    ]
    primary_r3 = [row for row in configurations if row["repeats"] == 3]
    repeat4 = [row for row in configurations if row["repeats"] == 4]
    passing_r3 = sorted(
        (
            row["components_per_required_corpus"]
            for row in primary_r3
            if row["decision"]["evaluable"] and row["decision"]["pass"]
        )
    )
    synthetic_smallest_passing_G = passing_r3[0] if passing_r3 else None
    sizing_evaluable = bool(primary_r3) and all(
        row["decision"]["evaluable"] for row in primary_r3
    )

    configuration_record = {
        key: value for key, value in resolved.items() if key != "configurations"
    }
    configuration_record["configurations"] = [
        {"components_per_required_corpus": int(value[0]), "repeats": int(value[1])}
        for value in resolved["configurations"]
    ]
    return {
        "schema_version": 2,
        "status": "SYNTHETIC SIZING ONLY; NO REAL COVARIANCE DEPLOYMENT",
        "scope": (
            "generic per-corpus synthetic sizing mechanism with three-row endpoint-disjoint "
            "components, stable split-contained prompt blocks of at most ten rows, shared "
            "request effects, repeated four-channel measurements, five-fold outer "
            "cross-fitting, and distinct residual and latent-state posterior NLL endpoints"
        ),
        "configuration": configuration_record,
        "provenance": _provenance(),
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
            "reason": (
                "R=3 can identify only a smallest passing two-corpus synthetic-primary-event "
                "grid point; R=4 is reported separately. No final campaign G or deployment "
                "recommendation is emitted until the unsimulated real-data gates pass."
            ),
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
            "s_safe spectral adapter, full call/prompt/wave incidence covariance, and delta_95",
            "statistical/numerical loading limits and cross-batch independence bound",
        ],
    }


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
    started = time.perf_counter()
    payload = build_scientific_payload(resolved)
    serialized = serialize_payload(payload).encode("utf-8")
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
