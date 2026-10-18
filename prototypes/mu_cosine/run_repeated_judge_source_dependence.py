#!/usr/bin/env python3
"""Audit topology-only dependence across full repeated-judge source regions.

This command consumes only the two frozen canonical graphs.  It does not read
historical labels, enumerate candidate triples, generate embeddings, or call a
judge.  The narrow structural bridge does not itself authorize any downstream
work: a completed audit exits with status 2 unless ``--audit-only`` is given.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from repeated_judge_campaign import FROZEN_WALK_WEIGHTS, content_record
from repeated_judge_candidate_capacity import (
    ENDPOINTS_PER_COMPONENT,
    SOURCE_COMPONENT_CAP_FRACTION,
)
from repeated_judge_source_dependence import audit_source_dependence
from repeated_judge_source_regions import DEFAULT_REGION_COUNT_GRID
from run_repeated_judge_candidate_capacity import (
    REGISTERED_COMPONENT_SIZES,
    load_frozen_graphs,
)


SCHEMA_VERSION = 1
ALGORITHM = "repeated-judge-topology-source-dependence-bridge-v1"
REQUIRED_CORPORA = ("exploratory", "fresh")
FROZEN_RHO_GRID = (0.0, 0.025, 0.05, 0.10, 0.20)
IMPLEMENTATION_FILES = (
    "repeated_judge_source_dependence.py",
    "run_repeated_judge_source_dependence.py",
    "repeated_judge_source_regions.py",
    "repeated_judge_candidate_capacity.py",
    "run_repeated_judge_candidate_capacity.py",
    "repeated_judge_campaign.py",
    "graph_geometry.py",
    "sample_sigma_hop_fresh_corpus.py",
    "sigma_hop_confirmatory.py",
    "run_product_kalman_realdata.py",
    "run_structured_residual_covariance.py",
    "mu_attention.py",
    "lmdb_id.py",
    "DESIGN_repeated_judge_source_dependence.md",
    "PREREG_graph_geometry_repeated_judge.md",
)


def _json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _implementation_records(root=HERE):
    root = Path(root)
    return {
        name: content_record((root / name).read_bytes())
        for name in IMPLEMENTATION_FILES
    }


def _assert_implementation_unchanged(expected):
    current = _implementation_records()
    if current != expected:
        changed = sorted(
            name
            for name in set(expected) | set(current)
            if expected.get(name) != current.get(name)
        )
        raise RuntimeError(
            "implementation or protocol provenance changed during source-dependence audit: "
            + ", ".join(changed)
        )


def _assert_graph_inputs_unchanged(expected, current):
    if current != expected:
        raise RuntimeError(
            "external graph content provenance changed during source-dependence audit"
        )


def _portable_blas_runtime(records):
    """Keep numerical runtime identity while excluding host-specific paths."""
    fields = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
    )
    return sorted(
        ({field: record.get(field) for field in fields} for record in records),
        key=lambda record: tuple(str(record[field]) for field in fields),
    )


def _blas_runtime_identity():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    return _portable_blas_runtime(threadpool_info())


def _single_blas_thread_context():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _graph_bundle_record(graph_inputs):
    canonical = json.dumps(
        graph_inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return content_record(canonical)


def _canonical_value_record(value):
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return content_record(data)


def _summary_projection(payload):
    """Return a reviewable projection with hashes for bulky exact arrays."""
    full_bytes = _json_bytes(payload)
    summary = json.loads(full_bytes)
    summary["artifact_projection"] = "tracked-summary-v1"
    summary["full_payload_record"] = content_record(full_bytes)
    for corpus_results in summary.get("results", {}).values():
        for audit in corpus_results.values():
            exposure = audit.get("exposure", {})
            if "matrix" in exposure:
                matrix = exposure.pop("matrix")
                exposure["matrix_shape"] = [len(matrix), len(matrix)]
                exposure["matrix_record"] = _canonical_value_record(matrix)
            for size_result in audit.get("registered_size_results", {}).values():
                if "capacities_by_region" in size_result:
                    capacities = size_result.pop("capacities_by_region")
                    size_result["capacities_by_region_record"] = (
                        _canonical_value_record(capacities)
                    )
                allocation = size_result.get("allocation")
                if not isinstance(allocation, dict):
                    continue
                for key in ("counts_by_region", "assignment_region_ids"):
                    if key in allocation:
                        value = allocation.pop(key)
                        allocation[f"{key}_record"] = _canonical_value_record(value)
    return summary


def _positive_unique_ints(values, label):
    values = tuple(values)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError(f"{label} must contain positive non-bool integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{label} must be increasing")
    return values


def _rho_grid(values):
    values = tuple(values)
    if not values:
        raise ValueError("rho_grid must be nonempty")
    output = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("rho_grid must contain finite numbers in [0,1)")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("rho_grid must contain finite numbers in [0,1)") from exc
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("rho_grid must contain finite numbers in [0,1)")
        output.append(value)
    output = tuple(output)
    if len(set(output)) != len(output):
        raise ValueError("rho_grid must be unique")
    if tuple(sorted(output)) != output:
        raise ValueError("rho_grid must be increasing")
    if output[0] != 0.0:
        raise ValueError("rho_grid must start at zero")
    return output


def _audit_passes(audit, *, corpus, region_count):
    if not isinstance(audit, dict):
        raise RuntimeError(
            f"{corpus} K={region_count} source-dependence audit is not an object"
        )
    gates = audit.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError(
            f"{corpus} K={region_count} source-dependence audit lacks gates"
        )
    passed = gates.get("all_registered_sizes_pass")
    if not isinstance(passed, bool):
        raise RuntimeError(
            f"{corpus} K={region_count} all_registered_sizes_pass must be boolean"
        )
    return passed


def build_source_dependence_payload(
    graphs,
    graph_inputs,
    *,
    region_count_grid=DEFAULT_REGION_COUNT_GRID,
    registered_sizes=REGISTERED_COMPONENT_SIZES,
    rho_grid=FROZEN_RHO_GRID,
    implementation=None,
    loader_metadata=None,
):
    """Build portable topology-only source-dependence audit JSON."""
    if set(graphs) != set(REQUIRED_CORPORA):
        raise ValueError("graphs must contain exactly exploratory and fresh")
    if set(graph_inputs) != set(REQUIRED_CORPORA):
        raise ValueError("graph_inputs must align exactly with graphs")
    region_count_grid = _positive_unique_ints(
        region_count_grid, "region_count_grid"
    )
    registered_sizes = _positive_unique_ints(
        registered_sizes, "registered_sizes"
    )
    rho_grid = _rho_grid(rho_grid)

    verify_live_implementation = implementation is None
    implementation_snapshot = (
        _implementation_records() if implementation is None else implementation
    )
    results = {
        corpus: {
            str(region_count): audit_source_dependence(
                graphs[corpus]["parents"],
                graphs[corpus]["children"],
                region_count,
                registered_sizes,
                rho_grid=rho_grid,
            )
            for region_count in region_count_grid
        }
        for corpus in REQUIRED_CORPORA
    }

    joint_grid = {}
    jointly_passing_region_counts = []
    for region_count in region_count_grid:
        key = str(region_count)
        passing_corpora = [
            corpus
            for corpus in REQUIRED_CORPORA
            if _audit_passes(
                results[corpus][key],
                corpus=corpus,
                region_count=region_count,
            )
        ]
        joint_passes = len(passing_corpora) == len(REQUIRED_CORPORA)
        joint_grid[key] = {
            "passing_corpora": passing_corpora,
            "required_corpora": list(REQUIRED_CORPORA),
            "joint_narrow_structural_gate_passes": joint_passes,
        }
        if joint_passes:
            jointly_passing_region_counts.append(region_count)

    bridge_defined = bool(jointly_passing_region_counts)
    if bridge_defined:
        status = "NO-SPEND SOURCE-DEPENDENCE STRUCTURAL BRIDGE DEFINED; DOWNSTREAM BLOCKED"
        reason = (
            "at least one identical frozen region count passes the narrow full-region "
            "capacity, allocation, and PSD gates in both required corpora; this audit "
            "does not include full-procedure source-dependent null calibration or power"
        )
    else:
        status = "NO-SPEND SOURCE-DEPENDENCE STRUCTURAL BRIDGE FAILED; DOWNSTREAM BLOCKED"
        reason = (
            "no identical frozen region count passes every narrow full-region structural "
            "gate in both required corpora"
        )

    metadata = loader_metadata or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "algorithm": ALGORITHM,
        "authorization": {
            "historical_inventory_unlocked": False,
            "candidate_enumeration_unlocked": False,
            "nomic_embedding_unlocked": False,
            "candidate_builder_unlocked": False,
            "judge_calls_authorized": False,
            "live_campaign_authorized": False,
            "covariance_deployment_unlocked": False,
            "independent_batching_unlocked": False,
            "qr_specialization_unlocked": False,
            "cuda_claim_unlocked": False,
        },
        "configuration": {
            "required_corpora": list(REQUIRED_CORPORA),
            "registered_components_per_corpus": list(registered_sizes),
            "region_count_grid": list(region_count_grid),
            "rho_grid": list(rho_grid),
            "region_count_selection_performed": False,
            "all_jointly_passing_region_counts_continue": True,
            "partition_inputs": "canonical undirected graph topology only",
            "full_source_regions_used_without_halo_exclusion": True,
            "source_region_cap_fraction": SOURCE_COMPONENT_CAP_FRACTION,
            "endpoints_charged_per_campaign_component": ENDPOINTS_PER_COMPONENT,
            "cumulative_walk_weights": list(FROZEN_WALK_WEIGHTS),
            "source_regions_claimed_independent": False,
            "true_weak_component_retained_as_separate_diagnostic": True,
            "graph_inputs_required_immutable_during_run": True,
            "external_graph_records_recheck_required_before_write": True,
        },
        "numerical_runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "blas_runtime": _blas_runtime_identity(),
            "blas_thread_limit_policy": (
                "CLI audit requests one BLAS thread via threadpoolctl when available"
            ),
        },
        "implementation_and_protocol": implementation_snapshot,
        "inputs": {
            "graph_bundle": _graph_bundle_record(graph_inputs),
            "graphs": graph_inputs,
            "outcomes_consumed": False,
            "historical_inventory_consumed": False,
            "nomic_cache_consumed": False,
            "candidate_pool_consumed": False,
            "judge_responses_consumed": False,
        },
        "loader_metadata": {
            corpus: dict(metadata.get(corpus, {}))
            for corpus in REQUIRED_CORPORA
        },
        "results": results,
        "joint_region_count_grid": joint_grid,
        "decision": {
            "structural_bridge_defined": bridge_defined,
            "jointly_passing_region_counts": jointly_passing_region_counts,
            "region_count_selected": None,
            "downstream_pipeline_must_stop": True,
            "candidate_builder_must_stop": True,
            "reason": reason,
            "required_next_action": (
                "extend the complete repeated-judge null and power procedure with "
                "source-atomic folds, registered source exposure, prompt incidence, and "
                "two-way prompt/source inference before unlocking history or candidates"
            ),
        },
        "non_claims": [
            "source regions are concentration and dependence-sensitivity units, not independent observations",
            "region-average topology exposure is not empirical residual covariance or a candidate-level upper bound",
            "effective component counts are information diagnostics, not power calculations",
            "the greedy allocation is a prospective diagnostic quota, not exact candidate packability",
            "no region count is selected or ranked by this audit",
            "no historical attempted-input identity inventory was consumed or unlocked",
            "no embedding model, candidate builder, judge, covariance conditioner, QR specialization, or CUDA kernel ran",
        ],
    }
    if verify_live_implementation:
        _assert_implementation_unchanged(implementation_snapshot)
    return payload


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = _json_bytes(payload)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return content_record(data)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lmdb-no-lock", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "write a reviewable projection with content records for bulky exact "
            "matrices/quotas; scientific computation and decisions are unchanged"
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="return zero after the blocked audit; does not unlock anything",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    implementation = _implementation_records()
    graphs, graph_inputs, metadata = load_frozen_graphs(
        args.artifact_repo, lmdb_no_lock=args.lmdb_no_lock
    )
    with _single_blas_thread_context():
        payload = build_source_dependence_payload(
            graphs,
            graph_inputs,
            implementation=implementation,
            loader_metadata=metadata,
        )
    del graphs
    end_graphs, end_graph_inputs, end_metadata = load_frozen_graphs(
        args.artifact_repo, lmdb_no_lock=args.lmdb_no_lock
    )
    del end_graphs
    _assert_graph_inputs_unchanged(graph_inputs, end_graph_inputs)
    if end_metadata != metadata:
        raise RuntimeError(
            "external graph loader metadata changed during source-dependence audit"
        )
    _assert_implementation_unchanged(implementation)
    output_payload = _summary_projection(payload) if args.summary_only else payload
    artifact = _atomic_write(args.out, output_payload)
    print(
        json.dumps(
            {
                "output": os.path.abspath(args.out),
                "artifact": artifact,
                "artifact_projection": output_payload.get(
                    "artifact_projection", "complete"
                ),
                "full_payload_record": (
                    output_payload.get("full_payload_record")
                    if args.summary_only
                    else content_record(_json_bytes(payload))
                ),
                "structural_bridge_defined": payload["decision"][
                    "structural_bridge_defined"
                ],
                "jointly_passing_region_counts": payload["decision"][
                    "jointly_passing_region_counts"
                ],
                "downstream_pipeline_must_stop": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.audit_only:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
