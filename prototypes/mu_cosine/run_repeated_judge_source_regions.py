#!/usr/bin/env python3
"""Audit deterministic topology-only source regions before candidate work.

This command consumes only the two frozen canonical graphs.  It does not read
historical labels, enumerate candidate triples, generate embeddings, or call a
judge.  A failed topology gate writes its complete audit and exits with status
2; ``--audit-only`` is the explicit reporting-mode escape hatch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from repeated_judge_campaign import FROZEN_WALK_WEIGHTS, content_record
from repeated_judge_source_regions import (
    CUMULATIVE_WALK_SUPPORT_RADIUS,
    DEFAULT_HALO_HOPS,
    DEFAULT_MIN_CORE_FRACTION,
    DEFAULT_MIN_EFFECTIVE_REGIONS,
    DEFAULT_REGION_COUNT_GRID,
    SOURCE_REGION_CAP_FRACTION,
    audit_source_region_partition,
)
from run_repeated_judge_candidate_capacity import (
    REGISTERED_COMPONENT_SIZES,
    load_frozen_graphs,
)


SCHEMA_VERSION = 1
ALGORITHM = "repeated-judge-topology-source-region-audit-v1"
IMPLEMENTATION_FILES = (
    "repeated_judge_source_regions.py",
    "run_repeated_judge_source_regions.py",
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
            "implementation provenance changed during source-region audit: "
            + ", ".join(changed)
        )


def _graph_bundle_record(graph_inputs):
    canonical = json.dumps(
        graph_inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return content_record(canonical)


def _positive_unique_ints(values, label):
    values = tuple(values)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError(f"{label} must contain positive non-bool integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def build_source_region_payload(
    graphs,
    graph_inputs,
    *,
    region_count_grid=DEFAULT_REGION_COUNT_GRID,
    registered_sizes=REGISTERED_COMPONENT_SIZES,
    halo_hops=DEFAULT_HALO_HOPS,
    min_core_fraction=DEFAULT_MIN_CORE_FRACTION,
    min_effective_regions=DEFAULT_MIN_EFFECTIVE_REGIONS,
    implementation=None,
    loader_metadata=None,
):
    """Build portable source-region audit JSON from already-loaded graphs."""
    corpora = ("exploratory", "fresh")
    if set(graphs) != set(corpora):
        raise ValueError("graphs must contain exactly exploratory and fresh")
    if set(graph_inputs) != set(corpora):
        raise ValueError("graph_inputs must align exactly with graphs")
    region_count_grid = _positive_unique_ints(region_count_grid, "region_count_grid")
    registered_sizes = _positive_unique_ints(registered_sizes, "registered_sizes")
    if tuple(sorted(region_count_grid)) != region_count_grid:
        raise ValueError("region_count_grid must be increasing from coarsest to finest")

    verify_live_implementation = implementation is None
    implementation_snapshot = (
        _implementation_records() if implementation is None else implementation
    )
    results = {
        corpus: {
            str(region_count): audit_source_region_partition(
                graphs[corpus]["parents"],
                graphs[corpus]["children"],
                region_count,
                registered_sizes,
                halo_hops=halo_hops,
                min_core_fraction=min_core_fraction,
                min_effective_regions=min_effective_regions,
            )
            for region_count in region_count_grid
        }
        for corpus in corpora
    }
    joint_grid = {}
    selected_region_count = None
    for region_count in region_count_grid:
        key = str(region_count)
        passing = [
            corpus
            for corpus in corpora
            if results[corpus][key]["gates"]["all_topology_gates_pass"]
        ]
        passes = len(passing) == len(corpora)
        joint_grid[key] = {
            "passing_corpora": passing,
            "required_corpora": list(corpora),
            "joint_topology_gate_passes": passes,
        }
        if selected_region_count is None and passes:
            selected_region_count = region_count

    topology_passed = selected_region_count is not None
    if topology_passed:
        status = "NO-SPEND SOURCE-REGION TOPOLOGY GATE PASSED; HISTORY INVENTORY ELIGIBLE"
        reason = (
            "the coarsest jointly passing frozen source-region count satisfies the "
            "three-hop core, effective-region, and optimistic four-endpoint capacity gates; "
            "this remains only a necessary structural result"
        )
        required_next_action = (
            "build the attempted-input historical inventory; independently migrate the "
            "selector to source_region plus weak_component_id before structural enumeration"
        )
    else:
        status = "NO-SPEND SOURCE-REGION TOPOLOGY AUDIT; DOWNSTREAM PIPELINE BLOCKED"
        reason = (
            "no preregistered region count passes every topology gate in both required "
            "corpora; later history, Nomic, and candidate filters cannot repair a failed "
            "core/capacity prerequisite for this frozen construction"
        )
        required_next_action = (
            "revise the outcome-blind topology partition family or powered dependence "
            "model under an explicit amendment; do not tune this grid or halo from outcomes"
        )

    metadata = loader_metadata or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "algorithm": ALGORITHM,
        "authorization": {
            "historical_inventory_unlocked": topology_passed,
            "candidate_enumeration_unlocked": False,
            "nomic_embedding_unlocked": False,
            "candidate_builder_unlocked": False,
            "judge_calls_authorized": False,
            "live_campaign_authorized": False,
            "covariance_deployment_unlocked": False,
            "qr_specialization_unlocked": False,
            "cuda_claim_unlocked": False,
        },
        "configuration": {
            "registered_components_per_corpus": list(registered_sizes),
            "region_count_grid_coarsest_to_finest": list(region_count_grid),
            "region_count_selection_rule": "first (coarsest) count passing jointly",
            "partition_inputs": "canonical undirected graph topology only",
            "partition_algorithm": "recursive metric-rooted spanning-tree cuts",
            "weak_component_region_allocation": (
                "start one per component, then repeatedly increment the eligible "
                "component maximizing node_count/current_regions; ties use canonical "
                "weak-component order"
            ),
            "source_region_cap_fraction": SOURCE_REGION_CAP_FRACTION,
            "graph_feature_support_radius_hops": halo_hops,
            "cumulative_walk_weights": list(FROZEN_WALK_WEIGHTS),
            "cumulative_walk_support_radius_hops": CUMULATIVE_WALK_SUPPORT_RADIUS,
            "core_definition": "B_radius(node) is wholly inside its assigned source region",
            "all_four_candidate_endpoints_must_share_one_core": True,
            "disconnected_distant_comparator_permitted": False,
            "minimum_core_node_fraction": min_core_fraction,
            "minimum_effective_regions_with_four_core_nodes": min_effective_regions,
            "true_weak_component_retained_as_separate_diagnostic": True,
            "source_regions_claimed_independent": False,
            "selector_source_region_schema_migration_complete": False,
            "graph_inputs_required_immutable_during_run": True,
        },
        "implementation": implementation_snapshot,
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
            corpus: dict(metadata.get(corpus, {})) for corpus in corpora
        },
        "results": results,
        "joint_region_count_grid": joint_grid,
        "decision": {
            "source_region_topology_gate_passed": topology_passed,
            "selected_region_count": selected_region_count,
            "candidate_builder_must_stop": True,
            "reason": reason,
            "required_next_action": required_next_action,
        },
        "non_claims": [
            "source regions are concentration, fold, and sensitivity units—not independent observations",
            "three-hop core separation applies only to radius-three graph-feature support",
            "Nomic, global, and within-weak-component dependence may cross source-region boundaries",
            "an optimistic U4 bound does not establish exact 32-cell candidate packability",
            "no historical attempted-input inventory or candidate universe was built",
            "no embedding model, judge, provider, covariance conditioner, QR specialization, or CUDA kernel ran",
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
        "--audit-only",
        action="store_true",
        help="return zero after a completed blocked audit; does not unlock anything",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    implementation = _implementation_records()
    graphs, graph_inputs, metadata = load_frozen_graphs(
        args.artifact_repo, lmdb_no_lock=args.lmdb_no_lock
    )
    payload = build_source_region_payload(
        graphs,
        graph_inputs,
        implementation=implementation,
        loader_metadata=metadata,
    )
    _assert_implementation_unchanged(implementation)
    artifact = _atomic_write(args.out, payload)
    print(
        json.dumps(
            {
                "output": os.path.abspath(args.out),
                "artifact": artifact,
                "source_region_topology_gate_passed": payload["decision"][
                    "source_region_topology_gate_passed"
                ],
                "selected_region_count": payload["decision"]["selected_region_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if (
        not args.audit_only
        and not payload["decision"]["source_region_topology_gate_passed"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
