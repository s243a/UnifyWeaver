#!/usr/bin/env python3
"""Audit repeated-judge structural capacity before candidates or Nomic.

The command uses the repository's canonical graph loaders and emits a portable,
content-addressed JSON result.  It consumes no scores, residuals, embeddings, or
judge responses and cannot authorize a live call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from repeated_judge_campaign import content_record
from repeated_judge_candidate_capacity import (
    ENDPOINTS_PER_COMPONENT,
    GUARANTEED_SOURCE_ENDPOINTS_PER_COMPONENT,
    SOURCE_COMPONENT_CAP_FRACTION,
    audit_graph_capacity,
)


SCHEMA_VERSION = 1
ALGORITHM = "repeated-judge-candidate-capacity-preflight-v1"
REGISTERED_COMPONENT_SIZES = (160, 320, 512, 800)
IMPLEMENTATION_FILES = (
    "repeated_judge_candidate_capacity.py",
    "run_repeated_judge_candidate_capacity.py",
    "repeated_judge_campaign.py",
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


def _graph_bundle_record(graph_inputs):
    canonical = json.dumps(
        graph_inputs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return content_record(canonical)


def build_capacity_payload(
    graphs,
    graph_inputs,
    *,
    registered_sizes=REGISTERED_COMPONENT_SIZES,
    implementation=None,
    loader_metadata=None,
):
    """Build path-free scientific JSON from already-loaded graph maps."""
    if set(graphs) != {"exploratory", "fresh"}:
        raise ValueError("graphs must contain exactly exploratory and fresh")
    if set(graph_inputs) != set(graphs):
        raise ValueError("graph_inputs must align exactly with graphs")
    registered_sizes = tuple(registered_sizes)
    if not registered_sizes:
        raise ValueError("registered_sizes must be non-empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in registered_sizes
    ):
        raise ValueError("registered_sizes must contain positive non-bool integers")
    if len(set(registered_sizes)) != len(registered_sizes):
        raise ValueError("registered_sizes must be unique")
    results = {
        corpus: audit_graph_capacity(
            graphs[corpus]["parents"],
            graphs[corpus]["children"],
            registered_sizes,
            cap_fraction=SOURCE_COMPONENT_CAP_FRACTION,
        )
        for corpus in ("exploratory", "fresh")
    }
    joint_grid = {}
    for size in registered_sizes:
        key = str(size)
        passing = [
            corpus
            for corpus in ("exploratory", "fresh")
            if results[corpus]["registered_grid"][key][
                "necessary_capacity_gate_passes"
            ]
        ]
        joint_grid[key] = {
            "passing_corpora": passing,
            "required_corpora": ["exploratory", "fresh"],
            "joint_necessary_capacity_gate_passes": len(passing) == 2,
        }
    all_feasible = all(
        value["joint_necessary_capacity_gate_passes"]
        for value in joint_grid.values()
    )
    metadata = loader_metadata or {}
    portable_metadata = {
        corpus: dict(metadata.get(corpus, {}))
        for corpus in ("exploratory", "fresh")
    }
    if all_feasible:
        status = "NO-SPEND STRUCTURAL CAPACITY PREFLIGHT PASSED; ENUMERATION ELIGIBLE"
        reason = (
            "the conservative one-endpoint/source-cap upper bound reaches G for every "
            "registered size; this necessary gate does not establish actual packability"
        )
        required_next_action = (
            "proceed only to the no-spend historical inventory and structural enumeration; "
            "judge calls and scientific deployment remain unauthorized"
        )
    else:
        status = "NO-SPEND STRUCTURAL CAPACITY PREFLIGHT; CANDIDATE BUILDER BLOCKED"
        reason = (
            "the conservative one-endpoint/source-cap upper bound is below G for at least "
            "one registered size in at least one required corpus; hop cells, degree matching, "
            "graph/Nomic agreement, history exclusions, and actual packing can only "
            "reduce capacity when source components remain frozen"
        )
        required_next_action = (
            "amend the preregistered dependency/source partition outcome-blindly; "
            "do not silently relabel branches as connected components or relax the cap"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "algorithm": ALGORITHM,
        "authorization": {
            "candidate_enumeration_unlocked": all_feasible,
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
            "source_definition": "weak connected component of the canonical retained graph",
            "source_component_cap_fraction": SOURCE_COMPONENT_CAP_FRACTION,
            "distinct_graph_endpoints_per_campaign_component": ENDPOINTS_PER_COMPONENT,
            "guaranteed_endpoints_charged_to_declared_source": (
                GUARANTEED_SOURCE_ENDPOINTS_PER_COMPONENT
            ),
            "source_components_frozen_before_endpoint_exclusions": True,
        },
        "implementation": (
            _implementation_records() if implementation is None else implementation
        ),
        "inputs": {
            "graph_bundle": _graph_bundle_record(graph_inputs),
            "graphs": graph_inputs,
            "outcomes_consumed": False,
            "historical_inventory_consumed": False,
            "nomic_cache_consumed": False,
        },
        "loader_metadata": portable_metadata,
        "results": results,
        "joint_grid": joint_grid,
        "decision": {
            "all_registered_sizes_feasible": all_feasible,
            "capacity_gate_passed": all_feasible,
            "candidate_builder_must_stop": not all_feasible,
            "reason": reason,
            "required_next_action": required_next_action,
        },
        "non_claims": [
            "this preflight did not enumerate or select candidate triples",
            "this preflight did not create or consume a Nomic embedding cache",
            "this preflight did not build a historical endpoint inventory",
            "this preflight did not consume judge scores or residuals",
            "an optimistic upper bound is not evidence that a passing graph is actually packable",
            "no branch/source partition has been substituted for connected components",
            "the sharper four-endpoint sensitivity is not used as the blocking proof because the preregistration permits a disconnected distant comparator",
        ],
    }
    return payload


def load_frozen_graphs(artifact_repo, *, lmdb_no_lock=False):
    """Load both frozen corpora and return portable content provenance."""
    from run_product_kalman_realdata import DATASETS
    from run_structured_residual_covariance import configure_artifact_repo
    from sigma_hop_confirmatory import FeatureGraphConfig, load_feature_graph

    artifacts = configure_artifact_repo(artifact_repo)
    DATASETS["fresh"]["graph"]["lmdb_no_lock"] = bool(lmdb_no_lock)
    graphs = {}
    metadata = {}
    for corpus in ("exploratory", "fresh"):
        parents, children, _degree, raw_metadata = load_feature_graph(
            FeatureGraphConfig(**DATASETS[corpus]["graph"])
        )
        graphs[corpus] = {"parents": parents, "children": children}
        if corpus == "fresh":
            metadata[corpus] = {
                "feature_graph_source": raw_metadata.get("feature_graph_source"),
                "feature_graph_root": raw_metadata.get("feature_graph_root"),
                "feature_graph_slice_nodes": raw_metadata.get("feature_graph_slice_nodes"),
                "feature_graph_lmdb_stats": raw_metadata.get("feature_graph_lmdb_stats", {}),
            }
        else:
            metadata[corpus] = {
                "feature_graph_source": "child-parent TSV",
            }
    exploratory_record = {
        key: artifacts["exploratory_graph"][key]
        for key in ("size_bytes", "sha256")
    }
    fresh_record = {
        key: artifacts["fresh_lmdb_data"][key]
        for key in ("size_bytes", "sha256")
    }
    exploratory_metadata = (
        Path(artifact_repo) / "data" / "benchmark" / "100k_cats" / "metadata.json"
    )
    if not exploratory_metadata.is_file():
        raise FileNotFoundError(
            f"missing exploratory dataset metadata: {exploratory_metadata}"
        )
    graph_inputs = {
        "exploratory": {
            "corpus_identity": "SimpleWiki 100k category graph",
            "format": "UTF-8 child-parent TSV",
            "artifacts": {
                "category_parent": exploratory_record,
                "dataset_metadata": content_record(exploratory_metadata.read_bytes()),
            },
        },
        "fresh": {
            "corpus_identity": "enwiki category LMDB retained under Behavior",
            "format": "LMDB uint32 graph with UTF-8 title layer",
            "artifacts": {
                "category_lmdb_data": fresh_record,
                "exploratory_title_exclusion_graph": exploratory_record,
            },
            "excluded_runtime_artifact": artifacts["fresh_lmdb_lock_excluded"],
        },
    }
    return graphs, graph_inputs, metadata


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
        "--require-feasible",
        action="store_true",
        help="return status 2 when the necessary structural capacity gate fails",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    graphs, graph_inputs, metadata = load_frozen_graphs(
        args.artifact_repo, lmdb_no_lock=args.lmdb_no_lock
    )
    payload = build_capacity_payload(graphs, graph_inputs, loader_metadata=metadata)
    artifact = _atomic_write(args.out, payload)
    print(json.dumps({
        "output": os.path.abspath(args.out),
        "artifact": artifact,
        "candidate_builder_must_stop": payload["decision"]["candidate_builder_must_stop"],
    }, indent=2, sort_keys=True))
    if args.require_feasible and payload["decision"]["candidate_builder_must_stop"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
