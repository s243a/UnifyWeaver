#!/usr/bin/env python3
"""Outcome-blind geometry redundancy/disagreement inventory on campaign rows.

No residual, judge score, or covariance outcome is consumed.  The campaign TSV
only supplies endpoint pairs and strata.  Results describe where graph,
shared-e5, MiniLM, and Nomic geometries agree or disagree so a later repeated-
judge campaign can sample informative cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from fine_tune_channel_heads import CAMPAIGN_E5_100K, load_campaign_datasets
from graph_geometry import (
    closed_neighborhood_kernel,
    cumulative_walk_feature_kernel,
    descendant_gated_item_kernel,
    embedding_item_kernel,
    median_pairwise_distance,
    role_aware_pair_features,
    walk_feature_kernel,
)
from independent_embedding_cache import MODEL_SPECS, load_embedding_cache
from run_product_kalman_realdata import DATASETS
from run_structured_residual_covariance import (
    configure_artifact_repo,
    semantic_pair_features,
)


def _content_record(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"size_bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def _undirected_neighbors(parents):
    output = {}
    for child, values in parents.items():
        output.setdefault(child, set()).update(values)
        for parent in values:
            output.setdefault(parent, set()).add(child)
    return {node: frozenset(values) for node, values in output.items()}


def _rbf_item_kernel(pairs, features):
    bandwidth = median_pairwise_distance(features)
    norm = np.sum(features * features, axis=1)
    squared = np.maximum(norm[:, None] + norm[None, :] - 2.0 * features @ features.T, 0.0)
    value = np.exp(-0.5 * squared / (bandwidth * bandwidth))
    left = [pair[0] for pair in pairs]
    value *= np.equal.outer(left, left)
    np.fill_diagonal(value, 1.0)
    return value, bandwidth


def _pearson(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left, right):
    return _pearson(rankdata(left, method="average"), rankdata(right, method="average"))


def summarize_geometry_inventory(pairs, kernels, neighbors):
    """Summarize within-descendant off-diagonal geometry without outcomes."""
    pairs = tuple(pairs)
    size = len(pairs)
    if any(np.asarray(value).shape != (size, size) for value in kernels.values()):
        raise ValueError("every kernel must align with pairs")
    rows, columns = np.triu_indices(size, 1)
    same_descendant = np.asarray([
        pairs[left][0] == pairs[right][0] for left, right in zip(rows, columns)
    ])
    rows, columns = rows[same_descendant], columns[same_descendant]
    if not len(rows):
        raise ValueError("inventory requires within-descendant row pairs")
    vectors = {
        name: 1.0 - np.asarray(kernel, dtype=float)[rows, columns]
        for name, kernel in kernels.items()
    }
    pearson = {
        left: {right: _pearson(vectors[left], vectors[right]) for right in vectors}
        for left in vectors
    }
    spearman = {
        left: {right: _spearman(vectors[left], vectors[right]) for right in vectors}
        for left in vectors
    }
    adjacent = np.asarray([
        pairs[right][1] in neighbors.get(pairs[left][1], ())
        or pairs[left][1] in neighbors.get(pairs[right][1], ())
        for left, right in zip(rows, columns)
    ])
    by_adjacency = {}
    for name, values in vectors.items():
        by_adjacency[name] = {
            "adjacent_mean_distance": float(np.mean(values[adjacent])) if np.any(adjacent) else None,
            "nonadjacent_mean_distance": (
                float(np.mean(values[~adjacent])) if np.any(~adjacent) else None
            ),
        }
    disagreement = {}
    graph_similarity = 1.0 - vectors["graph_walk_cumulative"]
    for name in ("minilm", "nomic", "shared_e5"):
        semantic_similarity = 1.0 - vectors[name]
        graph_high = graph_similarity >= np.quantile(graph_similarity, 0.75)
        graph_low = graph_similarity <= np.quantile(graph_similarity, 0.25)
        semantic_high = semantic_similarity >= np.quantile(semantic_similarity, 0.75)
        semantic_low = semantic_similarity <= np.quantile(semantic_similarity, 0.25)
        disagreement[name] = {
            "graph_high_semantic_low": int(np.sum(graph_high & semantic_low)),
            "graph_low_semantic_high": int(np.sum(graph_low & semantic_high)),
            "fraction": float(np.mean((graph_high & semantic_low) | (graph_low & semantic_high))),
        }
    return {
        "within_descendant_row_pairs": int(len(rows)),
        "directly_adjacent_root_pairs": int(np.sum(adjacent)),
        "distance_pearson": pearson,
        "distance_spearman": spearman,
        "mean_distance_by_graph_adjacency": by_adjacency,
        "graph_walk_quartile_disagreement": disagreement,
    }


def run_corpus(name, dataset, minilm, nomic):
    corpus = name.replace("-campaign", "")
    pairs = tuple(dataset["pairs"])
    if len(pairs) != len(set(pairs)):
        raise ValueError("geometry inventory requires unique campaign pairs")
    required = {node for pair in pairs for node in pair}
    minilm_missing = required - set(minilm.names)
    nomic_missing = required - set(nomic.names)
    if minilm_missing or nomic_missing:
        raise ValueError("independent embedding caches do not cover every campaign endpoint")
    neighbors = _undirected_neighbors(dataset["tok"].parents)
    roots = tuple(sorted({root for _left, root in pairs}))
    _, root_closed, _ = closed_neighborhood_kernel(roots, neighbors)
    _, root_walk_same_hop, _ = walk_feature_kernel(
        roots, neighbors, (1.0, 0.5, 0.25, 0.125)
    )
    _, root_walk_cumulative, _ = cumulative_walk_feature_kernel(
        roots, neighbors, (1.0, 0.5, 0.25, 0.125)
    )
    graph_closed = descendant_gated_item_kernel(pairs, roots, root_closed)
    graph_walk_same_hop = descendant_gated_item_kernel(
        pairs, roots, root_walk_same_hop
    )
    graph_walk_cumulative = descendant_gated_item_kernel(
        pairs, roots, root_walk_cumulative
    )

    minilm_map, nomic_map = minilm.by_name(), nomic.by_name()
    minilm_features = role_aware_pair_features(pairs, minilm_map)
    nomic_features = role_aware_pair_features(pairs, nomic_map)
    e5_features = semantic_pair_features(dataset, pairs)
    minilm_kernel = embedding_item_kernel(
        pairs,
        minilm_map,
        length_scale=median_pairwise_distance(minilm_features),
    )
    nomic_kernel = embedding_item_kernel(
        pairs,
        nomic_map,
        length_scale=median_pairwise_distance(nomic_features),
    )
    e5_kernel, e5_bandwidth = _rbf_item_kernel(pairs, e5_features)
    inventory = summarize_geometry_inventory(
        pairs,
        {
            "graph_closed": graph_closed,
            "graph_walk_same_hop": graph_walk_same_hop,
            "graph_walk_cumulative": graph_walk_cumulative,
            "minilm": minilm_kernel,
            "nomic": nomic_kernel,
            "shared_e5": e5_kernel,
        },
        neighbors,
    )
    inventory.update({
        "corpus": corpus,
        "campaign_rows": len(pairs),
        "unique_endpoints": len(required),
        "unique_roots": len(roots),
        "rbf_bandwidths": {
            "minilm": median_pairwise_distance(minilm_features),
            "nomic": median_pairwise_distance(nomic_features),
            "shared_e5": e5_bandwidth,
        },
    })
    return inventory


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo", required=True)
    parser.add_argument("--campaign", default="/tmp/mu_data/campaign_scored.tsv")
    parser.add_argument("--minilm-prefix", required=True)
    parser.add_argument("--nomic-prefix", required=True)
    parser.add_argument("--lmdb-no-lock", action="store_true")
    parser.add_argument("--out", default="/tmp/graph_geometry_inventory.json")
    return parser


def main():
    args = build_arg_parser().parse_args()
    started = time.perf_counter()
    artifacts = configure_artifact_repo(args.artifact_repo)
    if args.lmdb_no_lock:
        DATASETS["fresh"]["graph"]["lmdb_no_lock"] = True
    minilm = load_embedding_cache(args.minilm_prefix, expected_spec=MODEL_SPECS["minilm"])
    nomic = load_embedding_cache(args.nomic_prefix, expected_spec=MODEL_SPECS["nomic"])
    datasets = load_campaign_datasets(campaign_scored=args.campaign)
    results = [
        run_corpus(name, datasets[name], minilm, nomic)
        for name in ("exploratory-campaign", "fresh-campaign")
    ]
    root = os.path.dirname(os.path.abspath(__file__))
    payload = {
        "schema_version": 1,
        "status": "OUTCOME-BLIND GEOMETRY INVENTORY; NO RESIDUAL OR DEPLOYMENT CLAIM",
        "design": "DESIGN_graph_geometry_confirmatory.md",
        "implementation": {
            "design": _content_record(os.path.join(root, "DESIGN_graph_geometry_confirmatory.md")),
            "geometry": _content_record(os.path.join(root, "graph_geometry.py")),
            "cache": _content_record(os.path.join(root, "independent_embedding_cache.py")),
            "runner": _content_record(os.path.abspath(__file__)),
            "campaign_loader": _content_record(os.path.join(root, "fine_tune_channel_heads.py")),
            "graph_loader": _content_record(os.path.join(root, "run_product_kalman_realdata.py")),
            "pair_features": _content_record(
                os.path.join(root, "run_structured_residual_covariance.py")
            ),
        },
        "inputs": {
            "campaign": _content_record(args.campaign),
            "minilm_manifest_sha256": minilm.manifest_sha256,
            "nomic_manifest_sha256": nomic.manifest_sha256,
            "exploratory_e5": _content_record(CAMPAIGN_E5_100K),
            "fresh_e5": _content_record(DATASETS["fresh"]["e5_cache"]),
            "graph_artifacts": {
                "exploratory_graph": {
                    key: artifacts["exploratory_graph"][key] for key in ("size_bytes", "sha256")
                },
                "fresh_lmdb_data": {
                    key: artifacts["fresh_lmdb_data"][key] for key in ("size_bytes", "sha256")
                },
                "fresh_lmdb_lock_excluded": artifacts["fresh_lmdb_lock_excluded"],
            },
        },
        "configuration": {"lmdb_no_lock": bool(args.lmdb_no_lock)},
        "results": results,
        "deployment_gate_unlocked": False,
        "reason": "geometry redundancy and disagreement use no repeated residual fields",
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = os.path.abspath(args.out)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    os.replace(temporary, path)
    print(json.dumps({
        "output": path,
        "corpora": [
            {
                "corpus": row["corpus"],
                "within_descendant_row_pairs": row["within_descendant_row_pairs"],
                "graph_nomic_spearman": row["distance_spearman"][
                    "graph_walk_cumulative"
                ]["nomic"],
                "nomic_e5_spearman": row["distance_spearman"]["nomic"]["shared_e5"],
            }
            for row in results
        ],
        "wall_seconds": time.perf_counter() - started,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
