#!/usr/bin/env python3
"""
Test attachment criterion with foreign (non-curated) data.

This script:
1. Loads physics subset as the curated base tree
2. Finds items from OTHER accounts (true orphans) semantically close to physics
3. Attempts to attach them using semantic vs integrated modes
4. Compares where they attach and resulting tangent deviation

Usage:
    python3 scripts/mindmap/test_foreign_attachment.py --top-k 10
    python3 scripts/mindmap/test_foreign_attachment.py --top-k 20 --verbose
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.mindmap.mst_folder_grouping import (
    MSTFolderGrouper,
    load_hierarchy_paths,
)


def load_embeddings_with_metadata(
    embeddings_path: Path,
    targets_path: Path,
    physics_targets_path: Path
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[Dict], Set[str]]:
    """Load embeddings and identify physics vs foreign items.

    Returns:
        input_embeddings: Full input embeddings matrix
        output_embeddings: Full output embeddings matrix
        all_tree_ids: List of all tree IDs
        all_titles: List of all titles
        foreign_items: List of dicts for items NOT in physics and NOT in s243a
        physics_ids: Set of physics tree IDs
    """
    # Load physics IDs
    physics_ids = set()
    with open(physics_targets_path) as f:
        for line in f:
            d = json.loads(line)
            physics_ids.add(d.get("tree_id"))

    # Load all targets with account info
    all_targets = {}
    with open(targets_path) as f:
        for line in f:
            d = json.loads(line)
            tree_id = d.get("tree_id")
            all_targets[tree_id] = d

    # Load embeddings
    data = np.load(embeddings_path, allow_pickle=True)
    input_emb = data['input_nomic']  # Input embeddings
    output_emb = data['output_nomic']  # Output embeddings
    tree_ids = list(data['tree_ids'])
    titles = list(data['titles'])

    # Identify foreign items (not in physics, not in s243a accounts)
    foreign_items = []
    s243a_accounts = {"s243a", "s243a_groups"}

    for i, tree_id in enumerate(tree_ids):
        if tree_id in physics_ids:
            continue

        target = all_targets.get(tree_id, {})
        account = target.get("account", "unknown")

        if account not in s243a_accounts:
            foreign_items.append({
                "idx": i,
                "tree_id": tree_id,
                "title": titles[i],
                "account": account,
                "target_text": target.get("target_text", "")
            })

    return input_emb, output_emb, tree_ids, titles, foreign_items, physics_ids


def find_closest_foreign_to_physics(
    input_embeddings: np.ndarray,
    physics_indices: List[int],
    foreign_items: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    """Find foreign items closest to physics centroid."""

    # Compute physics centroid
    physics_emb = input_embeddings[physics_indices]

    # Normalize for cosine similarity
    norms = np.linalg.norm(physics_emb, axis=1, keepdims=True)
    physics_normalized = physics_emb / (norms + 1e-8)
    centroid = physics_normalized.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    # Compute similarity of each foreign item to centroid
    foreign_indices = [f["idx"] for f in foreign_items]
    foreign_emb = input_embeddings[foreign_indices]

    norms = np.linalg.norm(foreign_emb, axis=1, keepdims=True)
    foreign_normalized = foreign_emb / (norms + 1e-8)

    similarities = foreign_normalized @ centroid

    # Get top-k closest
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    closest = []
    for idx in top_indices:
        item = foreign_items[idx].copy()
        item["similarity_to_physics"] = float(similarities[idx])
        closest.append(item)

    return closest


def run_foreign_attachment_test(
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    tree_ids: List[str],
    titles: List[str],
    physics_ids: Set[str],
    foreign_orphans: List[Dict],
    hierarchy_paths: Dict[str, str],
    verbose: bool = True
) -> Dict:
    """Run attachment test with foreign orphans."""

    # Build index mappings
    tree_id_to_idx = {tid: i for i, tid in enumerate(tree_ids)}
    physics_indices = [tree_id_to_idx[tid] for tid in physics_ids if tid in tree_id_to_idx]

    # Create subset: physics + foreign orphans
    subset_indices = physics_indices + [f["idx"] for f in foreign_orphans]
    subset_embeddings = input_embeddings[subset_indices]
    subset_output_emb = output_embeddings[subset_indices]
    subset_tree_ids = [tree_ids[i] for i in subset_indices]
    subset_titles = [titles[i] for i in subset_indices]

    # Hierarchy paths only for physics (foreign have none)
    subset_hierarchy = {tid: path for tid, path in hierarchy_paths.items()
                       if tid in physics_ids}

    if verbose:
        print(f"\n{'='*60}")
        print(f"FOREIGN ATTACHMENT TEST")
        print(f"{'='*60}")
        print(f"\nPhysics base: {len(physics_indices)} items")
        print(f"Foreign orphans to attach: {len(foreign_orphans)}")
        print(f"\nForeign items:")
        for f in foreign_orphans[:10]:
            print(f"  [{f['account']}] {f['title'][:50]} (sim: {f['similarity_to_physics']:.3f})")
        if len(foreign_orphans) > 10:
            print(f"  ... and {len(foreign_orphans) - 10} more")

    results = {}

    # Run both modes
    for mode in ['semantic', 'integrated']:
        if verbose:
            print(f"\n{'-'*40}")
            print(f"Running {mode.upper()} mode...")
            print(f"{'-'*40}")

        grouper = MSTFolderGrouper(
            embeddings=subset_embeddings,
            titles=subset_titles,
            tree_ids=subset_tree_ids,
            target_size=8,
            max_depth=4,
            min_size=2,
            verbose=verbose,
            tree_source='hybrid',
            hierarchy_paths=subset_hierarchy,
            output_embeddings=subset_output_emb,
            embed_blend=0.3,
            attachment_cost=mode,
            tangent_lambda=1.0
        )

        # Build tree (triggers attachment)
        grouper.partition()

        # Analyze where foreign items attached
        attachments = {}
        n_physics = len(physics_indices)

        for i, foreign in enumerate(foreign_orphans):
            subset_idx = n_physics + i  # Index in subset
            neighbors = grouper.mst_adjacency.get(subset_idx, [])

            if neighbors:
                # Find attachment point (should be a physics node)
                for neighbor_idx, weight in neighbors:
                    if neighbor_idx < n_physics:  # Attached to physics node
                        attachments[foreign["tree_id"]] = {
                            "attached_to_idx": neighbor_idx,
                            "attached_to_id": subset_tree_ids[neighbor_idx],
                            "attached_to_title": subset_titles[neighbor_idx],
                            "weight": weight,
                            "foreign_title": foreign["title"],
                            "foreign_account": foreign["account"]
                        }
                        break

        # Compute tangent deviation
        deviation = grouper.compute_tangent_deviation()

        results[mode] = {
            "attachments": attachments,
            "mean_deviation": deviation["mean_deviation"],
            "max_deviation": deviation["max_deviation"],
            "n_nodes_compared": deviation["n_nodes_compared"]
        }

        if verbose:
            print(f"\nAttachment results ({mode}):")
            for fid, att in list(attachments.items())[:5]:
                print(f"  [{att['foreign_account']}] {att['foreign_title'][:25]:<25} -> {att['attached_to_title'][:30]}")
            if len(attachments) > 5:
                print(f"  ... and {len(attachments) - 5} more")

            print(f"\nTangent deviation ({mode}):")
            print(f"  Mean: {deviation['mean_deviation']:.6f}")
            print(f"  Max:  {deviation['max_deviation']:.6f}")

    # Compare results
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        sem_att = results['semantic']['attachments']
        int_att = results['integrated']['attachments']

        same = 0
        diff = 0
        differences = []

        for fid in sem_att:
            if fid in int_att:
                if sem_att[fid]['attached_to_id'] == int_att[fid]['attached_to_id']:
                    same += 1
                else:
                    diff += 1
                    differences.append({
                        "foreign": sem_att[fid]['foreign_title'],
                        "semantic_to": sem_att[fid]['attached_to_title'],
                        "integrated_to": int_att[fid]['attached_to_title']
                    })

        print(f"\nAttachment summary:")
        print(f"  Same attachment point: {same}")
        print(f"  Different attachment:  {diff}")

        if differences:
            print(f"\nDifferences:")
            for d in differences[:5]:
                print(f"  {d['foreign'][:30]}:")
                print(f"    Semantic:   -> {d['semantic_to'][:40]}")
                print(f"    Integrated: -> {d['integrated_to'][:40]}")

        print(f"\nTangent deviation comparison:")
        print(f"  Semantic mean:   {results['semantic']['mean_deviation']:.6f}")
        print(f"  Integrated mean: {results['integrated']['mean_deviation']:.6f}")

        sem_dev = results['semantic']['mean_deviation']
        int_dev = results['integrated']['mean_deviation']
        if sem_dev > 0:
            improvement = (sem_dev - int_dev) / sem_dev * 100
            print(f"  Improvement:     {improvement:+.2f}%")

    results['differences'] = diff
    results['same'] = same

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test attachment criterion with foreign data"
    )
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of closest foreign items to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    embeddings_path = project_root / "datasets/pearltrees_combined_2026-01-02_all_fixed_embeddings.npz"
    targets_path = project_root / "reports/pearltrees_targets_combined_2026-01-02_trees.jsonl"
    physics_targets_path = project_root / "reports/pearltrees_targets_physics_trees.jsonl"

    print("Loading data...")
    input_emb, output_emb, tree_ids, titles, foreign_items, physics_ids = \
        load_embeddings_with_metadata(embeddings_path, targets_path, physics_targets_path)

    print(f"Loaded {len(tree_ids)} total items")
    print(f"Physics: {len(physics_ids)} items")
    print(f"Foreign (non-s243a): {len(foreign_items)} items")

    # Find closest foreign items to physics
    tree_id_to_idx = {tid: i for i, tid in enumerate(tree_ids)}
    physics_indices = [tree_id_to_idx[tid] for tid in physics_ids if tid in tree_id_to_idx]

    print(f"\nFinding top {args.top_k} foreign items closest to physics...")
    closest_foreign = find_closest_foreign_to_physics(
        input_emb, physics_indices, foreign_items, top_k=args.top_k
    )

    # Load hierarchy paths for physics
    hierarchy_paths = load_hierarchy_paths(physics_targets_path)

    # Run test
    results = run_foreign_attachment_test(
        input_emb, output_emb, tree_ids, titles,
        physics_ids, closest_foreign, hierarchy_paths,
        verbose=True
    )

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Foreign items tested: {args.top_k}")
    print(f"Same attachments: {results['same']}")
    print(f"Different attachments: {results['differences']}")
    print(f"Semantic mean deviation:   {results['semantic']['mean_deviation']:.6f}")
    print(f"Integrated mean deviation: {results['integrated']['mean_deviation']:.6f}")


if __name__ == "__main__":
    main()
