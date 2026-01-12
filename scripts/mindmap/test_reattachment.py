#!/usr/bin/env python3
"""
Test reattachment criterion by artificially orphaning a branch.

This script:
1. Loads physics subset with curated hierarchy
2. Removes a specified branch (making nodes orphans)
3. Runs hybrid attachment with both semantic and integrated modes
4. Compares where nodes reattached and resulting tangent deviation

Usage:
    python3 scripts/mindmap/test_reattachment.py --branch 11457393  # Waves (physics)
    python3 scripts/mindmap/test_reattachment.py --branch 11477015  # mechanics
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.mindmap.mst_folder_grouping import (
    MSTFolderGrouper,
    load_physics_subset,
    load_hierarchy_paths,
)


def find_subtree(branch_root: str, hierarchy_paths: Dict[str, str]) -> Set[str]:
    """Find all nodes in the subtree rooted at branch_root."""
    subtree = set()

    # Build parent-child relationships
    children = defaultdict(list)
    for tree_id, path in hierarchy_paths.items():
        parts = [p for p in path.strip('/').split('/') if p]
        if len(parts) >= 2:
            parent_id = parts[-2]
            children[parent_id].append(tree_id)

    # BFS to find all descendants
    queue = [branch_root]
    while queue:
        node = queue.pop(0)
        if node in hierarchy_paths:
            subtree.add(node)
        for child in children.get(node, []):
            queue.append(child)

    return subtree


def remove_branch_from_hierarchy(
    hierarchy_paths: Dict[str, str],
    branch_nodes: Set[str]
) -> Dict[str, str]:
    """Remove branch nodes from hierarchy (making them orphans)."""
    return {k: v for k, v in hierarchy_paths.items() if k not in branch_nodes}


def run_reattachment_test(
    branch_root: str,
    embeddings: np.ndarray,
    titles: List[str],
    tree_ids: List[str],
    output_embeddings: np.ndarray,
    hierarchy_paths: Dict[str, str],
    verbose: bool = True
) -> Dict:
    """Run reattachment test comparing semantic vs integrated modes."""

    # Find subtree to orphan
    branch_nodes = find_subtree(branch_root, hierarchy_paths)

    if not branch_nodes:
        print(f"Error: Branch root {branch_root} not found or has no descendants")
        return {}

    # Get titles for branch nodes
    tree_id_to_idx = {tid: i for i, tid in enumerate(tree_ids)}
    branch_titles = []
    for node_id in branch_nodes:
        if node_id in tree_id_to_idx:
            idx = tree_id_to_idx[node_id]
            branch_titles.append((node_id, titles[idx]))

    if verbose:
        print(f"\n{'='*60}")
        print(f"REATTACHMENT TEST: {branch_root}")
        print(f"{'='*60}")
        print(f"\nBranch to orphan ({len(branch_nodes)} nodes):")
        for node_id, title in branch_titles[:10]:
            print(f"  - {node_id}: {title[:50]}")
        if len(branch_titles) > 10:
            print(f"  ... and {len(branch_titles) - 10} more")

    # Create modified hierarchy (without branch)
    modified_hierarchy = remove_branch_from_hierarchy(hierarchy_paths, branch_nodes)

    if verbose:
        print(f"\nOriginal hierarchy: {len(hierarchy_paths)} paths")
        print(f"Modified hierarchy: {len(modified_hierarchy)} paths (removed {len(branch_nodes)})")

    results = {}

    # Run both modes
    for mode in ['semantic', 'integrated']:
        if verbose:
            print(f"\n{'-'*40}")
            print(f"Running {mode.upper()} mode...")
            print(f"{'-'*40}")

        grouper = MSTFolderGrouper(
            embeddings=embeddings,
            titles=titles,
            tree_ids=tree_ids,
            target_size=8,
            max_depth=4,
            min_size=2,
            verbose=verbose,
            tree_source='hybrid',
            hierarchy_paths=modified_hierarchy,
            output_embeddings=output_embeddings,
            embed_blend=0.3,
            attachment_cost=mode,
            tangent_lambda=1.0
        )

        # Build the tree (this triggers attachment)
        grouper.partition()

        # Analyze where orphans attached
        attachments = {}
        for node_id in branch_nodes:
            if node_id not in tree_id_to_idx:
                continue
            idx = tree_id_to_idx[node_id]
            neighbors = grouper.mst_adjacency.get(idx, [])
            if neighbors:
                # Find the attachment point (neighbor not in branch)
                for neighbor_idx, weight in neighbors:
                    neighbor_id = tree_ids[neighbor_idx]
                    if neighbor_id not in branch_nodes:
                        attachments[node_id] = {
                            'attached_to': neighbor_id,
                            'attached_to_title': titles[neighbor_idx],
                            'weight': weight
                        }
                        break

        # Compute tangent deviation
        deviation = grouper.compute_tangent_deviation()

        results[mode] = {
            'attachments': attachments,
            'mean_deviation': deviation['mean_deviation'],
            'max_deviation': deviation['max_deviation'],
            'n_nodes_compared': deviation['n_nodes_compared']
        }

        if verbose:
            print(f"\nAttachment results ({mode}):")
            for node_id, att in list(attachments.items())[:5]:
                node_title = titles[tree_id_to_idx[node_id]]
                print(f"  {node_title[:30]:<30} -> {att['attached_to_title'][:30]}")
            if len(attachments) > 5:
                print(f"  ... and {len(attachments) - 5} more attachments")

            print(f"\nTangent deviation ({mode}):")
            print(f"  Mean: {deviation['mean_deviation']:.6f}")
            print(f"  Max:  {deviation['max_deviation']:.6f}")

    # Compare results
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        # Check if attachments differ
        sem_att = results['semantic']['attachments']
        int_att = results['integrated']['attachments']

        same_attachments = 0
        diff_attachments = 0
        for node_id in branch_nodes:
            if node_id in sem_att and node_id in int_att:
                if sem_att[node_id]['attached_to'] == int_att[node_id]['attached_to']:
                    same_attachments += 1
                else:
                    diff_attachments += 1
                    print(f"\nDifferent attachment for {node_id}:")
                    print(f"  Semantic:   -> {sem_att[node_id]['attached_to_title'][:40]}")
                    print(f"  Integrated: -> {int_att[node_id]['attached_to_title'][:40]}")

        print(f"\nAttachment summary:")
        print(f"  Same attachment point: {same_attachments}")
        print(f"  Different attachment:  {diff_attachments}")

        print(f"\nTangent deviation comparison:")
        print(f"  Semantic mean:   {results['semantic']['mean_deviation']:.6f}")
        print(f"  Integrated mean: {results['integrated']['mean_deviation']:.6f}")

        sem_dev = results['semantic']['mean_deviation']
        int_dev = results['integrated']['mean_deviation']
        if sem_dev > 0:
            improvement = (sem_dev - int_dev) / sem_dev * 100
            print(f"  Improvement:     {improvement:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test reattachment criterion by orphaning a branch"
    )
    parser.add_argument('--branch', type=str, required=True,
                        help='Tree ID of branch root to orphan')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    # Load physics subset
    project_root = Path(__file__).parent.parent.parent
    embeddings_path = project_root / "datasets/pearltrees_combined_2026-01-02_all_fixed_embeddings.npz"
    targets_path = project_root / "reports/pearltrees_targets_physics_trees.jsonl"

    print("Loading physics subset...")
    embeddings, titles, tree_ids, output_embeddings = load_physics_subset(
        embeddings_path, targets_path, include_output=True
    )
    hierarchy_paths = load_hierarchy_paths(targets_path)

    print(f"Loaded {len(tree_ids)} items with {len(hierarchy_paths)} hierarchy paths")

    # Run test
    results = run_reattachment_test(
        branch_root=args.branch,
        embeddings=embeddings,
        titles=titles,
        tree_ids=tree_ids,
        output_embeddings=output_embeddings,
        hierarchy_paths=hierarchy_paths,
        verbose=not args.quiet
    )

    # Output summary
    if results:
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Semantic mean deviation:   {results['semantic']['mean_deviation']:.6f}")
        print(f"Integrated mean deviation: {results['integrated']['mean_deviation']:.6f}")


if __name__ == "__main__":
    main()
