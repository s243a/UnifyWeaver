#!/usr/bin/env python3
"""
Test J-guided tree construction on Wikipedia physics data.

Compares:
1. Greedy MST (baseline)
2. Probability-ordered nearest neighbor
3. J-guided tree (our new algorithm)

Usage:
    python3 scripts/mindmap/test_j_guided_tree.py
    python3 scripts/mindmap/test_j_guided_tree.py --top-k 100 --use-bert
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts/mindmap"))

from hierarchy_objective import (
    JGuidedTreeBuilder,
    build_j_guided_tree,
    HierarchyObjective
)


def load_wikipedia_physics(top_k: int = 300) -> tuple:
    """Load Wikipedia physics embeddings."""
    npz_path = project_root / "datasets/wikipedia_physics.npz"

    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        print("Run: python3 scripts/fetch_wikipedia_physics.py --top-k 500")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)

    embeddings = data['embeddings'][:top_k]
    titles = list(data['titles'][:top_k])
    texts = list(data['texts'][:top_k]) if 'texts' in data else titles

    print(f"Loaded {len(embeddings)} articles ({embeddings.shape[1]}-dim embeddings)")

    return embeddings, titles, texts


def build_greedy_mst(embeddings: np.ndarray) -> dict:
    """Build greedy MST tree (baseline)."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    # Cosine distance
    similarity = emb_norm @ emb_norm.T
    cos_dist = 1 - similarity
    np.fill_diagonal(cos_dist, 0)

    # Build MST
    mst = minimum_spanning_tree(cos_dist)
    cx = mst.tocoo()

    # Convert to adjacency
    adj = {}
    for i, j, w in zip(cx.row, cx.col, cx.data):
        if i not in adj:
            adj[i] = []
        if j not in adj:
            adj[j] = []
        adj[i].append((j, w))
        adj[j].append((i, w))

    # Root at highest degree node
    degrees = [(len(adj.get(i, [])), i) for i in range(len(embeddings))]
    _, root = max(degrees)

    # BFS to create rooted tree
    parent = {root: None}
    children = {root: []}
    depth = {root: 0}
    visited = {root}
    queue = [root]

    while queue:
        node = queue.pop(0)
        for neighbor, _ in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                if node not in children:
                    children[node] = []
                children[node].append(neighbor)
                children[neighbor] = []
                queue.append(neighbor)

    return {"parent": parent, "children": children, "depth": depth, "root": root}


def build_probability_ordered_tree(embeddings: np.ndarray) -> dict:
    """Build tree with probability ordering but nearest-neighbor attachment."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    # Cosine distance
    similarity = emb_norm @ emb_norm.T
    cos_dist = 1 - similarity

    # Compute generality (similarity to centroid)
    centroid = emb_norm.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    generality = emb_norm @ centroid

    # Order by generality (high to low)
    order = np.argsort(-generality)

    # Build tree
    root = order[0]
    parent = {root: None}
    children = {root: []}
    depth = {root: 0}
    in_tree = {root}

    for node_idx in order[1:]:
        # Find nearest neighbor in tree
        tree_nodes = list(in_tree)
        dists_to_tree = cos_dist[node_idx, tree_nodes]
        nearest_in_tree = tree_nodes[np.argmin(dists_to_tree)]

        parent[node_idx] = nearest_in_tree
        depth[node_idx] = depth[nearest_in_tree] + 1
        if nearest_in_tree not in children:
            children[nearest_in_tree] = []
        children[nearest_in_tree].append(node_idx)
        children[node_idx] = []
        in_tree.add(node_idx)

    return {"parent": parent, "children": children, "depth": depth, "root": root}


def compute_depth_surprisal_correlation(tree: dict, embeddings: np.ndarray) -> tuple:
    """Compute correlation between depth and surprisal."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)

    # Compute probabilities from centroid similarity
    centroid = emb_norm.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    generality = emb_norm @ centroid

    # Softmax-like probabilities
    exp_gen = np.exp(generality / 0.5)
    probs = exp_gen / exp_gen.sum()

    # Compute correlation
    depths = np.array(list(tree["depth"].values()))
    surprisals = np.array([-np.log(probs[i] + 1e-10) for i in tree["depth"].keys()])

    if np.std(depths) < 1e-8:
        return 0.0, 0.0

    correlation = np.corrcoef(depths, surprisals)[0, 1]
    slope = np.polyfit(depths, surprisals, 1)[0]

    return correlation, slope


def compute_tree_stats(tree: dict, embeddings: np.ndarray) -> dict:
    """Compute statistics for a tree."""
    from collections import Counter

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)
    similarity = emb_norm @ emb_norm.T
    cos_dist = 1 - similarity

    # Depth stats
    depths = list(tree["depth"].values())
    depth_counts = Counter(depths)

    # Compute D (average parent-child distance)
    parent_child_dists = []
    for node, par in tree["parent"].items():
        if par is not None:
            parent_child_dists.append(cos_dist[node, par])

    D = np.mean(parent_child_dists) if parent_child_dists else 0

    # Depth-surprisal correlation
    corr, slope = compute_depth_surprisal_correlation(tree, embeddings)

    # Branching factor
    n_internal = sum(1 for n, c in tree["children"].items() if c)
    n_children_total = sum(len(c) for c in tree["children"].values())
    branching = n_children_total / n_internal if n_internal > 0 else 0

    return {
        "n_nodes": len(tree["depth"]),
        "max_depth": max(depths),
        "depth_counts": dict(depth_counts),
        "D": D,
        "depth_surprisal_corr": corr,
        "depth_surprisal_slope": slope,
        "branching_factor": branching
    }


def main():
    parser = argparse.ArgumentParser(description="Test J-guided tree construction")
    parser.add_argument("--top-k", type=int, default=300, help="Number of articles to use")
    parser.add_argument("--use-bert", action="store_true", help="Use BERT for entropy (slow)")
    parser.add_argument("--intermediate-threshold", type=float, default=0.5,
                        help="Threshold for intermediate node suggestions")
    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("Loading Wikipedia physics data...")
    embeddings, titles, texts = load_wikipedia_physics(args.top_k)

    # 1. Greedy MST baseline
    print("\n" + "=" * 60)
    print("Building Greedy MST (baseline)...")
    mst_tree = build_greedy_mst(embeddings)
    mst_stats = compute_tree_stats(mst_tree, embeddings)

    print(f"  Max depth: {mst_stats['max_depth']}")
    print(f"  Avg D: {mst_stats['D']:.4f}")
    print(f"  Depth-surprisal correlation: {mst_stats['depth_surprisal_corr']:.4f}")
    print(f"  Branching factor: {mst_stats['branching_factor']:.2f}")

    # 2. Probability-ordered nearest neighbor
    print("\n" + "=" * 60)
    print("Building Probability-Ordered Tree...")
    prob_tree = build_probability_ordered_tree(embeddings)
    prob_stats = compute_tree_stats(prob_tree, embeddings)

    print(f"  Max depth: {prob_stats['max_depth']}")
    print(f"  Avg D: {prob_stats['D']:.4f}")
    print(f"  Depth-surprisal correlation: {prob_stats['depth_surprisal_corr']:.4f}")
    print(f"  Branching factor: {prob_stats['branching_factor']:.2f}")

    # 3. J-guided tree
    print("\n" + "=" * 60)
    print("Building J-Guided Tree...")

    builder = JGuidedTreeBuilder(
        embeddings=embeddings,
        texts=texts if args.use_bert else None,
        titles=titles,
        use_bert_entropy=args.use_bert,
        intermediate_threshold=args.intermediate_threshold,
        verbose=True
    )

    j_tree = builder.build()
    j_corr, j_slope = builder.get_depth_surprisal_correlation()

    # Compute D for J-guided tree
    j_parent_child_dists = []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)
    similarity = emb_norm @ emb_norm.T
    cos_dist = 1 - similarity

    for node, par in builder.parent.items():
        if par is not None:
            j_parent_child_dists.append(cos_dist[node, par])
    j_D = np.mean(j_parent_child_dists) if j_parent_child_dists else 0

    # Branching factor
    n_internal = sum(1 for n, c in builder.children.items() if c)
    n_children_total = sum(len(c) for c in builder.children.values())
    j_branching = n_children_total / n_internal if n_internal > 0 else 0

    print(f"\n  Max depth: {max(builder.depth.values())}")
    print(f"  Avg D: {j_D:.4f}")
    print(f"  Depth-surprisal correlation: {j_corr:.4f}")
    print(f"  Branching factor: {j_branching:.2f}")
    print(f"  Entropy slope: {builder.entropy_slope:.4f}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'Max Depth':>10} {'Avg D':>10} {'Depth-Surp':>12} {'Branch':>10}")
    print("-" * 72)
    print(f"{'Greedy MST':<30} {mst_stats['max_depth']:>10} {mst_stats['D']:>10.4f} {mst_stats['depth_surprisal_corr']:>12.4f} {mst_stats['branching_factor']:>10.2f}")
    print(f"{'Probability-Ordered NN':<30} {prob_stats['max_depth']:>10} {prob_stats['D']:>10.4f} {prob_stats['depth_surprisal_corr']:>12.4f} {prob_stats['branching_factor']:>10.2f}")
    print(f"{'J-Guided':<30} {max(builder.depth.values()):>10} {j_D:>10.4f} {j_corr:>12.4f} {j_branching:>10.2f}")

    # Show intermediate suggestions
    if builder.intermediate_suggestions:
        print(f"\n{len(builder.intermediate_suggestions)} intermediate node suggestions:")
        for sugg in builder.intermediate_suggestions[:10]:
            print(f"  - {sugg['message'][:80]}")
        if len(builder.intermediate_suggestions) > 10:
            print(f"  ... and {len(builder.intermediate_suggestions) - 10} more")

    # Show root and top-level structure
    print("\n" + "=" * 60)
    print("J-GUIDED TREE STRUCTURE (top levels)")
    print("=" * 60)

    root_idx = [i for i, p in builder.parent.items() if p is None][0]
    print(f"Root: {titles[root_idx]}")

    # Show first 2 levels
    for child in builder.children.get(root_idx, [])[:5]:
        print(f"  └── {titles[child]}")
        for grandchild in builder.children.get(child, [])[:3]:
            print(f"      └── {titles[grandchild]}")
        if len(builder.children.get(child, [])) > 3:
            print(f"      └── ... ({len(builder.children[child]) - 3} more)")
    if len(builder.children.get(root_idx, [])) > 5:
        print(f"  └── ... ({len(builder.children[root_idx]) - 5} more)")


if __name__ == "__main__":
    main()
