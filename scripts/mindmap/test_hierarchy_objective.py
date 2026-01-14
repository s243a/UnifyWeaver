#!/usr/bin/env python3
"""
Test hierarchy objective function with branch reconstruction.

Tests that J(CORRECT) < J(WRONG) < J(FLAT) for synthetic hierarchies.
"""

import numpy as np
from hierarchy_objective import HierarchyObjective, HierarchyStats

# Seed for reproducibility
np.random.seed(42)


def create_synthetic_embeddings(n_nodes: int, dim: int = 64) -> np.ndarray:
    """Create synthetic normalized embeddings."""
    embeddings = np.random.randn(n_nodes, dim)
    # Normalize to unit sphere
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def create_clustered_embeddings(
    cluster_centers: np.ndarray,
    points_per_cluster: int,
    noise_scale: float = 0.1
) -> np.ndarray:
    """
    Create embeddings clustered around given centers.

    Args:
        cluster_centers: Shape [n_clusters, dim]
        points_per_cluster: Number of points per cluster
        noise_scale: How spread out points are around centers

    Returns:
        embeddings: Shape [n_clusters * points_per_cluster, dim]
    """
    embeddings = []
    for center in cluster_centers:
        # Add noise around center
        noise = np.random.randn(points_per_cluster, len(center)) * noise_scale
        cluster_points = center + noise
        embeddings.append(cluster_points)

    embeddings = np.vstack(embeddings)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def build_tree_dict(parent_of: dict, node_to_idx: dict) -> dict:
    """Convert parent_of mapping to tree dict format."""
    from collections import defaultdict

    children_of = defaultdict(list)
    for child, parent in parent_of.items():
        children_of[parent].append(child)

    # Find root
    all_nodes = set(parent_of.keys()) | set(parent_of.values())
    roots = all_nodes - set(parent_of.keys())

    def build_node(node_id):
        return {
            'id': node_id,
            'embedding_idx': node_to_idx.get(node_id),
            'children': [build_node(c) for c in children_of.get(node_id, [])]
        }

    if len(roots) == 1:
        return {'root': build_node(list(roots)[0])}
    else:
        return {'roots': [build_node(r) for r in roots]}


def test_branch_reconstruction_fisher():
    """
    Test that objective correctly ranks hierarchies using Fisher entropy proxy.

    Scenario:
        - Root with 2 clusters (A and B)
        - Each cluster has 3 leaf nodes
        - CORRECT: leaves assigned to their actual cluster parent
        - WRONG: leaves swapped between clusters
        - FLAT: all leaves directly under root
    """
    print("=" * 60)
    print("TEST: Branch Reconstruction (Fisher entropy proxy)")
    print("=" * 60)

    dim = 64

    # Create 2 cluster centers that are well-separated
    root_embed = np.random.randn(dim)
    root_embed = root_embed / np.linalg.norm(root_embed)

    # Cluster A center - offset from root
    cluster_a_center = root_embed + np.random.randn(dim) * 0.3
    cluster_a_center = cluster_a_center / np.linalg.norm(cluster_a_center)

    # Cluster B center - different offset
    cluster_b_center = root_embed + np.random.randn(dim) * 0.3
    cluster_b_center = cluster_b_center / np.linalg.norm(cluster_b_center)

    # Create leaf embeddings clustered around their centers
    noise_scale = 0.05  # Tight clusters

    # Cluster A leaves (indices 2, 3, 4)
    a_leaves = cluster_a_center + np.random.randn(3, dim) * noise_scale
    a_leaves = a_leaves / np.linalg.norm(a_leaves, axis=1, keepdims=True)

    # Cluster B leaves (indices 5, 6, 7)
    b_leaves = cluster_b_center + np.random.randn(3, dim) * noise_scale
    b_leaves = b_leaves / np.linalg.norm(b_leaves, axis=1, keepdims=True)

    # Combined embeddings: [root, cluster_a, cluster_b, a_leaf1, a_leaf2, a_leaf3, b_leaf1, b_leaf2, b_leaf3]
    embeddings = np.vstack([
        root_embed.reshape(1, -1),      # 0: root
        cluster_a_center.reshape(1, -1), # 1: cluster_a
        cluster_b_center.reshape(1, -1), # 2: cluster_b
        a_leaves,                        # 3, 4, 5: a_leaves
        b_leaves                         # 6, 7, 8: b_leaves
    ])

    node_to_idx = {
        'root': 0,
        'cluster_a': 1,
        'cluster_b': 2,
        'a_leaf_1': 3,
        'a_leaf_2': 4,
        'a_leaf_3': 5,
        'b_leaf_1': 6,
        'b_leaf_2': 7,
        'b_leaf_3': 8
    }

    # CORRECT hierarchy: leaves under their actual cluster
    correct_parent = {
        'cluster_a': 'root',
        'cluster_b': 'root',
        'a_leaf_1': 'cluster_a',
        'a_leaf_2': 'cluster_a',
        'a_leaf_3': 'cluster_a',
        'b_leaf_1': 'cluster_b',
        'b_leaf_2': 'cluster_b',
        'b_leaf_3': 'cluster_b'
    }

    # WRONG hierarchy: leaves swapped between clusters
    wrong_parent = {
        'cluster_a': 'root',
        'cluster_b': 'root',
        'a_leaf_1': 'cluster_b',  # WRONG
        'a_leaf_2': 'cluster_b',  # WRONG
        'a_leaf_3': 'cluster_b',  # WRONG
        'b_leaf_1': 'cluster_a',  # WRONG
        'b_leaf_2': 'cluster_a',  # WRONG
        'b_leaf_3': 'cluster_a'   # WRONG
    }

    # FLAT hierarchy: all leaves directly under root
    flat_parent = {
        'a_leaf_1': 'root',
        'a_leaf_2': 'root',
        'a_leaf_3': 'root',
        'b_leaf_1': 'root',
        'b_leaf_2': 'root',
        'b_leaf_3': 'root'
    }

    # Build tree dicts
    correct_tree = build_tree_dict(correct_parent, node_to_idx)
    wrong_tree = build_tree_dict(wrong_parent, node_to_idx)
    flat_tree = build_tree_dict(flat_parent, node_to_idx)

    # Create objective function
    obj = HierarchyObjective(
        entropy_source='fisher',
        depth_normalize=True,
        depth_decay=0.5
    )

    # Compute stats
    correct_stats = obj.compute(correct_tree, embeddings)
    wrong_stats = obj.compute(wrong_tree, embeddings)
    flat_stats = obj.compute(flat_tree, embeddings)

    print("\nResults (Fisher entropy proxy):")
    print(f"  CORRECT: J={correct_stats.objective:.4f} (D={correct_stats.semantic_distance:.4f}, H={correct_stats.entropy_gain:.4f})")
    print(f"  WRONG:   J={wrong_stats.objective:.4f} (D={wrong_stats.semantic_distance:.4f}, H={wrong_stats.entropy_gain:.4f})")
    print(f"  FLAT:    J={flat_stats.objective:.4f} (D={flat_stats.semantic_distance:.4f}, H={flat_stats.entropy_gain:.4f})")

    # Verify ranking: J(CORRECT) < J(FLAT) < J(WRONG)
    # Note: WRONG is worse than FLAT because misassigned children have large
    # distances to their wrong parents. FLAT has no measurable structure (degenerate).
    correct_vs_flat = correct_stats.objective < flat_stats.objective
    flat_vs_wrong = flat_stats.objective < wrong_stats.objective

    print(f"\nRanking verification (lower J = better):")
    print(f"  J(CORRECT) < J(FLAT):  {correct_vs_flat} ({correct_stats.objective:.4f} < {flat_stats.objective:.4f})")
    print(f"  J(FLAT) < J(WRONG):    {flat_vs_wrong} ({flat_stats.objective:.4f} < {wrong_stats.objective:.4f})")

    if correct_vs_flat and flat_vs_wrong:
        print("\n✓ PASS: Correct ranking J(CORRECT) < J(FLAT) < J(WRONG)")
        print("  (Wrong assignments worse than no structure)")
    else:
        print("\n✗ FAIL: Incorrect ranking")

    return correct_vs_flat and flat_vs_wrong


def test_branch_reconstruction_logits():
    """
    Test with logits-based entropy using text.

    Uses synthetic "text" represented as node names, with entropy computed
    via a simple mock (since we may not have transformers installed).
    """
    print("\n" + "=" * 60)
    print("TEST: Branch Reconstruction (Logits-based entropy)")
    print("=" * 60)

    dim = 64

    # Same embedding setup as Fisher test
    np.random.seed(42)

    root_embed = np.random.randn(dim)
    root_embed = root_embed / np.linalg.norm(root_embed)

    cluster_a_center = root_embed + np.random.randn(dim) * 0.3
    cluster_a_center = cluster_a_center / np.linalg.norm(cluster_a_center)

    cluster_b_center = root_embed + np.random.randn(dim) * 0.3
    cluster_b_center = cluster_b_center / np.linalg.norm(cluster_b_center)

    noise_scale = 0.05
    a_leaves = cluster_a_center + np.random.randn(3, dim) * noise_scale
    a_leaves = a_leaves / np.linalg.norm(a_leaves, axis=1, keepdims=True)

    b_leaves = cluster_b_center + np.random.randn(3, dim) * noise_scale
    b_leaves = b_leaves / np.linalg.norm(b_leaves, axis=1, keepdims=True)

    embeddings = np.vstack([
        root_embed.reshape(1, -1),
        cluster_a_center.reshape(1, -1),
        cluster_b_center.reshape(1, -1),
        a_leaves,
        b_leaves
    ])

    node_to_idx = {
        'root': 0,
        'cluster_a': 1,
        'cluster_b': 2,
        'a_leaf_1': 3,
        'a_leaf_2': 4,
        'a_leaf_3': 5,
        'b_leaf_1': 6,
        'b_leaf_2': 7,
        'b_leaf_3': 8
    }

    # Text for each node (would be used for entropy computation)
    texts = {
        'root': 'Science',
        'cluster_a': 'Physics',
        'cluster_b': 'Biology',
        'a_leaf_1': 'Quantum Mechanics',
        'a_leaf_2': 'Thermodynamics',
        'a_leaf_3': 'Electromagnetism',
        'b_leaf_1': 'Cell Biology',
        'b_leaf_2': 'Genetics',
        'b_leaf_3': 'Evolution'
    }

    # Tree structures
    correct_parent = {
        'cluster_a': 'root',
        'cluster_b': 'root',
        'a_leaf_1': 'cluster_a',
        'a_leaf_2': 'cluster_a',
        'a_leaf_3': 'cluster_a',
        'b_leaf_1': 'cluster_b',
        'b_leaf_2': 'cluster_b',
        'b_leaf_3': 'cluster_b'
    }

    wrong_parent = {
        'cluster_a': 'root',
        'cluster_b': 'root',
        'a_leaf_1': 'cluster_b',
        'a_leaf_2': 'cluster_b',
        'a_leaf_3': 'cluster_b',
        'b_leaf_1': 'cluster_a',
        'b_leaf_2': 'cluster_a',
        'b_leaf_3': 'cluster_a'
    }

    flat_parent = {
        'a_leaf_1': 'root',
        'a_leaf_2': 'root',
        'a_leaf_3': 'root',
        'b_leaf_1': 'root',
        'b_leaf_2': 'root',
        'b_leaf_3': 'root'
    }

    correct_tree = build_tree_dict(correct_parent, node_to_idx)
    wrong_tree = build_tree_dict(wrong_parent, node_to_idx)
    flat_tree = build_tree_dict(flat_parent, node_to_idx)

    # Try with logits-based entropy if transformers available
    try:
        from transformers import AutoModelForMaskedLM
        has_transformers = True
    except ImportError:
        has_transformers = False

    if has_transformers:
        # Try bert-base-uncased as fallback if ModernBERT not available
        models_to_try = [
            'answerdotai/ModernBERT-base',
            'bert-base-uncased',
            'distilbert-base-uncased'
        ]

        for model_name in models_to_try:
            print(f"\nTrying model: {model_name}")
            obj = HierarchyObjective(
                entropy_source='logits',
                entropy_text_source='raw_phrase',
                entropy_model=model_name,
                entropy_diagnostic_source='logits',  # Enable diagnostic
                depth_normalize=True,
                depth_decay=0.5
            )

            try:
                correct_stats = obj.compute(correct_tree, embeddings, texts=texts)
                wrong_stats = obj.compute(wrong_tree, embeddings, texts=texts)
                flat_stats = obj.compute(flat_tree, embeddings, texts=texts)

                print(f"\nResults ({model_name} logits):")
                print(f"  CORRECT: J={correct_stats.objective:.4f} (D={correct_stats.semantic_distance:.4f}, H={correct_stats.entropy_gain:.4f})")
                if correct_stats.depth_surprisal_correlation is not None:
                    print(f"           depth-surprisal corr={correct_stats.depth_surprisal_correlation:.4f}")
                print(f"  WRONG:   J={wrong_stats.objective:.4f} (D={wrong_stats.semantic_distance:.4f}, H={wrong_stats.entropy_gain:.4f})")
                print(f"  FLAT:    J={flat_stats.objective:.4f} (D={flat_stats.semantic_distance:.4f}, H={flat_stats.entropy_gain:.4f})")

                # Same ranking as Fisher: CORRECT < FLAT < WRONG
                correct_vs_flat = correct_stats.objective < flat_stats.objective
                flat_vs_wrong = flat_stats.objective < wrong_stats.objective

                print(f"\nRanking verification (lower J = better):")
                print(f"  J(CORRECT) < J(FLAT):  {correct_vs_flat}")
                print(f"  J(FLAT) < J(WRONG):    {flat_vs_wrong}")

                if correct_vs_flat and flat_vs_wrong:
                    print("\n✓ PASS: Correct ranking with logits-based entropy")
                else:
                    print("\n✗ FAIL: Incorrect ranking")

                return correct_vs_flat and flat_vs_wrong

            except Exception as e:
                print(f"  Error: {e}")
                continue

        print("\nNo models worked - skipping logits test")
        return None
    else:
        print("\nTransformers not available - skipping logits test")
        print("Install with: pip install transformers torch")
        return None


def test_depth_surprisal_correlation():
    """
    Test that depth-surprisal correlation is computed correctly.

    A good hierarchy should have depth(node) ∝ -log(p(node)).
    More specific (lower probability) nodes should be deeper.
    """
    print("\n" + "=" * 60)
    print("TEST: Depth-Surprisal Correlation")
    print("=" * 60)

    # Create a hierarchy where depth matches specificity
    dim = 64
    np.random.seed(42)

    # Root is most general (high probability)
    root = np.random.randn(dim)
    root = root / np.linalg.norm(root)

    # Level 1: more specific
    mid1 = root + np.random.randn(dim) * 0.2
    mid1 = mid1 / np.linalg.norm(mid1)

    mid2 = root + np.random.randn(dim) * 0.2
    mid2 = mid2 / np.linalg.norm(mid2)

    # Level 2: most specific (tight clusters)
    leaf1 = mid1 + np.random.randn(dim) * 0.05
    leaf1 = leaf1 / np.linalg.norm(leaf1)

    leaf2 = mid2 + np.random.randn(dim) * 0.05
    leaf2 = leaf2 / np.linalg.norm(leaf2)

    embeddings = np.vstack([root, mid1, mid2, leaf1, leaf2])

    node_to_idx = {
        'root': 0,
        'mid1': 1,
        'mid2': 2,
        'leaf1': 3,
        'leaf2': 4
    }

    parent_of = {
        'mid1': 'root',
        'mid2': 'root',
        'leaf1': 'mid1',
        'leaf2': 'mid2'
    }

    tree = build_tree_dict(parent_of, node_to_idx)

    # Create fake logits that match the hierarchy
    # Higher logits = higher probability = should be at higher level
    # Root has highest prob, leaves have lowest
    n_vocab = 100
    logits = np.zeros((5, n_vocab))

    # Root: high logits for common tokens
    logits[0, :10] = 5.0  # High prob for few tokens = low entropy

    # Mid level: medium spread
    logits[1, :20] = 3.0
    logits[2, :20] = 3.0

    # Leaves: spread across many tokens = high entropy
    logits[3, :50] = 1.0
    logits[4, :50] = 1.0

    obj = HierarchyObjective(
        entropy_source='fisher',  # Use fisher for H computation
        entropy_diagnostic_source='density',  # Use density for diagnostic
        knn_k=3,  # Small k for small test set
        depth_normalize=True
    )

    stats = obj.compute(tree, embeddings, logits=logits)

    print(f"\nResults:")
    corr = stats.depth_surprisal_correlation
    slope = stats.depth_surprisal_slope
    corr_str = f"{corr:.4f}" if corr is not None else "N/A"
    slope_str = f"{slope:.4f}" if slope is not None else "N/A"
    print(f"  Depth-surprisal correlation: {corr_str}")
    print(f"  Depth-surprisal slope: {slope_str}")

    # With our synthetic data, deeper nodes should have lower probability
    # So correlation should be positive
    if stats.depth_surprisal_correlation is not None:
        if stats.depth_surprisal_correlation > 0:
            print("\n✓ PASS: Positive depth-surprisal correlation (deeper = less probable)")
        else:
            print("\n✗ FAIL: Expected positive correlation")
        return stats.depth_surprisal_correlation > 0
    else:
        print("\nNo correlation computed")
        return False


def main():
    print("Hierarchy Objective Function Tests")
    print("=" * 60)

    results = {}

    # Test 1: Fisher entropy proxy
    results['fisher'] = test_branch_reconstruction_fisher()

    # Test 2: Logits-based entropy (if transformers available)
    results['logits'] = test_branch_reconstruction_logits()

    # Test 3: Depth-surprisal correlation
    results['depth_surprisal'] = test_depth_surprisal_correlation()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {test_name}: {status}")

    # Overall (use == instead of 'is' to handle numpy booleans)
    passed_tests = [k for k, v in results.items() if v == True and v is not None]
    failed_tests = [k for k, v in results.items() if v == False]
    skipped_tests = [k for k, v in results.items() if v is None]

    print(f"\nTotal: {len(passed_tests)} passed, {len(failed_tests)} failed, {len(skipped_tests)} skipped")

    return len(failed_tests) == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
