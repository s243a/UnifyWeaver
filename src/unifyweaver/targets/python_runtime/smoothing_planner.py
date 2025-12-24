# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
Smoothing Planner: Bridges Prolog policy with Python execution.

The Prolog module (smoothing_policy.pl) decides WHAT to do.
This Python module executes HOW to do it.

Workflow:
1. Build tree structure from FFT ordering
2. Export tree to JSON for Prolog
3. Query Prolog for smoothing plan
4. Execute plan using Python smoothing implementations
"""

import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import logging

# Import smoothing implementations
from fft_smoothing import FFTSmoothingProjection
from smoothing_basis import SmoothingBasisProjection, MultiHeadLDABaseline
from hierarchical_smoothing import HierarchicalSmoothing

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the smoothing tree."""
    id: str
    cluster_indices: List[int]  # Which clusters belong to this node
    cluster_count: int
    total_pairs: int
    depth: int
    avg_pairs: float
    similarity_score: float = 0.5
    children: List['TreeNode'] = field(default_factory=list)
    parent_id: Optional[str] = None

    # Computed after smoothing
    projection: Optional[Any] = None  # Trained projector


@dataclass
class SmoothingAction:
    """An action in the smoothing plan."""
    technique: str
    node_id: str


def build_tree_from_fft_ordering(
    clusters: List[Tuple[np.ndarray, np.ndarray]],
    order: np.ndarray,
    segment_threshold: float = 0.3
) -> TreeNode:
    """
    Build a tree from FFT's similarity ordering.

    Segments are created where similarity gaps exceed threshold.
    """
    from scipy.spatial.distance import cosine

    # Compute centroids
    centroids = np.array([np.mean(Q, axis=0) for Q, A in clusters])

    # Find segment boundaries (large similarity gaps)
    ordered_centroids = centroids[order]
    gaps = []
    for i in range(len(order) - 1):
        dist = cosine(ordered_centroids[i], ordered_centroids[i + 1])
        gaps.append((i, dist))

    # Segments are separated by large gaps
    segment_boundaries = [0]
    for i, dist in gaps:
        if dist > segment_threshold:
            segment_boundaries.append(i + 1)
    segment_boundaries.append(len(order))

    # Build tree: root -> segments -> (potentially more levels)
    root = TreeNode(
        id="root",
        cluster_indices=list(range(len(clusters))),
        cluster_count=len(clusters),
        total_pairs=sum(len(Q) for Q, A in clusters),
        depth=0,
        avg_pairs=sum(len(Q) for Q, A in clusters) / len(clusters)
    )

    # Compute root similarity (average pairwise similarity)
    if len(clusters) > 1:
        sims = []
        for i in range(min(100, len(clusters))):
            for j in range(i + 1, min(100, len(clusters))):
                sims.append(1 - cosine(centroids[i], centroids[j]))
        root.similarity_score = np.mean(sims) if sims else 0.5

    # Create segment children
    for seg_idx, (start, end) in enumerate(zip(segment_boundaries[:-1], segment_boundaries[1:])):
        segment_order = order[start:end]
        segment_clusters = [clusters[i] for i in segment_order]

        seg_node = TreeNode(
            id=f"segment_{seg_idx}",
            cluster_indices=list(segment_order),
            cluster_count=len(segment_order),
            total_pairs=sum(len(Q) for Q, A in segment_clusters),
            depth=1,
            avg_pairs=sum(len(Q) for Q, A in segment_clusters) / max(1, len(segment_order)),
            parent_id="root"
        )

        # Segment similarity
        if len(segment_order) > 1:
            seg_centroids = centroids[segment_order]
            sims = []
            for i in range(len(seg_centroids)):
                for j in range(i + 1, len(seg_centroids)):
                    sims.append(1 - cosine(seg_centroids[i], seg_centroids[j]))
            seg_node.similarity_score = np.mean(sims) if sims else 0.5

        root.children.append(seg_node)

    return root


def tree_to_json(root: TreeNode) -> Dict:
    """Convert tree to JSON format for Prolog."""
    nodes = []
    edges = []
    similarities = []

    def traverse(node: TreeNode):
        nodes.append({
            "id": node.id,
            "cluster_count": node.cluster_count,
            "total_pairs": node.total_pairs,
            "depth": node.depth,
            "avg_pairs": node.avg_pairs
        })
        similarities.append({
            "id": node.id,
            "score": node.similarity_score
        })

        for child in node.children:
            edges.append({
                "parent": node.id,
                "child": child.id
            })
            traverse(child)

    traverse(root)

    return {
        "nodes": nodes,
        "edges": edges,
        "similarities": similarities
    }


def query_prolog_plan(
    tree_json: Dict,
    prolog_module: Path,
    max_cost_ms: Optional[float] = None
) -> List[SmoothingAction]:
    """
    Query Prolog for a smoothing plan.

    Falls back to Python-based planning if Prolog unavailable.
    """
    try:
        return _query_prolog_plan_impl(tree_json, prolog_module, max_cost_ms)
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        logger.warning(f"Prolog query failed ({e}), using Python fallback")
        return _python_fallback_plan(tree_json, max_cost_ms)


def _query_prolog_plan_impl(
    tree_json: Dict,
    prolog_module: Path,
    max_cost_ms: Optional[float]
) -> List[SmoothingAction]:
    """Actual Prolog query implementation."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write tree JSON
        tree_path = Path(tmpdir) / "tree.json"
        with open(tree_path, 'w') as f:
            json.dump(tree_json, f)

        plan_path = Path(tmpdir) / "plan.json"

        # Build Prolog query
        if max_cost_ms:
            query = f"""
                use_module('{prolog_module}'),
                load_tree_from_json('{tree_path}'),
                optimized_plan(root, {max_cost_ms}, Plan),
                export_plan_to_json(Plan, '{plan_path}')
            """
        else:
            query = f"""
                use_module('{prolog_module}'),
                load_tree_from_json('{tree_path}'),
                smoothing_plan(root, Plan),
                export_plan_to_json(Plan, '{plan_path}')
            """

        # Run SWI-Prolog
        result = subprocess.run(
            ["swipl", "-g", query, "-t", "halt"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise subprocess.SubprocessError(f"Prolog failed: {result.stderr}")

        # Read plan
        with open(plan_path, 'r') as f:
            plan_data = json.load(f)

        return [
            SmoothingAction(technique=a["technique"], node_id=a["node"])
            for a in plan_data["actions"]
        ]


def _python_fallback_plan(
    tree_json: Dict,
    max_cost_ms: Optional[float]
) -> List[SmoothingAction]:
    """
    Python fallback when Prolog is unavailable.

    Implements the same logic as smoothing_policy.pl with depth-based transitions.

    Key insight: FFT's O(N log N) advantage diminishes at lower scales.
    At deeper levels, transitioning to basis methods gives better accuracy
    without the FFT overhead of MST construction and DFS ordering.
    """
    # Thresholds matching Prolog policy
    FFT_THRESHOLD = 30
    BASIS_SWEET_SPOT = (10, 50)

    actions = []

    for node in tree_json["nodes"]:
        cluster_count = node["cluster_count"]
        depth = node["depth"]
        avg_pairs = node["avg_pairs"]

        technique = None

        # Rule 1: Large clusters at shallow depths -> FFT
        if cluster_count >= FFT_THRESHOLD and depth < 3:
            technique = "fft"

        # Rule 2: Medium clusters at deeper levels -> basis_k8
        elif (BASIS_SWEET_SPOT[0] <= cluster_count <= BASIS_SWEET_SPOT[1]
              and depth >= 1 and avg_pairs >= 2):
            technique = "basis_k8"

        # Rule 3: Moderate clusters at deep levels -> basis_k4
        elif 5 <= cluster_count < 20 and depth >= 2 and avg_pairs >= 2:
            technique = "basis_k4"

        # Rule 4: Small clusters -> baseline
        elif cluster_count < 5:
            technique = "baseline"

        # Rule 5: Large clusters even at depth -> FFT still efficient
        elif cluster_count >= 50 and depth >= 3:
            technique = "fft"

        # Rule 6: Fallback
        else:
            technique = "basis_k4" if cluster_count >= 5 else "baseline"

        actions.append(SmoothingAction(technique=technique, node_id=node["id"]))

    return actions


def execute_plan(
    plan: List[SmoothingAction],
    tree: TreeNode,
    clusters: List[Tuple[np.ndarray, np.ndarray]],
    parent_constraint_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Execute a smoothing plan with optional parent soft constraints.

    Args:
        plan: List of smoothing actions from Prolog/fallback
        tree: Tree structure from FFT ordering
        clusters: Original cluster data
        parent_constraint_weight: How much to regularize toward parent (0-1)
            0 = no constraint, 1 = strong pull toward parent

    Returns dict mapping node_id -> trained projector.

    Key insight: Higher-level smoothings (parent) serve as soft constraints
    for lower-level smoothings (child). This regularizes child projections
    toward the global structure while still allowing local refinement.
    """
    projectors = {}

    # Build node lookup and parent mapping
    node_lookup = {}
    parent_of = {}  # child_id -> parent_id

    def build_lookup(node: TreeNode, parent_id: Optional[str] = None):
        node_lookup[node.id] = node
        if parent_id:
            parent_of[node.id] = parent_id
        for child in node.children:
            build_lookup(child, node.id)

    build_lookup(tree)

    # Sort actions by depth (parents first)
    def get_depth(action: SmoothingAction) -> int:
        return node_lookup[action.node_id].depth

    sorted_plan = sorted(plan, key=get_depth)

    # Execute each action (parents before children)
    for action in sorted_plan:
        node = node_lookup[action.node_id]
        node_clusters = [clusters[i] for i in node.cluster_indices]

        # Get parent projector if available (for soft constraints)
        parent_projector = None
        if action.node_id in parent_of:
            parent_id = parent_of[action.node_id]
            parent_projector = projectors.get(parent_id)

        # Parent constraint only applies to non-FFT techniques
        uses_parent_constraint = (parent_projector is not None and
                                  action.technique != "fft")
        logger.info(f"Executing {action.technique} on {action.node_id} "
                   f"({node.cluster_count} clusters)"
                   + (f" with parent constraint" if uses_parent_constraint else ""))

        if action.technique == "fft":
            projector = FFTSmoothingProjection(cutoff=0.5, blend_factor=0.6)
            projector.train(node_clusters)

            # NOTE: We don't apply parent constraint for FFT children.
            # FFT already does global smoothing within its segment via
            # frequency-domain filtering. Blending with parent's FFT
            # would be redundant - both are doing similar smoothing.
            # Parent constraint is more meaningful for basis/baseline
            # methods that learn local patterns.

        elif action.technique.startswith("basis_k"):
            k = int(action.technique.split("_k")[1])
            projector = SmoothingBasisProjection(num_basis=k)
            projector.train(node_clusters, num_iterations=50, log_interval=100)

            # Apply soft constraint by blending coefficients toward parent
            if parent_projector:
                _apply_parent_constraint_basis(projector, parent_projector,
                                              node.cluster_indices, parent_constraint_weight)

        elif action.technique == "hierarchical":
            # Convert to triple format
            triples = [(Q, A.flatten(), f"c{i}")
                      for i, (Q, A) in enumerate(node_clusters)]
            projector = HierarchicalSmoothing(num_levels=2)
            projector.train(triples, num_iterations=30)

        elif action.technique == "baseline":
            projector = MultiHeadLDABaseline()
            projector.train(node_clusters)

        else:
            logger.warning(f"Unknown technique: {action.technique}, using baseline")
            projector = MultiHeadLDABaseline()
            projector.train(node_clusters)

        projectors[action.node_id] = projector
        node.projection = projector

    return projectors


def _apply_parent_constraint_fft(child_proj, parent_proj, cluster_indices, weight):
    """
    Apply soft constraint by blending child's W matrices toward parent's.

    For FFT projectors, this blends the smoothed_W arrays.
    """
    if not hasattr(child_proj, 'smoothed_W') or not hasattr(parent_proj, 'smoothed_W'):
        return

    # Get parent's W for the relevant clusters
    parent_W = parent_proj.smoothed_W[cluster_indices]

    # Blend: child = (1-weight) * child + weight * parent
    child_proj.smoothed_W = (1 - weight) * child_proj.smoothed_W + weight * parent_W
    logger.debug(f"Applied FFT parent constraint with weight={weight}")


def _apply_parent_constraint_basis(child_proj, parent_proj, cluster_indices, weight):
    """
    Apply soft constraint for basis projection by adjusting coefficients.

    For basis projectors, we reconstruct what the parent would produce for
    each cluster and blend the child's reconstructed W toward it.
    """
    # Get parent's W for these clusters
    parent_W = _get_parent_W_for_clusters(parent_proj, cluster_indices)
    if parent_W is None:
        return

    # For basis projection: W_i = sum_k alpha_i,k * G_k
    # We can't directly modify basis matrices (shared), but we can:
    # 1. Compute what W each cluster currently produces
    # 2. Blend toward parent's W
    # 3. Re-fit coefficients to the blended W

    if not hasattr(child_proj, 'basis') or not hasattr(child_proj, 'coefficients'):
        return

    # Get current reconstruction
    current_W = np.einsum('nk,kij->nij', child_proj.coefficients, child_proj.basis)

    # Blend toward parent's W
    # parent_W shape depends on parent projector type
    if len(parent_W.shape) == 3:
        # Parent is also per-cluster (FFT or basis)
        target_W = (1 - weight) * current_W + weight * parent_W
    else:
        # Parent is single W matrix (baseline)
        target_W = (1 - weight) * current_W + weight * parent_W[np.newaxis, :, :]

    # Re-fit coefficients to the blended target
    # For each cluster i: target_W[i] ≈ sum_k alpha_i,k * G_k
    # This is a least-squares problem for alpha given G and target
    n_clusters = child_proj.coefficients.shape[0]
    n_basis = child_proj.basis.shape[0]

    # Flatten basis for lstsq: (K, d*d) -> solve for (N, K) coefficients
    G_flat = child_proj.basis.reshape(n_basis, -1)  # (K, d*d)
    target_flat = target_W.reshape(n_clusters, -1)  # (N, d*d)

    # Solve: target_flat ≈ coefficients @ G_flat
    # i.e., for each cluster i: target_flat[i] = sum_k coef[i,k] * G_flat[k]
    # Transpose to: G_flat.T @ coef.T = target_flat.T
    new_coef, _, _, _ = np.linalg.lstsq(G_flat.T, target_flat.T, rcond=None)
    child_proj.coefficients = new_coef.T

    logger.debug(f"Applied basis parent constraint with weight={weight}")


def _get_parent_W_for_clusters(parent_proj, cluster_indices):
    """
    Extract parent's W matrices for specific clusters.

    This provides the "target" for soft constraint regularization.
    """
    if hasattr(parent_proj, 'smoothed_W'):
        # FFT projector - get smoothed W
        return parent_proj.smoothed_W[cluster_indices]
    elif hasattr(parent_proj, 'W'):
        # Direct W attribute (baseline or single matrix)
        return parent_proj.W
    elif hasattr(parent_proj, 'basis') and hasattr(parent_proj, 'coefficients'):
        # Basis projector - reconstruct W
        return np.einsum('nk,kij->nij', parent_proj.coefficients[cluster_indices],
                        parent_proj.basis)
    else:
        return None


class HybridSmoothingProjection:
    """
    Hybrid projection using Prolog-planned smoothing.

    Combines FFT at top level with refined techniques at segment level.

    Key insight: When applying FFT to a subtree, we blend with parent's
    FFT smoothing using depth-based weights:
    - Parent FFT captures global cross-cluster patterns
    - Child FFT captures local within-segment patterns
    - Blending preserves both levels of structure
    """

    def __init__(self, segment_threshold: float = 0.3,
                 prolog_module: Optional[Path] = None,
                 max_cost_ms: Optional[float] = None,
                 parent_weight_decay: float = 0.5,
                 parent_constraint_weight: float = 0.3):
        """
        Args:
            segment_threshold: Cosine distance threshold for segment boundaries
            prolog_module: Path to Prolog policy module
            max_cost_ms: Budget constraint for plan optimization
            parent_weight_decay: How much to weight parent vs child at inference (0.5 = equal blend)
                                 Higher = more parent influence in projection blending
            parent_constraint_weight: How much to regularize toward parent during training (0-1)
                                     Higher = stronger pull toward parent's W matrices
        """
        self.segment_threshold = segment_threshold
        self.prolog_module = prolog_module or Path(__file__).parent.parent.parent / "core" / "smoothing_policy.pl"
        self.max_cost_ms = max_cost_ms
        self.parent_weight_decay = parent_weight_decay
        self.parent_constraint_weight = parent_constraint_weight

        self.tree: Optional[TreeNode] = None
        self.projectors: Dict[str, Any] = {}
        self.order: Optional[np.ndarray] = None
        self.clusters: List[Tuple[np.ndarray, np.ndarray]] = []

        # Track parent-child relationships for blending
        self.parent_projector: Dict[str, str] = {}  # child_id -> parent_id

    def train(self, clusters: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Train hybrid projection using Prolog-planned approach."""
        from fft_smoothing import order_clusters_by_similarity

        self.clusters = clusters
        logger.info(f"Training hybrid smoothing on {len(clusters)} clusters")

        # Step 1: Get FFT ordering
        centroids = np.array([np.mean(Q, axis=0) for Q, A in clusters])
        self.order = order_clusters_by_similarity(centroids)

        # Step 2: Build tree from ordering
        self.tree = build_tree_from_fft_ordering(
            clusters, self.order, self.segment_threshold
        )
        logger.info(f"Built tree with {len(self.tree.children)} segments")

        # Step 3: Query Prolog for plan
        tree_json = tree_to_json(self.tree)
        plan = query_prolog_plan(tree_json, self.prolog_module, self.max_cost_ms)
        logger.info(f"Prolog plan: {[(a.technique, a.node_id) for a in plan]}")

        # Step 4: Execute plan with soft constraints
        self.projectors = execute_plan(plan, self.tree, clusters,
                                       parent_constraint_weight=self.parent_constraint_weight)

        # Step 5: Build parent-child mapping for blending
        self._build_parent_mapping(self.tree)

        return {"tree": self.tree, "plan": plan, "projectors": self.projectors}

    def _build_parent_mapping(self, node: TreeNode):
        """Build mapping of child -> parent for hierarchical blending."""
        for child in node.children:
            if node.id in self.projectors and child.id in self.projectors:
                self.parent_projector[child.id] = node.id
            self._build_parent_mapping(child)

    def project(self, query_emb: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """
        Project query using hybrid approach with hierarchical blending.

        Routes to appropriate segment, blends segment projection with parent's.
        """
        if not self.projectors:
            return query_emb

        # Find nearest segment
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        best_segment = None
        best_sim = -1

        for child in self.tree.children:
            # Compute similarity to segment centroid
            seg_clusters = [self.clusters[i] for i in child.cluster_indices]
            seg_centroid = np.mean([np.mean(Q, axis=0) for Q, A in seg_clusters], axis=0)
            seg_centroid_norm = seg_centroid / (np.linalg.norm(seg_centroid) + 1e-8)

            sim = np.dot(query_norm, seg_centroid_norm)
            if sim > best_sim:
                best_sim = sim
                best_segment = child

        # Get segment projection
        if best_segment and best_segment.id in self.projectors:
            segment_proj = self.projectors[best_segment.id].project(query_emb, temperature)

            # Blend with parent if available (hierarchical FFT blending)
            if best_segment.id in self.parent_projector:
                parent_id = self.parent_projector[best_segment.id]
                parent_proj = self.projectors[parent_id].project(query_emb, temperature)

                # Weighted blend: parent captures global, child captures local
                # parent_weight_decay controls the balance
                parent_weight = self.parent_weight_decay
                child_weight = 1.0 - parent_weight

                return parent_weight * parent_proj + child_weight * segment_proj
            else:
                return segment_proj

        elif "root" in self.projectors:
            return self.projectors["root"].project(query_emb, temperature)
        else:
            return query_emb

    def project_at_depth(self, query_emb: np.ndarray, max_depth: int = 1,
                         temperature: float = 0.1) -> np.ndarray:
        """
        Project using only projectors up to a certain depth.

        Useful for comparing global (depth=0) vs local (depth=1+) effects.
        """
        # Find best segment as before
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        best_segment = None
        best_sim = -1

        for child in self.tree.children:
            seg_clusters = [self.clusters[i] for i in child.cluster_indices]
            seg_centroid = np.mean([np.mean(Q, axis=0) for Q, A in seg_clusters], axis=0)
            seg_centroid_norm = seg_centroid / (np.linalg.norm(seg_centroid) + 1e-8)
            sim = np.dot(query_norm, seg_centroid_norm)
            if sim > best_sim:
                best_sim = sim
                best_segment = child

        # Collect projections at each depth up to max_depth
        projections = []
        weights = []

        # Depth 0: root
        if max_depth >= 0 and "root" in self.projectors:
            projections.append(self.projectors["root"].project(query_emb, temperature))
            weights.append(self.parent_weight_decay ** 0)  # 1.0

        # Depth 1: segment
        if max_depth >= 1 and best_segment and best_segment.id in self.projectors:
            projections.append(self.projectors[best_segment.id].project(query_emb, temperature))
            weights.append(self.parent_weight_decay ** 1)

        # TODO: Deeper levels if tree has more depth

        if not projections:
            return query_emb

        # Weighted average
        total_weight = sum(weights)
        result = np.zeros_like(query_emb)
        for proj, w in zip(projections, weights):
            result += (w / total_weight) * proj

        return result


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=== Hybrid Smoothing Planner Test ===\n")

    np.random.seed(42)
    d = 64
    N = 50

    # Create clusters with segment structure
    clusters = []
    for i in range(N):
        segment = i // 10  # 5 segments of 10 clusters each
        n_q = np.random.randint(1, 4)
        base = np.random.randn(d) * 0.2 + segment * 1.0
        Q = base + np.random.randn(n_q, d) * 0.1
        A = base.reshape(1, -1) + np.random.randn(1, d) * 0.05
        clusters.append((Q, A))

    print(f"Created {N} clusters in 5 segments")

    # Train hybrid
    hybrid = HybridSmoothingProjection(segment_threshold=0.4)
    result = hybrid.train(clusters)

    print(f"\nTree: {len(result['tree'].children)} segments")
    print(f"Plan: {len(result['plan'])} actions")
    for action in result['plan']:
        print(f"  {action.technique} -> {action.node_id}")

    # Test projection
    query = np.random.randn(d)
    proj = hybrid.project(query)
    print(f"\nProjection: {np.linalg.norm(query):.3f} -> {np.linalg.norm(proj):.3f}")

    print("\n=== Test Complete ===")
