#!/usr/bin/env python3
"""
Benchmark hybrid planner vs single-level FFT smoothing.

Compares:
1. Single-level FFT (baseline winner from previous benchmarks)
2. Hybrid planner with distinguishability-based refinement
3. Hybrid planner with full refinement (no distinguishability check)

Measures:
- Training time
- Inference time
- Precision@1, P@3
- Cosine similarity

Usage:
    python scripts/benchmark_hybrid_planner.py
    python scripts/benchmark_hybrid_planner.py --max-clusters 100
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Add source paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "codegen" / "generated"))

from fft_smoothing import FFTSmoothingProjection
from smoothing_basis import SmoothingBasisProjection
from smoothing_policy import (
    NodeInfo, SmoothingTechnique, recommended_technique,
    clusters_distinguishable, refinement_needed, generate_smoothing_plan,
    estimate_cost_ms, plan_summary
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method: str
    train_time_ms: float
    inference_time_us: float
    precision_at_1: float
    precision_at_3: float
    cosine_sim: float
    num_clusters: int
    num_pairs: int
    plan_summary: Optional[Dict] = None


def load_training_data(data_dir: Path, max_clusters: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Load training data grouped by cluster_id."""
    clusters = defaultdict(list)

    for jsonl_file in sorted(data_dir.rglob("*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        pair = json.loads(line)
                        cluster_id = pair.get("cluster_id", "unknown")
                        clusters[cluster_id].append(pair)
                    except json.JSONDecodeError:
                        pass

    if max_clusters and len(clusters) > max_clusters:
        cluster_ids = list(clusters.keys())[:max_clusters]
        clusters = {k: clusters[k] for k in cluster_ids}

    return dict(clusters)


def get_text(field: Any) -> str:
    """Extract text from field."""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict):
        return field.get("text", str(field))
    return str(field) if field else ""


def simple_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Simple deterministic embedding for testing."""
    np.random.seed(hash(text) % (2**32))
    emb = np.random.randn(dim)
    return emb / (np.linalg.norm(emb) + 1e-8)


def prepare_cluster_data(clusters: Dict[str, List[Dict]], dim: int = 64):
    """Convert cluster data to embeddings."""
    cluster_data = []  # List of (Q, A, centroid, cluster_id)
    text_data = []     # List of (q_text, a_text, cluster_id)

    for cluster_id, pairs in clusters.items():
        questions = []
        answers = []

        for pair in pairs:
            q_text = get_text(pair.get("question", ""))
            a_text = get_text(pair.get("answer", ""))

            if q_text and a_text:
                q_emb = simple_embedding(q_text, dim)
                a_emb = simple_embedding(a_text, dim)
                questions.append(q_emb)
                answers.append(a_emb)
                text_data.append((q_text, a_text, cluster_id))

        if questions:
            Q = np.array(questions)
            A = np.mean(answers, axis=0)
            centroid = np.mean(questions, axis=0)
            cluster_data.append((Q, A, centroid, cluster_id))

    return cluster_data, text_data


def compute_similarity_scores(cluster_data: List[Tuple]) -> Dict[str, float]:
    """Compute intra-cluster similarity scores for distinguishability check."""
    scores = {}

    for Q, A, centroid, cluster_id in cluster_data:
        if len(Q) < 2:
            scores[cluster_id] = 0.0  # Single item = fully distinguishable
        else:
            # Average pairwise similarity of questions
            sims = []
            for i in range(len(Q)):
                for j in range(i + 1, len(Q)):
                    q1 = Q[i] / (np.linalg.norm(Q[i]) + 1e-8)
                    q2 = Q[j] / (np.linalg.norm(Q[j]) + 1e-8)
                    sims.append(np.dot(q1, q2))
            scores[cluster_id] = np.mean(sims) if sims else 0.0

    return scores


class HybridSmoothingProjector:
    """Hybrid projector using policy-based technique selection."""

    def __init__(self, use_distinguishability: bool = True):
        self.use_distinguishability = use_distinguishability
        self.projectors = {}  # segment_id -> projector
        self.cluster_to_segment = {}  # cluster_id -> segment_id
        self.centroids = {}  # cluster_id -> centroid
        self.plan = []

    def train(self, cluster_data: List[Tuple], similarity_scores: Dict[str, float]):
        """Train using policy-based technique selection."""
        # Build node info for policy
        root = NodeInfo(
            node_id="root",
            cluster_count=len(cluster_data),
            total_pairs=sum(len(Q) for Q, _, _, _ in cluster_data),
            depth=0,
            avg_pairs=np.mean([len(Q) for Q, _, _, _ in cluster_data]),
            similarity_score=np.mean(list(similarity_scores.values()))
        )

        # Generate plan
        children = {}  # For now, flat structure

        # If we should subdivide, create segments based on similarity
        if len(cluster_data) > 30:
            # Create segments by grouping similar clusters
            segments = self._create_segments(cluster_data, similarity_scores)
            children["root"] = []

            for seg_id, seg_clusters in segments.items():
                seg_sim = np.mean([similarity_scores.get(c[3], 0.5) for c in seg_clusters])
                seg_node = NodeInfo(
                    node_id=seg_id,
                    cluster_count=len(seg_clusters),
                    total_pairs=sum(len(c[0]) for c in seg_clusters),
                    depth=1,
                    avg_pairs=np.mean([len(c[0]) for c in seg_clusters]),
                    similarity_score=seg_sim
                )
                children["root"].append(seg_node)

                # Map clusters to segments
                for _, _, _, cluster_id in seg_clusters:
                    self.cluster_to_segment[cluster_id] = seg_id

        self.plan = generate_smoothing_plan(root, children)

        # Execute plan
        for action in self.plan:
            node_id = action.node_id
            technique = action.technique

            if node_id == "root":
                # Train on all clusters
                clusters_for_training = [(Q, A) for Q, A, _, _ in cluster_data]
            else:
                # Train on segment clusters
                seg_clusters = [c for c in cluster_data
                               if self.cluster_to_segment.get(c[3]) == node_id]
                clusters_for_training = [(Q, A) for Q, A, _, _ in seg_clusters]

            if not clusters_for_training:
                continue

            # Select and train projector based on technique
            if technique == SmoothingTechnique.FFT:
                proj = FFTSmoothingProjection(cutoff=0.5, blend_factor=0.7)
                proj.train(clusters_for_training)
            elif technique in (SmoothingTechnique.BASIS_K4, SmoothingTechnique.BASIS_K8):
                k = 4 if technique == SmoothingTechnique.BASIS_K4 else 8
                proj = SmoothingBasisProjection(num_basis=k)
                proj.train(clusters_for_training, num_iterations=50)
            else:
                # Baseline - just store centroids
                proj = None

            self.projectors[node_id] = proj

        # Store centroids for routing
        for Q, A, centroid, cluster_id in cluster_data:
            self.centroids[cluster_id] = centroid

    def _create_segments(self, cluster_data: List[Tuple],
                        similarity_scores: Dict[str, float],
                        num_segments: int = 5) -> Dict[str, List[Tuple]]:
        """Create segments by grouping clusters."""
        # Simple approach: divide by cluster index
        segments = defaultdict(list)
        clusters_per_seg = max(1, len(cluster_data) // num_segments)

        for i, cluster in enumerate(cluster_data):
            seg_id = f"segment_{i // clusters_per_seg}"
            segments[seg_id].append(cluster)

        return dict(segments)

    def project(self, query: np.ndarray, temperature: float = 0.1) -> np.ndarray:
        """Project query using best matching projector."""
        # Find nearest cluster
        best_cluster = None
        best_sim = -1

        for cluster_id, centroid in self.centroids.items():
            c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            q_norm = query / (np.linalg.norm(query) + 1e-8)
            sim = np.dot(q_norm, c_norm)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_id

        # Get segment for cluster
        segment = self.cluster_to_segment.get(best_cluster, "root")

        # Use segment projector if available, else root
        projector = self.projectors.get(segment) or self.projectors.get("root")

        if projector:
            return projector.project(query, temperature)
        else:
            return query  # Fallback


def benchmark_method(name: str, projector, cluster_data: List[Tuple],
                    text_data: List[Tuple], train_fn,
                    dim: int = 64, num_test: int = 100) -> BenchmarkResult:
    """Run benchmark for a method."""

    # Training
    start = time.perf_counter()
    train_fn()
    train_time = (time.perf_counter() - start) * 1000

    # Test data
    np.random.seed(42)
    test_indices = np.random.choice(len(text_data), min(num_test, len(text_data)), replace=False)

    test_queries = []
    test_answers = []
    test_clusters = []

    for idx in test_indices:
        q_text, a_text, cluster_id = text_data[idx]
        test_queries.append(simple_embedding(q_text, dim))
        test_answers.append(simple_embedding(a_text, dim))
        test_clusters.append(cluster_id)

    # All answers
    all_answers = []
    all_cluster_ids = []
    for _, A, _, cluster_id in cluster_data:
        all_answers.append(A)
        all_cluster_ids.append(cluster_id)
    all_answers = np.array(all_answers)

    # Inference timing
    start = time.perf_counter()
    projections = [projector.project(q) for q in test_queries]
    inference_time = (time.perf_counter() - start) * 1e6 / len(test_queries)

    # Cosine similarity
    cosine_values = []
    for proj, true_ans in zip(projections, test_answers):
        proj_norm = proj / (np.linalg.norm(proj) + 1e-8)
        ans_norm = true_ans / (np.linalg.norm(true_ans) + 1e-8)
        cosine_values.append(np.dot(proj_norm, ans_norm))

    # Precision@k
    p1_values = []
    p3_values = []

    for proj, true_cluster in zip(projections, test_clusters):
        proj_norm = proj / (np.linalg.norm(proj) + 1e-8)
        sims = []
        for i, ans in enumerate(all_answers):
            ans_norm = ans / (np.linalg.norm(ans) + 1e-8)
            sims.append((np.dot(proj_norm, ans_norm), all_cluster_ids[i]))

        sims.sort(reverse=True)
        top1 = [c for _, c in sims[:1]]
        top3 = [c for _, c in sims[:3]]

        p1_values.append(1.0 if true_cluster in top1 else 0.0)
        p3_values.append(sum(1 for c in top3 if c == true_cluster) / 3)

    plan_sum = None
    if hasattr(projector, 'plan'):
        plan_sum = plan_summary(projector.plan)

    return BenchmarkResult(
        method=name,
        train_time_ms=train_time,
        inference_time_us=inference_time,
        precision_at_1=np.mean(p1_values),
        precision_at_3=np.mean(p3_values),
        cosine_sim=np.mean(cosine_values),
        num_clusters=len(cluster_data),
        num_pairs=len(text_data),
        plan_summary=plan_sum
    )


def main():
    parser = argparse.ArgumentParser(description='Benchmark hybrid planner')
    parser.add_argument('--max-clusters', type=int, default=None)
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()

    # Find training data
    project_root = Path(__file__).parent.parent
    data_dirs = [
        project_root / "training-data" / "tailored",
        project_root / "training-data" / "tailored-gemini",
        project_root / "playbooks" / "lda-training-data",
    ]

    data_dir = None
    for d in data_dirs:
        if d.exists():
            data_dir = d
            break

    if not data_dir:
        print("No training data found. Creating synthetic data...")
        # Create synthetic data
        clusters = {}
        for i in range(50):
            clusters[f"cluster_{i}"] = [
                {"question": f"Question {j} for cluster {i}", "answer": f"Answer for cluster {i}"}
                for j in range(np.random.randint(2, 6))
            ]
    else:
        print(f"Loading data from {data_dir}")
        clusters = load_training_data(data_dir, args.max_clusters)

    print(f"Loaded {len(clusters)} clusters")

    # Prepare data
    cluster_data, text_data = prepare_cluster_data(clusters, args.dim)
    similarity_scores = compute_similarity_scores(cluster_data)

    print(f"Prepared {len(cluster_data)} clusters, {len(text_data)} Q-A pairs")
    print()

    results = []

    # Benchmark 1: Single-level FFT
    print("Benchmarking: Single-level FFT...")
    fft_proj = FFTSmoothingProjection(cutoff=0.5, blend_factor=0.7)
    clusters_tuple = [(Q, A.reshape(1, -1)) for Q, A, _, _ in cluster_data]
    result = benchmark_method(
        "FFT (single-level)",
        fft_proj,
        cluster_data,
        text_data,
        lambda: fft_proj.train(clusters_tuple),
        args.dim
    )
    results.append(result)
    print(f"  P@1: {result.precision_at_1:.1%}, Train: {result.train_time_ms:.1f}ms")

    # Benchmark 2: Hybrid planner with distinguishability
    print("Benchmarking: Hybrid planner (with distinguishability)...")
    hybrid_proj = HybridSmoothingProjector(use_distinguishability=True)
    result = benchmark_method(
        "Hybrid (distinguishability)",
        hybrid_proj,
        cluster_data,
        text_data,
        lambda: hybrid_proj.train(cluster_data, similarity_scores),
        args.dim
    )
    results.append(result)
    print(f"  P@1: {result.precision_at_1:.1%}, Train: {result.train_time_ms:.1f}ms")
    if result.plan_summary:
        print(f"  Plan: {result.plan_summary}")

    # Benchmark 3: Hybrid planner without distinguishability
    print("Benchmarking: Hybrid planner (full refinement)...")
    hybrid_full = HybridSmoothingProjector(use_distinguishability=False)
    result = benchmark_method(
        "Hybrid (full refinement)",
        hybrid_full,
        cluster_data,
        text_data,
        lambda: hybrid_full.train(cluster_data, similarity_scores),
        args.dim
    )
    results.append(result)
    print(f"  P@1: {result.precision_at_1:.1%}, Train: {result.train_time_ms:.1f}ms")

    # Summary table
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method':<35} {'P@1':>8} {'P@3':>8} {'Cosine':>8} {'Train(ms)':>10} {'Infer(us)':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r.method:<35} {r.precision_at_1:>7.1%} {r.precision_at_3:>7.1%} "
              f"{r.cosine_sim:>8.3f} {r.train_time_ms:>10.1f} {r.inference_time_us:>10.1f}")

    print("-" * 80)
    print(f"Clusters: {results[0].num_clusters}, Q-A pairs: {results[0].num_pairs}")
    print()

    # Save results
    output_file = project_root / "reports" / "hybrid_benchmark_results.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump([{
            'method': r.method,
            'precision_at_1': r.precision_at_1,
            'precision_at_3': r.precision_at_3,
            'cosine_sim': r.cosine_sim,
            'train_time_ms': r.train_time_ms,
            'inference_time_us': r.inference_time_us,
            'num_clusters': r.num_clusters,
            'num_pairs': r.num_pairs,
            'plan_summary': r.plan_summary
        } for r in results], f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
