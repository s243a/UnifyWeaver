#!/usr/bin/env python3
"""
Density Confidence Experiments for Paper Evaluation.

Runs experiments from PUBLICATION_PROPOSAL.md:
1. Confidence Calibration: Do confidence scores correlate with correctness?
2. Ranking Improvement: Does density confidence improve retrieval metrics?
3. Robustness to Sparse Regions: Does the method correctly penalize sparse results?

Usage:
    python scripts/density_confidence_experiments.py
    python scripts/density_confidence_experiments.py --experiment all
    python scripts/density_confidence_experiments.py --experiment calibration
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Add source paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

from density_scoring import (
    DensityConfig, BandwidthMethod, flux_softmax,
    compute_density_scores, two_stage_density_pipeline,
    silverman_bandwidth, pairwise_cosine_distances
)


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    name: str
    metrics: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


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


def prepare_data(clusters: Dict[str, List[Dict]], dim: int = 64):
    """Prepare embeddings and ground truth."""
    all_questions = []  # (embedding, cluster_id, text)
    all_answers = []    # (embedding, cluster_id, text)

    for cluster_id, pairs in clusters.items():
        for pair in pairs:
            q_text = get_text(pair.get("question", ""))
            a_text = get_text(pair.get("answer", ""))

            if q_text and a_text:
                q_emb = simple_embedding(q_text, dim)
                a_emb = simple_embedding(a_text, dim)
                all_questions.append((q_emb, cluster_id, q_text))
                all_answers.append((a_emb, cluster_id, a_text))

    return all_questions, all_answers


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


# =============================================================================
# Experiment 1: Confidence Calibration
# =============================================================================

def run_calibration_experiment(
    questions: List[Tuple],
    answers: List[Tuple],
    density_weight: float = 0.3,
    num_bins: int = 10
) -> ExperimentResult:
    """
    Test if confidence scores correlate with actual correctness.

    Method:
    1. For each query, compute confidence for all candidates
    2. Bin results by confidence
    3. Measure precision within each bin
    """
    print("Running Experiment 1: Confidence Calibration...")

    # Create answer embedding matrix
    answer_embs = np.array([a[0] for a in answers])
    answer_clusters = [a[1] for a in answers]

    # Collect (confidence, is_correct) pairs
    confidence_correct_pairs = []

    for q_emb, true_cluster, _ in questions:
        # Compute similarities to all answers
        sims = np.array([cosine_similarity(q_emb, a) for a in answer_embs])

        # Compute densities
        densities = compute_density_scores(answer_embs, DensityConfig())

        # Compute flux-softmax confidences
        probs = flux_softmax(sims, densities, density_weight)

        # Record (confidence, is_correct) for each candidate
        for i, (prob, cluster) in enumerate(zip(probs, answer_clusters)):
            is_correct = 1.0 if cluster == true_cluster else 0.0
            confidence_correct_pairs.append((prob, is_correct))

    # Bin by confidence
    confidences = np.array([p[0] for p in confidence_correct_pairs])
    corrects = np.array([p[1] for p in confidence_correct_pairs])

    # Create bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])

    # Compute precision per bin
    bin_precisions = []
    bin_counts = []
    bin_avg_conf = []

    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            precision = corrects[mask].mean()
            avg_conf = confidences[mask].mean()
            count = mask.sum()
        else:
            precision = 0.0
            avg_conf = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            count = 0

        bin_precisions.append(precision)
        bin_counts.append(count)
        bin_avg_conf.append(avg_conf)

    # Expected Calibration Error (ECE)
    total_samples = len(confidence_correct_pairs)
    ece = sum(
        (count / total_samples) * abs(prec - conf)
        for count, prec, conf in zip(bin_counts, bin_precisions, bin_avg_conf)
        if count > 0
    )

    # Compute correlation
    corr = np.corrcoef(confidences, corrects)[0, 1] if len(confidences) > 1 else 0.0

    return ExperimentResult(
        name="Calibration",
        metrics={
            'ece': ece,
            'correlation': corr,
            'num_samples': total_samples,
        },
        details={
            'bin_precisions': bin_precisions,
            'bin_counts': bin_counts,
            'bin_avg_conf': bin_avg_conf,
        }
    )


# =============================================================================
# Experiment 2: Ranking Improvement
# =============================================================================

def run_ranking_experiment(
    questions: List[Tuple],
    answers: List[Tuple],
    density_weights: List[float] = [0.0, 0.25, 0.5, 1.0, 2.0],
    temperatures: List[float] = [0.5, 1.0, 2.0]
) -> ExperimentResult:
    """
    Test if density confidence improves retrieval metrics.

    Compares:
    - Baseline: Rank by similarity only (w=0)
    - Proposed: Rank by similarity + density confidence (w>0)
    """
    print("Running Experiment 2: Ranking Improvement...")

    answer_embs = np.array([a[0] for a in answers])
    answer_clusters = [a[1] for a in answers]

    # Precompute densities once
    densities = compute_density_scores(answer_embs, DensityConfig())

    results = {}

    for w in density_weights:
        for temp in temperatures:
            key = f"w={w}_t={temp}"
            p1_scores = []
            p5_scores = []
            mrr_scores = []

            for q_emb, true_cluster, _ in questions:
                # Compute similarities
                sims = np.array([cosine_similarity(q_emb, a) for a in answer_embs])

                # Rank by flux-softmax
                probs = flux_softmax(sims, densities, w, temp)

                # Sort by probability
                ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
                ranked_clusters = [answer_clusters[i] for i, _ in ranked]

                # P@1
                p1 = 1.0 if ranked_clusters[0] == true_cluster else 0.0
                p1_scores.append(p1)

                # P@5
                top5 = ranked_clusters[:5]
                p5 = sum(1 for c in top5 if c == true_cluster) / 5
                p5_scores.append(p5)

                # MRR
                try:
                    rank = ranked_clusters.index(true_cluster) + 1
                    mrr = 1.0 / rank
                except ValueError:
                    mrr = 0.0
                mrr_scores.append(mrr)

            results[key] = {
                'p_at_1': np.mean(p1_scores),
                'p_at_5': np.mean(p5_scores),
                'mrr': np.mean(mrr_scores),
                'density_weight': w,
                'temperature': temp,
            }

    # Find best configuration
    best_key = max(results.keys(), key=lambda k: results[k]['mrr'])
    baseline_key = "w=0.0_t=1.0"

    improvement = results[best_key]['mrr'] - results[baseline_key]['mrr']

    return ExperimentResult(
        name="Ranking",
        metrics={
            'baseline_mrr': results[baseline_key]['mrr'],
            'best_mrr': results[best_key]['mrr'],
            'improvement': improvement,
            'best_config': best_key,
        },
        details={
            'all_results': results,
        }
    )


# =============================================================================
# Experiment 3: Robustness to Sparse Regions
# =============================================================================

def run_sparsity_experiment(
    questions: List[Tuple],
    answers: List[Tuple],
    density_weight: float = 0.5
) -> ExperimentResult:
    """
    Test if method correctly penalizes results in sparse regions.

    Method:
    1. Compute query density (how dense the region around the query is)
    2. Split queries into sparse vs dense
    3. Compare performance on each subset
    """
    print("Running Experiment 3: Sparse Region Robustness...")

    answer_embs = np.array([a[0] for a in answers])
    answer_clusters = [a[1] for a in answers]

    # Compute global densities
    densities = compute_density_scores(answer_embs, DensityConfig())

    # For each query, compute local density
    query_densities = []
    query_results = []  # (is_dense, baseline_correct, proposed_correct)

    for q_emb, true_cluster, _ in questions:
        # Compute similarity to all answers
        sims = np.array([cosine_similarity(q_emb, a) for a in answer_embs])

        # Query density: average density of top-k similar answers
        top_k = 5
        top_indices = np.argsort(sims)[-top_k:]
        query_density = np.mean(densities[top_indices])
        query_densities.append(query_density)

        # Baseline ranking (by similarity)
        baseline_ranked = np.argsort(sims)[::-1]
        baseline_top = answer_clusters[baseline_ranked[0]]
        baseline_correct = baseline_top == true_cluster

        # Proposed ranking (with density)
        probs = flux_softmax(sims, densities, density_weight)
        proposed_ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        proposed_top = answer_clusters[proposed_ranked[0][0]]
        proposed_correct = proposed_top == true_cluster

        query_results.append((query_density, baseline_correct, proposed_correct))

    # Split by median density
    median_density = np.median(query_densities)

    sparse_results = [r for r in query_results if r[0] < median_density]
    dense_results = [r for r in query_results if r[0] >= median_density]

    # Compute accuracy for each split
    sparse_baseline_acc = np.mean([r[1] for r in sparse_results]) if sparse_results else 0
    sparse_proposed_acc = np.mean([r[2] for r in sparse_results]) if sparse_results else 0

    dense_baseline_acc = np.mean([r[1] for r in dense_results]) if dense_results else 0
    dense_proposed_acc = np.mean([r[2] for r in dense_results]) if dense_results else 0

    return ExperimentResult(
        name="Sparsity",
        metrics={
            'sparse_baseline_acc': sparse_baseline_acc,
            'sparse_proposed_acc': sparse_proposed_acc,
            'sparse_improvement': sparse_proposed_acc - sparse_baseline_acc,
            'dense_baseline_acc': dense_baseline_acc,
            'dense_proposed_acc': dense_proposed_acc,
            'dense_improvement': dense_proposed_acc - dense_baseline_acc,
            'num_sparse': len(sparse_results),
            'num_dense': len(dense_results),
        },
        details={
            'median_density': median_density,
            'query_densities': query_densities,
        }
    )


def main():
    parser = argparse.ArgumentParser(description='Run density confidence experiments')
    parser.add_argument('--experiment', choices=['all', 'calibration', 'ranking', 'sparsity'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--max-clusters', type=int, default=None)
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()

    # Find training data
    project_root = Path(__file__).parent.parent
    data_dirs = [
        project_root / "training-data" / "tailored",
        project_root / "training-data" / "tailored-gemini",
    ]

    data_dir = None
    for d in data_dirs:
        if d.exists():
            data_dir = d
            break

    if not data_dir:
        print("No training data found. Creating synthetic data...")
        clusters = {}
        for i in range(30):
            clusters[f"cluster_{i}"] = [
                {"question": f"Question {j} about topic {i}", "answer": f"Answer about topic {i}"}
                for j in range(np.random.randint(2, 8))
            ]
    else:
        print(f"Loading data from {data_dir}")
        clusters = load_training_data(data_dir, args.max_clusters)

    print(f"Loaded {len(clusters)} clusters")

    # Prepare data
    questions, answers = prepare_data(clusters, args.dim)
    print(f"Prepared {len(questions)} questions, {len(answers)} answers")
    print()

    results = []

    # Run experiments
    if args.experiment in ['all', 'calibration']:
        result = run_calibration_experiment(questions, answers)
        results.append(result)
        print(f"\n{result.name} Results:")
        print(f"  ECE: {result.metrics['ece']:.4f}")
        print(f"  Correlation: {result.metrics['correlation']:.4f}")

    if args.experiment in ['all', 'ranking']:
        result = run_ranking_experiment(questions, answers)
        results.append(result)
        print(f"\n{result.name} Results:")
        print(f"  Baseline MRR: {result.metrics['baseline_mrr']:.4f}")
        print(f"  Best MRR: {result.metrics['best_mrr']:.4f} ({result.metrics['best_config']})")
        print(f"  Improvement: {result.metrics['improvement']:+.4f}")

        print("\n  Ablation Results:")
        for key, vals in sorted(result.details['all_results'].items()):
            print(f"    {key}: P@1={vals['p_at_1']:.3f}, MRR={vals['mrr']:.3f}")

    if args.experiment in ['all', 'sparsity']:
        result = run_sparsity_experiment(questions, answers)
        results.append(result)
        print(f"\n{result.name} Results:")
        print(f"  Sparse queries ({result.metrics['num_sparse']}):")
        print(f"    Baseline: {result.metrics['sparse_baseline_acc']:.1%}")
        print(f"    Proposed: {result.metrics['sparse_proposed_acc']:.1%}")
        print(f"    Improvement: {result.metrics['sparse_improvement']:+.1%}")
        print(f"  Dense queries ({result.metrics['num_dense']}):")
        print(f"    Baseline: {result.metrics['dense_baseline_acc']:.1%}")
        print(f"    Proposed: {result.metrics['dense_proposed_acc']:.1%}")
        print(f"    Improvement: {result.metrics['dense_improvement']:+.1%}")

    # Save results
    output_file = project_root / "reports" / "density_confidence_experiments.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump([{
            'name': r.name,
            'metrics': r.metrics,
        } for r in results], f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
