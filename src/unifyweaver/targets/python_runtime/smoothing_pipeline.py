"""
End-to-End Smoothing Pipeline

Integrates all components for semantic search with LDA projection:
1. FFT Smoothing for W matrix regularization (training)
2. DensityIndex for pre-computed density (training)
3. Policy-based technique selection (training)
4. Flux-softmax inference with pre-computed densities (inference)

Usage:
    from smoothing_pipeline import SmoothingPipeline

    # Training
    pipeline = SmoothingPipeline()
    pipeline.train(clusters)
    pipeline.save("pipeline.pkl")

    # Inference
    pipeline = SmoothingPipeline.load("pipeline.pkl")
    results = pipeline.search(query_embedding, k=10)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import time
import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .fft_smoothing import FFTSmoothingProjection
    from .smoothing_basis import SmoothingBasisProjection
    from .density_scoring import (
        DensityIndex, DensityIndexConfig, BandwidthMethod,
        flux_softmax, cosine_similarity
    )
except ImportError:
    from fft_smoothing import FFTSmoothingProjection
    from smoothing_basis import SmoothingBasisProjection
    from density_scoring import (
        DensityIndex, DensityIndexConfig, BandwidthMethod,
        flux_softmax, cosine_similarity
    )

# Import generated policy
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "codegen" / "generated"))

try:
    from smoothing_policy import (
        NodeInfo, SmoothingTechnique, recommended_technique,
        generate_smoothing_plan, clusters_distinguishable
    )
    POLICY_AVAILABLE = True
except ImportError:
    POLICY_AVAILABLE = False


@dataclass
class SearchResult:
    """A single search result."""
    answer_id: str
    cluster_id: str
    similarity: float
    density: float
    probability: float
    answer_embedding: Optional[np.ndarray] = None


@dataclass
class PipelineStats:
    """Statistics from pipeline training."""
    num_clusters: int
    num_answers: int
    train_time_ms: float
    smoothing_technique: str
    density_bandwidth_avg: float


class SmoothingPipeline:
    """
    End-to-end semantic search pipeline with LDA smoothing.

    Training pipeline:
        Input: Q-A pairs grouped by cluster
        1. Policy selects smoothing technique based on cluster structure
        2. FFT/basis smoothing regularizes W matrices
        3. DensityIndex pre-computes intra-cluster densities

    Inference pipeline:
        Input: Query embedding
        1. Soft routing to clusters (centroid similarity)
        2. Project query using smoothed W
        3. Find nearest answers
        4. Apply flux-softmax with pre-computed densities
        Output: Ranked results with probabilities
    """

    def __init__(
        self,
        fft_cutoff: float = 0.5,
        fft_blend: float = 0.7,
        density_weight: float = 0.3,
        temperature: float = 1.0,
        bandwidth_method: BandwidthMethod = BandwidthMethod.SILVERMAN,
        use_policy: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            fft_cutoff: FFT smoothing cutoff frequency
            fft_blend: Blend factor between original and smoothed W
            density_weight: Weight for density in flux-softmax
            temperature: Softmax temperature
            bandwidth_method: KDE bandwidth selection method
            use_policy: Whether to use policy-based technique selection
        """
        self.fft_cutoff = fft_cutoff
        self.fft_blend = fft_blend
        self.density_weight = density_weight
        self.temperature = temperature
        self.bandwidth_method = bandwidth_method
        self.use_policy = use_policy and POLICY_AVAILABLE

        # Components (populated during training)
        self.projector: Optional[FFTSmoothingProjection] = None
        self.density_index: Optional[DensityIndex] = None
        self.centroids: Dict[str, np.ndarray] = {}
        self.answers: Dict[str, Tuple[np.ndarray, str]] = {}  # id -> (embedding, cluster_id)
        self.stats: Optional[PipelineStats] = None

        self._trained = False

    def train(
        self,
        clusters: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        answer_ids: Optional[Dict[str, List[str]]] = None
    ) -> PipelineStats:
        """
        Train the pipeline.

        Args:
            clusters: Dict mapping cluster_id to (Q, A, centroid) tuples
                - Q: (n_questions, dim) question embeddings
                - A: (n_answers, dim) or (dim,) answer embeddings
                - centroid: Optional (dim,) cluster centroid
            answer_ids: Optional mapping cluster_id -> list of answer IDs

        Returns:
            Training statistics
        """
        start_time = time.perf_counter()

        # Convert to standard format
        clusters_for_smoothing = []
        clusters_for_density = {}

        for cluster_id, data in clusters.items():
            Q, A, centroid = data

            # Handle single vs multiple answers
            if A.ndim == 1:
                A = A.reshape(1, -1)

            # Store centroid
            if centroid is not None:
                self.centroids[cluster_id] = centroid
            else:
                self.centroids[cluster_id] = np.mean(Q, axis=0)

            # Store answers
            if answer_ids and cluster_id in answer_ids:
                ids = answer_ids[cluster_id]
            else:
                ids = [f"{cluster_id}_{i}" for i in range(len(A))]

            for i, (a_id, a_emb) in enumerate(zip(ids, A)):
                self.answers[a_id] = (a_emb, cluster_id)

            # Format for smoothing (Q, A) tuples
            clusters_for_smoothing.append((Q, A))

            # Format for density
            clusters_for_density[cluster_id] = (Q, A, self.centroids[cluster_id])

        # Select technique using policy
        technique = "fft"
        if self.use_policy:
            root = NodeInfo(
                node_id="root",
                cluster_count=len(clusters),
                total_pairs=sum(len(Q) for Q, _, _ in clusters.values()),
                depth=0,
                avg_pairs=np.mean([len(Q) for Q, _, _ in clusters.values()]),
                similarity_score=0.5
            )
            technique = recommended_technique(root).value

        # Train projector
        if technique == "fft":
            self.projector = FFTSmoothingProjection(
                cutoff=self.fft_cutoff,
                blend_factor=self.fft_blend
            )
            self.projector.train(clusters_for_smoothing)
        elif technique.startswith("basis"):
            k = int(technique.split("_k")[1]) if "_k" in technique else 4
            self.projector = SmoothingBasisProjection(num_basis=k)
            self.projector.train(clusters_for_smoothing, num_iterations=50)
        else:
            # Baseline - no projection, just store
            self.projector = None

        # Build density index
        self.density_index = DensityIndex(DensityIndexConfig(
            bandwidth_method=self.bandwidth_method,
            normalize_densities=True,
            store_embeddings=True
        ))
        density_stats = self.density_index.build(clusters_for_density, answer_ids)

        train_time = (time.perf_counter() - start_time) * 1000

        self.stats = PipelineStats(
            num_clusters=len(clusters),
            num_answers=len(self.answers),
            train_time_ms=train_time,
            smoothing_technique=technique,
            density_bandwidth_avg=density_stats.get('avg_bandwidth', 0.0)
        )

        self._trained = True
        return self.stats

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        return_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search for answers given a query.

        Args:
            query: Query embedding
            k: Number of results to return
            return_embeddings: Whether to include answer embeddings in results

        Returns:
            List of SearchResult ordered by probability
        """
        if not self._trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        # Project query
        if self.projector is not None:
            projected = self.projector.project(query, self.temperature)
        else:
            projected = query

        # Compute similarities to all answers
        candidates = []
        for answer_id, (answer_emb, cluster_id) in self.answers.items():
            sim = cosine_similarity(projected, answer_emb)
            density = self.density_index.lookup_density(answer_id)
            candidates.append((answer_id, cluster_id, answer_emb, sim, density))

        # Sort by similarity first
        candidates.sort(key=lambda x: x[3], reverse=True)

        # Take top candidates for flux-softmax
        top_candidates = candidates[:min(len(candidates), k * 3)]

        if not top_candidates:
            return []

        # Apply flux-softmax
        sims = np.array([c[3] for c in top_candidates])
        densities = np.array([c[4] for c in top_candidates])
        probs = flux_softmax(sims, densities, self.density_weight, self.temperature)

        # Build results
        results = []
        for (answer_id, cluster_id, answer_emb, sim, density), prob in zip(top_candidates, probs):
            results.append(SearchResult(
                answer_id=answer_id,
                cluster_id=cluster_id,
                similarity=float(sim),
                density=float(density),
                probability=float(prob),
                answer_embedding=answer_emb if return_embeddings else None
            ))

        # Sort by probability and take top k
        results.sort(key=lambda x: x.probability, reverse=True)
        return results[:k]

    def add_answer(
        self,
        answer_id: str,
        answer_embedding: np.ndarray,
        cluster_id: Optional[str] = None
    ) -> float:
        """
        Incrementally add an answer.

        Args:
            answer_id: ID for the answer
            answer_embedding: Embedding vector
            cluster_id: Optional cluster to add to

        Returns:
            Computed density for the new answer
        """
        if not self._trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        # Route to cluster if not specified
        if cluster_id is None:
            cluster_id = self._find_nearest_cluster(answer_embedding)

        # Add to density index
        density = self.density_index.add_answer(answer_id, answer_embedding, cluster_id)

        # Store answer
        self.answers[answer_id] = (answer_embedding, cluster_id)

        return density

    def _find_nearest_cluster(self, embedding: np.ndarray) -> str:
        """Find nearest cluster by centroid similarity."""
        best_cluster = None
        best_sim = -1.0

        for cluster_id, centroid in self.centroids.items():
            sim = cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_id

        return best_cluster or "default"

    def save(self, path: str) -> None:
        """Save pipeline to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'config': {
                    'fft_cutoff': self.fft_cutoff,
                    'fft_blend': self.fft_blend,
                    'density_weight': self.density_weight,
                    'temperature': self.temperature,
                    'bandwidth_method': self.bandwidth_method,
                    'use_policy': self.use_policy,
                },
                'projector': self.projector,
                'density_index': self.density_index,
                'centroids': self.centroids,
                'answers': self.answers,
                'stats': self.stats,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'SmoothingPipeline':
        """Load pipeline from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        config = data['config']
        pipeline = cls(
            fft_cutoff=config['fft_cutoff'],
            fft_blend=config['fft_blend'],
            density_weight=config['density_weight'],
            temperature=config['temperature'],
            bandwidth_method=config['bandwidth_method'],
            use_policy=config['use_policy'],
        )

        pipeline.projector = data['projector']
        pipeline.density_index = data['density_index']
        pipeline.centroids = data['centroids']
        pipeline.answers = data['answers']
        pipeline.stats = data['stats']
        pipeline._trained = True

        return pipeline

    def __len__(self) -> int:
        """Number of indexed answers."""
        return len(self.answers)


def create_pipeline_from_training_data(
    clusters: Dict[str, List[Dict]],
    dim: int = 64,
    embedding_fn=None
) -> SmoothingPipeline:
    """
    Convenience function to create pipeline from training data.

    Args:
        clusters: Dict mapping cluster_id to list of Q-A pair dicts
        dim: Embedding dimension
        embedding_fn: Optional function to compute embeddings

    Returns:
        Trained SmoothingPipeline
    """
    if embedding_fn is None:
        # Simple hash-based embedding
        def embedding_fn(text):
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(dim)
            return emb / (np.linalg.norm(emb) + 1e-8)

    # Convert to embeddings
    cluster_data = {}

    for cluster_id, pairs in clusters.items():
        questions = []
        answers = []

        for pair in pairs:
            q_text = pair.get("question", "")
            a_text = pair.get("answer", "")

            if isinstance(q_text, dict):
                q_text = q_text.get("text", "")
            if isinstance(a_text, dict):
                a_text = a_text.get("text", "")

            if q_text and a_text:
                questions.append(embedding_fn(q_text))
                answers.append(embedding_fn(a_text))

        if questions:
            Q = np.array(questions)
            A = np.array(answers)
            centroid = np.mean(Q, axis=0)
            cluster_data[cluster_id] = (Q, A, centroid)

    # Train pipeline
    pipeline = SmoothingPipeline()
    pipeline.train(cluster_data)

    return pipeline
