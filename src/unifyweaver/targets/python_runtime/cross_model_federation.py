# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Cross-Model Federation for KG Topology Phase 6e.

"""
Cross-Model Federation Engine for KG Topology.

Enables federation across nodes using different embedding models by:
1. Grouping nodes into model pools (same embedding space)
2. Running density-based federation within each pool
3. Fusing results across pools using score/rank-based methods

Key insight: While embeddings are incompatible across models, normalized
probabilities (softmax outputs) are comparable.

See: docs/proposals/CROSS_MODEL_FEDERATION.md
"""

import math
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import List, Dict, Any, Optional, Callable, Tuple

import numpy as np

try:
    from .federated_query import (
        FederatedQueryEngine,
        AggregationConfig,
        AggregationStrategy,
        AggregatedResult,
        AggregatedResponse,
        NodeResult,
        NodeResponse,
        ResultProvenance
    )
    from .kleinberg_router import KleinbergRouter, KGNode
    from .discovery_clients import DiscoveryClient
except ImportError:
    from federated_query import (
        FederatedQueryEngine,
        AggregationConfig,
        AggregationStrategy,
        AggregatedResult,
        AggregatedResponse,
        NodeResult,
        NodeResponse,
        ResultProvenance
    )
    from kleinberg_router import KleinbergRouter, KGNode
    from discovery_clients import DiscoveryClient


# =============================================================================
# FUSION METHODS
# =============================================================================

class FusionMethod(Enum):
    """Methods for combining results across model pools."""
    WEIGHTED_SUM = "weighted_sum"      # Σ w_m * P(answer|model_m)
    RECIPROCAL_RANK = "rrf"            # Σ 1/(k + rank_m)
    CONSENSUS = "consensus"            # Boost multi-pool agreement
    GEOMETRIC_MEAN = "geometric_mean"  # (Π score_m)^(1/n)
    MAX = "max"                        # max(score_m)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelPoolConfig:
    """Configuration for a single model pool."""
    model_name: str
    weight: float = 1.0

    # Intra-pool federation settings
    federation_k: int = 5
    aggregation_strategy: AggregationStrategy = AggregationStrategy.DENSITY_FLUX
    use_adaptive_k: bool = False
    use_hierarchical: bool = False

    # Optional node filter
    node_filter: Optional[Callable[[KGNode], bool]] = None


@dataclass
class CrossModelConfig:
    """Configuration for cross-model federation."""
    pools: List[ModelPoolConfig] = field(default_factory=list)

    # Fusion settings
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_SUM

    # RRF parameters
    rrf_k: int = 60  # Smoothing constant for reciprocal rank

    # Consensus parameters
    consensus_threshold: float = 0.1   # Min probability for consideration
    min_pools_for_consensus: int = 2   # Pools needed for boost
    consensus_boost_factor: float = 1.5  # Multiplier for consensus

    # Query settings
    pool_timeout_ms: int = 30000
    max_workers: int = 10

    # Weight learning (Phase 6e-iv)
    learn_weights: bool = False
    weight_learning_rate: float = 0.01


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PoolResult:
    """Result from a single model pool with density information."""
    answer_id: str
    answer_text: str
    answer_hash: str

    # Scores
    raw_score: float
    exp_score: float
    density_adjusted_score: float

    # Density information (computed within pool)
    density_score: float = 0.0
    cluster_id: int = -1
    cluster_size: int = 0
    cluster_confidence: float = 0.0

    # Provenance
    source_nodes: List[str] = field(default_factory=list)
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer_id': self.answer_id,
            'answer_text': self.answer_text,
            'answer_hash': self.answer_hash,
            'raw_score': self.raw_score,
            'exp_score': self.exp_score,
            'density_adjusted_score': self.density_adjusted_score,
            'density_score': self.density_score,
            'cluster_id': self.cluster_id,
            'cluster_size': self.cluster_size,
            'cluster_confidence': self.cluster_confidence,
            'source_nodes': self.source_nodes,
            'model_name': self.model_name
        }


@dataclass
class PoolResponse:
    """Aggregated response from a model pool."""
    model_name: str
    results: List[PoolResult]

    # Pool-level statistics
    total_results: int = 0
    num_clusters: int = 0
    avg_density: float = 0.0
    partition_sum: float = 0.0

    # Timing
    query_time_ms: float = 0.0
    nodes_queried: int = 0
    nodes_responded: int = 0

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'results': [r.to_dict() for r in self.results],
            'total_results': self.total_results,
            'num_clusters': self.num_clusters,
            'avg_density': self.avg_density,
            'partition_sum': self.partition_sum,
            'query_time_ms': self.query_time_ms,
            'nodes_queried': self.nodes_queried,
            'nodes_responded': self.nodes_responded,
            'error': self.error
        }


@dataclass
class PoolContribution:
    """A pool's contribution to a fused result."""
    model_name: str
    probability: float
    density: float
    cluster_size: int
    rank: int


@dataclass
class FusedResult:
    """Result after cross-model fusion."""
    answer_id: str
    answer_text: str
    answer_hash: str
    fused_score: float

    # Cross-pool information
    num_pools: int = 0
    consensus_strength: float = 0.0  # num_pools / total_pools
    pool_contributions: Dict[str, PoolContribution] = field(default_factory=dict)

    # For RRF
    ranks_by_pool: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer_id': self.answer_id,
            'answer_text': self.answer_text,
            'answer_hash': self.answer_hash,
            'fused_score': self.fused_score,
            'num_pools': self.num_pools,
            'consensus_strength': self.consensus_strength,
            'pool_contributions': {
                k: {
                    'model_name': v.model_name,
                    'probability': v.probability,
                    'density': v.density,
                    'cluster_size': v.cluster_size,
                    'rank': v.rank
                }
                for k, v in self.pool_contributions.items()
            },
            'ranks_by_pool': self.ranks_by_pool
        }


@dataclass
class CrossModelResponse:
    """Final response from cross-model federated query."""
    query_id: str
    results: List[FusedResult]
    pools_queried: List[str]
    pools_responded: int
    total_time_ms: float
    fusion_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__type': 'kg_cross_model_response',
            '__id': self.query_id,
            'results': [r.to_dict() for r in self.results],
            'pools_queried': self.pools_queried,
            'pools_responded': self.pools_responded,
            'total_time_ms': self.total_time_ms,
            'fusion_method': self.fusion_method
        }


# =============================================================================
# FUSION ALGORITHMS
# =============================================================================

def geometric_mean(values: List[float]) -> float:
    """Geometric mean - rewards consistency across pools."""
    if not values:
        return 0.0
    product = 1.0
    for v in values:
        product *= max(v, 1e-10)  # Avoid zero
    return product ** (1.0 / len(values))


def weighted_sum_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Weighted sum of density-adjusted probabilities."""

    # Build weight map from config
    weight_map = {p.model_name: p.weight for p in config.pools}

    answer_scores = defaultdict(float)
    answer_metadata: Dict[str, Dict] = {}

    for model, response in pool_responses.items():
        if response.error or not response.results:
            continue

        weight = weight_map.get(model, 1.0)

        # Normalize to probabilities within pool
        total = sum(r.density_adjusted_score for r in response.results)

        for rank, r in enumerate(sorted(
            response.results, key=lambda x: -x.density_adjusted_score
        ), start=1):
            prob = r.density_adjusted_score / total if total > 0 else 0
            answer_scores[r.answer_hash] += weight * prob

            # Track metadata
            if r.answer_hash not in answer_metadata:
                answer_metadata[r.answer_hash] = {
                    'answer_id': r.answer_id,
                    'answer_text': r.answer_text,
                    'pool_contributions': {}
                }

            answer_metadata[r.answer_hash]['pool_contributions'][model] = PoolContribution(
                model_name=model,
                probability=prob,
                density=r.density_score,
                cluster_size=r.cluster_size,
                rank=rank
            )

    # Build final results
    ranked = sorted(answer_scores.items(), key=lambda x: -x[1])
    total_pools = len(pool_responses)

    return [
        FusedResult(
            answer_id=answer_metadata[ahash]['answer_id'],
            answer_text=answer_metadata[ahash]['answer_text'],
            answer_hash=ahash,
            fused_score=score,
            num_pools=len(answer_metadata[ahash]['pool_contributions']),
            consensus_strength=len(answer_metadata[ahash]['pool_contributions']) / total_pools,
            pool_contributions=answer_metadata[ahash]['pool_contributions']
        )
        for ahash, score in ranked[:top_k]
    ]


def rrf_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Reciprocal Rank Fusion - combines rankings, not scores."""

    answer_rrf_scores = defaultdict(float)
    answer_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
    answer_metadata: Dict[str, Dict] = {}

    for model, response in pool_responses.items():
        if response.error or not response.results:
            continue

        # Sort by density-adjusted score to get ranks
        sorted_results = sorted(
            response.results,
            key=lambda r: -r.density_adjusted_score
        )

        for rank, r in enumerate(sorted_results, start=1):
            rrf_contribution = 1.0 / (config.rrf_k + rank)
            answer_rrf_scores[r.answer_hash] += rrf_contribution
            answer_ranks[r.answer_hash][model] = rank

            if r.answer_hash not in answer_metadata:
                answer_metadata[r.answer_hash] = {
                    'answer_id': r.answer_id,
                    'answer_text': r.answer_text
                }

    ranked = sorted(answer_rrf_scores.items(), key=lambda x: -x[1])
    total_pools = len(pool_responses)

    return [
        FusedResult(
            answer_id=answer_metadata[ahash]['answer_id'],
            answer_text=answer_metadata[ahash]['answer_text'],
            answer_hash=ahash,
            fused_score=score,
            num_pools=len(answer_ranks[ahash]),
            consensus_strength=len(answer_ranks[ahash]) / total_pools,
            ranks_by_pool=answer_ranks[ahash]
        )
        for ahash, score in ranked[:top_k]
    ]


def consensus_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Boost answers with high density across multiple pools."""

    # Build weight map from config
    weight_map = {p.model_name: p.weight for p in config.pools}

    answer_evidence: Dict[str, List[Dict]] = defaultdict(list)
    answer_metadata: Dict[str, Dict] = {}

    # Collect evidence from each pool
    for model, response in pool_responses.items():
        if response.error or not response.results:
            continue

        weight = weight_map.get(model, 1.0)

        # Normalize probabilities
        total = sum(r.density_adjusted_score for r in response.results)

        for rank, r in enumerate(sorted(
            response.results, key=lambda x: -x.density_adjusted_score
        ), start=1):
            prob = r.density_adjusted_score / total if total > 0 else 0

            if prob >= config.consensus_threshold:
                answer_evidence[r.answer_hash].append({
                    'model': model,
                    'probability': prob,
                    'density': r.density_score,
                    'cluster_size': r.cluster_size,
                    'cluster_confidence': r.cluster_confidence,
                    'weight': weight,
                    'rank': rank
                })

                if r.answer_hash not in answer_metadata:
                    answer_metadata[r.answer_hash] = {
                        'answer_id': r.answer_id,
                        'answer_text': r.answer_text
                    }

    # Compute fused scores with consensus boost
    final_scores: Dict[str, Dict] = {}
    total_pools = len(pool_responses)

    for answer_hash, evidences in answer_evidence.items():
        # Base score: weighted sum of probabilities
        base_score = sum(e['probability'] * e['weight'] for e in evidences)

        # Consensus boost: reward multi-pool agreement
        num_pools = len(evidences)

        if num_pools >= config.min_pools_for_consensus:
            # Geometric mean of densities (rewards consistent high density)
            densities = [e['density'] for e in evidences if e['density'] > 0]
            if densities:
                density_agreement = geometric_mean(densities)
                # Boost proportional to agreement strength
                boost = 1 + (config.consensus_boost_factor - 1) * density_agreement
                base_score *= boost

            # Additional boost for cluster size agreement
            cluster_sizes = [e['cluster_size'] for e in evidences]
            if min(cluster_sizes) >= 3:  # All pools have substantial clusters
                base_score *= 1.1

        # Build pool contributions
        pool_contributions = {}
        for e in evidences:
            pool_contributions[e['model']] = PoolContribution(
                model_name=e['model'],
                probability=e['probability'],
                density=e['density'],
                cluster_size=e['cluster_size'],
                rank=e['rank']
            )

        final_scores[answer_hash] = {
            'score': base_score,
            'num_pools': num_pools,
            'pool_contributions': pool_contributions
        }

    ranked = sorted(final_scores.items(), key=lambda x: -x[1]['score'])

    return [
        FusedResult(
            answer_id=answer_metadata[ahash]['answer_id'],
            answer_text=answer_metadata[ahash]['answer_text'],
            answer_hash=ahash,
            fused_score=data['score'],
            num_pools=data['num_pools'],
            consensus_strength=data['num_pools'] / total_pools,
            pool_contributions=data['pool_contributions']
        )
        for ahash, data in ranked[:top_k]
        if ahash in answer_metadata
    ]


def geometric_mean_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Geometric mean of probabilities across pools."""

    answer_probs: Dict[str, List[float]] = defaultdict(list)
    answer_metadata: Dict[str, Dict] = {}
    answer_contributions: Dict[str, Dict[str, PoolContribution]] = defaultdict(dict)

    for model, response in pool_responses.items():
        if response.error or not response.results:
            continue

        # Normalize to probabilities within pool
        total = sum(r.density_adjusted_score for r in response.results)

        for rank, r in enumerate(sorted(
            response.results, key=lambda x: -x.density_adjusted_score
        ), start=1):
            prob = r.density_adjusted_score / total if total > 0 else 0
            answer_probs[r.answer_hash].append(prob)

            if r.answer_hash not in answer_metadata:
                answer_metadata[r.answer_hash] = {
                    'answer_id': r.answer_id,
                    'answer_text': r.answer_text
                }

            answer_contributions[r.answer_hash][model] = PoolContribution(
                model_name=model,
                probability=prob,
                density=r.density_score,
                cluster_size=r.cluster_size,
                rank=rank
            )

    # Compute geometric mean for each answer
    answer_scores = {
        ahash: geometric_mean(probs)
        for ahash, probs in answer_probs.items()
    }

    ranked = sorted(answer_scores.items(), key=lambda x: -x[1])
    total_pools = len(pool_responses)

    return [
        FusedResult(
            answer_id=answer_metadata[ahash]['answer_id'],
            answer_text=answer_metadata[ahash]['answer_text'],
            answer_hash=ahash,
            fused_score=score,
            num_pools=len(answer_probs[ahash]),
            consensus_strength=len(answer_probs[ahash]) / total_pools,
            pool_contributions=answer_contributions[ahash]
        )
        for ahash, score in ranked[:top_k]
    ]


def max_fusion(
    pool_responses: Dict[str, PoolResponse],
    config: CrossModelConfig,
    top_k: int
) -> List[FusedResult]:
    """Take maximum probability across pools."""

    answer_max: Dict[str, Tuple[float, str]] = {}  # hash -> (max_prob, best_model)
    answer_metadata: Dict[str, Dict] = {}
    answer_contributions: Dict[str, Dict[str, PoolContribution]] = defaultdict(dict)

    for model, response in pool_responses.items():
        if response.error or not response.results:
            continue

        # Normalize to probabilities within pool
        total = sum(r.density_adjusted_score for r in response.results)

        for rank, r in enumerate(sorted(
            response.results, key=lambda x: -x.density_adjusted_score
        ), start=1):
            prob = r.density_adjusted_score / total if total > 0 else 0

            if r.answer_hash not in answer_max or prob > answer_max[r.answer_hash][0]:
                answer_max[r.answer_hash] = (prob, model)

            if r.answer_hash not in answer_metadata:
                answer_metadata[r.answer_hash] = {
                    'answer_id': r.answer_id,
                    'answer_text': r.answer_text
                }

            answer_contributions[r.answer_hash][model] = PoolContribution(
                model_name=model,
                probability=prob,
                density=r.density_score,
                cluster_size=r.cluster_size,
                rank=rank
            )

    ranked = sorted(answer_max.items(), key=lambda x: -x[1][0])
    total_pools = len(pool_responses)

    return [
        FusedResult(
            answer_id=answer_metadata[ahash]['answer_id'],
            answer_text=answer_metadata[ahash]['answer_text'],
            answer_hash=ahash,
            fused_score=max_prob,
            num_pools=len(answer_contributions[ahash]),
            consensus_strength=len(answer_contributions[ahash]) / total_pools,
            pool_contributions=answer_contributions[ahash]
        )
        for ahash, (max_prob, _) in ranked[:top_k]
    ]


# =============================================================================
# POOL ROUTER (filters nodes by model)
# =============================================================================

class PoolRouter:
    """A router that only sees nodes for a specific embedding model."""

    def __init__(
        self,
        base_router: KleinbergRouter,
        model_name: str,
        node_filter: Optional[Callable[[KGNode], bool]] = None
    ):
        self.base_router = base_router
        self.model_name = model_name
        self.node_filter = node_filter
        self._filtered_nodes: Optional[List[KGNode]] = None

    def discover_nodes(self, tags: Optional[List[str]] = None) -> List[KGNode]:
        """Discover nodes filtered by embedding model."""
        if self._filtered_nodes is None:
            all_nodes = self.base_router.discover_nodes(tags)
            self._filtered_nodes = [
                n for n in all_nodes
                if n.metadata.get('embedding_model') == self.model_name
                and (self.node_filter is None or self.node_filter(n))
            ]
        return self._filtered_nodes

    def get_stats(self) -> Dict[str, Any]:
        """Get pool-specific stats."""
        base_stats = self.base_router.get_stats()
        return {
            **base_stats,
            'model_name': self.model_name,
            'pool_size': len(self._filtered_nodes) if self._filtered_nodes else 0
        }

    # Delegate other methods to base router
    def __getattr__(self, name):
        return getattr(self.base_router, name)


# =============================================================================
# CROSS-MODEL FEDERATED ENGINE
# =============================================================================

class CrossModelFederatedEngine:
    """
    Federate queries across nodes with different embedding models.

    Two-phase architecture:
    1. Query each model pool in parallel (density scoring works within pool)
    2. Fuse results across pools using score/rank-based methods
    """

    def __init__(
        self,
        router: KleinbergRouter,
        config: CrossModelConfig,
        embedding_fn: Optional[Callable[[str, str], np.ndarray]] = None
    ):
        """
        Initialize cross-model engine.

        Args:
            router: Base router for node discovery
            config: Cross-model configuration
            embedding_fn: Function(text, model_name) -> embedding vector
        """
        self.router = router
        self.config = config
        self.embedding_fn = embedding_fn

        # Pool engines (initialized lazily)
        self.pool_engines: Dict[str, FederatedQueryEngine] = {}
        self.pool_routers: Dict[str, PoolRouter] = {}

        # Statistics
        self.queries_executed = 0
        self.total_latency_ms = 0.0

    def discover_pools(self) -> Dict[str, List[str]]:
        """
        Discover and group nodes by embedding model.

        Returns:
            Dict mapping model_name -> list of node_ids
        """
        all_nodes = self.router.discover_nodes()
        pools: Dict[str, List[str]] = defaultdict(list)

        for node in all_nodes:
            model = node.metadata.get('embedding_model', 'unknown')
            pools[model].append(node.node_id)

        return pools

    def _init_pool_engine(self, pool_config: ModelPoolConfig) -> Optional[FederatedQueryEngine]:
        """Initialize a FederatedQueryEngine for a model pool."""
        model = pool_config.model_name

        # Create pool-specific router
        pool_router = PoolRouter(
            base_router=self.router,
            model_name=model,
            node_filter=pool_config.node_filter
        )

        # Check if pool has any nodes
        nodes = pool_router.discover_nodes()
        if not nodes:
            return None

        self.pool_routers[model] = pool_router

        # Create aggregation config
        agg_config = AggregationConfig(
            strategy=pool_config.aggregation_strategy,
            dedup_key='answer_hash'
        )

        # Create engine (could use Adaptive/Hierarchical based on config)
        engine = FederatedQueryEngine(
            router=pool_router,
            aggregation_config=agg_config,
            timeout_ms=self.config.pool_timeout_ms
        )

        return engine

    def _ensure_pool_engines(self):
        """Lazily initialize pool engines."""
        if self.pool_engines:
            return

        for pool_config in self.config.pools:
            engine = self._init_pool_engine(pool_config)
            if engine:
                self.pool_engines[pool_config.model_name] = engine

    def _get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get embedding for text using specified model."""
        if self.embedding_fn:
            return self.embedding_fn(text, model)
        return None

    def _query_pool(
        self,
        model: str,
        engine: FederatedQueryEngine,
        query_text: str,
        embedding: Optional[np.ndarray],
        top_k: int,
        federation_k: int
    ) -> PoolResponse:
        """Query a single model pool."""
        start_time = time.time()

        try:
            # Execute federated query within pool
            response = engine.federated_query(
                query_text=query_text,
                embedding=embedding,
                top_k=top_k,
                federation_k=federation_k
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Convert to PoolResult format
            pool_results = []
            for r in response.results:
                # r is a dict from AggregatedResult.to_dict()
                pool_results.append(PoolResult(
                    answer_id=str(r.get('answer_hash', '')),  # Use hash as ID for cross-model
                    answer_text=r.get('answer_text', ''),
                    answer_hash=r.get('answer_hash', ''),
                    raw_score=r.get('combined_score', 0.0),
                    exp_score=r.get('combined_score', 0.0),
                    density_adjusted_score=r.get('combined_score', 0.0),
                    density_score=r.get('density_score', 0.0),
                    cluster_id=r.get('cluster_id', -1) or -1,
                    cluster_size=r.get('node_count', 0),
                    cluster_confidence=r.get('cluster_confidence', 0.0),
                    source_nodes=r.get('source_nodes', []),
                    model_name=model
                ))

            # Compute pool statistics
            densities = [r.density_score for r in pool_results if r.density_score > 0]
            cluster_ids = set(r.cluster_id for r in pool_results if r.cluster_id >= 0)

            return PoolResponse(
                model_name=model,
                results=pool_results,
                total_results=len(pool_results),
                num_clusters=len(cluster_ids),
                avg_density=sum(densities) / len(densities) if densities else 0.0,
                partition_sum=response.total_partition_sum,
                query_time_ms=elapsed_ms,
                nodes_queried=response.nodes_queried,
                nodes_responded=response.nodes_responded
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return PoolResponse(
                model_name=model,
                results=[],
                query_time_ms=elapsed_ms,
                error=str(e)
            )

    def federated_query(
        self,
        query_text: str,
        top_k: int = 10,
        federation_k: Optional[int] = None,
        timeout_ms: Optional[int] = None
    ) -> CrossModelResponse:
        """
        Execute cross-model federated query.

        Args:
            query_text: The query string
            top_k: Number of results to return
            federation_k: Nodes to query per pool (default from pool config)
            timeout_ms: Total timeout (default from config)

        Returns:
            CrossModelResponse with fused results
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        timeout_ms = timeout_ms or self.config.pool_timeout_ms

        # Ensure pool engines are initialized
        self._ensure_pool_engines()

        if not self.pool_engines:
            return CrossModelResponse(
                query_id=query_id,
                results=[],
                pools_queried=[],
                pools_responded=0,
                total_time_ms=0.0,
                fusion_method=self.config.fusion_method.value
            )

        # Phase 1: Query each pool in parallel
        pool_responses: Dict[str, PoolResponse] = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for pool_config in self.config.pools:
                model = pool_config.model_name
                engine = self.pool_engines.get(model)

                if not engine:
                    continue

                # Get embedding for this model
                embedding = self._get_embedding(query_text, model)

                # Use pool-specific federation_k or default
                pool_federation_k = federation_k or pool_config.federation_k

                future = executor.submit(
                    self._query_pool,
                    model, engine, query_text, embedding, top_k, pool_federation_k
                )
                futures[future] = model

            # Collect results
            for future in as_completed(futures, timeout=timeout_ms / 1000):
                model = futures[future]
                try:
                    pool_responses[model] = future.result()
                except Exception as e:
                    pool_responses[model] = PoolResponse(
                        model_name=model,
                        results=[],
                        error=str(e)
                    )

        # Phase 2: Fuse results
        if self.config.fusion_method == FusionMethod.WEIGHTED_SUM:
            fused = weighted_sum_fusion(pool_responses, self.config, top_k)
        elif self.config.fusion_method == FusionMethod.RECIPROCAL_RANK:
            fused = rrf_fusion(pool_responses, self.config, top_k)
        elif self.config.fusion_method == FusionMethod.CONSENSUS:
            fused = consensus_fusion(pool_responses, self.config, top_k)
        elif self.config.fusion_method == FusionMethod.GEOMETRIC_MEAN:
            fused = geometric_mean_fusion(pool_responses, self.config, top_k)
        elif self.config.fusion_method == FusionMethod.MAX:
            fused = max_fusion(pool_responses, self.config, top_k)
        else:
            fused = weighted_sum_fusion(pool_responses, self.config, top_k)

        elapsed_ms = (time.time() - start_time) * 1000

        # Update statistics
        self.queries_executed += 1
        self.total_latency_ms += elapsed_ms

        pools_responded = sum(
            1 for r in pool_responses.values()
            if r.error is None and r.results
        )

        return CrossModelResponse(
            query_id=query_id,
            results=fused,
            pools_queried=list(pool_responses.keys()),
            pools_responded=pools_responded,
            total_time_ms=elapsed_ms,
            fusion_method=self.config.fusion_method.value
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'queries_executed': self.queries_executed,
            'avg_latency_ms': (
                self.total_latency_ms / self.queries_executed
                if self.queries_executed > 0 else 0.0
            ),
            'pools_configured': len(self.config.pools),
            'pools_active': len(self.pool_engines),
            'pool_stats': {
                model: router.get_stats()
                for model, router in self.pool_routers.items()
            }
        }


# =============================================================================
# WEIGHT LEARNING (Phase 6e-iv)
# =============================================================================

class AdaptiveModelWeights:
    """Learn optimal model weights from user feedback."""

    def __init__(
        self,
        models: List[str],
        learning_rate: float = 0.01,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # Start with uniform weights
            n = len(models)
            self.weights = {m: 1.0 / n for m in models}

        self.learning_rate = learning_rate
        self.feedback_count = 0
        self.models = models

    def update(
        self,
        chosen_answer: str,
        pool_rankings: Dict[str, List[str]]  # model -> [answer_hashes in rank order]
    ):
        """
        Update weights based on which models ranked chosen answer highest.

        Args:
            chosen_answer: Hash of the answer the user selected
            pool_rankings: Rankings from each model pool
        """
        rewards = {}

        for model, ranking in pool_rankings.items():
            if chosen_answer in ranking:
                rank = ranking.index(chosen_answer) + 1
                # Higher reward for better rank (inverse rank)
                rewards[model] = 1.0 / rank
            else:
                rewards[model] = 0.0

        # Normalize rewards
        total_reward = sum(rewards.values())
        if total_reward > 0:
            rewards = {m: r / total_reward for m, r in rewards.items()}

        # Gradient update toward rewarded models
        for model in self.weights:
            target = rewards.get(model, 0.0)
            current = self.weights[model]
            self.weights[model] += self.learning_rate * (target - current)

        # Renormalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {m: w / total for m, w in self.weights.items()}

        self.feedback_count += 1

    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'weights': self.weights,
            'feedback_count': self.feedback_count,
            'learning_rate': self.learning_rate
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AdaptiveModelWeights':
        """Deserialize from dictionary."""
        obj = cls(
            models=list(d['weights'].keys()),
            learning_rate=d.get('learning_rate', 0.01),
            initial_weights=d['weights']
        )
        obj.feedback_count = d.get('feedback_count', 0)
        return obj
