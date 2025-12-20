# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Federated Query Engine for distributed KG topology.

"""
Federated Query Engine for KG Topology Phase 4.

Implements distributed query algebra with pluggable aggregation functions.
Treats federated queries as GROUP BY with configurable merge strategies.

Key concepts:
- Distributed softmax: partition_sum aggregation for probability normalization
- Pluggable aggregation: SUM, MAX, AVG, COUNT, FIRST, DIVERSITY_WEIGHTED
- Dedup as aggregation: duplicate handling is just another GROUP BY strategy

See: docs/proposals/FEDERATED_QUERY_ALGEBRA.md
"""

import hashlib
import math
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
import json
import urllib.request
import urllib.error

import numpy as np

try:
    from .kleinberg_router import KleinbergRouter, KGNode, RoutingEnvelope
    from .discovery_clients import DiscoveryClient
except ImportError:
    from kleinberg_router import KleinbergRouter, KGNode, RoutingEnvelope
    from discovery_clients import DiscoveryClient


# =============================================================================
# AGGREGATION STRATEGIES
# =============================================================================

class AggregationStrategy(Enum):
    """Strategies for merging duplicate results across nodes."""
    SUM = "sum"                    # exp(z_a) + exp(z_b) - boost consensus
    MAX = "max"                    # max(exp(z_a), exp(z_b)) - no boost
    MIN = "min"                    # min(exp(z_a), exp(z_b)) - pessimistic
    AVG = "avg"                    # average scores
    COUNT = "count"               # count occurrences only
    FIRST = "first"               # keep first seen
    COLLECT = "collect"           # keep all (no dedup)
    DIVERSITY_WEIGHTED = "diversity"  # boost only if sources differ
    DENSITY_FLUX = "density_flux"     # Phase 4d: density-weighted softmax


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    strategy: AggregationStrategy = AggregationStrategy.SUM
    dedup_key: str = "answer_hash"  # Field to group by
    consensus_threshold: Optional[int] = None  # Minimum node agreement
    diversity_field: str = "corpus_id"  # Field for diversity checking
    # Phase 4d: Density scoring options
    density_weight: float = 0.3  # Weight for density in flux-softmax (0 = ignore)
    clustering_enabled: bool = True  # Enable two-stage clustering
    similarity_threshold: float = 0.7  # Cluster assignment threshold
    min_cluster_size: int = 2  # Minimum cluster size (smaller = noise)


# =============================================================================
# AGGREGATION FUNCTIONS (Monoid-like operations)
# =============================================================================

class Aggregator(ABC):
    """Abstract base for aggregation functions.

    Aggregators must be:
    - Associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
    - Commutative: a ⊕ b = b ⊕ a (for order-independent merging)
    - Have identity: a ⊕ e = a
    """

    @abstractmethod
    def identity(self) -> Any:
        """Return the identity element."""
        pass

    @abstractmethod
    def merge(self, a: Any, b: Any) -> Any:
        """Merge two values."""
        pass

    @abstractmethod
    def finalize(self, value: Any) -> float:
        """Finalize accumulated value to score."""
        pass


class SumAggregator(Aggregator):
    """Sum aggregator - boosts consensus."""

    def identity(self) -> float:
        return 0.0

    def merge(self, a: float, b: float) -> float:
        return a + b

    def finalize(self, value: float) -> float:
        return value


class MaxAggregator(Aggregator):
    """Max aggregator - takes best, no boost."""

    def identity(self) -> float:
        return float('-inf')

    def merge(self, a: float, b: float) -> float:
        return max(a, b)

    def finalize(self, value: float) -> float:
        return value if value != float('-inf') else 0.0


class MinAggregator(Aggregator):
    """Min aggregator - pessimistic estimate."""

    def identity(self) -> float:
        return float('inf')

    def merge(self, a: float, b: float) -> float:
        return min(a, b)

    def finalize(self, value: float) -> float:
        return value if value != float('inf') else 0.0


class AvgAggregator(Aggregator):
    """Average aggregator - maintains (sum, count) tuple."""

    def identity(self) -> Tuple[float, int]:
        return (0.0, 0)

    def merge(self, a: Tuple[float, int], b: Tuple[float, int]) -> Tuple[float, int]:
        return (a[0] + b[0], a[1] + b[1])

    def finalize(self, value: Tuple[float, int]) -> float:
        total, count = value
        return total / count if count > 0 else 0.0


class CountAggregator(Aggregator):
    """Count aggregator - counts occurrences."""

    def identity(self) -> int:
        return 0

    def merge(self, a: int, b: int) -> int:
        return a + b

    def finalize(self, value: int) -> float:
        return float(value)


class FirstAggregator(Aggregator):
    """First aggregator - keeps first seen (by timestamp)."""

    def identity(self) -> Tuple[float, float]:
        return (0.0, float('inf'))  # (value, timestamp)

    def merge(self, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        return a if a[1] <= b[1] else b

    def finalize(self, value: Tuple[float, float]) -> float:
        return value[0]


def get_aggregator(strategy: AggregationStrategy) -> Aggregator:
    """Factory for aggregators."""
    return {
        AggregationStrategy.SUM: SumAggregator(),
        AggregationStrategy.MAX: MaxAggregator(),
        AggregationStrategy.MIN: MinAggregator(),
        AggregationStrategy.AVG: AvgAggregator(),
        AggregationStrategy.COUNT: CountAggregator(),
        AggregationStrategy.FIRST: FirstAggregator(),
    }.get(strategy, SumAggregator())


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NodeResult:
    """A single result from a node query."""
    answer_id: int
    answer_text: str
    answer_hash: str
    raw_score: float
    exp_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 4d: Embedding for density computation
    embedding: Optional[np.ndarray] = None
    local_density: float = 0.0  # Density computed at source node

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'answer_id': self.answer_id,
            'answer_text': self.answer_text,
            'answer_hash': self.answer_hash,
            'raw_score': self.raw_score,
            'exp_score': self.exp_score,
            'metadata': self.metadata,
            'local_density': self.local_density
        }
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NodeResult':
        embedding = None
        if 'embedding' in d and d['embedding'] is not None:
            embedding = np.array(d['embedding'])
        return cls(
            answer_id=d['answer_id'],
            answer_text=d['answer_text'],
            answer_hash=d['answer_hash'],
            raw_score=d['raw_score'],
            exp_score=d['exp_score'],
            metadata=d.get('metadata', {}),
            embedding=embedding,
            local_density=d.get('local_density', 0.0)
        )


@dataclass
class NodeResponse:
    """Response from a single node in federated query."""
    source_node: str
    results: List[NodeResult]
    partition_sum: float
    node_metadata: Dict[str, Any]
    response_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__type': 'kg_federated_response',
            'source_node': self.source_node,
            'results': [r.to_dict() for r in self.results],
            'partition_sum': self.partition_sum,
            'node_metadata': self.node_metadata,
            'response_time_ms': self.response_time_ms,
            'error': self.error
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NodeResponse':
        return cls(
            source_node=d['source_node'],
            results=[NodeResult.from_dict(r) for r in d.get('results', [])],
            partition_sum=d.get('partition_sum', 0.0),
            node_metadata=d.get('node_metadata', {}),
            response_time_ms=d.get('response_time_ms', 0.0),
            error=d.get('error')
        )


@dataclass
class ResultProvenance:
    """Tracks where a result came from for diversity analysis."""
    node_id: str
    exp_score: float
    corpus_id: Optional[str] = None
    data_sources: List[str] = field(default_factory=list)
    interface_id: Optional[int] = None
    embedding_model: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    # Phase 4d: Density information
    embedding: Optional[np.ndarray] = None
    density_score: float = 0.0
    cluster_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'node_id': self.node_id,
            'exp_score': self.exp_score,
            'corpus_id': self.corpus_id,
            'data_sources': self.data_sources,
            'interface_id': self.interface_id,
            'embedding_model': self.embedding_model,
            'density_score': self.density_score,
            'cluster_id': self.cluster_id
        }
        return result


@dataclass
class AggregatedResult:
    """A result aggregated from multiple nodes."""
    answer_text: str
    answer_hash: str
    combined_score: Any  # Type depends on aggregator
    source_nodes: List[str] = field(default_factory=list)
    provenance: List[ResultProvenance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Phase 4d: Density information
    semantic_centroid: Optional[np.ndarray] = None  # Mean embedding
    density_score: float = 0.0  # Global density after aggregation
    cluster_id: Optional[int] = None
    cluster_confidence: float = 0.0  # density * cluster_size

    @classmethod
    def from_single(cls, result: NodeResult, node_id: str,
                    node_metadata: Dict[str, Any]) -> 'AggregatedResult':
        """Create from a single node result."""
        return cls(
            answer_text=result.answer_text,
            answer_hash=result.answer_hash,
            combined_score=result.exp_score,
            source_nodes=[node_id],
            provenance=[ResultProvenance(
                node_id=node_id,
                exp_score=result.exp_score,
                corpus_id=node_metadata.get('corpus_id'),
                data_sources=node_metadata.get('data_sources', []),
                interface_id=node_metadata.get('interface_id'),
                embedding_model=node_metadata.get('embedding_model'),
                embedding=result.embedding,
                density_score=result.local_density
            )],
            metadata=result.metadata.copy(),
            semantic_centroid=result.embedding.copy() if result.embedding is not None else None,
            density_score=result.local_density
        )

    def to_dict(self, normalized_prob: float = 0.0) -> Dict[str, Any]:
        # Calculate diversity score based on unique corpus_ids
        unique_corpora = set(
            p.corpus_id for p in self.provenance if p.corpus_id is not None
        )
        diversity_score = len(unique_corpora) / len(self.provenance) if self.provenance else 0.0

        return {
            'answer_text': self.answer_text,
            'answer_hash': self.answer_hash,
            'combined_score': float(self.combined_score) if not isinstance(
                self.combined_score, tuple) else self.combined_score[0],
            'normalized_prob': normalized_prob,
            'source_nodes': self.source_nodes,
            'node_count': len(self.source_nodes),
            'diversity_score': round(diversity_score, 3),
            'unique_corpora': len(unique_corpora),
            'provenance': [p.to_dict() for p in self.provenance],
            'metadata': self.metadata,
            # Phase 4d: Density metrics
            'density_score': round(self.density_score, 4),
            'cluster_id': self.cluster_id,
            'cluster_confidence': round(self.cluster_confidence, 4)
        }


@dataclass
class AggregatedResponse:
    """Final aggregated response from federated query."""
    query_id: str
    results: List[Dict[str, Any]]
    total_partition_sum: float
    nodes_queried: int
    nodes_responded: int
    total_time_ms: float
    aggregation_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__type': 'kg_aggregated_response',
            '__id': self.query_id,
            'results': self.results,
            'total_partition_sum': self.total_partition_sum,
            'nodes_queried': self.nodes_queried,
            'nodes_responded': self.nodes_responded,
            'total_time_ms': self.total_time_ms,
            'aggregation_strategy': self.aggregation_strategy
        }


# =============================================================================
# FEDERATED QUERY ENGINE
# =============================================================================

class FederatedQueryEngine:
    """
    Engine for executing federated queries across distributed KG nodes.

    Implements distributed query algebra with pluggable aggregation.
    """

    def __init__(
        self,
        router: KleinbergRouter,
        aggregation_config: Optional[AggregationConfig] = None,
        federation_k: int = 3,
        timeout_ms: int = 5000,
        max_workers: int = 10
    ):
        """
        Initialize federated query engine.

        Args:
            router: KleinbergRouter for node discovery
            aggregation_config: Configuration for result aggregation
            federation_k: Number of nodes to query in parallel
            timeout_ms: Query timeout in milliseconds
            max_workers: Max parallel workers for queries
        """
        self.router = router
        self.config = aggregation_config or AggregationConfig()
        self.federation_k = federation_k
        self.timeout_ms = timeout_ms
        self.max_workers = max_workers

        # Statistics
        self._query_count = 0
        self._total_time_ms = 0.0
        self._node_response_times: Dict[str, List[float]] = {}

    def federated_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        federation_k: Optional[int] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None
    ) -> AggregatedResponse:
        """
        Execute a federated query across multiple KG nodes.

        Args:
            query_text: The query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            federation_k: Override default federation_k
            aggregation_strategy: Override default aggregation strategy

        Returns:
            AggregatedResponse with merged results from all nodes
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())

        k = federation_k or self.federation_k
        strategy = aggregation_strategy or self.config.strategy

        # 1. Discover top-k nodes by centroid similarity
        nodes = self.router.discover_nodes()
        if not nodes:
            return self._empty_response(query_id, strategy)

        ranked = self.router.compute_routing_probability(nodes, query_embedding)
        target_nodes = [node for node, _ in ranked[:k]]

        # 2. Query all nodes in parallel
        responses = self._parallel_query(
            target_nodes, query_text, query_embedding, query_id
        )

        # 3. Aggregate results
        aggregated, total_partition = self._aggregate(responses, strategy)

        # 4. Normalize and rank
        results = self._normalize_and_rank(aggregated, total_partition, top_k)

        # 5. Apply consensus threshold if configured
        if self.config.consensus_threshold:
            results = [
                r for r in results
                if r['node_count'] >= self.config.consensus_threshold
            ]

        total_time = (time.time() - start_time) * 1000
        self._query_count += 1
        self._total_time_ms += total_time

        return AggregatedResponse(
            query_id=query_id,
            results=results,
            total_partition_sum=total_partition,
            nodes_queried=len(target_nodes),
            nodes_responded=len([r for r in responses if r.error is None]),
            total_time_ms=total_time,
            aggregation_strategy=strategy.value
        )

    def _parallel_query(
        self,
        nodes: List[KGNode],
        query_text: str,
        query_embedding: np.ndarray,
        query_id: str
    ) -> List[NodeResponse]:
        """Query multiple nodes in parallel."""
        responses = []
        timeout_sec = self.timeout_ms / 1000.0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._query_node, node, query_text, query_embedding, query_id
                ): node
                for node in nodes
            }

            for future in as_completed(futures, timeout=timeout_sec):
                node = futures[future]
                try:
                    response = future.result()
                    responses.append(response)

                    # Track response times
                    if node.node_id not in self._node_response_times:
                        self._node_response_times[node.node_id] = []
                    self._node_response_times[node.node_id].append(
                        response.response_time_ms
                    )

                except TimeoutError:
                    responses.append(NodeResponse(
                        source_node=node.node_id,
                        results=[],
                        partition_sum=0.0,
                        node_metadata={},
                        error="Timeout"
                    ))
                except Exception as e:
                    responses.append(NodeResponse(
                        source_node=node.node_id,
                        results=[],
                        partition_sum=0.0,
                        node_metadata={},
                        error=str(e)
                    ))

        return responses

    def _query_node(
        self,
        node: KGNode,
        query_text: str,
        query_embedding: np.ndarray,
        query_id: str
    ) -> NodeResponse:
        """Query a single node."""
        start_time = time.time()

        request = {
            '__type': 'kg_federated_query',
            '__id': query_id,
            '__routing': {
                'origin_node': self.router.local_node_id,
                'federation_k': self.federation_k,
                'aggregation': {
                    'score_function': self.config.strategy.value,
                    'dedup_key': self.config.dedup_key
                }
            },
            '__embedding': {
                'model': node.embedding_model,
                'vector': query_embedding.tolist()
            },
            'payload': {
                'query_text': query_text,
                'top_k': 10  # Get more than needed for aggregation
            }
        }

        try:
            url = f"{node.endpoint}/kg/federated"
            req = urllib.request.Request(
                url,
                data=json.dumps(request).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=self.timeout_ms/1000) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                response_time = (time.time() - start_time) * 1000

                return NodeResponse(
                    source_node=node.node_id,
                    results=[NodeResult.from_dict(r) for r in data.get('results', [])],
                    partition_sum=data.get('partition_sum', 0.0),
                    node_metadata=data.get('node_metadata', {}),
                    response_time_ms=response_time
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return NodeResponse(
                source_node=node.node_id,
                results=[],
                partition_sum=0.0,
                node_metadata={},
                response_time_ms=response_time,
                error=str(e)
            )

    def _aggregate(
        self,
        responses: List[NodeResponse],
        strategy: AggregationStrategy
    ) -> Tuple[Dict[str, AggregatedResult], float]:
        """
        Aggregate results from multiple nodes.

        Returns:
            Tuple of (results dict keyed by answer_hash, total partition sum)
        """
        results: Dict[str, AggregatedResult] = {}
        total_partition = 0.0

        aggregator = get_aggregator(strategy)

        for resp in responses:
            if resp.error:
                continue

            total_partition += resp.partition_sum

            for r in resp.results:
                key = r.answer_hash

                if key in results:
                    existing = results[key]

                    # Handle diversity-weighted specially
                    if strategy == AggregationStrategy.DIVERSITY_WEIGHTED:
                        new_score = self._diversity_merge(
                            existing, r, resp.source_node, resp.node_metadata
                        )
                    elif strategy == AggregationStrategy.AVG:
                        # AVG needs tuple (sum, count)
                        if not isinstance(existing.combined_score, tuple):
                            existing.combined_score = (existing.combined_score, 1)
                        new_score = aggregator.merge(
                            existing.combined_score, (r.exp_score, 1)
                        )
                    elif strategy == AggregationStrategy.FIRST:
                        # FIRST needs tuple (value, timestamp)
                        if not isinstance(existing.combined_score, tuple):
                            existing.combined_score = (existing.combined_score,
                                                       existing.provenance[0].timestamp)
                        new_score = aggregator.merge(
                            existing.combined_score, (r.exp_score, time.time())
                        )
                    else:
                        new_score = aggregator.merge(
                            existing.combined_score, r.exp_score
                        )

                    existing.combined_score = new_score
                    existing.source_nodes.append(resp.source_node)
                    existing.provenance.append(ResultProvenance(
                        node_id=resp.source_node,
                        exp_score=r.exp_score,
                        corpus_id=resp.node_metadata.get('corpus_id'),
                        data_sources=resp.node_metadata.get('data_sources', []),
                        interface_id=resp.node_metadata.get('interface_id'),
                        embedding_model=resp.node_metadata.get('embedding_model')
                    ))
                else:
                    results[key] = AggregatedResult.from_single(
                        r, resp.source_node, resp.node_metadata
                    )

                    # Initialize tuple types for AVG/FIRST
                    if strategy == AggregationStrategy.AVG:
                        results[key].combined_score = (r.exp_score, 1)
                    elif strategy == AggregationStrategy.FIRST:
                        results[key].combined_score = (r.exp_score, time.time())

        return results, total_partition

    def _diversity_merge(
        self,
        existing: AggregatedResult,
        new_result: NodeResult,
        node_id: str,
        node_metadata: Dict[str, Any]
    ) -> float:
        """
        Merge with diversity weighting - boost only if sources differ.

        Diversity is determined by:
        1. Different corpus_id -> fully diverse -> full boost
        2. Same corpus_id but different data_sources -> partially diverse -> partial boost
        3. Same corpus_id and overlapping data_sources -> not diverse -> no boost

        Note: A related concept is result density (semantic clustering).
        High density of semantically similar results indicates stronger consensus.
        TODO: Add density-based confidence scoring in future phase.
        """
        new_corpus = node_metadata.get(self.config.diversity_field)
        new_data_sources = set(node_metadata.get('data_sources', []))

        # Check corpus diversity
        corpus_diverse = any(
            p.corpus_id != new_corpus
            for p in existing.provenance
            if p.corpus_id is not None
        )

        if corpus_diverse or new_corpus is None:
            # Different corpus - fully diverse, full boost
            return existing.combined_score + new_result.exp_score

        # Same corpus - check data source overlap
        if new_data_sources:
            existing_sources = set()
            for p in existing.provenance:
                if hasattr(p, 'data_sources'):
                    existing_sources.update(p.data_sources)

            if existing_sources and not existing_sources.intersection(new_data_sources):
                # Same corpus but disjoint data sources - partial boost (average of sum and max)
                sum_score = existing.combined_score + new_result.exp_score
                max_score = max(existing.combined_score, new_result.exp_score)
                return (sum_score + max_score) / 2

        # Same source - no boost, take MAX
        return max(existing.combined_score, new_result.exp_score)

    def _normalize_and_rank(
        self,
        aggregated: Dict[str, AggregatedResult],
        total_partition: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Normalize scores and return top-k results."""
        results = []

        aggregator = get_aggregator(self.config.strategy)

        for result in aggregated.values():
            final_score = aggregator.finalize(result.combined_score)
            normalized_prob = final_score / total_partition if total_partition > 0 else 0.0
            results.append(result.to_dict(normalized_prob))

        # Sort by normalized probability
        results.sort(key=lambda r: r['normalized_prob'], reverse=True)

        return results[:top_k]

    def _empty_response(
        self,
        query_id: str,
        strategy: AggregationStrategy
    ) -> AggregatedResponse:
        """Return empty response when no nodes available."""
        return AggregatedResponse(
            query_id=query_id,
            results=[],
            total_partition_sum=0.0,
            nodes_queried=0,
            nodes_responded=0,
            total_time_ms=0.0,
            aggregation_strategy=strategy.value
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        avg_response_times = {
            node_id: sum(times) / len(times)
            for node_id, times in self._node_response_times.items()
            if times
        }

        return {
            'query_count': self._query_count,
            'avg_query_time_ms': (
                self._total_time_ms / self._query_count
                if self._query_count > 0 else 0.0
            ),
            'federation_k': self.federation_k,
            'aggregation_strategy': self.config.strategy.value,
            'node_response_times': avg_response_times
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_answer_hash(answer_text: str) -> str:
    """Compute hash for answer deduplication."""
    return hashlib.sha256(answer_text.encode()).hexdigest()[:16]


def compute_exp_scores(raw_scores: List[float]) -> Tuple[List[float], float]:
    """
    Compute exp scores and partition sum for softmax.

    Uses log-sum-exp trick for numerical stability.

    Returns:
        Tuple of (exp_scores list, partition_sum)
    """
    if not raw_scores:
        return [], 0.0

    # Log-sum-exp trick: subtract max for stability
    max_score = max(raw_scores)
    exp_scores = [math.exp(s - max_score) for s in raw_scores]
    partition_sum = sum(exp_scores)

    # Scale back
    scale = math.exp(max_score)
    exp_scores = [e * scale for e in exp_scores]
    partition_sum *= scale

    return exp_scores, partition_sum


def create_federated_engine(
    router: KleinbergRouter,
    strategy: str = "sum",
    federation_k: int = 3,
    timeout_ms: int = 5000,
    consensus_threshold: Optional[int] = None,
    diversity_field: str = "corpus_id",
    # Phase 4d: Density options
    density_weight: float = 0.3,
    clustering_enabled: bool = True,
    similarity_threshold: float = 0.7
) -> FederatedQueryEngine:
    """Factory function to create a FederatedQueryEngine."""
    strategy_enum = AggregationStrategy(strategy)

    config = AggregationConfig(
        strategy=strategy_enum,
        consensus_threshold=consensus_threshold,
        diversity_field=diversity_field,
        density_weight=density_weight,
        clustering_enabled=clustering_enabled,
        similarity_threshold=similarity_threshold
    )

    return FederatedQueryEngine(
        router=router,
        aggregation_config=config,
        federation_k=federation_k,
        timeout_ms=timeout_ms
    )


# =============================================================================
# PHASE 4D: DENSITY-WEIGHTED AGGREGATION
# =============================================================================

try:
    from .density_scoring import (
        DensityConfig,
        compute_density_scores,
        flux_softmax,
        cluster_by_similarity,
        compute_cluster_density,
        two_stage_density_pipeline,
        compute_cluster_stats,
        TransactionManager,
        get_transaction_manager
    )
    DENSITY_AVAILABLE = True
except ImportError:
    try:
        from density_scoring import (
            DensityConfig,
            compute_density_scores,
            flux_softmax,
            cluster_by_similarity,
            compute_cluster_density,
            two_stage_density_pipeline,
            compute_cluster_stats,
            TransactionManager,
            get_transaction_manager
        )
        DENSITY_AVAILABLE = True
    except ImportError:
        DENSITY_AVAILABLE = False


def apply_density_scoring(
    results: List[AggregatedResult],
    config: AggregationConfig
) -> List[AggregatedResult]:
    """
    Apply two-stage density scoring to aggregated results.

    Stage 1: Cluster results by semantic similarity
    Stage 2: Compute density within each cluster

    This implements the "flux-based softmax" where density acts as a
    multiplicative factor that concentrates probability in coherent regions.

    Args:
        results: List of aggregated results with embeddings
        config: Aggregation configuration with density options

    Returns:
        Results with updated density_score, cluster_id, cluster_confidence
    """
    if not DENSITY_AVAILABLE:
        return results

    if not results:
        return results

    # Extract embeddings
    embeddings = []
    scores = []
    valid_indices = []

    for i, r in enumerate(results):
        if r.semantic_centroid is not None:
            embeddings.append(r.semantic_centroid)
            score = r.combined_score
            if isinstance(score, tuple):
                score = score[0]
            scores.append(float(score))
            valid_indices.append(i)

    if len(embeddings) < 2:
        # Not enough embeddings for density
        return results

    embeddings = np.array(embeddings)
    scores = np.array(scores)

    # Create density config from aggregation config
    density_config = DensityConfig(
        density_weight=config.density_weight,
        clustering_enabled=config.clustering_enabled,
        similarity_threshold=config.similarity_threshold,
        min_cluster_size=config.min_cluster_size
    )

    # Run two-stage pipeline
    flux_probs, densities, labels, centroids = two_stage_density_pipeline(
        embeddings, scores, density_config
    )

    # Update results with density information
    for j, i in enumerate(valid_indices):
        results[i].density_score = float(densities[j])
        results[i].cluster_id = int(labels[j]) if labels[j] >= 0 else None

        # Cluster confidence = density * cluster_size
        if labels[j] >= 0:
            cluster_size = int((labels == labels[j]).sum())
            results[i].cluster_confidence = float(densities[j] * cluster_size)
        else:
            results[i].cluster_confidence = 0.0

    return results


def density_flux_normalize(
    results: List[AggregatedResult],
    config: AggregationConfig
) -> List[Tuple[AggregatedResult, float]]:
    """
    Normalize results using flux-softmax (density-weighted probabilities).

    P(i) = exp(sᵢ) * (1 + w * dᵢ) / Z

    Args:
        results: Aggregated results with density scores
        config: Aggregation config

    Returns:
        List of (result, probability) tuples sorted by probability
    """
    if not DENSITY_AVAILABLE or not results:
        # Fallback to standard softmax
        total = sum(
            r.combined_score if not isinstance(r.combined_score, tuple)
            else r.combined_score[0]
            for r in results
        )
        return [
            (r, (r.combined_score if not isinstance(r.combined_score, tuple)
                 else r.combined_score[0]) / total if total > 0 else 0.0)
            for r in results
        ]

    scores = np.array([
        r.combined_score if not isinstance(r.combined_score, tuple)
        else r.combined_score[0]
        for r in results
    ])
    densities = np.array([r.density_score for r in results])

    probs = flux_softmax(scores, densities, config.density_weight)

    result_probs = list(zip(results, probs.tolist()))
    result_probs.sort(key=lambda x: x[1], reverse=True)

    return result_probs


class DensityAwareFederatedEngine(FederatedQueryEngine):
    """
    Extended federated query engine with density-based scoring.

    Implements Phase 4d density scoring:
    - Two-stage clustering pipeline
    - Flux-softmax normalization
    - Cluster confidence metrics
    """

    def federated_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        federation_k: Optional[int] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        return_embeddings: bool = True
    ) -> AggregatedResponse:
        """
        Execute federated query with density scoring.

        Args:
            query_text: Query text
            query_embedding: Query embedding vector
            top_k: Number of results
            federation_k: Number of nodes to query
            aggregation_strategy: Override strategy
            return_embeddings: Request embeddings from nodes for density

        Returns:
            AggregatedResponse with density-enhanced results
        """
        # Use parent implementation for basic query
        response = super().federated_query(
            query_text, query_embedding, top_k * 2,  # Get more for clustering
            federation_k, aggregation_strategy
        )

        # Apply density scoring if using DENSITY_FLUX strategy
        strategy = aggregation_strategy or self.config.strategy
        if strategy == AggregationStrategy.DENSITY_FLUX and DENSITY_AVAILABLE:
            # Convert dict results back to AggregatedResult for processing
            # This is a simplified version - full implementation would
            # preserve AggregatedResult objects through the pipeline
            pass

        return response

    def _normalize_and_rank(
        self,
        aggregated: Dict[str, AggregatedResult],
        total_partition: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Override to use flux-softmax when DENSITY_FLUX strategy is set."""
        results_list = list(aggregated.values())

        if self.config.strategy == AggregationStrategy.DENSITY_FLUX:
            # Apply density scoring
            results_list = apply_density_scoring(results_list, self.config)

            # Use flux-softmax normalization
            result_probs = density_flux_normalize(results_list, self.config)

            results = []
            for result, prob in result_probs[:top_k]:
                results.append(result.to_dict(normalized_prob=prob))

            return results

        # Fallback to parent implementation
        return super()._normalize_and_rank(aggregated, total_partition, top_k)


def create_density_aware_engine(
    router: KleinbergRouter,
    federation_k: int = 3,
    timeout_ms: int = 5000,
    density_weight: float = 0.3,
    clustering_enabled: bool = True,
    similarity_threshold: float = 0.7,
    consensus_threshold: Optional[int] = None
) -> DensityAwareFederatedEngine:
    """Factory for density-aware federated engine."""
    config = AggregationConfig(
        strategy=AggregationStrategy.DENSITY_FLUX,
        consensus_threshold=consensus_threshold,
        density_weight=density_weight,
        clustering_enabled=clustering_enabled,
        similarity_threshold=similarity_threshold
    )

    return DensityAwareFederatedEngine(
        router=router,
        aggregation_config=config,
        federation_k=federation_k,
        timeout_ms=timeout_ms
    )


# =============================================================================
# PHASE 5b: ADAPTIVE FEDERATION-K
# =============================================================================

@dataclass
class QueryMetrics:
    """Metrics for adaptive k selection.

    These metrics help determine how many nodes to query based on
    query characteristics and historical performance.
    """
    entropy: float              # Semantic diversity/ambiguity of query (0-1)
    top_similarity: float       # Max similarity to any node centroid (0-1)
    similarity_variance: float  # Variance in node similarities
    historical_consensus: float  # Avg consensus from similar queries (0-1)
    avg_node_latency_ms: float  # Expected response time per node


@dataclass
class AdaptiveKConfig:
    """Configuration for adaptive federation-k selection."""
    base_k: int = 3                   # Default number of nodes
    min_k: int = 1                    # Minimum nodes to query
    max_k: int = 10                   # Maximum nodes to query
    entropy_weight: float = 0.3       # Weight for entropy factor
    latency_weight: float = 0.2       # Weight for latency factor
    consensus_weight: float = 0.5     # Weight for consensus factor
    entropy_threshold: float = 0.7    # High entropy triggers more nodes
    similarity_threshold: float = 0.5  # Low similarity triggers more nodes
    consensus_threshold: float = 0.6  # Low consensus triggers more nodes
    history_size: int = 100           # Max queries to keep in history


class AdaptiveKCalculator:
    """Computes optimal federation_k based on query metrics.

    Uses multiple factors to dynamically adjust how many nodes to query:
    - High entropy (ambiguous query) → more nodes needed
    - Low top similarity (no strong match) → more nodes needed
    - Historical low consensus → more nodes needed
    - Tight latency budget → fewer nodes

    Implements a feedback loop: records query outcomes to improve future
    k selection for similar queries.
    """

    def __init__(self, config: Optional[AdaptiveKConfig] = None):
        """
        Initialize adaptive k calculator.

        Args:
            config: Configuration for k selection. Uses defaults if None.
        """
        self.config = config or AdaptiveKConfig()
        self.query_history: List[Tuple[np.ndarray, float, int]] = []  # (embedding, consensus, k_used)
        self._latency_cache: Dict[str, List[float]] = {}  # node_id -> latencies

    def compute_k(
        self,
        query_embedding: np.ndarray,
        nodes: List[KGNode],
        latency_budget_ms: Optional[int] = None
    ) -> int:
        """
        Compute optimal federation_k based on query characteristics.

        Args:
            query_embedding: The query embedding vector
            nodes: Available KG nodes to query
            latency_budget_ms: Optional time budget for query

        Returns:
            Optimal number of nodes to query
        """
        if not nodes:
            return self.config.min_k

        metrics = self._compute_metrics(query_embedding, nodes)

        # Start with base k
        k = self.config.base_k

        # Adjust based on entropy (ambiguity)
        if metrics.entropy > self.config.entropy_threshold:
            k += int(2 * self.config.entropy_weight * 10)  # Up to +2 nodes

        # Adjust based on similarity distribution
        if metrics.top_similarity < self.config.similarity_threshold:
            k += 1  # No strong match, query more

        if metrics.similarity_variance > 0.1:
            k += 1  # High variance suggests need for exploration

        # Adjust based on historical consensus
        if metrics.historical_consensus < self.config.consensus_threshold:
            k += int(self.config.consensus_weight * 2)  # Past queries needed more nodes

        # Adjust based on latency budget
        if latency_budget_ms and metrics.avg_node_latency_ms > 0:
            max_nodes_in_budget = int(latency_budget_ms / metrics.avg_node_latency_ms)
            k = min(k, max(self.config.min_k, max_nodes_in_budget))

        # Clamp to valid range
        return max(self.config.min_k, min(k, self.config.max_k, len(nodes)))

    def _compute_metrics(
        self,
        query_embedding: np.ndarray,
        nodes: List[KGNode]
    ) -> QueryMetrics:
        """Compute metrics for k selection."""
        # Compute similarities to all nodes
        similarities = []
        for node in nodes:
            if node.centroid is not None:
                sim = self._cosine_similarity(query_embedding, node.centroid)
                similarities.append(sim)
            else:
                similarities.append(0.0)

        similarities = np.array(similarities)

        # Entropy: normalized entropy of similarity distribution
        # High entropy = query is ambiguous (similar to many topics)
        if len(similarities) > 1 and similarities.sum() > 0:
            probs = np.abs(similarities) / (np.abs(similarities).sum() + 1e-10)
            probs = probs + 1e-10  # Avoid log(0)
            entropy = -np.sum(probs * np.log(probs)) / np.log(len(probs))
        else:
            entropy = 0.5  # Default for single node

        # Top similarity
        top_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0

        # Variance
        variance = float(np.var(similarities)) if len(similarities) > 1 else 0.0

        # Historical consensus from similar queries
        historical_consensus = self._get_historical_consensus(query_embedding)

        # Average node latency
        avg_latency = self._get_avg_latency(nodes)

        return QueryMetrics(
            entropy=float(entropy),
            top_similarity=top_sim,
            similarity_variance=variance,
            historical_consensus=historical_consensus,
            avg_node_latency_ms=avg_latency
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _get_historical_consensus(self, query_embedding: np.ndarray) -> float:
        """Get average consensus from similar past queries."""
        if not self.query_history:
            return 0.8  # Optimistic default

        # Find similar queries in history
        similar_consensus = []
        for hist_emb, consensus, _ in self.query_history[-self.config.history_size:]:
            sim = self._cosine_similarity(query_embedding, hist_emb)
            if sim > 0.7:  # Similar query
                similar_consensus.append(consensus)

        if similar_consensus:
            return float(np.mean(similar_consensus))
        return 0.8  # Default if no similar queries

    def _get_avg_latency(self, nodes: List[KGNode]) -> float:
        """Get average latency for the given nodes."""
        latencies = []
        for node in nodes:
            if node.node_id in self._latency_cache:
                node_latencies = self._latency_cache[node.node_id]
                if node_latencies:
                    latencies.append(np.mean(node_latencies))

        if latencies:
            return float(np.mean(latencies))
        return 100.0  # Default 100ms if no data

    def record_query_outcome(
        self,
        query_embedding: np.ndarray,
        consensus_score: float,
        k_used: int,
        node_latencies: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record query outcome for future adaptive decisions.

        Args:
            query_embedding: The query embedding used
            consensus_score: Resulting consensus (0-1, higher = better)
            k_used: Number of nodes that were queried
            node_latencies: Optional dict of node_id -> latency_ms
        """
        # Add to history
        self.query_history.append((query_embedding.copy(), consensus_score, k_used))

        # Trim history if needed
        if len(self.query_history) > self.config.history_size:
            self.query_history = self.query_history[-self.config.history_size:]

        # Update latency cache
        if node_latencies:
            for node_id, latency in node_latencies.items():
                if node_id not in self._latency_cache:
                    self._latency_cache[node_id] = []
                self._latency_cache[node_id].append(latency)
                # Keep only recent latencies
                if len(self._latency_cache[node_id]) > 20:
                    self._latency_cache[node_id] = self._latency_cache[node_id][-20:]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive k selection."""
        if not self.query_history:
            return {
                'queries_recorded': 0,
                'avg_k_used': self.config.base_k,
                'avg_consensus': 0.0,
                'nodes_tracked': 0
            }

        k_values = [k for _, _, k in self.query_history]
        consensus_values = [c for _, c, _ in self.query_history]

        return {
            'queries_recorded': len(self.query_history),
            'avg_k_used': float(np.mean(k_values)),
            'avg_consensus': float(np.mean(consensus_values)),
            'nodes_tracked': len(self._latency_cache),
            'config': {
                'base_k': self.config.base_k,
                'min_k': self.config.min_k,
                'max_k': self.config.max_k
            }
        }


class AdaptiveFederatedEngine(FederatedQueryEngine):
    """Federated query engine with adaptive federation_k selection.

    Dynamically adjusts the number of nodes queried based on:
    - Query ambiguity (entropy of similarity distribution)
    - Historical query performance
    - Node latency characteristics
    - Optional latency budget constraints

    Includes a feedback loop to improve k selection over time.
    """

    def __init__(
        self,
        router: KleinbergRouter,
        aggregation_config: Optional[AggregationConfig] = None,
        adaptive_config: Optional[AdaptiveKConfig] = None,
        timeout_ms: int = 5000,
        max_workers: int = 10
    ):
        """
        Initialize adaptive federated engine.

        Args:
            router: KleinbergRouter for node discovery
            aggregation_config: Configuration for result aggregation
            adaptive_config: Configuration for adaptive k selection
            timeout_ms: Query timeout in milliseconds
            max_workers: Max parallel workers for queries
        """
        # Use base_k from adaptive config as default federation_k
        adaptive_cfg = adaptive_config or AdaptiveKConfig()
        super().__init__(
            router=router,
            aggregation_config=aggregation_config,
            federation_k=adaptive_cfg.base_k,
            timeout_ms=timeout_ms,
            max_workers=max_workers
        )
        self.adaptive = AdaptiveKCalculator(adaptive_cfg)

    def federated_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        federation_k: Optional[int] = None,
        latency_budget_ms: Optional[int] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None
    ) -> AggregatedResponse:
        """
        Execute a federated query with adaptive k selection.

        Args:
            query_text: The query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            federation_k: Override adaptive k (None = use adaptive)
            latency_budget_ms: Optional time budget for query
            aggregation_strategy: Override default aggregation strategy

        Returns:
            AggregatedResponse with merged results from all nodes
        """
        # Discover nodes
        nodes = self.router.discover_nodes()

        # Compute adaptive k if not overridden
        if federation_k is None:
            k = self.adaptive.compute_k(query_embedding, nodes, latency_budget_ms)
        else:
            k = federation_k

        # Execute query with computed k
        start_time = time.time()
        response = super().federated_query(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            federation_k=k,
            aggregation_strategy=aggregation_strategy
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Record outcome for learning
        # Compute consensus score from response
        consensus_score = self._compute_consensus_score(response)
        self.adaptive.record_query_outcome(
            query_embedding=query_embedding,
            consensus_score=consensus_score,
            k_used=k,
            node_latencies=self._get_recent_latencies()
        )

        return response

    def _compute_consensus_score(self, response: AggregatedResponse) -> float:
        """Compute consensus score from response."""
        if not response.results:
            return 0.0

        # Use diversity score if available
        if hasattr(response, 'diversity_score'):
            # Higher diversity = lower consensus (from different sources)
            # But for adaptive k, we want to measure result quality
            pass

        # Simple heuristic: ratio of top result score to total
        if len(response.results) >= 2:
            top_score = response.results[0].get('normalized_prob', 0.5)
            second_score = response.results[1].get('normalized_prob', 0.0)
            # High gap = high consensus on top result
            return min(1.0, top_score / (second_score + 0.1))

        return 0.5  # Default

    def _get_recent_latencies(self) -> Dict[str, float]:
        """Get recent node latencies from parent class stats."""
        latencies = {}
        for node_id, times in self._node_response_times.items():
            if times:
                latencies[node_id] = times[-1]  # Most recent
        return latencies

    def get_stats(self) -> Dict[str, Any]:
        """Get combined engine and adaptive k statistics."""
        base_stats = super().get_stats()
        adaptive_stats = self.adaptive.get_stats()
        return {
            **base_stats,
            'adaptive': adaptive_stats
        }


def create_adaptive_engine(
    router: KleinbergRouter,
    base_k: int = 3,
    min_k: int = 1,
    max_k: int = 10,
    entropy_weight: float = 0.3,
    latency_weight: float = 0.2,
    consensus_weight: float = 0.5,
    timeout_ms: int = 5000,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.SUM
) -> AdaptiveFederatedEngine:
    """Factory for adaptive federated engine.

    Args:
        router: KleinbergRouter for node discovery
        base_k: Default number of nodes to query
        min_k: Minimum nodes to query
        max_k: Maximum nodes to query
        entropy_weight: Weight for entropy factor in k computation
        latency_weight: Weight for latency factor in k computation
        consensus_weight: Weight for consensus factor in k computation
        timeout_ms: Query timeout in milliseconds
        aggregation_strategy: Default aggregation strategy

    Returns:
        AdaptiveFederatedEngine configured with given parameters
    """
    adaptive_config = AdaptiveKConfig(
        base_k=base_k,
        min_k=min_k,
        max_k=max_k,
        entropy_weight=entropy_weight,
        latency_weight=latency_weight,
        consensus_weight=consensus_weight
    )

    aggregation_config = AggregationConfig(
        strategy=aggregation_strategy
    )

    return AdaptiveFederatedEngine(
        router=router,
        aggregation_config=aggregation_config,
        adaptive_config=adaptive_config,
        timeout_ms=timeout_ms
    )


# =============================================================================
# PHASE 5a: HIERARCHICAL FEDERATION
# =============================================================================

@dataclass
class RegionalNode:
    """A node that aggregates results from child nodes.

    Regional nodes form a hierarchy where queries are first routed
    to regional aggregators, then drilled down to specialized nodes.
    """
    region_id: str
    centroid: np.ndarray           # Average centroid of child nodes
    topics: List[str]              # Combined topics from children
    child_nodes: List[str]         # Node IDs of children
    parent_region: Optional[str] = None
    level: int = 0                 # Hierarchy level (0 = top)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'region_id': self.region_id,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'topics': self.topics,
            'child_nodes': self.child_nodes,
            'parent_region': self.parent_region,
            'level': self.level
        }


@dataclass
class HierarchyConfig:
    """Configuration for node hierarchy."""
    max_levels: int = 3                    # Maximum hierarchy depth
    min_nodes_per_region: int = 2          # Minimum children per region
    max_nodes_per_region: int = 10         # Maximum children per region
    topic_similarity_threshold: float = 0.5  # Topic overlap for grouping
    centroid_similarity_threshold: float = 0.6  # Centroid similarity for grouping


class NodeHierarchy:
    """Manages hierarchical node relationships.

    Builds a tree structure from flat node lists based on:
    - Topic overlap (nodes with similar topics grouped together)
    - Centroid similarity (semantically close nodes grouped)

    The hierarchy enables efficient query routing:
    1. Query top-level regional nodes
    2. Drill down into best-matching region
    3. Query leaf nodes in that region
    """

    def __init__(self, config: Optional[HierarchyConfig] = None):
        """
        Initialize node hierarchy.

        Args:
            config: Hierarchy configuration
        """
        self.config = config or HierarchyConfig()
        self.regions: Dict[str, RegionalNode] = {}
        self.node_to_region: Dict[str, str] = {}  # node_id -> region_id
        self._leaf_nodes: Dict[str, KGNode] = {}  # Original nodes

    def build_from_nodes(self, nodes: List[KGNode]) -> None:
        """
        Build hierarchy from a list of KG nodes.

        Uses topic clustering first, then centroid similarity
        for nodes without clear topic matches.

        Args:
            nodes: List of KGNode to organize into hierarchy
        """
        if not nodes:
            return

        # Store leaf nodes
        self._leaf_nodes = {n.node_id: n for n in nodes}

        # Group by topic overlap
        topic_groups = self._group_by_topics(nodes)

        # Create regional nodes from topic groups
        for group_id, group_nodes in topic_groups.items():
            if len(group_nodes) >= self.config.min_nodes_per_region:
                self._create_region(group_id, group_nodes, level=0)

        # Handle ungrouped nodes by centroid similarity
        ungrouped = [n for n in nodes if n.node_id not in self.node_to_region]
        if ungrouped:
            self._group_by_centroid(ungrouped)

    def _group_by_topics(self, nodes: List[KGNode]) -> Dict[str, List[KGNode]]:
        """Group nodes by topic overlap."""
        groups: Dict[str, List[KGNode]] = {}

        for node in nodes:
            if not node.topics:
                continue

            # Find existing group with topic overlap
            best_group = None
            best_overlap = 0

            for group_id, group_nodes in groups.items():
                # Calculate topic overlap with group
                group_topics = set()
                for gn in group_nodes:
                    group_topics.update(gn.topics)

                overlap = len(set(node.topics) & group_topics)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_group = group_id

            # Add to best group or create new
            if best_group and best_overlap > 0:
                groups[best_group].append(node)
            else:
                # Create new group named after primary topic
                group_id = f"topic_{node.topics[0]}" if node.topics else f"group_{len(groups)}"
                groups[group_id] = [node]

        return groups

    def _group_by_centroid(self, nodes: List[KGNode]) -> None:
        """Group remaining nodes by centroid similarity."""
        if not nodes:
            return

        # Simple greedy clustering
        remaining = list(nodes)
        group_id = 0

        while remaining:
            # Start new cluster with first node
            seed = remaining.pop(0)
            cluster = [seed]

            # Add similar nodes
            i = 0
            while i < len(remaining):
                node = remaining[i]
                sim = self._cosine_similarity(seed.centroid, node.centroid)
                if sim >= self.config.centroid_similarity_threshold:
                    cluster.append(remaining.pop(i))
                else:
                    i += 1

                if len(cluster) >= self.config.max_nodes_per_region:
                    break

            # Create region if enough nodes
            if len(cluster) >= self.config.min_nodes_per_region:
                region_id = f"centroid_region_{group_id}"
                self._create_region(region_id, cluster, level=0)
                group_id += 1
            else:
                # Add to nearest existing region or create singleton region
                for node in cluster:
                    nearest = self._find_nearest_region(node)
                    if nearest:
                        self.regions[nearest].child_nodes.append(node.node_id)
                        self.node_to_region[node.node_id] = nearest
                    else:
                        # Create singleton region
                        region_id = f"singleton_{node.node_id}"
                        self._create_region(region_id, [node], level=0)

    def _create_region(
        self,
        region_id: str,
        nodes: List[KGNode],
        level: int,
        parent: Optional[str] = None
    ) -> RegionalNode:
        """Create a regional node from child nodes."""
        # Compute average centroid
        centroids = [n.centroid for n in nodes if n.centroid is not None]
        if centroids:
            avg_centroid = np.mean(centroids, axis=0)
        else:
            avg_centroid = np.zeros(384)  # Default dimension

        # Combine topics
        all_topics = []
        for n in nodes:
            all_topics.extend(n.topics)
        unique_topics = list(set(all_topics))

        region = RegionalNode(
            region_id=region_id,
            centroid=avg_centroid,
            topics=unique_topics,
            child_nodes=[n.node_id for n in nodes],
            parent_region=parent,
            level=level
        )

        self.regions[region_id] = region
        for node in nodes:
            self.node_to_region[node.node_id] = region_id

        return region

    def _find_nearest_region(self, node: KGNode) -> Optional[str]:
        """Find region with most similar centroid."""
        if not self.regions or node.centroid is None:
            return None

        best_region = None
        best_sim = -1

        for region_id, region in self.regions.items():
            sim = self._cosine_similarity(node.centroid, region.centroid)
            if sim > best_sim:
                best_sim = sim
                best_region = region_id

        return best_region if best_sim >= self.config.centroid_similarity_threshold else None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_regional_nodes(self, level: int = 0) -> List[RegionalNode]:
        """Get all regional nodes at a specific hierarchy level."""
        return [r for r in self.regions.values() if r.level == level]

    def get_children(self, region_id: str) -> List[str]:
        """Get child node IDs for a region."""
        if region_id in self.regions:
            return self.regions[region_id].child_nodes
        return []

    def get_child_nodes(self, region_id: str) -> List[KGNode]:
        """Get actual KGNode objects for a region's children."""
        child_ids = self.get_children(region_id)
        return [self._leaf_nodes[nid] for nid in child_ids if nid in self._leaf_nodes]

    def get_region_for_node(self, node_id: str) -> Optional[str]:
        """Get the region ID containing a node."""
        return self.node_to_region.get(node_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchy statistics."""
        if not self.regions:
            return {
                'num_regions': 0,
                'num_nodes': 0,
                'avg_nodes_per_region': 0.0,
                'levels': 0
            }

        sizes = [len(r.child_nodes) for r in self.regions.values()]
        levels = set(r.level for r in self.regions.values())

        return {
            'num_regions': len(self.regions),
            'num_nodes': len(self._leaf_nodes),
            'avg_nodes_per_region': float(np.mean(sizes)),
            'min_region_size': min(sizes),
            'max_region_size': max(sizes),
            'levels': max(levels) + 1 if levels else 0
        }


class HierarchicalFederatedEngine(FederatedQueryEngine):
    """Federated query engine with hierarchical query routing.

    Executes queries in multiple levels:
    1. Query regional aggregators at top level
    2. Select best-matching region(s)
    3. Query child nodes within selected region(s)
    4. Aggregate results from all levels

    This approach reduces network overhead for large federations
    by pruning unrelated regions early.
    """

    def __init__(
        self,
        router: KleinbergRouter,
        hierarchy: Optional[NodeHierarchy] = None,
        hierarchy_config: Optional[HierarchyConfig] = None,
        aggregation_config: Optional[AggregationConfig] = None,
        federation_k: int = 3,
        timeout_ms: int = 5000,
        max_workers: int = 10,
        drill_down_k: int = 2  # Number of regions to drill into
    ):
        """
        Initialize hierarchical federated engine.

        Args:
            router: KleinbergRouter for node discovery
            hierarchy: Pre-built hierarchy (built from nodes if None)
            hierarchy_config: Configuration for hierarchy building
            aggregation_config: Aggregation configuration
            federation_k: Nodes to query per level
            timeout_ms: Query timeout
            max_workers: Max parallel workers
            drill_down_k: Number of top regions to drill into
        """
        super().__init__(
            router=router,
            aggregation_config=aggregation_config,
            federation_k=federation_k,
            timeout_ms=timeout_ms,
            max_workers=max_workers
        )
        self.hierarchy = hierarchy
        self.hierarchy_config = hierarchy_config or HierarchyConfig()
        self.drill_down_k = drill_down_k
        self._hierarchy_built = hierarchy is not None

    def _ensure_hierarchy(self) -> None:
        """Build hierarchy from discovered nodes if not already built."""
        if self._hierarchy_built:
            return

        nodes = self.router.discover_nodes()
        if nodes:
            self.hierarchy = NodeHierarchy(self.hierarchy_config)
            self.hierarchy.build_from_nodes(nodes)
            self._hierarchy_built = True

    def federated_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        federation_k: Optional[int] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        use_hierarchy: bool = True
    ) -> AggregatedResponse:
        """
        Execute a hierarchical federated query.

        Args:
            query_text: The query text
            query_embedding: Query embedding vector
            top_k: Number of results to return
            federation_k: Override nodes per level
            aggregation_strategy: Override aggregation strategy
            use_hierarchy: If False, bypass hierarchy and query flat

        Returns:
            AggregatedResponse with merged results
        """
        if not use_hierarchy:
            return super().federated_query(
                query_text, query_embedding, top_k,
                federation_k, aggregation_strategy
            )

        self._ensure_hierarchy()

        if not self.hierarchy or not self.hierarchy.regions:
            # No hierarchy available, fall back to flat query
            return super().federated_query(
                query_text, query_embedding, top_k,
                federation_k, aggregation_strategy
            )

        start_time = time.time()
        query_id = str(uuid.uuid4())
        k = federation_k or self.federation_k

        # Level 1: Query regional nodes
        regions = self.hierarchy.get_regional_nodes(level=0)
        if not regions:
            return super().federated_query(
                query_text, query_embedding, top_k,
                federation_k, aggregation_strategy
            )

        # Rank regions by similarity to query
        ranked_regions = self._rank_regions(query_embedding, regions)

        # Select top regions to drill into
        selected_regions = ranked_regions[:self.drill_down_k]

        # Level 2: Query child nodes in selected regions
        all_responses = []
        for region, sim in selected_regions:
            child_nodes = self.hierarchy.get_child_nodes(region.region_id)
            if child_nodes:
                # Query children using parent class
                response = self._query_region_children(
                    query_text, query_embedding, child_nodes,
                    k, aggregation_strategy, query_id
                )
                all_responses.append((response, sim))

        # Aggregate across regions
        final_response = self._aggregate_hierarchical(
            all_responses, query_id, top_k, aggregation_strategy
        )

        elapsed_ms = (time.time() - start_time) * 1000
        final_response.total_time_ms = elapsed_ms

        return final_response

    def _rank_regions(
        self,
        query_embedding: np.ndarray,
        regions: List[RegionalNode]
    ) -> List[Tuple[RegionalNode, float]]:
        """Rank regions by similarity to query."""
        ranked = []
        for region in regions:
            sim = self._cosine_similarity(query_embedding, region.centroid)
            ranked.append((region, sim))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _query_region_children(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        child_nodes: List[KGNode],
        k: int,
        aggregation_strategy: Optional[AggregationStrategy],
        query_id: str
    ) -> AggregatedResponse:
        """Query child nodes within a region."""
        # Use parent's parallel query mechanism
        responses = self._parallel_query(
            child_nodes[:k], query_text, query_embedding, query_id
        )

        # Aggregate responses
        strategy = aggregation_strategy or self.config.strategy
        aggregator = get_aggregator(strategy)

        aggregated = self._aggregate(responses, aggregator)
        total_partition = sum(r.partition_sum for r in responses)

        return AggregatedResponse(
            query_id=query_id,
            results=self._normalize_and_rank(aggregated, total_partition, k * 2),
            total_partition_sum=total_partition,
            nodes_queried=len(child_nodes[:k]),
            nodes_responded=len(responses),
            total_time_ms=0.0,
            aggregation_strategy=strategy.value
        )

    def _aggregate_hierarchical(
        self,
        region_responses: List[Tuple[AggregatedResponse, float]],
        query_id: str,
        top_k: int,
        aggregation_strategy: Optional[AggregationStrategy]
    ) -> AggregatedResponse:
        """Aggregate results from multiple regions."""
        if not region_responses:
            return AggregatedResponse(
                query_id=query_id,
                results=[],
                total_partition_sum=0.0,
                nodes_queried=0,
                nodes_responded=0,
                total_time_ms=0.0,
                aggregation_strategy=(aggregation_strategy or self.config.strategy).value
            )

        # Merge results from all regions
        # Weight by region similarity
        all_results = {}
        total_partition = 0.0
        total_nodes_queried = 0
        total_nodes_responded = 0

        for response, region_sim in region_responses:
            total_partition += response.total_partition_sum
            total_nodes_queried += response.nodes_queried
            total_nodes_responded += response.nodes_responded

            for result in response.results:
                key = result.get('answer_hash', result.get('answer_id', str(result)))
                if key in all_results:
                    # Merge: boost by region similarity
                    existing = all_results[key]
                    existing_prob = existing.get('normalized_prob', 0.0)
                    new_prob = result.get('normalized_prob', 0.0) * region_sim
                    existing['normalized_prob'] = existing_prob + new_prob
                else:
                    # New result: scale by region similarity
                    result_copy = dict(result)
                    result_copy['normalized_prob'] = result.get('normalized_prob', 0.0) * region_sim
                    result_copy['source_region'] = region_sim
                    all_results[key] = result_copy

        # Sort and take top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda r: r.get('normalized_prob', 0.0),
            reverse=True
        )[:top_k]

        return AggregatedResponse(
            query_id=query_id,
            results=sorted_results,
            total_partition_sum=total_partition,
            nodes_queried=total_nodes_queried,
            nodes_responded=total_nodes_responded,
            total_time_ms=0.0,
            aggregation_strategy=(aggregation_strategy or self.config.strategy).value
        )

    def rebuild_hierarchy(self) -> None:
        """Force rebuild of hierarchy from current nodes."""
        self._hierarchy_built = False
        self._ensure_hierarchy()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined engine and hierarchy statistics."""
        base_stats = super().get_stats()
        hierarchy_stats = self.hierarchy.get_stats() if self.hierarchy else {}
        return {
            **base_stats,
            'hierarchy': hierarchy_stats,
            'drill_down_k': self.drill_down_k
        }


def create_hierarchical_engine(
    router: KleinbergRouter,
    max_levels: int = 3,
    min_nodes_per_region: int = 2,
    max_nodes_per_region: int = 10,
    drill_down_k: int = 2,
    federation_k: int = 3,
    timeout_ms: int = 5000,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.SUM
) -> HierarchicalFederatedEngine:
    """Factory for hierarchical federated engine.

    Args:
        router: KleinbergRouter for node discovery
        max_levels: Maximum hierarchy depth
        min_nodes_per_region: Minimum children per region
        max_nodes_per_region: Maximum children per region
        drill_down_k: Number of top regions to query in detail
        federation_k: Nodes to query per level
        timeout_ms: Query timeout
        aggregation_strategy: Default aggregation strategy

    Returns:
        HierarchicalFederatedEngine configured with given parameters
    """
    hierarchy_config = HierarchyConfig(
        max_levels=max_levels,
        min_nodes_per_region=min_nodes_per_region,
        max_nodes_per_region=max_nodes_per_region
    )

    aggregation_config = AggregationConfig(
        strategy=aggregation_strategy
    )

    return HierarchicalFederatedEngine(
        router=router,
        hierarchy_config=hierarchy_config,
        aggregation_config=aggregation_config,
        federation_k=federation_k,
        timeout_ms=timeout_ms,
        drill_down_k=drill_down_k
    )
