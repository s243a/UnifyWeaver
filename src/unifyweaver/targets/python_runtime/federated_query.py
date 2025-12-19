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


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    strategy: AggregationStrategy = AggregationStrategy.SUM
    dedup_key: str = "answer_hash"  # Field to group by
    consensus_threshold: Optional[int] = None  # Minimum node agreement
    diversity_field: str = "corpus_id"  # Field for diversity checking


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'answer_id': self.answer_id,
            'answer_text': self.answer_text,
            'answer_hash': self.answer_hash,
            'raw_score': self.raw_score,
            'exp_score': self.exp_score,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NodeResult':
        return cls(
            answer_id=d['answer_id'],
            answer_text=d['answer_text'],
            answer_hash=d['answer_hash'],
            raw_score=d['raw_score'],
            exp_score=d['exp_score'],
            metadata=d.get('metadata', {})
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'exp_score': self.exp_score,
            'corpus_id': self.corpus_id,
            'data_sources': self.data_sources,
            'interface_id': self.interface_id,
            'embedding_model': self.embedding_model
        }


@dataclass
class AggregatedResult:
    """A result aggregated from multiple nodes."""
    answer_text: str
    answer_hash: str
    combined_score: Any  # Type depends on aggregator
    source_nodes: List[str] = field(default_factory=list)
    provenance: List[ResultProvenance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
                embedding_model=node_metadata.get('embedding_model')
            )],
            metadata=result.metadata.copy()
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
            'metadata': self.metadata
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
    diversity_field: str = "corpus_id"
) -> FederatedQueryEngine:
    """Factory function to create a FederatedQueryEngine."""
    strategy_enum = AggregationStrategy(strategy)

    config = AggregationConfig(
        strategy=strategy_enum,
        consensus_threshold=consensus_threshold,
        diversity_field=diversity_field
    )

    return FederatedQueryEngine(
        router=router,
        aggregation_config=config,
        federation_k=federation_k,
        timeout_ms=timeout_ms
    )
