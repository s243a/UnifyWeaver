# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Query Plan Optimization for federated KG queries.

"""
Query Plan Optimization for KG Topology Phase 5c.

Builds optimized query plans based on query characteristics:
- SPECIFIC queries: High similarity to one node, use MAX aggregation
- EXPLORATORY queries: Low/varied similarity, broad parallel query
- CONSENSUS queries: Two-stage - broad query then density refinement

Key concepts:
- Query classification based on similarity distribution
- DAG-based execution plans with cost estimation
- Cost-based node selection and strategy choice

See: docs/proposals/ROADMAP_KG_TOPOLOGY.md (Phase 5)
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    from .federated_query import (
        FederatedQueryEngine,
        AggregationStrategy,
        AggregationConfig,
        AggregatedResponse,
        AggregatedResult
    )
    from .kleinberg_router import KleinbergRouter, KGNode
except ImportError:
    from federated_query import (
        FederatedQueryEngine,
        AggregationStrategy,
        AggregationConfig,
        AggregatedResponse,
        AggregatedResult
    )
    from kleinberg_router import KleinbergRouter, KGNode


# =============================================================================
# QUERY CLASSIFICATION
# =============================================================================

class QueryType(Enum):
    """Classification of query characteristics."""
    SPECIFIC = "specific"        # High similarity to one node, few nodes needed
    EXPLORATORY = "exploratory"  # Low/varied similarity, broad exploration
    CONSENSUS = "consensus"      # Medium similarity, density-focused aggregation


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    max_similarity: float
    similarity_variance: float
    top_nodes: List[str]  # Node IDs of best matches
    confidence: float     # Confidence in classification (0-1)


# =============================================================================
# QUERY PLAN DATA STRUCTURES
# =============================================================================

@dataclass
class NodeStats:
    """Statistics for a node used in cost estimation."""
    node_id: str
    avg_latency_ms: float = 100.0
    success_rate: float = 1.0
    avg_result_count: float = 10.0
    last_query_time: float = 0.0


@dataclass
class QueryPlanStage:
    """A stage in the query execution plan."""
    stage_id: int
    nodes: List[str]              # Node IDs to query
    strategy: AggregationStrategy
    parallel: bool = True
    depends_on: List[int] = field(default_factory=list)
    estimated_cost_ms: float = 0.0
    estimated_results: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'stage_id': self.stage_id,
            'nodes': self.nodes,
            'strategy': self.strategy.value,
            'parallel': self.parallel,
            'depends_on': self.depends_on,
            'estimated_cost_ms': self.estimated_cost_ms,
            'estimated_results': self.estimated_results,
            'description': self.description
        }


@dataclass
class QueryPlan:
    """DAG of query execution stages."""
    plan_id: str
    query_type: QueryType
    stages: List[QueryPlanStage]
    total_estimated_cost_ms: float
    created_at: float = field(default_factory=time.time)

    def get_execution_order(self) -> List[List[QueryPlanStage]]:
        """Return stages grouped by dependency level.

        Stages with no dependencies come first, then stages that
        depend only on completed stages, etc.
        """
        if not self.stages:
            return []

        # Build dependency graph
        completed = set()
        levels = []
        remaining = list(self.stages)

        while remaining:
            # Find stages whose dependencies are all completed
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.depends_on)
            ]

            if not ready:
                # Circular dependency or bug - just take remaining
                levels.append(remaining)
                break

            levels.append(ready)
            for stage in ready:
                completed.add(stage.stage_id)
                remaining.remove(stage)

        return levels

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'plan_id': self.plan_id,
            'query_type': self.query_type.value,
            'stages': [s.to_dict() for s in self.stages],
            'total_estimated_cost_ms': self.total_estimated_cost_ms,
            'created_at': self.created_at,
            'execution_order': [
                [s.stage_id for s in level]
                for level in self.get_execution_order()
            ]
        }


# =============================================================================
# QUERY PLANNER
# =============================================================================

@dataclass
class PlannerConfig:
    """Configuration for query planner."""
    # Classification thresholds
    specific_threshold: float = 0.8     # Max sim above this = SPECIFIC
    exploratory_variance: float = 0.1   # Variance above this = EXPLORATORY
    consensus_min_nodes: int = 3        # Min nodes for CONSENSUS queries

    # Plan building
    specific_max_nodes: int = 2         # Max nodes for SPECIFIC queries
    exploratory_max_nodes: int = 7      # Max nodes for EXPLORATORY
    consensus_stage1_nodes: int = 5     # Nodes for CONSENSUS stage 1
    consensus_stage2_nodes: int = 3     # Nodes for CONSENSUS stage 2

    # Cost estimation
    default_latency_ms: float = 100.0   # Default per-node latency
    parallel_overhead_ms: float = 10.0  # Overhead for parallel execution


class QueryPlanner:
    """Builds optimized query plans based on query characteristics.

    Analyzes query embeddings against node centroids to determine
    the best execution strategy:
    - SPECIFIC: Greedy approach, query 1-2 most similar nodes
    - EXPLORATORY: Broad parallel query across many nodes
    - CONSENSUS: Two-stage with initial broad query, then refinement
    """

    def __init__(
        self,
        router: KleinbergRouter,
        config: Optional[PlannerConfig] = None
    ):
        """
        Initialize query planner.

        Args:
            router: KleinbergRouter for node discovery
            config: Planner configuration
        """
        self.router = router
        self.config = config or PlannerConfig()
        self.node_stats: Dict[str, NodeStats] = {}
        self._plan_count = 0

    def classify_query(
        self,
        query_embedding: np.ndarray,
        nodes: List[KGNode]
    ) -> QueryClassification:
        """
        Classify query based on similarity distribution.

        Args:
            query_embedding: Query embedding vector
            nodes: Available nodes with centroids

        Returns:
            QueryClassification with type and metrics
        """
        if not nodes:
            return QueryClassification(
                query_type=QueryType.SPECIFIC,
                max_similarity=0.0,
                similarity_variance=0.0,
                top_nodes=[],
                confidence=0.0
            )

        # Compute similarities
        similarities = []
        for node in nodes:
            if node.centroid is not None:
                sim = self._cosine_similarity(query_embedding, node.centroid)
            else:
                sim = 0.0
            similarities.append((node.node_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        max_sim = similarities[0][1] if similarities else 0.0
        sims_array = np.array([s[1] for s in similarities])
        variance = float(np.var(sims_array)) if len(sims_array) > 1 else 0.0

        # Get top node IDs
        top_nodes = [node_id for node_id, _ in similarities[:5]]

        # Classify
        if max_sim >= self.config.specific_threshold:
            query_type = QueryType.SPECIFIC
            confidence = min(1.0, max_sim)
        elif variance >= self.config.exploratory_variance:
            query_type = QueryType.EXPLORATORY
            confidence = min(1.0, variance * 5)  # Scale variance to confidence
        else:
            query_type = QueryType.CONSENSUS
            confidence = 0.7  # Medium confidence for consensus

        return QueryClassification(
            query_type=query_type,
            max_similarity=max_sim,
            similarity_variance=variance,
            top_nodes=top_nodes,
            confidence=confidence
        )

    def build_plan(
        self,
        query_embedding: np.ndarray,
        nodes: Optional[List[KGNode]] = None,
        latency_budget_ms: Optional[float] = None,
        force_type: Optional[QueryType] = None
    ) -> QueryPlan:
        """
        Build execution plan based on query characteristics.

        Args:
            query_embedding: Query embedding vector
            nodes: Available nodes (discovered if None)
            latency_budget_ms: Optional latency constraint
            force_type: Force a specific query type (for testing)

        Returns:
            QueryPlan with optimized execution stages
        """
        if nodes is None:
            nodes = self.router.discover_nodes()

        if not nodes:
            return self._build_empty_plan()

        # Classify query
        classification = self.classify_query(query_embedding, nodes)
        query_type = force_type or classification.query_type

        # Build plan based on type
        if query_type == QueryType.SPECIFIC:
            plan = self._build_specific_plan(classification, nodes)
        elif query_type == QueryType.EXPLORATORY:
            plan = self._build_exploratory_plan(classification, nodes)
        else:
            plan = self._build_consensus_plan(classification, nodes)

        # Apply latency budget constraint if specified
        if latency_budget_ms:
            plan = self._apply_latency_budget(plan, latency_budget_ms)

        self._plan_count += 1
        return plan

    def _build_specific_plan(
        self,
        classification: QueryClassification,
        nodes: List[KGNode]
    ) -> QueryPlan:
        """Build plan for SPECIFIC query type.

        Strategy: Query top 1-2 nodes with highest similarity,
        use MAX aggregation (take best result).
        """
        # Select top nodes
        num_nodes = min(self.config.specific_max_nodes, len(classification.top_nodes))
        selected_nodes = classification.top_nodes[:num_nodes]

        # Single stage with MAX aggregation
        stage = QueryPlanStage(
            stage_id=0,
            nodes=selected_nodes,
            strategy=AggregationStrategy.MAX,
            parallel=True,
            estimated_cost_ms=self._estimate_stage_cost(selected_nodes),
            estimated_results=10,
            description="Greedy query to top matching nodes"
        )

        return QueryPlan(
            plan_id=self._generate_plan_id(),
            query_type=QueryType.SPECIFIC,
            stages=[stage],
            total_estimated_cost_ms=stage.estimated_cost_ms
        )

    def _build_exploratory_plan(
        self,
        classification: QueryClassification,
        nodes: List[KGNode]
    ) -> QueryPlan:
        """Build plan for EXPLORATORY query type.

        Strategy: Broad parallel query to many nodes,
        use SUM aggregation to boost consensus.
        """
        # Select more nodes for exploration
        num_nodes = min(self.config.exploratory_max_nodes, len(nodes))
        selected_nodes = classification.top_nodes[:num_nodes]

        # If not enough from classification, add more
        if len(selected_nodes) < num_nodes:
            remaining = [n.node_id for n in nodes if n.node_id not in selected_nodes]
            selected_nodes.extend(remaining[:num_nodes - len(selected_nodes)])

        # Single broad stage with SUM aggregation
        stage = QueryPlanStage(
            stage_id=0,
            nodes=selected_nodes,
            strategy=AggregationStrategy.SUM,
            parallel=True,
            estimated_cost_ms=self._estimate_stage_cost(selected_nodes),
            estimated_results=num_nodes * 5,
            description="Broad exploration across diverse nodes"
        )

        return QueryPlan(
            plan_id=self._generate_plan_id(),
            query_type=QueryType.EXPLORATORY,
            stages=[stage],
            total_estimated_cost_ms=stage.estimated_cost_ms
        )

    def _build_consensus_plan(
        self,
        classification: QueryClassification,
        nodes: List[KGNode]
    ) -> QueryPlan:
        """Build plan for CONSENSUS query type.

        Strategy: Two-stage execution:
        1. Broad query with SUM aggregation
        2. Density refinement on promising results
        """
        # Stage 1: Broad initial query
        stage1_nodes = min(self.config.consensus_stage1_nodes, len(nodes))
        stage1_node_ids = classification.top_nodes[:stage1_nodes]

        if len(stage1_node_ids) < stage1_nodes:
            remaining = [n.node_id for n in nodes if n.node_id not in stage1_node_ids]
            stage1_node_ids.extend(remaining[:stage1_nodes - len(stage1_node_ids)])

        stage1 = QueryPlanStage(
            stage_id=0,
            nodes=stage1_node_ids,
            strategy=AggregationStrategy.SUM,
            parallel=True,
            estimated_cost_ms=self._estimate_stage_cost(stage1_node_ids),
            estimated_results=stage1_nodes * 5,
            description="Initial broad query for consensus candidates"
        )

        # Stage 2: Density refinement (nodes determined at runtime)
        # We plan for additional nodes based on stage 1 results
        stage2_nodes = min(self.config.consensus_stage2_nodes, len(nodes))
        # Use remaining top nodes not in stage 1
        stage2_node_ids = [
            nid for nid in classification.top_nodes
            if nid not in stage1_node_ids
        ][:stage2_nodes]

        stage2 = QueryPlanStage(
            stage_id=1,
            nodes=stage2_node_ids,
            strategy=AggregationStrategy.DENSITY_FLUX,
            parallel=True,
            depends_on=[0],
            estimated_cost_ms=self._estimate_stage_cost(stage2_node_ids),
            estimated_results=stage2_nodes * 3,
            description="Density refinement on consensus clusters"
        )

        total_cost = stage1.estimated_cost_ms + stage2.estimated_cost_ms

        return QueryPlan(
            plan_id=self._generate_plan_id(),
            query_type=QueryType.CONSENSUS,
            stages=[stage1, stage2],
            total_estimated_cost_ms=total_cost
        )

    def _build_empty_plan(self) -> QueryPlan:
        """Build empty plan when no nodes available."""
        return QueryPlan(
            plan_id=self._generate_plan_id(),
            query_type=QueryType.SPECIFIC,
            stages=[],
            total_estimated_cost_ms=0.0
        )

    def _apply_latency_budget(
        self,
        plan: QueryPlan,
        budget_ms: float
    ) -> QueryPlan:
        """Apply latency budget constraint to plan.

        Reduces node count in stages if estimated cost exceeds budget.
        """
        if plan.total_estimated_cost_ms <= budget_ms:
            return plan

        # Reduce nodes in each stage proportionally
        ratio = budget_ms / plan.total_estimated_cost_ms
        new_stages = []

        for stage in plan.stages:
            new_node_count = max(1, int(len(stage.nodes) * ratio))
            new_stage = QueryPlanStage(
                stage_id=stage.stage_id,
                nodes=stage.nodes[:new_node_count],
                strategy=stage.strategy,
                parallel=stage.parallel,
                depends_on=stage.depends_on,
                estimated_cost_ms=self._estimate_stage_cost(stage.nodes[:new_node_count]),
                estimated_results=int(stage.estimated_results * ratio),
                description=stage.description + " (budget-constrained)"
            )
            new_stages.append(new_stage)

        total_cost = sum(s.estimated_cost_ms for s in new_stages)

        return QueryPlan(
            plan_id=plan.plan_id,
            query_type=plan.query_type,
            stages=new_stages,
            total_estimated_cost_ms=total_cost,
            created_at=plan.created_at
        )

    def _estimate_stage_cost(self, node_ids: List[str]) -> float:
        """Estimate execution cost for a stage."""
        if not node_ids:
            return 0.0

        # Get latencies for nodes
        latencies = []
        for node_id in node_ids:
            if node_id in self.node_stats:
                latencies.append(self.node_stats[node_id].avg_latency_ms)
            else:
                latencies.append(self.config.default_latency_ms)

        # Parallel execution: cost is max latency + overhead
        max_latency = max(latencies)
        return max_latency + self.config.parallel_overhead_ms

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        return f"plan_{self._plan_count}_{uuid.uuid4().hex[:8]}"

    def update_node_stats(
        self,
        node_id: str,
        latency_ms: float,
        success: bool = True,
        result_count: int = 0
    ) -> None:
        """Update statistics for a node after query execution."""
        if node_id not in self.node_stats:
            self.node_stats[node_id] = NodeStats(node_id=node_id)

        stats = self.node_stats[node_id]

        # Exponential moving average for latency
        alpha = 0.3
        stats.avg_latency_ms = alpha * latency_ms + (1 - alpha) * stats.avg_latency_ms

        # Update success rate
        if success:
            stats.success_rate = alpha * 1.0 + (1 - alpha) * stats.success_rate
        else:
            stats.success_rate = alpha * 0.0 + (1 - alpha) * stats.success_rate

        # Update result count average
        if result_count > 0:
            stats.avg_result_count = alpha * result_count + (1 - alpha) * stats.avg_result_count

        stats.last_query_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            'plans_created': self._plan_count,
            'nodes_tracked': len(self.node_stats),
            'node_stats': {
                node_id: {
                    'avg_latency_ms': stats.avg_latency_ms,
                    'success_rate': stats.success_rate,
                    'avg_result_count': stats.avg_result_count
                }
                for node_id, stats in self.node_stats.items()
            },
            'config': {
                'specific_threshold': self.config.specific_threshold,
                'exploratory_variance': self.config.exploratory_variance
            }
        }


# =============================================================================
# PLAN EXECUTOR
# =============================================================================

class PlanExecutor:
    """Executes query plans using a federated engine.

    Handles multi-stage execution with dependency ordering,
    parallel execution within stages, and result aggregation.
    """

    def __init__(
        self,
        engine: FederatedQueryEngine,
        planner: Optional[QueryPlanner] = None
    ):
        """
        Initialize plan executor.

        Args:
            engine: FederatedQueryEngine for executing queries
            planner: Optional QueryPlanner for stats feedback
        """
        self.engine = engine
        self.planner = planner
        self._executions = 0

    def execute(
        self,
        plan: QueryPlan,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> AggregatedResponse:
        """
        Execute a query plan.

        Args:
            plan: QueryPlan to execute
            query_text: Query text
            query_embedding: Query embedding
            top_k: Number of results to return

        Returns:
            AggregatedResponse with merged results
        """
        start_time = time.time()
        self._executions += 1

        if not plan.stages:
            return AggregatedResponse(
                query_id=f"exec_{self._executions}",
                results=[],
                total_partition_sum=0.0,
                nodes_queried=0,
                nodes_responded=0,
                total_time_ms=0.0,
                aggregation_strategy=AggregationStrategy.SUM.value
            )

        # Execute stages in dependency order
        stage_results: Dict[int, AggregatedResponse] = {}
        execution_order = plan.get_execution_order()

        for level_stages in execution_order:
            level_results = self._execute_level(
                level_stages, query_text, query_embedding, top_k, stage_results
            )
            stage_results.update(level_results)

        # Final aggregation across all stages
        final_response = self._aggregate_stages(
            stage_results, plan, query_text, query_embedding, top_k
        )

        elapsed_ms = (time.time() - start_time) * 1000
        final_response.query_time_ms = elapsed_ms

        return final_response

    def _execute_level(
        self,
        stages: List[QueryPlanStage],
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        previous_results: Dict[int, AggregatedResponse]
    ) -> Dict[int, AggregatedResponse]:
        """Execute all stages at a dependency level."""
        results = {}

        if len(stages) == 1 and not stages[0].parallel:
            # Sequential single stage
            stage = stages[0]
            response = self._execute_stage(
                stage, query_text, query_embedding, top_k, previous_results
            )
            results[stage.stage_id] = response
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=len(stages)) as executor:
                futures = {
                    executor.submit(
                        self._execute_stage,
                        stage, query_text, query_embedding, top_k, previous_results
                    ): stage
                    for stage in stages
                }

                for future in as_completed(futures):
                    stage = futures[future]
                    try:
                        response = future.result()
                        results[stage.stage_id] = response
                    except Exception as e:
                        # Log error and continue
                        results[stage.stage_id] = AggregatedResponse(
                            query_id=f"error_{stage.stage_id}",
                            results=[],
                            total_partition_sum=0.0,
                            nodes_queried=0,
                            nodes_responded=0,
                            total_time_ms=0.0,
                            aggregation_strategy=stage.strategy.value
                        )

        return results

    def _execute_stage(
        self,
        stage: QueryPlanStage,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        previous_results: Dict[int, AggregatedResponse]
    ) -> AggregatedResponse:
        """Execute a single stage."""
        start_time = time.time()

        if not stage.nodes:
            return AggregatedResponse(
                query_id=f"stage_{stage.stage_id}",
                results=[],
                total_partition_sum=0.0,
                nodes_queried=0,
                nodes_responded=0,
                total_time_ms=0.0,
                aggregation_strategy=stage.strategy.value
            )

        # Use engine to query the stage's nodes
        response = self.engine.federated_query(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            federation_k=len(stage.nodes),
            aggregation_strategy=stage.strategy
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Update planner stats if available
        if self.planner:
            for node_id in stage.nodes:
                self.planner.update_node_stats(
                    node_id=node_id,
                    latency_ms=elapsed_ms / len(stage.nodes),
                    success=len(response.results) > 0,
                    result_count=len(response.results)
                )

        return response

    def _aggregate_stages(
        self,
        stage_results: Dict[int, AggregatedResponse],
        plan: QueryPlan,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int
    ) -> AggregatedResponse:
        """Aggregate results from all stages."""
        if not stage_results:
            return AggregatedResponse(
                query_id=f"exec_{self._executions}",
                results=[],
                total_partition_sum=0.0,
                nodes_queried=0,
                nodes_responded=0,
                total_time_ms=0.0,
                aggregation_strategy=AggregationStrategy.SUM.value
            )

        # For multi-stage plans, merge results
        # Use last stage's results as primary (it has refinement)
        if len(stage_results) == 1:
            return list(stage_results.values())[0]

        # Merge all results, preferring later stages
        all_results = {}
        total_partition = 0.0
        nodes_queried = 0

        for stage_id in sorted(stage_results.keys()):
            response = stage_results[stage_id]
            total_partition += response.total_partition_sum
            nodes_queried += response.nodes_queried

            for result in response.results:
                key = result.get('answer_hash', result.get('answer_id', str(result)))
                # Later stages override earlier (refinement)
                all_results[key] = result

        # Sort by probability and take top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda r: r.get('normalized_prob', 0.0),
            reverse=True
        )[:top_k]

        return AggregatedResponse(
            query_id=f"exec_{self._executions}",
            results=sorted_results,
            total_partition_sum=total_partition,
            nodes_queried=nodes_queried,
            nodes_responded=nodes_queried,  # Assume all queried nodes responded
            total_time_ms=0.0,  # Will be set by caller
            aggregation_strategy=plan.query_type.value
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            'executions': self._executions
        }


# =============================================================================
# PLANNED QUERY ENGINE
# =============================================================================

class PlannedQueryEngine:
    """Query engine that uses query planning for optimization.

    Combines QueryPlanner and PlanExecutor for automatic
    query classification and optimized execution.
    """

    def __init__(
        self,
        router: KleinbergRouter,
        aggregation_config: Optional[AggregationConfig] = None,
        planner_config: Optional[PlannerConfig] = None,
        federation_k: int = 3,
        timeout_ms: int = 5000
    ):
        """
        Initialize planned query engine.

        Args:
            router: KleinbergRouter for node discovery
            aggregation_config: Aggregation configuration
            planner_config: Query planner configuration
            federation_k: Default federation k (used in stages)
            timeout_ms: Query timeout
        """
        self.router = router
        self.planner = QueryPlanner(router, planner_config)

        # Create underlying engine
        self.engine = FederatedQueryEngine(
            router=router,
            aggregation_config=aggregation_config,
            federation_k=federation_k,
            timeout_ms=timeout_ms
        )

        self.executor = PlanExecutor(self.engine, self.planner)

    def query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        latency_budget_ms: Optional[float] = None,
        force_type: Optional[QueryType] = None
    ) -> Tuple[AggregatedResponse, QueryPlan]:
        """
        Execute a query with automatic planning.

        Args:
            query_text: Query text
            query_embedding: Query embedding
            top_k: Number of results
            latency_budget_ms: Optional latency constraint
            force_type: Force a specific query type

        Returns:
            Tuple of (response, plan used)
        """
        # Discover nodes
        nodes = self.router.discover_nodes()

        # Build plan
        plan = self.planner.build_plan(
            query_embedding=query_embedding,
            nodes=nodes,
            latency_budget_ms=latency_budget_ms,
            force_type=force_type
        )

        # Execute plan
        response = self.executor.execute(
            plan=plan,
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k
        )

        return response, plan

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            'planner': self.planner.get_stats(),
            'executor': self.executor.get_stats(),
            'engine': self.engine.get_stats()
        }


def create_planned_engine(
    router: KleinbergRouter,
    specific_threshold: float = 0.8,
    exploratory_variance: float = 0.1,
    federation_k: int = 3,
    timeout_ms: int = 5000,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.SUM
) -> PlannedQueryEngine:
    """Factory for planned query engine.

    Args:
        router: KleinbergRouter for node discovery
        specific_threshold: Similarity threshold for SPECIFIC queries
        exploratory_variance: Variance threshold for EXPLORATORY
        federation_k: Default federation k
        timeout_ms: Query timeout
        aggregation_strategy: Default aggregation strategy

    Returns:
        PlannedQueryEngine configured with given parameters
    """
    planner_config = PlannerConfig(
        specific_threshold=specific_threshold,
        exploratory_variance=exploratory_variance
    )

    aggregation_config = AggregationConfig(
        strategy=aggregation_strategy
    )

    return PlannedQueryEngine(
        router=router,
        aggregation_config=aggregation_config,
        planner_config=planner_config,
        federation_k=federation_k,
        timeout_ms=timeout_ms
    )
