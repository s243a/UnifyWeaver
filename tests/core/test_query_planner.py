# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)

"""Unit tests for Phase 5c: Query Plan Optimization."""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

try:
    from query_planner import (
        QueryType,
        QueryClassification,
        NodeStats,
        QueryPlanStage,
        QueryPlan,
        PlannerConfig,
        QueryPlanner,
        PlanExecutor,
        PlannedQueryEngine,
        create_planned_engine
    )
    from federated_query import AggregationStrategy, AggregatedResponse
    from kleinberg_router import KleinbergRouter, KGNode
except ImportError:
    from unifyweaver.targets.python_runtime.query_planner import (
        QueryType,
        QueryClassification,
        NodeStats,
        QueryPlanStage,
        QueryPlan,
        PlannerConfig,
        QueryPlanner,
        PlanExecutor,
        PlannedQueryEngine,
        create_planned_engine
    )
    from unifyweaver.targets.python_runtime.federated_query import (
        AggregationStrategy, AggregatedResponse
    )
    from unifyweaver.targets.python_runtime.kleinberg_router import KleinbergRouter, KGNode


class TestQueryType(unittest.TestCase):
    """Tests for QueryType enum."""

    def test_query_types(self):
        """Test QueryType values."""
        self.assertEqual(QueryType.SPECIFIC.value, "specific")
        self.assertEqual(QueryType.EXPLORATORY.value, "exploratory")
        self.assertEqual(QueryType.CONSENSUS.value, "consensus")


class TestQueryClassification(unittest.TestCase):
    """Tests for QueryClassification dataclass."""

    def test_classification_creation(self):
        """Test creating QueryClassification."""
        classification = QueryClassification(
            query_type=QueryType.SPECIFIC,
            max_similarity=0.9,
            similarity_variance=0.05,
            top_nodes=["node_1", "node_2"],
            confidence=0.85
        )
        self.assertEqual(classification.query_type, QueryType.SPECIFIC)
        self.assertEqual(classification.max_similarity, 0.9)
        self.assertEqual(len(classification.top_nodes), 2)


class TestNodeStats(unittest.TestCase):
    """Tests for NodeStats dataclass."""

    def test_default_values(self):
        """Test default NodeStats values."""
        stats = NodeStats(node_id="test_node")
        self.assertEqual(stats.node_id, "test_node")
        self.assertEqual(stats.avg_latency_ms, 100.0)
        self.assertEqual(stats.success_rate, 1.0)


class TestQueryPlanStage(unittest.TestCase):
    """Tests for QueryPlanStage dataclass."""

    def test_stage_creation(self):
        """Test creating QueryPlanStage."""
        stage = QueryPlanStage(
            stage_id=0,
            nodes=["node_1", "node_2"],
            strategy=AggregationStrategy.SUM,
            parallel=True
        )
        self.assertEqual(stage.stage_id, 0)
        self.assertEqual(len(stage.nodes), 2)
        self.assertEqual(stage.strategy, AggregationStrategy.SUM)

    def test_stage_to_dict(self):
        """Test stage serialization."""
        stage = QueryPlanStage(
            stage_id=1,
            nodes=["node_a"],
            strategy=AggregationStrategy.MAX,
            depends_on=[0],
            description="Test stage"
        )
        d = stage.to_dict()
        self.assertEqual(d['stage_id'], 1)
        self.assertEqual(d['strategy'], "max")
        self.assertEqual(d['depends_on'], [0])


class TestQueryPlan(unittest.TestCase):
    """Tests for QueryPlan dataclass."""

    def test_plan_creation(self):
        """Test creating QueryPlan."""
        stage = QueryPlanStage(
            stage_id=0,
            nodes=["node_1"],
            strategy=AggregationStrategy.SUM
        )
        plan = QueryPlan(
            plan_id="test_plan",
            query_type=QueryType.SPECIFIC,
            stages=[stage],
            total_estimated_cost_ms=100.0
        )
        self.assertEqual(plan.plan_id, "test_plan")
        self.assertEqual(len(plan.stages), 1)

    def test_execution_order_single_stage(self):
        """Test execution order with single stage."""
        stage = QueryPlanStage(stage_id=0, nodes=["n1"], strategy=AggregationStrategy.SUM)
        plan = QueryPlan(
            plan_id="p1",
            query_type=QueryType.SPECIFIC,
            stages=[stage],
            total_estimated_cost_ms=50.0
        )
        order = plan.get_execution_order()
        self.assertEqual(len(order), 1)
        self.assertEqual(len(order[0]), 1)

    def test_execution_order_multi_stage(self):
        """Test execution order with dependencies."""
        stage1 = QueryPlanStage(stage_id=0, nodes=["n1"], strategy=AggregationStrategy.SUM)
        stage2 = QueryPlanStage(
            stage_id=1, nodes=["n2"], strategy=AggregationStrategy.MAX, depends_on=[0]
        )
        stage3 = QueryPlanStage(
            stage_id=2, nodes=["n3"], strategy=AggregationStrategy.DENSITY_FLUX, depends_on=[1]
        )

        plan = QueryPlan(
            plan_id="p2",
            query_type=QueryType.CONSENSUS,
            stages=[stage1, stage2, stage3],
            total_estimated_cost_ms=150.0
        )
        order = plan.get_execution_order()
        self.assertEqual(len(order), 3)
        self.assertEqual(order[0][0].stage_id, 0)
        self.assertEqual(order[1][0].stage_id, 1)
        self.assertEqual(order[2][0].stage_id, 2)

    def test_execution_order_parallel_stages(self):
        """Test execution order with parallel stages."""
        stage1 = QueryPlanStage(stage_id=0, nodes=["n1"], strategy=AggregationStrategy.SUM)
        stage2 = QueryPlanStage(stage_id=1, nodes=["n2"], strategy=AggregationStrategy.SUM)
        # Both depend on nothing - should run in parallel

        plan = QueryPlan(
            plan_id="p3",
            query_type=QueryType.EXPLORATORY,
            stages=[stage1, stage2],
            total_estimated_cost_ms=100.0
        )
        order = plan.get_execution_order()
        self.assertEqual(len(order), 1)  # Both in first level
        self.assertEqual(len(order[0]), 2)

    def test_plan_to_dict(self):
        """Test plan serialization."""
        stage = QueryPlanStage(stage_id=0, nodes=["n1"], strategy=AggregationStrategy.SUM)
        plan = QueryPlan(
            plan_id="p4",
            query_type=QueryType.CONSENSUS,
            stages=[stage],
            total_estimated_cost_ms=75.0
        )
        d = plan.to_dict()
        self.assertEqual(d['plan_id'], "p4")
        self.assertEqual(d['query_type'], "consensus")
        self.assertIn('execution_order', d)


class TestPlannerConfig(unittest.TestCase):
    """Tests for PlannerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = PlannerConfig()
        self.assertEqual(config.specific_threshold, 0.8)
        self.assertEqual(config.exploratory_variance, 0.1)
        self.assertEqual(config.specific_max_nodes, 2)

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlannerConfig(
            specific_threshold=0.9,
            exploratory_max_nodes=10
        )
        self.assertEqual(config.specific_threshold, 0.9)
        self.assertEqual(config.exploratory_max_nodes, 10)


class TestQueryPlanner(unittest.TestCase):
    """Tests for QueryPlanner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_router = Mock(spec=KleinbergRouter)
        np.random.seed(42)

        # Create nodes with different centroids
        self.nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.random.randn(384),
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]
        self.mock_router.discover_nodes.return_value = self.nodes

        self.planner = QueryPlanner(self.mock_router)

    def test_classify_specific(self):
        """Test SPECIFIC query classification."""
        # Create query very similar to node_0
        query = self.nodes[0].centroid.copy()
        classification = self.planner.classify_query(query, self.nodes)
        self.assertEqual(classification.query_type, QueryType.SPECIFIC)
        self.assertGreater(classification.max_similarity, 0.8)

    def test_classify_exploratory(self):
        """Test EXPLORATORY query classification."""
        # Create diverse nodes for high variance
        diverse_nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.eye(384)[i] if i < 384 else np.random.randn(384),
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]

        query = np.random.randn(384)
        query = query / np.linalg.norm(query)

        classification = self.planner.classify_query(query, diverse_nodes)
        # With very different node centroids, variance should be high
        self.assertIn(classification.query_type, [QueryType.EXPLORATORY, QueryType.CONSENSUS])

    def test_classify_empty_nodes(self):
        """Test classification with empty nodes."""
        query = np.random.randn(384)
        classification = self.planner.classify_query(query, [])
        self.assertEqual(classification.query_type, QueryType.SPECIFIC)
        self.assertEqual(classification.confidence, 0.0)

    def test_build_specific_plan(self):
        """Test building SPECIFIC plan."""
        query = self.nodes[0].centroid.copy()
        plan = self.planner.build_plan(query, self.nodes, force_type=QueryType.SPECIFIC)

        self.assertEqual(plan.query_type, QueryType.SPECIFIC)
        self.assertEqual(len(plan.stages), 1)
        self.assertEqual(plan.stages[0].strategy, AggregationStrategy.MAX)
        self.assertLessEqual(len(plan.stages[0].nodes), 2)

    def test_build_exploratory_plan(self):
        """Test building EXPLORATORY plan."""
        query = np.random.randn(384)
        plan = self.planner.build_plan(query, self.nodes, force_type=QueryType.EXPLORATORY)

        self.assertEqual(plan.query_type, QueryType.EXPLORATORY)
        self.assertEqual(len(plan.stages), 1)
        self.assertEqual(plan.stages[0].strategy, AggregationStrategy.SUM)
        self.assertGreater(len(plan.stages[0].nodes), 2)

    def test_build_consensus_plan(self):
        """Test building CONSENSUS plan."""
        query = np.random.randn(384)
        plan = self.planner.build_plan(query, self.nodes, force_type=QueryType.CONSENSUS)

        self.assertEqual(plan.query_type, QueryType.CONSENSUS)
        self.assertEqual(len(plan.stages), 2)
        self.assertEqual(plan.stages[0].strategy, AggregationStrategy.SUM)
        self.assertEqual(plan.stages[1].strategy, AggregationStrategy.DENSITY_FLUX)
        self.assertIn(0, plan.stages[1].depends_on)

    def test_build_empty_plan(self):
        """Test building plan with no nodes."""
        self.mock_router.discover_nodes.return_value = []
        query = np.random.randn(384)
        plan = self.planner.build_plan(query)

        self.assertEqual(len(plan.stages), 0)
        self.assertEqual(plan.total_estimated_cost_ms, 0.0)

    def test_latency_budget_constraint(self):
        """Test latency budget reduces nodes."""
        query = np.random.randn(384)

        # Build plan without budget
        plan_no_budget = self.planner.build_plan(
            query, self.nodes, force_type=QueryType.EXPLORATORY
        )

        # Build plan with tight budget
        plan_with_budget = self.planner.build_plan(
            query, self.nodes,
            latency_budget_ms=50.0,
            force_type=QueryType.EXPLORATORY
        )

        # Budget-constrained should have fewer nodes
        self.assertLessEqual(
            len(plan_with_budget.stages[0].nodes),
            len(plan_no_budget.stages[0].nodes)
        )

    def test_update_node_stats(self):
        """Test updating node statistics."""
        self.planner.update_node_stats(
            node_id="node_0",
            latency_ms=50.0,
            success=True,
            result_count=10
        )

        self.assertIn("node_0", self.planner.node_stats)
        stats = self.planner.node_stats["node_0"]
        self.assertLess(stats.avg_latency_ms, 100.0)  # Should decrease from default

    def test_get_stats(self):
        """Test getting planner statistics."""
        # Build a few plans
        query = np.random.randn(384)
        self.planner.build_plan(query, self.nodes)
        self.planner.build_plan(query, self.nodes)

        stats = self.planner.get_stats()
        self.assertEqual(stats['plans_created'], 2)
        self.assertIn('config', stats)


class TestPlanExecutor(unittest.TestCase):
    """Tests for PlanExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_engine.federated_query.return_value = AggregatedResponse(
            query_id="test",
            results=[{'answer_id': '1', 'normalized_prob': 0.8}],
            total_partition_sum=1.0,
            nodes_queried=2,
            nodes_responded=2,
            total_time_ms=50.0,
            aggregation_strategy="sum"
        )

        self.executor = PlanExecutor(self.mock_engine)

    def test_execute_empty_plan(self):
        """Test executing empty plan."""
        plan = QueryPlan(
            plan_id="empty",
            query_type=QueryType.SPECIFIC,
            stages=[],
            total_estimated_cost_ms=0.0
        )

        response = self.executor.execute(
            plan=plan,
            query_text="test",
            query_embedding=np.random.randn(384),
            top_k=10
        )

        self.assertEqual(len(response.results), 0)

    def test_execute_single_stage(self):
        """Test executing single-stage plan."""
        stage = QueryPlanStage(
            stage_id=0,
            nodes=["node_1", "node_2"],
            strategy=AggregationStrategy.SUM
        )
        plan = QueryPlan(
            plan_id="single",
            query_type=QueryType.SPECIFIC,
            stages=[stage],
            total_estimated_cost_ms=100.0
        )

        response = self.executor.execute(
            plan=plan,
            query_text="test query",
            query_embedding=np.random.randn(384),
            top_k=10
        )

        self.mock_engine.federated_query.assert_called_once()
        self.assertGreater(len(response.results), 0)

    def test_execute_multi_stage(self):
        """Test executing multi-stage plan."""
        stage1 = QueryPlanStage(
            stage_id=0,
            nodes=["node_1"],
            strategy=AggregationStrategy.SUM
        )
        stage2 = QueryPlanStage(
            stage_id=1,
            nodes=["node_2"],
            strategy=AggregationStrategy.DENSITY_FLUX,
            depends_on=[0]
        )
        plan = QueryPlan(
            plan_id="multi",
            query_type=QueryType.CONSENSUS,
            stages=[stage1, stage2],
            total_estimated_cost_ms=150.0
        )

        response = self.executor.execute(
            plan=plan,
            query_text="test query",
            query_embedding=np.random.randn(384),
            top_k=10
        )

        # Should call engine twice (once per stage)
        self.assertEqual(self.mock_engine.federated_query.call_count, 2)

    def test_get_stats(self):
        """Test executor statistics."""
        stats = self.executor.get_stats()
        self.assertEqual(stats['executions'], 0)

        # Execute something
        stage = QueryPlanStage(stage_id=0, nodes=["n1"], strategy=AggregationStrategy.SUM)
        plan = QueryPlan(
            plan_id="test",
            query_type=QueryType.SPECIFIC,
            stages=[stage],
            total_estimated_cost_ms=50.0
        )
        self.executor.execute(plan, "test", np.random.randn(384), 10)

        stats = self.executor.get_stats()
        self.assertEqual(stats['executions'], 1)


class TestPlannedQueryEngine(unittest.TestCase):
    """Tests for PlannedQueryEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_router = Mock(spec=KleinbergRouter)
        np.random.seed(42)

        self.nodes = [
            KGNode(
                node_id=f"node_{i}",
                endpoint=f"http://node{i}:8080",
                centroid=np.random.randn(384),
                topics=[f"topic_{i}"],
                embedding_model="test-model"
            )
            for i in range(5)
        ]
        self.mock_router.discover_nodes.return_value = self.nodes

    def test_engine_creation(self):
        """Test creating PlannedQueryEngine."""
        engine = PlannedQueryEngine(router=self.mock_router)
        self.assertIsNotNone(engine.planner)
        self.assertIsNotNone(engine.executor)

    def test_factory_creation(self):
        """Test factory function."""
        engine = create_planned_engine(
            router=self.mock_router,
            specific_threshold=0.9,
            exploratory_variance=0.15
        )
        self.assertEqual(engine.planner.config.specific_threshold, 0.9)
        self.assertEqual(engine.planner.config.exploratory_variance, 0.15)

    @patch('query_planner.FederatedQueryEngine.federated_query')
    def test_query_returns_plan(self, mock_federated):
        """Test that query returns both response and plan."""
        mock_federated.return_value = AggregatedResponse(
            query_id="test",
            results=[{'answer_id': '1', 'normalized_prob': 0.8}],
            total_partition_sum=1.0,
            nodes_queried=2,
            nodes_responded=2,
            total_time_ms=50.0,
            aggregation_strategy="sum"
        )

        engine = PlannedQueryEngine(router=self.mock_router)
        response, plan = engine.query(
            query_text="test query",
            query_embedding=np.random.randn(384),
            top_k=10
        )

        self.assertIsInstance(response, AggregatedResponse)
        self.assertIsInstance(plan, QueryPlan)

    @patch('query_planner.FederatedQueryEngine.federated_query')
    def test_force_query_type(self, mock_federated):
        """Test forcing query type."""
        mock_federated.return_value = AggregatedResponse(
            query_id="test",
            results=[],
            total_partition_sum=0.0,
            nodes_queried=1,
            nodes_responded=1,
            total_time_ms=25.0,
            aggregation_strategy="max"
        )

        engine = PlannedQueryEngine(router=self.mock_router)
        _, plan = engine.query(
            query_text="test",
            query_embedding=np.random.randn(384),
            force_type=QueryType.EXPLORATORY
        )

        self.assertEqual(plan.query_type, QueryType.EXPLORATORY)

    def test_get_stats(self):
        """Test getting combined statistics."""
        engine = PlannedQueryEngine(router=self.mock_router)
        stats = engine.get_stats()

        self.assertIn('planner', stats)
        self.assertIn('executor', stats)
        self.assertIn('engine', stats)


if __name__ == '__main__':
    unittest.main()
