"""
Unit tests for Phase 5d: Streaming Aggregation.

Tests:
- PartialResult data structure
- StreamingConfig options
- StreamingFederatedEngine creation
- Async streaming query execution (with mock nodes)
- SSE formatting
- create_streaming_engine factory
- Prolog validation
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import numpy as np
from dataclasses import asdict

# Import Phase 5d classes
try:
    from federated_query import (
        PartialResult, StreamingConfig, StreamingFederatedEngine,
        create_streaming_engine, AggregationStrategy, AggregationConfig,
        NodeResult, NodeResponse, get_aggregator
    )
    from kleinberg_router import KGNode, KleinbergRouter
except ImportError:
    from unifyweaver.targets.python_runtime.federated_query import (
        PartialResult, StreamingConfig, StreamingFederatedEngine,
        create_streaming_engine, AggregationStrategy, AggregationConfig,
        NodeResult, NodeResponse, get_aggregator
    )
    from unifyweaver.targets.python_runtime.kleinberg_router import KGNode, KleinbergRouter


class TestPartialResult(unittest.TestCase):
    """Test PartialResult data structure."""

    def test_creation(self):
        """Test creating a partial result."""
        result = PartialResult(
            results=[{'answer_id': 'a1', 'normalized_prob': 0.5}],
            confidence=0.5,
            nodes_responded=2,
            nodes_total=4,
            elapsed_ms=150.0,
            is_final=False
        )
        self.assertEqual(result.confidence, 0.5)
        self.assertEqual(result.nodes_responded, 2)
        self.assertEqual(result.nodes_total, 4)
        self.assertEqual(result.elapsed_ms, 150.0)
        self.assertFalse(result.is_final)

    def test_final_result(self):
        """Test marking result as final."""
        result = PartialResult(
            results=[],
            confidence=1.0,
            nodes_responded=3,
            nodes_total=3,
            elapsed_ms=200.0,
            is_final=True
        )
        self.assertTrue(result.is_final)
        self.assertEqual(result.confidence, 1.0)

    def test_to_dict(self):
        """Test serialization."""
        result = PartialResult(
            results=[{'id': 1}],
            confidence=0.75,
            nodes_responded=3,
            nodes_total=4,
            elapsed_ms=100.0,
            is_final=False
        )
        d = result.to_dict()
        self.assertEqual(d['confidence'], 0.75)
        self.assertEqual(d['nodes_responded'], 3)
        self.assertEqual(d['nodes_total'], 4)
        self.assertEqual(d['elapsed_ms'], 100.0)
        self.assertFalse(d['is_final'])
        self.assertEqual(d['results'], [{'id': 1}])


class TestStreamingConfig(unittest.TestCase):
    """Test StreamingConfig defaults and options."""

    def test_defaults(self):
        """Test default configuration values."""
        config = StreamingConfig()
        self.assertEqual(config.yield_interval_ms, 100)
        self.assertEqual(config.min_confidence, 0.1)
        self.assertEqual(config.max_wait_ms, 5000)
        self.assertTrue(config.eager_yield)

    def test_custom_values(self):
        """Test custom configuration."""
        config = StreamingConfig(
            yield_interval_ms=50,
            min_confidence=0.2,
            max_wait_ms=10000,
            eager_yield=False
        )
        self.assertEqual(config.yield_interval_ms, 50)
        self.assertEqual(config.min_confidence, 0.2)
        self.assertEqual(config.max_wait_ms, 10000)
        self.assertFalse(config.eager_yield)


class TestStreamingFederatedEngine(unittest.TestCase):
    """Test StreamingFederatedEngine creation and methods."""

    def _make_mock_router(self, nodes=None):
        """Create mock router."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = nodes or []
        return router

    def _make_node(self, node_id):
        """Helper to create KGNode."""
        return KGNode(
            node_id=node_id,
            endpoint=f"http://{node_id}:8080",
            centroid=np.random.randn(384),
            topics=["test"],
            embedding_model="test-model"
        )

    def test_engine_creation(self):
        """Test creating streaming engine."""
        router = self._make_mock_router()
        engine = StreamingFederatedEngine(router=router)
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.streaming_config)
        self.assertEqual(engine.streaming_config.yield_interval_ms, 100)

    def test_engine_with_custom_config(self):
        """Test engine with custom streaming config."""
        router = self._make_mock_router()
        config = StreamingConfig(
            yield_interval_ms=50,
            eager_yield=False
        )
        engine = StreamingFederatedEngine(
            router=router,
            streaming_config=config
        )
        self.assertEqual(engine.streaming_config.yield_interval_ms, 50)
        self.assertFalse(engine.streaming_config.eager_yield)

    def test_get_stats(self):
        """Test getting engine stats."""
        router = self._make_mock_router()
        engine = StreamingFederatedEngine(
            router=router,
            streaming_config=StreamingConfig(
                yield_interval_ms=75,
                min_confidence=0.25
            )
        )
        stats = engine.get_stats()
        self.assertIn('streaming', stats)
        self.assertEqual(stats['streaming']['yield_interval_ms'], 75)
        self.assertEqual(stats['streaming']['min_confidence'], 0.25)

    def test_merge_response_streaming(self):
        """Test merging responses in streaming mode."""
        router = self._make_mock_router()
        engine = StreamingFederatedEngine(router=router)
        aggregator = get_aggregator(AggregationStrategy.SUM)

        # First response
        aggregated = {}
        response1 = NodeResponse(
            source_node="node1",
            results=[
                NodeResult("a1", "Answer 1", "hash1", 0.8, 2.0, None, 0.0)
            ],
            partition_sum=3.0,
            node_metadata={}
        )
        engine._merge_response_streaming(aggregated, response1, aggregator)

        self.assertIn("hash1", aggregated)
        self.assertEqual(aggregated["hash1"]["exp_score"], 2.0)
        self.assertEqual(aggregated["hash1"]["node_count"], 1)

        # Second response with same answer
        response2 = NodeResponse(
            source_node="node2",
            results=[
                NodeResult("a1", "Answer 1", "hash1", 0.7, 1.5, None, 0.0)
            ],
            partition_sum=2.5,
            node_metadata={}
        )
        engine._merge_response_streaming(aggregated, response2, aggregator)

        # SUM aggregation
        self.assertEqual(aggregated["hash1"]["exp_score"], 3.5)
        self.assertEqual(aggregated["hash1"]["node_count"], 2)

    def test_normalize_streaming(self):
        """Test normalizing streaming results."""
        router = self._make_mock_router()
        engine = StreamingFederatedEngine(router=router)

        aggregated = {
            "h1": {"answer_id": "a1", "answer_text": "A1", "answer_hash": "h1",
                   "exp_score": 2.0, "raw_score": 0.8, "node_count": 1},
            "h2": {"answer_id": "a2", "answer_text": "A2", "answer_hash": "h2",
                   "exp_score": 1.0, "raw_score": 0.5, "node_count": 1}
        }

        results = engine._normalize_streaming(aggregated, 5.0, 10)

        # Should be sorted by normalized_prob descending
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["answer_hash"], "h1")
        self.assertAlmostEqual(results[0]["normalized_prob"], 0.4, places=5)
        self.assertAlmostEqual(results[1]["normalized_prob"], 0.2, places=5)

    def test_normalize_streaming_top_k(self):
        """Test top_k limiting in normalization."""
        router = self._make_mock_router()
        engine = StreamingFederatedEngine(router=router)

        aggregated = {
            f"h{i}": {"answer_id": f"a{i}", "answer_text": f"A{i}", "answer_hash": f"h{i}",
                      "exp_score": float(i), "raw_score": 0.5, "node_count": 1}
            for i in range(10)
        }

        results = engine._normalize_streaming(aggregated, 55.0, 3)
        self.assertEqual(len(results), 3)
        # Should have top 3 by score
        hashes = [r["answer_hash"] for r in results]
        self.assertEqual(hashes, ["h9", "h8", "h7"])

    def test_parse_node_response(self):
        """Test parsing JSON response."""
        router = self._make_mock_router()
        engine = StreamingFederatedEngine(router=router)

        data = {
            "results": [
                {
                    "answer_id": "a1",
                    "answer_text": "Answer 1",
                    "answer_hash": "h1",
                    "raw_score": 0.9,
                    "exp_score": 2.5,
                    "local_density": 0.1
                }
            ],
            "partition_sum": 5.0,
            "metadata": {"model": "test"}
        }

        response = engine._parse_node_response(data, "node1")
        self.assertEqual(response.source_node, "node1")
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].answer_id, "a1")
        self.assertEqual(response.partition_sum, 5.0)


class TestStreamingQueryExecution(unittest.TestCase):
    """Test async streaming query execution."""

    def _make_mock_router(self, nodes=None):
        """Create mock router."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = nodes or []
        return router

    def _make_node(self, node_id):
        """Helper to create KGNode."""
        return KGNode(
            node_id=node_id,
            endpoint=f"http://{node_id}:8080",
            centroid=np.random.randn(384),
            topics=["test"],
            embedding_model="test-model"
        )

    def test_streaming_empty_nodes(self):
        """Test streaming with no nodes."""
        router = self._make_mock_router([])
        engine = StreamingFederatedEngine(router=router)

        async def run_test():
            results = []
            async for partial in engine.federated_query_streaming(
                "test query",
                np.random.randn(384)
            ):
                results.append(partial)
            return results

        results = asyncio.run(run_test())
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_final)
        self.assertEqual(results[0].nodes_total, 0)

    def test_streaming_with_mock_nodes(self):
        """Test streaming node selection and ranking logic."""
        # Create nodes with distinct centroids for similarity ranking
        query_embedding = np.array([1.0, 0.0, 0.0] + [0.0] * 381)

        # node0: perfectly aligned with query
        centroid0 = np.array([1.0, 0.0, 0.0] + [0.0] * 381)
        # node1: partially aligned
        centroid1 = np.array([0.7, 0.7, 0.0] + [0.0] * 381)
        # node2: orthogonal
        centroid2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381)

        nodes = [
            KGNode("node0", "http://node0:8080", centroid0, ["test"], "test-model"),
            KGNode("node1", "http://node1:8080", centroid1, ["test"], "test-model"),
            KGNode("node2", "http://node2:8080", centroid2, ["test"], "test-model"),
        ]

        router = self._make_mock_router(nodes)
        engine = StreamingFederatedEngine(
            router=router,
            streaming_config=StreamingConfig(eager_yield=True)
        )

        # Verify node selection logic works
        all_nodes = router.discover_nodes()
        self.assertEqual(len(all_nodes), 3)

        # Verify cosine similarity ranking
        ranked = [(n, engine._cosine_similarity(query_embedding, n.centroid))
                  for n in all_nodes]
        ranked.sort(key=lambda x: x[1], reverse=True)

        # node0 should be most similar (perfect alignment = 1.0)
        self.assertEqual(ranked[0][0].node_id, "node0")
        self.assertAlmostEqual(ranked[0][1], 1.0, places=5)
        # node1 should be second (partial alignment ~0.7)
        self.assertEqual(ranked[1][0].node_id, "node1")
        # node2 should be last (orthogonal = 0.0)
        self.assertEqual(ranked[2][0].node_id, "node2")
        self.assertAlmostEqual(ranked[2][1], 0.0, places=5)

    def test_streaming_integration_simple(self):
        """Test streaming with timeout handling."""
        # Create nodes
        np.random.seed(42)
        nodes = [
            KGNode(
                node_id="node0",
                endpoint="http://node0:8080",
                centroid=np.ones(384),
                topics=["test"],
                embedding_model="test-model"
            )
        ]
        router = self._make_mock_router(nodes)
        engine = StreamingFederatedEngine(
            router=router,
            timeout_ms=100,  # Short timeout
            streaming_config=StreamingConfig(eager_yield=True)
        )

        async def run_test():
            results = []
            # The streaming should handle timeouts gracefully
            async for partial in engine.federated_query_streaming(
                "test query",
                np.ones(384),
                federation_k=1
            ):
                results.append(partial)
            return results

        # This should complete without error (timeouts are handled)
        results = asyncio.run(run_test())

        # At minimum, should yield final result with 0 responses
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(results[-1].is_final)


class TestSSEFormatting(unittest.TestCase):
    """Test Server-Sent Events formatting."""

    def _make_mock_router(self, nodes=None):
        """Create mock router."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = nodes or []
        return router

    def test_sse_format(self):
        """Test SSE output format."""
        router = self._make_mock_router([])
        engine = StreamingFederatedEngine(router=router)

        async def run_test():
            results = []
            async for sse_line in engine.federated_query_sse(
                "test query",
                np.random.randn(384)
            ):
                results.append(sse_line)
            return results

        results = asyncio.run(run_test())
        self.assertEqual(len(results), 1)
        # SSE format: "data: {...}\n\n"
        self.assertTrue(results[0].startswith("data: "))
        self.assertTrue(results[0].endswith("\n\n"))

        # Should be valid JSON
        import json
        json_str = results[0][6:-2]  # Strip "data: " and "\n\n"
        data = json.loads(json_str)
        self.assertIn("is_final", data)
        self.assertTrue(data["is_final"])


class TestCreateStreamingEngine(unittest.TestCase):
    """Test create_streaming_engine factory."""

    def test_factory_defaults(self):
        """Test factory with default parameters."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = []

        engine = create_streaming_engine(router)

        self.assertIsInstance(engine, StreamingFederatedEngine)
        self.assertEqual(engine.streaming_config.yield_interval_ms, 100)
        self.assertEqual(engine.streaming_config.min_confidence, 0.1)
        self.assertTrue(engine.streaming_config.eager_yield)

    def test_factory_custom_params(self):
        """Test factory with custom parameters."""
        router = Mock(spec=KleinbergRouter)
        router.discover_nodes.return_value = []

        engine = create_streaming_engine(
            router,
            yield_interval_ms=50,
            min_confidence=0.25,
            max_wait_ms=10000,
            eager_yield=False,
            federation_k=5,
            timeout_ms=3000,
            aggregation_strategy=AggregationStrategy.MAX
        )

        self.assertEqual(engine.streaming_config.yield_interval_ms, 50)
        self.assertEqual(engine.streaming_config.min_confidence, 0.25)
        self.assertEqual(engine.streaming_config.max_wait_ms, 10000)
        self.assertFalse(engine.streaming_config.eager_yield)
        self.assertEqual(engine.federation_k, 5)
        self.assertEqual(engine.timeout_ms, 3000)
        self.assertEqual(engine.config.strategy, AggregationStrategy.MAX)


class TestPrologValidation(unittest.TestCase):
    """Test Prolog validation for Phase 5d options."""

    def test_streaming_true(self):
        """Test streaming(true) validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_federation_option(streaming(true)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_streaming_false(self):
        """Test streaming(false) validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_federation_option(streaming(false)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_streaming_options_list(self):
        """Test streaming options list validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_federation_option(streaming([yield_interval_ms(100), eager_yield(true)])), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_streaming_option_yield_interval(self):
        """Test yield_interval_ms validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_streaming_option(yield_interval_ms(50)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_streaming_option_min_confidence(self):
        """Test min_confidence validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_streaming_option(min_confidence(0.25)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)

    def test_streaming_option_protocol_flags(self):
        """Test SSE and websocket flags validation."""
        import subprocess
        result = subprocess.run([
            'swipl', '-g',
            "use_module('src/unifyweaver/core/service_validation'), "
            "is_valid_streaming_option(sse_enabled(true)), "
            "is_valid_streaming_option(websocket_enabled(false)), "
            "writeln('SUCCESS'), halt."
        ], capture_output=True, text=True, cwd='/home/s243a/Projects/UnifyWeaver/context/claude/UnifyWeaver')
        self.assertIn('SUCCESS', result.stdout)


if __name__ == '__main__':
    unittest.main()
