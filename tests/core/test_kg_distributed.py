# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Unit tests for KG Topology Phase 3: Distributed Network

"""
Unit tests for distributed KG topology components.

Tests:
- LocalDiscoveryClient: In-memory service discovery
- ConsulDiscoveryClient: Consul API (mocked)
- KleinbergRouter: Routing logic and shortcuts
- DistributedKGTopologyAPI: Schema and methods
"""

import unittest
import tempfile
import os
import sys
import json
import time
import base64

import numpy as np

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'unifyweaver', 'targets', 'python_runtime'))

from discovery_clients import (
    DiscoveryClient, LocalDiscoveryClient, ConsulDiscoveryClient,
    ServiceInstance, HealthStatus, create_discovery_client
)
from kleinberg_router import (
    KleinbergRouter, KGNode, RoutingEnvelope, QueryResult
)


class TestServiceInstance(unittest.TestCase):
    """Tests for ServiceInstance dataclass."""

    def test_to_dict(self):
        """Test ServiceInstance serialization."""
        instance = ServiceInstance(
            service_id='test-123',
            service_name='kg_topology',
            host='localhost',
            port=8080,
            tags=['kg_node', 'expert'],
            metadata={'key': 'value'},
            health_status=HealthStatus.HEALTHY
        )

        d = instance.to_dict()
        self.assertEqual(d['service_id'], 'test-123')
        self.assertEqual(d['service_name'], 'kg_topology')
        self.assertEqual(d['host'], 'localhost')
        self.assertEqual(d['port'], 8080)
        self.assertEqual(d['tags'], ['kg_node', 'expert'])
        self.assertEqual(d['metadata'], {'key': 'value'})
        self.assertEqual(d['health_status'], 'HEALTHY')

    def test_from_dict(self):
        """Test ServiceInstance deserialization."""
        d = {
            'service_id': 'test-456',
            'service_name': 'kg_topology',
            'host': '192.168.1.1',
            'port': 9090,
            'tags': ['tag1'],
            'metadata': {'foo': 'bar'},
            'health_status': 'UNHEALTHY'
        }

        instance = ServiceInstance.from_dict(d)
        self.assertEqual(instance.service_id, 'test-456')
        self.assertEqual(instance.health_status, HealthStatus.UNHEALTHY)


class TestLocalDiscoveryClient(unittest.TestCase):
    """Tests for LocalDiscoveryClient."""

    def setUp(self):
        """Set up test client."""
        self.client = LocalDiscoveryClient()
        self.client.clear()

    def tearDown(self):
        """Clean up."""
        self.client.clear()

    def test_register_and_discover(self):
        """Test basic registration and discovery."""
        # Register a service
        success = self.client.register(
            service_name='kg_topology',
            service_id='node-a',
            host='localhost',
            port=8081,
            tags=['kg_node'],
            metadata={'semantic_centroid': 'abc123'}
        )
        self.assertTrue(success)

        # Discover it
        instances = self.client.discover('kg_topology')
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].service_id, 'node-a')
        self.assertEqual(instances[0].metadata['semantic_centroid'], 'abc123')

    def test_discover_with_tags(self):
        """Test discovery filtering by tags."""
        # Register services with different tags
        self.client.register('kg_topology', 'node-a', 'localhost', 8081,
                            tags=['kg_node', 'csv_expert'])
        self.client.register('kg_topology', 'node-b', 'localhost', 8082,
                            tags=['kg_node', 'json_expert'])
        self.client.register('kg_topology', 'node-c', 'localhost', 8083,
                            tags=['kg_node', 'csv_expert', 'json_expert'])

        # Filter by single tag
        csv_nodes = self.client.discover('kg_topology', tags=['csv_expert'])
        self.assertEqual(len(csv_nodes), 2)

        # Filter by multiple tags (AND)
        both_nodes = self.client.discover('kg_topology',
                                         tags=['csv_expert', 'json_expert'])
        self.assertEqual(len(both_nodes), 1)
        self.assertEqual(both_nodes[0].service_id, 'node-c')

    def test_deregister(self):
        """Test service deregistration."""
        self.client.register('kg_topology', 'node-a', 'localhost', 8081)

        # Verify registered
        self.assertEqual(len(self.client.discover('kg_topology')), 1)

        # Deregister
        success = self.client.deregister('node-a')
        self.assertTrue(success)

        # Verify gone
        self.assertEqual(len(self.client.discover('kg_topology')), 0)

        # Deregister non-existent
        success = self.client.deregister('node-x')
        self.assertFalse(success)

    def test_heartbeat(self):
        """Test heartbeat updates."""
        self.client.register('kg_topology', 'node-a', 'localhost', 8081)

        # Get initial heartbeat
        instances = self.client.discover('kg_topology')
        initial_hb = instances[0].last_heartbeat

        # Small delay
        time.sleep(0.01)

        # Send heartbeat
        success = self.client.heartbeat('node-a')
        self.assertTrue(success)

        # Check updated
        instances = self.client.discover('kg_topology')
        self.assertGreater(instances[0].last_heartbeat, initial_hb)

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Register with very short TTL
        self.client.register('kg_topology', 'node-a', 'localhost', 8081, ttl=0)

        # Small delay to exceed TTL
        time.sleep(0.01)

        # Should be filtered out by healthy_only
        instances = self.client.discover('kg_topology', healthy_only=True)
        self.assertEqual(len(instances), 0)

        # But still exists if healthy_only=False
        instances = self.client.discover('kg_topology', healthy_only=False)
        self.assertEqual(len(instances), 1)

    def test_shared_registry(self):
        """Test shared registry mode."""
        LocalDiscoveryClient.clear_shared()

        client1 = LocalDiscoveryClient(use_shared=True)
        client2 = LocalDiscoveryClient(use_shared=True)

        # Register via client1
        client1.register('kg_topology', 'node-a', 'localhost', 8081)

        # Discover via client2
        instances = client2.discover('kg_topology')
        self.assertEqual(len(instances), 1)

        LocalDiscoveryClient.clear_shared()


class TestCreateDiscoveryClient(unittest.TestCase):
    """Tests for discovery client factory."""

    def test_create_local(self):
        """Test creating local client."""
        client = create_discovery_client('local')
        self.assertIsInstance(client, LocalDiscoveryClient)

    def test_create_consul(self):
        """Test creating Consul client."""
        client = create_discovery_client('consul', host='127.0.0.1', port=8500)
        self.assertIsInstance(client, ConsulDiscoveryClient)

    def test_create_unknown(self):
        """Test error on unknown backend."""
        with self.assertRaises(ValueError):
            create_discovery_client('unknown_backend')


class TestKGNode(unittest.TestCase):
    """Tests for KGNode dataclass."""

    def test_to_dict(self):
        """Test KGNode serialization."""
        centroid = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        node = KGNode(
            node_id='node-1',
            endpoint='http://localhost:8080',
            centroid=centroid,
            topics=['csv', 'data'],
            embedding_model='all-MiniLM-L6-v2',
            similarity=0.85
        )

        d = node.to_dict()
        self.assertEqual(d['node_id'], 'node-1')
        self.assertEqual(d['endpoint'], 'http://localhost:8080')
        self.assertAlmostEqual(d['centroid'][0], 0.1, places=5)
        self.assertEqual(d['topics'], ['csv', 'data'])


class TestRoutingEnvelope(unittest.TestCase):
    """Tests for RoutingEnvelope dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        envelope = RoutingEnvelope(
            origin_node='node-a',
            htl=8,
            visited={'node-a', 'node-b'},
            path_folding_enabled=True
        )

        d = envelope.to_dict()
        self.assertEqual(d['origin_node'], 'node-a')
        self.assertEqual(d['htl'], 8)
        self.assertIn('node-a', d['visited'])
        self.assertIn('node-b', d['visited'])
        self.assertTrue(d['path_folding_enabled'])

    def test_from_dict(self):
        """Test deserialization."""
        d = {
            'origin_node': 'node-x',
            'htl': 5,
            'visited': ['node-x', 'node-y'],
            'path_folding_enabled': False
        }

        envelope = RoutingEnvelope.from_dict(d)
        self.assertEqual(envelope.origin_node, 'node-x')
        self.assertEqual(envelope.htl, 5)
        self.assertIn('node-y', envelope.visited)
        self.assertFalse(envelope.path_folding_enabled)


class TestKleinbergRouter(unittest.TestCase):
    """Tests for KleinbergRouter."""

    def setUp(self):
        """Set up test router with mock discovery."""
        self.discovery = LocalDiscoveryClient()
        self.discovery.clear()

        # Register some mock nodes
        self._register_mock_nodes()

        self.router = KleinbergRouter(
            local_node_id='local-node',
            discovery_client=self.discovery,
            alpha=2.0,
            max_hops=5,
            parallel_paths=1,
            similarity_threshold=0.3,
            path_folding_enabled=True
        )

    def _register_mock_nodes(self):
        """Register mock KG nodes."""
        # Node A: CSV expert
        csv_centroid = np.array([0.8, 0.2, 0.1], dtype=np.float32)
        self.discovery.register(
            'kg_topology', 'node-a', 'localhost', 8081,
            tags=['kg_node'],
            metadata={
                'semantic_centroid': base64.b64encode(csv_centroid.tobytes()).decode(),
                'interface_topics': ['csv', 'data'],
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        )

        # Node B: JSON expert
        json_centroid = np.array([0.1, 0.8, 0.2], dtype=np.float32)
        self.discovery.register(
            'kg_topology', 'node-b', 'localhost', 8082,
            tags=['kg_node'],
            metadata={
                'semantic_centroid': base64.b64encode(json_centroid.tobytes()).decode(),
                'interface_topics': ['json', 'api'],
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        )

    def tearDown(self):
        """Clean up."""
        self.discovery.clear()

    def test_discover_nodes(self):
        """Test node discovery."""
        nodes = self.router.discover_nodes()
        self.assertEqual(len(nodes), 2)

        node_ids = {n.node_id for n in nodes}
        self.assertIn('node-a', node_ids)
        self.assertIn('node-b', node_ids)

    def test_discover_nodes_decodes_centroid(self):
        """Test centroid decoding from base64."""
        nodes = self.router.discover_nodes()

        for node in nodes:
            self.assertIsInstance(node.centroid, np.ndarray)
            self.assertEqual(node.centroid.dtype, np.float32)
            self.assertEqual(len(node.centroid), 3)

    def test_shortcut_management(self):
        """Test shortcut creation and lookup."""
        # No shortcut initially
        self.assertIsNone(self.router.check_shortcut('test query'))

        # Create shortcut
        self.router.create_shortcut('test query', 'node-a')

        # Check exists
        self.assertEqual(self.router.check_shortcut('test query'), 'node-a')

        # Remove shortcut
        self.router.remove_shortcut('test query')
        self.assertIsNone(self.router.check_shortcut('test query'))

    def test_clear_shortcuts(self):
        """Test clearing all shortcuts."""
        self.router.create_shortcut('query1', 'node-a')
        self.router.create_shortcut('query2', 'node-b')

        self.router.clear_shortcuts()

        self.assertIsNone(self.router.check_shortcut('query1'))
        self.assertIsNone(self.router.check_shortcut('query2'))

    def test_compute_routing_probability(self):
        """Test softmax routing probability computation."""
        nodes = self.router.discover_nodes()

        # Query closer to CSV node
        csv_query = np.array([0.9, 0.1, 0.0], dtype=np.float32)

        probs = self.router.compute_routing_probability(nodes, csv_query)

        # Should have probabilities summing to 1
        total_prob = sum(p for _, p in probs)
        self.assertAlmostEqual(total_prob, 1.0, places=5)

        # First node should have highest probability (CSV expert)
        self.assertGreater(probs[0][1], probs[1][1])

    def test_get_stats(self):
        """Test router statistics."""
        # Create some shortcuts
        self.router.create_shortcut('q1', 'node-a')
        self.router.create_shortcut('q2', 'node-b')

        stats = self.router.get_stats()

        self.assertEqual(stats['local_node_id'], 'local-node')
        self.assertEqual(stats['shortcuts'], 2)
        self.assertEqual(stats['config']['alpha'], 2.0)
        self.assertEqual(stats['config']['max_hops'], 5)


class TestDistributedKGTopologyAPI(unittest.TestCase):
    """Tests for DistributedKGTopologyAPI."""

    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name

        # Import here to avoid circular imports
        from kg_topology_api import DistributedKGTopologyAPI
        self.api = DistributedKGTopologyAPI(
            db_path=self.db_path,
            node_id='test-node',
            discovery_backend='local'
        )

    def tearDown(self):
        """Clean up."""
        self.api.conn.close()
        os.unlink(self.db_path)

    def test_schema_creation(self):
        """Test distributed schema tables are created."""
        cursor = self.api.conn.cursor()

        # Check query_shortcuts table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='query_shortcuts'
        """)
        self.assertIsNotNone(cursor.fetchone())

        # Check remote_nodes table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='remote_nodes'
        """)
        self.assertIsNotNone(cursor.fetchone())

        # Check distributed_query_log table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='distributed_query_log'
        """)
        self.assertIsNotNone(cursor.fetchone())

    def test_node_id_generation(self):
        """Test node ID is generated correctly."""
        self.assertEqual(self.api.node_id, 'test-node')

        # Auto-generate from db path
        from kg_topology_api import DistributedKGTopologyAPI
        temp_db2 = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_db2.close()
        try:
            api2 = DistributedKGTopologyAPI(db_path=temp_db2.name)
            self.assertTrue(api2.node_id.startswith('node_'))
            api2.conn.close()
        finally:
            os.unlink(temp_db2.name)

    def test_create_shortcut(self):
        """Test shortcut creation."""
        self.api.create_shortcut('test query', 'target-node', 42)

        shortcut = self.api._get_shortcut(
            __import__('hashlib').sha256('test query'.encode()).hexdigest()[:16]
        )
        self.assertIsNotNone(shortcut)
        self.assertEqual(shortcut['target_node_id'], 'target-node')
        self.assertEqual(shortcut['target_interface_id'], 42)

    def test_shortcut_usage_tracking(self):
        """Test shortcut hit count updates."""
        query_hash = __import__('hashlib').sha256('test query'.encode()).hexdigest()[:16]

        self.api.create_shortcut('test query', 'target-node')

        # Get initial hit count
        cursor = self.api.conn.cursor()
        cursor.execute("SELECT hit_count FROM query_shortcuts WHERE query_hash = ?",
                      (query_hash,))
        initial_count = cursor.fetchone()[0]
        self.assertEqual(initial_count, 1)

        # Update usage
        self.api._update_shortcut_usage(query_hash)

        cursor.execute("SELECT hit_count FROM query_shortcuts WHERE query_hash = ?",
                      (query_hash,))
        new_count = cursor.fetchone()[0]
        self.assertEqual(new_count, 2)

    def test_query_stats(self):
        """Test query statistics."""
        stats = self.api.get_query_stats()

        self.assertEqual(stats['total_queries'], 0)
        self.assertEqual(stats['avg_hops'], 0)
        self.assertEqual(stats['shortcuts']['count'], 0)
        self.assertEqual(stats['node_id'], 'test-node')

    def test_prune_shortcuts(self):
        """Test shortcut pruning."""
        # Create shortcuts
        self.api.create_shortcut('query1', 'node-a')
        self.api.create_shortcut('query2', 'node-b')

        # Prune with high min_hits (removes everything)
        removed = self.api.prune_shortcuts(max_age_days=0, min_hits=100)
        self.assertEqual(removed, 2)

        # Verify gone
        stats = self.api.get_query_stats()
        self.assertEqual(stats['shortcuts']['count'], 0)

    def test_get_router(self):
        """Test router creation."""
        router = self.api.get_router()
        self.assertIsNotNone(router)
        self.assertEqual(router.local_node_id, 'test-node')

        # Same instance on second call
        router2 = self.api.get_router()
        self.assertIs(router, router2)


if __name__ == '__main__':
    unittest.main()
