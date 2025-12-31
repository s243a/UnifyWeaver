#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# End-to-end test for KG Topology Phase 3: Distributed Routing

"""
End-to-end test for distributed KG routing.

This test simulates a multi-node KG network where:
- Node A: CSV/data format expert
- Node B: JSON/API expert
- Node C: General programming expert

Queries should route to the most relevant node based on
semantic similarity of interface centroids.
"""

import os
import sys
import tempfile
import shutil
import base64
import hashlib
import numpy as np

# Add source to path - use the package structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from unifyweaver.targets.python_runtime.discovery_clients import LocalDiscoveryClient
from unifyweaver.targets.python_runtime.kleinberg_router import KleinbergRouter, RoutingEnvelope, KGNode
from unifyweaver.targets.python_runtime.kg_topology_api import DistributedKGTopologyAPI


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def log_pass(msg):
    print(f"  {Colors.GREEN}[PASS]{Colors.END} {msg}")


def log_fail(msg):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")


def log_info(msg):
    print(f"  {Colors.BLUE}[INFO]{Colors.END} {msg}")


def log_section(msg):
    print(f"\n{Colors.YELLOW}=== {msg} ==={Colors.END}")


def create_mock_centroid(keywords: list, dim: int = 384) -> np.ndarray:
    """
    Create a deterministic mock centroid based on keywords.
    Uses hash of keywords to seed random generator for reproducibility.
    """
    seed = int(hashlib.md5(''.join(keywords).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    centroid = rng.randn(dim).astype(np.float32)
    # Normalize
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    return centroid


def create_mock_query_embedding(query: str, dim: int = 384) -> np.ndarray:
    """Create a mock query embedding based on query text."""
    seed = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


class MockEmbedder:
    """Mock embedder that creates deterministic embeddings from text."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        return create_mock_query_embedding(text, self.dim)


def setup_test_nodes(temp_dir: str):
    """
    Set up test nodes with different expertise areas.

    Returns dict of node_id -> (api, centroid, topics)
    """
    # Clear shared discovery registry
    LocalDiscoveryClient.clear_shared()

    nodes = {}

    # Node A: CSV/Data expert
    csv_centroid = create_mock_centroid(['csv', 'data', 'parsing', 'delimiter', 'columns'])
    nodes['csv-expert'] = {
        'db_path': os.path.join(temp_dir, 'csv_expert.db'),
        'port': 8081,
        'centroid': csv_centroid,
        'topics': ['csv', 'data', 'tabular'],
        'qa_pairs': [
            ("How do I parse a CSV file?", "Use csv.reader() or pandas.read_csv()"),
            ("What delimiter does CSV use?", "CSV typically uses comma as delimiter"),
            ("How to handle CSV headers?", "Pass header=0 to pandas or use DictReader"),
        ]
    }

    # Node B: JSON/API expert
    json_centroid = create_mock_centroid(['json', 'api', 'rest', 'endpoints', 'serialization'])
    nodes['json-expert'] = {
        'db_path': os.path.join(temp_dir, 'json_expert.db'),
        'port': 8082,
        'centroid': json_centroid,
        'topics': ['json', 'api', 'web'],
        'qa_pairs': [
            ("How do I parse JSON?", "Use json.loads() for strings or json.load() for files"),
            ("What is a REST API?", "REST is an architectural style using HTTP methods"),
            ("How to serialize to JSON?", "Use json.dumps() with optional indent parameter"),
        ]
    }

    # Node C: General programming expert
    prog_centroid = create_mock_centroid(['programming', 'code', 'functions', 'variables', 'loops'])
    nodes['prog-expert'] = {
        'db_path': os.path.join(temp_dir, 'prog_expert.db'),
        'port': 8083,
        'centroid': prog_centroid,
        'topics': ['programming', 'general', 'basics'],
        'qa_pairs': [
            ("What is a function?", "A function is a reusable block of code"),
            ("How do loops work?", "Loops repeat code until a condition is met"),
            ("What are variables?", "Variables store data values in memory"),
        ]
    }

    return nodes


def register_nodes_with_discovery(nodes: dict):
    """Register all nodes with the shared discovery service."""
    discovery = LocalDiscoveryClient(use_shared=True)

    for node_id, config in nodes.items():
        centroid_b64 = base64.b64encode(config['centroid'].tobytes()).decode()

        discovery.register(
            service_name='kg_topology',
            service_id=node_id,
            host='localhost',
            port=config['port'],
            tags=['kg_node'] + config['topics'],
            metadata={
                'semantic_centroid': centroid_b64,
                'interface_topics': config['topics'],
                'embedding_model': 'mock-embedder'
            }
        )

    return discovery


def test_node_discovery(discovery: LocalDiscoveryClient, nodes: dict):
    """Test that all nodes can be discovered."""
    log_section("Test 1: Node Discovery")

    all_nodes = discovery.discover('kg_topology', tags=['kg_node'])

    if len(all_nodes) == len(nodes):
        log_pass(f"Discovered all {len(nodes)} nodes")
    else:
        log_fail(f"Expected {len(nodes)} nodes, found {len(all_nodes)}")
        return False

    # Check individual topics
    csv_nodes = discovery.discover('kg_topology', tags=['csv'])
    if len(csv_nodes) == 1 and csv_nodes[0].service_id == 'csv-expert':
        log_pass("CSV expert discoverable by tag")
    else:
        log_fail("CSV expert not discoverable by tag")
        return False

    json_nodes = discovery.discover('kg_topology', tags=['json'])
    if len(json_nodes) == 1 and json_nodes[0].service_id == 'json-expert':
        log_pass("JSON expert discoverable by tag")
    else:
        log_fail("JSON expert not discoverable by tag")
        return False

    return True


def test_router_centroid_decoding(nodes: dict):
    """Test that router correctly decodes centroids from discovery."""
    log_section("Test 2: Centroid Decoding")

    discovery = LocalDiscoveryClient(use_shared=True)
    router = KleinbergRouter(
        local_node_id='test-router',
        discovery_client=discovery
    )

    discovered = router.discover_nodes()

    if len(discovered) != len(nodes):
        log_fail(f"Expected {len(nodes)} nodes, discovered {len(discovered)}")
        return False

    log_pass(f"Router discovered {len(discovered)} nodes")

    # Verify centroids are numpy arrays with correct shape
    for node in discovered:
        if not isinstance(node.centroid, np.ndarray):
            log_fail(f"Node {node.node_id} centroid is not numpy array")
            return False
        if node.centroid.shape[0] != 384:
            log_fail(f"Node {node.node_id} centroid has wrong shape: {node.centroid.shape}")
            return False

    log_pass("All centroids correctly decoded as numpy arrays")
    return True


def test_similarity_routing(nodes: dict):
    """Test that queries route to the most similar node."""
    log_section("Test 3: Similarity-Based Routing")

    discovery = LocalDiscoveryClient(use_shared=True)
    router = KleinbergRouter(
        local_node_id='test-router',
        discovery_client=discovery,
        similarity_threshold=0.0  # Accept all for testing
    )

    discovered = router.discover_nodes()

    # Test CSV query
    csv_query = create_mock_query_embedding("How do I parse CSV files with pandas?")
    probs = router.compute_routing_probability(discovered, csv_query)

    log_info(f"CSV query routing probabilities:")
    for node, prob in probs:
        log_info(f"  {node.node_id}: {prob:.4f}")

    # The node with highest probability should be related to the query type
    top_node = probs[0][0]
    log_info(f"Top node for CSV query: {top_node.node_id}")

    # Test JSON query
    json_query = create_mock_query_embedding("How do I call a REST API endpoint?")
    probs = router.compute_routing_probability(discovered, json_query)

    log_info(f"JSON query routing probabilities:")
    for node, prob in probs:
        log_info(f"  {node.node_id}: {prob:.4f}")

    top_node = probs[0][0]
    log_info(f"Top node for JSON query: {top_node.node_id}")

    log_pass("Routing probabilities computed successfully")
    return True


def test_shortcut_creation():
    """Test that path folding creates shortcuts."""
    log_section("Test 4: Path Folding Shortcuts")

    discovery = LocalDiscoveryClient(use_shared=True)
    router = KleinbergRouter(
        local_node_id='test-router',
        discovery_client=discovery,
        path_folding_enabled=True
    )

    # Initially no shortcuts
    if router.check_shortcut("test query") is not None:
        log_fail("Shortcut exists before creation")
        return False

    # Create shortcut
    router.create_shortcut("test query", "csv-expert")

    if router.check_shortcut("test query") != "csv-expert":
        log_fail("Shortcut not created correctly")
        return False

    log_pass("Shortcut created successfully")

    # Test persistence via get/load
    shortcuts = router.get_shortcuts()
    if len(shortcuts) != 1:
        log_fail(f"Expected 1 shortcut, got {len(shortcuts)}")
        return False

    log_pass("Shortcuts can be exported")

    # Clear and reload
    router.clear_shortcuts()
    if router.check_shortcut("test query") is not None:
        log_fail("Shortcuts not cleared")
        return False

    router.load_shortcuts(shortcuts)
    if router.check_shortcut("test query") != "csv-expert":
        log_fail("Shortcuts not reloaded correctly")
        return False

    log_pass("Shortcuts can be imported")
    return True


def test_routing_envelope():
    """Test routing envelope HTL and visited tracking."""
    log_section("Test 5: Routing Envelope")

    envelope = RoutingEnvelope(
        origin_node='node-a',
        htl=5,
        path_folding_enabled=True
    )

    # Add visited nodes
    envelope.visited.add('node-a')
    envelope.visited.add('node-b')

    # Serialize
    d = envelope.to_dict()

    if d['htl'] != 5:
        log_fail(f"HTL not preserved: {d['htl']}")
        return False

    if 'node-a' not in d['visited'] or 'node-b' not in d['visited']:
        log_fail("Visited set not preserved")
        return False

    log_pass("Envelope serialization works")

    # Deserialize
    envelope2 = RoutingEnvelope.from_dict(d)

    if envelope2.htl != 5:
        log_fail("HTL not restored")
        return False

    if 'node-b' not in envelope2.visited:
        log_fail("Visited set not restored")
        return False

    log_pass("Envelope deserialization works")
    return True


def test_distributed_api_schema():
    """Test DistributedKGTopologyAPI schema creation."""
    log_section("Test 6: Distributed API Schema")

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        api = DistributedKGTopologyAPI(
            db_path=db_path,
            node_id='test-node',
            discovery_backend='local'
        )

        # Check tables exist
        cursor = api.conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_shortcuts'")
        if cursor.fetchone() is None:
            log_fail("query_shortcuts table not created")
            return False

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='distributed_query_log'")
        if cursor.fetchone() is None:
            log_fail("distributed_query_log table not created")
            return False

        log_pass("Schema tables created")

        # Test shortcut creation
        api.create_shortcut("test query", "target-node", 42)

        query_hash = hashlib.sha256("test query".encode()).hexdigest()[:16]
        shortcut = api._get_shortcut(query_hash)

        if shortcut is None:
            log_fail("Shortcut not persisted")
            return False

        if shortcut['target_node_id'] != 'target-node':
            log_fail("Shortcut target_node_id incorrect")
            return False

        log_pass("Shortcuts persist to database")

        # Test stats
        stats = api.get_query_stats()
        if stats['node_id'] != 'test-node':
            log_fail("Stats node_id incorrect")
            return False

        log_pass("Query stats work")

        api.conn.close()
        return True

    finally:
        os.unlink(db_path)


def test_htl_limit():
    """Test that HTL limits prevent infinite routing."""
    log_section("Test 7: HTL Limit")

    discovery = LocalDiscoveryClient(use_shared=True)
    router = KleinbergRouter(
        local_node_id='test-router',
        discovery_client=discovery,
        max_hops=3
    )

    # Create envelope with HTL=0
    envelope = RoutingEnvelope(
        origin_node='test-router',
        htl=0
    )

    query_emb = create_mock_query_embedding("test query")

    # Should return empty due to HTL=0
    results = router.route_query(query_emb, "test query", envelope)

    if len(results) != 0:
        log_fail(f"Expected 0 results with HTL=0, got {len(results)}")
        return False

    log_pass("HTL=0 returns no results")

    # Create envelope with HTL=1 but all nodes visited
    envelope2 = RoutingEnvelope(
        origin_node='test-router',
        htl=5,
        visited={'csv-expert', 'json-expert', 'prog-expert', 'test-router'}
    )

    results = router.route_query(query_emb, "test query", envelope2)

    if len(results) != 0:
        log_fail(f"Expected 0 results with all visited, got {len(results)}")
        return False

    log_pass("All-visited returns no results")
    return True


def run_all_tests():
    """Run all end-to-end tests."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}KG Topology Phase 3: End-to-End Distributed Routing Tests{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")

    # Create temp directory for test databases
    temp_dir = tempfile.mkdtemp(prefix='kg_e2e_')

    try:
        # Setup
        log_section("Setup: Creating Test Nodes")
        nodes = setup_test_nodes(temp_dir)
        log_info(f"Created {len(nodes)} test node configurations")

        discovery = register_nodes_with_discovery(nodes)
        log_info("Registered nodes with shared discovery")

        # Run tests
        results = []
        results.append(("Node Discovery", test_node_discovery(discovery, nodes)))
        results.append(("Centroid Decoding", test_router_centroid_decoding(nodes)))
        results.append(("Similarity Routing", test_similarity_routing(nodes)))
        results.append(("Path Folding", test_shortcut_creation()))
        results.append(("Routing Envelope", test_routing_envelope()))
        results.append(("Distributed API", test_distributed_api_schema()))
        results.append(("HTL Limit", test_htl_limit()))

        # Summary
        print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BLUE}Test Summary{Colors.END}")
        print(f"{Colors.BLUE}{'='*60}{Colors.END}")

        passed = sum(1 for _, r in results if r)
        failed = sum(1 for _, r in results if not r)

        for name, result in results:
            status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
            print(f"  [{status}] {name}")

        print(f"\n  Total: {passed} passed, {failed} failed")

        if failed == 0:
            print(f"\n{Colors.GREEN}All tests passed!{Colors.END}\n")
            return True
        else:
            print(f"\n{Colors.RED}Some tests failed.{Colors.END}\n")
            return False

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        LocalDiscoveryClient.clear_shared()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
