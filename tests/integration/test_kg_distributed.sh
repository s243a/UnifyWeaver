#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# Integration tests for KG Topology Phase 3: Distributed Network

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Helper function for test results
pass() {
    echo -e "  ${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

fail() {
    echo -e "  ${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

skip() {
    echo -e "  ${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

echo "=========================================="
echo "KG Topology Phase 3: Distributed Network"
echo "Integration Tests"
echo "=========================================="
echo ""

# =============================================================================
# Group 1: Prolog Validation Tests
# =============================================================================

echo -e "${BLUE}[Group 1] Prolog Validation Tests${NC}"

# Test 1.1: Valid routing strategies
echo -n "  Testing valid routing strategies... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/core/service_validation'),
    (   is_valid_routing_strategy(direct),
        is_valid_routing_strategy(round_robin),
        is_valid_routing_strategy(kleinberg),
        is_valid_routing_strategy(kleinberg([alpha(2.0), max_hops(10)]))
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Valid routing strategies"
else
    fail "Valid routing strategies"
fi

# Test 1.2: Invalid routing strategies rejected
echo -n "  Testing invalid strategies rejected... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/core/service_validation'),
    (   \+ is_valid_routing_strategy(invalid_strategy),
        \+ is_valid_routing_strategy(kleinberg([invalid_option(123)]))
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Invalid strategies rejected"
else
    fail "Invalid strategies rejected"
fi

# Test 1.3: Kleinberg options validation
echo -n "  Testing Kleinberg options validation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/core/service_validation'),
    (   is_valid_kleinberg_option(alpha(2.5)),
        is_valid_kleinberg_option(max_hops(15)),
        is_valid_kleinberg_option(parallel_paths(3)),
        is_valid_kleinberg_option(similarity_threshold(0.7)),
        is_valid_kleinberg_option(path_folding(true)),
        \+ is_valid_kleinberg_option(alpha(-1)),
        \+ is_valid_kleinberg_option(max_hops(0)),
        \+ is_valid_kleinberg_option(similarity_threshold(1.5))
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Kleinberg options validation"
else
    fail "Kleinberg options validation"
fi

# Test 1.4: Discovery metadata validation
echo -n "  Testing discovery metadata validation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/core/service_validation'),
    (   is_valid_discovery_metadata_entry(semantic_centroid([0.1, 0.2, 0.3])),
        is_valid_discovery_metadata_entry(semantic_centroid('base64string')),
        is_valid_discovery_metadata_entry(interface_topics([csv, json, xml])),
        is_valid_discovery_metadata_entry(embedding_model('all-MiniLM-L6-v2'))
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Discovery metadata validation"
else
    fail "Discovery metadata validation"
fi

# Test 1.5: is_kleinberg_routed helper
echo -n "  Testing is_kleinberg_routed helper... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/core/service_validation'),
    Service1 = service(test, [routing(kleinberg)], handler),
    Service2 = service(test, [routing(kleinberg([alpha(2.0)]))], handler),
    Service3 = service(test, [routing(direct)], handler),
    (   is_kleinberg_routed(Service1),
        is_kleinberg_routed(Service2),
        \+ is_kleinberg_routed(Service3)
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "is_kleinberg_routed helper"
else
    fail "is_kleinberg_routed helper"
fi

# Test 1.6: Kleinberg option extraction
echo -n "  Testing Kleinberg option extraction... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/core/service_validation'),
    Service = service(test, [routing(kleinberg([alpha(3.0), max_hops(5)]))], handler),
    (   get_kleinberg_alpha(Service, 3.0),
        get_kleinberg_max_hops(Service, 5),
        get_kleinberg_parallel_paths(Service, 1),  % default
        get_kleinberg_similarity_threshold(Service, 0.5)  % default
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Kleinberg option extraction"
else
    fail "Kleinberg option extraction"
fi

echo ""

# =============================================================================
# Group 2: Python Discovery Client Tests
# =============================================================================

echo -e "${BLUE}[Group 2] Python Discovery Client Tests${NC}"

# Test 2.1: LocalDiscoveryClient
echo -n "  Testing LocalDiscoveryClient... "
RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from discovery_clients import LocalDiscoveryClient

client = LocalDiscoveryClient()
client.clear()

# Register
success = client.register('kg_topology', 'node-a', 'localhost', 8081,
                         tags=['kg_node'], metadata={'key': 'value'})
assert success, 'Registration failed'

# Discover
instances = client.discover('kg_topology')
assert len(instances) == 1, f'Expected 1 instance, got {len(instances)}'
assert instances[0].service_id == 'node-a', 'Wrong service_id'

# Deregister
success = client.deregister('node-a')
assert success, 'Deregistration failed'

instances = client.discover('kg_topology')
assert len(instances) == 0, 'Instance still present after deregister'

print('SUCCESS')
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "LocalDiscoveryClient"
else
    fail "LocalDiscoveryClient: $RESULT"
fi

# Test 2.2: Tag filtering
echo -n "  Testing tag filtering... "
RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from discovery_clients import LocalDiscoveryClient

client = LocalDiscoveryClient()
client.clear()

client.register('kg_topology', 'csv-node', 'localhost', 8081, tags=['kg_node', 'csv'])
client.register('kg_topology', 'json-node', 'localhost', 8082, tags=['kg_node', 'json'])
client.register('kg_topology', 'both-node', 'localhost', 8083, tags=['kg_node', 'csv', 'json'])

# Filter by single tag
csv_nodes = client.discover('kg_topology', tags=['csv'])
assert len(csv_nodes) == 2, f'Expected 2 CSV nodes, got {len(csv_nodes)}'

# Filter by multiple tags (AND)
both = client.discover('kg_topology', tags=['csv', 'json'])
assert len(both) == 1, f'Expected 1 both node, got {len(both)}'

print('SUCCESS')
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Tag filtering"
else
    fail "Tag filtering: $RESULT"
fi

# Test 2.3: create_discovery_client factory
echo -n "  Testing discovery client factory... "
RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from discovery_clients import create_discovery_client, LocalDiscoveryClient, ConsulDiscoveryClient

local = create_discovery_client('local')
assert isinstance(local, LocalDiscoveryClient)

consul = create_discovery_client('consul', host='127.0.0.1', port=8500)
assert isinstance(consul, ConsulDiscoveryClient)

try:
    create_discovery_client('unknown')
    assert False, 'Should have raised ValueError'
except ValueError:
    pass

print('SUCCESS')
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Discovery client factory"
else
    fail "Discovery client factory: $RESULT"
fi

echo ""

# =============================================================================
# Group 3: Kleinberg Router Tests
# =============================================================================

echo -e "${BLUE}[Group 3] Kleinberg Router Tests${NC}"

# Test 3.1: Router initialization
echo -n "  Testing router initialization... "
RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from kleinberg_router import KleinbergRouter
from discovery_clients import LocalDiscoveryClient

discovery = LocalDiscoveryClient()
router = KleinbergRouter(
    local_node_id='test-node',
    discovery_client=discovery,
    alpha=2.5,
    max_hops=8
)

assert router.local_node_id == 'test-node'
assert router.alpha == 2.5
assert router.max_hops == 8

print('SUCCESS')
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Router initialization"
else
    fail "Router initialization: $RESULT"
fi

# Test 3.2: Shortcut management
echo -n "  Testing shortcut management... "
RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from kleinberg_router import KleinbergRouter
from discovery_clients import LocalDiscoveryClient

discovery = LocalDiscoveryClient()
router = KleinbergRouter('test-node', discovery)

# No shortcut initially
assert router.check_shortcut('test query') is None

# Create shortcut
router.create_shortcut('test query', 'target-node')
assert router.check_shortcut('test query') == 'target-node'

# Remove shortcut
router.remove_shortcut('test query')
assert router.check_shortcut('test query') is None

# Clear all
router.create_shortcut('q1', 'n1')
router.create_shortcut('q2', 'n2')
router.clear_shortcuts()
assert len(router.get_shortcuts()) == 0

print('SUCCESS')
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Shortcut management"
else
    fail "Shortcut management: $RESULT"
fi

# Test 3.3: RoutingEnvelope serialization
echo -n "  Testing RoutingEnvelope serialization... "
RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from kleinberg_router import RoutingEnvelope

envelope = RoutingEnvelope(
    origin_node='node-a',
    htl=8,
    visited={'node-a', 'node-b'},
    path_folding_enabled=True
)

d = envelope.to_dict()
assert d['origin_node'] == 'node-a'
assert d['htl'] == 8
assert 'node-a' in d['visited']
assert 'node-b' in d['visited']
assert d['path_folding_enabled'] == True

# Deserialize
envelope2 = RoutingEnvelope.from_dict(d)
assert envelope2.origin_node == 'node-a'
assert envelope2.htl == 8
assert 'node-b' in envelope2.visited

print('SUCCESS')
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "RoutingEnvelope serialization"
else
    fail "RoutingEnvelope serialization: $RESULT"
fi

echo ""

# =============================================================================
# Group 4: DistributedKGTopologyAPI Tests
# =============================================================================

echo -e "${BLUE}[Group 4] DistributedKGTopologyAPI Tests${NC}"

# Test 4.1: Schema creation
echo -n "  Testing distributed schema creation... "
RESULT=$(python3 -c "
import sys
import tempfile
import os
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from kg_topology_api import DistributedKGTopologyAPI

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

try:
    api = DistributedKGTopologyAPI(db_path, node_id='test-node')

    # Check tables exist
    cursor = api.conn.cursor()

    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='query_shortcuts'\")
    assert cursor.fetchone() is not None, 'query_shortcuts table missing'

    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='remote_nodes'\")
    assert cursor.fetchone() is not None, 'remote_nodes table missing'

    cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='distributed_query_log'\")
    assert cursor.fetchone() is not None, 'distributed_query_log table missing'

    api.conn.close()
    print('SUCCESS')
finally:
    os.unlink(db_path)
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Distributed schema creation"
else
    fail "Distributed schema creation: $RESULT"
fi

# Test 4.2: Shortcut persistence
echo -n "  Testing shortcut persistence... "
RESULT=$(python3 -c "
import sys
import tempfile
import os
import hashlib
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from kg_topology_api import DistributedKGTopologyAPI

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

try:
    api = DistributedKGTopologyAPI(db_path, node_id='test-node')

    # Create shortcut
    api.create_shortcut('test query', 'target-node', 42)

    # Verify in database
    query_hash = hashlib.sha256('test query'.encode()).hexdigest()[:16]
    shortcut = api._get_shortcut(query_hash)
    assert shortcut is not None
    assert shortcut['target_node_id'] == 'target-node'
    assert shortcut['target_interface_id'] == 42

    api.conn.close()
    print('SUCCESS')
finally:
    os.unlink(db_path)
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Shortcut persistence"
else
    fail "Shortcut persistence: $RESULT"
fi

# Test 4.3: Query statistics
echo -n "  Testing query statistics... "
RESULT=$(python3 -c "
import sys
import tempfile
import os
sys.path.insert(0, '$PROJECT_ROOT/src/unifyweaver/targets/python_runtime')
from kg_topology_api import DistributedKGTopologyAPI

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

try:
    api = DistributedKGTopologyAPI(db_path, node_id='test-node')

    stats = api.get_query_stats()
    assert stats['total_queries'] == 0
    assert stats['avg_hops'] == 0
    assert stats['shortcuts']['count'] == 0
    assert stats['node_id'] == 'test-node'

    api.conn.close()
    print('SUCCESS')
finally:
    os.unlink(db_path)
" 2>&1)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Query statistics"
else
    fail "Query statistics: $RESULT"
fi

echo ""

# =============================================================================
# Group 5: Code Generation Tests
# =============================================================================

echo -e "${BLUE}[Group 5] Code Generation Tests${NC}"

# Test 5.1: Python Kleinberg router generation
echo -n "  Testing Python router code generation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/python_target'),
    compile_kleinberg_router_python([alpha(3.0), max_hops(8)], Code),
    (   sub_string(Code, _, _, _, 'alpha=3.0'),
        sub_string(Code, _, _, _, 'max_hops=8'),
        sub_string(Code, _, _, _, 'KleinbergRouter')
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Python router code generation"
else
    fail "Python router code generation"
fi

# Test 5.2: Go Kleinberg router generation
echo -n "  Testing Go router code generation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/go_target'),
    compile_kleinberg_router_go([alpha(2.5), parallel_paths(3)], Code),
    (   sub_string(Code, _, _, _, 'Alpha'),
        sub_string(Code, _, _, _, 'KleinbergRouter'),
        sub_string(Code, _, _, _, 'sync.RWMutex')
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Go router code generation"
else
    fail "Go router code generation"
fi

# Test 5.3: Rust Kleinberg router generation
echo -n "  Testing Rust router code generation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/rust_target'),
    compile_kleinberg_router_rust([similarity_threshold(0.6)], Code),
    (   sub_string(Code, _, _, _, 'KleinbergRouter'),
        sub_string(Code, _, _, _, 'RwLock'),
        sub_string(Code, _, _, _, 'cosine_similarity')
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Rust router code generation"
else
    fail "Rust router code generation"
fi

# Test 5.4: KG endpoint generation (Python)
echo -n "  Testing Python KG endpoint generation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/glue/network_glue'),
    generate_kg_query_endpoint(python, [], Code),
    (   sub_string(Code, _, _, _, '/kg/query'),
        sub_string(Code, _, _, _, '/kg/register'),
        sub_string(Code, _, _, _, '/kg/health'),
        sub_string(Code, _, _, _, 'handle_remote_query')
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Python KG endpoint generation"
else
    fail "Python KG endpoint generation"
fi

# Test 5.5: KG endpoint generation (Go)
echo -n "  Testing Go KG endpoint generation... "
RESULT=$(swipl -g "
    use_module('$PROJECT_ROOT/src/unifyweaver/glue/network_glue'),
    generate_kg_query_endpoint(go, [], Code),
    (   sub_string(Code, _, _, _, 'handleKGQuery'),
        sub_string(Code, _, _, _, 'handleKGHealth'),
        sub_string(Code, _, _, _, 'RegisterKGRoutes')
    ->  writeln('SUCCESS')
    ;   writeln('FAILED')
    ),
    halt.
" 2>/dev/null)

if [[ "$RESULT" == *"SUCCESS"* ]]; then
    pass "Go KG endpoint generation"
else
    fail "Go KG endpoint generation"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "  ${GREEN}Passed:${NC}  $TESTS_PASSED"
echo -e "  ${RED}Failed:${NC}  $TESTS_FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
