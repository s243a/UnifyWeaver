#!/bin/bash
# Test suite for Phase 6: Distributed Services
# Tests sharding, replication, and cluster management for Python, Go, and Rust

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TESTS_PASSED=0
TESTS_FAILED=0

# Test function with timeout
run_test() {
    local test_name="$1"
    local test_file="$2"

    echo -n "  Testing $test_name... "
    if timeout 30 swipl -s "$test_file" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++))
    fi
}

# Create temp directory for test files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=========================================="
echo "Phase 6: Distributed Services Test Suite"
echo "=========================================="
echo ""

# ==========================================
# TEST GROUP 1: Validation Tests
# ==========================================
echo -e "${YELLOW}[1/5] Validation Tests${NC}"

# Test 1.1: Valid consistency levels
cat > "$TMPDIR/test_1_1.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    ( is_valid_consistency_level(eventual),
      is_valid_consistency_level(strong),
      is_valid_consistency_level(quorum),
      is_valid_consistency_level(causal),
      is_valid_consistency_level(read_your_writes) -> halt(0) ; halt(1) ).
EOF
run_test "valid consistency levels" "$TMPDIR/test_1_1.pl"

# Test 1.2: Valid sharding strategies
cat > "$TMPDIR/test_1_2.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    ( is_valid_sharding_strategy(hash),
      is_valid_sharding_strategy(range),
      is_valid_sharding_strategy(consistent_hash),
      is_valid_sharding_strategy(geographic) -> halt(0) ; halt(1) ).
EOF
run_test "valid sharding strategies" "$TMPDIR/test_1_2.pl"

# Test 1.3: Distributed service options
cat > "$TMPDIR/test_1_3.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Options = [
        distributed(true),
        sharding(consistent_hash),
        replication(3),
        consistency(quorum),
        partition_key(user_id)
    ],
    ( maplist(is_valid_service_option, Options) -> halt(0) ; halt(1) ).
EOF
run_test "distributed service options" "$TMPDIR/test_1_3.pl"

# Test 1.4: Get replication factor helper
cat > "$TMPDIR/test_1_4.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service = service(test, [replication(3)], [receive(_X), respond(_X)]),
    ( get_replication_factor(Service, 3) -> halt(0) ; halt(1) ).
EOF
run_test "get_replication_factor helper" "$TMPDIR/test_1_4.pl"

# Test 1.5: Get sharding strategy helper
cat > "$TMPDIR/test_1_5.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service = service(test, [sharding(consistent_hash)], [receive(_X), respond(_X)]),
    ( get_sharding_strategy(Service, consistent_hash) -> halt(0) ; halt(1) ).
EOF
run_test "get_sharding_strategy helper" "$TMPDIR/test_1_5.pl"

# Test 1.6: Is distributed service
cat > "$TMPDIR/test_1_6.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service1 = service(test, [distributed(true)], [receive(_X), respond(_X)]),
    Service2 = service(test, [sharding(hash)], [receive(_X), respond(_X)]),
    ( is_distributed_service(Service1), is_distributed_service(Service2) -> halt(0) ; halt(1) ).
EOF
run_test "is_distributed_service helper" "$TMPDIR/test_1_6.pl"

echo ""

# ==========================================
# TEST GROUP 2: Python Distributed Compilation
# ==========================================
echo -e "${YELLOW}[2/5] Python Distributed Compilation Tests${NC}"

# Test 2.1: Basic distributed service
cat > "$TMPDIR/test_2_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "ShardingStrategy") -> halt(0) ; halt(1) ).
EOF
run_test "Python distributed basic" "$TMPDIR/test_2_1.pl"

# Test 2.2: Sharding configuration
cat > "$TMPDIR/test_2_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(store, [sharding(consistent_hash), partition_key(user_id)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "ConsistentHashRing"),
      sub_string(Code, _, _, _, "partition_key") -> halt(0) ; halt(1) ).
EOF
run_test "Python sharding config" "$TMPDIR/test_2_2.pl"

# Test 2.3: Replication manager
cat > "$TMPDIR/test_2_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true), replication(3)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "ReplicationManager"),
      sub_string(Code, _, _, _, "write_quorum") -> halt(0) ; halt(1) ).
EOF
run_test "Python replication manager" "$TMPDIR/test_2_3.pl"

# Test 2.4: Consistency level
cat > "$TMPDIR/test_2_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true), consistency(quorum)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "ConsistencyLevel"),
      sub_string(Code, _, _, _, "QUORUM") -> halt(0) ; halt(1) ).
EOF
run_test "Python consistency level" "$TMPDIR/test_2_4.pl"

# Test 2.5: Cluster node management
cat > "$TMPDIR/test_2_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "ClusterNode"),
      sub_string(Code, _, _, _, "add_node") -> halt(0) ; halt(1) ).
EOF
run_test "Python cluster nodes" "$TMPDIR/test_2_5.pl"

echo ""

# ==========================================
# TEST GROUP 3: Go Distributed Compilation
# ==========================================
echo -e "${YELLOW}[3/5] Go Distributed Compilation Tests${NC}"

# Test 3.1: Basic distributed service
cat > "$TMPDIR/test_3_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "ShardingStrategy") -> halt(0) ; halt(1) ).
EOF
run_test "Go distributed basic" "$TMPDIR/test_3_1.pl"

# Test 3.2: Sharding router
cat > "$TMPDIR/test_3_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(store, [sharding(consistent_hash), partition_key(user_id)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "ShardRouter"),
      sub_string(Code, _, _, _, "GetShard") -> halt(0) ; halt(1) ).
EOF
run_test "Go shard router" "$TMPDIR/test_3_2.pl"

# Test 3.3: Consistent hash ring
cat > "$TMPDIR/test_3_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "ConsistentHashRing"),
      sub_string(Code, _, _, _, "AddNode") -> halt(0) ; halt(1) ).
EOF
run_test "Go consistent hash ring" "$TMPDIR/test_3_3.pl"

# Test 3.4: Replication with quorum
cat > "$TMPDIR/test_3_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true), replication(3), consistency(quorum)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "ReplicationManager"),
      sub_string(Code, _, _, _, "WriteQuorum") -> halt(0) ; halt(1) ).
EOF
run_test "Go replication quorum" "$TMPDIR/test_3_4.pl"

# Test 3.5: Thread safety (sync)
cat > "$TMPDIR/test_3_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "sync.RWMutex"),
      sub_string(Code, _, _, _, "atomic") -> halt(0) ; halt(1) ).
EOF
run_test "Go thread safety" "$TMPDIR/test_3_5.pl"

echo ""

# ==========================================
# TEST GROUP 4: Rust Distributed Compilation
# ==========================================
echo -e "${YELLOW}[4/5] Rust Distributed Compilation Tests${NC}"

# Test 4.1: Basic distributed service
cat > "$TMPDIR/test_4_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "ShardingStrategy") -> halt(0) ; halt(1) ).
EOF
run_test "Rust distributed basic" "$TMPDIR/test_4_1.pl"

# Test 4.2: Sharding with BTreeMap
cat > "$TMPDIR/test_4_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [sharding(consistent_hash)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "BTreeMap"),
      sub_string(Code, _, _, _, "ConsistentHashRing") -> halt(0) ; halt(1) ).
EOF
run_test "Rust sharding BTreeMap" "$TMPDIR/test_4_2.pl"

# Test 4.3: Replication manager
cat > "$TMPDIR/test_4_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true), replication(3)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "ReplicationManager"),
      sub_string(Code, _, _, _, "write_quorum") -> halt(0) ; halt(1) ).
EOF
run_test "Rust replication manager" "$TMPDIR/test_4_3.pl"

# Test 4.4: RwLock thread safety
cat > "$TMPDIR/test_4_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "RwLock"),
      sub_string(Code, _, _, _, "AtomicU64") -> halt(0) ; halt(1) ).
EOF
run_test "Rust thread safety" "$TMPDIR/test_4_4.pl"

# Test 4.5: Partition key routing
cat > "$TMPDIR/test_4_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true), partition_key(user_id)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "get_partition_key"),
      sub_string(Code, _, _, _, "route_request") -> halt(0) ; halt(1) ).
EOF
run_test "Rust partition key routing" "$TMPDIR/test_4_5.pl"

echo ""

# ==========================================
# TEST GROUP 5: Cross-Target Consistency
# ==========================================
echo -e "${YELLOW}[5/5] Cross-Target Consistency Tests${NC}"

# Test 5.1: All targets support hash sharding
cat > "$TMPDIR/test_5_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- use_module('src/unifyweaver/targets/go_target').
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [sharding(hash)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, PyCode),
      compile_service_to_go(Service, GoCode),
      compile_service_to_rust(Service, RsCode),
      sub_string(PyCode, _, _, _, "HASH"),
      sub_string(GoCode, _, _, _, "ShardHash"),
      sub_string(RsCode, _, _, _, "Hash") -> halt(0) ; halt(1) ).
EOF
run_test "all targets hash sharding" "$TMPDIR/test_5_1.pl"

# Test 5.2: All targets support consistent hash
cat > "$TMPDIR/test_5_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- use_module('src/unifyweaver/targets/go_target').
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [sharding(consistent_hash)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, PyCode),
      compile_service_to_go(Service, GoCode),
      compile_service_to_rust(Service, RsCode),
      sub_string(PyCode, _, _, _, "ConsistentHashRing"),
      sub_string(GoCode, _, _, _, "ConsistentHashRing"),
      sub_string(RsCode, _, _, _, "ConsistentHashRing") -> halt(0) ; halt(1) ).
EOF
run_test "all targets consistent hash" "$TMPDIR/test_5_2.pl"

# Test 5.3: All targets support replication
cat > "$TMPDIR/test_5_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- use_module('src/unifyweaver/targets/go_target').
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(store, [distributed(true), replication(3)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, PyCode),
      compile_service_to_go(Service, GoCode),
      compile_service_to_rust(Service, RsCode),
      sub_string(PyCode, _, _, _, "replication_factor"),
      sub_string(GoCode, _, _, _, "ReplicationFactor"),
      sub_string(RsCode, _, _, _, "replication_factor") -> halt(0) ; halt(1) ).
EOF
run_test "all targets replication" "$TMPDIR/test_5_3.pl"

echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo "Total:  $((TESTS_PASSED + TESTS_FAILED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
