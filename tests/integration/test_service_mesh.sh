#!/bin/bash
# Phase 4: Service Mesh Integration Tests
# Tests load balancing, circuit breaker, and retry with backoff

# Don't exit on error - we handle test failures ourselves
cd "$(dirname "$0")/../.."

PASSED=0
FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

run_test() {
    local name="$1"
    local cmd="$2"
    if timeout 20 bash -c "$cmd" >/dev/null 2>&1; then
        pass "$name"
    else
        fail "$name"
    fi
}

echo "=============================================="
echo "Phase 4: Service Mesh Integration Tests"
echo "=============================================="
echo ""

# ==============================================================================
# VALIDATION TESTS
# ==============================================================================
echo "--- Validation Tests ---"

# Load balance strategy validation
run_test "Valid load_balance round_robin" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(load_balance(round_robin)), halt(0)\" -t \"halt(1)\""

run_test "Valid load_balance random" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(load_balance(random)), halt(0)\" -t \"halt(1)\""

run_test "Valid load_balance least_connections" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(load_balance(least_connections)), halt(0)\" -t \"halt(1)\""

run_test "Valid load_balance weighted" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(load_balance(weighted)), halt(0)\" -t \"halt(1)\""

run_test "Valid load_balance ip_hash" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(load_balance(ip_hash)), halt(0)\" -t \"halt(1)\""

# Circuit breaker validation
run_test "Valid circuit_breaker basic" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(circuit_breaker(threshold(5), timeout(30000))), halt(0)\" -t \"halt(1)\""

run_test "Valid circuit_breaker with half_open" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(circuit_breaker(threshold(3), timeout(10000), half_open_requests(2))), halt(0)\" -t \"halt(1)\""

run_test "Valid circuit_breaker with success_threshold" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(circuit_breaker(threshold(5), timeout(30000), success_threshold(3))), halt(0)\" -t \"halt(1)\""

# Retry validation
run_test "Valid retry fixed" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(retry(3, fixed)), halt(0)\" -t \"halt(1)\""

run_test "Valid retry linear" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(retry(5, linear)), halt(0)\" -t \"halt(1)\""

run_test "Valid retry exponential" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(retry(3, exponential)), halt(0)\" -t \"halt(1)\""

run_test "Valid retry with delay" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(retry(3, fixed, delay(1000))), halt(0)\" -t \"halt(1)\""

run_test "Valid retry with max_delay" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(retry(3, exponential, delay(100), max_delay(10000))), halt(0)\" -t \"halt(1)\""

# Discovery validation
run_test "Valid discovery static" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(discovery(static)), halt(0)\" -t \"halt(1)\""

run_test "Valid discovery dns" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(discovery(dns)), halt(0)\" -t \"halt(1)\""

run_test "Valid discovery consul" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(discovery(consul)), halt(0)\" -t \"halt(1)\""

run_test "Valid discovery etcd" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(discovery(etcd)), halt(0)\" -t \"halt(1)\""

# Backends validation
run_test "Valid backends single" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(backends([backend(server1, tcp(localhost, 8080))])), halt(0)\" -t \"halt(1)\""

run_test "Valid backends multiple" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(backends([backend(server1, tcp(localhost, 8080)), backend(server2, tcp(localhost, 8081))])), halt(0)\" -t \"halt(1)\""

run_test "Valid backends with options" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_valid_service_option(backends([backend(server1, tcp(localhost, 8080), [weight(3)]), backend(server2, tcp(localhost, 8081), [weight(1)])])), halt(0)\" -t \"halt(1)\""

# ==============================================================================
# HELPER PREDICATE TESTS
# ==============================================================================
echo ""
echo "--- Helper Predicate Tests ---"

run_test "get_load_balance_strategy extracts strategy" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_load_balance_strategy(service(test, [load_balance(round_robin)], []), round_robin), halt(0)\" -t \"halt(1)\""

run_test "get_load_balance_strategy defaults to none" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_load_balance_strategy(service(test, [], []), none), halt(0)\" -t \"halt(1)\""

run_test "get_circuit_breaker_config extracts config" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_circuit_breaker_config(service(test, [circuit_breaker(threshold(5), timeout(30000))], []), config(5, 30000)), halt(0)\" -t \"halt(1)\""

run_test "get_retry_config extracts config" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_retry_config(service(test, [retry(3, exponential)], []), config(3, exponential, 100, 30000, false)), halt(0)\" -t \"halt(1)\""

run_test "get_retry_config with custom delay" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_retry_config(service(test, [retry(5, fixed, [delay(500)])], []), config(5, fixed, 500, 30000, false)), halt(0)\" -t \"halt(1)\""

run_test "has_load_balancing detects option" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), has_load_balancing(service(test, [load_balance(random)], [])), halt(0)\" -t \"halt(1)\""

run_test "has_circuit_breaker detects option" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), has_circuit_breaker(service(test, [circuit_breaker(threshold(5), timeout(30000))], [])), halt(0)\" -t \"halt(1)\""

run_test "has_retry detects option" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), has_retry(service(test, [retry(3, fixed)], [])), halt(0)\" -t \"halt(1)\""

run_test "is_service_mesh_service detects mesh service" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_service_mesh_service(service(test, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000))], [])), halt(0)\" -t \"halt(1)\""

run_test "is_service_mesh_service with retry only" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_service_mesh_service(service(test, [retry(3, exponential)], [])), halt(0)\" -t \"halt(1)\""

# ==============================================================================
# PYTHON SERVICE MESH COMPILATION TESTS
# ==============================================================================
echo ""
echo "--- Python Service Mesh Compilation Tests ---"

run_test "Python service mesh with round_robin" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'GatewayService'), sub_string(Code, _, _, _, 'CircuitState'), halt(0)\" -t \"halt(1)\""

run_test "Python service mesh with random" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(loadbalancer, [load_balance(random), circuit_breaker(threshold(3), timeout(10000)), retry(5, fixed)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'LoadbalancerService'), halt(0)\" -t \"halt(1)\""

run_test "Python service mesh with least_connections" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(proxy, [load_balance(least_connections), circuit_breaker(threshold(10), timeout(60000)), retry(2, linear)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'ProxyService'), halt(0)\" -t \"halt(1)\""

run_test "Python service mesh has _select_backend method" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_select_backend'), halt(0)\" -t \"halt(1)\""

run_test "Python service mesh has _check_circuit method" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_check_circuit'), halt(0)\" -t \"halt(1)\""

run_test "Python service mesh has _calculate_delay method" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, '_calculate_delay'), halt(0)\" -t \"halt(1)\""

run_test "Python service mesh has register_service call" \
    "swipl -g \"use_module('src/unifyweaver/targets/python_target'), compile_service_mesh_python(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'register_service'), halt(0)\" -t \"halt(1)\""

# ==============================================================================
# GO SERVICE MESH COMPILATION TESTS
# ==============================================================================
echo ""
echo "--- Go Service Mesh Compilation Tests ---"

run_test "Go service mesh with round_robin" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'GatewayService'), sub_string(Code, _, _, _, 'CircuitState'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh with random" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(loadbalancer, [load_balance(random), circuit_breaker(threshold(3), timeout(10000)), retry(5, fixed)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'LoadbalancerService'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh with least_connections" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(proxy, [load_balance(least_connections), circuit_breaker(threshold(10), timeout(60000)), retry(2, linear)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'ProxyService'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh has selectBackend method" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'selectBackend'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh has checkCircuit method" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'checkCircuit'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh has calculateDelay method" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'calculateDelay'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh has atomic operations" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'atomic'), halt(0)\" -t \"halt(1)\""

run_test "Go service mesh registers with RegisterService" \
    "swipl -g \"use_module('src/unifyweaver/targets/go_target'), compile_service_mesh_go(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'RegisterService'), halt(0)\" -t \"halt(1)\""

# ==============================================================================
# RUST SERVICE MESH COMPILATION TESTS
# ==============================================================================
echo ""
echo "--- Rust Service Mesh Compilation Tests ---"

run_test "Rust service mesh with round_robin" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'GatewayService'), sub_string(Code, _, _, _, 'CircuitState'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh with random" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(loadbalancer, [load_balance(random), circuit_breaker(threshold(3), timeout(10000)), retry(5, fixed)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'LoadbalancerService'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh with least_connections" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(proxy, [load_balance(least_connections), circuit_breaker(threshold(10), timeout(60000)), retry(2, linear)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'ProxyService'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh has select_backend method" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'select_backend'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh has check_circuit method" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'check_circuit'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh has calculate_delay method" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'calculate_delay'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh has RwLock for circuit state" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'RwLock'), halt(0)\" -t \"halt(1)\""

run_test "Rust service mesh has lazy_static registration" \
    "swipl -g \"use_module('src/unifyweaver/targets/rust_target'), compile_service_mesh_rust(service(gateway, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]), Code), sub_string(Code, _, _, _, 'lazy_static'), halt(0)\" -t \"halt(1)\""

# ==============================================================================
# CROSS-TARGET CONSISTENCY TESTS
# ==============================================================================
echo ""
echo "--- Cross-Target Consistency Tests ---"

run_test "All targets produce CircuitState enum" \
    "swipl -g \"
        use_module('src/unifyweaver/targets/python_target'),
        use_module('src/unifyweaver/targets/go_target'),
        use_module('src/unifyweaver/targets/rust_target'),
        Svc = service(test, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]),
        compile_service_mesh_python(Svc, PyCode), sub_string(PyCode, _, _, _, 'CircuitState'),
        compile_service_mesh_go(Svc, GoCode), sub_string(GoCode, _, _, _, 'CircuitState'),
        compile_service_mesh_rust(Svc, RsCode), sub_string(RsCode, _, _, _, 'CircuitState'),
        halt(0)\" -t \"halt(1)\""

run_test "All targets have backends list" \
    "swipl -g \"
        use_module('src/unifyweaver/targets/python_target'),
        use_module('src/unifyweaver/targets/go_target'),
        use_module('src/unifyweaver/targets/rust_target'),
        Svc = service(test, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]),
        compile_service_mesh_python(Svc, PyCode), sub_string(PyCode, _, _, _, 'backends'),
        compile_service_mesh_go(Svc, GoCode), sub_string(GoCode, _, _, _, 'backends'),
        compile_service_mesh_rust(Svc, RsCode), sub_string(RsCode, _, _, _, 'backends'),
        halt(0)\" -t \"halt(1)\""

run_test "All targets have select_backend method" \
    "swipl -g \"
        use_module('src/unifyweaver/targets/python_target'),
        use_module('src/unifyweaver/targets/go_target'),
        use_module('src/unifyweaver/targets/rust_target'),
        Svc = service(test, [load_balance(round_robin), circuit_breaker(threshold(5), timeout(30000)), retry(3, exponential)], [receive(X), respond(X)]),
        compile_service_mesh_python(Svc, PyCode), sub_string(PyCode, _, _, _, 'select_backend'),
        compile_service_mesh_go(Svc, GoCode), sub_string(GoCode, _, _, _, 'selectBackend'),
        compile_service_mesh_rust(Svc, RsCode), sub_string(RsCode, _, _, _, 'select_backend'),
        halt(0)\" -t \"halt(1)\""

# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================
echo ""
echo "--- Edge Case Tests ---"

run_test "Service mesh with only load_balance" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_service_mesh_service(service(test, [load_balance(round_robin)], [])), halt(0)\" -t \"halt(1)\""

run_test "Service mesh with only circuit_breaker" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_service_mesh_service(service(test, [circuit_breaker(threshold(5), timeout(30000))], [])), halt(0)\" -t \"halt(1)\""

run_test "Service mesh with only retry" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), is_service_mesh_service(service(test, [retry(3, fixed)], [])), halt(0)\" -t \"halt(1)\""

run_test "Default circuit breaker config when none specified" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_circuit_breaker_config(service(test, [], []), none), halt(0)\" -t \"halt(1)\""

run_test "Default retry config when none specified" \
    "swipl -g \"use_module('src/unifyweaver/core/service_validation'), get_retry_config(service(test, [], []), none), halt(0)\" -t \"halt(1)\""

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
