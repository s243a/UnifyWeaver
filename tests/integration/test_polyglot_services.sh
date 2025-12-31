#!/bin/bash
# Test suite for Phase 5: Polyglot Services
# Tests cross-language service compilation for Python, Go, and Rust

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
echo "Phase 5: Polyglot Services Test Suite"
echo "=========================================="
echo ""

# ==========================================
# TEST GROUP 1: Validation Tests
# ==========================================
echo -e "${YELLOW}[1/4] Validation Tests${NC}"

# Test 1.1: Valid target language
cat > "$TMPDIR/test_1_1.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    ( is_valid_target_language(python),
      is_valid_target_language(go),
      is_valid_target_language(rust),
      is_valid_target_language(javascript),
      is_valid_target_language(csharp) -> halt(0) ; halt(1) ).
EOF
run_test "valid target languages" "$TMPDIR/test_1_1.pl"

# Test 1.2: Valid service dependency
cat > "$TMPDIR/test_1_2.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Dep = dep(user_service, python, tcp('localhost', 8001)),
    ( is_valid_service_dependency(Dep) -> halt(0) ; halt(1) ).
EOF
run_test "valid service dependency" "$TMPDIR/test_1_2.pl"

# Test 1.3: Valid polyglot service options
cat > "$TMPDIR/test_1_3.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Options = [
        polyglot(true),
        target_language(python),
        depends_on([dep(other_service, go, tcp('localhost', 8080))])
    ],
    ( maplist(is_valid_service_option, Options) -> halt(0) ; halt(1) ).
EOF
run_test "polyglot service validation" "$TMPDIR/test_1_3.pl"

# Test 1.4: Get target language helper
cat > "$TMPDIR/test_1_4.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service = service(test, [target_language(python)], [receive(_X), respond(_X)]),
    ( get_target_language(Service, python) -> halt(0) ; halt(1) ).
EOF
run_test "get_target_language helper" "$TMPDIR/test_1_4.pl"

# Test 1.5: Get service dependencies helper
cat > "$TMPDIR/test_1_5.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service = service(test, [depends_on([dep(a, python, tcp(h, 1))])], [receive(_X), respond(_X)]),
    ( get_service_dependencies(Service, Deps), Deps = [dep(a, python, tcp(h, 1))] -> halt(0) ; halt(1) ).
EOF
run_test "get_service_dependencies helper" "$TMPDIR/test_1_5.pl"

echo ""

# ==========================================
# TEST GROUP 2: Python Polyglot Compilation
# ==========================================
echo -e "${YELLOW}[2/4] Python Polyglot Compilation Tests${NC}"

# Test 2.1: Basic polyglot service
cat > "$TMPDIR/test_2_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(gateway, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "ServiceRegistry") -> halt(0) ; halt(1) ).
EOF
run_test "Python polyglot basic" "$TMPDIR/test_2_1.pl"

# Test 2.2: Polyglot with dependencies
cat > "$TMPDIR/test_2_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(api, [
        polyglot(true),
        depends_on([
            dep(user_svc, go, tcp(localhost, 8001)),
            dep(order_svc, rust, tcp(localhost, 8002))
        ])
    ], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "user_svc"),
      sub_string(Code, _, _, _, "order_svc") -> halt(0) ; halt(1) ).
EOF
run_test "Python polyglot with deps" "$TMPDIR/test_2_2.pl"

# Test 2.3: Python ServiceClient class
cat > "$TMPDIR/test_2_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "class ServiceClient"),
      sub_string(Code, _, _, _, "urllib.request") -> halt(0) ; halt(1) ).
EOF
run_test "Python ServiceClient class" "$TMPDIR/test_2_3.pl"

# Test 2.4: Python call_service method
cat > "$TMPDIR/test_2_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "call_service") -> halt(0) ; halt(1) ).
EOF
run_test "Python call_service method" "$TMPDIR/test_2_4.pl"

# Test 2.5: Python target language attribute
cat > "$TMPDIR/test_2_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true), target_language(python)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "Target language: python") -> halt(0) ; halt(1) ).
EOF
run_test "Python target language" "$TMPDIR/test_2_5.pl"

echo ""

# ==========================================
# TEST GROUP 3: Go Polyglot Compilation
# ==========================================
echo -e "${YELLOW}[3/4] Go Polyglot Compilation Tests${NC}"

# Test 3.1: Basic Go polyglot service
cat > "$TMPDIR/test_3_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(gateway, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "ServiceRegistry") -> halt(0) ; halt(1) ).
EOF
run_test "Go polyglot basic" "$TMPDIR/test_3_1.pl"

# Test 3.2: Go polyglot with dependencies
cat > "$TMPDIR/test_3_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(api, [
        polyglot(true),
        depends_on([
            dep(user_svc, python, tcp(localhost, 8001)),
            dep(order_svc, rust, tcp(localhost, 8002))
        ])
    ], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "user_svc"),
      sub_string(Code, _, _, _, "order_svc") -> halt(0) ; halt(1) ).
EOF
run_test "Go polyglot with deps" "$TMPDIR/test_3_2.pl"

# Test 3.3: Go ServiceClient struct
cat > "$TMPDIR/test_3_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "type ServiceClient struct") -> halt(0) ; halt(1) ).
EOF
run_test "Go ServiceClient struct" "$TMPDIR/test_3_3.pl"

# Test 3.4: Go sync.RWMutex for thread safety
cat > "$TMPDIR/test_3_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "sync.RWMutex") -> halt(0) ; halt(1) ).
EOF
run_test "Go thread safety" "$TMPDIR/test_3_4.pl"

# Test 3.5: Go CallService method
cat > "$TMPDIR/test_3_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "CallService") -> halt(0) ; halt(1) ).
EOF
run_test "Go CallService method" "$TMPDIR/test_3_5.pl"

echo ""

# ==========================================
# TEST GROUP 4: Rust Polyglot Compilation
# ==========================================
echo -e "${YELLOW}[4/4] Rust Polyglot Compilation Tests${NC}"

# Test 4.1: Basic Rust polyglot service
cat > "$TMPDIR/test_4_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(gateway, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "ServiceRegistry") -> halt(0) ; halt(1) ).
EOF
run_test "Rust polyglot basic" "$TMPDIR/test_4_1.pl"

# Test 4.2: Rust polyglot with dependencies
cat > "$TMPDIR/test_4_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(api, [
        polyglot(true),
        depends_on([
            dep(user_svc, python, tcp(localhost, 8001)),
            dep(order_svc, go, tcp(localhost, 8002))
        ])
    ], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "user_svc"),
      sub_string(Code, _, _, _, "order_svc") -> halt(0) ; halt(1) ).
EOF
run_test "Rust polyglot with deps" "$TMPDIR/test_4_2.pl"

# Test 4.3: Rust ServiceClient struct
cat > "$TMPDIR/test_4_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "pub struct ServiceClient") -> halt(0) ; halt(1) ).
EOF
run_test "Rust ServiceClient struct" "$TMPDIR/test_4_3.pl"

# Test 4.4: Rust RwLock for thread safety
cat > "$TMPDIR/test_4_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "RwLock") -> halt(0) ; halt(1) ).
EOF
run_test "Rust thread safety" "$TMPDIR/test_4_4.pl"

# Test 4.5: Rust reqwest client
cat > "$TMPDIR/test_4_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "reqwest") -> halt(0) ; halt(1) ).
EOF
run_test "Rust reqwest HTTP client" "$TMPDIR/test_4_5.pl"

# Test 4.6: Rust call_service method
cat > "$TMPDIR/test_4_6.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "call_service") -> halt(0) ; halt(1) ).
EOF
run_test "Rust call_service method" "$TMPDIR/test_4_6.pl"

# Test 4.7: Rust lazy_static for registry
cat > "$TMPDIR/test_4_7.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(test, [polyglot(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "lazy_static"),
      sub_string(Code, _, _, _, "SERVICE_REGISTRY") -> halt(0) ; halt(1) ).
EOF
run_test "Rust lazy_static registry" "$TMPDIR/test_4_7.pl"

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
