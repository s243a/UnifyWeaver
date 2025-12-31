#!/bin/bash
# Test suite for Phase 8: Service Tracing
# Tests distributed tracing compilation for Python, Go, and Rust

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
echo "Phase 8: Service Tracing Test Suite"
echo "=========================================="
echo ""

# ==========================================
# TEST GROUP 1: Validation Tests
# ==========================================
echo -e "${YELLOW}[1/4] Validation Tests${NC}"

# Test 1.1: Valid trace exporters
cat > "$TMPDIR/test_1_1.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    ( is_valid_trace_exporter(otlp),
      is_valid_trace_exporter(jaeger),
      is_valid_trace_exporter(zipkin),
      is_valid_trace_exporter(console) -> halt(0) ; halt(1) ).
EOF
run_test "valid trace exporters" "$TMPDIR/test_1_1.pl"

# Test 1.2: Valid trace propagation formats
cat > "$TMPDIR/test_1_2.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    ( is_valid_trace_propagation(w3c),
      is_valid_trace_propagation(b3),
      is_valid_trace_propagation(jaeger) -> halt(0) ; halt(1) ).
EOF
run_test "valid trace propagation" "$TMPDIR/test_1_2.pl"

# Test 1.3: Tracing service options
cat > "$TMPDIR/test_1_3.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Options = [
        tracing(true),
        trace_exporter(otlp),
        trace_sampling(0.1),
        trace_propagation(w3c)
    ],
    ( maplist(is_valid_service_option, Options) -> halt(0) ; halt(1) ).
EOF
run_test "tracing service options" "$TMPDIR/test_1_3.pl"

# Test 1.4: is_tracing_enabled helper
cat > "$TMPDIR/test_1_4.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service = service(test, [tracing(true)], [receive(_X), respond(_X)]),
    ( is_tracing_enabled(Service) -> halt(0) ; halt(1) ).
EOF
run_test "is_tracing_enabled helper" "$TMPDIR/test_1_4.pl"

# Test 1.5: get_trace_exporter helper
cat > "$TMPDIR/test_1_5.pl" << 'EOF'
:- use_module('src/unifyweaver/core/service_validation').
:- initialization(main, main).
main :-
    Service = service(test, [trace_exporter(jaeger)], [receive(_X), respond(_X)]),
    ( get_trace_exporter(Service, jaeger) -> halt(0) ; halt(1) ).
EOF
run_test "get_trace_exporter helper" "$TMPDIR/test_1_5.pl"

echo ""

# ==========================================
# TEST GROUP 2: Python Tracing Compilation
# ==========================================
echo -e "${YELLOW}[2/4] Python Tracing Compilation Tests${NC}"

# Test 2.1: Basic tracing service
cat > "$TMPDIR/test_2_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "Tracer") -> halt(0) ; halt(1) ).
EOF
run_test "Python tracing basic" "$TMPDIR/test_2_1.pl"

# Test 2.2: SpanContext class
cat > "$TMPDIR/test_2_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "SpanContext") -> halt(0) ; halt(1) ).
EOF
run_test "Python SpanContext class" "$TMPDIR/test_2_2.pl"

# Test 2.3: Span class
cat > "$TMPDIR/test_2_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "class Span") -> halt(0) ; halt(1) ).
EOF
run_test "Python Span class" "$TMPDIR/test_2_3.pl"

# Test 2.4: Trace exporter
cat > "$TMPDIR/test_2_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(api, [trace_exporter(otlp)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "SpanExporter") -> halt(0) ; halt(1) ).
EOF
run_test "Python SpanExporter" "$TMPDIR/test_2_4.pl"

# Test 2.5: Sampling rate
cat > "$TMPDIR/test_2_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/python_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true), trace_sampling(0.5)], [receive(_X), respond(_X)]),
    ( compile_service_to_python(Service, Code),
      sub_string(Code, _, _, _, "sampling_rate") -> halt(0) ; halt(1) ).
EOF
run_test "Python sampling rate" "$TMPDIR/test_2_5.pl"

echo ""

# ==========================================
# TEST GROUP 3: Go Tracing Compilation
# ==========================================
echo -e "${YELLOW}[3/4] Go Tracing Compilation Tests${NC}"

# Test 3.1: Basic Go tracing service
cat > "$TMPDIR/test_3_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "Tracer") -> halt(0) ; halt(1) ).
EOF
run_test "Go tracing basic" "$TMPDIR/test_3_1.pl"

# Test 3.2: Go SpanContext
cat > "$TMPDIR/test_3_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "SpanContext") -> halt(0) ; halt(1) ).
EOF
run_test "Go SpanContext" "$TMPDIR/test_3_2.pl"

# Test 3.3: Go Span struct
cat > "$TMPDIR/test_3_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "type Span struct") -> halt(0) ; halt(1) ).
EOF
run_test "Go Span struct" "$TMPDIR/test_3_3.pl"

# Test 3.4: Go SpanExporter interface
cat > "$TMPDIR/test_3_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(api, [trace_exporter(jaeger)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "SpanExporter") -> halt(0) ; halt(1) ).
EOF
run_test "Go SpanExporter interface" "$TMPDIR/test_3_4.pl"

# Test 3.5: Go context propagation
cat > "$TMPDIR/test_3_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/go_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_go(Service, Code),
      sub_string(Code, _, _, _, "context.Context") -> halt(0) ; halt(1) ).
EOF
run_test "Go context propagation" "$TMPDIR/test_3_5.pl"

echo ""

# ==========================================
# TEST GROUP 4: Rust Tracing Compilation
# ==========================================
echo -e "${YELLOW}[4/4] Rust Tracing Compilation Tests${NC}"

# Test 4.1: Basic Rust tracing service
cat > "$TMPDIR/test_4_1.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "Tracer") -> halt(0) ; halt(1) ).
EOF
run_test "Rust tracing basic" "$TMPDIR/test_4_1.pl"

# Test 4.2: Rust SpanContext
cat > "$TMPDIR/test_4_2.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "SpanContext") -> halt(0) ; halt(1) ).
EOF
run_test "Rust SpanContext" "$TMPDIR/test_4_2.pl"

# Test 4.3: Rust Span struct
cat > "$TMPDIR/test_4_3.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "struct Span") -> halt(0) ; halt(1) ).
EOF
run_test "Rust Span struct" "$TMPDIR/test_4_3.pl"

# Test 4.4: Rust SpanExporter trait
cat > "$TMPDIR/test_4_4.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(api, [trace_exporter(zipkin)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "trait SpanExporter") -> halt(0) ; halt(1) ).
EOF
run_test "Rust SpanExporter trait" "$TMPDIR/test_4_4.pl"

# Test 4.5: Rust SpanKind enum
cat > "$TMPDIR/test_4_5.pl" << 'EOF'
:- use_module('src/unifyweaver/targets/rust_target').
:- initialization(main, main).
main :-
    Service = service(api, [tracing(true)], [receive(_X), respond(_X)]),
    ( compile_service_to_rust(Service, Code),
      sub_string(Code, _, _, _, "SpanKind") -> halt(0) ; halt(1) ).
EOF
run_test "Rust SpanKind enum" "$TMPDIR/test_4_5.pl"

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
