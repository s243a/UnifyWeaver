#!/bin/bash
# test_enhanced_pipeline_chaining.sh - End-to-end tests for enhanced pipeline chaining
# Tests fan-out, merge, conditional routing, and filter stages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_enhanced_pipeline_test"
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

cleanup() {
    rm -rf "$OUTPUT_DIR"
}

setup() {
    cleanup
    mkdir -p "$OUTPUT_DIR"
}

# Test 1: Unit tests pass
test_unit_tests() {
    log_info "Test 1: Enhanced pipeline chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/python_target), test_enhanced_pipeline_chaining" -t halt 2>&1 | grep -q "All Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "All unit tests pass"
    else
        log_fail "Unit tests failed"
    fi
}

# Test 2: Compile enhanced pipeline with fan-out
test_fanout_pipeline() {
    log_info "Test 2: Fan-out pipeline compilation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/fanout_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_enhanced_pipeline([
        extract/1,
        fan_out([validate/1, enrich/1]),
        merge,
        transform/1
    ], [pipeline_name(fanout_pipeline)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_pipeline_test/fanout_pipeline.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_pipeline_test/fanout_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/fanout_pipeline.py" ]; then
            if grep -q "def fanout_pipeline" "$OUTPUT_DIR/fanout_pipeline.py" && \
               grep -q "fan_out_records" "$OUTPUT_DIR/fanout_pipeline.py" && \
               grep -q "Fan-out to 2 parallel stages" "$OUTPUT_DIR/fanout_pipeline.py"; then
                log_pass "Fan-out pipeline generates correct Python code"
            else
                log_fail "Missing fan-out patterns"
            fi
        else
            log_fail "fanout_pipeline.py not generated"
        fi
    else
        log_fail "Pipeline compilation failed"
    fi
}

# Test 3: Verify fan_out_records helper
test_fanout_helper() {
    log_info "Test 3: fan_out_records helper function"

    if [ -f "$OUTPUT_DIR/fanout_pipeline.py" ]; then
        if grep -q "def fan_out_records" "$OUTPUT_DIR/fanout_pipeline.py" && \
           grep -q "Send record to all stages" "$OUTPUT_DIR/fanout_pipeline.py"; then
            log_pass "fan_out_records helper present"
        else
            log_fail "fan_out_records helper missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 4: Verify merge_streams helper
test_merge_helper() {
    log_info "Test 4: merge_streams helper function"

    if [ -f "$OUTPUT_DIR/fanout_pipeline.py" ]; then
        if grep -q "def merge_streams" "$OUTPUT_DIR/fanout_pipeline.py" && \
           grep -q "Combine multiple streams" "$OUTPUT_DIR/fanout_pipeline.py"; then
            log_pass "merge_streams helper present"
        else
            log_fail "merge_streams helper missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 5: Compile pipeline with conditional routing
test_routing_pipeline() {
    log_info "Test 5: Conditional routing pipeline"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/routing_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_enhanced_pipeline([
        extract/1,
        route_by(has_error, [(true, error_handler/1), (false, success_handler/1)])
    ], [pipeline_name(routing_pipeline)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_pipeline_test/routing_pipeline.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_pipeline_test/routing_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/routing_pipeline.py" ]; then
            if grep -q "def routing_pipeline" "$OUTPUT_DIR/routing_pipeline.py" && \
               grep -q "Conditional routing" "$OUTPUT_DIR/routing_pipeline.py" && \
               grep -q "route_map" "$OUTPUT_DIR/routing_pipeline.py"; then
                log_pass "Routing pipeline generates correct Python code"
            else
                log_fail "Missing routing patterns"
            fi
        else
            log_fail "routing_pipeline.py not generated"
        fi
    else
        log_fail "Routing pipeline compilation failed"
    fi
}

# Test 6: Verify route_record helper
test_route_helper() {
    log_info "Test 6: route_record helper function"

    if [ -f "$OUTPUT_DIR/routing_pipeline.py" ]; then
        if grep -q "def route_record" "$OUTPUT_DIR/routing_pipeline.py" && \
           grep -q "Direct record to appropriate stage" "$OUTPUT_DIR/routing_pipeline.py"; then
            log_pass "route_record helper present"
        else
            log_fail "route_record helper missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 7: Compile pipeline with filter
test_filter_pipeline() {
    log_info "Test 7: Filter pipeline"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/filter_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_enhanced_pipeline([
        extract/1,
        filter_by(is_valid),
        transform/1
    ], [pipeline_name(filter_pipeline)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_pipeline_test/filter_pipeline.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_pipeline_test/filter_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/filter_pipeline.py" ]; then
            if grep -q "def filter_pipeline" "$OUTPUT_DIR/filter_pipeline.py" && \
               grep -q "Filter by is_valid" "$OUTPUT_DIR/filter_pipeline.py" && \
               grep -q "filter_records" "$OUTPUT_DIR/filter_pipeline.py"; then
                log_pass "Filter pipeline generates correct Python code"
            else
                log_fail "Missing filter patterns"
            fi
        else
            log_fail "filter_pipeline.py not generated"
        fi
    else
        log_fail "Filter pipeline compilation failed"
    fi
}

# Test 8: Verify filter_records helper
test_filter_helper() {
    log_info "Test 8: filter_records helper function"

    if [ -f "$OUTPUT_DIR/filter_pipeline.py" ]; then
        if grep -q "def filter_records" "$OUTPUT_DIR/filter_pipeline.py" && \
           grep -q "Only yield records that satisfy" "$OUTPUT_DIR/filter_pipeline.py"; then
            log_pass "filter_records helper present"
        else
            log_fail "filter_records helper missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 9: Verify tee_stream helper
test_tee_helper() {
    log_info "Test 9: tee_stream helper function"

    if [ -f "$OUTPUT_DIR/fanout_pipeline.py" ]; then
        if grep -q "def tee_stream" "$OUTPUT_DIR/fanout_pipeline.py" && \
           grep -q "Materialize to allow multiple iterations" "$OUTPUT_DIR/fanout_pipeline.py"; then
            log_pass "tee_stream helper present"
        else
            log_fail "tee_stream helper missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 10: Complex pipeline with all patterns
test_complex_pipeline() {
    log_info "Test 10: Complex pipeline with all patterns"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/complex_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(complex_pipeline)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_pipeline_test/complex_pipeline.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_pipeline_test/complex_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/complex_pipeline.py" ]; then
            if grep -q "def complex_pipeline" "$OUTPUT_DIR/complex_pipeline.py" && \
               grep -q "Fan-out to 3 parallel stages" "$OUTPUT_DIR/complex_pipeline.py" && \
               grep -q "Filter by is_active" "$OUTPUT_DIR/complex_pipeline.py" && \
               grep -q "Conditional routing" "$OUTPUT_DIR/complex_pipeline.py"; then
                log_pass "Complex pipeline generates all patterns"
            else
                log_fail "Missing patterns in complex pipeline"
            fi
        else
            log_fail "complex_pipeline.py not generated"
        fi
    else
        log_fail "Complex pipeline compilation failed"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Enhanced Pipeline Chaining E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_unit_tests
    test_fanout_pipeline
    test_fanout_helper
    test_merge_helper
    test_routing_pipeline
    test_route_helper
    test_filter_pipeline
    test_filter_helper
    test_tee_helper
    test_complex_pipeline

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    cleanup

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
