#!/bin/bash
# test_ironpython_enhanced_chaining.sh - End-to-end tests for IronPython enhanced pipeline chaining
# Tests enhanced chaining (fan-out, merge, routing, filter) with .NET CLR integration

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_ironpython_enhanced_chaining_test"
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

# Test 1: IronPython unit tests pass
test_ironpython_unit_tests() {
    log_info "Test 1: IronPython enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/python_target), test_ironpython_enhanced_chaining" -t halt 2>&1 | grep -q "All IronPython Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "IronPython unit tests pass"
    else
        log_fail "IronPython unit tests failed"
    fi
}

# Test 2: IronPython generates valid code with fan-out
test_ironpython_fanout_code_generation() {
    log_info "Test 2: IronPython fan-out code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ironpython_fanout_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_ironpython_enhanced_pipeline([
        extract/1,
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(fanout_pipe)], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_enhanced_chaining_test/ironpython_fanout.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_enhanced_chaining_test/ironpython_fanout_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ironpython_fanout.py" ]; then
            if grep -q "fan_out_records" "$OUTPUT_DIR/ironpython_fanout.py" && \
               grep -q "Fan-out to 2" "$OUTPUT_DIR/ironpython_fanout.py" && \
               grep -q "List\[object\]" "$OUTPUT_DIR/ironpython_fanout.py"; then
                log_pass "IronPython fan-out code generated correctly"
            else
                log_fail "IronPython fan-out code missing expected patterns"
            fi
        else
            log_fail "IronPython fan-out file not generated"
        fi
    else
        log_fail "IronPython fan-out compilation failed"
    fi
}

# Test 3: IronPython generates valid code with parallel (.NET Tasks)
test_ironpython_parallel_code_generation() {
    log_info "Test 3: IronPython parallel code generation (.NET Tasks)"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ironpython_parallel_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_ironpython_enhanced_pipeline([
        extract/1,
        parallel([validate/1, enrich/1, audit/1]),
        merge,
        output/1
    ], [pipeline_name(parallel_pipe)], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_enhanced_chaining_test/ironpython_parallel.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_enhanced_chaining_test/ironpython_parallel_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ironpython_parallel.py" ]; then
            if grep -q "parallel_records" "$OUTPUT_DIR/ironpython_parallel.py" && \
               grep -q "Parallel execution of 3 stages" "$OUTPUT_DIR/ironpython_parallel.py" && \
               grep -q "concurrent via .NET Tasks" "$OUTPUT_DIR/ironpython_parallel.py"; then
                log_pass "IronPython parallel code generated correctly"
            else
                log_fail "IronPython parallel code missing expected patterns"
            fi
        else
            log_fail "IronPython parallel file not generated"
        fi
    else
        log_fail "IronPython parallel compilation failed"
    fi
}

# Test 4: IronPython generates valid code with filter
test_ironpython_filter_code_generation() {
    log_info "Test 4: IronPython filter code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ironpython_filter_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_ironpython_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        output/1
    ], [pipeline_name(filter_pipe)], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_enhanced_chaining_test/ironpython_filter.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_enhanced_chaining_test/ironpython_filter_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ironpython_filter.py" ]; then
            if grep -q "Filter by is_active" "$OUTPUT_DIR/ironpython_filter.py" && \
               grep -q "filter_records" "$OUTPUT_DIR/ironpython_filter.py"; then
                log_pass "IronPython filter code generated correctly"
            else
                log_fail "IronPython filter code missing expected patterns"
            fi
        else
            log_fail "IronPython filter file not generated"
        fi
    else
        log_fail "IronPython filter compilation failed"
    fi
}

# Test 5: IronPython generates valid code with routing
test_ironpython_routing_code_generation() {
    log_info "Test 5: IronPython routing code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ironpython_routing_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_ironpython_enhanced_pipeline([
        extract/1,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(routing_pipe)], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_enhanced_chaining_test/ironpython_routing.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_enhanced_chaining_test/ironpython_routing_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ironpython_routing.py" ]; then
            if grep -q "Conditional routing" "$OUTPUT_DIR/ironpython_routing.py" && \
               grep -q "route_map" "$OUTPUT_DIR/ironpython_routing.py" && \
               grep -q "route_record" "$OUTPUT_DIR/ironpython_routing.py"; then
                log_pass "IronPython routing code generated correctly"
            else
                log_fail "IronPython routing code missing expected patterns"
            fi
        else
            log_fail "IronPython routing file not generated"
        fi
    else
        log_fail "IronPython routing compilation failed"
    fi
}

# Test 6: Complex IronPython pipeline with all patterns
test_ironpython_complex_pipeline() {
    log_info "Test 6: Complex IronPython pipeline with all patterns"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ironpython_complex_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_ironpython_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(complex_pipe)], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_enhanced_chaining_test/ironpython_complex.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_enhanced_chaining_test/ironpython_complex_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ironpython_complex.py" ]; then
            if grep -q "Fan-out to 3" "$OUTPUT_DIR/ironpython_complex.py" && \
               grep -q "Conditional routing" "$OUTPUT_DIR/ironpython_complex.py" && \
               grep -q "Filter by is_active" "$OUTPUT_DIR/ironpython_complex.py" && \
               grep -q "Merge" "$OUTPUT_DIR/ironpython_complex.py"; then
                log_pass "Complex IronPython pipeline has all patterns"
            else
                log_fail "Complex IronPython pipeline missing patterns"
            fi
        else
            log_fail "Complex IronPython file not generated"
        fi
    else
        log_fail "Complex IronPython compilation failed"
    fi
}

# Test 7: IronPython helpers include .NET collections and parallel support
test_ironpython_helpers_dotnet() {
    log_info "Test 7: IronPython helpers use .NET collections and parallel"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ironpython_helpers_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_helpers :-
    ironpython_enhanced_helpers(Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_enhanced_chaining_test/ironpython_helpers.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_enhanced_chaining_test/ironpython_helpers_test.pl'), test_helpers" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ironpython_helpers.py" ]; then
            if grep -q "List\[object\]" "$OUTPUT_DIR/ironpython_helpers.py" && \
               grep -q "Dictionary\[object, object\]" "$OUTPUT_DIR/ironpython_helpers.py" && \
               grep -q "to_dotnet_list" "$OUTPUT_DIR/ironpython_helpers.py" && \
               grep -q "from_dotnet_list" "$OUTPUT_DIR/ironpython_helpers.py" && \
               grep -q "parallel_records" "$OUTPUT_DIR/ironpython_helpers.py" && \
               grep -q "ConcurrentBag" "$OUTPUT_DIR/ironpython_helpers.py"; then
                log_pass "IronPython helpers use .NET collections and parallel"
            else
                log_fail "IronPython helpers missing .NET patterns"
            fi
        else
            log_fail "IronPython helpers file not generated"
        fi
    else
        log_fail "IronPython helpers compilation failed"
    fi
}

# Test 8: IronPython generates proper shebang and CLR imports
test_ironpython_header() {
    log_info "Test 8: IronPython header structure"

    if [ -f "$OUTPUT_DIR/ironpython_complex.py" ]; then
        if grep -q "#!/usr/bin/env ipy" "$OUTPUT_DIR/ironpython_complex.py" && \
           grep -q "import clr" "$OUTPUT_DIR/ironpython_complex.py" && \
           grep -q "clr.AddReference" "$OUTPUT_DIR/ironpython_complex.py"; then
            log_pass "IronPython has proper header with CLR imports"
        else
            log_fail "IronPython missing proper header"
        fi
    else
        log_fail "Complex IronPython file not available for header test"
    fi
}

# Test 9: IronPython code has proper main block
test_ironpython_main_block() {
    log_info "Test 9: IronPython main block"

    if [ -f "$OUTPUT_DIR/ironpython_complex.py" ]; then
        if grep -q '__name__ == "__main__"' "$OUTPUT_DIR/ironpython_complex.py" && \
           grep -q "read_input()" "$OUTPUT_DIR/ironpython_complex.py" && \
           grep -q "write_output" "$OUTPUT_DIR/ironpython_complex.py"; then
            log_pass "IronPython has proper main block"
        else
            log_fail "IronPython missing main block"
        fi
    else
        log_fail "Complex IronPython file not available for main block test"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  IronPython Enhanced Chaining E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_ironpython_unit_tests
    test_ironpython_fanout_code_generation
    test_ironpython_parallel_code_generation
    test_ironpython_filter_code_generation
    test_ironpython_routing_code_generation
    test_ironpython_complex_pipeline
    test_ironpython_helpers_dotnet
    test_ironpython_header
    test_ironpython_main_block

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
