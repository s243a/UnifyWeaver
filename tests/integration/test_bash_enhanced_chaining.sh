#!/bin/bash
# test_bash_enhanced_chaining.sh - End-to-end tests for Bash enhanced pipeline chaining
# Tests enhanced chaining (fan-out, merge, routing, filter) for Bash target

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_bash_enhanced_chaining_test"
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

# Test 1: Bash unit tests pass
test_bash_unit_tests() {
    log_info "Test 1: Bash enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/bash_target), test_bash_enhanced_chaining" -t halt 2>&1 | grep -q "All Bash Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "Bash unit tests pass"
    else
        log_fail "Bash unit tests failed"
    fi
}

# Test 2: Bash generates valid code with fan-out
test_bash_fanout_code_generation() {
    log_info "Test 2: Bash fan-out code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/bash_fanout_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/bash_target).

test_compile :-
    compile_bash_enhanced_pipeline([
        extract/1,
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(fanout_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_bash_enhanced_chaining_test/bash_fanout.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_enhanced_chaining_test/bash_fanout_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/bash_fanout.sh" ]; then
            if grep -q "fan_out_records" "$OUTPUT_DIR/bash_fanout.sh" && \
               grep -q "Fan-out to 2" "$OUTPUT_DIR/bash_fanout.sh" && \
               grep -q "merge_streams" "$OUTPUT_DIR/bash_fanout.sh"; then
                log_pass "Bash fan-out code generated correctly"
            else
                log_fail "Bash fan-out code missing expected patterns"
            fi
        else
            log_fail "Bash fan-out file not generated"
        fi
    else
        log_fail "Bash fan-out compilation failed"
    fi
}

# Test 3: Bash generates valid code with filter
test_bash_filter_code_generation() {
    log_info "Test 3: Bash filter code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/bash_filter_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/bash_target).

test_compile :-
    compile_bash_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        output/1
    ], [pipeline_name(filter_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_bash_enhanced_chaining_test/bash_filter.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_enhanced_chaining_test/bash_filter_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/bash_filter.sh" ]; then
            if grep -q "Filter by is_active" "$OUTPUT_DIR/bash_filter.sh" && \
               grep -q "filter_record" "$OUTPUT_DIR/bash_filter.sh"; then
                log_pass "Bash filter code generated correctly"
            else
                log_fail "Bash filter code missing expected patterns"
            fi
        else
            log_fail "Bash filter file not generated"
        fi
    else
        log_fail "Bash filter compilation failed"
    fi
}

# Test 4: Bash generates valid code with routing
test_bash_routing_code_generation() {
    log_info "Test 4: Bash routing code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/bash_routing_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/bash_target).

test_compile :-
    compile_bash_enhanced_pipeline([
        extract/1,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(routing_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_bash_enhanced_chaining_test/bash_routing.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_enhanced_chaining_test/bash_routing_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/bash_routing.sh" ]; then
            if grep -q "Conditional routing" "$OUTPUT_DIR/bash_routing.sh" && \
               grep -q "ROUTE_MAP" "$OUTPUT_DIR/bash_routing.sh" && \
               grep -q "route_record" "$OUTPUT_DIR/bash_routing.sh"; then
                log_pass "Bash routing code generated correctly"
            else
                log_fail "Bash routing code missing expected patterns"
            fi
        else
            log_fail "Bash routing file not generated"
        fi
    else
        log_fail "Bash routing compilation failed"
    fi
}

# Test 5: Complex Bash pipeline with all patterns
test_bash_complex_pipeline() {
    log_info "Test 5: Complex Bash pipeline with all patterns"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/bash_complex_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/bash_target).

test_compile :-
    compile_bash_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(complex_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_bash_enhanced_chaining_test/bash_complex.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_enhanced_chaining_test/bash_complex_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/bash_complex.sh" ]; then
            if grep -q "Fan-out to 3" "$OUTPUT_DIR/bash_complex.sh" && \
               grep -q "Conditional routing" "$OUTPUT_DIR/bash_complex.sh" && \
               grep -q "Filter by is_active" "$OUTPUT_DIR/bash_complex.sh" && \
               grep -q "Merge" "$OUTPUT_DIR/bash_complex.sh"; then
                log_pass "Complex Bash pipeline has all patterns"
            else
                log_fail "Complex Bash pipeline missing patterns"
            fi
        else
            log_fail "Complex Bash file not generated"
        fi
    else
        log_fail "Complex Bash compilation failed"
    fi
}

# Test 6: Bash helpers include all required functions
test_bash_helpers_completeness() {
    log_info "Test 6: Bash helpers completeness"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/bash_helpers_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/bash_target).

test_helpers :-
    bash_enhanced_helpers(Code),
    atom_string(CodeAtom, Code),
    open('output_bash_enhanced_chaining_test/bash_helpers.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_enhanced_chaining_test/bash_helpers_test.pl'), test_helpers" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/bash_helpers.sh" ]; then
            local all_helpers=true
            for helper in "fan_out_records" "route_record" "filter_record" "merge_streams" "tee_stream" "parse_jsonl" "format_jsonl"; do
                if ! grep -q "$helper()" "$OUTPUT_DIR/bash_helpers.sh"; then
                    log_fail "Missing Bash helper: $helper"
                    all_helpers=false
                fi
            done
            if $all_helpers; then
                log_pass "All Bash helpers present"
            fi
        else
            log_fail "Bash helpers file not generated"
        fi
    else
        log_fail "Bash helpers compilation failed"
    fi
}

# Test 7: Bash generates proper shebang and header
test_bash_header() {
    log_info "Test 7: Bash header structure"

    if [ -f "$OUTPUT_DIR/bash_complex.sh" ]; then
        if grep -q "#!/bin/bash" "$OUTPUT_DIR/bash_complex.sh" && \
           grep -q "set -euo pipefail" "$OUTPUT_DIR/bash_complex.sh"; then
            log_pass "Bash has proper header"
        else
            log_fail "Bash missing proper header"
        fi
    else
        log_fail "Complex Bash file not available for header test"
    fi
}

# Test 8: Bash code has proper main function
test_bash_main_function() {
    log_info "Test 8: Bash main function"

    if [ -f "$OUTPUT_DIR/bash_complex.sh" ]; then
        if grep -q "main()" "$OUTPUT_DIR/bash_complex.sh" && \
           grep -q 'BASH_SOURCE\[0\]' "$OUTPUT_DIR/bash_complex.sh"; then
            log_pass "Bash has main function with proper entry point"
        else
            log_fail "Bash missing main function"
        fi
    else
        log_fail "Complex Bash file not available for main function test"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Bash Enhanced Chaining E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_bash_unit_tests
    test_bash_fanout_code_generation
    test_bash_filter_code_generation
    test_bash_routing_code_generation
    test_bash_complex_pipeline
    test_bash_helpers_completeness
    test_bash_header
    test_bash_main_function

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
