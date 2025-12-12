#!/bin/bash
# test_awk_enhanced_chaining.sh - End-to-end tests for AWK enhanced pipeline chaining
# Tests enhanced chaining (fan-out, merge, routing, filter) for AWK target

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_awk_enhanced_chaining_test"
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

# Test 1: AWK unit tests pass
test_awk_unit_tests() {
    log_info "Test 1: AWK enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/awk_target), test_awk_enhanced_chaining" -t halt 2>&1 | grep -q "All AWK Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "AWK unit tests pass"
    else
        log_fail "AWK unit tests failed"
    fi
}

# Test 2: AWK generates valid code with fan-out
test_awk_fanout_code_generation() {
    log_info "Test 2: AWK fan-out code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/awk_fanout_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/awk_target).

test_compile :-
    compile_awk_enhanced_pipeline([
        extract/1,
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(fanout_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_awk_enhanced_chaining_test/awk_fanout.awk', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_awk_enhanced_chaining_test/awk_fanout_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/awk_fanout.awk" ]; then
            if grep -q "fan_out_records" "$OUTPUT_DIR/awk_fanout.awk" && \
               grep -q "Fan-out to 2" "$OUTPUT_DIR/awk_fanout.awk" && \
               grep -q "Merge" "$OUTPUT_DIR/awk_fanout.awk"; then
                log_pass "AWK fan-out code generated correctly"
            else
                log_fail "AWK fan-out code missing expected patterns"
            fi
        else
            log_fail "AWK fan-out file not generated"
        fi
    else
        log_fail "AWK fan-out compilation failed"
    fi
}

# Test 3: AWK generates valid code with filter
test_awk_filter_code_generation() {
    log_info "Test 3: AWK filter code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/awk_filter_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/awk_target).

test_compile :-
    compile_awk_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        output/1
    ], [pipeline_name(filter_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_awk_enhanced_chaining_test/awk_filter.awk', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_awk_enhanced_chaining_test/awk_filter_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/awk_filter.awk" ]; then
            if grep -q "Filter by is_active" "$OUTPUT_DIR/awk_filter.awk" && \
               grep -q "filter_pipe" "$OUTPUT_DIR/awk_filter.awk"; then
                log_pass "AWK filter code generated correctly"
            else
                log_fail "AWK filter code missing expected patterns"
            fi
        else
            log_fail "AWK filter file not generated"
        fi
    else
        log_fail "AWK filter compilation failed"
    fi
}

# Test 4: AWK generates valid code with routing
test_awk_routing_code_generation() {
    log_info "Test 4: AWK routing code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/awk_routing_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/awk_target).

test_compile :-
    compile_awk_enhanced_pipeline([
        extract/1,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(routing_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_awk_enhanced_chaining_test/awk_routing.awk', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_awk_enhanced_chaining_test/awk_routing_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/awk_routing.awk" ]; then
            if grep -q "Conditional routing" "$OUTPUT_DIR/awk_routing.awk" && \
               grep -q "route_map" "$OUTPUT_DIR/awk_routing.awk" && \
               grep -q "route_record" "$OUTPUT_DIR/awk_routing.awk"; then
                log_pass "AWK routing code generated correctly"
            else
                log_fail "AWK routing code missing expected patterns"
            fi
        else
            log_fail "AWK routing file not generated"
        fi
    else
        log_fail "AWK routing compilation failed"
    fi
}

# Test 5: Complex AWK pipeline with all patterns
test_awk_complex_pipeline() {
    log_info "Test 5: Complex AWK pipeline with all patterns"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/awk_complex_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/awk_target).

test_compile :-
    compile_awk_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(has_error, [(true, error_log/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(complex_pipe), record_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_awk_enhanced_chaining_test/awk_complex.awk', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_awk_enhanced_chaining_test/awk_complex_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/awk_complex.awk" ]; then
            if grep -q "Fan-out to 3" "$OUTPUT_DIR/awk_complex.awk" && \
               grep -q "Conditional routing" "$OUTPUT_DIR/awk_complex.awk" && \
               grep -q "Filter by is_active" "$OUTPUT_DIR/awk_complex.awk" && \
               grep -q "Merge" "$OUTPUT_DIR/awk_complex.awk"; then
                log_pass "Complex AWK pipeline has all patterns"
            else
                log_fail "Complex AWK pipeline missing patterns"
            fi
        else
            log_fail "Complex AWK file not generated"
        fi
    else
        log_fail "Complex AWK compilation failed"
    fi
}

# Test 6: AWK helpers include all required functions
test_awk_helpers_completeness() {
    log_info "Test 6: AWK helpers completeness"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/awk_helpers_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/awk_target).

test_helpers :-
    awk_enhanced_helpers(Code),
    atom_string(CodeAtom, Code),
    open('output_awk_enhanced_chaining_test/awk_helpers.awk', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_awk_enhanced_chaining_test/awk_helpers_test.pl'), test_helpers" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/awk_helpers.awk" ]; then
            local all_helpers=true
            for helper in "fan_out_records" "route_record" "filter_record" "merge_streams" "tee_stream" "parse_jsonl" "format_jsonl"; do
                if ! grep -q "function $helper" "$OUTPUT_DIR/awk_helpers.awk"; then
                    log_fail "Missing AWK helper: $helper"
                    all_helpers=false
                fi
            done
            if $all_helpers; then
                log_pass "All AWK helpers present"
            fi
        else
            log_fail "AWK helpers file not generated"
        fi
    else
        log_fail "AWK helpers compilation failed"
    fi
}

# Test 7: AWK generates syntactically valid BEGIN block
test_awk_begin_block() {
    log_info "Test 7: AWK BEGIN block structure"

    if [ -f "$OUTPUT_DIR/awk_complex.awk" ]; then
        if grep -q "BEGIN {" "$OUTPUT_DIR/awk_complex.awk" && \
           grep -q "END {" "$OUTPUT_DIR/awk_complex.awk"; then
            log_pass "AWK has proper BEGIN/END blocks"
        else
            log_fail "AWK missing BEGIN/END blocks"
        fi
    else
        log_fail "Complex AWK file not available for BEGIN block test"
    fi
}

# Test 8: AWK code has proper main processing
test_awk_main_processing() {
    log_info "Test 8: AWK main processing block"

    if [ -f "$OUTPUT_DIR/awk_complex.awk" ]; then
        if grep -q "# Main processing" "$OUTPUT_DIR/awk_complex.awk" || \
           grep -q "complex_pipe" "$OUTPUT_DIR/awk_complex.awk"; then
            log_pass "AWK has main processing block"
        else
            log_fail "AWK missing main processing block"
        fi
    else
        log_fail "Complex AWK file not available for main processing test"
    fi
}

# Test 9: AWK GNU Parallel mode generates bash script
test_awk_gnu_parallel_mode() {
    log_info "Test 9: AWK GNU Parallel mode code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/awk_gnu_parallel_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/awk_target).

test_compile :-
    compile_awk_enhanced_pipeline([
        extract/1,
        parallel([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(gnu_parallel_pipe), parallel_mode(gnu_parallel)], Code),
    atom_string(CodeAtom, Code),
    open('output_awk_enhanced_chaining_test/awk_gnu_parallel.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_awk_enhanced_chaining_test/awk_gnu_parallel_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/awk_gnu_parallel.sh" ]; then
            if grep -q "#!/bin/bash" "$OUTPUT_DIR/awk_gnu_parallel.sh" && \
               grep -q "GNU Parallel" "$OUTPUT_DIR/awk_gnu_parallel.sh" && \
               grep -q "parallel --keep-order" "$OUTPUT_DIR/awk_gnu_parallel.sh" && \
               grep -q "stage_validate" "$OUTPUT_DIR/awk_gnu_parallel.sh" && \
               grep -q "stage_enrich" "$OUTPUT_DIR/awk_gnu_parallel.sh"; then
                log_pass "AWK GNU Parallel mode generates correct bash script"
            else
                log_fail "AWK GNU Parallel script missing expected patterns"
            fi
        else
            log_fail "AWK GNU Parallel file not generated"
        fi
    else
        log_fail "AWK GNU Parallel compilation failed"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  AWK Enhanced Chaining E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_awk_unit_tests
    test_awk_fanout_code_generation
    test_awk_filter_code_generation
    test_awk_routing_code_generation
    test_awk_complex_pipeline
    test_awk_helpers_completeness
    test_awk_begin_block
    test_awk_main_processing
    test_awk_gnu_parallel_mode

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
