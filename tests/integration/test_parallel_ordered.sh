#!/bin/bash
# test_parallel_ordered.sh - Integration tests for parallel(Stages, [ordered(true)]) option
# Tests ordered parallel compilation across multiple targets

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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

# Test 1: Pipeline validation recognizes parallel(Stages, Options)
test_validation_valid() {
    log_info "Test 1: Pipeline validation accepts parallel(Stages, [ordered(true)])"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([a/1, b/1], [ordered(true)]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)

    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Valid ordered parallel accepted"
    else
        log_fail "Valid ordered parallel rejected: $output"
    fi
}

# Test 2: Pipeline validation detects invalid options
test_validation_invalid_option() {
    log_info "Test 2: Pipeline validation detects invalid parallel options"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([a/1, b/1], [invalid_option]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)

    if echo "$output" | grep -q "invalid_parallel_option"; then
        log_pass "Invalid parallel option detected"
    else
        log_fail "Invalid parallel option not detected: $output"
    fi
}

# Test 3: Python ordered parallel compilation
test_python_ordered() {
    log_info "Test 3: Python ordered parallel compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([
            parse/1,
            parallel([process_a/1, process_b/1], [ordered(true)]),
            output/1
        ], [pipeline_name(ordered_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "parallel_records_ordered" && \
       echo "$output" | grep -q "indexed_results" && \
       echo "$output" | grep -q "Parallel execution (ordered)"; then
        log_pass "Python ordered parallel compiles correctly"
    else
        log_fail "Python ordered parallel compilation failed"
    fi
}

# Test 4: Python unordered parallel (default) compilation
test_python_unordered() {
    log_info "Test 4: Python unordered parallel (default) compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([
            parse/1,
            parallel([process_a/1, process_b/1]),
            output/1
        ], [pipeline_name(unordered_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    # Should use parallel_records (not ordered version)
    if echo "$output" | grep -q "parallel_records(record, \[process_a, process_b\])" && \
       ! echo "$output" | grep -q "parallel_records_ordered(record, \[process_a, process_b\])"; then
        log_pass "Python unordered parallel compiles correctly"
    else
        log_fail "Python unordered parallel compilation failed"
    fi
}

# Test 5: Go ordered parallel compilation
test_go_ordered() {
    log_info "Test 5: Go ordered parallel compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([
            parse/1,
            parallel([process_a/1, process_b/1], [ordered(true)]),
            output/1
        ], [pipeline_name(orderedTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "parallelRecordsOrdered" && \
       echo "$output" | grep -q "indexedResults" && \
       echo "$output" | grep -q "Parallel execution (ordered)"; then
        log_pass "Go ordered parallel compiles correctly"
    else
        log_fail "Go ordered parallel compilation failed"
    fi
}

# Test 6: Rust ordered parallel compilation
test_rust_ordered() {
    log_info "Test 6: Rust ordered parallel compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([
            parse/1,
            parallel([process_a/1, process_b/1], [ordered(true)]),
            output/1
        ], [pipeline_name(ordered_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "parallel_records_ordered" && \
       echo "$output" | grep -q "indexed_results" && \
       echo "$output" | grep -q "Parallel execution (ordered)"; then
        log_pass "Rust ordered parallel compiles correctly"
    else
        log_fail "Rust ordered parallel compilation failed"
    fi
}

# Test 7: ordered(false) uses unordered version
test_ordered_false() {
    log_info "Test 7: ordered(false) uses unordered version"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([
            parse/1,
            parallel([process_a/1, process_b/1], [ordered(false)]),
            output/1
        ], [pipeline_name(explicit_unordered)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    # Should use parallel_records (not ordered version)
    if echo "$output" | grep -q "parallel_records(record, \[process_a, process_b\])" && \
       ! echo "$output" | grep -q "parallel_records_ordered(record, \[process_a, process_b\])"; then
        log_pass "ordered(false) correctly uses unordered version"
    else
        log_fail "ordered(false) did not use unordered version"
    fi
}

# Test 8: Empty options list uses default (unordered)
test_empty_options() {
    log_info "Test 8: Empty options list uses default (unordered)"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([
            parse/1,
            parallel([process_a/1, process_b/1], []),
            output/1
        ], [pipeline_name(empty_options)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    # Should use parallel_records (not ordered version)
    if echo "$output" | grep -q "parallel_records(record, \[process_a, process_b\])" && \
       ! echo "$output" | grep -q "parallel_records_ordered(record, \[process_a, process_b\])"; then
        log_pass "Empty options uses default unordered"
    else
        log_fail "Empty options did not use default unordered"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Parallel Result Ordering Tests"
    echo "=========================================="
    echo ""

    test_validation_valid
    test_validation_invalid_option
    test_python_ordered
    test_python_unordered
    test_go_ordered
    test_rust_ordered
    test_ordered_false
    test_empty_options

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
