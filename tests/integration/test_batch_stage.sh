#!/bin/bash
# test_batch_stage.sh - Integration tests for batch(N) and unbatch stages
# Tests batch stage compilation across multiple targets

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

# Test 1: Pipeline validation recognizes batch(N) and unbatch
test_validation() {
    log_info "Test 1: Pipeline validation recognizes batch(N) and unbatch"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/core/pipeline_validation), test_pipeline_validation" -t halt 2>&1 | grep -q "All Pipeline Validation Tests Passed"; then
        log_pass "Pipeline validation tests pass (includes batch tests)"
    else
        log_fail "Pipeline validation tests failed"
    fi
}

# Test 2: Python batch stage compilation
test_python_batch() {
    log_info "Test 2: Python batch stage compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([
            extract/1,
            batch(100),
            process_batch/1,
            unbatch,
            output/1
        ], [pipeline_name(batch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "batch_records" && \
       echo "$output" | grep -q "unbatch_records" && \
       echo "$output" | grep -q "Batch records into groups of 100"; then
        log_pass "Python batch stage compiles correctly"
    else
        log_fail "Python batch stage compilation failed"
    fi
}

# Test 3: Go batch stage compilation
test_go_batch() {
    log_info "Test 3: Go batch stage compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([
            extract/1,
            batch(50),
            process/1,
            unbatch,
            output/1
        ], [pipeline_name(batchPipe)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "batchRecords" && \
       echo "$output" | grep -q "unbatchRecords" && \
       echo "$output" | grep -q "Batch records into groups of 50"; then
        log_pass "Go batch stage compiles correctly"
    else
        log_fail "Go batch stage compilation failed"
    fi
}

# Test 4: Rust batch stage compilation
test_rust_batch() {
    log_info "Test 4: Rust batch stage compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([
            extract/1,
            batch(200),
            bulk_process/1,
            unbatch,
            output/1
        ], [pipeline_name(batch_pipe)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "batch_records" && \
       echo "$output" | grep -q "unbatch_records" && \
       echo "$output" | grep -q "Batch records into groups of 200"; then
        log_pass "Rust batch stage compiles correctly"
    else
        log_fail "Rust batch stage compilation failed"
    fi
}

# Test 5: Bash batch stage compilation
test_bash_batch() {
    log_info "Test 5: Bash batch stage compilation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/bash_target),
        compile_bash_enhanced_pipeline([
            extract/1,
            batch(10),
            process/1,
            unbatch,
            output/1
        ], [pipeline_name(batch_pipe)], Code),
        format('~w', [Code])
    " -t halt 2>&1)

    if echo "$output" | grep -q "batch_records" && \
       echo "$output" | grep -q "unbatch_records" && \
       echo "$output" | grep -q "Batch records into groups of 10"; then
        log_pass "Bash batch stage compiles correctly"
    else
        log_fail "Bash batch stage compilation failed"
    fi
}

# Test 6: Invalid batch size validation
test_invalid_batch_size() {
    log_info "Test 6: Invalid batch size validation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, batch(0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)

    if echo "$output" | grep -q "invalid_batch_size"; then
        log_pass "Invalid batch size is detected"
    else
        log_fail "Invalid batch size not detected"
    fi
}

# Test 7: Negative batch size validation
test_negative_batch_size() {
    log_info "Test 7: Negative batch size validation"

    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, batch(-5), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)

    if echo "$output" | grep -q "invalid_batch_size"; then
        log_pass "Negative batch size is detected"
    else
        log_fail "Negative batch size not detected"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Batch Stage Integration Tests"
    echo "=========================================="
    echo ""

    test_validation
    test_python_batch
    test_go_batch
    test_rust_batch
    test_bash_batch
    test_invalid_batch_size
    test_negative_batch_size

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
