#!/bin/bash
# test_merge_sorted_stage.sh - Integration tests for merge_sorted pipeline stage
# Tests: merge_sorted(Stages, Field), merge_sorted(Stages, Field, Dir)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PASS_COUNT=0
FAIL_COUNT=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# ============================================
# VALIDATION TESTS
# ============================================

# Test 1: Validation accepts merge_sorted(Stages, Field)
test_validation_merge_sorted() {
    log_info "Test 1: Validation accepts merge_sorted(Stages, Field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([sorted_a/1, sorted_b/1], timestamp), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "merge_sorted(Stages, Field) validation accepted"
    else
        log_fail "merge_sorted(Stages, Field) validation failed: $output"
    fi
}

# Test 2: Validation accepts merge_sorted(Stages, Field, Dir)
test_validation_merge_sorted_dir() {
    log_info "Test 2: Validation accepts merge_sorted(Stages, Field, Dir)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([sorted_a/1, sorted_b/1], timestamp, desc), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "merge_sorted(Stages, Field, desc) validation accepted"
    else
        log_fail "merge_sorted with direction validation failed: $output"
    fi
}

# Test 3: Validation rejects empty merge_sorted
test_validation_merge_sorted_empty() {
    log_info "Test 3: Validation rejects empty merge_sorted"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([], timestamp), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_merge_sorted\|invalid_stage\|empty"; then
        log_pass "Empty merge_sorted correctly rejected"
    else
        log_fail "Empty merge_sorted should be rejected: $output"
    fi
}

# Test 4: Validation rejects invalid direction
test_validation_merge_sorted_invalid_dir() {
    log_info "Test 4: Validation rejects invalid direction"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([a/1, b/1], timestamp, invalid), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid\|error"; then
        log_pass "Invalid direction correctly rejected"
    else
        log_fail "Invalid direction should be rejected: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 5: Python merge_sorted compilation (ascending)
test_python_merge_sorted() {
    log_info "Test 5: Python merge_sorted stage compilation (ascending)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, merge_sorted([stage_a/1, stage_b/1], timestamp), output/1], [pipeline_name(merge_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "merge_sorted_stage"; then
        log_pass "Python merge_sorted compiles correctly"
    else
        log_fail "Python merge_sorted compilation failed"
    fi
}

# Test 6: Python merge_sorted with direction
test_python_merge_sorted_desc() {
    log_info "Test 6: Python merge_sorted stage compilation (descending)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, merge_sorted([stage_a/1, stage_b/1], timestamp, desc), output/1], [pipeline_name(merge_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "reverse=True"; then
        log_pass "Python merge_sorted desc compiles correctly"
    else
        log_fail "Python merge_sorted desc compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 7: Go merge_sorted compilation (ascending)
test_go_merge_sorted() {
    log_info "Test 7: Go merge_sorted stage compilation (ascending)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, merge_sorted([stageA/1, stageB/1], timestamp), output/1], [pipeline_name(mergeTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "mergeSortedStage"; then
        log_pass "Go merge_sorted compiles correctly"
    else
        log_fail "Go merge_sorted compilation failed"
    fi
}

# Test 8: Go merge_sorted with direction
test_go_merge_sorted_desc() {
    log_info "Test 8: Go merge_sorted stage compilation (descending)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, merge_sorted([stageA/1, stageB/1], timestamp, desc), output/1], [pipeline_name(mergeTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "false"; then
        log_pass "Go merge_sorted desc compiles correctly"
    else
        log_fail "Go merge_sorted desc compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 9: Rust merge_sorted compilation (ascending)
test_rust_merge_sorted() {
    log_info "Test 9: Rust merge_sorted stage compilation (ascending)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, merge_sorted([stage_a/1, stage_b/1], timestamp), output/1], [pipeline_name(merge_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "merge_sorted_stage"; then
        log_pass "Rust merge_sorted compiles correctly"
    else
        log_fail "Rust merge_sorted compilation failed"
    fi
}

# Test 10: Rust merge_sorted with direction
test_rust_merge_sorted_desc() {
    log_info "Test 10: Rust merge_sorted stage compilation (descending)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, merge_sorted([stage_a/1, stage_b/1], timestamp, desc), output/1], [pipeline_name(merge_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "false"; then
        log_pass "Rust merge_sorted desc compiles correctly"
    else
        log_fail "Rust merge_sorted desc compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 11: merge_sorted with multiple streams
test_merge_sorted_multiple() {
    log_info "Test 11: merge_sorted with 3+ stages"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([a/1, b/1, c/1, d/1], ts), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "merge_sorted with multiple stages validation accepted"
    else
        log_fail "merge_sorted with multiple stages failed: $output"
    fi
}

# Test 12: merge_sorted with distinct
test_merge_sorted_with_distinct() {
    log_info "Test 12: merge_sorted combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([a/1, b/1], ts), distinct, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "merge_sorted + distinct validation accepted"
    else
        log_fail "merge_sorted + distinct validation failed: $output"
    fi
}

# Test 13: merge_sorted with filter_by
test_merge_sorted_with_filter() {
    log_info "Test 13: merge_sorted combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([a/1, b/1], ts), filter_by(is_valid), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "merge_sorted + filter_by validation accepted"
    else
        log_fail "merge_sorted + filter_by validation failed: $output"
    fi
}

# Test 14: merge_sorted with take/skip
test_merge_sorted_with_take_skip() {
    log_info "Test 14: merge_sorted with take and skip"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, merge_sorted([a/1, b/1], ts), skip(10), take(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "merge_sorted + skip + take validation accepted"
    else
        log_fail "merge_sorted + skip + take validation failed: $output"
    fi
}

# Test 15: merge_sorted with try_catch
test_merge_sorted_with_try_catch() {
    log_info "Test 15: merge_sorted with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(merge_sorted([a/1, b/1], ts), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(merge_sorted, handler) validation accepted"
    else
        log_fail "try_catch with merge_sorted validation failed: $output"
    fi
}

# Test 16: merge_sorted in parallel
test_merge_sorted_in_parallel() {
    log_info "Test 16: merge_sorted within parallel"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([merge_sorted([a/1, b/1], ts), concat([c/1, d/1])]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "parallel([merge_sorted, concat]) validation accepted"
    else
        log_fail "parallel with merge_sorted validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Merge Sorted Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_merge_sorted
    test_validation_merge_sorted_dir
    test_validation_merge_sorted_empty
    test_validation_merge_sorted_invalid_dir

    # Python compilation tests
    test_python_merge_sorted
    test_python_merge_sorted_desc

    # Go compilation tests
    test_go_merge_sorted
    test_go_merge_sorted_desc

    # Rust compilation tests
    test_rust_merge_sorted
    test_rust_merge_sorted_desc

    # Combined stage tests
    test_merge_sorted_multiple
    test_merge_sorted_with_distinct
    test_merge_sorted_with_filter
    test_merge_sorted_with_take_skip
    test_merge_sorted_with_try_catch
    test_merge_sorted_in_parallel

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
