#!/bin/bash
# test_interleave_concat_stages.sh - Integration tests for interleave and concat pipeline stages
# Tests: interleave(Stages), concat(Stages)

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

# Test 1: Validation accepts interleave(Stages)
test_validation_interleave() {
    log_info "Test 1: Validation accepts interleave(Stages)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, interleave([filter_a/1, filter_b/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "interleave(Stages) validation accepted"
    else
        log_fail "interleave(Stages) validation failed: $output"
    fi
}

# Test 2: Validation accepts concat(Stages)
test_validation_concat() {
    log_info "Test 2: Validation accepts concat(Stages)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, concat([source_a/1, source_b/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "concat(Stages) validation accepted"
    else
        log_fail "concat(Stages) validation failed: $output"
    fi
}

# Test 3: Validation rejects empty interleave
test_validation_interleave_empty() {
    log_info "Test 3: Validation rejects empty interleave"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, interleave([]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_interleave\|invalid_stage\|empty"; then
        log_pass "Empty interleave correctly rejected"
    else
        log_fail "Empty interleave should be rejected: $output"
    fi
}

# Test 4: Validation rejects empty concat
test_validation_concat_empty() {
    log_info "Test 4: Validation rejects empty concat"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, concat([]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_concat\|invalid_stage\|empty"; then
        log_pass "Empty concat correctly rejected"
    else
        log_fail "Empty concat should be rejected: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 5: Python interleave compilation
test_python_interleave() {
    log_info "Test 5: Python interleave stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, interleave([stage_a/1, stage_b/1]), output/1], [pipeline_name(interleave_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "interleave_stage"; then
        log_pass "Python interleave compiles correctly"
    else
        log_fail "Python interleave compilation failed"
    fi
}

# Test 6: Python concat compilation
test_python_concat() {
    log_info "Test 6: Python concat stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, concat([stage_a/1, stage_b/1]), output/1], [pipeline_name(concat_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "concat_stage"; then
        log_pass "Python concat compiles correctly"
    else
        log_fail "Python concat compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 7: Go interleave compilation
test_go_interleave() {
    log_info "Test 7: Go interleave stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, interleave([stageA/1, stageB/1]), output/1], [pipeline_name(interleaveTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "interleaveStage"; then
        log_pass "Go interleave compiles correctly"
    else
        log_fail "Go interleave compilation failed"
    fi
}

# Test 8: Go concat compilation
test_go_concat() {
    log_info "Test 8: Go concat stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, concat([stageA/1, stageB/1]), output/1], [pipeline_name(concatTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "concatStage"; then
        log_pass "Go concat compiles correctly"
    else
        log_fail "Go concat compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 9: Rust interleave compilation
test_rust_interleave() {
    log_info "Test 9: Rust interleave stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, interleave([stage_a/1, stage_b/1]), output/1], [pipeline_name(interleave_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "interleave_stage"; then
        log_pass "Rust interleave compiles correctly"
    else
        log_fail "Rust interleave compilation failed"
    fi
}

# Test 10: Rust concat compilation
test_rust_concat() {
    log_info "Test 10: Rust concat stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, concat([stage_a/1, stage_b/1]), output/1], [pipeline_name(concat_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "concat_stage"; then
        log_pass "Rust concat compiles correctly"
    else
        log_fail "Rust concat compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 11: Interleave with multiple stages
test_interleave_multiple() {
    log_info "Test 11: Interleave with 3+ stages"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, interleave([a/1, b/1, c/1, d/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Interleave with multiple stages validation accepted"
    else
        log_fail "Interleave with multiple stages failed: $output"
    fi
}

# Test 12: Concat with filter_by
test_concat_with_filter() {
    log_info "Test 12: Concat combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, concat([source_a/1, source_b/1]), filter_by(is_valid), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "concat + filter_by validation accepted"
    else
        log_fail "concat + filter_by validation failed: $output"
    fi
}

# Test 13: Interleave with distinct
test_interleave_with_distinct() {
    log_info "Test 13: Interleave combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, interleave([a/1, b/1]), distinct, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "interleave + distinct validation accepted"
    else
        log_fail "interleave + distinct validation failed: $output"
    fi
}

# Test 14: Nested interleave/concat
test_nested_interleave_concat() {
    log_info "Test 14: Nested interleave within concat"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, concat([interleave([a/1, b/1]), c/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Nested interleave/concat validation accepted"
    else
        log_fail "Nested interleave/concat validation failed: $output"
    fi
}

# Test 15: Concat with take/skip
test_concat_with_take_skip() {
    log_info "Test 15: Concat with take and skip"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, concat([a/1, b/1]), skip(5), take(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "concat + skip + take validation accepted"
    else
        log_fail "concat + skip + take validation failed: $output"
    fi
}

# Test 16: Interleave in parallel
test_interleave_in_parallel() {
    log_info "Test 16: Interleave within parallel"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([interleave([a/1, b/1]), concat([c/1, d/1])]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "parallel([interleave, concat]) validation accepted"
    else
        log_fail "parallel with interleave/concat validation failed: $output"
    fi
}

# Test 17: Concat with error handling
test_concat_with_try_catch() {
    log_info "Test 17: Concat with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(concat([a/1, b/1]), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(concat, handler) validation accepted"
    else
        log_fail "try_catch with concat validation failed: $output"
    fi
}

# Test 18: Interleave with window
test_interleave_with_window() {
    log_info "Test 18: Interleave with window stage"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, interleave([a/1, b/1]), window(10), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "interleave + window validation accepted"
    else
        log_fail "interleave + window validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Interleave/Concat Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_interleave
    test_validation_concat
    test_validation_interleave_empty
    test_validation_concat_empty

    # Python compilation tests
    test_python_interleave
    test_python_concat

    # Go compilation tests
    test_go_interleave
    test_go_concat

    # Rust compilation tests
    test_rust_interleave
    test_rust_concat

    # Combined stage tests
    test_interleave_multiple
    test_concat_with_filter
    test_interleave_with_distinct
    test_nested_interleave_concat
    test_concat_with_take_skip
    test_interleave_in_parallel
    test_concat_with_try_catch
    test_interleave_with_window

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
