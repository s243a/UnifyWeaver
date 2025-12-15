#!/bin/bash
# test_tap_stage.sh - Integration tests for tap pipeline stage
# Tests: tap(Pred), tap(Pred/Arity) - Execute side effects without modifying stream

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

# Test 1: Validation accepts tap(Pred) with atom
test_validation_tap_atom() {
    log_info "Test 1: Validation accepts tap(Pred) with atom"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(log_record), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap(atom) validation accepted"
    else
        log_fail "tap(atom) validation failed: $output"
    fi
}

# Test 2: Validation accepts tap(Pred/Arity)
test_validation_tap_pred_arity() {
    log_info "Test 2: Validation accepts tap(Pred/Arity)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(log_record/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap(pred/arity) validation accepted"
    else
        log_fail "tap(pred/arity) validation failed: $output"
    fi
}

# Test 3: Validation rejects invalid tap
test_validation_tap_invalid() {
    log_info "Test 3: Validation rejects invalid tap"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(123), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_tap\|invalid_stage\|error"; then
        log_pass "Invalid tap correctly rejected"
    else
        log_fail "Invalid tap should be rejected: $output"
    fi
}

# Test 4: Stage type detection for tap
test_stage_type_tap() {
    log_info "Test 4: Stage type detection for tap"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        stage_type(tap(logger), Type),
        format('Type: ~w~n', [Type])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Type: tap"; then
        log_pass "tap stage type correctly detected"
    else
        log_fail "tap stage type detection failed: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 5: Python tap stage compilation
test_python_tap() {
    log_info "Test 5: Python tap stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, tap(log_record), output/1], [pipeline_name(tap_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tap_stage"; then
        log_pass "Python tap compiles correctly"
    else
        log_fail "Python tap compilation failed"
    fi
}

# Test 6: Python tap with pred/arity
test_python_tap_pred_arity() {
    log_info "Test 6: Python tap stage with pred/arity compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, tap(log_record/1), output/1], [pipeline_name(tap_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tap_stage.*log_record"; then
        log_pass "Python tap with pred/arity compiles correctly"
    else
        log_fail "Python tap with pred/arity compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 7: Go tap stage compilation
test_go_tap() {
    log_info "Test 7: Go tap stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, tap(logRecord), output/1], [pipeline_name(tapTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tapStage"; then
        log_pass "Go tap compiles correctly"
    else
        log_fail "Go tap compilation failed"
    fi
}

# Test 8: Go tap with pred/arity
test_go_tap_pred_arity() {
    log_info "Test 8: Go tap stage with pred/arity compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, tap(logRecord/1), output/1], [pipeline_name(tapTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tapStage.*logRecord"; then
        log_pass "Go tap with pred/arity compiles correctly"
    else
        log_fail "Go tap with pred/arity compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 9: Rust tap stage compilation
test_rust_tap() {
    log_info "Test 9: Rust tap stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, tap(log_record), output/1], [pipeline_name(tap_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tap_stage"; then
        log_pass "Rust tap compiles correctly"
    else
        log_fail "Rust tap compilation failed"
    fi
}

# Test 10: Rust tap with pred/arity
test_rust_tap_pred_arity() {
    log_info "Test 10: Rust tap stage with pred/arity compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, tap(log_record/1), output/1], [pipeline_name(tap_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tap_stage.*log_record"; then
        log_pass "Rust tap with pred/arity compiles correctly"
    else
        log_fail "Rust tap with pred/arity compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 11: Multiple taps in pipeline
test_multiple_taps() {
    log_info "Test 11: Multiple tap stages in pipeline"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(logger1), tap(logger2), tap(metrics), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Multiple taps validation accepted"
    else
        log_fail "Multiple taps validation failed: $output"
    fi
}

# Test 12: tap combined with filter_by
test_tap_with_filter() {
    log_info "Test 12: tap combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(pre_filter_log), filter_by(is_valid), tap(post_filter_log), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap + filter_by validation accepted"
    else
        log_fail "tap + filter_by validation failed: $output"
    fi
}

# Test 13: tap combined with predicate stage
test_tap_with_predicate() {
    log_info "Test 13: tap combined with predicate stage"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(debug_input), normalize/1, tap(debug_output), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap + predicate validation accepted"
    else
        log_fail "tap + predicate validation failed: $output"
    fi
}

# Test 14: tap with parallel
test_tap_with_parallel() {
    log_info "Test 14: tap within parallel"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([tap(branch1_log), tap(branch2_log)]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "parallel([tap, tap]) validation accepted"
    else
        log_fail "parallel with tap validation failed: $output"
    fi
}

# Test 15: tap with try_catch
test_tap_with_try_catch() {
    log_info "Test 15: tap with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(tap(risky_log), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(tap, handler) validation accepted"
    else
        log_fail "try_catch with tap validation failed: $output"
    fi
}

# Test 16: tap with distinct
test_tap_with_distinct() {
    log_info "Test 16: tap combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(pre_dedup), distinct, tap(post_dedup), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap + distinct validation accepted"
    else
        log_fail "tap + distinct validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Tap Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_tap_atom
    test_validation_tap_pred_arity
    test_validation_tap_invalid
    test_stage_type_tap

    # Python compilation tests
    test_python_tap
    test_python_tap_pred_arity

    # Go compilation tests
    test_go_tap
    test_go_tap_pred_arity

    # Rust compilation tests
    test_rust_tap
    test_rust_tap_pred_arity

    # Combined stage tests
    test_multiple_taps
    test_tap_with_filter
    test_tap_with_predicate
    test_tap_with_parallel
    test_tap_with_try_catch
    test_tap_with_distinct

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
