#!/bin/bash
# test_flatten_stage.sh - Integration tests for flatten pipeline stage
# Tests: flatten, flatten(Field) - Flatten nested collections into individual records

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

# Test 1: Validation accepts flatten (simple)
test_validation_flatten() {
    log_info "Test 1: Validation accepts flatten (simple)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "flatten validation accepted"
    else
        log_fail "flatten validation failed: $output"
    fi
}

# Test 2: Validation accepts flatten(Field)
test_validation_flatten_field() {
    log_info "Test 2: Validation accepts flatten(Field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten(items), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "flatten(Field) validation accepted"
    else
        log_fail "flatten(Field) validation failed: $output"
    fi
}

# Test 3: Validation rejects invalid flatten(Field)
test_validation_flatten_invalid() {
    log_info "Test 3: Validation rejects invalid flatten(Field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten(123), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_flatten\|invalid_stage\|error"; then
        log_pass "Invalid flatten(Field) correctly rejected"
    else
        log_fail "Invalid flatten(Field) should be rejected: $output"
    fi
}

# Test 4: Stage type detection for flatten
test_stage_type_flatten() {
    log_info "Test 4: Stage type detection for flatten"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        stage_type(flatten, Type1),
        stage_type(flatten(items), Type2),
        format('Type1: ~w, Type2: ~w~n', [Type1, Type2])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Type1: flatten, Type2: flatten"; then
        log_pass "flatten stage type correctly detected"
    else
        log_fail "flatten stage type detection failed: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 5: Python flatten stage compilation
test_python_flatten() {
    log_info "Test 5: Python flatten stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, flatten, output/1], [pipeline_name(flatten_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "flatten_stage"; then
        log_pass "Python flatten compiles correctly"
    else
        log_fail "Python flatten compilation failed"
    fi
}

# Test 6: Python flatten(Field) compilation
test_python_flatten_field() {
    log_info "Test 6: Python flatten(Field) stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, flatten(items), output/1], [pipeline_name(flatten_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "flatten_field_stage.*items"; then
        log_pass "Python flatten(Field) compiles correctly"
    else
        log_fail "Python flatten(Field) compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 7: Go flatten stage compilation
test_go_flatten() {
    log_info "Test 7: Go flatten stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, flatten, output/1], [pipeline_name(flattenTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "flattenStage"; then
        log_pass "Go flatten compiles correctly"
    else
        log_fail "Go flatten compilation failed"
    fi
}

# Test 8: Go flatten(Field) compilation
test_go_flatten_field() {
    log_info "Test 8: Go flatten(Field) stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, flatten(items), output/1], [pipeline_name(flattenTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "flattenFieldStage.*items"; then
        log_pass "Go flatten(Field) compiles correctly"
    else
        log_fail "Go flatten(Field) compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 9: Rust flatten stage compilation
test_rust_flatten() {
    log_info "Test 9: Rust flatten stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, flatten, output/1], [pipeline_name(flatten_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "flatten_stage"; then
        log_pass "Rust flatten compiles correctly"
    else
        log_fail "Rust flatten compilation failed"
    fi
}

# Test 10: Rust flatten(Field) compilation
test_rust_flatten_field() {
    log_info "Test 10: Rust flatten(Field) stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, flatten(items), output/1], [pipeline_name(flatten_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "flatten_field_stage.*items"; then
        log_pass "Rust flatten(Field) compiles correctly"
    else
        log_fail "Rust flatten(Field) compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 11: Multiple flatten stages in pipeline
test_multiple_flattens() {
    log_info "Test 11: Multiple flatten stages in pipeline"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten(outer), flatten(inner), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Multiple flattens validation accepted"
    else
        log_fail "Multiple flattens validation failed: $output"
    fi
}

# Test 12: flatten combined with filter_by
test_flatten_with_filter() {
    log_info "Test 12: flatten combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten(items), filter_by(is_valid), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "flatten + filter_by validation accepted"
    else
        log_fail "flatten + filter_by validation failed: $output"
    fi
}

# Test 13: flatten combined with distinct
test_flatten_with_distinct() {
    log_info "Test 13: flatten combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten, distinct, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "flatten + distinct validation accepted"
    else
        log_fail "flatten + distinct validation failed: $output"
    fi
}

# Test 14: flatten with parallel
test_flatten_with_parallel() {
    log_info "Test 14: flatten within parallel"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([flatten(items), flatten(tags)]), merge, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "parallel([flatten, flatten]) validation accepted"
    else
        log_fail "parallel with flatten validation failed: $output"
    fi
}

# Test 15: flatten with try_catch
test_flatten_with_try_catch() {
    log_info "Test 15: flatten with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(flatten(items), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(flatten, handler) validation accepted"
    else
        log_fail "try_catch with flatten validation failed: $output"
    fi
}

# Test 16: flatten with tap
test_flatten_with_tap() {
    log_info "Test 16: flatten combined with tap"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(log_nested), flatten(items), tap(log_flattened), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "flatten + tap validation accepted"
    else
        log_fail "flatten + tap validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Flatten Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_flatten
    test_validation_flatten_field
    test_validation_flatten_invalid
    test_stage_type_flatten

    # Python compilation tests
    test_python_flatten
    test_python_flatten_field

    # Go compilation tests
    test_go_flatten
    test_go_flatten_field

    # Rust compilation tests
    test_rust_flatten
    test_rust_flatten_field

    # Combined stage tests
    test_multiple_flattens
    test_flatten_with_filter
    test_flatten_with_distinct
    test_flatten_with_parallel
    test_flatten_with_try_catch
    test_flatten_with_tap

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
