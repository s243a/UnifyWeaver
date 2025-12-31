#!/bin/bash
# test_debounce_stage.sh - Integration tests for debounce pipeline stage
# Tests: debounce(Ms), debounce(Ms, Field) - Emit only after silence period

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

# Test 1: Validation accepts debounce(Ms)
test_validation_debounce() {
    log_info "Test 1: Validation accepts debounce(Ms)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce(Ms) validation accepted"
    else
        log_fail "debounce(Ms) validation failed: $output"
    fi
}

# Test 2: Validation accepts debounce(Ms, Field)
test_validation_debounce_field() {
    log_info "Test 2: Validation accepts debounce(Ms, Field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(500, timestamp), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce(Ms, Field) validation accepted"
    else
        log_fail "debounce(Ms, Field) validation failed: $output"
    fi
}

# Test 3: Validation rejects invalid debounce (zero ms)
test_validation_debounce_zero() {
    log_info "Test 3: Validation rejects invalid debounce (zero ms)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_debounce\|invalid_stage\|error"; then
        log_pass "Invalid debounce(0) correctly rejected"
    else
        log_fail "Invalid debounce(0) should be rejected: $output"
    fi
}

# Test 4: Validation rejects invalid debounce (negative ms)
test_validation_debounce_negative() {
    log_info "Test 4: Validation rejects invalid debounce (negative ms)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(-100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "invalid_debounce\|invalid_stage\|error"; then
        log_pass "Invalid debounce(-100) correctly rejected"
    else
        log_fail "Invalid debounce(-100) should be rejected: $output"
    fi
}

# Test 5: Stage type detection for debounce
test_stage_type_debounce() {
    log_info "Test 5: Stage type detection for debounce"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        stage_type(debounce(100), Type1),
        stage_type(debounce(100, ts), Type2),
        format('Type1: ~w, Type2: ~w~n', [Type1, Type2])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Type1: debounce, Type2: debounce"; then
        log_pass "debounce stage type correctly detected"
    else
        log_fail "debounce stage type detection failed: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 6: Python debounce stage compilation
test_python_debounce() {
    log_info "Test 6: Python debounce stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, debounce(100), output/1], [pipeline_name(debounce_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "debounce_stage"; then
        log_pass "Python debounce compiles correctly"
    else
        log_fail "Python debounce compilation failed"
    fi
}

# Test 7: Python debounce(Ms, Field) compilation
test_python_debounce_field() {
    log_info "Test 7: Python debounce(Ms, Field) stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, debounce(500, timestamp), output/1], [pipeline_name(debounce_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "debounce_stage.*timestamp"; then
        log_pass "Python debounce(Ms, Field) compiles correctly"
    else
        log_fail "Python debounce(Ms, Field) compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 8: Go debounce stage compilation
test_go_debounce() {
    log_info "Test 8: Go debounce stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, debounce(100), output/1], [pipeline_name(debounceTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "debounceStage"; then
        log_pass "Go debounce compiles correctly"
    else
        log_fail "Go debounce compilation failed"
    fi
}

# Test 9: Go debounce(Ms, Field) compilation
test_go_debounce_field() {
    log_info "Test 9: Go debounce(Ms, Field) stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, debounce(500, timestamp), output/1], [pipeline_name(debounceTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "debounceStage.*timestamp"; then
        log_pass "Go debounce(Ms, Field) compiles correctly"
    else
        log_fail "Go debounce(Ms, Field) compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 10: Rust debounce stage compilation
test_rust_debounce() {
    log_info "Test 10: Rust debounce stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, debounce(100), output/1], [pipeline_name(debounce_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "debounce_stage"; then
        log_pass "Rust debounce compiles correctly"
    else
        log_fail "Rust debounce compilation failed"
    fi
}

# Test 11: Rust debounce(Ms, Field) compilation
test_rust_debounce_field() {
    log_info "Test 11: Rust debounce(Ms, Field) stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, debounce(500, timestamp), output/1], [pipeline_name(debounce_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "debounce_stage.*timestamp"; then
        log_pass "Rust debounce(Ms, Field) compiles correctly"
    else
        log_fail "Rust debounce(Ms, Field) compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 12: debounce combined with filter_by
test_debounce_with_filter() {
    log_info "Test 12: debounce combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(100), filter_by(is_valid), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce + filter_by validation accepted"
    else
        log_fail "debounce + filter_by validation failed: $output"
    fi
}

# Test 13: debounce combined with distinct
test_debounce_with_distinct() {
    log_info "Test 13: debounce combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(100), distinct, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce + distinct validation accepted"
    else
        log_fail "debounce + distinct validation failed: $output"
    fi
}

# Test 14: debounce combined with tap
test_debounce_with_tap() {
    log_info "Test 14: debounce combined with tap"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(log_input), debounce(100), tap(log_debounced), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce + tap validation accepted"
    else
        log_fail "debounce + tap validation failed: $output"
    fi
}

# Test 15: debounce with try_catch
test_debounce_with_try_catch() {
    log_info "Test 15: debounce with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(debounce(100), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(debounce, handler) validation accepted"
    else
        log_fail "try_catch with debounce validation failed: $output"
    fi
}

# Test 16: debounce with flatten
test_debounce_with_flatten() {
    log_info "Test 16: debounce combined with flatten"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, flatten(items), debounce(100, timestamp), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce + flatten validation accepted"
    else
        log_fail "debounce + flatten validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Debounce Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_debounce
    test_validation_debounce_field
    test_validation_debounce_zero
    test_validation_debounce_negative
    test_stage_type_debounce

    # Python compilation tests
    test_python_debounce
    test_python_debounce_field

    # Go compilation tests
    test_go_debounce
    test_go_debounce_field

    # Rust compilation tests
    test_rust_debounce
    test_rust_debounce_field

    # Combined stage tests
    test_debounce_with_filter
    test_debounce_with_distinct
    test_debounce_with_tap
    test_debounce_with_try_catch
    test_debounce_with_flatten

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
