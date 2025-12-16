#!/bin/bash
# test_tee_stage.sh - Integration tests for tee pipeline stage
# Tests: tee(Stage) - Fork stream to side destination, pass original through

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

# Test 1: Validation accepts tee(Stage)
test_validation_tee() {
    log_info "Test 1: Validation accepts tee(Stage)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tee(log_to_file/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tee(Stage) validation accepted"
    else
        log_fail "tee validation failed: $output"
    fi
}

# Test 2: Validation accepts tee with complex stage
test_validation_tee_complex() {
    log_info "Test 2: Validation accepts tee with complex stage"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tee(filter_by(active)), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tee with filter_by validation accepted"
    else
        log_fail "tee with filter_by validation failed: $output"
    fi
}

# Test 3: Stage type detection for tee
test_stage_type_tee() {
    log_info "Test 3: Stage type detection for tee"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        stage_type(tee(log/1), Type),
        format('Type: ~w~n', [Type])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Type: tee"; then
        log_pass "tee stage type correctly detected"
    else
        log_fail "tee stage type detection failed: $output"
    fi
}

# Test 4: Validation validates nested tee stages
test_validation_tee_nested() {
    log_info "Test 4: Validation validates nested tee in tee"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tee(tee(inner_log/1)), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "nested tee validation accepted"
    else
        log_fail "nested tee validation failed: $output"
    fi
}

# Test 5: Validation accepts tee with branch stage
test_validation_tee_branch() {
    log_info "Test 5: Validation accepts tee with branch stage"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tee(branch(cond/1, a/1, b/1)), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tee with branch validation accepted"
    else
        log_fail "tee with branch validation failed: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 6: Python tee stage compilation
test_python_tee() {
    log_info "Test 6: Python tee stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, tee(log_to_file/1), output/1], [pipeline_name(tee_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tee_stage"; then
        log_pass "Python tee compiles correctly"
    else
        log_fail "Python tee compilation failed"
    fi
}

# Test 7: Python tee generates side function
test_python_tee_side_fn() {
    log_info "Test 7: Python tee generates side function"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, tee(log_to_file/1), output/1], [pipeline_name(tee_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "_tee_side" && echo "$output" | grep -q "log_to_file"; then
        log_pass "Python tee generates side function"
    else
        log_fail "Python tee side function not generated"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 8: Go tee stage compilation
test_go_tee() {
    log_info "Test 8: Go tee stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, tee(log_to_file/1), output/1], [pipeline_name(teeTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "teeStage"; then
        log_pass "Go tee compiles correctly"
    else
        log_fail "Go tee compilation failed"
    fi
}

# Test 9: Go tee generates function parameter
test_go_tee_fn_param() {
    log_info "Test 9: Go tee generates function parameter"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, tee(log_to_file/1), output/1], [pipeline_name(teeTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "func(rs \[\]Record)"; then
        log_pass "Go tee generates function parameter"
    else
        log_fail "Go tee function parameter not generated"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 10: Rust tee stage compilation
test_rust_tee() {
    log_info "Test 10: Rust tee stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, tee(log_to_file/1), output/1], [pipeline_name(tee_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tee_stage"; then
        log_pass "Rust tee compiles correctly"
    else
        log_fail "Rust tee compilation failed"
    fi
}

# Test 11: Rust tee generates closure
test_rust_tee_closure() {
    log_info "Test 11: Rust tee generates closure"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, tee(log_to_file/1), output/1], [pipeline_name(tee_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "|rs| log_to_file(rs)"; then
        log_pass "Rust tee generates closure"
    else
        log_fail "Rust tee closure not generated"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 12: tee combined with filter_by
test_tee_with_filter() {
    log_info "Test 12: tee combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, filter_by(active), tee(log/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "filter_by + tee validation accepted"
    else
        log_fail "filter_by + tee validation failed: $output"
    fi
}

# Test 13: tee combined with distinct
test_tee_with_distinct() {
    log_info "Test 13: tee combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tee(log/1), distinct, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tee + distinct validation accepted"
    else
        log_fail "tee + distinct validation failed: $output"
    fi
}

# Test 14: tee combined with tap
test_tee_with_tap() {
    log_info "Test 14: tee combined with tap"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(log_record), tee(archive/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap + tee validation accepted"
    else
        log_fail "tap + tee validation failed: $output"
    fi
}

# Test 15: tee with try_catch
test_tee_with_try_catch() {
    log_info "Test 15: tee with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(tee(risky/1), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(tee, handler) validation accepted"
    else
        log_fail "try_catch with tee validation failed: $output"
    fi
}

# Test 16: multiple tee stages
test_multiple_tee() {
    log_info "Test 16: multiple tee stages in pipeline"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tee(log1/1), tee(log2/1), tee(log3/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "multiple tee stages validation accepted"
    else
        log_fail "multiple tee stages validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Tee Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_tee
    test_validation_tee_complex
    test_stage_type_tee
    test_validation_tee_nested
    test_validation_tee_branch

    # Python compilation tests
    test_python_tee
    test_python_tee_side_fn

    # Go compilation tests
    test_go_tee
    test_go_tee_fn_param

    # Rust compilation tests
    test_rust_tee
    test_rust_tee_closure

    # Combined stage tests
    test_tee_with_filter
    test_tee_with_distinct
    test_tee_with_tap
    test_tee_with_try_catch
    test_multiple_tee

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
