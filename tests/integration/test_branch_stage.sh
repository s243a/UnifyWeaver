#!/bin/bash
# test_branch_stage.sh - Integration tests for branch pipeline stage
# Tests: branch(Cond, TrueStage, FalseStage) - Conditional routing

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

# Test 1: Validation accepts branch(Cond, TrueStage, FalseStage)
test_validation_branch() {
    log_info "Test 1: Validation accepts branch(Cond, TrueStage, FalseStage)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "branch(Cond, TrueStage, FalseStage) validation accepted"
    else
        log_fail "branch validation failed: $output"
    fi
}

# Test 2: Validation accepts branch(Cond/Arity, TrueStage, FalseStage)
test_validation_branch_arity() {
    log_info "Test 2: Validation accepts branch(Cond/Arity, TrueStage, FalseStage)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, branch(is_active/1, transform/1, skip/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "branch(Cond/Arity, TrueStage, FalseStage) validation accepted"
    else
        log_fail "branch with arity validation failed: $output"
    fi
}

# Test 3: Stage type detection for branch
test_stage_type_branch() {
    log_info "Test 3: Stage type detection for branch"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        stage_type(branch(is_active, transform/1, skip/1), Type),
        format('Type: ~w~n', [Type])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Type: branch"; then
        log_pass "branch stage type correctly detected"
    else
        log_fail "branch stage type detection failed: $output"
    fi
}

# Test 4: Validation validates nested stages
test_validation_branch_nested() {
    log_info "Test 4: Validation validates nested branch stages"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, branch(cond/1, filter_by(pred), distinct), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "branch with nested stages validation accepted"
    else
        log_fail "branch with nested stages validation failed: $output"
    fi
}

# Test 5: Validation validates nested branch in branch
test_validation_branch_recursive() {
    log_info "Test 5: Validation validates nested branch in branch"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, branch(cond1/1, branch(cond2/1, a/1, b/1), c/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "nested branch in branch validation accepted"
    else
        log_fail "nested branch validation failed: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 6: Python branch stage compilation
test_python_branch() {
    log_info "Test 6: Python branch stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], [pipeline_name(branch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "branch_stage"; then
        log_pass "Python branch compiles correctly"
    else
        log_fail "Python branch compilation failed"
    fi
}

# Test 7: Python branch generates condition function
test_python_branch_cond() {
    log_info "Test 7: Python branch generates condition function"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], [pipeline_name(branch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "_branch_cond" && echo "$output" | grep -q "is_active"; then
        log_pass "Python branch generates condition function"
    else
        log_fail "Python branch condition function not generated"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 8: Go branch stage compilation
test_go_branch() {
    log_info "Test 8: Go branch stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], [pipeline_name(branchTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "branchStage"; then
        log_pass "Go branch compiles correctly"
    else
        log_fail "Go branch compilation failed"
    fi
}

# Test 9: Go branch generates function parameters
test_go_branch_params() {
    log_info "Test 9: Go branch generates function parameters"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], [pipeline_name(branchTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "func(r Record) bool"; then
        log_pass "Go branch generates function parameters"
    else
        log_fail "Go branch function parameters not generated"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 10: Rust branch stage compilation
test_rust_branch() {
    log_info "Test 10: Rust branch stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], [pipeline_name(branch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "branch_stage"; then
        log_pass "Rust branch compiles correctly"
    else
        log_fail "Rust branch compilation failed"
    fi
}

# Test 11: Rust branch generates closures
test_rust_branch_closures() {
    log_info "Test 11: Rust branch generates closures"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, branch(is_active, transform/1, skip/1), output/1], [pipeline_name(branch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "|r| is_active(r)"; then
        log_pass "Rust branch generates closures"
    else
        log_fail "Rust branch closures not generated"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 12: branch combined with filter_by
test_branch_with_filter() {
    log_info "Test 12: branch combined with filter_by"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, filter_by(active), branch(cond/1, a/1, b/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "filter_by + branch validation accepted"
    else
        log_fail "filter_by + branch validation failed: $output"
    fi
}

# Test 13: branch combined with distinct
test_branch_with_distinct() {
    log_info "Test 13: branch combined with distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, branch(cond/1, distinct, dedup), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "branch with distinct/dedup stages validation accepted"
    else
        log_fail "branch with distinct/dedup validation failed: $output"
    fi
}

# Test 14: branch combined with tap
test_branch_with_tap() {
    log_info "Test 14: branch combined with tap"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, tap(log_input), branch(cond/1, a/1, b/1), tap(log_output), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "tap + branch + tap validation accepted"
    else
        log_fail "tap + branch + tap validation failed: $output"
    fi
}

# Test 15: branch with try_catch
test_branch_with_try_catch() {
    log_info "Test 15: branch with try_catch error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(branch(cond/1, a/1, b/1), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(branch, handler) validation accepted"
    else
        log_fail "try_catch with branch validation failed: $output"
    fi
}

# Test 16: branch with debounce
test_branch_with_debounce() {
    log_info "Test 16: branch combined with debounce"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(100), branch(cond/1, a/1, b/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce + branch validation accepted"
    else
        log_fail "debounce + branch validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Branch Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_branch
    test_validation_branch_arity
    test_stage_type_branch
    test_validation_branch_nested
    test_validation_branch_recursive

    # Python compilation tests
    test_python_branch
    test_python_branch_cond

    # Go compilation tests
    test_go_branch
    test_go_branch_params

    # Rust compilation tests
    test_rust_branch
    test_rust_branch_closures

    # Combined stage tests
    test_branch_with_filter
    test_branch_with_distinct
    test_branch_with_tap
    test_branch_with_try_catch
    test_branch_with_debounce

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
