#!/bin/bash
# test_distinct_dedup_stages.sh - Integration tests for distinct and dedup pipeline stages
# Tests: distinct, distinct_by(Field), dedup, dedup_by(Field)

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

# Test 1: Validation accepts distinct
test_validation_distinct() {
    log_info "Test 1: Validation accepts distinct"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, distinct, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "distinct validation accepted"
    else
        log_fail "distinct validation failed: $output"
    fi
}

# Test 2: Validation accepts distinct_by(Field)
test_validation_distinct_by() {
    log_info "Test 2: Validation accepts distinct_by(Field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, distinct_by(id), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "distinct_by(id) validation accepted"
    else
        log_fail "distinct_by(id) validation failed: $output"
    fi
}

# Test 3: Validation accepts dedup
test_validation_dedup() {
    log_info "Test 3: Validation accepts dedup"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, dedup, output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "dedup validation accepted"
    else
        log_fail "dedup validation failed: $output"
    fi
}

# Test 4: Validation accepts dedup_by(Field)
test_validation_dedup_by() {
    log_info "Test 4: Validation accepts dedup_by(Field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, dedup_by(category), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "dedup_by(category) validation accepted"
    else
        log_fail "dedup_by(category) validation failed: $output"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 5: Python distinct compilation
test_python_distinct() {
    log_info "Test 5: Python distinct stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, distinct, output/1], [pipeline_name(distinct_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "distinct_stage"; then
        log_pass "Python distinct compiles correctly"
    else
        log_fail "Python distinct compilation failed"
    fi
}

# Test 6: Python distinct_by compilation
test_python_distinct_by() {
    log_info "Test 6: Python distinct_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, distinct_by(id), output/1], [pipeline_name(distinct_by_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "distinct_by_stage"; then
        log_pass "Python distinct_by compiles correctly"
    else
        log_fail "Python distinct_by compilation failed"
    fi
}

# Test 7: Python dedup compilation
test_python_dedup() {
    log_info "Test 7: Python dedup stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, dedup, output/1], [pipeline_name(dedup_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "dedup_stage"; then
        log_pass "Python dedup compiles correctly"
    else
        log_fail "Python dedup compilation failed"
    fi
}

# Test 8: Python dedup_by compilation
test_python_dedup_by() {
    log_info "Test 8: Python dedup_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, dedup_by(category), output/1], [pipeline_name(dedup_by_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "dedup_by_stage"; then
        log_pass "Python dedup_by compiles correctly"
    else
        log_fail "Python dedup_by compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 9: Go distinct compilation
test_go_distinct() {
    log_info "Test 9: Go distinct stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, distinct, output/1], [pipeline_name(distinctTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "distinctStage"; then
        log_pass "Go distinct compiles correctly"
    else
        log_fail "Go distinct compilation failed"
    fi
}

# Test 10: Go distinct_by compilation
test_go_distinct_by() {
    log_info "Test 10: Go distinct_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, distinct_by(id), output/1], [pipeline_name(distinctByTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "distinctByStage"; then
        log_pass "Go distinct_by compiles correctly"
    else
        log_fail "Go distinct_by compilation failed"
    fi
}

# Test 11: Go dedup compilation
test_go_dedup() {
    log_info "Test 11: Go dedup stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, dedup, output/1], [pipeline_name(dedupTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "dedupStage"; then
        log_pass "Go dedup compiles correctly"
    else
        log_fail "Go dedup compilation failed"
    fi
}

# Test 12: Go dedup_by compilation
test_go_dedup_by() {
    log_info "Test 12: Go dedup_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, dedup_by(category), output/1], [pipeline_name(dedupByTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "dedupByStage"; then
        log_pass "Go dedup_by compiles correctly"
    else
        log_fail "Go dedup_by compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 13: Rust distinct compilation
test_rust_distinct() {
    log_info "Test 13: Rust distinct stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, distinct, output/1], [pipeline_name(distinct_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "distinct_stage"; then
        log_pass "Rust distinct compiles correctly"
    else
        log_fail "Rust distinct compilation failed"
    fi
}

# Test 14: Rust distinct_by compilation
test_rust_distinct_by() {
    log_info "Test 14: Rust distinct_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, distinct_by(id), output/1], [pipeline_name(distinct_by_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "distinct_by_stage"; then
        log_pass "Rust distinct_by compiles correctly"
    else
        log_fail "Rust distinct_by compilation failed"
    fi
}

# Test 15: Rust dedup compilation
test_rust_dedup() {
    log_info "Test 15: Rust dedup stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, dedup, output/1], [pipeline_name(dedup_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "dedup_stage"; then
        log_pass "Rust dedup compiles correctly"
    else
        log_fail "Rust dedup compilation failed"
    fi
}

# Test 16: Rust dedup_by compilation
test_rust_dedup_by() {
    log_info "Test 16: Rust dedup_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, dedup_by(category), output/1], [pipeline_name(dedup_by_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "dedup_by_stage"; then
        log_pass "Rust dedup_by compiles correctly"
    else
        log_fail "Rust dedup_by compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 17: Combined distinct and take
test_combined_distinct_take() {
    log_info "Test 17: Combined distinct and take validation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, distinct, take(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Combined distinct and take validation accepted"
    else
        log_fail "Combined distinct and take validation failed: $output"
    fi
}

# Test 18: Dedup with error handling
test_dedup_with_try_catch() {
    log_info "Test 18: Dedup with error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(dedup, error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(dedup, handler) validation accepted"
    else
        log_fail "try_catch with dedup validation failed: $output"
    fi
}

# Test 19: Distinct with sorting
test_distinct_with_sort() {
    log_info "Test 19: Distinct with sorting"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, order_by(name), distinct_by(id), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "order_by + distinct_by validation accepted"
    else
        log_fail "order_by + distinct_by validation failed: $output"
    fi
}

# Test 20: Dedup with window
test_dedup_with_window() {
    log_info "Test 20: Dedup with window"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, window(10), dedup_by(type), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "window + dedup_by validation accepted"
    else
        log_fail "window + dedup_by validation failed: $output"
    fi
}

# Test 21: Multiple distinct/dedup stages
test_multiple_dedup_stages() {
    log_info "Test 21: Multiple distinct/dedup stages"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, dedup_by(user_id), distinct_by(session_id), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Multiple dedup/distinct stages validation accepted"
    else
        log_fail "Multiple dedup/distinct stages validation failed: $output"
    fi
}

# Test 22: Distinct in parallel
test_distinct_parallel() {
    log_info "Test 22: Distinct in parallel"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, parallel([distinct, dedup]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "parallel([distinct, dedup]) validation accepted"
    else
        log_fail "parallel with distinct/dedup validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "==========================================="
    echo "  Pipeline Distinct/Dedup Stage Tests"
    echo "==========================================="
    echo ""

    # Validation tests
    test_validation_distinct
    test_validation_distinct_by
    test_validation_dedup
    test_validation_dedup_by

    # Python compilation tests
    test_python_distinct
    test_python_distinct_by
    test_python_dedup
    test_python_dedup_by

    # Go compilation tests
    test_go_distinct
    test_go_distinct_by
    test_go_dedup
    test_go_dedup_by

    # Rust compilation tests
    test_rust_distinct
    test_rust_distinct_by
    test_rust_dedup
    test_rust_dedup_by

    # Combined stage tests
    test_combined_distinct_take
    test_dedup_with_try_catch
    test_distinct_with_sort
    test_dedup_with_window
    test_multiple_dedup_stages
    test_distinct_parallel

    echo ""
    echo "==========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "==========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
