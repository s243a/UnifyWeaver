#!/bin/bash
# test_error_handling_stages.sh - Integration tests for pipeline error handling stages
# Tests try_catch, retry, and on_error stages

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

# Test 1: Validation accepts try_catch(stage, handler)
test_validation_try_catch() {
    log_info "Test 1: Validation accepts try_catch(stage, handler)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(process/1, error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(stage, handler) validation accepted"
    else
        log_fail "try_catch(stage, handler) validation failed: $output"
    fi
}

# Test 2: Validation accepts retry(stage, N)
test_validation_retry_simple() {
    log_info "Test 2: Validation accepts retry(stage, N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, retry(fetch/1, 3), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "retry(stage, N) validation accepted"
    else
        log_fail "retry(stage, N) validation failed: $output"
    fi
}

# Test 3: Validation accepts retry(stage, N, options)
test_validation_retry_options() {
    log_info "Test 3: Validation accepts retry(stage, N, options)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, retry(fetch/1, 3, [delay(1000), backoff(exponential)]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "retry(stage, N, options) validation accepted"
    else
        log_fail "retry(stage, N, options) validation failed: $output"
    fi
}

# Test 4: Validation accepts on_error(handler)
test_validation_on_error() {
    log_info "Test 4: Validation accepts on_error(handler)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, on_error(log_error/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "on_error(handler) validation accepted"
    else
        log_fail "on_error(handler) validation failed: $output"
    fi
}

# Test 5: Validation rejects retry with invalid count
test_validation_retry_invalid() {
    log_info "Test 5: Validation rejects retry with invalid count"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, retry(fetch/1, 0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "retry with invalid count correctly rejected"
    else
        log_fail "retry with invalid count should have been rejected"
    fi
}

# Test 6: Python try_catch compilation
test_python_try_catch() {
    log_info "Test 6: Python try_catch stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, try_catch(process/1, error_handler/1), output/1], [pipeline_name(try_catch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "try_catch_stage" && \
       echo "$output" | grep -q "Try-Catch"; then
        log_pass "Python try_catch compiles correctly"
    else
        log_fail "Python try_catch compilation failed"
    fi
}

# Test 7: Python retry compilation
test_python_retry() {
    log_info "Test 7: Python retry stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, retry(fetch/1, 3), output/1], [pipeline_name(retry_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "retry_stage" && \
       echo "$output" | grep -q "Retry"; then
        log_pass "Python retry compiles correctly"
    else
        log_fail "Python retry compilation failed"
    fi
}

# Test 8: Python retry with options compilation
test_python_retry_options() {
    log_info "Test 8: Python retry with options compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, retry(fetch/1, 3, [delay(1000), backoff(exponential)]), output/1], [pipeline_name(retry_opts_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "retry_stage" && \
       echo "$output" | grep -q "1000" && \
       echo "$output" | grep -q "exponential"; then
        log_pass "Python retry with options compiles correctly"
    else
        log_fail "Python retry with options compilation failed"
    fi
}

# Test 9: Python on_error compilation
test_python_on_error() {
    log_info "Test 9: Python on_error stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, on_error(log_error/1), output/1], [pipeline_name(on_error_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "on_error_stage" && \
       echo "$output" | grep -q "On-Error"; then
        log_pass "Python on_error compiles correctly"
    else
        log_fail "Python on_error compilation failed"
    fi
}

# Test 10: Go try_catch compilation
test_go_try_catch() {
    log_info "Test 10: Go try_catch stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, try_catch(process/1, errorHandler/1), output/1], [pipeline_name(tryCatchTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "tryCatchStage" && \
       echo "$output" | grep -q "Try-Catch"; then
        log_pass "Go try_catch compiles correctly"
    else
        log_fail "Go try_catch compilation failed"
    fi
}

# Test 11: Go retry compilation
test_go_retry() {
    log_info "Test 11: Go retry stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, retry(fetch/1, 3), output/1], [pipeline_name(retryTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "retryStage" && \
       echo "$output" | grep -q "Retry"; then
        log_pass "Go retry compiles correctly"
    else
        log_fail "Go retry compilation failed"
    fi
}

# Test 12: Go on_error compilation
test_go_on_error() {
    log_info "Test 12: Go on_error stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, on_error(logError/1), output/1], [pipeline_name(onErrorTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "onErrorStage" && \
       echo "$output" | grep -q "On-Error"; then
        log_pass "Go on_error compiles correctly"
    else
        log_fail "Go on_error compilation failed"
    fi
}

# Test 13: Rust try_catch compilation
test_rust_try_catch() {
    log_info "Test 13: Rust try_catch stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, try_catch(process/1, error_handler/1), output/1], [pipeline_name(try_catch_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "try_catch_stage" && \
       echo "$output" | grep -q "Try-Catch"; then
        log_pass "Rust try_catch compiles correctly"
    else
        log_fail "Rust try_catch compilation failed"
    fi
}

# Test 14: Rust retry compilation
test_rust_retry() {
    log_info "Test 14: Rust retry stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, retry(fetch/1, 3), output/1], [pipeline_name(retry_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "retry_stage" && \
       echo "$output" | grep -q "Retry"; then
        log_pass "Rust retry compiles correctly"
    else
        log_fail "Rust retry compilation failed"
    fi
}

# Test 15: Rust on_error compilation
test_rust_on_error() {
    log_info "Test 15: Rust on_error stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, on_error(log_error/1), output/1], [pipeline_name(on_error_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "on_error_stage" && \
       echo "$output" | grep -q "On-Error"; then
        log_pass "Rust on_error compiles correctly"
    else
        log_fail "Rust on_error compilation failed"
    fi
}

# Test 16: Nested try_catch validation
test_nested_try_catch() {
    log_info "Test 16: Nested try_catch validation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(retry(fetch/1, 3), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Nested try_catch(retry(...), handler) validation accepted"
    else
        log_fail "Nested try_catch validation failed: $output"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Pipeline Error Handling Stages Tests"
    echo "=========================================="
    echo ""

    test_validation_try_catch
    test_validation_retry_simple
    test_validation_retry_options
    test_validation_on_error
    test_validation_retry_invalid
    test_python_try_catch
    test_python_retry
    test_python_retry_options
    test_python_on_error
    test_go_try_catch
    test_go_retry
    test_go_on_error
    test_rust_try_catch
    test_rust_retry
    test_rust_on_error
    test_nested_try_catch

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
