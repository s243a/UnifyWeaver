#!/bin/bash
# test_timeout_stage.sh - Integration tests for pipeline timeout stage
# Tests timeout(stage, ms) and timeout(stage, ms, fallback)

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

# Test 1: Validation accepts timeout(stage, ms)
test_validation_timeout_simple() {
    log_info "Test 1: Validation accepts timeout(stage, ms)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, timeout(fetch/1, 5000), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "timeout(stage, ms) validation accepted"
    else
        log_fail "timeout(stage, ms) validation failed: $output"
    fi
}

# Test 2: Validation accepts timeout(stage, ms, fallback)
test_validation_timeout_fallback() {
    log_info "Test 2: Validation accepts timeout(stage, ms, fallback)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, timeout(fetch/1, 5000, use_cache/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "timeout(stage, ms, fallback) validation accepted"
    else
        log_fail "timeout(stage, ms, fallback) validation failed: $output"
    fi
}

# Test 3: Validation rejects timeout with invalid ms
test_validation_timeout_invalid_ms() {
    log_info "Test 3: Validation rejects timeout with invalid ms"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, timeout(fetch/1, 0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "timeout with invalid ms correctly rejected"
    else
        log_fail "timeout with invalid ms should have been rejected"
    fi
}

# Test 4: Validation rejects timeout with negative ms
test_validation_timeout_negative_ms() {
    log_info "Test 4: Validation rejects timeout with negative ms"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, timeout(fetch/1, -100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "timeout with negative ms correctly rejected"
    else
        log_fail "timeout with negative ms should have been rejected"
    fi
}

# Test 5: Python timeout compilation
test_python_timeout() {
    log_info "Test 5: Python timeout stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, timeout(fetch/1, 5000), output/1], [pipeline_name(timeout_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "timeout_stage" && \
       echo "$output" | grep -q "5000"; then
        log_pass "Python timeout compiles correctly"
    else
        log_fail "Python timeout compilation failed"
    fi
}

# Test 6: Python timeout with fallback compilation
test_python_timeout_fallback() {
    log_info "Test 6: Python timeout with fallback compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, timeout(fetch/1, 5000, use_cache/1), output/1], [pipeline_name(timeout_fb_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "timeout_stage_with_fallback" && \
       echo "$output" | grep -q "5000" && \
       echo "$output" | grep -q "use_cache"; then
        log_pass "Python timeout with fallback compiles correctly"
    else
        log_fail "Python timeout with fallback compilation failed"
    fi
}

# Test 7: Go timeout compilation
test_go_timeout() {
    log_info "Test 7: Go timeout stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, timeout(fetch/1, 5000), output/1], [pipeline_name(timeoutTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "timeoutStage" && \
       echo "$output" | grep -q "5000"; then
        log_pass "Go timeout compiles correctly"
    else
        log_fail "Go timeout compilation failed"
    fi
}

# Test 8: Go timeout with fallback compilation
test_go_timeout_fallback() {
    log_info "Test 8: Go timeout with fallback compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, timeout(fetch/1, 5000, useCache/1), output/1], [pipeline_name(timeoutFbTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "timeoutStageWithFallback" && \
       echo "$output" | grep -q "5000" && \
       echo "$output" | grep -q "useCache"; then
        log_pass "Go timeout with fallback compiles correctly"
    else
        log_fail "Go timeout with fallback compilation failed"
    fi
}

# Test 9: Rust timeout compilation
test_rust_timeout() {
    log_info "Test 9: Rust timeout stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, timeout(fetch/1, 5000), output/1], [pipeline_name(timeout_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "timeout_stage" && \
       echo "$output" | grep -q "5000"; then
        log_pass "Rust timeout compiles correctly"
    else
        log_fail "Rust timeout compilation failed"
    fi
}

# Test 10: Rust timeout with fallback compilation
test_rust_timeout_fallback() {
    log_info "Test 10: Rust timeout with fallback compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, timeout(fetch/1, 5000, use_cache/1), output/1], [pipeline_name(timeout_fb_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "timeout_stage_with_fallback" && \
       echo "$output" | grep -q "5000" && \
       echo "$output" | grep -q "use_cache"; then
        log_pass "Rust timeout with fallback compiles correctly"
    else
        log_fail "Rust timeout with fallback compilation failed"
    fi
}

# Test 11: Nested timeout validation
test_nested_timeout() {
    log_info "Test 11: Nested timeout validation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, timeout(retry(fetch/1, 3), 10000), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Nested timeout(retry(...)) validation accepted"
    else
        log_fail "Nested timeout validation failed: $output"
    fi
}

# Test 12: Timeout combined with try_catch
test_timeout_with_try_catch() {
    log_info "Test 12: Timeout combined with try_catch"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(timeout(fetch/1, 5000), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(timeout(...), handler) validation accepted"
    else
        log_fail "try_catch with timeout validation failed: $output"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Pipeline Timeout Stage Tests"
    echo "=========================================="
    echo ""

    test_validation_timeout_simple
    test_validation_timeout_fallback
    test_validation_timeout_invalid_ms
    test_validation_timeout_negative_ms
    test_python_timeout
    test_python_timeout_fallback
    test_go_timeout
    test_go_timeout_fallback
    test_rust_timeout
    test_rust_timeout_fallback
    test_nested_timeout
    test_timeout_with_try_catch

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
