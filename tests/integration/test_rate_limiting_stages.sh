#!/bin/bash
# test_rate_limiting_stages.sh - Integration tests for pipeline rate limiting stages
# Tests rate_limit(N, Per) and throttle(Ms)

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

# Test 1: Validation accepts rate_limit(N, second)
test_validation_rate_limit_second() {
    log_info "Test 1: Validation accepts rate_limit(N, second)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(10, second), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "rate_limit(10, second) validation accepted"
    else
        log_fail "rate_limit(10, second) validation failed: $output"
    fi
}

# Test 2: Validation accepts rate_limit(N, minute)
test_validation_rate_limit_minute() {
    log_info "Test 2: Validation accepts rate_limit(N, minute)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(100, minute), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "rate_limit(100, minute) validation accepted"
    else
        log_fail "rate_limit(100, minute) validation failed: $output"
    fi
}

# Test 3: Validation accepts rate_limit(N, hour)
test_validation_rate_limit_hour() {
    log_info "Test 3: Validation accepts rate_limit(N, hour)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(1000, hour), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "rate_limit(1000, hour) validation accepted"
    else
        log_fail "rate_limit(1000, hour) validation failed: $output"
    fi
}

# Test 4: Validation accepts rate_limit(N, ms(X))
test_validation_rate_limit_ms() {
    log_info "Test 4: Validation accepts rate_limit(N, ms(X))"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(5, ms(500)), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "rate_limit(5, ms(500)) validation accepted"
    else
        log_fail "rate_limit(5, ms(500)) validation failed: $output"
    fi
}

# Test 5: Validation accepts throttle(Ms)
test_validation_throttle() {
    log_info "Test 5: Validation accepts throttle(Ms)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, throttle(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "throttle(100) validation accepted"
    else
        log_fail "throttle(100) validation failed: $output"
    fi
}

# Test 6: Validation rejects rate_limit with invalid N
test_validation_rate_limit_invalid_n() {
    log_info "Test 6: Validation rejects rate_limit with invalid N"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(0, second), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "rate_limit(0, second) correctly rejected"
    else
        log_fail "rate_limit(0, second) should have been rejected"
    fi
}

# Test 7: Validation rejects throttle with invalid ms
test_validation_throttle_invalid_ms() {
    log_info "Test 7: Validation rejects throttle with invalid ms"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, throttle(-10), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "throttle(-10) correctly rejected"
    else
        log_fail "throttle(-10) should have been rejected"
    fi
}

# Test 8: Python rate_limit compilation
test_python_rate_limit() {
    log_info "Test 8: Python rate_limit stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, rate_limit(10, second), output/1], [pipeline_name(rate_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "rate_limit_stage"; then
        log_pass "Python rate_limit compiles correctly"
    else
        log_fail "Python rate_limit compilation failed"
    fi
}

# Test 9: Python throttle compilation
test_python_throttle() {
    log_info "Test 9: Python throttle stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, throttle(100), output/1], [pipeline_name(throttle_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "throttle_stage" && \
       echo "$output" | grep -q "100"; then
        log_pass "Python throttle compiles correctly"
    else
        log_fail "Python throttle compilation failed"
    fi
}

# Test 10: Go rate_limit compilation
test_go_rate_limit() {
    log_info "Test 10: Go rate_limit stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, rate_limit(10, second), output/1], [pipeline_name(rateTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "rateLimitStage"; then
        log_pass "Go rate_limit compiles correctly"
    else
        log_fail "Go rate_limit compilation failed"
    fi
}

# Test 11: Go throttle compilation
test_go_throttle() {
    log_info "Test 11: Go throttle stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, throttle(100), output/1], [pipeline_name(throttleTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "throttleStage" && \
       echo "$output" | grep -q "100"; then
        log_pass "Go throttle compiles correctly"
    else
        log_fail "Go throttle compilation failed"
    fi
}

# Test 12: Rust rate_limit compilation
test_rust_rate_limit() {
    log_info "Test 12: Rust rate_limit stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, rate_limit(10, second), output/1], [pipeline_name(rate_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "rate_limit_stage"; then
        log_pass "Rust rate_limit compiles correctly"
    else
        log_fail "Rust rate_limit compilation failed"
    fi
}

# Test 13: Rust throttle compilation
test_rust_throttle() {
    log_info "Test 13: Rust throttle stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, throttle(100), output/1], [pipeline_name(throttle_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "throttle_stage" && \
       echo "$output" | grep -q "100"; then
        log_pass "Rust throttle compiles correctly"
    else
        log_fail "Rust throttle compilation failed"
    fi
}

# Test 14: Combined rate limiting with other stages
test_combined_rate_limiting() {
    log_info "Test 14: Combined rate limiting with other stages"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(10, second), filter_by(active), throttle(50), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Combined rate_limit and throttle validation accepted"
    else
        log_fail "Combined rate limiting validation failed: $output"
    fi
}

# Test 15: Rate limiting with error handling
test_rate_limit_with_try_catch() {
    log_info "Test 15: Rate limiting with error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(rate_limit(5, second), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(rate_limit(...), handler) validation accepted"
    else
        log_fail "try_catch with rate_limit validation failed: $output"
    fi
}

# Test 16: Rate limiting with timeout
test_rate_limit_with_timeout() {
    log_info "Test 16: Rate limiting with timeout"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, timeout(rate_limit(10, second), 5000), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "timeout(rate_limit(...), ms) validation accepted"
    else
        log_fail "timeout with rate_limit validation failed: $output"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Pipeline Rate Limiting Stage Tests"
    echo "=========================================="
    echo ""

    test_validation_rate_limit_second
    test_validation_rate_limit_minute
    test_validation_rate_limit_hour
    test_validation_rate_limit_ms
    test_validation_throttle
    test_validation_rate_limit_invalid_n
    test_validation_throttle_invalid_ms
    test_python_rate_limit
    test_python_throttle
    test_go_rate_limit
    test_go_throttle
    test_rust_rate_limit
    test_rust_throttle
    test_combined_rate_limiting
    test_rate_limit_with_try_catch
    test_rate_limit_with_timeout

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
