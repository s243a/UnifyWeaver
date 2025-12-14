#!/bin/bash
# test_buffer_zip_stages.sh - Integration tests for buffer and zip pipeline stages
# Tests buffer(N), debounce(Ms), and zip(Stages)

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

# Test 1: Validation accepts buffer(N)
test_validation_buffer() {
    log_info "Test 1: Validation accepts buffer(N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, buffer(10), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "buffer(10) validation accepted"
    else
        log_fail "buffer(10) validation failed: $output"
    fi
}

# Test 2: Validation accepts debounce(Ms)
test_validation_debounce() {
    log_info "Test 2: Validation accepts debounce(Ms)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "debounce(100) validation accepted"
    else
        log_fail "debounce(100) validation failed: $output"
    fi
}

# Test 3: Validation accepts zip(Stages)
test_validation_zip() {
    log_info "Test 3: Validation accepts zip(Stages)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, zip([transform/1, enrich/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "zip([transform/1, enrich/1]) validation accepted"
    else
        log_fail "zip validation failed: $output"
    fi
}

# Test 4: Validation rejects buffer with invalid N
test_validation_buffer_invalid() {
    log_info "Test 4: Validation rejects buffer with invalid N"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, buffer(0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "buffer(0) correctly rejected"
    else
        log_fail "buffer(0) should have been rejected"
    fi
}

# Test 5: Validation rejects debounce with invalid Ms
test_validation_debounce_invalid() {
    log_info "Test 5: Validation rejects debounce with invalid Ms"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, debounce(-10), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "debounce(-10) correctly rejected"
    else
        log_fail "debounce(-10) should have been rejected"
    fi
}

# Test 6: Validation rejects empty zip
test_validation_zip_empty() {
    log_info "Test 6: Validation rejects empty zip"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, zip([]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "zip([]) correctly rejected"
    else
        log_fail "zip([]) should have been rejected"
    fi
}

# Test 7: Python buffer compilation
test_python_buffer() {
    log_info "Test 7: Python buffer stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, buffer(10), output/1], [pipeline_name(buffer_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "buffer_stage"; then
        log_pass "Python buffer compiles correctly"
    else
        log_fail "Python buffer compilation failed"
    fi
}

# Test 8: Python debounce compilation
test_python_debounce() {
    log_info "Test 8: Python debounce stage compilation"
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

# Test 9: Python zip compilation
test_python_zip() {
    log_info "Test 9: Python zip stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, zip([transform/1, enrich/1]), output/1], [pipeline_name(zip_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "zip_stage"; then
        log_pass "Python zip compiles correctly"
    else
        log_fail "Python zip compilation failed"
    fi
}

# Test 10: Go buffer compilation
test_go_buffer() {
    log_info "Test 10: Go buffer stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, buffer(10), output/1], [pipeline_name(bufferTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "bufferStage"; then
        log_pass "Go buffer compiles correctly"
    else
        log_fail "Go buffer compilation failed"
    fi
}

# Test 11: Go debounce compilation
test_go_debounce() {
    log_info "Test 11: Go debounce stage compilation"
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

# Test 12: Go zip compilation
test_go_zip() {
    log_info "Test 12: Go zip stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, zip([transform/1, enrich/1]), output/1], [pipeline_name(zipTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "zipStage"; then
        log_pass "Go zip compiles correctly"
    else
        log_fail "Go zip compilation failed"
    fi
}

# Test 13: Rust buffer compilation
test_rust_buffer() {
    log_info "Test 13: Rust buffer stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, buffer(10), output/1], [pipeline_name(buffer_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "buffer_stage"; then
        log_pass "Rust buffer compiles correctly"
    else
        log_fail "Rust buffer compilation failed"
    fi
}

# Test 14: Rust debounce compilation
test_rust_debounce() {
    log_info "Test 14: Rust debounce stage compilation"
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

# Test 15: Rust zip compilation
test_rust_zip() {
    log_info "Test 15: Rust zip stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, zip([transform/1, enrich/1]), output/1], [pipeline_name(zip_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "zip_stage"; then
        log_pass "Rust zip compiles correctly"
    else
        log_fail "Rust zip compilation failed"
    fi
}

# Test 16: Combined buffer and zip
test_combined_buffer_zip() {
    log_info "Test 16: Combined buffer and zip validation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, buffer(5), zip([transform/1, enrich/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Combined buffer and zip validation accepted"
    else
        log_fail "Combined buffer and zip validation failed: $output"
    fi
}

# Test 17: Buffer with error handling
test_buffer_with_try_catch() {
    log_info "Test 17: Buffer with error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(buffer(10), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(buffer(...), handler) validation accepted"
    else
        log_fail "try_catch with buffer validation failed: $output"
    fi
}

# Test 18: Zip with rate limiting
test_zip_with_rate_limit() {
    log_info "Test 18: Zip with rate limiting"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, rate_limit(10, second), zip([transform/1, enrich/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "rate_limit + zip validation accepted"
    else
        log_fail "rate_limit + zip validation failed: $output"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Pipeline Buffer and Zip Stage Tests"
    echo "=========================================="
    echo ""

    test_validation_buffer
    test_validation_debounce
    test_validation_zip
    test_validation_buffer_invalid
    test_validation_debounce_invalid
    test_validation_zip_empty
    test_python_buffer
    test_python_debounce
    test_python_zip
    test_go_buffer
    test_go_debounce
    test_go_zip
    test_rust_buffer
    test_rust_debounce
    test_rust_zip
    test_combined_buffer_zip
    test_buffer_with_try_catch
    test_zip_with_rate_limit

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
