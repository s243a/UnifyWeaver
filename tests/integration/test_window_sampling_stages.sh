#!/bin/bash
# test_window_sampling_stages.sh - Integration tests for window, sampling, partition, take/skip pipeline stages
# Tests: window(N), sliding_window(N, Step), sample(N), take_every(N), partition(Pred),
#        take(N), skip(N), take_while(Pred), skip_while(Pred)

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

# Test 1: Validation accepts window(N)
test_validation_window() {
    log_info "Test 1: Validation accepts window(N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, window(5), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "window(5) validation accepted"
    else
        log_fail "window(5) validation failed: $output"
    fi
}

# Test 2: Validation accepts sliding_window(N, Step)
test_validation_sliding_window() {
    log_info "Test 2: Validation accepts sliding_window(N, Step)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, sliding_window(10, 2), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "sliding_window(10, 2) validation accepted"
    else
        log_fail "sliding_window(10, 2) validation failed: $output"
    fi
}

# Test 3: Validation accepts sample(N)
test_validation_sample() {
    log_info "Test 3: Validation accepts sample(N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, sample(100), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "sample(100) validation accepted"
    else
        log_fail "sample(100) validation failed: $output"
    fi
}

# Test 4: Validation accepts take_every(N)
test_validation_take_every() {
    log_info "Test 4: Validation accepts take_every(N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, take_every(3), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "take_every(3) validation accepted"
    else
        log_fail "take_every(3) validation failed: $output"
    fi
}

# Test 5: Validation accepts partition(Pred)
test_validation_partition() {
    log_info "Test 5: Validation accepts partition(Pred)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, partition(is_valid), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "partition(is_valid) validation accepted"
    else
        log_fail "partition(is_valid) validation failed: $output"
    fi
}

# Test 6: Validation accepts take(N)
test_validation_take() {
    log_info "Test 6: Validation accepts take(N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, take(10), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "take(10) validation accepted"
    else
        log_fail "take(10) validation failed: $output"
    fi
}

# Test 7: Validation accepts skip(N)
test_validation_skip() {
    log_info "Test 7: Validation accepts skip(N)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, skip(5), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "skip(5) validation accepted"
    else
        log_fail "skip(5) validation failed: $output"
    fi
}

# Test 8: Validation accepts take_while(Pred)
test_validation_take_while() {
    log_info "Test 8: Validation accepts take_while(Pred)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, take_while(is_active), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "take_while(is_active) validation accepted"
    else
        log_fail "take_while(is_active) validation failed: $output"
    fi
}

# Test 9: Validation accepts skip_while(Pred)
test_validation_skip_while() {
    log_info "Test 9: Validation accepts skip_while(Pred)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, skip_while(is_header), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "skip_while(is_header) validation accepted"
    else
        log_fail "skip_while(is_header) validation failed: $output"
    fi
}

# Test 10: Validation rejects window with invalid N
test_validation_window_invalid() {
    log_info "Test 10: Validation rejects window with invalid N"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, window(0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "window(0) correctly rejected"
    else
        log_fail "window(0) should have been rejected"
    fi
}

# Test 11: Validation rejects sample with invalid N
test_validation_sample_invalid() {
    log_info "Test 11: Validation rejects sample with invalid N"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, sample(0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "sample(0) correctly rejected"
    else
        log_fail "sample(0) should have been rejected"
    fi
}

# Test 12: Validation rejects take_every with invalid N
test_validation_take_every_invalid() {
    log_info "Test 12: Validation rejects take_every with invalid N"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, take_every(0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -qv "Errors: \[\]"; then
        log_pass "take_every(0) correctly rejected"
    else
        log_fail "take_every(0) should have been rejected"
    fi
}

# ============================================
# PYTHON COMPILATION TESTS
# ============================================

# Test 13: Python window compilation
test_python_window() {
    log_info "Test 13: Python window stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, window(5), output/1], [pipeline_name(window_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "window_stage"; then
        log_pass "Python window compiles correctly"
    else
        log_fail "Python window compilation failed"
    fi
}

# Test 14: Python sliding_window compilation
test_python_sliding_window() {
    log_info "Test 14: Python sliding_window stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, sliding_window(10, 2), output/1], [pipeline_name(sliding_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sliding_window_stage"; then
        log_pass "Python sliding_window compiles correctly"
    else
        log_fail "Python sliding_window compilation failed"
    fi
}

# Test 15: Python sample compilation
test_python_sample() {
    log_info "Test 15: Python sample stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, sample(100), output/1], [pipeline_name(sample_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sample_stage"; then
        log_pass "Python sample compiles correctly"
    else
        log_fail "Python sample compilation failed"
    fi
}

# Test 16: Python take/skip compilation
test_python_take_skip() {
    log_info "Test 16: Python take/skip stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, take(10), skip(5), output/1], [pipeline_name(take_skip_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "take_stage" && echo "$output" | grep -q "skip_stage"; then
        log_pass "Python take/skip compiles correctly"
    else
        log_fail "Python take/skip compilation failed"
    fi
}

# Test 17: Python partition compilation
test_python_partition() {
    log_info "Test 17: Python partition stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, partition(is_valid), output/1], [pipeline_name(partition_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "partition_stage"; then
        log_pass "Python partition compiles correctly"
    else
        log_fail "Python partition compilation failed"
    fi
}

# ============================================
# GO COMPILATION TESTS
# ============================================

# Test 18: Go window compilation
test_go_window() {
    log_info "Test 18: Go window stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, window(5), output/1], [pipeline_name(windowTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "windowStage"; then
        log_pass "Go window compiles correctly"
    else
        log_fail "Go window compilation failed"
    fi
}

# Test 19: Go sliding_window compilation
test_go_sliding_window() {
    log_info "Test 19: Go sliding_window stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, sliding_window(10, 2), output/1], [pipeline_name(slidingTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "slidingWindowStage"; then
        log_pass "Go sliding_window compiles correctly"
    else
        log_fail "Go sliding_window compilation failed"
    fi
}

# Test 20: Go sample compilation
test_go_sample() {
    log_info "Test 20: Go sample stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, sample(100), output/1], [pipeline_name(sampleTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sampleStage"; then
        log_pass "Go sample compiles correctly"
    else
        log_fail "Go sample compilation failed"
    fi
}

# Test 21: Go take/skip compilation
test_go_take_skip() {
    log_info "Test 21: Go take/skip stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, take(10), skip(5), output/1], [pipeline_name(takeSkipTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "takeStage" && echo "$output" | grep -q "skipStage"; then
        log_pass "Go take/skip compiles correctly"
    else
        log_fail "Go take/skip compilation failed"
    fi
}

# Test 22: Go partition compilation
test_go_partition() {
    log_info "Test 22: Go partition stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, partition(is_valid), output/1], [pipeline_name(partitionTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "partitionStage"; then
        log_pass "Go partition compiles correctly"
    else
        log_fail "Go partition compilation failed"
    fi
}

# ============================================
# RUST COMPILATION TESTS
# ============================================

# Test 23: Rust window compilation
test_rust_window() {
    log_info "Test 23: Rust window stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, window(5), output/1], [pipeline_name(window_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "window_stage"; then
        log_pass "Rust window compiles correctly"
    else
        log_fail "Rust window compilation failed"
    fi
}

# Test 24: Rust sliding_window compilation
test_rust_sliding_window() {
    log_info "Test 24: Rust sliding_window stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, sliding_window(10, 2), output/1], [pipeline_name(sliding_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sliding_window_stage"; then
        log_pass "Rust sliding_window compiles correctly"
    else
        log_fail "Rust sliding_window compilation failed"
    fi
}

# Test 25: Rust sample compilation
test_rust_sample() {
    log_info "Test 25: Rust sample stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, sample(100), output/1], [pipeline_name(sample_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sample_stage"; then
        log_pass "Rust sample compiles correctly"
    else
        log_fail "Rust sample compilation failed"
    fi
}

# Test 26: Rust take/skip compilation
test_rust_take_skip() {
    log_info "Test 26: Rust take/skip stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, take(10), skip(5), output/1], [pipeline_name(take_skip_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "take_stage" && echo "$output" | grep -q "skip_stage"; then
        log_pass "Rust take/skip compiles correctly"
    else
        log_fail "Rust take/skip compilation failed"
    fi
}

# Test 27: Rust partition compilation
test_rust_partition() {
    log_info "Test 27: Rust partition stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, partition(is_valid), output/1], [pipeline_name(partition_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "partition_stage"; then
        log_pass "Rust partition compiles correctly"
    else
        log_fail "Rust partition compilation failed"
    fi
}

# ============================================
# COMBINED STAGE TESTS
# ============================================

# Test 28: Combined window and take
test_combined_window_take() {
    log_info "Test 28: Combined window and take validation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, window(10), take(5), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "Combined window and take validation accepted"
    else
        log_fail "Combined window and take validation failed: $output"
    fi
}

# Test 29: Window with error handling
test_window_with_try_catch() {
    log_info "Test 29: Window with error handling"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, try_catch(window(5), error_handler/1), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "try_catch(window(...), handler) validation accepted"
    else
        log_fail "try_catch with window validation failed: $output"
    fi
}

# Test 30: Sampling with buffer
test_sample_with_buffer() {
    log_info "Test 30: Sampling with buffer"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, buffer(10), sample(50), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "buffer + sample validation accepted"
    else
        log_fail "buffer + sample validation failed: $output"
    fi
}

# Test 31: Take while and skip while
test_take_while_skip_while() {
    log_info "Test 31: Take while and skip while validation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, skip_while(is_header), take_while(is_active), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "skip_while + take_while validation accepted"
    else
        log_fail "skip_while + take_while validation failed: $output"
    fi
}

# Test 32: Partition with parallel processing
test_partition_parallel() {
    log_info "Test 32: Partition with parallel processing"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, partition(is_valid), parallel([process_valid/1, process_invalid/1]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "partition + parallel validation accepted"
    else
        log_fail "partition + parallel validation failed: $output"
    fi
}

# ============================================
# MAIN TEST RUNNER
# ============================================

main() {
    echo "=================================================="
    echo "  Pipeline Window/Sampling/Partition Stage Tests"
    echo "=================================================="
    echo ""

    # Validation tests
    test_validation_window
    test_validation_sliding_window
    test_validation_sample
    test_validation_take_every
    test_validation_partition
    test_validation_take
    test_validation_skip
    test_validation_take_while
    test_validation_skip_while
    test_validation_window_invalid
    test_validation_sample_invalid
    test_validation_take_every_invalid

    # Python compilation tests
    test_python_window
    test_python_sliding_window
    test_python_sample
    test_python_take_skip
    test_python_partition

    # Go compilation tests
    test_go_window
    test_go_sliding_window
    test_go_sample
    test_go_take_skip
    test_go_partition

    # Rust compilation tests
    test_rust_window
    test_rust_sliding_window
    test_rust_sample
    test_rust_take_skip
    test_rust_partition

    # Combined stage tests
    test_combined_window_take
    test_window_with_try_catch
    test_sample_with_buffer
    test_take_while_skip_while
    test_partition_parallel

    echo ""
    echo "=================================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=================================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main
