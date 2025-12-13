#!/bin/bash
# test_aggregation_stages.sh - Integration tests for pipeline aggregation stages
# Tests unique, first, last, group_by, reduce, scan stages

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

# Test 1: Validation accepts unique(Field)
test_validation_unique() {
    log_info "Test 1: Validation accepts unique(field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, unique(user_id), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "unique(field) validation accepted"
    else
        log_fail "unique(field) validation failed: $output"
    fi
}

# Test 2: Validation accepts group_by(Field, Agg)
test_validation_group_by() {
    log_info "Test 2: Validation accepts group_by(field, aggregations)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, group_by(category, [count, sum(amount)]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "group_by validation accepted"
    else
        log_fail "group_by validation failed: $output"
    fi
}

# Test 3: Validation accepts reduce(Pred, Init)
test_validation_reduce() {
    log_info "Test 3: Validation accepts reduce(pred, init)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, reduce(running_total, 0), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "reduce validation accepted"
    else
        log_fail "reduce validation failed: $output"
    fi
}

# Test 4: Python unique compilation
test_python_unique() {
    log_info "Test 4: Python unique stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, unique(user_id), output/1], [pipeline_name(unique_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "unique_by_field" && \
       echo "$output" | grep -q "Unique: keep first record"; then
        log_pass "Python unique compiles correctly"
    else
        log_fail "Python unique compilation failed"
    fi
}

# Test 5: Python group_by compilation
test_python_group_by() {
    log_info "Test 5: Python group_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, group_by(category, [count, sum(amount)]), output/1], [pipeline_name(groupby_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "group_by_field" && \
       echo "$output" | grep -q "'count', 'count'" && \
       echo "$output" | grep -q "'sum', 'sum', 'amount'"; then
        log_pass "Python group_by compiles correctly"
    else
        log_fail "Python group_by compilation failed"
    fi
}

# Test 6: Python reduce compilation
test_python_reduce() {
    log_info "Test 6: Python reduce stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, reduce(running_total, 0), output/1], [pipeline_name(reduce_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "reduce_records" && \
       echo "$output" | grep -q "running_total, 0"; then
        log_pass "Python reduce compiles correctly"
    else
        log_fail "Python reduce compilation failed"
    fi
}

# Test 7: Go unique compilation
test_go_unique() {
    log_info "Test 7: Go unique stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, unique(user_id), output/1], [pipeline_name(uniqueTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "uniqueByField" && \
       echo "$output" | grep -q "Unique: keep first record"; then
        log_pass "Go unique compiles correctly"
    else
        log_fail "Go unique compilation failed"
    fi
}

# Test 8: Go group_by compilation
test_go_group_by() {
    log_info "Test 8: Go group_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, group_by(category, [count, sum(amount)]), output/1], [pipeline_name(groupbyTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "groupByField" && \
       echo "$output" | grep -q "Aggregation"; then
        log_pass "Go group_by compiles correctly"
    else
        log_fail "Go group_by compilation failed"
    fi
}

# Test 9: Rust unique compilation
test_rust_unique() {
    log_info "Test 9: Rust unique stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, unique(user_id), output/1], [pipeline_name(unique_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "unique_by_field" && \
       echo "$output" | grep -q "Unique: keep first record"; then
        log_pass "Rust unique compiles correctly"
    else
        log_fail "Rust unique compilation failed"
    fi
}

# Test 10: Rust group_by compilation
test_rust_group_by() {
    log_info "Test 10: Rust group_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, group_by(category, [count, sum(amount)]), output/1], [pipeline_name(groupby_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "group_by_field" && \
       echo "$output" | grep -q "AggType::Count" && \
       echo "$output" | grep -q "AggType::Sum"; then
        log_pass "Rust group_by compiles correctly"
    else
        log_fail "Rust group_by compilation failed"
    fi
}

# Test 11: Python last stage
test_python_last() {
    log_info "Test 11: Python last stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, last(user_id), output/1], [pipeline_name(last_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "last_by_field" && \
       echo "$output" | grep -q "Last: keep last record"; then
        log_pass "Python last compiles correctly"
    else
        log_fail "Python last compilation failed"
    fi
}

# Test 12: Python scan stage
test_python_scan() {
    log_info "Test 12: Python scan stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, scan(running_sum, 0), output/1], [pipeline_name(scan_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "scan_records" && \
       echo "$output" | grep -q "Scan: running fold"; then
        log_pass "Python scan compiles correctly"
    else
        log_fail "Python scan compilation failed"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Pipeline Aggregation Stages Tests"
    echo "=========================================="
    echo ""

    test_validation_unique
    test_validation_group_by
    test_validation_reduce
    test_python_unique
    test_python_group_by
    test_python_reduce
    test_go_unique
    test_go_group_by
    test_rust_unique
    test_rust_group_by
    test_python_last
    test_python_scan

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
