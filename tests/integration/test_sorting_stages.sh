#!/bin/bash
# test_sorting_stages.sh - Integration tests for pipeline sorting stages
# Tests order_by and sort_by stages

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

# Test 1: Validation accepts order_by(field)
test_validation_order_by_simple() {
    log_info "Test 1: Validation accepts order_by(field)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, order_by(timestamp), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "order_by(field) validation accepted"
    else
        log_fail "order_by(field) validation failed: $output"
    fi
}

# Test 2: Validation accepts order_by(field, direction)
test_validation_order_by_direction() {
    log_info "Test 2: Validation accepts order_by(field, direction)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, order_by(timestamp, desc), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "order_by(field, direction) validation accepted"
    else
        log_fail "order_by(field, direction) validation failed: $output"
    fi
}

# Test 3: Validation accepts order_by(list)
test_validation_order_by_list() {
    log_info "Test 3: Validation accepts order_by(field_list)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, order_by([(timestamp, desc), (user_id, asc)]), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "order_by(field_list) validation accepted"
    else
        log_fail "order_by(field_list) validation failed: $output"
    fi
}

# Test 4: Validation accepts sort_by(pred)
test_validation_sort_by() {
    log_info "Test 4: Validation accepts sort_by(comparator)"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/core/pipeline_validation),
        validate_pipeline([parse/1, sort_by(compare_priority), output/1], Errors),
        format('Errors: ~w~n', [Errors])
    " -t halt 2>&1)
    if echo "$output" | grep -q "Errors: \[\]"; then
        log_pass "sort_by(comparator) validation accepted"
    else
        log_fail "sort_by(comparator) validation failed: $output"
    fi
}

# Test 5: Python order_by compilation
test_python_order_by() {
    log_info "Test 5: Python order_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, order_by(timestamp), output/1], [pipeline_name(order_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "order_by_field" && \
       echo "$output" | grep -q "Order by"; then
        log_pass "Python order_by compiles correctly"
    else
        log_fail "Python order_by compilation failed"
    fi
}

# Test 6: Python order_by with direction compilation
test_python_order_by_direction() {
    log_info "Test 6: Python order_by with direction compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, order_by(timestamp, desc), output/1], [pipeline_name(order_desc_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "order_by_field" && \
       echo "$output" | grep -q "desc"; then
        log_pass "Python order_by with direction compiles correctly"
    else
        log_fail "Python order_by with direction compilation failed"
    fi
}

# Test 7: Python sort_by compilation
test_python_sort_by() {
    log_info "Test 7: Python sort_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, sort_by(compare_priority), output/1], [pipeline_name(sort_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sort_by_comparator" && \
       echo "$output" | grep -q "compare_priority"; then
        log_pass "Python sort_by compiles correctly"
    else
        log_fail "Python sort_by compilation failed"
    fi
}

# Test 8: Go order_by compilation
test_go_order_by() {
    log_info "Test 8: Go order_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, order_by(timestamp), output/1], [pipeline_name(orderTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "orderByField" && \
       echo "$output" | grep -q "Order by"; then
        log_pass "Go order_by compiles correctly"
    else
        log_fail "Go order_by compilation failed"
    fi
}

# Test 9: Go sort_by compilation
test_go_sort_by() {
    log_info "Test 9: Go sort_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/go_target),
        compile_go_enhanced_pipeline([parse/1, sort_by(comparePriority), output/1], [pipeline_name(sortTest)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sortByComparator" && \
       echo "$output" | grep -q "comparePriority"; then
        log_pass "Go sort_by compiles correctly"
    else
        log_fail "Go sort_by compilation failed"
    fi
}

# Test 10: Rust order_by compilation
test_rust_order_by() {
    log_info "Test 10: Rust order_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, order_by(timestamp), output/1], [pipeline_name(order_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "order_by_field" && \
       echo "$output" | grep -q "Order by"; then
        log_pass "Rust order_by compiles correctly"
    else
        log_fail "Rust order_by compilation failed"
    fi
}

# Test 11: Rust sort_by compilation
test_rust_sort_by() {
    log_info "Test 11: Rust sort_by stage compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/rust_target),
        compile_rust_enhanced_pipeline([parse/1, sort_by(compare_priority), output/1], [pipeline_name(sort_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "sort_by_comparator" && \
       echo "$output" | grep -q "compare_priority"; then
        log_pass "Rust sort_by compiles correctly"
    else
        log_fail "Rust sort_by compilation failed"
    fi
}

# Test 12: Python multi-field order_by compilation
test_python_order_by_multi() {
    log_info "Test 12: Python multi-field order_by compilation"
    cd "$PROJECT_ROOT"
    local output=$(swipl -g "
        use_module(src/unifyweaver/targets/python_target),
        compile_enhanced_pipeline([parse/1, order_by([(timestamp, desc), (user_id, asc)]), output/1], [pipeline_name(multi_order_test)], Code),
        format('~w', [Code])
    " -t halt 2>&1)
    if echo "$output" | grep -q "order_by_fields" && \
       echo "$output" | grep -q "timestamp" && \
       echo "$output" | grep -q "user_id"; then
        log_pass "Python multi-field order_by compiles correctly"
    else
        log_fail "Python multi-field order_by compilation failed"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Pipeline Sorting Stages Tests"
    echo "=========================================="
    echo ""

    test_validation_order_by_simple
    test_validation_order_by_direction
    test_validation_order_by_list
    test_validation_sort_by
    test_python_order_by
    test_python_order_by_direction
    test_python_sort_by
    test_go_order_by
    test_go_sort_by
    test_rust_order_by
    test_rust_sort_by
    test_python_order_by_multi

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
