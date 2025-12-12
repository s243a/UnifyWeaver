#!/bin/bash
# test_enhanced_chaining_multi_target.sh - End-to-end tests for enhanced pipeline chaining
# Tests enhanced chaining (fan-out, merge, routing, filter) across Go, C#, Rust, PowerShell

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_enhanced_chaining_multi_target_test"
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

cleanup() {
    rm -rf "$OUTPUT_DIR"
}

setup() {
    cleanup
    mkdir -p "$OUTPUT_DIR"
}

# Test 1: Go unit tests pass
test_go_unit_tests() {
    log_info "Test 1: Go enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/go_target), test_go_enhanced_chaining" -t halt 2>&1 | grep -q "All Go Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "Go unit tests pass"
    else
        log_fail "Go unit tests failed"
    fi
}

# Test 2: C# unit tests pass
test_csharp_unit_tests() {
    log_info "Test 2: C# enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/csharp_target), test_csharp_enhanced_chaining" -t halt 2>&1 | grep -q "All C# Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "C# unit tests pass"
    else
        log_fail "C# unit tests failed"
    fi
}

# Test 3: Rust unit tests pass
test_rust_unit_tests() {
    log_info "Test 3: Rust enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/rust_target), test_rust_enhanced_chaining" -t halt 2>&1 | grep -q "All Rust Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "Rust unit tests pass"
    else
        log_fail "Rust unit tests failed"
    fi
}

# Test 4: PowerShell unit tests pass
test_powershell_unit_tests() {
    log_info "Test 4: PowerShell enhanced chaining unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/powershell_target), test_powershell_enhanced_chaining" -t halt 2>&1 | grep -q "All PowerShell Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "PowerShell unit tests pass"
    else
        log_fail "PowerShell unit tests failed"
    fi
}

# Test 5: Go generates valid code
test_go_code_generation() {
    log_info "Test 5: Go code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/go_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/go_target).

test_compile :-
    compile_go_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(enhancedPipe), output_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_chaining_multi_target_test/go_enhanced.go', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_chaining_multi_target_test/go_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/go_enhanced.go" ]; then
            if grep -q "func fanOutRecords" "$OUTPUT_DIR/go_enhanced.go" && \
               grep -q "func filterRecords" "$OUTPUT_DIR/go_enhanced.go" && \
               grep -q "func enhancedPipe" "$OUTPUT_DIR/go_enhanced.go"; then
                log_pass "Go code generated correctly"
            else
                log_fail "Go code missing expected patterns"
            fi
        else
            log_fail "Go file not generated"
        fi
    else
        log_fail "Go compilation failed"
    fi
}

# Test 6: C# generates valid code
test_csharp_code_generation() {
    log_info "Test 6: C# code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/csharp_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/csharp_target).

test_compile :-
    compile_csharp_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name('EnhancedPipe'), output_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_chaining_multi_target_test/csharp_enhanced.cs', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_chaining_multi_target_test/csharp_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/csharp_enhanced.cs" ]; then
            if grep -q "FanOutRecords" "$OUTPUT_DIR/csharp_enhanced.cs" && \
               grep -q "FilterRecords" "$OUTPUT_DIR/csharp_enhanced.cs" && \
               grep -q "EnhancedPipe" "$OUTPUT_DIR/csharp_enhanced.cs"; then
                log_pass "C# code generated correctly"
            else
                log_fail "C# code missing expected patterns"
            fi
        else
            log_fail "C# file not generated"
        fi
    else
        log_fail "C# compilation failed"
    fi
}

# Test 7: Rust generates valid code
test_rust_code_generation() {
    log_info "Test 7: Rust code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/rust_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/rust_target).

test_compile :-
    compile_rust_enhanced_pipeline([
        extract/1,
        filter_by(is_active),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name(enhanced_pipe), output_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_chaining_multi_target_test/rust_enhanced.rs', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_chaining_multi_target_test/rust_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/rust_enhanced.rs" ]; then
            if grep -q "fn fan_out_records" "$OUTPUT_DIR/rust_enhanced.rs" && \
               grep -q "fn filter_records" "$OUTPUT_DIR/rust_enhanced.rs" && \
               grep -q "fn enhanced_pipe" "$OUTPUT_DIR/rust_enhanced.rs"; then
                log_pass "Rust code generated correctly"
            else
                log_fail "Rust code missing expected patterns"
            fi
        else
            log_fail "Rust file not generated"
        fi
    else
        log_fail "Rust compilation failed"
    fi
}

# Test 8: PowerShell generates valid code
test_powershell_code_generation() {
    log_info "Test 8: PowerShell code generation"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/ps_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/powershell_target).

test_compile :-
    compile_powershell_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name('Invoke-EnhancedPipe'), output_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_chaining_multi_target_test/ps_enhanced.ps1', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_chaining_multi_target_test/ps_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/ps_enhanced.ps1" ]; then
            if grep -q "Invoke-FanOut" "$OUTPUT_DIR/ps_enhanced.ps1" && \
               grep -q "Select-FilteredRecords" "$OUTPUT_DIR/ps_enhanced.ps1" && \
               grep -q "Invoke-EnhancedPipe" "$OUTPUT_DIR/ps_enhanced.ps1"; then
                log_pass "PowerShell code generated correctly"
            else
                log_fail "PowerShell code missing expected patterns"
            fi
        else
            log_fail "PowerShell file not generated"
        fi
    else
        log_fail "PowerShell compilation failed"
    fi
}

# Test 9: Complex pipeline with all patterns (Go)
test_complex_pipeline_go() {
    log_info "Test 9: Complex pipeline (Go)"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/complex_go_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/go_target).

test_compile :-
    compile_go_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(hasError, [(true, errorLog/1), (false, transform/1)]),
        output/1
    ], [pipeline_name(complexPipe), output_format(jsonl)], Code),
    atom_string(CodeAtom, Code),
    open('output_enhanced_chaining_multi_target_test/complex_go.go', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_enhanced_chaining_multi_target_test/complex_go_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/complex_go.go" ]; then
            if grep -q "Fan-out to 3" "$OUTPUT_DIR/complex_go.go" && \
               grep -q "Conditional routing" "$OUTPUT_DIR/complex_go.go" && \
               grep -q "Filter by isActive" "$OUTPUT_DIR/complex_go.go" && \
               grep -q "Merge" "$OUTPUT_DIR/complex_go.go"; then
                log_pass "Complex Go pipeline has all patterns"
            else
                log_fail "Complex Go pipeline missing patterns"
            fi
        else
            log_fail "Complex Go file not generated"
        fi
    else
        log_fail "Complex Go compilation failed"
    fi
}

# Test 10: Python enhanced chaining (existing feature)
test_python_enhanced_chaining() {
    log_info "Test 10: Python enhanced chaining (existing feature)"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/python_target), test_enhanced_pipeline_chaining" -t halt 2>&1 | grep -q "All Enhanced Pipeline Chaining Tests Passed"; then
        log_pass "Python enhanced chaining tests pass"
    else
        log_fail "Python enhanced chaining tests failed"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Enhanced Chaining Multi-Target E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_go_unit_tests
    test_csharp_unit_tests
    test_rust_unit_tests
    test_powershell_unit_tests
    test_go_code_generation
    test_csharp_code_generation
    test_rust_code_generation
    test_powershell_code_generation
    test_complex_pipeline_go
    test_python_enhanced_chaining

    echo ""
    echo "=========================================="
    echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo "=========================================="

    cleanup

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
