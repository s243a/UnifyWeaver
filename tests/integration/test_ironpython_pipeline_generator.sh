#!/bin/bash
# test_ironpython_pipeline_generator.sh - End-to-end tests for IronPython pipeline generator mode
# Tests fixpoint evaluation with .NET HashSet for recursive pipeline stages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_ironpython_pipeline_generator_test"
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

# Test 1: Unit tests pass
test_unit_tests() {
    log_info "Test 1: IronPython pipeline generator unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/python_target), test_ironpython_pipeline_generator" -t halt 2>&1 | grep -q "All IronPython Pipeline Generator Mode Tests Passed"; then
        log_pass "All unit tests pass"
    else
        log_fail "Unit tests failed"
    fi
}

# Test 2: Compile pipeline with generator mode
test_generator_pipeline() {
    log_info "Test 2: IronPython pipeline with generator mode"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/gen_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_same_runtime_pipeline([transform/1, derive/1], [
        pipeline_name(gen_pipeline),
        pipeline_mode(generator),
        runtime(ironpython)
    ], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_pipeline_generator_test/gen_pipeline.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_pipeline_generator_test/gen_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
            if grep -q "RecordSet" "$OUTPUT_DIR/gen_pipeline.py" && \
               grep -q "while changed" "$OUTPUT_DIR/gen_pipeline.py" && \
               grep -q "def gen_pipeline" "$OUTPUT_DIR/gen_pipeline.py"; then
                log_pass "Generator pipeline generates correct IronPython code"
            else
                log_fail "Missing generator mode patterns"
            fi
        else
            log_fail "gen_pipeline.py not generated"
        fi
    else
        log_fail "Pipeline compilation failed"
    fi
}

# Test 3: Verify RecordSet class
test_recordset_class() {
    log_info "Test 3: RecordSet class"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if grep -q "class RecordSet" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "HashSet\[String\]" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "def __contains__" "$OUTPUT_DIR/gen_pipeline.py"; then
            log_pass "RecordSet class has correct structure"
        else
            log_fail "RecordSet class missing expected patterns"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 4: Verify record_key function
test_record_key_function() {
    log_info "Test 4: record_key function"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if grep -q "def record_key" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "json.dumps" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "sort_keys=True" "$OUTPUT_DIR/gen_pipeline.py"; then
            log_pass "record_key function uses JSON serialization"
        else
            log_fail "record_key function missing expected patterns"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 5: Verify fixpoint iteration structure
test_fixpoint_structure() {
    log_info "Test 5: Fixpoint iteration structure"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if grep -q "all_records = \[\]" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "changed = True" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "while changed" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "all_records.append" "$OUTPUT_DIR/gen_pipeline.py"; then
            log_pass "Fixpoint iteration structure correct"
        else
            log_fail "Fixpoint structure missing patterns"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 6: Verify IronPython shebang
test_shebang_header() {
    log_info "Test 6: IronPython shebang header"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if head -1 "$OUTPUT_DIR/gen_pipeline.py" | grep -q "#!/usr/bin/env ipy"; then
            log_pass "IronPython shebang header present"
        else
            log_fail "IronPython shebang header missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 7: Verify CLR imports
test_clr_imports() {
    log_info "Test 7: CLR imports"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if grep -q "import clr" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "clr.AddReference" "$OUTPUT_DIR/gen_pipeline.py" && \
           grep -q "from System import" "$OUTPUT_DIR/gen_pipeline.py"; then
            log_pass "CLR imports present"
        else
            log_fail "CLR imports missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 8: Verify HashSet import for generator mode
test_hashset_import() {
    log_info "Test 8: HashSet import for generator mode"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if grep -q "from System.Collections.Generic import HashSet" "$OUTPUT_DIR/gen_pipeline.py"; then
            log_pass "HashSet import present"
        else
            log_fail "HashSet import missing"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Test 9: Sequential mode still works
test_sequential_mode() {
    log_info "Test 9: Sequential mode still works"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/seq_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/python_target).

test_compile :-
    compile_same_runtime_pipeline([filter/1, format/1], [
        pipeline_name(seq_pipeline),
        pipeline_mode(sequential),
        runtime(ironpython)
    ], Code),
    atom_string(CodeAtom, Code),
    open('output_ironpython_pipeline_generator_test/seq_pipeline.py', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_ironpython_pipeline_generator_test/seq_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/seq_pipeline.py" ]; then
            if grep -q "def seq_pipeline" "$OUTPUT_DIR/seq_pipeline.py" && \
               grep -q "yield from" "$OUTPUT_DIR/seq_pipeline.py" && \
               ! grep -q "RecordSet" "$OUTPUT_DIR/seq_pipeline.py"; then
                log_pass "Sequential mode works correctly"
            else
                log_fail "Sequential mode incorrect"
            fi
        else
            log_fail "seq_pipeline.py not generated"
        fi
    else
        log_fail "Sequential pipeline compilation failed"
    fi
}

# Test 10: Verify no dataclass import for IronPython
test_no_dataclass() {
    log_info "Test 10: No dataclass import for IronPython"

    if [ -f "$OUTPUT_DIR/gen_pipeline.py" ]; then
        if ! grep -q "from dataclasses import" "$OUTPUT_DIR/gen_pipeline.py" && \
           ! grep -q "@dataclass" "$OUTPUT_DIR/gen_pipeline.py"; then
            log_pass "No dataclass import (IronPython compatible)"
        else
            log_fail "Dataclass found (not IronPython compatible)"
        fi
    else
        log_fail "No Python file to check"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  IronPython Pipeline Generator E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_unit_tests
    test_generator_pipeline
    test_recordset_class
    test_record_key_function
    test_fixpoint_structure
    test_shebang_header
    test_clr_imports
    test_hashset_import
    test_sequential_mode
    test_no_dataclass

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
