#!/bin/bash
# test_csharp_pipeline_generator.sh - End-to-end tests for C# pipeline generator mode
# Tests fixpoint evaluation for recursive pipeline stages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_csharp_pipeline_generator_test"
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
    log_info "Test 1: Pipeline generator unit tests"

    cd "$PROJECT_ROOT"
    if swipl -g "use_module(src/unifyweaver/targets/csharp_target), test_csharp_pipeline_generator" -t halt 2>&1 | grep -q "All C# Pipeline Generator Mode Tests Passed"; then
        log_pass "All unit tests pass"
    else
        log_fail "Unit tests failed"
    fi
}

# Test 2: Compile pipeline with generator mode
test_generator_pipeline() {
    log_info "Test 2: Pipeline with generator mode"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/gen_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/csharp_target).

test_compile :-
    compile_csharp_pipeline([transform/1, derive/1], [
        pipeline_name('GenPipeline'),
        pipeline_mode(generator),
        output_format(jsonl)
    ], Code),
    atom_string(CodeAtom, Code),
    open('output_csharp_pipeline_generator_test/gen_pipeline.cs', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_csharp_pipeline_generator_test/gen_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
            if grep -q "GetRecordKey" "$OUTPUT_DIR/gen_pipeline.cs" && \
               grep -q "while (changed)" "$OUTPUT_DIR/gen_pipeline.cs" && \
               grep -q "GenPipeline" "$OUTPUT_DIR/gen_pipeline.cs"; then
                log_pass "Generator pipeline generates correct C# code"
            else
                log_fail "Missing generator mode patterns"
            fi
        else
            log_fail "gen_pipeline.cs not generated"
        fi
    else
        log_fail "Pipeline compilation failed"
    fi
}

# Test 3: Verify RecordKeyHelper class
test_record_key_helper() {
    log_info "Test 3: RecordKeyHelper class"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "class RecordKeyHelper" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "GetRecordKey" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "string.Join" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "RecordKeyHelper has correct structure"
        else
            log_fail "RecordKeyHelper missing expected patterns"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Test 4: Verify JsonlHelper class
test_jsonl_helper() {
    log_info "Test 4: JsonlHelper class"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "class JsonlHelper" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "ReadJsonlStream" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "WriteJsonlStream" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "JsonlHelper present"
        else
            log_fail "JsonlHelper missing"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Test 5: Verify fixpoint iteration structure
test_fixpoint_structure() {
    log_info "Test 5: Fixpoint iteration structure"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "HashSet<string>" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "changed = true" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "while (changed)" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "total.Contains" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "Fixpoint iteration structure correct"
        else
            log_fail "Fixpoint structure missing patterns"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Test 6: Verify using statements
test_using_statements() {
    log_info "Test 6: Using statements"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "using System;" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "using System.Collections.Generic;" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "using System.Linq;" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "using System.Text.Json;" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "Using statements present"
        else
            log_fail "Using statements missing"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Test 7: Verify pipeline connector function
test_pipeline_connector() {
    log_info "Test 7: Pipeline connector function"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "GenPipeline" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "Fixpoint pipeline" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "IEnumerable<Dictionary<string, object?>>" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "Pipeline connector has correct structure"
        else
            log_fail "Connector structure incorrect"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Test 8: Verify stage functions
test_stage_functions() {
    log_info "Test 8: Stage functions"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "public static IEnumerable" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "Transform" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "Derive" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "Stage functions generated"
        else
            log_fail "Stage functions missing"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Test 9: Sequential mode still works
test_sequential_mode() {
    log_info "Test 9: Sequential mode still works"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/seq_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/csharp_target).

test_compile :-
    compile_csharp_pipeline([filter/1, format/1], [
        pipeline_name('SeqPipeline'),
        pipeline_mode(sequential),
        output_format(jsonl)
    ], Code),
    atom_string(CodeAtom, Code),
    open('output_csharp_pipeline_generator_test/seq_pipeline.cs', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_csharp_pipeline_generator_test/seq_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/seq_pipeline.cs" ]; then
            if grep -q "SeqPipeline" "$OUTPUT_DIR/seq_pipeline.cs" && \
               grep -q "sequential mode" "$OUTPUT_DIR/seq_pipeline.cs" && \
               ! grep -q "while (changed)" "$OUTPUT_DIR/seq_pipeline.cs"; then
                log_pass "Sequential mode works correctly"
            else
                log_fail "Sequential mode incorrect"
            fi
        else
            log_fail "seq_pipeline.cs not generated"
        fi
    else
        log_fail "Sequential pipeline compilation failed"
    fi
}

# Test 10: Main execution block
test_main_execution() {
    log_info "Test 10: Main execution block"

    if [ -f "$OUTPUT_DIR/gen_pipeline.cs" ]; then
        if grep -q "JsonlHelper.ReadJsonlStream" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "Console.In" "$OUTPUT_DIR/gen_pipeline.cs" && \
           grep -q "Main(string\[\] args)" "$OUTPUT_DIR/gen_pipeline.cs"; then
            log_pass "Main execution block correct"
        else
            log_fail "Main execution block incorrect"
        fi
    else
        log_fail "No C# file to check"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  C# Pipeline Generator E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_unit_tests
    test_generator_pipeline
    test_record_key_helper
    test_jsonl_helper
    test_fixpoint_structure
    test_using_statements
    test_pipeline_connector
    test_stage_functions
    test_sequential_mode
    test_main_execution

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
