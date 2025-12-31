#!/bin/bash
# test_bash_pipeline_generator.sh - End-to-end tests for Bash pipeline generator mode
# Tests fixpoint evaluation for recursive pipeline stages

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_bash_pipeline_generator_test"
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
    if swipl -g "use_module(src/unifyweaver/targets/bash_target), test_bash_pipeline_generator" -t halt 2>&1 | grep -q "All Bash Pipeline Generator Mode Tests Passed"; then
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
:- use_module(src/unifyweaver/targets/bash_target).

test_compile :-
    compile_bash_pipeline([transform/1, derive/1], [
        pipeline_name(gen_pipeline),
        pipeline_mode(generator),
        record_format(jsonl)
    ], Code),
    atom_string(CodeAtom, Code),
    open('output_bash_pipeline_generator_test/gen_pipeline.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_pipeline_generator_test/gen_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
            if grep -q "record_key" "$OUTPUT_DIR/gen_pipeline.sh" && \
               grep -q "while" "$OUTPUT_DIR/gen_pipeline.sh" && \
               grep -q "run_gen_pipeline" "$OUTPUT_DIR/gen_pipeline.sh"; then
                log_pass "Generator pipeline generates correct Bash code"
            else
                log_fail "Missing generator mode patterns"
            fi
        else
            log_fail "gen_pipeline.sh not generated"
        fi
    else
        log_fail "Pipeline compilation failed"
    fi
}

# Test 3: Verify record_key function
test_record_key_function() {
    log_info "Test 3: record_key function"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if grep -q "record_key()" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "jq" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "sort" "$OUTPUT_DIR/gen_pipeline.sh"; then
            log_pass "record_key function has correct structure"
        else
            log_fail "record_key function missing expected patterns"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Test 4: Verify JSONL helpers
test_jsonl_helpers() {
    log_info "Test 4: JSONL helper functions"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if grep -q "parse_jsonl()" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "format_jsonl()" "$OUTPUT_DIR/gen_pipeline.sh"; then
            log_pass "JSONL helpers present"
        else
            log_fail "JSONL helpers missing"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Test 5: Verify fixpoint iteration structure
test_fixpoint_structure() {
    log_info "Test 5: Fixpoint iteration structure"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if grep -q "declare -A seen" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "changed=1" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "while \[\[" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q 'seen\[' "$OUTPUT_DIR/gen_pipeline.sh"; then
            log_pass "Fixpoint iteration structure correct"
        else
            log_fail "Fixpoint structure missing patterns"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Test 6: Verify shebang header
test_shebang_header() {
    log_info "Test 6: Shebang header"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if head -1 "$OUTPUT_DIR/gen_pipeline.sh" | grep -q "#!/bin/bash"; then
            log_pass "Shebang header present"
        else
            log_fail "Shebang header missing"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Test 7: Verify pipeline connector function
test_pipeline_connector() {
    log_info "Test 7: Pipeline connector function"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if grep -q "run_gen_pipeline()" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "Fixpoint pipeline" "$OUTPUT_DIR/gen_pipeline.sh"; then
            log_pass "Pipeline connector has correct structure"
        else
            log_fail "Connector structure incorrect"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Test 8: Verify stage functions
test_stage_functions() {
    log_info "Test 8: Stage functions"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if grep -q "stage_transform()" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "stage_derive()" "$OUTPUT_DIR/gen_pipeline.sh"; then
            log_pass "Stage functions generated"
        else
            log_fail "Stage functions missing"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Test 9: Sequential mode still works
test_sequential_mode() {
    log_info "Test 9: Sequential mode still works"

    cd "$PROJECT_ROOT"

    cat > "$OUTPUT_DIR/seq_test.pl" << 'EOF'
:- use_module(src/unifyweaver/targets/bash_target).

test_compile :-
    compile_bash_pipeline([filter/1, format/1], [
        pipeline_name(seq_pipeline),
        pipeline_mode(sequential),
        record_format(tsv)
    ], Code),
    atom_string(CodeAtom, Code),
    open('output_bash_pipeline_generator_test/seq_pipeline.sh', write, S),
    write(S, CodeAtom),
    close(S).
EOF

    if swipl -g "consult('output_bash_pipeline_generator_test/seq_test.pl'), test_compile" -t halt 2>/dev/null; then
        if [ -f "$OUTPUT_DIR/seq_pipeline.sh" ]; then
            if grep -q "run_seq_pipeline" "$OUTPUT_DIR/seq_pipeline.sh" && \
               grep -q "sequential mode" "$OUTPUT_DIR/seq_pipeline.sh" && \
               ! grep -q "while \[\[" "$OUTPUT_DIR/seq_pipeline.sh"; then
                log_pass "Sequential mode works correctly"
            else
                log_fail "Sequential mode incorrect"
            fi
        else
            log_fail "seq_pipeline.sh not generated"
        fi
    else
        log_fail "Sequential pipeline compilation failed"
    fi
}

# Test 10: Main execution block
test_main_execution() {
    log_info "Test 10: Main execution block"

    if [ -f "$OUTPUT_DIR/gen_pipeline.sh" ]; then
        if grep -q "main()" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "input_records" "$OUTPUT_DIR/gen_pipeline.sh" && \
           grep -q "output_records" "$OUTPUT_DIR/gen_pipeline.sh"; then
            log_pass "Main execution block correct"
        else
            log_fail "Main execution block incorrect"
        fi
    else
        log_fail "No Bash file to check"
    fi
}

# Main test runner
main() {
    echo "=========================================="
    echo "  Bash Pipeline Generator E2E Tests"
    echo "=========================================="
    echo ""

    setup

    test_unit_tests
    test_generator_pipeline
    test_record_key_function
    test_jsonl_helpers
    test_fixpoint_structure
    test_shebang_header
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
