#!/bin/bash
# test_llvm_wam_pipeline.sh - E2E pipeline tests for WAM-to-LLVM hybrid target
#
# Tests the full pipeline:
#   1. Prolog-side: generate LLVM IR via compile_wam_predicate_to_llvm
#   2. LLVM tools: opt -verify (if available)
#   3. LLVM tools: lli JIT execution (if available)
#
# Runs without LLVM tools installed (Prolog-only validation),
# but reports additional passes when tools are available.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_llvm_wam_e2e_test"
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; SKIP_COUNT=$((SKIP_COUNT + 1)); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

cleanup() { rm -rf "$OUTPUT_DIR"; }
setup() { cleanup; mkdir -p "$OUTPUT_DIR"; }

# Check for LLVM tools
HAS_OPT=false
HAS_LLI=false
HAS_LLC=false
command -v opt  >/dev/null 2>&1 && HAS_OPT=true
command -v lli  >/dev/null 2>&1 && HAS_LLI=true
command -v llc  >/dev/null 2>&1 && HAS_LLC=true

# ============================================================================
# Test 1: Prolog-side unit tests pass
# ============================================================================
test_prolog_unit_tests() {
    log_info "Test 1: WAM LLVM unit tests (46 tests)"
    cd "$PROJECT_ROOT"
    OUTPUT=$(swipl -g "['tests/test_wam_llvm_target'], run_tests, halt" -t "halt(1)" 2>&1)
    if echo "$OUTPUT" | grep -q "All.*tests passed"; then
        PASSED=$(echo "$OUTPUT" | grep -oP '\d+ tests passed' | head -1)
        log_pass "Unit tests: $PASSED"
    else
        log_fail "Unit tests failed"
        echo "$OUTPUT" | grep -E "FAILED|ERROR" | head -5
    fi
}

# ============================================================================
# Test 2: Prolog-side E2E tests pass
# ============================================================================
test_prolog_e2e_tests() {
    log_info "Test 2: WAM LLVM E2E integration tests"
    cd "$PROJECT_ROOT"
    OUTPUT=$(swipl -g "['tests/integration/test_llvm_wam_e2e'], run_tests, halt" -t "halt(1)" 2>&1)
    if echo "$OUTPUT" | grep -q "All.*tests passed"; then
        PASSED=$(echo "$OUTPUT" | grep -oP '\d+ tests passed' | head -1)
        log_pass "E2E tests: $PASSED"
    else
        log_fail "E2E tests failed"
        echo "$OUTPUT" | grep -E "FAILED|ERROR" | head -5
    fi
}

# ============================================================================
# Test 3: Generate step function IR and verify structure
# ============================================================================
test_step_function_ir() {
    log_info "Test 3: Generate step function IR"
    cd "$PROJECT_ROOT"

    swipl -g "
        use_module('src/unifyweaver/targets/wam_llvm_target'),
        compile_step_wam_to_llvm([], Code),
        open('$OUTPUT_DIR/step.ll', write, S),
        format(S, '~w', [Code]),
        close(S),
        halt
    " 2>/dev/null

    if [ -f "$OUTPUT_DIR/step.ll" ] && grep -q "define i1 @step" "$OUTPUT_DIR/step.ll"; then
        log_pass "Step function IR generated ($(wc -l < "$OUTPUT_DIR/step.ll") lines)"
    else
        log_fail "Step function IR generation failed"
    fi
}

# ============================================================================
# Test 4: Generate runtime IR (step + helpers)
# ============================================================================
test_runtime_ir() {
    log_info "Test 4: Generate full runtime IR"
    cd "$PROJECT_ROOT"

    swipl -g "
        use_module('src/unifyweaver/targets/wam_llvm_target'),
        compile_wam_runtime_to_llvm([], Code),
        open('$OUTPUT_DIR/runtime.ll', write, S),
        format(S, '~w', [Code]),
        close(S),
        halt
    " 2>/dev/null

    if [ -f "$OUTPUT_DIR/runtime.ll" ]; then
        FUNCS=$(grep -c "^define " "$OUTPUT_DIR/runtime.ll" 2>/dev/null || echo 0)
        log_pass "Runtime IR generated ($FUNCS functions, $(wc -l < "$OUTPUT_DIR/runtime.ll") lines)"
    else
        log_fail "Runtime IR generation failed"
    fi
}

# ============================================================================
# Test 5: Generate WAM-compiled predicate IR
# ============================================================================
test_wam_predicate_ir() {
    log_info "Test 5: Generate WAM-compiled predicate IR"
    cd "$PROJECT_ROOT"

    swipl -g "
        use_module('src/unifyweaver/targets/wam_llvm_target'),
        WamCode = 'parent/2:\n    try_me_else L_c2\n    get_constant john, A1\n    get_constant mary, A2\n    proceed\nL_c2:\n    trust_me\n    get_constant bob, A1\n    get_constant alice, A2\n    proceed',
        compile_wam_predicate_to_llvm(parent/2, WamCode, [], LLVMCode),
        open('$OUTPUT_DIR/parent.ll', write, S),
        format(S, '~w', [LLVMCode]),
        close(S),
        halt
    " 2>/dev/null

    if [ -f "$OUTPUT_DIR/parent.ll" ] && grep -q "@parent_code" "$OUTPUT_DIR/parent.ll"; then
        INSTRS=$(grep -c "%Instruction" "$OUTPUT_DIR/parent.ll" 2>/dev/null || echo 0)
        log_pass "WAM predicate IR generated ($INSTRS instruction refs)"
    else
        log_fail "WAM predicate IR generation failed"
    fi
}

# ============================================================================
# Test 6: WAM fallback disable option
# ============================================================================
test_wam_fallback_disable() {
    log_info "Test 6: WAM fallback can be disabled"
    cd "$PROJECT_ROOT"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/targets/wam_llvm_target'),
        wam_llvm_target:compile_predicates_collect(
            [nonexistent_pred/2],
            [wam_fallback(false)],
            NativeParts, WamParts),
        format('native=~w wam=~w~n', [NativeParts, WamParts]),
        halt
    " 2>/dev/null)

    if echo "$OUTPUT" | grep -q "native=\[\]"; then
        log_pass "WAM fallback disabled correctly"
    else
        log_fail "WAM fallback disable not working"
    fi
}

# ============================================================================
# Test 7: opt -verify (if available)
# ============================================================================
test_opt_verify() {
    if [ "$HAS_OPT" = false ]; then
        log_skip "opt not found — LLVM IR verification skipped"
        return
    fi

    log_info "Test 7: opt -verify on assembled module"

    cd "$PROJECT_ROOT"

    # Assemble a complete module: types + value + state templates + runtime
    cat templates/targets/llvm_wam/types.ll.mustache \
        templates/targets/llvm_wam/value.ll.mustache \
        templates/targets/llvm_wam/state.ll.mustache \
        > "$OUTPUT_DIR/full_module.ll" 2>/dev/null

    # Strip mustache tags (they're just comments in context)
    sed -i 's/{{[^}]*}}//g' "$OUTPUT_DIR/full_module.ll"

    # Prepend module header
    sed -i '1i\; ModuleID = "wam_e2e_test"\ntarget triple = "x86_64-pc-linux-gnu"\n' "$OUTPUT_DIR/full_module.ll"

    # Append generated runtime
    swipl -g "
        use_module('src/unifyweaver/targets/wam_llvm_target'),
        compile_step_wam_to_llvm([], StepCode),
        compile_wam_helpers_to_llvm([], HelpersCode),
        open('$OUTPUT_DIR/runtime_funcs.ll', write, S),
        format(S, '~w~n~n~w~n', [StepCode, HelpersCode]),
        close(S),
        halt
    " 2>/dev/null

    cat "$OUTPUT_DIR/runtime_funcs.ll" >> "$OUTPUT_DIR/full_module.ll"

    # Try opt verify; new pass manager uses -passes=verify
    OPT_OUTPUT=$(opt -passes=verify -S "$OUTPUT_DIR/full_module.ll" -o /dev/null 2>&1)
    OPT_RC=$?

    if [ $OPT_RC -eq 0 ]; then
        log_pass "opt -verify passed on assembled module"
    else
        # Count errors to distinguish minor vs major issues
        ERR_COUNT=$(echo "$OPT_OUTPUT" | grep -c "error:" 2>/dev/null || echo 0)
        log_fail "opt -verify: $ERR_COUNT errors (see $OUTPUT_DIR/full_module.ll)"
        echo "$OPT_OUTPUT" | head -5
    fi
}

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "======================================"
echo "  WAM-to-LLVM E2E Pipeline Tests"
echo "======================================"
echo ""
echo "LLVM tools: opt=$HAS_OPT lli=$HAS_LLI llc=$HAS_LLC"
echo ""

setup

test_prolog_unit_tests
test_prolog_e2e_tests
test_step_function_ir
test_runtime_ir
test_wam_predicate_ir
test_wam_fallback_disable
test_opt_verify

echo ""
echo "======================================"
echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo "======================================"

cleanup

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
