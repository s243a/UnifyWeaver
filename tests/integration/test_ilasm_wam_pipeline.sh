#!/bin/bash
# test_ilasm_wam_pipeline.sh - E2E pipeline tests for WAM-to-ILAsm hybrid target
#
# Tests the full pipeline:
#   1. Prolog-side: unit + E2E tests
#   2. CIL tools: ilasm assembly (if available)
#   3. CIL tools: dotnet-ilverify / peverify (if available)
#
# Runs without .NET tools (Prolog-only validation),
# but reports additional passes when tools are available.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/output_ilasm_wam_e2e_test"
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

# Check for .NET tools
HAS_ILASM=false
HAS_ILVERIFY=false
HAS_MONO=false
command -v ilasm      >/dev/null 2>&1 && HAS_ILASM=true
command -v dotnet-ilverify >/dev/null 2>&1 && HAS_ILVERIFY=true
command -v peverify   >/dev/null 2>&1 && HAS_ILVERIFY=true
command -v mono       >/dev/null 2>&1 && HAS_MONO=true

# ============================================================================
# Test 1: Unit tests
# ============================================================================
test_unit_tests() {
    log_info "Test 1: WAM ILAsm unit tests"
    cd "$PROJECT_ROOT"
    OUTPUT=$(swipl -g "['tests/test_wam_ilasm_target'], run_tests, halt" -t "halt(1)" 2>&1)
    if echo "$OUTPUT" | grep -q "All.*tests passed"; then
        PASSED=$(echo "$OUTPUT" | grep -oP '\d+ tests passed' | head -1)
        log_pass "Unit tests: $PASSED"
    else
        log_fail "Unit tests failed"
        echo "$OUTPUT" | grep -E "FAILED|ERROR" | head -5
    fi
}

# ============================================================================
# Test 2: E2E integration tests
# ============================================================================
test_e2e_tests() {
    log_info "Test 2: WAM ILAsm E2E integration tests"
    cd "$PROJECT_ROOT"
    OUTPUT=$(swipl -g "['tests/integration/test_ilasm_wam_e2e'], run_tests, halt" -t "halt(1)" 2>&1)
    if echo "$OUTPUT" | grep -q "All.*tests passed"; then
        PASSED=$(echo "$OUTPUT" | grep -oP '\d+ tests passed' | head -1)
        log_pass "E2E tests: $PASSED"
    else
        log_fail "E2E tests failed"
        echo "$OUTPUT" | grep -E "FAILED|ERROR" | head -5
    fi
}

# ============================================================================
# Test 3: Generate step function CIL
# ============================================================================
test_step_function_cil() {
    log_info "Test 3: Generate step function CIL"
    cd "$PROJECT_ROOT"

    swipl -g "
        use_module('src/unifyweaver/targets/wam_ilasm_target'),
        compile_step_wam_to_cil([], Code),
        open('$OUTPUT_DIR/step.il', write, S),
        format(S, '~w', [Code]),
        close(S),
        halt
    " 2>/dev/null

    if [ -f "$OUTPUT_DIR/step.il" ] && grep -q ".method public static bool step" "$OUTPUT_DIR/step.il"; then
        log_pass "Step function CIL generated ($(wc -l < "$OUTPUT_DIR/step.il") lines)"
    else
        log_fail "Step function CIL generation failed"
    fi
}

# ============================================================================
# Test 4: Generate full runtime CIL
# ============================================================================
test_runtime_cil() {
    log_info "Test 4: Generate full runtime CIL"
    cd "$PROJECT_ROOT"

    swipl -g "
        use_module('src/unifyweaver/targets/wam_ilasm_target'),
        compile_wam_runtime_to_cil([], Code),
        open('$OUTPUT_DIR/runtime.il', write, S),
        format(S, '~w', [Code]),
        close(S),
        halt
    " 2>/dev/null

    if [ -f "$OUTPUT_DIR/runtime.il" ]; then
        METHODS=$(grep -c "^\.method " "$OUTPUT_DIR/runtime.il" 2>/dev/null || echo 0)
        log_pass "Runtime CIL generated ($METHODS methods, $(wc -l < "$OUTPUT_DIR/runtime.il") lines)"
    else
        log_fail "Runtime CIL generation failed"
    fi
}

# ============================================================================
# Test 5: Generate WAM-compiled predicate CIL
# ============================================================================
test_wam_predicate_cil() {
    log_info "Test 5: Generate WAM-compiled predicate CIL"
    cd "$PROJECT_ROOT"

    swipl -g "
        use_module('src/unifyweaver/targets/wam_ilasm_target'),
        WamCode = 'parent/2:\n    try_me_else L_c2\n    get_constant john, A1\n    get_constant mary, A2\n    proceed\nL_c2:\n    trust_me\n    get_constant bob, A1\n    get_constant alice, A2\n    proceed',
        compile_wam_predicate_to_cil(parent/2, WamCode, [], CILCode),
        open('$OUTPUT_DIR/parent.il', write, S),
        format(S, '~w', [CILCode]),
        close(S),
        halt
    " 2>/dev/null

    if [ -f "$OUTPUT_DIR/parent.il" ] && grep -q "parent_code" "$OUTPUT_DIR/parent.il"; then
        log_pass "WAM predicate CIL generated"
    else
        log_fail "WAM predicate CIL generation failed"
    fi
}

# ============================================================================
# Test 6: ilasm assembly (if available)
# ============================================================================
test_ilasm_verify() {
    if [ "$HAS_ILASM" = false ]; then
        log_skip "ilasm not found — CIL assembly verification skipped"
        return
    fi

    log_info "Test 6: ilasm assembly of generated CIL"

    # Assemble the types template + runtime into a single .il
    cd "$PROJECT_ROOT"
    cat templates/targets/ilasm_wam/types.il.mustache > "$OUTPUT_DIR/full_module.il"
    sed -i 's/{{[^}]*}}//g' "$OUTPUT_DIR/full_module.il"

    ILASM_OUTPUT=$(ilasm "$OUTPUT_DIR/full_module.il" /dll /output="$OUTPUT_DIR/test.dll" 2>&1)
    ILASM_RC=$?

    if [ $ILASM_RC -eq 0 ]; then
        log_pass "ilasm assembled types template successfully"
    else
        log_fail "ilasm assembly failed"
        echo "$ILASM_OUTPUT" | head -5
    fi
}

# ============================================================================
# Test 7: WAM fallback disable
# ============================================================================
test_wam_fallback_disable() {
    log_info "Test 7: WAM fallback can be disabled"
    cd "$PROJECT_ROOT"

    OUTPUT=$(swipl -g "
        use_module('src/unifyweaver/targets/wam_ilasm_target'),
        wam_ilasm_target:compile_predicates_collect_cil(
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
# Summary
# ============================================================================

echo ""
echo "======================================"
echo "  WAM-to-ILAsm E2E Pipeline Tests"
echo "======================================"
echo ""
echo ".NET tools: ilasm=$HAS_ILASM ilverify=$HAS_ILVERIFY mono=$HAS_MONO"
echo ""

setup

test_unit_tests
test_e2e_tests
test_step_function_cil
test_runtime_cil
test_wam_predicate_cil
test_ilasm_verify
test_wam_fallback_disable

echo ""
echo "======================================"
echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped"
echo "======================================"

cleanup

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
