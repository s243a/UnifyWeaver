#!/bin/bash
# test_pipeline_validation.sh - Integration tests for pipeline validation
# Tests that validation works correctly across all targets

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PASS_COUNT=0
FAIL_COUNT=0

pass() {
    echo "  PASS: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo "  FAIL: $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

run_test() {
    local name="$1"
    local goal="$2"
    local pattern="$3"

    echo "Running: $name..."
    if timeout 30 swipl -g "$goal" -t halt 2>&1 | grep -q "$pattern"; then
        pass "$name"
    else
        fail "$name"
    fi
}

echo "=== Pipeline Validation Integration Tests ==="
echo ""

# Test 1: Core validation module tests
run_test "Core validation tests" \
    "use_module('src/unifyweaver/core/pipeline_validation'), test_pipeline_validation" \
    "All Pipeline Validation Tests Passed"

# Test 2: Empty pipeline detection (Python)
run_test "Empty pipeline detection (Python)" \
    "use_module('src/unifyweaver/targets/python_target'), catch(compile_enhanced_pipeline([], [], _), pipeline_validation_failed(_), true)" \
    "empty_pipeline"

# Test 3: Empty pipeline detection (Go)
run_test "Empty pipeline detection (Go)" \
    "use_module('src/unifyweaver/targets/go_target'), catch(compile_go_enhanced_pipeline([], [], _), pipeline_validation_failed(_), true)" \
    "empty_pipeline"

# Test 4: Empty fan_out detection (Bash)
run_test "Empty fan_out detection (Bash)" \
    "use_module('src/unifyweaver/targets/bash_target'), catch(compile_bash_enhanced_pipeline([parse/1, fan_out([])], [], _), pipeline_validation_failed(_), true)" \
    "empty_fan_out"

# Test 5: Empty routes detection (AWK)
run_test "Empty routes detection (AWK)" \
    "use_module('src/unifyweaver/targets/awk_target'), catch(compile_awk_enhanced_pipeline([parse/1, route_by(check, [])], [], _), pipeline_validation_failed(_), true)" \
    "empty_routes"

# Test 6: Invalid route format detection (PowerShell)
run_test "Invalid route format detection (PowerShell)" \
    "use_module('src/unifyweaver/targets/powershell_target'), catch(compile_powershell_enhanced_pipeline([parse/1, route_by(check, [bad_route])], [], _), pipeline_validation_failed(_), true)" \
    "invalid_route_format"

# Test 7: Warning for fan_out without merge (C#)
run_test "Warning for fan_out without merge (C#)" \
    "use_module('src/unifyweaver/targets/csharp_target'), compile_csharp_enhanced_pipeline([parse/1, fan_out([a/1, b/1]), output/1], [], _)" \
    "fan_out_without_merge"

# Test 8: Warning for merge without fan_out (Rust)
run_test "Warning for merge without fan_out (Rust)" \
    "use_module('src/unifyweaver/targets/rust_target'), compile_rust_enhanced_pipeline([parse/1, merge, output/1], [], _)" \
    "merge_without_fan_out"

# Test 9: Validation can be disabled
run_test "Validation can be disabled" \
    "use_module('src/unifyweaver/targets/python_target'), compile_enhanced_pipeline([], [validate(false)], _), format('Validation disabled OK~n', [])" \
    "Validation disabled OK"

# Test 10: Strict mode converts warnings to errors
run_test "Strict mode converts warnings to errors" \
    "use_module('src/unifyweaver/targets/python_target'), catch(compile_enhanced_pipeline([parse/1, merge, output/1], [strict(true)], _), pipeline_validation_failed(_), true)" \
    "validation errors"

# Test 11: Valid pipeline passes (IronPython)
run_test "Valid pipeline passes (IronPython)" \
    "use_module('src/unifyweaver/targets/python_target'), compile_ironpython_enhanced_pipeline([parse/1, filter_by(is_active), fan_out([validate/1, enrich/1]), merge, output/1], [], _), format('Valid pipeline generated OK~n', [])" \
    "Valid pipeline generated OK"

# Test 12: Complex valid pipeline with routing
run_test "Complex valid pipeline with routing (Go)" \
    "use_module('src/unifyweaver/targets/go_target'), compile_go_enhanced_pipeline([parse/1, filter_by(is_valid), fan_out([transform_a/1, transform_b/1]), merge, route_by(priority, [(high, fast/1), (low, slow/1)]), output/1], [], _), format('Complex pipeline generated OK~n', [])" \
    "Complex pipeline generated OK"

echo ""
echo "=== Results ==="
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "SOME TESTS FAILED"
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED"
    exit 0
fi
