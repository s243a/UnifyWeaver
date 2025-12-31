#!/bin/bash
# Phase 7-8 Integration Test Runner
# Tests all generated code across Python, Go, and Rust

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Phase 7-8 Integration Tests"
echo "========================================"
echo ""

# Track results
PYTHON_OK=0
GO_OK=0
RUST_OK=0
PROLOG_OK=0

# Run Prolog code generation tests
echo "--- Prolog Code Generation Tests ---"
if command -v swipl &> /dev/null; then
    cd "$SCRIPT_DIR/../../tests/prolog"
    if swipl -g run_tests -t halt test_small_world_codegen.pl 2>&1; then
        PROLOG_OK=1
        echo "[PASS] Prolog code generation tests"
    else
        echo "[FAIL] Prolog code generation tests"
    fi
    cd "$SCRIPT_DIR"
else
    echo "[SKIP] swipl not found"
fi
echo ""

# Run Python tests
echo "--- Python Runtime Tests ---"
if command -v python3 &> /dev/null; then
    cd "$SCRIPT_DIR/../.."
    if python3 -m unittest tests.integration.test_small_world_integration -v 2>&1; then
        PYTHON_OK=1
        echo "[PASS] Python integration tests"
    else
        echo "[FAIL] Python integration tests"
    fi
    cd "$SCRIPT_DIR"
else
    echo "[SKIP] python3 not found"
fi
echo ""

# Run Go tests
echo "--- Go Generated Code Tests ---"
if command -v go &> /dev/null; then
    cd "$SCRIPT_DIR/generated/smallworld"
    if go test -v 2>&1; then
        GO_OK=1
        echo "[PASS] Go generated code tests"
    else
        echo "[FAIL] Go generated code tests"
    fi
    cd "$SCRIPT_DIR"
else
    echo "[SKIP] go not found"
fi
echo ""

# Run Rust tests
echo "--- Rust Generated Code Tests ---"
if command -v cargo &> /dev/null; then
    cd "$SCRIPT_DIR/generated/rust_smallworld"
    if cargo test 2>&1; then
        RUST_OK=1
        echo "[PASS] Rust generated code tests"
    else
        echo "[FAIL] Rust generated code tests"
    fi
    cd "$SCRIPT_DIR"
else
    echo "[SKIP] cargo not found"
fi
echo ""

# Summary
echo "========================================"
echo "Summary"
echo "========================================"
echo "Prolog: $([ $PROLOG_OK -eq 1 ] && echo 'PASS' || echo 'FAIL/SKIP')"
echo "Python: $([ $PYTHON_OK -eq 1 ] && echo 'PASS' || echo 'FAIL/SKIP')"
echo "Go:     $([ $GO_OK -eq 1 ] && echo 'PASS' || echo 'FAIL/SKIP')"
echo "Rust:   $([ $RUST_OK -eq 1 ] && echo 'PASS' || echo 'FAIL/SKIP')"
echo ""

# Exit code
TOTAL=$((PROLOG_OK + PYTHON_OK + GO_OK + RUST_OK))
if [ $TOTAL -eq 4 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed or skipped."
    exit 1
fi
