#!/usr/bin/env bash
#
# Integration tests for C# target compilation
# Uses build-first-then-execute approach to work around dotnet run hang
#
# Usage: ./test_csharp_targets.sh
# Returns: 0 on success, 1 on failure

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Test output directory
TEST_OUTPUT_DIR="$PROJECT_ROOT/tmp/unifyweaver_csharp_tests_$$"
mkdir -p "$TEST_OUTPUT_DIR"

# Cleanup on exit
cleanup() {
    if [ -n "${KEEP_TEST_DATA:-}" ]; then
        echo -e "${YELLOW}Keeping test data in: $TEST_OUTPUT_DIR${NC}"
    else
        rm -rf "$TEST_OUTPUT_DIR"
    fi
}
trap cleanup EXIT

# Print test header
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Print test result
print_result() {
    local test_name="$1"
    local status="$2"
    local message="${3:-}"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        if [ -n "$message" ]; then
            echo "  Error: $message"
        fi
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Build and run a C# program from generated code
build_and_run_csharp() {
    local code="$1"
    local test_name="$2"
    local expected_output="$3"

    local test_dir="$TEST_OUTPUT_DIR/$test_name"
    mkdir -p "$test_dir"

    # Write the generated code to Program.cs
    echo "$code" > "$test_dir/Program.cs"

    # Create a simple .csproj file
    cat > "$test_dir/test.csproj" <<'EOF'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net9.0</TargetFramework>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
EOF

    # Build the project
    if ! dotnet build "$test_dir/test.csproj" --nologo -v quiet -o "$test_dir/bin" > "$test_dir/build.log" 2>&1; then
        print_result "$test_name" "FAIL" "Build failed - see $test_dir/build.log"
        return 1
    fi

    # Find the executable
    local exe_path="$test_dir/bin/test"
    if [ ! -f "$exe_path" ]; then
        exe_path="$test_dir/bin/test.dll"
        if [ ! -f "$exe_path" ]; then
            print_result "$test_name" "FAIL" "Executable not found after build"
            return 1
        fi
        # Run with dotnet if it's a DLL
        local actual_output
        if ! actual_output=$(dotnet "$exe_path" 2>&1); then
            print_result "$test_name" "FAIL" "Execution failed"
            return 1
        fi
    else
        # Run native executable directly
        local actual_output
        if ! actual_output=$("$exe_path" 2>&1); then
            print_result "$test_name" "FAIL" "Execution failed"
            return 1
        fi
    fi

    # Check output
    if echo "$actual_output" | grep -qF "$expected_output"; then
        print_result "$test_name" "PASS"
        return 0
    else
        print_result "$test_name" "FAIL" "Expected '$expected_output', got '$actual_output'"
        return 1
    fi
}

# Compile a Prolog predicate to C# and run tests
test_predicate_compilation() {
    local pred="$1"
    local arity="$2"
    local target="${3:-csharp_native}"
    local test_name="$4"
    local expected_output="$5"

    local prolog_query="
        use_module('$PROJECT_ROOT/src/unifyweaver/targets/csharp_native_target'),
        compile_predicate_to_csharp($pred/$arity, [target($target)], Code),
        write(Code)
    "

    local generated_code
    if ! generated_code=$(swipl -q -g "$prolog_query" -t halt 2>&1); then
        print_result "$test_name" "FAIL" "Compilation failed: $generated_code"
        return 1
    fi

    build_and_run_csharp "$generated_code" "$test_name" "$expected_output"
}

# ==========================================
# Test Suite
# ==========================================

print_header "C# Target Integration Tests"

# Test 1: Simple facts (Stream Target)
print_header "Test 1: Stream Target - Simple Binary Facts"

# Create a temporary Prolog file with test facts
cat > "$TEST_OUTPUT_DIR/test_facts.pl" <<'EOF'
:- dynamic link/2.
link(a, b).
link(b, c).
link(c, d).
EOF

# Compile using Prolog
PROLOG_TEST_1="
    ['$TEST_OUTPUT_DIR/test_facts.pl'],
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/csharp_native_target'),
    compile_predicate_to_csharp(link/2, [], Code),
    write(Code)
"

CODE_1=$(swipl -q -g "$PROLOG_TEST_1" -t halt 2>&1) || {
    print_result "Simple facts compilation" "FAIL" "Compilation error"
    CODE_1=""
}

if [ -n "$CODE_1" ]; then
    build_and_run_csharp "$CODE_1" "simple_facts" "a:b"
fi

# Test 2: Join query (Stream Target)
print_header "Test 2: Stream Target - Join Query"

cat > "$TEST_OUTPUT_DIR/test_join.pl" <<'EOF'
:- dynamic parent/2.
parent(alice, bob).
parent(bob, charlie).
parent(alice, diane).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
EOF

PROLOG_TEST_2="
    ['$TEST_OUTPUT_DIR/test_join.pl'],
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/csharp_native_target'),
    compile_predicate_to_csharp(grandparent/2, [], Code),
    write(Code)
"

CODE_2=$(swipl -q -g "$PROLOG_TEST_2" -t halt 2>&1) || {
    print_result "Join query compilation" "FAIL" "Compilation error"
    CODE_2=""
}

if [ -n "$CODE_2" ]; then
    build_and_run_csharp "$CODE_2" "join_query" "alice:charlie"
fi

# Test 3: Query Runtime - Transitive Closure
print_header "Test 3: Query Runtime - Recursive Transitive Closure"

cat > "$TEST_OUTPUT_DIR/test_recursive.pl" <<'EOF'
:- dynamic link/2.
link(a, b).
link(b, c).
link(c, d).

path(X, Y) :- link(X, Y).
path(X, Z) :- link(X, Y), path(Y, Z).
EOF

# Note: This test expects the Query Runtime to work
# If constraint operand issue exists, this will fail
PROLOG_TEST_3="
    ['$TEST_OUTPUT_DIR/test_recursive.pl'],
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/csharp_query_target'),
    compile_predicate_to_csharp(path/2, [target(csharp_query)], Code),
    write(Code)
"

echo -e "${YELLOW}Note: This test may fail due to 'unsupported constraint operand' issue${NC}"

CODE_3=$(swipl -q -g "$PROLOG_TEST_3" -t halt 2>&1) || {
    if echo "$CODE_3" | grep -q "unsupported constraint operand"; then
        print_result "Recursive query compilation" "FAIL" "Known issue: unsupported constraint operand"
    else
        print_result "Recursive query compilation" "FAIL" "Compilation error: $CODE_3"
    fi
    CODE_3=""
}

if [ -n "$CODE_3" ]; then
    # This should generate QueryRuntime-based code
    build_and_run_csharp "$CODE_3" "recursive_query" "a:d"
fi

# Test 4: Stream Target Error Handling
print_header "Test 4: Stream Target - Error Handling for Recursion"

# This should fail gracefully with a helpful error message
PROLOG_TEST_4="
    ['$TEST_OUTPUT_DIR/test_recursive.pl'],
    use_module('$PROJECT_ROOT/src/unifyweaver/targets/csharp_native_target'),
    compile_predicate_to_csharp(path/2, [mode(procedural)], Code),
        write(Code)
"

if swipl -q -g "$PROLOG_TEST_4" -t halt > /dev/null 2>&1; then
    print_result "Error handling for recursive in Stream Target" "FAIL" "Should have failed but didn't"
else
    print_result "Error handling for recursive in Stream Target" "PASS"
fi

# Test 5: Verify dotnet is available
print_header "Test 5: Environment Check"

if command -v dotnet > /dev/null 2>&1; then
    print_result "dotnet CLI available" "PASS"
    dotnet --version > "$TEST_OUTPUT_DIR/dotnet_version.txt"
else
    print_result "dotnet CLI available" "FAIL" "dotnet not found in PATH"
fi

if command -v swipl > /dev/null 2>&1; then
    print_result "SWI-Prolog available" "PASS"
else
    print_result "SWI-Prolog available" "FAIL" "swipl not found in PATH"
fi

# ==========================================
# Test Summary
# ==========================================

print_header "Test Summary"

echo "Tests run:    $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some tests failed.${NC}"
    echo "Set KEEP_TEST_DATA=1 to preserve test artifacts in $TEST_OUTPUT_DIR"
    exit 1
fi
