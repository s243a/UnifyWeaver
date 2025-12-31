#!/usr/bin/env bash
#
# Integration tests for IronPython pipeline code generation (Phase 3)
# Tests runtime-specific headers and CLR integration
#
# Usage: ./test_ironpython_pipeline.sh
# Returns: 0 on success, 1 on failure
#
# Requirements:
#   - IronPython 3.4+ (ipy command)
#   - .NET 8.0+ runtime
#   - Must be run in proot-distro debian environment

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

# Test output directory (use project-local directory for Termux compatibility)
TEST_OUTPUT_DIR="$PROJECT_ROOT/.test_output/ironpython_tests_$$"
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

# Check if IronPython is available (also check in proot debian)
check_ironpython() {
    if command -v ipy &> /dev/null; then
        return 0
    elif [ -x "$HOME/.dotnet/tools/ipy" ]; then
        # Add to PATH for this session
        export PATH="$HOME/.dotnet/tools:$PATH"
        return 0
    else
        echo -e "${YELLOW}Warning: IronPython not found (ipy command not available)${NC}"
        echo "To install IronPython:"
        echo "  1. Install .NET 8.0: https://dotnet.microsoft.com/download"
        echo "  2. Run: dotnet tool install -g IronPython.Console"
        echo "  3. Add ~/.dotnet/tools to PATH"
        return 1
    fi
}

# Check if CPython is available (fallback)
check_cpython() {
    if command -v python3 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Run Python code with IronPython
run_ironpython() {
    local code="$1"
    local test_name="$2"
    local input="${3:-}"

    local test_file="$TEST_OUTPUT_DIR/${test_name}.py"
    echo "$code" > "$test_file"

    if [ -n "$input" ]; then
        echo "$input" | ipy "$test_file" 2>&1
    else
        ipy "$test_file" 2>&1
    fi
}

# Run Python code with CPython (fallback for syntax tests)
run_cpython() {
    local code="$1"
    local test_name="$2"
    local input="${3:-}"

    local test_file="$TEST_OUTPUT_DIR/${test_name}.py"
    echo "$code" > "$test_file"

    if [ -n "$input" ]; then
        echo "$input" | python3 "$test_file" 2>&1
    else
        python3 "$test_file" 2>&1
    fi
}

print_header "UnifyWeaver IronPython Integration Tests (Phase 3)"

# Check for IronPython
if ! check_ironpython; then
    echo -e "${YELLOW}Skipping IronPython tests - IronPython not available${NC}"

    # Run syntax validation tests with CPython if available
    if check_cpython; then
        echo "Running syntax validation tests with CPython..."
    else
        echo -e "${RED}Neither IronPython nor CPython available. Exiting.${NC}"
        exit 1
    fi
fi

HAS_IRONPYTHON=false
if check_ironpython; then
    HAS_IRONPYTHON=true
fi

# =============================================================================
# Test 1: Generate IronPython header via Prolog
# =============================================================================
print_header "Test 1: Generate IronPython Header"

cd "$PROJECT_ROOT"
IRONPYTHON_HEADER=$(swipl -g "use_module('src/unifyweaver/targets/python_target'), python_target:clear_binding_imports, python_target:pipeline_header(jsonl, ironpython, Header), write(Header), halt" 2>&1)

if echo "$IRONPYTHON_HEADER" | grep -q "import clr" && \
   echo "$IRONPYTHON_HEADER" | grep -q "clr.AddReference" && \
   echo "$IRONPYTHON_HEADER" | grep -q "from System import"; then
    print_result "IronPython header has CLR imports" "PASS"
else
    print_result "IronPython header has CLR imports" "FAIL" "Missing CLR import statements"
fi

if echo "$IRONPYTHON_HEADER" | grep -q "to_dotnet_dict"; then
    print_result "IronPython header has conversion helpers" "PASS"
else
    print_result "IronPython header has conversion helpers" "FAIL" "Missing to_dotnet_dict helper"
fi

# =============================================================================
# Test 2: Generate full pipeline with IronPython runtime
# =============================================================================
print_header "Test 2: Full Pipeline with IronPython Runtime"

# Create a test file for the pipeline compilation
TEST_PROLOG_FILE="$TEST_OUTPUT_DIR/test_pipeline.pl"
cat > "$TEST_PROLOG_FILE" <<'PROLOG_EOF'
:- use_module('src/unifyweaver/targets/python_target').

test :-
    abolish(user:greet/2),
    assert((greet(Name, Greeting) :- atom_concat('Hello, ', Name, Greeting))),
    python_target:compile_predicate_to_python(greet/2, [
        pipeline_input(true),
        output_format(object),
        arg_names(['Name', 'Greeting']),
        runtime(ironpython)
    ], Code),
    write(Code),
    abolish(user:greet/2),
    halt.

:- test.
PROLOG_EOF

PIPELINE_CODE=$(swipl "$TEST_PROLOG_FILE" 2>&1)

if echo "$PIPELINE_CODE" | grep -q "#!/usr/bin/env ipy"; then
    print_result "Pipeline uses IronPython shebang" "PASS"
else
    print_result "Pipeline uses IronPython shebang" "FAIL" "Expected #!/usr/bin/env ipy"
fi

if echo "$PIPELINE_CODE" | grep -q "def greet(stream"; then
    print_result "Pipeline has greet function" "PASS"
else
    print_result "Pipeline has greet function" "FAIL" "Missing greet function"
fi

# =============================================================================
# Test 3: IronPython CLR Integration (if IronPython available)
# =============================================================================
if [ "$HAS_IRONPYTHON" = true ]; then
    print_header "Test 3: IronPython CLR Integration"

    CLR_TEST=$(cat <<'PYTHON_EOF'
#!/usr/bin/env ipy
import clr
clr.AddReference('System')
from System import Math, DateTime, String
from System.Collections.Generic import Dictionary, List

# Test Math
sqrt_result = Math.Sqrt(16.0)
print(f"Math.Sqrt(16) = {sqrt_result}")

# Test DateTime
now = DateTime.Now
print(f"DateTime.Now.Year = {now.Year}")

# Test Dictionary
d = Dictionary[String, object]()
d["name"] = "test"
d["value"] = 42
print(f"Dictionary: name={d['name']}, value={d['value']}")

# Test List
lst = List[int]()
lst.Add(1)
lst.Add(2)
lst.Add(3)
print(f"List count: {lst.Count}")

print("CLR_TEST_SUCCESS")
PYTHON_EOF
)

    OUTPUT=$(run_ironpython "$CLR_TEST" "clr_test" 2>&1) || true

    if echo "$OUTPUT" | grep -q "CLR_TEST_SUCCESS"; then
        print_result "IronPython CLR Math integration" "PASS"
        print_result "IronPython CLR DateTime integration" "PASS"
        print_result "IronPython CLR Dictionary integration" "PASS"
        print_result "IronPython CLR List integration" "PASS"
    else
        # Check individual results
        if echo "$OUTPUT" | grep -q "Math.Sqrt(16) = 4"; then
            print_result "IronPython CLR Math integration" "PASS"
        else
            print_result "IronPython CLR Math integration" "FAIL" "$OUTPUT"
        fi

        if echo "$OUTPUT" | grep -q "DateTime.Now.Year"; then
            print_result "IronPython CLR DateTime integration" "PASS"
        else
            print_result "IronPython CLR DateTime integration" "FAIL" "$OUTPUT"
        fi

        if echo "$OUTPUT" | grep -q "Dictionary:"; then
            print_result "IronPython CLR Dictionary integration" "PASS"
        else
            print_result "IronPython CLR Dictionary integration" "FAIL" "$OUTPUT"
        fi

        if echo "$OUTPUT" | grep -q "List count:"; then
            print_result "IronPython CLR List integration" "PASS"
        else
            print_result "IronPython CLR List integration" "FAIL" "$OUTPUT"
        fi
    fi

    # =============================================================================
    # Test 4: Pipeline execution with IronPython
    # =============================================================================
    print_header "Test 4: Pipeline Execution with IronPython"

    PIPELINE_EXEC_TEST=$(cat <<'PYTHON_EOF'
#!/usr/bin/env ipy
import sys
import json
import clr

clr.AddReference('System')
from System import String
from System.Collections.Generic import Dictionary

def to_dotnet_dict(py_dict):
    result = Dictionary[String, object]()
    for k, v in py_dict.items():
        result[str(k)] = v
    return result

def process_record(record):
    # Simple transformation: uppercase name
    name = record.get('name', '').upper()
    return {'name': name, 'processed': True}

# Test with JSONL input
test_input = '{"name": "alice"}\n{"name": "bob"}'
for line in test_input.split('\n'):
    if line.strip():
        record = json.loads(line)
        result = process_record(record)
        # Convert to .NET dict and back to verify helpers work
        dotnet_dict = to_dotnet_dict(result)
        print(json.dumps(result))

print("PIPELINE_EXEC_SUCCESS")
PYTHON_EOF
)

    OUTPUT=$(run_ironpython "$PIPELINE_EXEC_TEST" "pipeline_exec_test" 2>&1) || true

    if echo "$OUTPUT" | grep -q "PIPELINE_EXEC_SUCCESS" && \
       echo "$OUTPUT" | grep -q '"name": "ALICE"'; then
        print_result "Pipeline execution with IronPython" "PASS"
    else
        print_result "Pipeline execution with IronPython" "FAIL" "$OUTPUT"
    fi

else
    echo -e "${YELLOW}Skipping IronPython runtime tests - IronPython not available${NC}"
    echo "Tests 3-4 require IronPython to be installed"
fi

# =============================================================================
# Test 5: Jython header generation
# =============================================================================
print_header "Test 5: Jython Header Generation"

JYTHON_HEADER=$(swipl -g "use_module('src/unifyweaver/targets/python_target'), python_target:clear_binding_imports, python_target:pipeline_header(jsonl, jython, Header), write(Header), halt" 2>&1)

if echo "$JYTHON_HEADER" | grep -q "from java.lang import" && \
   echo "$JYTHON_HEADER" | grep -q "HashMap, ArrayList"; then
    print_result "Jython header has Java imports" "PASS"
else
    print_result "Jython header has Java imports" "FAIL" "Missing Java import statements"
fi

if echo "$JYTHON_HEADER" | grep -q "to_java_map"; then
    print_result "Jython header has conversion helpers" "PASS"
else
    print_result "Jython header has conversion helpers" "FAIL" "Missing to_java_map helper"
fi

# =============================================================================
# Test 6: PyPy header generation
# =============================================================================
print_header "Test 6: PyPy Header Generation"

PYPY_HEADER=$(swipl -g "use_module('src/unifyweaver/targets/python_target'), python_target:clear_binding_imports, python_target:pipeline_header(jsonl, pypy, Header), write(Header), halt" 2>&1)

if echo "$PYPY_HEADER" | grep -q "#!/usr/bin/env pypy3" && \
   echo "$PYPY_HEADER" | grep -q "JIT-optimized"; then
    print_result "PyPy header has correct shebang and docstring" "PASS"
else
    print_result "PyPy header has correct shebang and docstring" "FAIL" "Missing pypy3 shebang or docstring"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "Test Summary"

echo ""
echo "Tests run:    $TESTS_RUN"
echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
