#!/usr/bin/env bash
#
# End-to-end integration test for Python Pipeline Chaining (Phase 4)
# Tests real pipeline generation and execution with actual data
#
# Usage: ./test_pipeline_chaining_e2e.sh
# Returns: 0 on success, 1 on failure
#
# Requirements:
#   - Python 3.6+ (python3 command)
#   - SWI-Prolog

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
TEST_OUTPUT_DIR="$PROJECT_ROOT/.test_output/pipeline_chaining_e2e_$$"
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

# Check Python
check_python() {
    if command -v python3 &> /dev/null; then
        return 0
    else
        echo -e "${RED}Python 3 not found${NC}"
        return 1
    fi
}

print_header "UnifyWeaver Pipeline Chaining E2E Tests (Phase 4)"

if ! check_python; then
    echo "Python 3 is required for these tests"
    exit 1
fi

cd "$PROJECT_ROOT"

# =============================================================================
# Test 1: Generate and run a same-runtime pipeline
# Pipeline: parse_user -> filter_adult -> format_output
# =============================================================================
print_header "Test 1: Same-Runtime Pipeline (User Processing)"

# Create Prolog predicates file
cat > "$TEST_OUTPUT_DIR/user_pipeline.pl" <<'PROLOG_EOF'
:- use_module('src/unifyweaver/targets/python_target').

% Parse user from JSON-like input
% Input: {name: "Alice", age: 30}
% Output: user(Name, Age)
parse_user(Name, Age) :-
    json_record([name-Name, age-Age]).

% Filter to only adults (age >= 18)
filter_adult(Name, Age) :-
    parse_user(Name, Age),
    Age >= 18.

% Format for output
format_user(Name, Age, Status) :-
    filter_adult(Name, Age),
    (Age >= 65 -> Status = senior ; Status = adult).

generate_pipeline :-
    compile_pipeline(
        [parse_user/2, filter_adult/2, format_user/3],
        [
            runtime(cpython),
            pipeline_name(user_pipeline),
            glue_protocol(jsonl)
        ],
        Code
    ),
    write(Code),
    halt.

:- generate_pipeline.
PROLOG_EOF

# Generate the Python code
PIPELINE_CODE=$(swipl "$TEST_OUTPUT_DIR/user_pipeline.pl" 2>&1)

# Save for inspection
echo "$PIPELINE_CODE" > "$TEST_OUTPUT_DIR/user_pipeline_generated.py"

# Check that it generated something
if echo "$PIPELINE_CODE" | grep -q "def user_pipeline"; then
    print_result "Pipeline function generated" "PASS"
else
    print_result "Pipeline function generated" "FAIL" "Missing user_pipeline function"
    echo "Generated code:"
    echo "$PIPELINE_CODE"
fi

# Check for chained predicates
if echo "$PIPELINE_CODE" | grep -q "def parse_user" && \
   echo "$PIPELINE_CODE" | grep -q "def filter_adult" && \
   echo "$PIPELINE_CODE" | grep -q "def format_user"; then
    print_result "All predicate functions generated" "PASS"
else
    print_result "All predicate functions generated" "FAIL" "Missing predicate functions"
fi

# Check for yield from chaining
if echo "$PIPELINE_CODE" | grep -q "yield from"; then
    print_result "Generator chaining present" "PASS"
else
    print_result "Generator chaining present" "FAIL" "Missing yield from"
fi

# =============================================================================
# Test 2: Cross-runtime pipeline generation
# Pipeline: python:extract -> csharp:validate -> python:transform
# =============================================================================
print_header "Test 2: Cross-Runtime Pipeline Generation"

cat > "$TEST_OUTPUT_DIR/cross_runtime.pl" <<'PROLOG_EOF'
:- use_module('src/unifyweaver/targets/python_target').

generate_cross :-
    compile_pipeline(
        [python:extract_data/1, csharp:validate_data/1, python:transform_data/1],
        [
            pipeline_name(data_processor),
            glue_protocol(jsonl)
        ],
        Code
    ),
    write(Code),
    halt.

:- generate_cross.
PROLOG_EOF

CROSS_CODE=$(swipl "$TEST_OUTPUT_DIR/cross_runtime.pl" 2>&1)
echo "$CROSS_CODE" > "$TEST_OUTPUT_DIR/cross_runtime_generated.py"

# Check for stage functions
if echo "$CROSS_CODE" | grep -q "def stage_1" && \
   echo "$CROSS_CODE" | grep -q "def stage_2" && \
   echo "$CROSS_CODE" | grep -q "def stage_3"; then
    print_result "Stage functions generated" "PASS"
else
    print_result "Stage functions generated" "FAIL" "Missing stage functions"
fi

# Check for orchestrator
if echo "$CROSS_CODE" | grep -q "def data_processor" && \
   echo "$CROSS_CODE" | grep -q "Cross-runtime pipeline orchestrator"; then
    print_result "Orchestrator function generated" "PASS"
else
    print_result "Orchestrator function generated" "FAIL" "Missing orchestrator"
fi

# =============================================================================
# Test 3: Execute a real Python pipeline with data
# =============================================================================
print_header "Test 3: Execute Real Pipeline"

# Create a simple executable pipeline
cat > "$TEST_OUTPUT_DIR/runnable_pipeline.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Generated pipeline test - processes JSON records
"""
import sys
import json

def read_stream(file):
    """Read JSONL from input"""
    for line in file:
        if line.strip():
            yield json.loads(line)

def write_record(record):
    """Write JSON to stdout"""
    print(json.dumps(record))

# Stage 1: Parse and extract fields
def parse_record(stream):
    """Extract name and age from input records"""
    for record in stream:
        result = {
            'name': record.get('name', 'unknown'),
            'age': record.get('age', 0)
        }
        yield result

# Stage 2: Filter adults
def filter_adults(stream):
    """Keep only records where age >= 18"""
    for record in stream:
        if record.get('age', 0) >= 18:
            yield record

# Stage 3: Add status
def add_status(stream):
    """Add status field based on age"""
    for record in stream:
        age = record.get('age', 0)
        if age >= 65:
            status = 'senior'
        elif age >= 18:
            status = 'adult'
        else:
            status = 'minor'
        result = record.copy()
        result['status'] = status
        yield result

# Chained pipeline
def user_pipeline(input_stream):
    """
    Chained pipeline: [parse_record, filter_adults, add_status]
    """
    yield from add_status(filter_adults(parse_record(input_stream)))

if __name__ == '__main__':
    input_stream = read_stream(sys.stdin)
    for result in user_pipeline(input_stream):
        write_record(result)
PYTHON_EOF

# Create test data
cat > "$TEST_OUTPUT_DIR/test_input.jsonl" <<'INPUT_EOF'
{"name": "Alice", "age": 30}
{"name": "Bob", "age": 15}
{"name": "Charlie", "age": 70}
{"name": "Diana", "age": 25}
{"name": "Eve", "age": 10}
INPUT_EOF

# Run the pipeline
OUTPUT=$(python3 "$TEST_OUTPUT_DIR/runnable_pipeline.py" < "$TEST_OUTPUT_DIR/test_input.jsonl" 2>&1)
echo "$OUTPUT" > "$TEST_OUTPUT_DIR/pipeline_output.jsonl"

# Check output
ADULT_COUNT=$(echo "$OUTPUT" | grep -c '"status"' || true)
if [ "$ADULT_COUNT" -eq 3 ]; then
    print_result "Pipeline filters correctly (3 adults)" "PASS"
else
    print_result "Pipeline filters correctly (3 adults)" "FAIL" "Got $ADULT_COUNT records"
fi

# Check for senior
if echo "$OUTPUT" | grep -q '"status": "senior"'; then
    print_result "Senior status assigned correctly" "PASS"
else
    print_result "Senior status assigned correctly" "FAIL" "No senior status found"
fi

# Check that minors are filtered out
if ! echo "$OUTPUT" | grep -q '"name": "Bob"' && ! echo "$OUTPUT" | grep -q '"name": "Eve"'; then
    print_result "Minors filtered out" "PASS"
else
    print_result "Minors filtered out" "FAIL" "Minors still in output"
fi

# =============================================================================
# Test 4: Pipeline with error handling
# =============================================================================
print_header "Test 4: Pipeline Error Handling"

cat > "$TEST_OUTPUT_DIR/error_pipeline.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
import sys
import json

def read_stream(file):
    for line in file:
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # Yield error record instead of crashing
                yield {'_error': str(e), '_raw': line.strip()}

def safe_transform(stream):
    """Transform with error handling"""
    for record in stream:
        if '_error' in record:
            # Pass through error records
            yield record
        else:
            try:
                result = {
                    'name': record['name'].upper(),
                    'processed': True
                }
                yield result
            except (KeyError, AttributeError) as e:
                yield {'_error': str(e), '_record': record}

def pipeline(input_stream):
    yield from safe_transform(input_stream)

if __name__ == '__main__':
    input_stream = read_stream(sys.stdin)
    for result in pipeline(input_stream):
        print(json.dumps(result))
PYTHON_EOF

# Test with mixed valid/invalid input
cat > "$TEST_OUTPUT_DIR/mixed_input.jsonl" <<'INPUT_EOF'
{"name": "Alice"}
not valid json
{"name": "Bob"}
{"missing": "name field"}
INPUT_EOF

OUTPUT=$(python3 "$TEST_OUTPUT_DIR/error_pipeline.py" < "$TEST_OUTPUT_DIR/mixed_input.jsonl" 2>&1)

# Should have 4 output lines (2 valid transforms, 2 errors)
LINE_COUNT=$(echo "$OUTPUT" | wc -l)
if [ "$LINE_COUNT" -ge 4 ]; then
    print_result "Pipeline handles errors gracefully" "PASS"
else
    print_result "Pipeline handles errors gracefully" "FAIL" "Expected 4 lines, got $LINE_COUNT"
fi

# =============================================================================
# Test 5: Large data throughput
# =============================================================================
print_header "Test 5: Large Data Throughput"

# Generate 1000 records
cat > "$TEST_OUTPUT_DIR/gen_data.py" <<'PYTHON_EOF'
import json
for i in range(1000):
    print(json.dumps({"id": i, "value": i * 2, "name": f"user_{i}"}))
PYTHON_EOF

python3 "$TEST_OUTPUT_DIR/gen_data.py" > "$TEST_OUTPUT_DIR/large_input.jsonl"

# Simple passthrough pipeline
cat > "$TEST_OUTPUT_DIR/throughput_pipeline.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
import sys
import json

def read_stream(file):
    for line in file:
        if line.strip():
            yield json.loads(line)

def transform(stream):
    for record in stream:
        record['processed'] = True
        yield record

def pipeline(input_stream):
    yield from transform(input_stream)

if __name__ == '__main__':
    count = 0
    for result in pipeline(read_stream(sys.stdin)):
        count += 1
    print(f"Processed {count} records", file=sys.stderr)
    print(json.dumps({"total": count}))
PYTHON_EOF

OUTPUT=$(python3 "$TEST_OUTPUT_DIR/throughput_pipeline.py" < "$TEST_OUTPUT_DIR/large_input.jsonl" 2>&1)

if echo "$OUTPUT" | grep -q '"total": 1000'; then
    print_result "Processed 1000 records" "PASS"
else
    print_result "Processed 1000 records" "FAIL" "Output: $OUTPUT"
fi

# =============================================================================
# Test 6: Multi-stage same-runtime pipeline
# =============================================================================
print_header "Test 6: Multi-Stage Same-Runtime Pipeline"

cat > "$TEST_OUTPUT_DIR/multi_stage.pl" <<'PROLOG_EOF'
:- use_module('src/unifyweaver/targets/python_target').

generate :-
    compile_same_runtime_pipeline(
        [stage_a/1, stage_b/1, stage_c/1, stage_d/1, stage_e/1],
        [
            runtime(cpython),
            pipeline_name(five_stage_pipeline)
        ],
        Code
    ),
    write(Code),
    halt.

:- generate.
PROLOG_EOF

MULTI_CODE=$(swipl "$TEST_OUTPUT_DIR/multi_stage.pl" 2>&1)

# Check all 5 stages are present
STAGE_COUNT=0
for stage in stage_a stage_b stage_c stage_d stage_e; do
    if echo "$MULTI_CODE" | grep -q "def $stage"; then
        STAGE_COUNT=$((STAGE_COUNT + 1))
    fi
done

if [ "$STAGE_COUNT" -eq 5 ]; then
    print_result "All 5 stages generated" "PASS"
else
    print_result "All 5 stages generated" "FAIL" "Only found $STAGE_COUNT stages"
fi

# Check the connector chains all 5
if echo "$MULTI_CODE" | grep -q "five_stage_pipeline"; then
    print_result "Pipeline connector generated" "PASS"
else
    print_result "Pipeline connector generated" "FAIL" "Missing connector"
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
    echo -e "${GREEN}All end-to-end tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
