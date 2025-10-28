# Test Runner Inference System

## Overview

The test runner inference system automatically generates test runner scripts for compiled Bash scripts. It analyzes function signatures, infers appropriate test cases, and generates executable test scripts with dependency-aware sourcing and exit code validation.

## Location

- **Generator**: `src/unifyweaver/core/advanced/test_runner_inference.pl`
- **Generated output**: Typically `education/output/advanced/test_runner.sh` or `output/advanced/test_runner.sh`

## Features

### 1. Dependency-Aware Sourcing

The test runner ensures dependencies are available before testing:

1. **Global preload**: All scripts are sourced once at the top
2. **Per-block sourcing**: Before each test block, all OTHER scripts are sourced
3. **Target script last**: The script under test is sourced last (takes precedence in name conflicts)

This pattern ensures:
- Dependencies like `parent.sh` are available before testing `ancestor.sh`
- Name conflicts favor the script being tested
- Functions defined in dependencies are accessible

### 2. Exit Code Checking

The test runner validates both output and exit codes:

**For `*_check` functions** (functions ending in `_check`):
- These return exit codes (0=success, 1=failure), not output
- Test runner checks the exit code directly
- Example: `ancestor_check "isaac" "judah"` → checks if exit code is 0

**For regular functions**:
- Captures both stdout and stderr
- Displays any output produced
- Checks exit code: 0=success, non-zero=failure
- Example: `ancestor "isaac" "judah"` → displays output and checks exit code

### 3. Expected Failure Handling

Test descriptions indicating expected failures are detected:
- Descriptions containing "should fail" or "NOT" expect the function to fail
- When function fails as expected: "Result: PASS (correctly failed)"
- When function succeeds unexpectedly: "Result: FAIL (expected failure but succeeded)"

Example test case:
```prolog
test('Check ishmael is NOT ancestor of jacob (should fail)', ['ishmael', 'jacob'])
```

### 4. Intelligent Test Case Inference

The system infers appropriate test cases based on:

#### Pattern Recognition
- **Transitive closure** (ancestor, reachable): Tests with meaningful names from family tree
- **Linear recursion**: Tests with base cases and recursive cases
- **Mutual recursion** (even/odd): Tests with valid/invalid inputs
- **Tree recursion**: Tests with empty, single-node, and complex trees
- **Accumulator patterns**: Tests with empty lists and accumulators

#### Example Inferred Tests

For `ancestor/2` (transitive closure):
```bash
Test 1: Check isaac is ancestor of judah
ancestor "isaac" "judah"
    Result: PASS

Test 3: Check ishmael is NOT ancestor of jacob (should fail)
ancestor "ishmael" "jacob"
    Result: PASS (correctly failed)
```

For `ancestor_check/2` (exit code function):
```bash
Test 1: Check isaac is ancestor of judah
if ancestor_check "isaac" "judah"; then
    echo "    Result: PASS"
else
    echo "    Result: FAIL"
fi
```

## Usage

### Basic Usage

```prolog
% Initialize environment
?- ['education/init'].

% Load the test runner inference module
?- use_module('src/unifyweaver/core/advanced/test_runner_inference').

% Generate test runner for education examples
?- generate_test_runner_inferred(
    'education/output/advanced/test_runner.sh',
    [output_dir('education/output/advanced')]
).
```

### Running the Generated Tests

```bash
cd education/output/advanced
bash test_runner.sh
```

### Output Format

```
=== Testing Generated Bash Scripts ===

--- Testing ancestor.sh ---
Test 1: Check isaac is ancestor of judah
isaac:judah
    Result: PASS

Test 2: Check sarah is ancestor of esau
sarah:esau
    Result: PASS

Test 3: Check ishmael is NOT ancestor of jacob (should fail)
    Result: PASS (correctly failed)

=== All Tests Complete ===
```

## Extending Test Inference

To add new test inference rules, edit `infer_test_cases/2` in `test_runner_inference.pl`:

```prolog
% Rule N: Your new pattern
infer_test_cases(function(Name, Arity, metadata(pattern_type(YourPattern), _, _)),
                 TestCases) :-
    % Your matching logic
    TestCases = [
        test('Description 1', ['arg1', 'arg2']),
        test('Description 2', ['arg3', 'arg4'])
    ].
```

## Implementation Details

### Function Signature Extraction

The system:
1. Scans directory for `.sh` files
2. Parses bash functions using regex
3. Extracts function names and arity (parameter count)
4. Detects helper functions (e.g., `*_stream`, `*_memo`) and excludes them
5. Classifies pattern type from header comments

### Test Generation Modes

- **explicit**: One test per block (current default)
- **concise**: Loop-based test runner (more compact)
- **hybrid**: Mix of explicit and concise (planned)

### Dependency Detection

Dependencies are detected by:
1. Parsing `source` statements in bash scripts
2. Extracting referenced script names
3. Filtering out self-references
4. Building dependency list for sourcing order

## See Also

- `education/04_your_first_program.md` - Tutorial using the test runner
- `src/unifyweaver/core/advanced/advanced_recursive_compiler.pl` - Generates the scripts being tested
- `src/unifyweaver/core/stream_compiler.pl` - Generates fact-based scripts (like parent.sh)
