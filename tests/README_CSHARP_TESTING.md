# C# Target Testing Guide

**Branch:** `feat/csharp-automated-testing`
**Date:** 2025-11-02
**Status:** Active Development

## Overview

This document describes the automated testing infrastructure for UnifyWeaver's C# target language implementation. We provide three complementary testing approaches:

1. **Bash Integration Tests** - Simple, fast validation of compilation and execution
2. **Python Helper via Janus** - Sophisticated output validation and subprocess management
3. **Extended Prolog Test Suite** - Integration with existing PLUnit framework

## Test Infrastructure Files

### Bash Integration Tests

**File:** `tests/integration/test_csharp_targets.sh`

Standalone Bash script that compiles Prolog predicates to C#, builds them with `dotnet build`, and executes the compiled binaries.

**Usage:**
```bash
# Run all tests
./tests/integration/test_csharp_targets.sh

# Keep test artifacts for debugging
KEEP_TEST_DATA=1 ./tests/integration/test_csharp_targets.sh
```

**Test cases:**
- Simple binary facts (Stream Target)
- Join queries (Stream Target)
- Recursive transitive closure (Query Runtime)
- Error handling for unsupported features
- Environment validation (dotnet, swipl)

**When to use:**
- Quick validation during development
- CI/CD pipelines
- Manual testing with artifact preservation

### Python Test Helper

**File:** `tests/helpers/csharp_test_helper.py`

Python module that provides robust C# compilation and execution utilities. Can be called standalone or via Prolog's Janus bridge.

**Standalone usage:**
```bash
# Compile and run a C# file
python3 tests/helpers/csharp_test_helper.py Program.cs

# Compile and check for expected output
python3 tests/helpers/csharp_test_helper.py Program.cs "expected text"

# Check dotnet version
python3 tests/helpers/csharp_test_helper.py --version
```

**From Prolog via Janus:**
```prolog
:- use_module(library(janus)).
:- py_call(importlib:import_module('csharp_test_helper'), _).

% Compile and run
?- Code = "using System; class Program { static void Main() { Console.WriteLine(\"Hello\"); } }",
   py_call(csharp_test_helper:compile_and_run(Code, 'test'), Result),
   format('Success: ~w~nOutput: ~w~n', [Result.success, Result.stdout]).

% Assert output contains substring
?- py_call(csharp_test_helper:assert_output_contains(Code, "Hello", 'test'), Result),
   assertion(Result.assertion_passed == @(true)).
```

**Key features:**
- Automatic dotnet detection (including WSL paths)
- Temporary directory management
- Build and execution in one call
- Rich result dictionary with stdout/stderr/build_output
- Output validation helpers

**When to use:**
- Complex output validation
- From Prolog test suites
- When you need detailed error information

### Prolog-Janus Bridge Tests

**File:** `tests/core/test_csharp_janus.pl`

PLUnit test suite that uses the Python helper via Janus bridge for end-to-end testing.

**Usage:**
```prolog
% Run all Janus-based tests
?- use_module('tests/core/test_csharp_janus').
?- test_csharp_compilation.

% Run specific test
?- test_stream_target_simple_facts.
```

**Test cases:**
- `stream_target_simple_facts` - Binary fact compilation
- `stream_target_join_query` - Join query compilation
- `stream_target_error_on_recursion` - Error handling
- `query_runtime_basic_recursion` - Recursive queries (blocked: known issue)
- `dotnet_cli_available` - Environment check

**When to use:**
- Integration testing from Prolog
- Automated test runs
- When you want PLUnit reporting

### Extended Prolog Test Suite

**File:** `tests/core/test_csharp_query_target.pl` (enhanced)

Original test suite enhanced with:
- Build-first execution approach (no more `dotnet run` hang)
- `SKIP_CSHARP_EXECUTION` environment variable support
- Improved error reporting
- Helper predicates for finding and executing compiled binaries

**Usage:**
```prolog
% Run all Query Runtime tests
?- use_module('tests/core/test_csharp_query_target').
?- test_csharp_query_target.

% Skip execution phase (plan validation only)
$ SKIP_CSHARP_EXECUTION=1 swipl -l tests/core/test_csharp_query_target.pl \
                                -g test_csharp_query_target -t halt
```

**Test cases:**
- `verify_fact_plan` - Simple fact query plans
- `verify_join_plan` - Join query plans
- `verify_selection_plan` - Filter query plans
- `verify_recursive_plan` - Recursive fixpoint plans

**When to use:**
- Core query planner validation
- Regression testing for IR generation
- Performance benchmarking (with `KEEP_TEST_DATA`)

### Constraint Operand Diagnostic Tests

**File:** `tests/core/test_csharp_constraint_operand.pl`

Specialized test suite for diagnosing the "unsupported constraint operand" issue encountered by Codex.

**Usage:**
```prolog
% Run all diagnostic tests
?- use_module('tests/core/test_csharp_constraint_operand').
?- test_constraint_operand_issue.

% Run individual tests
?- test_simple_equality.
?- test_variable_unification.
?- test_recursive_with_constraints.
```

**Test cases:**
- Simple equality constraints (`Job = engineer`)
- Variable unification (`X \= Y`)
- Recursive queries with constraints

**When to use:**
- Investigating constraint-related compilation failures
- Adding new constraint support
- Debugging query planner issues

## Testing Workflow

### 1. Quick Validation (30 seconds)

```bash
# Bash tests only
./tests/integration/test_csharp_targets.sh
```

### 2. Comprehensive Testing (2-3 minutes)

```bash
# All test approaches
./tests/integration/test_csharp_targets.sh && \
swipl -l tests/core/test_csharp_janus.pl -g test_csharp_compilation -t halt && \
swipl -l tests/core/test_csharp_query_target.pl -g test_csharp_query_target -t halt
```

### 3. Debugging Failures

```bash
# Keep artifacts and run verbose
KEEP_TEST_DATA=1 ./tests/integration/test_csharp_targets.sh

# Check generated files
ls -la /tmp/unifyweaver_csharp_tests_*/

# Run Prolog tests with tracing
swipl -l tests/core/test_csharp_query_target.pl
?- trace, test_csharp_query_target.
```

### 4. CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Test C# Target
  run: |
    # Install dependencies
    dotnet --version
    swipl --version
    python3 --version

    # Run tests
    ./tests/integration/test_csharp_targets.sh
    swipl -l tests/core/test_csharp_query_target.pl -g test_csharp_query_target -t halt
```

## Known Issues and Workarounds

### Issue 1: `dotnet run` Hang

**Problem:** `dotnet run` hangs indefinitely when called from `process_create/3` in WSL.

**Solution:** Use build-first-then-execute approach:
1. `dotnet build --no-restore` to compile
2. Find compiled executable in `bin/Debug/net9.0/`
3. Execute binary directly (or with `dotnet <dll>`)

**Documentation:** `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md`

**Status:** ✅ Implemented in all test frameworks

### Issue 2: Unsupported Constraint Operand

**Problem:** Query planner throws "unsupported constraint operand _168" error when compiling certain recursive queries with constraints.

**Location:** `constraint_operand/3` in `src/unifyweaver/targets/csharp_query_target.pl`

**Workaround:** Use `test_csharp_constraint_operand.pl` to isolate the specific pattern causing the error.

**Status:** ⚠️ Under investigation

### Issue 3: Janus Bridge Availability

**Problem:** SWI-Prolog may not have Janus support compiled in all distributions.

**Check:**
```prolog
?- use_module(library(janus)).
% Should succeed if Janus is available
```

**Workaround:** Use Bash or pure Prolog tests if Janus is unavailable.

**Status:** ℹ️ Optional dependency

## Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `SKIP_CSHARP_EXECUTION` | Skip binary execution, plan validation only | (unset) | `SKIP_CSHARP_EXECUTION=1` |
| `KEEP_TEST_DATA` | Preserve test artifacts for debugging | (unset) | `KEEP_TEST_DATA=1` |
| `CSHARP_QUERY_OUTPUT_DIR` | Custom output directory for test artifacts | `tmp` | `CSHARP_QUERY_OUTPUT_DIR=/tmp/my_tests` |
| `CSHARP_QUERY_KEEP_ARTIFACTS` | Keep artifacts (alternative to KEEP_TEST_DATA) | `false` | `CSHARP_QUERY_KEEP_ARTIFACTS=true` |

## Test Output Examples

### Successful Bash Test Run

```
==========================================
C# Target Integration Tests
==========================================

==========================================
Test 1: Stream Target - Simple Binary Facts
==========================================
✓ Simple facts compilation
✓ simple_facts

==========================================
Test 2: Stream Target - Join Query
==========================================
✓ Join query compilation
✓ join_query

==========================================
Test Summary
==========================================
Tests run:    5
Tests passed: 5
Tests failed: 0

All tests passed!
```

### Failed Test with Artifacts

```
✗ recursive_query
  Error: Compilation error: unsupported constraint operand _168

Set KEEP_TEST_DATA=1 to preserve test artifacts in /tmp/unifyweaver_csharp_tests_12345
```

### Prolog Test Output

```
=== Testing C# query target ===
Plan generation: SUCCESS
  (query runtime execution: PASS)
Plan generation: SUCCESS
  (query runtime execution: PASS)
=== C# query target tests complete ===
```

## Adding New Tests

### Bash Test

1. Edit `tests/integration/test_csharp_targets.sh`
2. Add test case following existing pattern
3. Update `TESTS_RUN` counter
4. Add expected output validation

### Python Helper Function

1. Edit `tests/helpers/csharp_test_helper.py`
2. Add new function (e.g., `assert_output_matches_regex`)
3. Add to module exports
4. Document in docstring

### Prolog PLUnit Test

1. Edit `tests/core/test_csharp_janus.pl` or create new file
2. Add test in PLUnit format:
```prolog
test(my_new_test, [
    condition(check_dotnet_available)
]) :-
    % Test body
    assertion(Result == Expected).
```

### Diagnostic Test

1. Edit `tests/core/test_csharp_constraint_operand.pl`
2. Add test predicate following `test_simple_equality` pattern
3. Add to `test_constraint_operand_issue/0` main suite

## Performance Considerations

**Test execution times (approximate):**

| Test Suite | Time | When to Use |
|------------|------|-------------|
| Bash integration tests | 15-30s | Every commit |
| Janus bridge tests | 20-40s | Before push |
| Full Prolog suite | 30-60s | Before PR |
| Constraint diagnostics | 5-10s | When debugging |

**Optimization tips:**
- Use `SKIP_CSHARP_EXECUTION=1` for quick plan validation
- Run Bash tests first (fastest feedback)
- Keep test data only when debugging (faster cleanup)
- Use parallel test execution in CI/CD

## Troubleshooting

### "dotnet not found"

**Symptoms:** Tests fail with "dotnet CLI not found"

**Solutions:**
1. Install .NET SDK 9.0+
2. Check `which dotnet`
3. For WSL, ensure Linux native dotnet is installed (not Windows version)

### "swipl not found"

**Symptoms:** Bash tests can't run Prolog

**Solutions:**
1. Install SWI-Prolog 9.0+
2. Check `which swipl`
3. Add to PATH if needed

### "Janus module not found"

**Symptoms:** Janus tests fail to load

**Solutions:**
1. Check if Janus is available: `swipl -g "use_module(library(janus))" -t halt`
2. Install SWI-Prolog with Python support
3. Use Bash or pure Prolog tests instead

### "Permission denied" executing test scripts

**Symptoms:** Can't execute `.sh` scripts

**Solutions:**
```bash
chmod +x tests/integration/test_csharp_targets.sh
chmod +x tests/helpers/csharp_test_helper.py
```

### Tests hang indefinitely

**Symptoms:** Tests never complete

**Possible causes:**
1. Still using `dotnet run` instead of build-first approach
2. Infinite loop in generated C# code
3. Deadlock in fixpoint iteration

**Solutions:**
1. Verify using build-first approach (check `run_dotnet_plan_build_first`)
2. Set timeout: `timeout 60 ./tests/integration/test_csharp_targets.sh`
3. Examine generated code in kept artifacts

## Future Improvements

### Planned Enhancements

1. **Parallel test execution** - Run independent tests concurrently
2. **Code coverage tracking** - Measure test coverage of IR generation
3. **Performance regression tests** - Detect slowdowns in query execution
4. **Cross-platform testing** - Validate on Linux, macOS, Windows
5. **Automated issue reporting** - Create GitHub issues from test failures

### Contributing

When adding new C# target features:

1. ✅ Add test case to appropriate suite
2. ✅ Document any new environment variables
3. ✅ Update this README with new test patterns
4. ✅ Ensure tests pass on clean checkout

## References

- **Build-First Solution:** `docs/CSHARP_DOTNET_RUN_HANG_SOLUTION.md`
- **C# Target Review:** `context/codex/UnifyWeaver/context/csharp-target-review/README.md`
- **Janus Bridge Examples:** `context/other-projects/JanusBridge/src/examples/`
- **PLUnit Documentation:** https://www.swi-prolog.org/pldoc/package/plunit.html

---

**Last Updated:** 2025-11-02
**Maintainer:** UnifyWeaver C# Target Team
**Branch:** `feat/csharp-automated-testing`
