---
file_type: UnifyWeaver Example Library
spec_version: 1.0
---

# Testing and Verification Examples

This file contains examples of testing compiled bash scripts using UnifyWeaver's test infrastructure.

---

### Factorial Test Runner

> [!example-record]
> id: 20251108-factorial-test
> name: unifyweaver.testing.factorial_runner
> pattern: test_runner_inference
> parent_example: unifyweaver.compilation.factorial

After compiling factorial.sh, use test_runner_inference to automatically generate and execute tests.

**Prerequisites:**
- Compiled factorial.sh exists in `education/output/advanced/`
- See [unifyweaver.compilation.factorial](compilation_examples.md#factorial-compilation) for compilation

**Generate Test Runner:**

From project root:

```bash
cd $UNIFYWEAVER_HOME

swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/test_runner.sh', [
        mode(explicit),
        output_dir('education/output/advanced')
    ]),
    halt"
```

**What This Does:**
1. Scans `education/output/advanced/` for `.sh` files
2. Extracts function signatures from each script
3. Infers appropriate test cases based on pattern recognition
4. Generates executable test runner script
5. Includes dependency management (sources scripts in correct order)

**Execute Tests:**

```bash
chmod +x education/output/advanced/test_runner.sh
./education/output/advanced/test_runner.sh
```

**Expected Output:**

```
=== Testing Generated Bash Scripts ===

--- Testing factorial.sh ---
Test 1: Base case 0
0:1
    Result: PASS

Test 2: Base case 1
1:1
    Result: PASS

Test 3: Larger value
5:120
    Result: PASS

=== All Tests Complete ===
```

(Note: Helper functions like factorial_op, build_range_down, and fold_left also get generic tests, which may fail if they require specific input formats)

**Test Runner Features:**
- **Automatic test case inference**: Recognizes factorial pattern, generates base case (0, 1) and recursive cases
- **Exit code validation**: Checks both output and return codes
- **Dependency handling**: Sources required scripts before testing
- **Clear reporting**: Shows expected vs actual results

---

### Manual Test Execution

> [!example-record]
> id: 20251108-manual-test
> name: unifyweaver.testing.manual_factorial
> pattern: manual_verification

For quick verification without generating a full test runner, you can execute the compiled script directly.

**Direct Execution:**

The compiled scripts define bash functions but don't have a main execution block by default. To test manually:

**Method 1: Source and call function**
```bash
cd $UNIFYWEAVER_HOME
source education/output/advanced/factorial.sh

# The function is now available
# But factorial/2 expects 2 arguments and bash functions don't work that way
# So this shows the limitation of direct execution
```

**Method 2: Use test_runner (Recommended)**
See [unifyweaver.testing.factorial_runner](#factorial-test-runner)

**Why test_runner is better:**
- Handles argument passing correctly
- Validates results automatically
- Works with all predicate arities
- Manages dependencies

---

### Test Runner Modes

> [!example-record]
> id: 20251108-test-modes
> name: unifyweaver.testing.runner_modes
> pattern: test_configuration

The test_runner_inference module supports different output modes.

**Explicit Mode (Default):**
```prolog
generate_test_runner_inferred(Path, [mode(explicit)])
```

Shows detailed test execution:
```
Test 1: factorial 5
Command: factorial_stream <<< "5"
Expected: 120
Output: 120
    Result: PASS
```

**Concise Mode:**
```prolog
generate_test_runner_inferred(Path, [mode(concise)])
```

Shows only summary:
```
factorial.sh: 4/4 PASSED
```

**Hybrid Mode:**
```prolog
generate_test_runner_inferred(Path, [mode(hybrid)])
```

Shows failures in detail, successes concisely:
```
factorial.sh: Test 1-3 PASSED
Test 4: factorial 100
    Result: FAIL (expected 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000, got stack overflow)
```

---

### Expected Failure Handling

> [!example-record]
> id: 20251108-expected-fail
> name: unifyweaver.testing.expected_failures
> pattern: negative_testing

Test runner can detect and handle expected failures correctly.

**Example: Testing Invalid Input**

When a test description contains "should fail" or "NOT", the test runner expects the function to fail.

**Generated Test:**
```bash
# Test: Check ishmael is NOT ancestor of jacob (should fail)
ancestor_check "ishmael" "jacob"
if [ $? -eq 0 ]; then
    echo "    Result: FAIL (expected failure but succeeded)"
else
    echo "    Result: PASS (correctly failed)"
fi
```

**Pattern Recognition:**
The test_runner_inference module recognizes these patterns in test descriptions:
- "should fail"
- "NOT"
- "invalid"
- "error"

And inverts the success criteria accordingly.

---

### Multi-Script Testing

> [!example-record]
> id: 20251108-multi-script
> name: unifyweaver.testing.multiple_scripts
> pattern: batch_testing

Testing multiple scripts with dependencies.

**Scenario:** Testing ancestor.sh which depends on parent.sh

**Generate Test Runner:**
```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/test_runner.sh', [
        output_dir('education/output/advanced')
    ]),
    halt"
```

**What Happens:**
1. Scanner finds: parent.sh, ancestor.sh, factorial.sh
2. Test runner sources ALL scripts at top
3. For each test block:
   - Sources all OTHER scripts
   - Sources target script LAST (for precedence)
   - Runs tests

**Dependency Management:**
```bash
# Global preload
source parent.sh
source ancestor.sh
source factorial.sh

# Test block for ancestor
source parent.sh  # dependency
source ancestor.sh  # target (last = highest precedence)

# Run ancestor tests...
```

This ensures dependencies are always available.

---

### Custom Output Directory

> [!example-record]
> id: 20251108-custom-output
> name: unifyweaver.testing.custom_directory
> pattern: configuration

Testing scripts in non-default locations.

**Compile to Custom Directory:**
```prolog
compile(factorial/2, [output_directory('/tmp/my_scripts')], Scripts)
```

**Generate Test Runner for Custom Directory:**
```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('/tmp/my_scripts/test_runner.sh', [
        output_dir('/tmp/my_scripts')
    ]),
    halt"
```

**Execute:**
```bash
chmod +x /tmp/my_scripts/test_runner.sh
/tmp/my_scripts/test_runner.sh
```
