# Playbook: Generate and Transpile Factorial Prolog

## Goal

Generate a Prolog program for calculating the factorial of a number, transpile it using UnifyWeaver's `compiler_driver.pl`, and verify its execution using automated test infrastructure.

## Context

This playbook demonstrates the agent's ability to generate declarative Prolog code based on a natural language specification, integrate with UnifyWeaver's transpilation capabilities, and use the automated testing infrastructure to verify correctness.

**Key Insight:** UnifyWeaver provides a complete workflow from Prolog → Bash → Automated Testing. You don't need to manually execute scripts; use the test_runner_inference infrastructure instead.

## Strategy

1.  **Generate Prolog:** Based on the goal, generate a Prolog program for the factorial function.
2.  **Save Prolog:** Save the generated Prolog to a temporary file.
3.  **Transpile Prolog:** Use `compiler_driver.pl` to transpile the Prolog file into a Bash script.
4.  **Generate Test Runner:** Use `test_runner_inference` to automatically generate tests.
5.  **Execute and Verify:** Run the generated test runner and verify results.

## Tools and Infrastructure

*   **Compiler Driver:** `src/unifyweaver/core/compiler_driver.pl`
    *   Transpiles Prolog predicates to Bash scripts
    *   See documentation: `docs/development/COMPILER_DRIVER.md`

*   **Test Runner Inference:** `src/unifyweaver/core/advanced/test_runner_inference.pl`
    *   Automatically generates test cases from compiled scripts
    *   See documentation: `docs/TEST_RUNNER_INFERENCE.md`

*   **Example Database:** `../UnifyWeaver_Education-sandbox/book-workflow/examples_library/`
    *   Compilation examples: `compilation_examples.md`
    *   Testing examples: `testing_examples.md`

## Expected Output (Prolog Generation)

> [!output]
> language: prolog
> purpose: Generated Prolog code for the factorial function.
> format: declarative_logic
>
> ```prolog
> % factorial(N, F) :- F is N!
> factorial(0, 1).
> factorial(N, F) :-
>     N > 0,
>     N1 is N - 1,
>     factorial(N1, F1),
>     F is N * F1.
> ```

## Minimal Playbook

**See example:** `unifyweaver.workflow.factorial_complete` in [compilation_examples.md](../UnifyWeaver_Education-sandbox/book-workflow/examples_library/compilation_examples.md#factorial-complete-workflow)

The minimal playbook contains executable steps. The sections below provide detailed explanations for learning purposes.

## Detailed Execution Steps (for Agent)

### Prerequisites

**Environment:** Ensure you're in the project root. See [workflow_environment.md](../docs/development/ai-skills/workflow_environment.md) for details.

```bash
cd $UNIFYWEAVER_HOME
# or
cd /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver/context/sandbox/UnifyWeaver-sandbox
```

### Step 1: Generate Prolog Code

Generate the Prolog code as shown in the "Expected Output" section above, or reference the example:

**Example Reference:** `unifyweaver.compilation.factorial` in [compilation_examples.md](../UnifyWeaver_Education-sandbox/book-workflow/examples_library/compilation_examples.md#factorial-compilation)

**Save to file:**
```bash
cat > /tmp/factorial.pl <<'EOF'
% factorial(N, F) - F is N!
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
EOF
```

### Step 2: Transpile Prolog to Bash

Use the compiler_driver with inline initialization:

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/factorial.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [], Scripts),
    format('Generated scripts: ~w~n', [Scripts]),
    halt"
```

**What this does:**
- Sets up the file_search_path (inline initialization)
- Loads your Prolog file
- Compiles factorial/2 to bash
- Reports generated script locations
- Exits SWI-Prolog

**Expected output:**
```
Generated scripts: [education/output/advanced/factorial.sh]
```

### Step 3: Generate Test Runner

Use test_runner_inference to automatically create tests:

**Example Reference:** `unifyweaver.testing.factorial_runner` in [testing_examples.md](../UnifyWeaver_Education-sandbox/book-workflow/examples_library/testing_examples.md#factorial-test-runner)

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/test_runner.sh', [
        mode(explicit),
        output_dir('education/output/advanced')
    ]),
    halt"
```

**What this does:**
- Scans education/output/advanced/ for compiled scripts
- Analyzes factorial.sh's function signature
- Infers appropriate test cases (base case 0, recursive cases 1, 5, 10, etc.)
- Generates executable test_runner.sh

**Expected output:**
```
Generated test runner (inferred, explicit mode): education/output/advanced/test_runner.sh
```

### Step 4: Execute Tests

Make the test runner executable and run it:

```bash
chmod +x education/output/advanced/test_runner.sh
./education/output/advanced/test_runner.sh
```

**Expected output:**
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

## Verification

**Success criteria:**
- All factorial tests should PASS
- factorial(0) = 1 (base case)
- factorial(1) = 1 (base case)
- factorial(5) = 120 (larger value)
- Test runner exit code = 0
- Note: Helper functions may show FAIL for generic tests (this is expected)

**If tests fail:**
- Check the generated factorial.sh for correctness
- Verify the Prolog source matches the expected pattern
- Check docs/TEST_RUNNER_INFERENCE.md for troubleshooting
