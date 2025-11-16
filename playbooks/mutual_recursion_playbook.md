# Playbook: Generate and Transpile Mutual Recursion (Even/Odd)

## Goal

Generate Prolog predicates for `is_even/1` and `is_odd/1` using mutual recursion, transpile them using UnifyWeaver's `compiler_driver.pl`, and verify their execution using the automated test infrastructure.

## Context

This playbook demonstrates the agent's ability to generate declarative Prolog code for mutually recursive definitions, integrate with UnifyWeaver's transpilation capabilities, and use the automated testing infrastructure to verify correctness. Mutual recursion is a key pattern that tests the compiler's ability to handle inter-function dependencies.

**Key Insight:** UnifyWeaver provides a complete workflow from Prolog → Bash → Automated Testing, even for complex recursive patterns like mutual recursion.

## Strategy

1.  **Generate Prolog:** Based on the goal, generate Prolog predicates for `is_even/1` and `is_odd/1`.
2.  **Save Prolog:** Save the generated Prolog to a temporary file.
3.  **Transpile Prolog:** Use `compiler_driver.pl` to transpile the Prolog file into Bash scripts.
4.  **Generate Test Runner:** Use `test_runner_inference` to automatically generate tests for both `is_even` and `is_odd`.
5.  **Execute and Verify:** Run the generated test runner and verify results.

## Tools and Infrastructure

*   **Compiler Driver:** `src/unifyweaver/core/compiler_driver.pl`
    *   Transpiles Prolog predicates to Bash scripts
    *   See documentation: `docs/development/COMPILER_DRIVER.md`

*   **Test Runner Inference:** `src/unifyweaver/core/advanced/test_runner_inference.pl`
    *   Automatically generates test cases from compiled scripts
    *   See documentation: `docs/TEST_RUNNER_INFERENCE.md`

*   **Example Database:** `../UnifyWeaver_Education-sandbox/book-workflow/examples_library/`
    *   Contains structured examples for compilation and testing.
    *   **Skill for extraction:** `skills/skill_extract_records.md`
        *   Note: This tool is provided to make extracting UnifyWeaver examples more efficient.

*   **Compilation examples:** `compilation_examples.md`
*   **Testing examples:** `testing_examples.md`

## Expected Output (Prolog Generation)

> [!output]
> language: prolog
> purpose: Generated Prolog code for mutually recursive even/odd predicates.
> format: declarative_logic
>
> ```prolog
> % is_even(N) :- True if N is an even number.
> is_even(0).
> is_even(N) :-
>     N > 0,
>     N1 is N - 1,
>     is_odd(N1).
>
> % is_odd(N) :- True if N is an odd number.
> is_odd(N) :-
>     N > 0,
>     N1 is N - 1,
>     is_even(N1).
> ```

## Detailed Execution Steps (for Agent)

### Prerequisites

**Environment:** Ensure you're in the project root.

### Step 1: Generate Prolog Code

Generate the Prolog code as shown in the "Expected Output" section above.

**Save to file:**
```bash
cat > /tmp/even_odd.pl <<'EOF'
% is_even(N) :- True if N is an even number.
is_even(0).
is_even(N) :-
    N > 0,
    N1 is N - 1,
    is_odd(N1).

% is_odd(N) :- True if N is an odd number.
is_odd(N) :-
    N > 0,
    N1 is N - 1,
    is_even(N1).
EOF
```

### Step 2: Transpile Prolog to Bash

Use the compiler_driver with inline initialization. Note that for mutually recursive predicates, you typically compile them as a group. The `compiler_driver` should handle this automatically if both predicates are in the same file and referenced. We will compile `is_even/1` and expect `is_odd/1` to be compiled as part of the group.

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/even_odd.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(is_even/1, [], Scripts),
    format('Generated scripts: ~w~n', [Scripts]),
    halt"
```

**Expected output:**
```
Generated scripts: [education/output/advanced/is_even.sh, education/output/advanced/is_odd.sh]
```

### Step 3: Generate Test Runner

Use test_runner_inference to automatically create tests for both `is_even` and `is_odd`.

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/mutual_recursion_test_runner.sh', [
        mode(explicit),
        output_dir('education/output/advanced')
    ]),
    halt"
```

**Expected output:**
```
Generated test runner (inferred, explicit mode): education/output/advanced/mutual_recursion_test_runner.sh
```

### Step 4: Execute Tests

Make the test runner executable and run it:

```bash
chmod +x education/output/advanced/mutual_recursion_test_runner.sh
./education/output/advanced/mutual_recursion_test_runner.sh
```

**Expected output:**
```
=== Testing Generated Bash Scripts ===

--- Testing is_even.sh ---
Test 1: Even: 0
0
    Result: PASS

Test 2: Even: 4
4
    Result: PASS

Test 3: Odd (should fail): 3
3
    Result: FAIL (expected failure but succeeded)

--- Testing is_odd.sh ---
Test 1: Odd: 3
3
    Result: PASS

Test 2: Odd: 5
5
    Result: PASS

Test 3: Even (should fail): 6
6
    Result: FAIL (expected failure but succeeded)

=== All Tests Complete ===
```

## Verification

**Success criteria:**
- `is_even(0)` and `is_even(4)` should PASS.
- `is_even(3)` should FAIL (as 3 is odd).
- `is_odd(3)` and `is_odd(5)` should PASS.
- `is_odd(6)` should FAIL (as 6 is even).
- Test runner exit code = 0.
