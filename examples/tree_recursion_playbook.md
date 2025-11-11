# Playbook: Generate and Transpile Tree Recursion (Tree Sum)

## Goal

Generate a Prolog predicate for `tree_sum/2` using tree recursion, transpile it using UnifyWeaver's `compiler_driver.pl`, and verify its execution using the automated test infrastructure.

## Context

This playbook demonstrates the agent's ability to generate declarative Prolog code for tree-based recursive definitions, integrate with UnifyWeaver's transpilation capabilities, and use the automated testing infrastructure to verify correctness. Tree recursion is a fundamental pattern for processing hierarchical data structures.

**Key Insight:** UnifyWeaver can handle structural recursion, like tree recursion, by generating Bash code that parses and processes the data structure.

## Strategy

1.  **Generate Prolog:** Based on the goal, generate a Prolog predicate for `tree_sum/2`.
2.  **Save Prolog:** Save the generated Prolog to a temporary file.
3.  **Transpile Prolog:** Use `compiler_driver.pl` to transpile the Prolog file into a Bash script.
4.  **Generate Test Runner:** Use `test_runner_inference` to automatically generate tests for `tree_sum`.
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

## Expected Output (Prolog Generation)

> [!output]
> language: prolog
> purpose: Generated Prolog code for tree recursion (tree_sum).
> format: declarative_logic
>
> ```prolog
> % tree_sum(Tree, Sum) :- Sum is the sum of all nodes in the tree.
> % Tree representation: [Value, LeftSubtree, RightSubtree]
> tree_sum([], 0).
> tree_sum([Value, Left, Right], Sum) :-
>     tree_sum(Left, LeftSum),
>     tree_sum(Right, RightSum),
>     Sum is Value + LeftSum + RightSum.
> ```

## Detailed Execution Steps (for Agent)

### Prerequisites

**Environment:** Ensure you're in the project root.

### Step 1: Generate Prolog Code

Generate the Prolog code as shown in the "Expected Output" section above.

**Save to file:**
```bash
cat > /tmp/tree_sum.pl <<'EOF'
% tree_sum(Tree, Sum) :- Sum is the sum of all nodes in the tree.
% Tree representation: [Value, LeftSubtree, RightSubtree]
tree_sum([], 0).
tree_sum([Value, Left, Right], Sum) :-
    tree_sum(Left, LeftSum),
    tree_sum(Right, RightSum),
    Sum is Value + LeftSum + RightSum.
EOF
```

### Step 2: Transpile Prolog to Bash

Use the compiler_driver with inline initialization.

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/tree_sum.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(tree_sum/2, [], Scripts),
    format('Generated scripts: ~w~n', [Scripts]),
    halt"
```

**Expected output:**
```
Generated scripts: [education/output/advanced/tree_sum.sh]
```

### Step 3: Generate Test Runner

Use test_runner_inference to automatically create tests for `tree_sum`.

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/tree_sum_test_runner.sh', [
        mode(explicit),
        output_dir('education/output/advanced')
    ]),
    halt"
```

**Expected output:**
```
Generated test runner (inferred, explicit mode): education/output/advanced/tree_sum_test_runner.sh
```

### Step 4: Execute Tests

Make the test runner executable and run it:

```bash
chmod +x education/output/advanced/tree_sum_test_runner.sh
./education/output/advanced/tree_sum_test_runner.sh
```

**Expected output:**
```
=== Testing Generated Bash Scripts ===

--- Testing tree_sum.sh ---
Test 1: Empty tree
[]:0
    Result: PASS

Test 2: Single node
[10,[],[]]:10
    Result: PASS

Test 3: Complex tree
[10,[5,[],[]],[15,[],[]]]:30
    Result: PASS

=== All Tests Complete ===
```

## Verification

**Success criteria:**
- `tree_sum([], 0)` should PASS.
- `tree_sum([10,[],[]], 10)` should PASS.
- `tree_sum([10,[5,[],[]],[15,[],[]]], 30)` should PASS.
- Test runner exit code = 0.
