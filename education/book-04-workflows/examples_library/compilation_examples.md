---
file_type: UnifyWeaver Example Library
spec_version: 1.0
---

# Prolog Compilation Examples

This file contains examples of compiling Prolog predicates to bash using UnifyWeaver's compiler_driver.

---

### Factorial Compilation

> [!example-record]
> id: 20251108-factorial-compile
> name: unifyweaver.compilation.factorial
> pattern: linear_recursion
> child_examples: [unifyweaver.testing.factorial_runner, unifyweaver.workflow.factorial_complete]

This example demonstrates compiling a linear recursive predicate (factorial) to bash.

**Prolog Source:**
```prolog
% factorial(N, F) - F is the factorial of N
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
```

**Save to File:**
```bash
cat > /tmp/factorial.pl <<'EOF'
% factorial(N, F) - F is the factorial of N
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
EOF
```

**Compilation Command:**

From project root (`$UNIFYWEAVER_HOME`):

```bash
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/factorial.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [], Scripts),
    format('Generated scripts: ~w~n', [Scripts]),
    halt"
```

**Expected Output:**
```
Generated scripts: [education/output/advanced/factorial.sh]
```

**Generated Bash Script Location:**
`education/output/advanced/factorial.sh`

**What the Compiler Does:**
- Analyzes the recursive pattern (linear recursion with base case)
- Translates arithmetic operations (`is`, `>`) to bash
- Generates both a standalone function and stream-compatible version
- Creates optimized bash using `$((...))` for arithmetic

**Next Steps:**
- See [unifyweaver.testing.factorial_runner](testing_examples.md#factorial-test-runner) for testing
- See [unifyweaver.workflow.factorial_complete](#factorial-complete-workflow) for full workflow

---

### Factorial Complete Workflow

> [!example-record]
> id: 20251108-factorial-workflow
> name: unifyweaver.workflow.factorial_complete
> pattern: compile_and_test
> parent_example: unifyweaver.compilation.factorial

**Minimal Playbook: Complete factorial generation, compilation, and testing**

**Execution Steps:**
1. Generate Prolog code (see example: `unifyweaver.compilation.factorial`)
2. Save to `/tmp/factorial.pl`
3. Transpile using compiler_driver
4. Generate test runner using test_runner_inference (see: `unifyweaver.testing.factorial_runner`)
5. Execute and verify

**Full Command Sequence:**

```bash
# Navigate to project root
cd $UNIFYWEAVER_HOME

# Step 1-2: Create Prolog file
cat > /tmp/factorial.pl <<'EOF'
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
EOF

# Step 3: Compile
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/factorial.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(factorial/2, [], Scripts),
    format('Generated: ~w~n', [Scripts]),
    halt"

# Step 4: Generate test runner
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    use_module(unifyweaver(core/advanced/test_runner_inference)),
    generate_test_runner_inferred('education/output/advanced/test_runner.sh', [
        mode(explicit),
        output_dir('education/output/advanced')
    ]),
    halt"

# Step 5: Execute tests
chmod +x education/output/advanced/test_runner.sh
./education/output/advanced/test_runner.sh
```

**Expected Test Output:**
```
Testing factorial.sh...
Test 1: factorial 0 → 1
    Result: PASS
Test 2: factorial 5 → 120
    Result: PASS
```

**Verification:**
- All tests should PASS
- Output values should match expected factorials
- Exit code should be 0

---

### Ancestor Compilation

> [!example-record]
> id: 20251108-ancestor-compile
> name: unifyweaver.compilation.ancestor
> pattern: transitive_closure
> related: [unifyweaver.recursion.ancestor from recursion_examples.md]

This example shows compiling a transitive closure (ancestor relationship) which UnifyWeaver optimizes to breadth-first search.

**Prolog Source:**
```prolog
% Base facts
parent(alice, bob).
parent(bob, charlie).
parent(charlie, diana).

% Recursive rules
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

**Compilation:**

```bash
cat > /tmp/ancestor.pl <<'EOF'
parent(alice, bob).
parent(bob, charlie).
parent(charlie, diana).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
EOF

cd $UNIFYWEAVER_HOME
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    ['/tmp/ancestor.pl'],
    use_module(unifyweaver(core/compiler_driver)),
    compile(ancestor/2, [], Scripts),
    format('Generated: ~w~n', [Scripts]),
    halt"
```

**What Makes This Special:**
- UnifyWeaver recognizes the transitive closure pattern
- Generates optimized breadth-first search in bash
- Handles dependency on `parent/2` automatically
- Creates efficient hash-based lookups

**Generated Files:**
- `education/output/advanced/parent.sh` (dependency)
- `education/output/advanced/ancestor.sh` (main predicate)
