# UnifyWeaver Testing Guide

This document lists all tests that should be run when making changes to the UnifyWeaver codebase. As features are added, this list will grow to cover new test scenarios.

## Quick Test Suite

Run these tests after any significant change:

```bash
cd /path/to/UnifyWeaver

# 1. Core constraint system
swipl -q -g "use_module('src/unifyweaver/core/constraint_analyzer'), test_constraint_analyzer, halt."

# 2. Stream compiler (non-recursive predicates)
swipl -q -g "use_module('src/unifyweaver/core/stream_compiler'), test_stream_compiler, halt."

# 3. Recursive compiler (transitive closure, tail recursion)
swipl -q -g "use_module('src/unifyweaver/core/recursive_compiler'), test_recursive_compiler, halt."

# 4. Advanced recursion (linear, mutual, tail with accumulators)
swipl -q -g "use_module('src/unifyweaver/core/advanced/test_advanced'), test_all_advanced, halt."

# 5. Constraint integration tests
swipl -q -g "use_module('src/unifyweaver/core/test_constraints'), test_constraints, halt."

# 6. Generate and run inferred test runner (validates generated bash scripts)
swipl -g "use_module('src/unifyweaver/core/advanced/test_runner_inference'), \
         generate_test_runner_inferred('output/advanced/inferred_test_runner.sh'), halt."
bash output/advanced/inferred_test_runner.sh
```

## Test Categories

### 1. Constraint System Tests

**Module:** `src/unifyweaver/core/constraint_analyzer.pl`

**What it tests:**
- Default constraints (unique=true, unordered=true)
- Constraint declarations (pragma and programmatic)
- Deduplication strategy selection (sort -u, hash, none)
- Runtime option overrides
- Global default changes

**Run:**
```bash
swipl -q -g "use_module('src/unifyweaver/core/constraint_analyzer'), test_constraint_analyzer, halt."
```

**Expected output:** All 6 tests pass with ✓ marks

---


### 2. Stream Compiler Tests

**Module:** `src/unifyweaver/core/stream_compiler.pl`

**What it tests:**
- Facts compilation (associative arrays)
- Single rule compilation (streaming pipelines)
- Multiple rules (OR patterns)
- Inequality constraints (X \= Y)
- Constraint-aware code generation
- Generated bash script correctness

**Run:**
```bash
swipl -q -g "use_module('src/unifyweaver/core/stream_compiler'), test_stream_compiler, halt."
```

**Expected output:**
- Compiles parent/2, grandparent/2, sibling/2, related/2
- Generates files in `output/` directory
- Shows constraint information for each predicate

**Verify generated code:**
```bash
bash output/test.sh  # Run the generated test script
```


---


### 3. Recursive Compiler Tests

**Module:** `src/unifyweaver/core/recursive_compiler.pl`

**What it tests:**
- Transitive closure detection and compilation
- Tail recursion detection and compilation
- BFS-based graph traversal code generation
- Integration with stream compiler for base predicates

**Run:**
```bash
swipl -q -g "use_module('src/unifyweaver/core/recursive_compiler'), test_recursive_compiler, halt."
```

**Expected output:**
- Classifies ancestor/2 as transitive_closure(parent)
- Classifies descendant/2 as tail_recursion
- Classifies reachable/2 as transitive_closure(related)
- Generates bash scripts with BFS logic

**Verify generated code:**
```bash
bash output/test_recursive.sh  # Run the generated test script
```


---


### 4. Advanced Recursion Tests

**Module:** `src/unifyweaver/core/advanced/test_advanced.pl`

**What it tests:**

#### 4.1 Call Graph Analysis
- Non-recursive predicate detection
- Self-recursive predicate detection
- Mutual recursion detection
- Call graph building
- Predicate group identification

#### 4.2 SCC Detection (Tarjan's Algorithm)
- Simple two-node cycles
- Acyclic graphs
- Self-loops
- Complex graphs with multiple SCCs
- Topological ordering

#### 4.3 Pattern Matching
- Tail recursion with accumulators
- Linear recursion patterns
- Recursive call counting
- Accumulator position detection

#### 4.4 Tail Recursion Compilation
- Count-style predicates
- Sum-style predicates
- Accumulator-based bash code generation

#### 4.5 Linear Recursion Compilation
- List length calculation
- Factorial calculation
- Memoization-based bash code generation

#### 4.6 Mutual Recursion Compilation
- Even/odd style mutual recursion
- Multi-predicate group compilation

**Run:**
```bash
swipl -q -g "use_module('src/unifyweaver/core/advanced/test_advanced'), test_all_advanced, halt."
```

**Expected output:**
- Call Graph: 5/5 tests pass ✅
- SCC Detection: 5/5 tests pass ✅
- Pattern Matchers: 4/4 tests pass ✅
- Tail Recursion: 2/2 tests pass ✅
- Linear Recursion: 2/2 tests pass ✅
- Mutual Recursion: 2/2 tests pass ✅
- Integration: 4/4 tests pass ✅

**Total: 24/24 tests passing (100%)**

**Verify generated code:**
```bash
ls -lh output/advanced/
head -30 output/advanced/list_length.sh
bash output/advanced/test_runner.sh  # Run all generated scripts
```


---


### 4.7 Automatic Test Runner Generation

**Module:** `src/unifyweaver/core/advanced/test_runner_inference.pl`

**What it does:**
Automatically generates comprehensive test runners by analyzing compiled bash scripts - no manual configuration required!

**Features:**
- **Automatic Discovery:** Scans `output/advanced/` for all `.sh` files
- **Script Classification:** Distinguishes between function libraries, demos, and test wrappers
- **Function Extraction:** Finds all callable functions in each script
- **Multi-Function Support:** Handles mutual recursion (multiple functions per file)
- **Smart Deduplication:** Skips test_* wrapper scripts when base script exists
- **Intelligent Inference:** Generates appropriate test cases based on function arity and pattern type

**How to use:**
```bash
# Generate inferred test runner
swipl -g "use_module('src/unifyweaver/core/advanced/test_runner_inference'), \
         generate_test_runner_inferred('output/advanced/inferred_test_runner.sh'), \
         halt."

# Run the generated test runner
bash output/advanced/inferred_test_runner.sh
```

**Generated Output Modes:**

1. **Explicit Mode** (default)
   - One test case per line
   - Easy to read and debug
   - Clear error messages with line numbers

2. **Concise Mode**
   - Compact test definitions
   - Loops for multiple test cases
   - Smaller file size

3. **Hybrid Mode**
   - Mix of explicit and concise styles
   - Balances readability and size

**Example Generated Test:**
```bash
# Test count_items.sh (multi-function: 2 functions)
if [[ -f count_items.sh ]]; then
    echo "--- Testing count_items.sh ---"
    source count_items.sh

    # Function: count_items_eval
    echo "Test 1: Generic test"
    count_items_eval

    # Function: count_items
    echo "Test 1: Empty list with accumulator 0"
    count_items "[]" "0" ""

    echo "Test 2: List with elements"
    count_items "[a,b,c]" "0" ""
fi
```

**What gets tested:**
- `count_items.sh` - Tail recursive accumulator pattern
- `sum_list.sh` - Tail recursive with arithmetic
- `list_length.sh` - Linear recursion with memoization
- `factorial.sh` - Linear recursion
- `even_odd.sh` - Mutual recursion (is_even + is_odd)

**What gets skipped:**
- `demo.sh` - Demo scripts (inline execution, no functions)
- `test_*.sh` - Test wrappers (duplicates of base scripts)
- Helper functions (e.g., `*_stream`, `*_memo`)

**Implementation Details:**

Three-phase implementation:

**Phase 1: Classification & Deduplication** (Commit 0fbf0a0)
```prolog
% Classify scripts by type
classify_script_type(Content, Type) :-
    % Returns: function_library, demo, test_wrapper, or standalone

% Skip duplicates
is_test_wrapper_duplicate(FileName) :-
    atom_concat('test_', _BaseName, FileName).
```

**Phase 2: Multi-Function Extraction** (Commit ebac072)
```prolog
% Extract ALL functions from a script
extract_all_functions(Content, Functions) :-
    re_foldl(collect_function, "^(\w+)\(\)\s*\{", Content, [], Functions, [...]).

% Get arity for each function
extract_function_arity(Content, FuncName, Arity) :-
    % Count "local var=\"$N\"" patterns in function body
```

**Phase 3: File Grouping** (Commit c1c5e64)
```prolog
% Group test configs by source file
group_configs_by_file(Configs, GroupedConfigs) :-
    % Source each file only once, test all its functions

% Write tests for multi-function files
write_file_tests(Stream, FilePath, FunctionConfigs) :-
    format(Stream, '# Test ~w (multi-function: ~w functions)~n', [...])
```

**Advantages over manual configuration:**
- ✅ Zero maintenance - adapts to new scripts automatically
- ✅ No typos in function names - extracted from actual code
- ✅ Correct arity detection - parsed from bash parameter usage
- ✅ Proper multi-function handling - mutual recursion works correctly
- ✅ Clean output - intelligent deduplication and grouping

**When to use:**
- **Inference-based:** Quick iteration, exploratory development, changing codebase
- **Config-based:** Precise control, complex test cases, stable API

**Troubleshooting:**
- If a script isn't tested: Check if it's classified as `function_library`
- If wrong function called: Verify header comment matches actual function name
- If arity is wrong: Check for `local var=\"$N\"` patterns in function body

---


### 5. Constraint Integration Tests

**Module:** `src/unifyweaver/core/test_constraints.pl`

**What it tests:**
- Default constraint behavior in compilation
- Ordered constraint usage (hash-based dedup)
- Non-unique constraint usage (no dedup)
- Runtime option override of declared constraints
- Proper bash code generation for each strategy

**Run:**
```bash
swipl -q -g "use_module('src/unifyweaver/core/test_constraints'), test_constraints, halt."
```

**Expected output:** All 4 integration tests pass with ✓ marks

**Constraint Support in Advanced Compilers:**
- All advanced recursion compilers (tail, linear, mutual) now query constraints via `get_constraints/2`.
- The `tail_recursion` compiler now actively uses the `unique(true)` constraint to generate optimized code with an early exit.
- For other advanced compilers, constraints are logged but not yet acted upon. This is reserved for future enhancements.
- See [ADVANCED_RECURSION.md](ADVANCED_RECURSION.md#options-and-constraints) for details.

---


### 6. Constraint Demo (Visual Verification)

**Module:** `examples/constraints_demo.pl`

**What it demonstrates:**
- Default constraints in action
- Temporal/ordered data handling
- Path finding with duplicates
- Visual comparison of generated code

**Run:**
```bash
swipl -q -g "['examples/constraints_demo.pl'], demo, halt."
```

**Verify generated files:**
```bash
ls -lh output/demo_*.sh
head -20 output/demo_default.sh   # Should show: sort -u
head -20 output/demo_ordered.sh   # Should show: declare -A seen
head -20 output/demo_no_dedup.sh  # Should show: no deduplication
```

---


## Testing Checklist

After making changes, verify:

- [ ] All constraint analyzer tests pass
- [ ] All stream compiler tests pass
- [ ] All recursive compiler tests pass
- [ ] All advanced recursion tests pass
- [ ] All integration tests pass
- [ ] No new compiler warnings
- [ ] Generated bash scripts are valid (syntax check with `bash -n`)
- [ ] Generated bash scripts produce correct output
- [ ] Constraint comments appear in generated code
- [ ] No regressions in existing functionality

## Test Organization

### Unit Tests
Individual module functionality:
- `constraint_analyzer.pl:test_constraint_analyzer/0`
- `call_graph.pl:test_call_graph/0`
- `scc_detection.pl:test_scc_detection/0`
- `pattern_matchers.pl:test_pattern_matchers/0`
- `tail_recursion.pl:test_tail_recursion/0`
- `linear_recursion.pl:test_linear_recursion/0`
- `mutual_recursion.pl:test_mutual_recursion/0`

### Integration Tests
Cross-module functionality:
- `stream_compiler.pl:test_stream_compiler/0`
- `recursive_compiler.pl:test_recursive_compiler/0`
- `advanced_recursive_compiler.pl:test_advanced_compiler/0`
- `test_constraints.pl:test_constraints/0`
- `test_advanced.pl:test_all_advanced/0`

### Regression Tests
Ensure no breaking changes:
- All existing examples still compile
- Generated bash scripts still run correctly
- Performance hasn't degraded

---


## Adding New Tests

When adding a new feature, update this document with:

1. **New test module** - Add to appropriate category
2. **Run command** - Show how to execute the test
3. **Expected output** - Describe what success looks like
4. **Verification steps** - How to verify generated code
5. **Known issues** - Document any expected failures

### Example Template

```markdown
### X. New Feature Tests

**Module:** `src/unifyweaver/core/new_feature.pl`

**What it tests:**
- Feature aspect 1
- Feature aspect 2

**Run:**
```bash
swipl -q -g "use_module('src/unifyweaver/core/new_feature'), test_new_feature, halt."
```

**Expected output:** Description of success

**Verify:**
```bash
# Commands to verify functionality
```
```

---


## Test Environment (test_env)

UnifyWeaver provides test environment setup scripts that create standalone test directories with auto-discovery:

### Setup

**Bash/Linux:**
```bash
cd scripts/testing
./init_testing.sh           # Creates test_env/ in scripts/testing/
./init_testing.sh -d /tmp   # Creates /tmp/test_env/
```

**PowerShell/Windows:**
```powershell
cd scripts\testing
.\Init-TestEnvironment.ps1  # Creates test_env/ in scripts\testing\
```

### Test Environment Features

**Hybrid Test Discovery:**
- **Manual tests** (hardcoded): `test_stream`, `test_recursive`, `test_advanced`, `test_constraints` - always available, guaranteed to work
- **Auto-discovery**: Automatically finds `test_*.pl` files in `src/unifyweaver/core/` and `src/unifyweaver/core/advanced/`
- **Fallback safe**: If auto-discovery fails, manual tests still work

**Available Commands:**

```prolog
% Core loaders
?- load_stream.         # Load stream compiler
?- load_recursive.      # Load recursive compiler
?- load_all_core.       # Load all core modules

% Manual tests (reliable fallback)
?- test_stream.         # Test stream compiler
?- test_recursive.      # Test recursive compiler
?- test_advanced.       # Test advanced recursion (24 tests)
?- test_constraints.    # Test constraint system

% Auto-discovered tests (if available)
?- test_auto.           # Run all auto-discovered tests

% Run everything
?- test_all.            # Run ALL tests (manual + auto)

% Help
?- help.                # Show all available commands
```

### test_all Command

The `test_all` command runs all tests in sequence:

```prolog
?- test_all.

╔════════════════════════════════════════╗
║  Running All UnifyWeaver Tests        ║
╚════════════════════════════════════════╝

═══ Manual Tests (Core) ═══

┌─ Stream Compiler ────────────────────┐
=== STREAM COMPILER TESTS ===
... tests run ...
└─ Stream Compiler Complete ──────────┘

┌─ Recursive Compiler ─────────────────┐
... tests run ...
└─ Recursive Compiler Complete ───────┘

┌─ Advanced Recursion ─────────────────┐
... 24 tests run ...
└─ Advanced Recursion Complete ───────┘

┌─ Constraint System ──────────────────┐
... tests run ...
└─ Constraint System Complete ────────┘

═══ Auto-Discovered Tests ═══
(runs any additional test modules found)

╔════════════════════════════════════════╗
║  All Tests Complete                    ║
╚════════════════════════════════════════╝
```

### Adding New Tests

**No configuration needed!** Just create your test file:

1. Create `src/unifyweaver/core/test_myfeature.pl`
2. Export `test_myfeature/0` predicate
3. Regenerate test environment or copy to existing one
4. New test is automatically discovered

Example:
```prolog
:- module(test_myfeature, [test_myfeature/0]).

test_myfeature :-
    writeln('=== MY FEATURE TESTS ==='),
    % ... your tests ...
    writeln('=== TESTS COMPLETE ===').
```

Then in test environment:
```prolog
?- help.
% Shows: test_myfeature.  - Test myfeature module

?- test_myfeature.
% Runs your tests

?- test_all.
% Includes your tests automatically
```

---


## Continuous Integration Notes

For CI/CD integration, all tests can be run in sequence:

```bash
#!/bin/bash
# run_all_tests.sh

set -e  # Exit on first failure

echo "Running UnifyWeaver test suite..."

# Core tests
echo "1. Constraint analyzer..."
swipl -q -g "use_module('src/unifyweaver/core/constraint_analyzer'), test_constraint_analyzer, halt."

echo "2. Stream compiler..."
swipl -q -g "use_module('src/unifyweaver/core/stream_compiler'), test_stream_compiler, halt."

echo "3. Recursive compiler..."
swipl -q -g "use_module('src/unifyweaver/core/recursive_compiler'), test_recursive_compiler, halt."

echo "4. Advanced recursion..."
swipl -q -g "use_module('src/unifyweaver/core/advanced/test_advanced'), test_all_advanced, halt."

echo "5. Constraint integration..."
swipl -q -g "use_module('src/unifyweaver/core/test_constraints'), test_constraints, halt."

echo ""
echo "✓ All tests passed!"
```

---


## Test Coverage Goals

- **Unit tests:** Cover all public predicates in each module
- **Integration tests:** Cover all compilation paths
- **Regression tests:** Cover all examples in `examples/` directory
- **Performance tests:** Ensure compilation time remains reasonable
- **Output validation:** Generated bash scripts are syntactically correct and produce expected output

---

## 5. C# Query Runtime Tests

See the dedicated plan at [`docs/development/testing/v0_1_csharp_test_plan.md`](development/testing/v0_1_csharp_test_plan.md) for full details.

### 5.1 Automated regression (skip dotnet execution)
```bash
SKIP_CSHARP_EXECUTION=1 \
swipl -q \
     -f tests/core/test_csharp_query_target.pl \
     -g test_csharp_query_target:test_csharp_query_target \
     -t halt
```

### 5.2 Optional full-suite check
```bash
SKIP_CSHARP_EXECUTION=1 swipl -q -f run_all_tests.pl -g main -t halt
```

### 5.3 Manual runtime validation (optional)
Follow the “Manual Runtime Validation” section in the plan to:
1. Generate a C# project under `output/csharp/<uuid>/`.
2. Run `dotnet build --no-restore`.
3. Execute the compiled binary or DLL (`dotnet bin/Debug/net9.0/<module>.dll` or the self-contained executable) and verify outputs (`alice,charlie`, `0`, `2`, `4`).

---


## Troubleshooting Tests

### Common Issues

**Module not found:**
```prolog
ERROR: source_sink `...` does not exist
```
**Solution:** Ensure you're running from the UnifyWeaver root directory

**Test predicates not exported:**
```prolog
ERROR: Unknown procedure: test_foo/0
```
**Solution:** Check module declaration exports the test predicate

**Output directory missing:**
```prolog
ERROR: existence_error(directory,output)
```
**Solution:** Create directory: `mkdir -p output/advanced`

**Bash script errors:**
```bash
bash: output/test.sh: No such file or directory
```
**Solution:** Run the Prolog test first to generate the files

---


## 6. Go Target Tests

See [`docs/development/testing/v0_2_go_target_test_plan.md`](development/testing/v0_2_go_target_test_plan.md) for the full plan.

### Quick Test
```bash
swipl -g "use_module('tests/core/test_go_generator'), run_tests" -t halt
```

---

## 7. Python Target Tests

See [`docs/development/testing/v0_2_python_target_test_plan.md`](development/testing/v0_2_python_target_test_plan.md) for the full plan.

### Quick Test
```bash
swipl -g "[tests/core/test_python_generator], run_tests(python_generator)" -t halt
```

---

## 8. Rust Target Tests

See [`docs/development/testing/v0_2_rust_target_test_plan.md`](development/testing/v0_2_rust_target_test_plan.md) for the full plan.

### Quick Test
```bash
swipl -g run_tests -t halt tests/test_rust_target.pl
```

---

## 9. Cross-Target Glue Tests

See [`docs/development/testing/v0_2_cross_target_glue_test_plan.md`](development/testing/v0_2_cross_target_glue_test_plan.md) for the full plan.

### Quick Test
```bash
# Shell glue
swipl -g "use_module('tests/integration/glue/test_shell_glue'), run_tests" -t halt

# .NET glue
swipl -g "use_module('tests/integration/glue/test_dotnet_glue'), run_tests" -t halt
```

---

## Future Test Additions

As features are added, tests should be created for:

- [ ] Negation handling (partial coverage in Go/Python targets)
- [ ] Complex built-ins (arithmetic, string operations)
- [ ] Module system integration
- [ ] Incremental compilation
- [ ] Query optimization
- [ ] Parallel execution strategies
- [ ] Error handling and reporting
- [x] Cross-platform compatibility (WSL, Cygwin, native Linux, macOS)
- [x] Go target code generation
- [x] Python target code generation
- [x] Rust target code generation
- [x] Cross-target glue integration

---

*Last updated: 2025-12-08*
