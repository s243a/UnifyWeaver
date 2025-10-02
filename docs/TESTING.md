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
- Call Graph: 5/5 tests pass
- SCC Detection: 5/5 tests pass
- Pattern Matchers: 3/4 tests pass (1 known issue)
- Linear Recursion: 2/2 tests pass
- Integration: Most tests pass

**Known issues:**
- Tail recursion with accumulator detection needs refinement
- Mutual recursion requires explicit group compilation

**Verify generated code:**
```bash
ls -lh output/advanced/
head -30 output/advanced/list_length.sh
bash output/advanced/demo.sh  # If demo script exists
```

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
\`\`\`bash
swipl -q -g "use_module('src/unifyweaver/core/new_feature'), test_new_feature, halt."
\`\`\`

**Expected output:** Description of success

**Verify:**
\`\`\`bash
# Commands to verify functionality
\`\`\`
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

## Future Test Additions

As features are added, tests should be created for:

- [ ] Negation handling
- [ ] Complex built-ins (arithmetic, string operations)
- [ ] Module system integration
- [ ] Incremental compilation
- [ ] Query optimization
- [ ] Parallel execution strategies
- [ ] Error handling and reporting
- [ ] Cross-platform compatibility (WSL, Cygwin, native Linux, macOS)

---

*Last updated: 2025-10-02*
