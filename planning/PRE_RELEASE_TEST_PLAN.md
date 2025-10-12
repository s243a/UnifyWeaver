# Pre-Release Test Plan

**Version:** For any release (alpha, beta, or stable)
**Last Updated:** 2025-10-12
**Estimated Time:** 10-15 minutes (full suite), 3-5 minutes (quick suite)

---

## Quick Test Suite (3-5 minutes)

Run this minimal suite before any release:

### 1. Core Tests
```bash
cd scripts/testing
./init_testing.sh

# In SWI-Prolog:
?- test_all.
?- halt.
```

**Expected:** All tests pass (stream, recursive, advanced, constraints)

### 2. Pattern Matchers (forbid system)
```prolog
cd scripts/testing
./init_testing.sh

# In SWI-Prolog:
?- use_module('src/unifyweaver/core/advanced/pattern_matchers').
?- test_pattern_matchers.
?- halt.
```

**Expected:** All 5 tests pass, including forbid/unforbid cycle

### 3. Smoke Test Generated Script
```bash
cd output/advanced
source fibonacci.sh
fibonacci 10
```

**Expected:** Output `10:55` (or similar - verifies bash generation works)

---

## Comprehensive Test Suite (10-15 minutes)

Run before major releases or when significant changes made:

### Test 1: Core Prolog Unit Tests (~2 minutes)

**Location:** Test environment
**Command:**
```bash
cd scripts/testing
./init_testing.sh

# In SWI-Prolog:
?- test_all.
```

**Expected Results:**
- ✅ Stream compilation tests pass
- ✅ Basic recursion tests pass
- ✅ Advanced recursion tests pass (24 tests)
- ✅ Constraint system tests pass (6 tests)
- ✅ All pattern detection tests pass (6/6)

**Fixed in recent releases:**
- ✅ `list_length/2` - Now correctly detected as linear recursion (fixed in fold-based PR)
- ✅ `descendant/2` - Now correctly classified as transitive closure (fixed in fold-based PR)
- ✅ `factorial/2` - Now compiles with working fold-based code generation
- ✅ Fibonacci exclusion - Now correctly excluded from linear recursion (2+ calls)

**If fails:** Check error messages, verify module paths, check for syntax errors

---

### Test 2: Advanced Recursion Detailed (~1 minute)

**Location:** Test environment
**Command:**
```prolog
?- test_advanced.
```

**Expected Results:**
- ✅ Tail recursion: `count_items/3`, `sum_list/3` compile and test
- ✅ Linear recursion: `fibonacci/2`, `factorial/2` compile with memoization
- ✅ Tree recursion: `tree_sum/2` compiles with structure parsing
- ✅ Mutual recursion: `is_even/1`, `is_odd/1` compile with shared memo
- ✅ Pattern detection works correctly

**Specific Checks:**
```prolog
% Verify fibonacci does NOT match linear recursion (has 2 recursive calls)
?- use_module('src/unifyweaver/core/advanced/pattern_matchers').
?- \+ is_linear_recursive_streamable(fibonacci/2).
% Should succeed (fibonacci has 2 calls, not linear)

% Verify tree_sum uses tree recursion
?- is_tree_recursive(tree_sum/2).
% Should succeed
```

---

### Test 3: Pattern Matcher Tests (~30 seconds)

**Location:** Test environment
**Command:**
```prolog
?- use_module('src/unifyweaver/core/advanced/pattern_matchers').
?- test_pattern_matchers.
```

**Expected Results:**

**Test 1: Tail recursion detection**
- ✅ `count_items/3` detected as tail recursive

**Test 2: Linear recursion detection**
- ✅ `factorial/2` detected as linear recursive

**Test 3: Mutual recursion detection**
- ✅ `is_even/1` and `is_odd/1` in same SCC

**Test 4: Tree recursion detection**
- ✅ `tree_sum/2` detected as tree recursive

**Test 5: Forbid linear recursion system**
- ✅ `test_fib/2` matches linear pattern initially
- ✅ After `forbid_linear_recursion(test_fib/2)`, doesn't match
- ✅ `is_forbidden_linear_recursion/1` returns true
- ✅ After `clear_linear_recursion_forbid/1`, matches again

---

### Test 4: Generated Bash Scripts (~2 minutes)

**Location:** `output/advanced/`
**Command:**
```bash
cd output/advanced
bash test_runner.sh
```

**Expected Results:**
- ✅ All bash scripts execute without syntax errors
- ✅ Functions are callable
- ✅ Results match expected values
- ✅ No permission errors or missing files

**Manual Spot Checks:**
```bash
# Test fibonacci
source fibonacci.sh
fibonacci 5    # Should output: 5:5
fibonacci 10   # Should output: 10:55

# Test factorial
source factorial.sh
factorial 5    # Should output: 5:120

# Test tree_sum
source tree_sum.sh
tree_sum "[10,[5,[],[3,[],[]]],[7,[],[]]]"  # Should output: ...25

# Test even/odd
source even_odd.sh
is_even 4 && echo "YES" || echo "NO"  # Should output: YES
is_odd 4 && echo "YES" || echo "NO"   # Should output: NO
```

---

### Test 5: Educational Materials Workflow (~3 minutes)

**Location:** `education/`
**Purpose:** Verify Chapter 4 tutorial workflow

**Commands:**
```bash
cd education
swipl
```

```prolog
% 1. Initialize environment
?- ['init'].
% Expected: "Educational environment initialized"

% 2. Load compilers
?- use_module(unifyweaver(core/stream_compiler)).
?- use_module(unifyweaver(core/recursive_compiler)).

% 3. Load family tree
?- ['family_tree'].

% 4. Compile parent facts
?- stream_compiler:compile_facts(parent, 2, [], BashCode),
   open('output/advanced/parent.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).
% Expected: true

% 5. Compile ancestor rule
?- compile_recursive(ancestor/2, [], BashCode),
   open('output/advanced/ancestor.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).
% Expected: true

% 6. Generate test runner with custom output_dir
?- use_module('src/unifyweaver/core/advanced/test_runner_inference').
?- generate_test_runner_inferred('output/advanced/test_runner.sh', [output_dir('output/advanced')]).
% Expected: true

?- halt.
```

**Run generated scripts:**
```bash
# Test parent
source education/output/advanced/parent.sh
parent abraham  # Should list: ishmael, isaac

# Test ancestor
source education/output/advanced/ancestor.sh
ancestor abraham  # Should list: ishmael, isaac, esau, jacob, ...

# Run test runner
bash education/output/advanced/test_runner.sh
# Expected: Tests execute successfully
```

---

### Test 6: Constraint System Integration (~1 minute)

**Location:** Test environment
**Command:**
```prolog
?- use_module('src/unifyweaver/core/constraint_analyzer').
?- test_constraint_analyzer.
```

**Expected Results:**
- ✅ 6/6 constraint tests pass
- ✅ Pragma parsing works
- ✅ Programmatic constraints work
- ✅ Defaults apply correctly
- ✅ Override behavior correct

---

### Test 7: Control Plane (Firewall + Preferences) (~1 minute)

**Location:** Test environment
**Command:**
```prolog
?- use_module('src/unifyweaver/core/firewall').
?- use_module('src/unifyweaver/core/preferences').
```

**Manual Checks:**
```prolog
% Test firewall
?- firewall:is_backend_allowed(bash).
% Expected: true

?- firewall:is_backend_allowed(evil_backend).
% Expected: false (if not in whitelist)

% Test preferences
?- preferences:get_preference(default_backend, Backend).
% Expected: Backend = bash (or configured value)
```

---

## Test Failure Triage

### If Core Tests Fail:
1. Check SWI-Prolog version (need 8.0+)
2. Verify library paths are correct
3. Check for module import errors
4. Review error messages for syntax issues

### If Advanced Tests Fail:
1. Verify `output/advanced/` directory exists
2. Check file permissions
3. Ensure Bash 4.0+ (need associative arrays)
4. Review generated .sh files for syntax errors

### If Pattern Matcher Tests Fail:
1. Check dynamic predicate assertions work
2. Verify `forbid_linear_recursion/1` system loaded
3. Test with fresh Prolog session
4. Check for module conflicts

### If Bash Script Tests Fail:
1. Check bash version: `bash --version` (need 4.0+)
2. Verify scripts have executable permissions
3. Check for missing dependencies
4. Source scripts in correct order (dependencies first)

### If Education Tests Fail:
1. Verify `education/init.pl` loads correctly
2. Check library alias setup
3. Ensure `output/advanced/` directory exists
4. Verify module paths relative to education/

---

## GitHub Actions CI/CD

**Automated tests run on:**
- Push to main
- Pull requests
- Manual workflow dispatch

**View results:**
```bash
# Check latest CI run
gh run list --limit 5

# View specific run
gh run view <run-id>
```

---

## Pre-Release Checklist

Before tagging a release:

- [ ] All Quick Test Suite tests pass
- [ ] (Optional) Comprehensive Test Suite passes
- [ ] No uncommitted changes: `git status`
- [ ] CHANGELOG.md updated with version
- [ ] VERSION file updated
- [ ] README.md reflects current features
- [ ] All documentation up to date
- [ ] GitHub Actions CI passing

---

## Post-Release Verification

After creating a release tag:

```bash
# 1. Clone fresh copy
git clone https://github.com/s243a/UnifyWeaver.git test-release
cd test-release
git checkout v0.0.1-alpha  # or specific tag

# 2. Run quick test suite
cd scripts/testing
./init_testing.sh
# In SWI-Prolog: test_all.

# 3. Verify one example works
cd ../../output/advanced
bash test_runner.sh
```

**Expected:** Everything works from clean clone

---

## Notes

- **Test Environment:** Uses `scripts/testing/init_testing.sh` for automatic setup
- **Platform:** Tests should pass on Linux, WSL, and macOS (Windows may need adjustments)
- **Bash Version:** Requires Bash 4.0+ for associative arrays
- **SWI-Prolog:** Requires 8.0+ for module system features

---

## Updating This Document

When adding new features:
1. Add corresponding test to appropriate suite
2. Document expected results
3. Update triage section if new failure modes possible
4. Update checklist if new requirements
