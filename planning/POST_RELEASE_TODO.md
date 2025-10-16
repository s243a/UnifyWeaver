# Post-Release TODO List

**Created:** 2025-10-12
**Target Release:** v0.0.2 or later

This tracks work to be done after v0.0.1-alpha release.

---

## Priority 1: Known Limitations (Fix These First)

### 1. Fix `list_length/2` Linear Recursion Detection

**Status:** ❌ FAILING TEST
**Location:** `test_advanced.pl` - Linear Recursion Compiler, Test 1
**Current Behavior:** Pattern matcher fails to detect `list_length/2` as linear recursion

**Predicate:**
```prolog
list_length([], 0).
list_length([_|T], N) :-
    list_length(T, N1),
    N is N1 + 1.
```

**Issue:** The linear recursion pattern matcher doesn't recognize this pattern, even though:
- It has exactly 1 recursive call per clause
- The recursive call is independent (no data flow between calls)
- It computes a scalar result via arithmetic

**Why It Fails:**
- Pattern detection in `pattern_matchers.pl:is_linear_recursive_streamable/1` may be too strict
- Possible issue with how arithmetic operations are analyzed
- May need to relax independence checks for post-computation patterns

**Fix Strategy:**
1. Debug pattern matcher with `list_length/2` specifically
2. Compare with working `factorial/2` pattern detection
3. Adjust independence or pattern matching rules
4. Add regression test to ensure fix doesn't break other patterns

**Estimated Effort:** 2-3 hours

---

### 2. Fix `descendant/2` Advanced Pattern Detection

**Status:** ✗ No advanced pattern matched (falls back to basic recursion)
**Location:** `test_recursive.pl` - Recursive Predicates Test
**Current Behavior:** Classified as `tail_recursion` but fails all advanced pattern matchers

**Predicate:**
```prolog
descendant(X, Y) :- parent(X, Y).
descendant(X, Z) :- parent(X, Y), descendant(Y, Z).
```

**Issue:** This is a classic transitive closure pattern but in reverse order:
- `ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)` ✅ Works (forward chaining)
- `descendant(X, Z) :- parent(X, Y), descendant(Y, Z)` ❌ Fails (same structure!)

**Current Classification:** `tail_recursion` (incorrect)

**Why It Fails:**
- Tail recursion detector: Not actually tail recursive (recursive call in body, not tail position)
- Linear recursion detector: Multiple calls (base case + recursive case)
- Tree recursion detector: Not a tree structure pattern
- Mutual recursion detector: Only calls itself, not mutual

**The Real Pattern:** This is a **basic recursion** case that should use BFS optimization (transitive closure), but the classifier marks it as `tail_recursion` incorrectly.

**Fix Strategy:**
1. Improve transitive closure detection in `recursive_compiler.pl`
2. Check if predicate matches `P(X, Z) :- Q(X, Y), P(Y, Z)` pattern
3. Mark as transitive closure even when Q is different from P
4. Apply BFS optimization for this pattern
5. Consider it a variant of ancestor/descendant symmetry

**Estimated Effort:** 3-4 hours

---

## Priority 2: Bash Code Generation Completion

### 3. Complete Linear Recursion Bash Generation

**Status:** ⚠️ Scaffold generated, logic incomplete
**Location:** `src/unifyweaver/core/advanced/linear_recursion.pl`

**Current Behavior:**
- Pattern detection works correctly
- Bash scaffolding with memoization structure is generated
- Actual recursive logic has TODO placeholders

**Example:** `factorial.sh` generates but doesn't compute results
- Expected: `factorial("5", "")` → `5:120`
- Actual: No output (base case logic incomplete)

**Fix Strategy:**
1. Implement bash code generation for arithmetic operations in recursive case
2. Handle `N is N1 * F1` style operations
3. Generate proper base case handling for numeric predicates
4. Test with factorial, fibonacci examples

**Estimated Effort:** 4-6 hours

---

### 4. Complete Mutual Recursion Bash Generation

**Status:** ✅ COMPLETE
**Location:** `src/unifyweaver/core/advanced/mutual_recursion.pl`
**Branch:** `feat/mutual-recursion-codegen` (merged to main)

**Completed Work:**
- ✅ Implemented base case code generation from Prolog facts
- ✅ Implemented recursive case generation with variable translation
- ✅ Proper head variable tracking (maps to `$arg1` in bash)
- ✅ Arithmetic expression translation (`N - 1` to bash `$(( $arg1 - 1 ))`)
- ✅ Condition generation (`N > 0` to bash `[[ $arg1 -gt 0 ]]`)
- ✅ Recursive call generation with proper argument passing
- ✅ Shared memoization table correctly expanded in templates
- ✅ All tests passing

**Generated Code Works:**
- `is_even(0)` → true (base case)
- `is_even(2)` → true (via mutual recursion through is_odd)
- `is_odd(3)` → true (via mutual recursion through is_even)
- `is_even(5)` → false (correctly fails)

**Implementation Details:**
- Added `is_mutual_recursive_clause/2` for proper clause classification
- Implemented `contains_goal/2` to search clause bodies
- Created `*_with_var` family of functions for variable translation
- Fixed infinite loop in expression translation by checking var() first
- Handled negative numbers in expressions (N + -1 → N - 1)

**Effort:** 6 hours (completed 2025-10-12)

---

## Priority 3: Code Quality Improvements

### 5. Fix Singleton Variable Warnings

**Status:** ⚠️ Warnings during test runs

**Locations:**
- `tree_recursion.pl:179` - Singleton variables: `MemoCheck`, `MemoStore`
- `tree_recursion.pl:227` - Singleton variables: `Arity`, `Operation`
- `mutual_recursion.pl:46` - Singleton variable: `AllOptions`
- `test_advanced.pl:73` - Singleton variables: `FibCode`, `TreeCode`

**Fix Strategy:**
- Use underscore prefix for intentionally unused variables: `_MemoCheck`
- Or remove unused variables from patterns
- Verify logic is correct (not masking real bugs)

**Estimated Effort:** 30 minutes

---

### 4. Update init_template.pl to Working Pattern

**Status:** ✅ FIXED in templates/, but working pattern documented

**Current Working Pattern:**
```prolog
unifyweaver_init :-
    % ... setup code ...
    help.

:- dynamic unifyweaver_initialized/0.
:- asserta(unifyweaver_initialized).

% ... rest of predicates ...

:- initialization(unifyweaver_init, now).
```

**Note:** This pattern works and is committed. Keeping here for reference.

---

## Priority 3: Documentation Updates

### 5. Update Test Plan with Known Failures

**Location:** `planning/PRE_RELEASE_TEST_PLAN.md`

**Add Section:**
```markdown
## Known Test Failures

### Expected Failures (Do Not Block Release)

1. **list_length/2** - Linear recursion pattern not detected
   - Test: Advanced Recursion → Linear Recursion Compiler → Test 1
   - Workaround: Falls back to basic recursion with memoization
   - Tracked in: POST_RELEASE_TODO.md #1

2. **descendant/2** - Misclassified as tail_recursion, fails all patterns
   - Test: Recursive Compiler → test_recursive
   - Workaround: Falls back to basic recursion (no BFS optimization)
   - Tracked in: POST_RELEASE_TODO.md #2
```

**Estimated Effort:** 15 minutes

---

### 6. Add Known Limitations to README.md

**Location:** `README.md` - Current Limitations section (lines 254-275)

**Add to Limitations:**
```markdown
**Pattern Detection:**
- `list_length/2` pattern not detected by linear recursion matcher (issue #1)
- `descendant/2` misclassified as tail recursion, should be transitive closure (issue #2)
- Some arithmetic post-computation patterns may not be recognized
```

**Estimated Effort:** 10 minutes

---

## Priority 4: Testing Infrastructure

### 7. Add Regression Tests for Fixes

**When fixing #1 and #2 above:**

Add tests to prevent regressions:
```prolog
% In test_advanced.pl or new test_regressions.pl

test_list_length_pattern :-
    % Verify list_length is detected as linear recursion
    is_linear_recursive_streamable(list_length/2),
    writeln('✓ list_length pattern detected').

test_descendant_classification :-
    % Verify descendant gets transitive closure optimization
    classify_recursion(descendant/2, Classification),
    Classification = transitive_closure(_),
    writeln('✓ descendant classified correctly').
```

**Estimated Effort:** 1 hour

---

## Priority 5: Post v0.0.2 Improvements

### 10. Clean Up Singleton Variable and Code Quality Warnings

**Status:** 📋 IDENTIFIED - Code quality cleanup needed
**Location:** Multiple files across core and advanced modules
**Created:** 2025-10-15

**Warnings Identified During Testing:**

**Singleton Variable Warnings:**
- `stream_compiler.pl:130` - Singleton: `[Pred]`
- `linear_recursion.pl:196, 336` - Singleton: `[FoldExpr]`
- `fold_helper_generator.pl:69` - Singleton: `[Arity,RecHead]`
- `fold_helper_generator.pl:116` - Singleton: `[Goal,Body]`
- `fold_helper_generator.pl:532` - Singleton: `[Arity]`
- `advanced_recursive_compiler.pl:195` - Singleton: `[PredStr]`
- `advanced_recursive_compiler.pl:220` - Singleton: `[Arity,Options]`
- `advanced_recursive_compiler.pl:259` - Singleton: `[GraphClauses]`
- `advanced_recursive_compiler.pl:288` - Singleton: `[FoldClauses]`
- `advanced_recursive_compiler.pl:323` - Singleton: `[BasePredStr]`
- `firewall.pl:198` - Singleton: `[P,Ps]`
- `firewall.pl:223` - Singleton: `[M,Ms]`
- `firewall.pl:268` - Singleton: `[D,Ds]`

**Singleton-Marked Variable Warnings:**
- `fold_helper_generator.pl:301` - Variables marked as singleton but used multiple times:
  - `_OutputVar`, `_V`, `_FL`, `_VL`, `_FR`, `_VR`, `_WOutputVar`, `_Graph`

**Code Organization Warnings:**
- `fold_helper_generator.pl:532` - Clauses not together: `generate_fold_computer/3`
- `fold_helper_generator.pl:633` - Clauses not together: `generate_wrapper/2`

**Import Override Warning:**
- `advanced_recursive_compiler.pl:352` - Local definition overrides weak import: `extract_goal/2`

**Impact:**
- No functional issues - all tests pass
- Code quality and maintainability issue
- Could mask real bugs if not addressed

**Fix Strategy:**
1. **Singleton variables:** Prefix with underscore (`_Pred`) or remove if truly unused
2. **Singleton-marked but used:** Remove underscore prefix (these are actual variables)
3. **Discontiguous clauses:** Add `:- discontiguous` directives or reorganize code
4. **Import overrides:** Either rename local predicate or explicitly handle the override

**Estimated Effort:** 2-3 hours for thorough cleanup

**Priority:** Medium - doesn't block release but improves code quality

---

### 11. Add Negative Test Cases for Mutual Recursion

**Status:** 📋 IDENTIFIED - Needs review and implementation
**Location:** `src/unifyweaver/core/advanced/test_advanced.pl` or test runner
**Created:** 2025-10-15

**Current Test Coverage:**
- ✅ `is_even(0)` → true (base case)
- ✅ `is_even(4)` → true (positive case)
- ✅ `is_odd(3)` → true (positive case)
- ✅ `is_odd(6)` → empty (correctly fails)

**Missing Negative Test Cases:**
- ❌ `is_even(3)` → should fail/return nothing
- ❌ `is_even(5)` → should fail/return nothing
- ❌ `is_odd(2)` → should fail/return nothing
- ❌ `is_odd(4)` → should fail/return nothing

**Discussion:**
Returning nothing (empty result) may be valid behavior for these predicates - they succeed for valid cases and fail silently for invalid cases. This follows Prolog semantics where predicates can succeed (with bindings), fail (no results), or error.

**Review Needed:**
1. Verify empty result is correct/expected behavior
2. Consider if we want explicit false/failure indicators
3. Evaluate if bash exit codes should indicate success/failure
4. Decide if documentation should clarify this behavior

**Fix Strategy (If Needed):**
1. Add negative test cases to test_runner.sh
2. Verify expected behavior (empty vs false vs error)
3. Document the mutual recursion failure semantics
4. Consider adding assertion-based tests if needed

**Estimated Effort:** 1-2 hours (depends on semantic decisions)

---

### 12. Review test_auto Auto-Discovery Behavior

**Status:** 📋 IDENTIFIED - Needs investigation
**Location:** `templates/init_template.pl` auto-discovery system
**Created:** 2025-10-15

**Current Behavior:**
- `test_auto.` reports: `[INFO] No auto-discovered tests available`
- Falls back to manual tests (test_stream, test_recursive, etc.)
- Auto-discovery appears to be failing in test environments

**Expected Behavior:**
- Should auto-discover test files in `tests/` and `tests/core/`
- Should find predicates matching test patterns
- Should provide list of discovered tests

**Investigation Needed:**
1. Check if `auto_discover_tests/0` predicate exists and is callable
2. Verify test file pattern matching is working
3. Review directory scanning in test environments
4. Check if initialization properly sets up auto-discovery

**Estimated Effort:** 1-2 hours

---

## Priority 6: Future Enhancements (Post v0.0.2)

See `context/FUTURE_WORK.md` for:
- Tree recursion with fold helper pattern (fibonacci, binomial coefficients)
- Constraint system integration with advanced recursion
- Education materials completion
- Additional recursion patterns

---

## Tracking

- [x] Priority 1, Item 1: Fix list_length pattern detection ✅ (feat/fold-based-linear-recursion PR)
- [x] Priority 1, Item 2: Fix descendant classification ✅ (feat/fold-based-linear-recursion PR)
- [x] Priority 2, Item 3: Complete linear recursion bash generation ✅ (feat/fold-based-linear-recursion PR)
- [x] Priority 2, Item 4: Complete mutual recursion bash generation ✅ (feat/mutual-recursion-codegen PR)
- [x] Priority 3, Item 5: Fix singleton warnings ✅ (feat/fold-based-linear-recursion PR)
- [x] Priority 3, Item 6: ✅ Already fixed (init_template.pl)
- [x] Priority 4, Item 7: Update test plan docs ✅ (docs commits on main)
- [x] Priority 4, Item 8: Update README limitations ✅ (docs commits on main)
- [x] Priority 5, Item 9: Add regression tests ✅ (test_regressions.pl on main)
- [x] Priority 6, Item 10: Update documentation ✅ (README.md, POST_RELEASE_TODO.md updated)

**Completed (v0.0.1 cycle):** 10/10 items ✅
**New Items (v0.0.2+):** 1 item identified for review
**Total Effort (v0.0.1):** ~20 hours across two feature branches

---

*This document will be updated as items are completed and new issues are discovered.*
