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

### 13. Firewall Philosophy - Blocking vs Guidance

**Status:** 📋 DESIGN DECISION NEEDED
**Location:** `src/unifyweaver/core/firewall.pl`, `tests/core/test_firewall_enhanced.pl`
**Created:** 2025-10-15

**Issue:**
Test expects firewall to throw exceptions for denied services, but current implementation prints message and fails silently. This reveals a deeper question about firewall philosophy.

**Documentation:** See `context/firewall_behavior_design.md` for full analysis

**Two Approaches:**
1. **Security Firewall (Hard Blocking)** - Throw exceptions, stop compilation
2. **Preference Guidance (Soft Constraints)** - Help select best option, only block when no alternatives

**Recommended: Hybrid Approach**
- ALLOW: Explicitly allowed or preferred → succeed
- WARN: Works but not preferred → succeed with warning  
- DENY: Explicitly denied or no alternatives → throw exception
- UNKNOWN: Depends on mode (strict vs permissive)

**Implementation Plan:**
1. Add `mode` configuration: strict/permissive/disabled
2. Add `preferred` vs `fallback` vs `denied` lists
3. Implement warning system for non-preferred tools
4. Create policy templates for different environments
5. Update tests to match chosen philosophy

**Current Fix (v0.0.2):**
- ✅ Updated test to match current behavior (fail without exception)
- ✅ Added comment referencing design document
- Defer full implementation to v0.0.3+

**Estimated Effort:** 4-6 hours for full hybrid implementation

---

### 14. Fix PowerShell Compatibility Layer WSL Backend Invocation from Bash

**Status:** 📋 IDENTIFIED - Known limitation
**Location:** `scripts/powershell-compat/test_compat_layer.ps1`
**Created:** 2025-10-17

**Current Behavior:**
- ✅ Works perfectly when called from PowerShell directly
- ✅ Default Cygwin backend works from both PowerShell and WSL/Bash
- ❌ Setting WSL backend via env var fails when invoked from WSL/Bash

**Error When Called from WSL:**
```bash
powershell.exe -Command "$env:UNIFYWEAVER_EXEC_MODE='wsl'; .\test_compat_layer.ps1"
# Error: The term ':UNIFYWEAVER_EXEC_MODE=wsl' is not recognized...
```

**Root Cause:**
Bash shell escaping adds a `:` prefix when parsing the PowerShell command string, causing PowerShell to interpret it as a malformed command.

**Workaround (Current):**
Set environment variable in Windows before calling, or use PowerShell directly.

**Proposed Fix:**
1. Create a wrapper script approach using `-File` parameter
2. Use a temporary PowerShell script to set env var and invoke test
3. Or document as limitation with recommended usage patterns

**Estimated Effort:** 1-2 hours

**Priority:** Low - The primary use case (running from PowerShell) works correctly. Cross-environment invocation is edge case.

---

## Priority 6: Future Enhancements (Post v0.0.2)

### 15. Implement firewall_implies - Higher-Order Firewall Policies

**Status:** 📋 DESIGN PROPOSAL - Showcase Prolog's Unique Advantages
**Location:** `src/unifyweaver/core/firewall.pl`
**Documentation:** `docs/FIREWALL_GUIDE.md` (Future Enhancements section)
**Created:** 2025-10-19

**Concept:**
Higher-order firewall rules that derive security policies from other policies using Prolog's logical inference capabilities. This would be **extremely difficult or impossible** to implement cleanly in traditional imperative languages, making it a compelling showcase for why Prolog was chosen for UnifyWeaver.

**Example Usage:**
```prolog
% If python3 is denied, automatically deny any source type that uses python3
firewall_implies(denied(python3), denied_source_type(python)).

% If network access is denied, block all HTTP sources
firewall_implies(network_access(denied), denied_source_type(http)).

% If a Python module is blocked, deny any code that imports it
firewall_implies(denied_python_module(requests),
                 block_python_imports_matching('import requests')).

% Transitive implications: If requests is blocked, block urllib3 too (dependency)
firewall_implies(denied_python_module(requests),
                 denied_python_module(urllib3)).
```

**Why This Showcases Prolog's Power:**

1. **Declarative Security Policies** - Express "what" not "how"
   ```prolog
   % In Prolog (elegant):
   firewall_implies(denied(X), denied_source_type(Type)) :-
       source_uses_service(Type, X).

   % In Python (imperative mess):
   def check_firewall(source_type, denied_services):
       for service in denied_services:
           if source_type_uses_service(source_type, service):
               for rule in firewall_rules:
                   if rule.matches(source_type):
                       return DENIED
       return ALLOWED
   ```

2. **Logical Inference** - Prolog automatically derives policies
   ```prolog
   % Define one rule about Python
   firewall_implies(denied(python3), denied_source_type(python)).

   % Prolog automatically knows:
   % - If python3 is denied
   % - Then python source type is denied
   % - Therefore any source(python, ...) should be blocked

   % No manual checking needed!
   ```

3. **Transitive Reasoning** - Automatically handles chains
   ```prolog
   firewall_implies(A, B).
   firewall_implies(B, C).
   % Prolog can infer: A implies C (if we want transitivity)
   ```

4. **Pattern Matching** - Natural syntax for security rules
   ```prolog
   % Block any network access to certain TLDs
   firewall_implies(
       network_hosts(Hosts),
       denied_url_pattern(Pattern)
   ) :-
       member('*.cn', Hosts),
       Pattern = '*.cn'.
   ```

**Design Questions to Address:**

1. **Evaluation Strategy:**
   - Eager (at policy definition time)?
   - Lazy (at validation time)?
   - Trade-offs: Performance vs flexibility

2. **Transitivity:**
   - Should `firewall_implies` be transitive automatically?
   - How to prevent circular implications?
   - Example: A→B, B→C, should A→C be automatic?

3. **Scope:**
   - Apply to `rule_firewall` only?
   - Apply to `firewall_default` only?
   - Apply globally across all policies?

4. **Conflict Resolution:**
   - What if `firewall_implies` creates contradictions?
   - Example: One rule allows, one denies via implication
   - Use "deny always wins" principle?

**Implementation Plan:**

**Phase 1: Basic Implementation (2-3 hours)**
```prolog
% In firewall.pl
:- dynamic firewall_implies/2.

% Expand implications when validating
validate_against_firewall(Target, Options, Firewall) :-
    % Expand firewall with implied rules
    expand_firewall_implications(Firewall, ExpandedFirewall),
    % ... existing validation logic ...

expand_firewall_implications(Firewall, Expanded) :-
    findall(Implied,
        (member(Rule, Firewall),
         firewall_implies(Rule, Implied)),
        ImpliedRules),
    append(Firewall, ImpliedRules, Expanded).
```

**Phase 2: Transitive Closure (1-2 hours)**
```prolog
% Compute transitive closure of implications
expand_firewall_implications_transitive(Firewall, Expanded) :-
    % Iteratively expand until fixed point
    expand_once(Firewall, Step1),
    (   Firewall = Step1
    ->  Expanded = Firewall  % Fixed point reached
    ;   expand_firewall_implications_transitive(Step1, Expanded)
    ).
```

**Phase 3: Cycle Detection (1-2 hours)**
```prolog
% Detect circular implications
check_firewall_implies_cycles :-
    findall(A-B, firewall_implies(A, B), Edges),
    (   has_cycle(Edges)
    ->  format(user_error, 'Warning: Circular firewall implications detected~n', [])
    ;   true
    ).
```

**Testing Strategy:**
```prolog
% In test_firewall_implies.pl

test_basic_implication :-
    assertz(firewall_implies(denied(python3), denied_source_type(python))),
    assertz(firewall_default([denied([python3])])),

    % Should block Python sources due to implication
    source(python, test_source, [python_inline('print("test")')]),
    \+ compile_dynamic_source(test_source/2, [], _),

    writeln('✅ Basic implication works').

test_transitive_implication :-
    assertz(firewall_implies(denied(A), denied_module(A))),
    assertz(firewall_implies(denied_module(M), block_import(M))),
    assertz(firewall_default([denied([requests])])),

    % Should block imports due to transitive implication
    Python = 'import requests',
    \+ validate_python_imports(Python, [denied([requests])]),

    writeln('✅ Transitive implication works').

test_no_circular_implications :-
    assertz(firewall_implies(denied(A), denied(B))),
    assertz(firewall_implies(denied(B), denied(A))),

    % Should detect and warn about cycle
    check_firewall_implies_cycles,

    writeln('✅ Cycle detection works').
```

**Documentation Updates:**
- Update `docs/FIREWALL_GUIDE.md` with `firewall_implies` examples
- Add to README.md as "Why Prolog?" showcase
- Create blog post/article: "Security Policies as Logic Programs"

**Marketing Value:**
This feature directly addresses "Why not just use Python/JavaScript?" by showing:
- Prolog does this **elegantly** in a few lines
- Imperative languages would need complex rule engines
- **Logical inference is Prolog's superpower**
- **Declarative security > imperative security checks**

**Estimated Effort:** 6-10 hours total
- Basic implementation: 2-3 hours
- Transitive closure: 1-2 hours
- Cycle detection: 1-2 hours
- Testing: 2 hours
- Documentation: 2 hours

**Priority:** High - Excellent showcase of Prolog's unique advantages

**Dependencies:** None - can be implemented independently

---

See also `context/FUTURE_WORK.md` for:
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

## Priority 7: Research & Improvement

### 16. Research Pure Prolog Alternatives to External Tool Calls

**Status:** 📋 Research Task  
**Reference:** `docs/development/LANGUAGE_IDIOSYNCRASIES.md` - Anti-Declarative Patterns section  
**Current Situation:** We use external tools (bash, cygpath, etc.) to work around Prolog's automatic cleanup

**Context:**
Currently in `bash_executor.pl`, we bypass Prolog's file I/O and use external bash to create temporary scripts. See the Anti-Declarative Patterns section in LANGUAGE_IDIOSYNCRASIES.md for full details.

**Why We Do This:**
- Prolog's `tmp_file/2` and `open/3` trigger automatic cleanup
- External processes can't access files that Prolog manages internally  
- Path namespace issues between Windows and Cygwin filesystems

**Research Goal:** Investigate pure Prolog alternatives to understand:
1. Can we control Prolog's cleanup behavior?
2. Can we prevent file locking issues?
3. Can we handle paths without external tools like `cygpath`?
4. Are there alternative file creation methods that avoid cleanup hooks?

**Benefits of Pure Prolog Solutions:**
- **Cross-Platform Robustness** - Less dependence on bash/cygpath availability
- **Simpler Architecture** - Fewer subprocess invocations, easier debugging
- **Better Understanding** - Learn Prolog's file system model, document best practices

**Important:** This is research, not a mandate. If external tools work better, we keep them. Don't sacrifice reliability for "purity."

**Estimated Effort:** 10-15 hours (investigation + testing + documentation)

**Priority:** Medium-Low - Current solution works; research valuable for future multiplatform work

**Deliverables:**
- Updated LANGUAGE_IDIOSYNCRASIES.md with research findings
- Test suite comparing different approaches
- Decision matrix: when to use Prolog vs external tools

---

## Priority 8: Testing Infrastructure Enhancement

### 17. Implement Data Source Test Runner Generator

**Status:** 📋 DESIGN NEEDED - Post-Release Enhancement
**Location:** New module `src/unifyweaver/core/data_sources/test_generator.pl` (proposed)
**Reference:** `examples/test_generated_scripts.sh` (current ad-hoc implementation)
**Created:** 2025-10-23

**Current Situation:**
We have an ad-hoc test script (`examples/test_generated_scripts.sh`) that tests the integration test's generated bash scripts:
- `test_output/products.sh` (CSV source)
- `test_output/orders.sh` (JSON source)
- `test_output/analyze_orders.sh` (Python ETL)
- `test_output/top_products.sh` (SQLite query)

This works but is not automatically generated like advanced recursion tests.

**Why This Is Different from test_runner_generator.pl:**

The existing `test_runner_generator.pl` is designed for **unit testing recursive predicates**:
- Tests pure predicates with specific arguments: `factorial "5" ""`
- Multiple test cases per function (0, 1, 5, etc.)
- Self-contained (no external data dependencies)
- Uses `source script.sh` pattern

Data source testing requires **integration testing pipelines**:
- Tests end-to-end data flows: `orders.sh | analyze.sh`
- Single execution per pipeline (with test data)
- Requires test data files (CSV, JSON)
- Uses `bash script.sh` and pipe patterns
- Validates output correctness

**Architectural Considerations:**

Before implementing, these architectural decisions must be addressed:

1. **Module Organization & Responsibility:**
   - Where does this fit in the module hierarchy?
   - Should it be `core/data_sources/test_generator.pl` or `core/testing/data_source_generator.pl`?
   - Who owns test generation: the data source system or the testing system?
   - How does this interact with the existing `core/advanced/test_runner_generator.pl`?

2. **Abstraction Level:**
   - Should the generator be **data-source-aware** (knows about CSV, JSON, Python)?
   - Or should it be **generic** (just generates bash script tests from metadata)?
   - Trade-off: Specific knowledge vs flexibility for future source types

3. **Test Discovery vs Configuration:**
   - **Configuration-based** (current test_runner_generator.pl approach):
     - Hardcoded test cases for known scripts
     - Pro: Explicit control, predictable
     - Con: Manual updates needed for new sources
   - **Discovery-based** (scan generated_file/2 facts):
     - Automatically detect all generated scripts
     - Pro: Adapts to new sources automatically
     - Con: May need heuristics for pipeline ordering
   - **Hybrid** (discover + annotate):
     - Discover scripts but allow metadata annotations
     - Pro: Best of both worlds
     - Con: More complex

4. **Metadata Storage:**
   - Where to store test metadata (expected outputs, pipeline dependencies)?
   - Options:
     - In source definitions (`source(csv, products, [..., test_cases([...])])`)
     - In separate test spec file (`test_specs.pl`)
     - In `generated_file/2` with extended format
     - In comments/annotations in integration_test.pl
   - Trade-off: Colocation vs separation of concerns

5. **Test Execution Model:**
   - **Self-contained bash script** (current approach):
     - Pro: Runs independently, easy to distribute
     - Con: No feedback to Prolog, hard to validate programmatically
   - **Prolog test runner**:
     - Pro: Can validate outputs, integrate with test framework
     - Con: Requires Prolog to run tests, more complex
   - **Hybrid** (bash script that reports to Prolog):
     - Pro: Portable but validatable
     - Con: Complex inter-process communication

6. **Reusability Across Test Types:**
   - Can the same generator support:
     - Unit tests (single source execution)?
     - Integration tests (multi-source pipelines)?
     - Regression tests (compare with expected outputs)?
   - Or should we have specialized generators for each?
   - Trade-off: One flexible generator vs multiple focused generators

7. **Relationship to Dynamic Source Compiler:**
   - The `dynamic_source_compiler.pl` generates bash scripts
   - Should the test generator be part of that workflow?
   - Should script generation automatically trigger test generation?
   - Or keep them completely separate?
   - Trade-off: Tight coupling vs loose coupling

8. **Extensibility for Future Source Types:**
   - How does the generator handle new source types (e.g., future SQL, XML, YAML sources)?
   - Should the generator use a plugin architecture?
   - Should each source type provide its own test generation logic?
   - Or should there be a common test generation interface?

**Design Questions to Answer:**

1. **Architecture:**
   - Should this be a separate module (`data_source_test_generator.pl`)?
   - Or extend `test_runner_generator.pl` with pipeline support?
   - Or integrate with `integration_test.pl` to auto-generate its own test runner?

2. **Pipeline Representation:**
   - How to specify multi-stage pipelines in Prolog?
   - Example: Need to represent `orders.sh | analyze_orders.sh`
   ```prolog
   % Possible syntax:
   test_pipeline(etl_demo, [
       stage(extract, 'test_output/orders.sh', []),
       stage(transform, 'test_output/analyze_orders.sh', [stdin]),
       expected_output_contains('Mouse')
   ]).
   ```

3. **Test Data Management:**
   - Should the generator create test data files?
   - Or assume `integration_test.pl` has already created them?
   - How to handle setup/teardown?

4. **Output Validation:**
   - Current ad-hoc script just runs and shows output
   - Should generated tests validate output programmatically?
   - How to specify expected results?

5. **Integration Points:**
   - Should `integration_test.pl` call the generator at the end?
   - Or is test generation a separate workflow?
   - How does this fit into the overall test plan?

**Proposed Implementation Approach:**

**Option A: Separate Module (Recommended)**
```prolog
% src/unifyweaver/core/data_sources/test_generator.pl
:- module(data_source_test_generator, [
    generate_integration_test_runner/0,
    generate_integration_test_runner/1
]).

% Generate test runner from integration_test.pl's generated_file/2 facts
generate_integration_test_runner(OutputPath) :-
    findall(Type-Path, generated_file(Type, Path), Files),
    open(OutputPath, write, Stream),
    write_test_header(Stream),
    write_data_source_tests(Stream, Files),
    write_test_footer(Stream),
    close(Stream).

% Generate individual tests based on source type
write_data_source_tests(Stream, Files) :-
    % CSV sources: source script && call function
    % JSON sources: bash script
    % Python pipelines: bash a.sh | bash b.sh
    % SQLite queries: bash script
    ...
```

**Option B: Extend test_runner_generator.pl**
- Add `generate_integration_tests/1` predicate
- Add pipeline support to existing generator
- Risk: Mixing unit tests and integration tests in one module

**Option C: Integration Test Self-Generation**
- Modify `integration_test.pl` to generate its own test runner
- Uses `generated_file/2` facts it already tracks
- Simplest but least reusable

**Recommended: Option A** - Separate module for clarity and extensibility

**Example Generated Output:**
```bash
#!/bin/bash
# AUTO-GENERATED by data_source_test_generator.pl
# DO NOT EDIT MANUALLY

echo "=== Data Source Integration Tests ==="

echo "1. CSV Source (Products):"
source test_output/products.sh && products
echo

echo "2. JSON Source (Orders):"
bash test_output/orders.sh
echo

echo "3. Python ETL Pipeline:"
bash test_output/orders.sh | bash test_output/analyze_orders.sh
echo

echo "4. SQLite Query (Top Products):"
bash test_output/top_products.sh
echo

echo "✅ All integration tests complete"
```

**Testing the Generator:**
```prolog
% After running integration_test.pl:
?- use_module(data_source_test_generator).
?- generate_integration_test_runner('test_output/test_runner.sh').
% Generated test runner: test_output/test_runner.sh

% Then in bash:
$ bash test_output/test_runner.sh
```

**Implementation Plan:**

**Phase 1: Basic Generation (3-4 hours)**
- Create `data_source_test_generator.pl` module
- Query `generated_file/2` facts
- Generate simple test runner (like current ad-hoc)
- Write tests for the generator itself

**Phase 2: Pipeline Support (2-3 hours)**
- Add pipeline specification format
- Handle multi-stage data flows
- Support stdin piping between stages

**Phase 3: Validation (2-3 hours)**
- Add expected output specifications
- Generate assertions in test runner
- Report pass/fail instead of just showing output

**Phase 4: Integration (1-2 hours)**
- Integrate with `integration_test.pl`
- Optionally auto-generate runner at end of test
- Update test plans to reference generated runner

**Current Workaround:**
✅ Ad-hoc `test_generated_scripts.sh` works and is documented
✅ Script is copied to test environments by `init_testing.sh`
✅ Comments clearly mark it as temporary and reference this TODO

**Estimated Total Effort:** 8-12 hours

**Priority:** Medium - Improves testing infrastructure but not blocking release

**Dependencies:** None - can be implemented independently after v0.0.2

**See Also:**
- `src/unifyweaver/core/advanced/test_runner_generator.pl` - Existing generator for recursion tests
- `examples/test_generated_scripts.sh` - Current ad-hoc implementation
- `examples/integration_test.pl` - Uses `generated_file/2` tracking
