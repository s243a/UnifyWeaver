# Post-Release TODO List

**Created:** 2025-10-12
**Target Release:** v0.2 or later

This tracks work to be done after v0.1 release.

---

## Priority 1: Known Limitations (Fix These First)

### 1. ✅ RESOLVED: `list_length/2` Linear Recursion Detection

**Status:** ✅ WORKING CORRECTLY (Verified 2025-10-26)
**Location:** `test_advanced.pl` - Linear Recursion Compiler, Test 1
**Resolution:** Pattern detection works correctly. Issue was **incorrectly documented**.

**Verification:**
```prolog
list_length([], 0).
list_length([_|T], N) :- list_length(T, N1), N is N1 + 1.

% Compiles successfully as linear recursion with memoization
compile_advanced_recursive(list_length/2, [], Code).
% Result: ✓ Compiled as linear recursion
```

**What Actually Works:**
- ✅ Detected as linear recursion
- ✅ Generated with memoization (associative arrays)
- ✅ Efficient O(n) implementation

**Test Results:** All pattern detection tests passing.

---

### 2. ✅ RESOLVED: `descendant/2` Pattern Detection

**Status:** ✅ WORKING CORRECTLY (Verified 2025-10-26)
**Location:** `test_recursive.pl` - Recursive Predicates Test
**Resolution:** Transitive closure detection and BFS optimization work correctly. Issue was **incorrectly documented**.

**Verification:**
```prolog
descendant(X, Y) :- parent(Y, X).
descendant(X, Z) :- parent(Y, X), descendant(Y, Z).

% Compiles successfully with BFS optimization
compile_recursive(descendant/2, [], Code).
% Result: Classification: transitive_closure(parent)
% Generated code includes work queue, visited tracking, BFS loop
```

**What Actually Works:**
- ✅ Correctly detected as transitive_closure
- ✅ Generates BFS-optimized code with work queues
- ✅ Includes visited tracking (no cycles/duplicates)
- ✅ Efficient O(V+E) graph traversal

**Generated Code Features:**
- Work queue for iterative BFS
- Visited hash table for cycle prevention
- No recursion (bash-safe)

**Test Results:** All tests passing. See `examples/test_pattern_detection_issues.pl`.

---

## Priority 2: Bash Code Generation Completion

### 3. ✅ RESOLVED: Linear Recursion Bash Generation

**Status:** ✅ WORKING CORRECTLY (Verified 2025-10-26)
**Location:** `src/unifyweaver/core/advanced/linear_recursion.pl`
**Resolution:** Testing confirmed linear recursion works correctly. Issue was incorrectly documented.

**Verification:**
```bash
# factorial(5) = 120 ✓
# factorial(6) = 720 ✓
# All test cases pass
```

**What Actually Works:**
- ✅ Pattern detection correctly identifies linear recursion (exactly 1 recursive call)
- ✅ Bash code generation complete with memoization
- ✅ Factorial compiles and executes correctly
- ✅ All arithmetic operations handled properly

**Note:** Fibonacci was incorrectly thought to be linear recursion, but it has 2 recursive calls, making it tree/fold recursion. The confusion arose from a misleading test comment.

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

### 5. ✅ RESOLVED: Fix Singleton Variable Warnings

**Status:** ✅ ALREADY FIXED (Verified 2025-10-26)
**Resolution:** Singleton warnings have already been addressed in the codebase.

**Verification:**
- Loaded all modules and ran tests - no singleton warnings appear
- Code inspection shows all variables are properly used
- `MemoCheckCode` and `MemoStoreCode` are used in line 206 of tree_recursion.pl
- No current singleton warnings in test runs

**Note:** This issue was likely fixed during prior development but not marked resolved in TODO.

---

### 5a. ✅ RESOLVED: Fix Module Import Conflicts in Source Plugins

**Status:** ✅ FIXED (Completed 2025-10-26)
**Solution:** Implemented Option C - source plugins export nothing

**Changes Made:**
All source plugins updated to export empty list:
- ✅ csv_source.pl
- ✅ json_source.pl
- ✅ http_source.pl
- ✅ python_source.pl
- ✅ awk_source.pl

**Implementation:**
```prolog
% Before (caused conflicts):
:- module(csv_source, [
    compile_source/4,
    validate_config/1,
    source_info/1
]).

% After (no conflicts):
% Export nothing - all access goes through plugin registry
:- module(csv_source, []).
```

**Verification:**
```prolog
% Test: Load all plugins without conflicts
:- use_module(sources/csv_source).
:- use_module(sources/json_source).
:- use_module(sources/python_source).
:- use_module(sources/http_source).
:- use_module(sources/awk_source).

% Result: ✓ All plugins load without errors or warnings
% See: examples/test_source_plugins_no_conflict.pl
```

**How It Works:**
- Interface predicates (`compile_source/4`, `validate_config/1`, `source_info/1`) remain in each plugin but are not exported
- Plugin system uses dynamic dispatch via `register_source_type/2`
- Core system calls predicates via module qualification: `csv_source:compile_source/4`
- Users only need to import `sources.pl`, never individual plugins directly

**Benefits:**
- ✅ No import conflicts when using multiple source types
- ✅ Cleaner API - users interact only with `sources.pl`
- ✅ Plugins remain self-contained and testable
- ✅ No breaking changes to existing functionality

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

### 5b. ✅ RESOLVED: Fix PowerShell Integration Test Sequential Execution Hang

**Status:** ✅ FIXED (Completed 2025-11-03)
**Discovered:** PowerShell test plan, full integration test
**Priority:** Medium (workaround exists)
**Solution:** Added 0.5s delays between test stages (Commit 7348270)

**Issue:**
The full integration test (`examples/integration_test.pl`) hangs when running in PowerShell/Windows environments. The test crashes consistently at the Python Source Test stage after successfully completing CSV and JSON tests.

**Current Behavior:**
- ✅ Individual tests pass when run in isolation
- ✅ All generated scripts are correct and execute properly
- ✅ WSL/Linux environment passes complete integration test
- ❌ PowerShell environment hangs during 3rd sequential bash execution

**Crash Point:**
```prolog
test_python_source :-
    % ...
    compile_dynamic_source(orders/4, [], OrdersCode),  % Compiles successfully
    write_and_execute_bash(OrdersCode, '', OrdersOutput),  % ← HANGS HERE
```

This is the **third call** to `write_and_execute_bash` in sequence (after CSV and JSON tests).

**Investigation Performed:**
1. ❌ Not console buffer overflow - redirecting to file doesn't fix it
2. ❌ Not debug output - disabling all DEBUG statements doesn't fix it
3. ❌ Not temp file accumulation - only 2 temp files exist
4. ❌ Not PowerShell compatibility layer - hangs even without `init_unify_compat.ps1`
5. ✅ Specific to sequential execution - individual tests work fine

**Root Cause (Suspected):**
SWI-Prolog's `process_create/3` on Windows appears to have a resource leak or deadlock issue when called multiple times in rapid succession. This is likely a limitation of SWI-Prolog's Windows process management, not UnifyWeaver code.

**Workaround (v0.0.2):**
Run integration tests individually in PowerShell:
```powershell
swipl -l init.pl -l examples/integration_test.pl -g "test_csv_source, halt" -t halt
swipl -l init.pl -l examples/integration_test.pl -g "test_json_source, halt" -t halt
swipl -l init.pl -l examples/integration_test.pl -g "test_python_source, halt" -t halt
swipl -l init.pl -l examples/integration_test.pl -g "test_sqlite_source, halt" -t halt
```

**Fix Applied (2025-11-03):**
✅ **Option A:** Added 0.5s delays between `write_and_execute_bash` calls
   - Implemented `sleep(0.5)` after each test stage
   - Gives Windows process cleanup time between bash executions
   - Total overhead: ~2 seconds for full integration test
   - Solution is simple, effective, and harmless on all platforms

**Changes Made:**
- `examples/integration_test.pl`: Added `sleep(0.5)` after each of 4 test stages
  - After test_csv_source (line 217)
  - After test_json_source (line 243)
  - After test_python_source (line 271)
  - After test_sqlite_source (line 295)

**Result:**
- ✅ Full integration test now runs successfully on PowerShell/Windows
- ✅ No functional changes (delays are harmless on Linux/macOS)
- ✅ Simple, maintainable solution
- ✅ No need for upstream bug report or complex refactoring

**Remaining Options (not needed but documented for reference):**
- Option B: Alternative process execution - not needed
- Option C: Refactor test architecture - not needed
- Option D: Report to SWI-Prolog - may still be useful for upstream awareness

**Related Files:**
- `src/unifyweaver/core/bash_executor.pl` - `write_and_execute_bash/3`
- `examples/integration_test.pl` - Full test suite
- `docs/development/testing/v0_0_2_powershell_test_plan.md`

---

## Priority 3: Documentation Updates

### 5. ✅ RESOLVED: Update Test Plan with Known Failures

**Status:** ✅ NO LONGER NEEDED (2025-10-26)
**Reason:** Issues #1 and #2 (list_length/2 and descendant/2) have been verified as working correctly. These were documentation errors, not actual bugs.

**Verification:**
- list_length/2: Correctly detected as linear recursion ✓
- descendant/2: Correctly classified as transitive_closure with BFS ✓
- No known test failures to document

---

### 6. ✅ RESOLVED: Add Known Limitations to README.md

**Status:** ✅ COMPLETED (2025-10-26)
**Outcome:** README.md limitations section reviewed and verified accurate.

**Current Limitations (Verified as Accurate):**
- Divide-and-conquer patterns (quicksort, mergesort) not yet supported ✓
- Requires Bash 4.0+ for associative arrays ✓
- Tree recursion uses list representation only ✓

**Note:** Pattern detection issues mentioned in original proposal were documentation errors and have been resolved.

---

## Priority 4: Testing Infrastructure

### 7. ✅ COMPLETE: Add Regression Tests for Pattern Detection Verification

**Status:** ✅ COMPLETE (2025-10-26)
**Location:** `examples/test_pattern_detection_regression.pl`

**Tests Implemented:**

1. **test_list_length_pattern** - Verifies list_length/2 linear recursion detection
   - ✓ Pattern detection works (is_linear_recursive_streamable)
   - ✓ Compilation succeeds
   - ✓ Generated code includes memoization

2. **test_descendant_classification** - Verifies descendant/2 transitive closure
   - ✓ Compilation succeeds
   - ✓ Generated code uses BFS optimization (queue-based)
   - ✓ Includes visited tracking for cycle prevention

3. **test_factorial_linear_recursion** - Verifies factorial compilation and execution
   - ✓ Compiles as linear recursion
   - ✓ Generated bash script executes correctly
   - ✓ Produces correct results (5! = 120)

**Usage:**
```bash
swipl -g main -t halt examples/test_pattern_detection_regression.pl
```

**Result:** All tests pass ✓

---

## Priority 5: Post v0.0.2 Improvements

### 10. ✅ RESOLVED: Clean Up Singleton Variable and Code Quality Warnings

**Status:** ✅ COMPLETE (Verified 2025-11-03)
**Location:** Multiple files across core and advanced modules
**Created:** 2025-10-15

**Resolution:**
All singleton warnings mentioned in this item have been previously addressed. Verification performed 2025-11-03 shows:
- ✅ No singleton warnings in `stream_compiler.pl`
- ✅ No singleton warnings in `linear_recursion.pl`
- ✅ No singleton warnings in `firewall.pl`
- ✅ Fixed remaining warning in `fixed_size.pl:101` (commit 9cb9068)

**Verification Commands:**
```bash
# Test core modules - no warnings
swipl -q -g "use_module('src/unifyweaver/core/stream_compiler'), halt"
swipl -q -g "use_module('src/unifyweaver/core/advanced/linear_recursion'), halt"
swipl -q -g "use_module('src/unifyweaver/core/firewall'), halt"
swipl -q -g "use_module('src/unifyweaver/core/partitioners/fixed_size'), halt"
```

**Final Fix (2025-11-03):**
- `fixed_size.pl:101` - Marked `CurrentSize` as `_CurrentSize` (intentionally unused in base case)

**Conclusion:**
The warnings listed in this item were already fixed during prior development. The singleton warning cleanup has been completed across the codebase.

---

### 11. ✅ RESOLVED: Add Negative Test Cases for Mutual Recursion

**Status:** ✅ COMPLETE (Already implemented, verified 2025-11-03)
**Location:** `examples/test_mutual_recursion_negative.pl`
**Created:** 2025-10-15
**Completed:** Added SPDX header (commit effc273)

**Test Coverage (All Implemented):**

**Positive Cases:**
- ✅ `is_even(0)` → succeeds (base case)
- ✅ `is_even(2)` → succeeds
- ✅ `is_even(4)` → succeeds
- ✅ `is_odd(1)` → succeeds (base case)
- ✅ `is_odd(3)` → succeeds
- ✅ `is_odd(5)` → succeeds

**Negative Cases (All Implemented):**
- ✅ `is_even(1)` → correctly fails
- ✅ `is_even(3)` → correctly fails
- ✅ `is_even(5)` → correctly fails
- ✅ `is_odd(0)` → correctly fails
- ✅ `is_odd(2)` → correctly fails
- ✅ `is_odd(4)` → correctly fails
- ✅ `is_odd(6)` → correctly fails

**Edge Cases:**
- ✅ Large numbers: `is_even(100)`, `is_odd(99)`
- ✅ Negative inputs: `is_even(-2)`, `is_odd(-3)` (correctly fail)

**Bash Execution Tests:**
- ✅ Compiles to bash correctly
- ✅ Positive cases exit with code 0
- ✅ Negative cases fail appropriately

**Behavior Documented:**
The file confirms that returning nothing (failing) is correct Prolog semantics:
- Predicates succeed with bindings for valid inputs
- Predicates fail (no results) for invalid inputs
- This matches expected Prolog behavior

**Test File:**
- `examples/test_mutual_recursion_negative.pl` - Complete test suite (262 lines)
- Run with: `swipl -q -l examples/test_mutual_recursion_negative.pl -g main -t halt`

---

### 12. ✅ RESOLVED: Review test_auto Auto-Discovery Behavior

**Status:** ✅ NOT APPLICABLE - Template no longer used (Resolved 2025-10-29)
**Location:** `templates/init_template.pl` auto-discovery system
**Created:** 2025-10-15

**Investigation Results:**
- `test_auto` and `auto_discover_tests` are defined in `templates/init_template.pl`
- **No `init.pl` exists in project root** - template is not actively used
- Current test architecture uses individual test files in `examples/` directory
- Each test is run independently (e.g., `swipl -l test_firewall.pl -g main -t halt`)

**Current Architecture (Working):**
```
examples/
├── test_firewall_implies.pl          (278 lines, 6 tests)
├── test_firewall_network_access.pl   (449 lines, 8 tests)
├── test_firewall_powershell.pl       (4 tests)
├── test_firewall_tools.pl            (5 tests)
└── ... (30+ individual test files)
```

**Conclusion:**
The `test_auto` auto-discovery system is an **obsolete template** from an earlier design iteration. The current approach of individual test files in `examples/` is:
- ✅ Simpler and more maintainable
- ✅ Better for CI/CD (run specific tests)
- ✅ Easier to debug (isolated failures)
- ✅ Already working well

**Resolution:** No action needed. The template can remain for reference but is not part of the active codebase.

**Recommendation:** If auto-discovery is desired in the future, implement it as a test runner script rather than initialization-time discovery.

---

### 13. ✅ REVIEWED: Firewall Philosophy - Blocking vs Guidance

**Status:** ✅ DESIGN REVIEW COMPLETE (2025-10-26)
**Location:** `src/unifyweaver/core/firewall.pl`, `tests/core/test_firewall_enhanced.pl`
**Created:** 2025-10-15
**Review Document:** `context/firewall_philosophy_review.md` (comprehensive analysis)

**Issue:**
Test expects firewall to throw exceptions for denied services, but current implementation prints message and fails silently. This reveals a deeper question about firewall philosophy.

**Review Findings:**
✅ Current implementation is **consistent and correct** within its design
✅ Operates on **fundamental security policies** (not derived predicates)
✅ Validation behavior well-documented: print error, fail silently
✅ Test expectations were incorrect (have been fixed in v0.0.2)
✅ Architecture supports future hybrid approach

**Current Behavior (v0.0.2):**
- Fundamental rules: allowed/denied services, network, file access
- Allowlist/denylist validation
- Print error message to stderr
- Fail silently (returns `false`, no exceptions)
- Separation of fundamental rules from derived preferences

**Recommended Future Approach: Hybrid (3-Level Response)**
- **ALLOW:** Explicitly allowed or preferred → succeed
- **WARN:** Works but not preferred → succeed with warning
- **DENY:** Explicitly denied or no alternatives → fail/throw (mode-dependent)
- **UNKNOWN:** Depends on mode (strict vs permissive)

**Implementation Roadmap:**
- **Phase 1 (v0.0.2):** ✅ COMPLETE - Clarify current behavior, fix tests
- **Phase 2 (v0.0.3):** Implement hybrid design (allow/warn/deny + modes)
- **Phase 3 (v0.0.4):** Preference chains with fallback logic
- **Phase 4 (v0.1.0):** Higher-order firewall policies (showcase feature)

**Key Documentation:**
- `context/firewall_philosophy_review.md` - Comprehensive analysis (new)
- `context/firewall_behavior_design.md` - Original design analysis
- `docs/FIREWALL_GUIDE.md` - User guide
- `planning/FIREWALL_TODO.md` - Implementation plan

**Implementation Plan (Phase 2 - v0.0.3):**
1. Add `mode` configuration: strict/permissive/disabled
2. Add `preferred` vs `fallback` vs `denied` lists
3. Implement warning system for non-preferred tools
4. Create policy templates for different environments
5. Update tests to match new three-level system

**Estimated Effort:**
- Phase 2 (hybrid): 4-6 hours
- Phase 3 (preferences): 6-8 hours
- Phase 4 (higher-order): 10-15 hours
- **Total:** 20-29 hours across v0.0.3 to v0.1.0

**Why This Showcases Prolog:**
- Declarative security policies
- Logical inference (automatic derivation from fundamental rules)
- Transitive reasoning (A→B, B→C ⟹ A→C)
- Pattern matching for complex security rules
- Difficult to implement cleanly in imperative languages

---

### 14. ✅ RESOLVED: Fix PowerShell Compatibility Layer WSL Backend Invocation from Bash

**Status:** ✅ FIXED (Already implemented in commit ce324e3, verified 2025-11-03)
**Location:** `scripts/powershell-compat/`
**Created:** 2025-10-17
**Solution:** Wrapper script approach using `-File` parameter

**Original Issue:**
- ✅ Works perfectly when called from PowerShell directly
- ✅ Default Cygwin backend works from both PowerShell and WSL/Bash
- ❌ Setting WSL backend via env var fails when invoked from WSL/Bash

**Error When Called from WSL (before fix):**
```bash
powershell.exe -Command "$env:UNIFYWEAVER_EXEC_MODE='wsl'; .\test_compat_layer.ps1"
# Error: The term ':UNIFYWEAVER_EXEC_MODE=wsl' is not recognized...
```

**Root Cause:**
Bash shell escaping adds a `:` prefix when parsing the PowerShell command string.

**Fix Implemented:**
✅ Created wrapper scripts using `-File` parameter approach:
- `test_compat_layer_wsl.ps1` - Sets WSL backend and runs test
- `test_compat_layer_cygwin.ps1` - Sets Cygwin backend and runs test
- `test_from_bash.sh` - Bash script that invokes wrapper via `-File`

**Usage:**
```bash
# From Bash/WSL - now works correctly
./scripts/powershell-compat/test_from_bash.sh wsl
./scripts/powershell-compat/test_from_bash.sh cygwin

# Or directly with PowerShell -File
powershell.exe -File ./scripts/powershell-compat/test_compat_layer_wsl.ps1
```

**Benefits:**
1. ✅ Avoids shell escaping issues
2. ✅ Clean cross-environment invocation
3. ✅ Easy to test different backends
4. ✅ Well-documented in README.md

**Files:**
- `scripts/powershell-compat/test_compat_layer_wsl.ps1` (wrapper)
- `scripts/powershell-compat/test_compat_layer_cygwin.ps1` (wrapper)
- `scripts/powershell-compat/test_from_bash.sh` (bash invoker)
- `scripts/powershell-compat/README.md` (documentation)

---

## Priority 6: Future Enhancements (Post v0.0.2)

### 14a. Fix PowerShell Integration Test Hang After 3rd Data Source Test

**Status:** 📋 CRITICAL BUG - WORKAROUND IN PLACE
**Created:** 2025-11-15
**Priority:** High (blocks full integration testing on Windows)
**Platform:** PowerShell/Windows only (works fine on Linux/WSL bash)

**Issue:**
Integration test hangs during SQLite source compilation after running CSV, JSON, and Python tests. The compilation starts but never completes, causing the entire test suite to hang.

**Symptoms:**
```
[CODE] === Python Source Test ===
  ✅ Python processing working

[SAVE] === SQLite Source Test ===
DEBUG: About to compile top_products/3
Compiling dynamic source: top_products/3 using python
  Compiling Python source: top_products/3
<HANGS HERE - never returns>
```

**What Works:**
- ✅ SQLite test runs successfully in ISOLATION
- ✅ SQLite test runs successfully on LINUX
- ✅ All previous tests (CSV, JSON, Python) pass on PowerShell
- ✅ Sequential compilation works in isolation
- ✅ Database operations work in isolation
- ✅ Mode switching (inline → sqlite) works in isolation

**What Fails:**
- ❌ SQLite compilation ONLY when run after CSV + JSON + Python tests on PowerShell
- ❌ Hangs inside `compile_source/4` in `python_source.pl`
- ❌ Never reaches the end of compilation
- ❌ No error message - just silent hang

**Investigation Performed:**
1. ✅ Ruled out timing issues (increasing sleep didn't help)
2. ✅ Ruled out mode switching (inline→sqlite works in isolation)
3. ✅ Ruled out database locking (DB operations work in sequence)
4. ✅ Ruled out process execution issues (works with execution in isolation)
5. ✅ Confirmed compilation starts but doesn't complete
6. ✅ Confirmed it's a cumulative state issue (only after 3 previous tests)

**Root Cause (Suspected):**
State corruption in python_source.pl or template system after multiple compilations on PowerShell/Windows. The exact failure point is inside `compile_source/4` after printing "Compiling Python source: top_products/3" but before returning.

**Workaround (v0.0.2):**
Skip `test_sqlite_source` on Windows platform in integration test:
```prolog
(   get_platform(Platform),
    Platform = windows
->  safe_format('  ⚠ Skipped on Windows (known issue - works on Linux)~n', [])
;   test_sqlite_source
)
```

**Files Affected:**
- `examples/integration_test.pl` - Contains workaround
- `src/unifyweaver/sources/python_source.pl` - Suspected location of bug

**Next Steps:**
1. Add extensive debug logging inside `python_source.pl:compile_source/4`
2. Check for global state or caching issues in template system
3. Check for resource leaks after multiple `render_named_template` calls
4. Test with different PowerShell versions
5. Compare SWI-Prolog behavior on Windows vs Linux
6. Potentially refactor to avoid state accumulation

**Estimated Effort:** 6-10 hours
- Investigation: 4-6 hours (trace execution, add logging)
- Fix: 2-3 hours (once root cause found)
- Testing: 1 hour (verify on both platforms)

**Related:**
- All isolated test files in `test_env_ps9/test_*.pl` work correctly
- Linux integration test passes completely (test_env14)

---

### 14b. Investigate C# Test Cleanup Permission Error on WSL/Dropbox

**Status:** 📋 INVESTIGATION NEEDED
**Created:** 2025-11-15
**Priority:** Low (doesn't affect test results, just cleanup)
**Platform:** WSL + Dropbox sync

**Issue:**
During C# query target tests, cleanup occasionally fails with permission errors:
```
warning: could not cleanup tmp/csharp_query_test_reachable_8b6ea072-c24d-11f0-aac7-00155d6d13b7:
error(permission_error(delete,directory,tmp/csharp_query_test_reachable_8b6ea072-c24d-11f0-aac7-00155d6d13b7),
context(system:delete_directory/1,Permission denied))
```

**Observed During:**
- `test_csharp_query_target.pl` execution
- Specifically on `reachable` test case
- Only affects cleanup, not test execution or results
- All tests pass successfully despite cleanup warning

**Possible Causes:**
1. **Dropbox Sync Conflict** - Dropbox holding file handles during sync
2. **WSL File System Timing** - Windows/Linux file system sync delay
3. **dotnet Process Cleanup** - .NET runtime still holding files
4. **Prolog Cleanup Bug** - SWI-Prolog's `delete_directory/1` issue on WSL

**Current Impact:**
- ✅ Tests pass correctly
- ✅ C# code generation works
- ✅ C# execution works
- ⚠️ Temporary directories not cleaned up (minor disk space issue)
- ⚠️ Warning message in test output (cosmetic)

**Investigation Steps:**
1. Check if issue is specific to Dropbox-synced directories
2. Test with `--csharp-query-keep` flag (intentionally preserve dirs)
3. Add delay before cleanup to allow Windows/WSL sync
4. Check if dotnet process cleanup is complete before directory deletion
5. Test on native Linux (non-WSL) to isolate WSL-specific issues
6. Test on non-Dropbox directory to isolate sync issues

**Potential Solutions:**
- Add retry logic with delay for directory cleanup
- Skip cleanup on WSL + Dropbox (log warning instead)
- Use different temp directory location (outside Dropbox)
- Improve process cleanup sequencing

**Workaround:**
- Use `--csharp-query-keep` flag to intentionally preserve directories
- Manual cleanup: `rm -rf tmp/csharp_query_test_*`

**Estimated Effort:** 3-5 hours (investigation + fix + testing)

**Related:**
- `tests/core/test_csharp_query_target.pl` - Test file with cleanup code
- WSL + Dropbox sync behavior
- SWI-Prolog's `delete_directory/1` implementation

---

### 15. Optimization Strategy Predicates

**Status:** 📋 FUTURE ENHANCEMENT
**Created:** 2025-10-26
**Priority:** Low (current fallback patterns work correctly)

**Concept:**
Allow users to configure optimization strategies for recursive predicates. While pattern detection currently works correctly (linear recursion uses memoization, transitive closure uses BFS), there may be cases where users want explicit control over optimization tradeoffs.

**Example Usage:**
```prolog
% Global default
:- set_optimization_strategy(speed).  % Prefer memoization

% Per-predicate override
:- optimization_strategy(fibonacci/2, speed).      % Use memoization
:- optimization_strategy(large_graph/2, memory).   % Use streaming/fold
:- optimization_strategy(simple_list/2, readability). % Use simple recursion
```

**Strategy Options:**
- `optimize(speed)` - Prefer memoization for faster lookups (more memory)
- `optimize(memory)` - Prefer streaming/fold patterns (less memory)
- `optimize(readability)` - Prefer simpler patterns even if less efficient
- `auto` - Let pattern detection choose (current default)

**Configuration Levels:**
1. Global default (apply to all predicates)
2. Firewall policy level (apply to specific source types)
3. Per-predicate directive (most specific, highest priority)

**Current Behavior:**
- Pattern detection chooses appropriate optimizations automatically
- Linear recursion → memoization (efficient)
- Transitive closure → BFS with work queue (efficient)
- Fold pattern → graph-based (used when linear recursion forbidden for testing)

**Background - Why We Considered This:**

During investigation of Issues #1 and #2, we examined fibonacci compilation with the fold pattern and initially misinterpreted the results:

1. **Initial Observation:** fibonacci compiled with fold pattern showed O(2^n) complexity (recursive calls without memoization)
   ```bash
   # Generated code made repeated recursive calls
   local left=$($0 "$n1")   # No caching
   local right=$($0 "$n2")  # Recomputes same values
   ```

2. **Initial Misunderstanding:** Thought this was a bug - shouldn't fibonacci use memoization?

3. **Key Insight:** Found `forbid_linear_recursion(test_fib/2)` directive in the code
   - This is **intentional** for testing graph recursion capabilities
   - Fibonacci naturally fits linear recursion pattern (which would use memoization)
   - By forbidding linear recursion, it forces the fold pattern (graph-based)
   - This tests that the fold pattern works, even if it's not optimal for fibonacci

4. **Realization:** The "inefficiency" is deliberate - it's a test case, not production usage
   - Pattern detection works correctly
   - Each pattern (linear, fold, transitive closure) is optimized appropriately
   - The fold pattern isn't meant to be optimal for fibonacci - it's testing alternative compilation paths

5. **Future Consideration:** While not needed now, there might be cases where users want explicit control:
   - Choose speed vs memory tradeoffs
   - Override automatic pattern selection
   - Test different compilation strategies
   - Hence this future enhancement proposal

**Why Not Now:**
- Current automatic pattern detection produces good code
- Optimization strategies are correctly applied by default
- The fibonacci "issue" was actually a test case working as designed
- Adding explicit control adds complexity without clear immediate benefit
- Can be added later if users request fine-grained control

**Estimated Effort:** 6-8 hours

---

### 16. Multi-Call Linear Recursion with Independent Arguments

**Status:** 📋 FUTURE ENHANCEMENT
**Created:** 2025-10-26
**Priority:** Medium (optimization opportunity, not a bug)
**Documentation:** `docs/RECURSION_PATTERN_THEORY.md` (theory exists)

**Concept:**
Extend linear recursion detection to handle multiple recursive calls when the calls are **independent**. Currently, fibonacci and similar patterns compile as "fold pattern" but could use simpler linear recursion with memoization.

**Theory - Independence Criteria:**

A predicate with 2+ recursive calls can use linear recursion + memoization when ALL of:

1. **Scalar Arguments** - Recursive call arguments are computed values, NOT structural parts
   - ✓ `N1 is N - 1, fib(N1, F1)` - computed via `is`
   - ✗ `tree_sum([V,L,R], S) :- tree_sum(L, ...)` - L from pattern matching

2. **Arguments Computed Before Calls** - All recursive call arguments determined before any call
   - ✓ `N1 is N-1, N2 is N-2, fib(N1, F1), fib(N2, F2)` - N1, N2 computed first
   - ✗ `bad(N, R1), X is R1+1, bad(X, R2)` - X depends on R1's output

3. **No Variable Dependencies Across Calls** - Each recursive call argument is a **single variable** not shared with other calls
   - ✓ `fib(N1, F1), fib(N2, F2)` - N1 and N2 are distinct variables
   - ✗ `bad(X, R1), bad(X, R2)` - both calls use same variable X (potential dependency)

4. **Pure Aggregation** - Results only combined AFTER all calls complete
   - ✓ `fib(N1, F1), fib(N2, F2), F is F1 + F2` - aggregation after both calls
   - ✗ `bad(N, R1), X is R1 + 1, ...` - uses R1 before second call

**Proof Sketch:**

Independence follows from single-variable arguments:
- Each recursive call receives a **distinct computed variable** (N1, N2, N3, ...)
- No variable appears in multiple recursive calls as an argument
- Therefore: No data flow between calls → calls are independent
- Memoization works: Each call can be cached separately by its unique argument

**Examples:**

```prolog
% Should use linear + memo (currently uses fold)
fib(N, F) :-
    N1 is N - 1, N2 is N - 2,      % Computed scalars
    fib(N1, F1), fib(N2, F2),      % Distinct variables: N1 ≠ N2
    F is F1 + F2.                  % Pure aggregation

% Tribonacci - 3 independent calls
trib(N, T) :-
    N1 is N - 1, N2 is N - 2, N3 is N - 3,  % All distinct
    trib(N1, T1), trib(N2, T2), trib(N3, T3),
    T is T1 + T2 + T3.

% Tree recursion - structural arguments (NOT linear)
tree_sum([V,L,R], S) :-
    tree_sum(L, LS),               % L from pattern match (structural)
    tree_sum(R, RS),               % R from pattern match (structural)
    S is V + LS + RS.
```

**Detection Algorithm:**

```prolog
is_multi_call_linear_recursion(Pred/Arity) :-
    % Has 2+ recursive calls
    count_recursive_calls(Pred/Arity, Count),
    Count >= 2,

    % All recursive call arguments are:
    % 1. Computed via 'is' expressions (scalar, not structural)
    % 2. Single distinct variables (no shared variables across calls)
    all_recursive_args_are_computed_scalars(Pred/Arity),
    all_recursive_args_are_distinct_variables(Pred/Arity),

    % Results aggregated after all calls
    has_pure_aggregation(Pred/Arity).
```

**Benefits:**
- Simpler code generation (reuse existing linear recursion compiler)
- Same or better performance (memoization vs fold)
- More intuitive mapping to mathematical pattern

**Current Workaround:**
Fold pattern works correctly but is more complex than needed.

**Implementation Tasks:**
1. Extend `is_linear_recursive_streamable` to allow 2+ calls
2. Add `has_structural_arguments` check (distinguish from tree recursion)
3. Add `args_are_distinct_variables` check (prove independence)
4. Update tests to verify fibonacci/tribonacci use linear + memo
5. Document the extended pattern in ADVANCED_RECURSION.md

**Estimated Effort:** 8-12 hours

**References:**
- Theory: `docs/RECURSION_PATTERN_THEORY.md`
- Current code: `src/unifyweaver/core/advanced/pattern_matchers.pl`

---

### 17. ✅ COMPLETE: Implement firewall_implies - Higher-Order Firewall Policies

**Status:** ✅ COMPLETE (Verified 2025-11-03)
**Location:** `src/unifyweaver/core/firewall.pl`
**Documentation:** `docs/FIREWALL_GUIDE.md` (Future Enhancements section)
**Created:** 2025-10-19
**Completed:** Already implemented prior to verification

**Implementation Status:**
This feature is **fully implemented and working**. Higher-order firewall rules that derive security policies from other policies using Prolog's logical inference capabilities are already in the codebase.

**Implemented Features:**
1. ✅ **`firewall_implies/2`** - User-defined custom implications (dynamic predicate)
2. ✅ **`firewall_implies_default/2`** - 30+ built-in default implications for common scenarios
3. ✅ **`firewall_implies_disabled/2`** - Mechanism to disable default implications
4. ✅ **`derive_policy/2`** - Policy derivation from conditions
5. ✅ **Full test suite** - `examples/test_firewall_implies.pl` (all tests passing)

**Test Results (2025-11-03):**
```bash
$ swipl -q -l examples/test_firewall_implies.pl -g main -t halt

[Test 1] Default Implications - ✓ PASS
[Test 2] User-Defined Implications - ✓ PASS
[Test 3] Override Default Implications - ✓ PASS
[Test 4] Disable Default Implications - ✓ PASS
[Test 5] Derive Policy from Conditions - ✓ PASS
[Test 6] Complex Multi-Condition Scenarios - ✓ PASS

All Tests Passed ✓
```

**Concept:**
Higher-order firewall rules that derive security policies from other policies using Prolog's logical inference capabilities. This is **extremely difficult or impossible** to implement cleanly in traditional imperative languages, making it a compelling showcase for why Prolog was chosen for UnifyWeaver.

**Example Usage:**

**Basic Implications:**
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

**Preference Chains with Fallback Modes:**
```prolog
% Tool selection - graceful degradation across platforms
firewall_default([
    tool_preferences([
        preferred([jq, python3], mode(fallback)),
        fallback([awk, sed], mode(fallback)),
        denied([bash_eval], mode(exception))  % Never use this
    ])
]).

% When jq is blocked/unavailable, system automatically tries:
% 1. jq (preferred) - fail → fallback
% 2. python3 (preferred) - fail → fallback
% 3. awk (fallback) - fail → fallback
% 4. sed (fallback) - fail → exception (all options exhausted)
% 5. bash_eval - immediate exception (explicitly denied)
```

**Network Rules with Different Modes:**
```prolog
firewall_default([
    % URL blocking - strict security (throw exception)
    network_hosts(['*.typicode.com', '*.github.com'], mode(exception)),

    % SSH port preferences - try alternatives (Termux use case)
    ssh_ports([
        preferred(22, mode(fallback)),      % Standard port
        fallback(2222, mode(fallback)),     % Termux default (non-root accessible)
        fallback(8022, mode(fallback))      % Alternative
    ], mode(exception_if_all_fail))
]).

% Real-world scenario: Termux on Android
% Port 22 blocked (requires root) → try 2222 (Termux default)
% Port 2222 blocked → try 8022
% All ports blocked → throw exception
firewall_implies(
    ssh_connection(Host),
    try_ports([22, 2222, 8022])
) :-
    detect_platform(android).
```

**Tool Selection Across Platforms:**
```prolog
% Minimal systems - degrade gracefully
firewall_default([
    json_processing([
        preferred(jq, mode(fallback)),      % Fast, clean
        fallback(python3, mode(fallback)),  % More powerful
        fallback(awk, mode(warn))           % Always available but warn
    ])
]).

% Air-gapped environments - prefer local/cached
firewall_default([
    data_source([
        preferred(local_cache, mode(fallback)),
        fallback(network, mode(exception))  % Block network in air-gapped
    ])
]).
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

5. **Rule-Level Modes (NEW - 2025-10-24):**
   - Each rule should support its own failure mode
   - Different behaviors for different rule types:
     - **Security rules** (URLs, file access): `mode(exception)` - throw error
     - **Tool preferences**: `mode(fallback)` - try next option
     - **Platform adaptation**: `mode(warn)` - succeed with warning
   - Examples:
     ```prolog
     firewall_default([
         % URL blocking - strict security
         network_hosts(['*.typicode.com', '*.github.com'], mode(exception)),

         % Tool selection - graceful fallback
         tools([
             preferred(jq, mode(fallback)),
             fallback(python3, mode(fallback)),
             denied(eval, mode(exception))
         ]),

         % SSH port selection - try alternatives (Termux use case)
         ssh_ports([
             preferred(22, mode(fallback)),      % Standard port
             fallback(2222, mode(fallback)),     % Termux default (non-root)
             fallback(8022, mode(fallback))      % Alternative
         ], mode(exception_if_all_fail))
     ]).
     ```

6. **Preference Chains vs Hard Blocking:**
   - Most rules should support **preference/fallback chains**
   - System tries options in order until one succeeds
   - Only throw exception when:
     - Explicit `mode(exception)` on the rule
     - All fallback options exhausted
     - Security-critical violation (URL blocking, etc.)
   - This allows graceful degradation across platforms
   - Example use cases:
     - **Android/Termux**: Standard ports blocked → try high-numbered ports
     - **Minimal systems**: `jq` not available → fall back to `python3` → fall back to `awk`
     - **Air-gapped environments**: Network access denied → use cached/local alternatives

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

**Estimated Effort:** 10-15 hours total (updated 2025-10-24)
- Basic implementation: 2-3 hours
- Rule-level modes: 3-4 hours (NEW)
- Preference chain logic: 2-3 hours (NEW)
- Transitive closure: 1-2 hours
- Cycle detection: 1-2 hours
- Testing: 3 hours (expanded for modes)
- Documentation: 3 hours (expanded for preference examples)

**Priority:** High - Excellent showcase of Prolog's unique advantages
- Demonstrates declarative security policies
- Shows graceful cross-platform degradation
- Highlights logical inference capabilities
- Solves real-world problems (Termux SSH ports, minimal systems, air-gapped)

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

### 17. ✅ COMPLETE: Implement Data Source Test Runner Generator

**Status:** ✅ IMPLEMENTED (2025-11-03)
**Location:** `src/unifyweaver/core/advanced/data_source_test_runner.pl`
**Reference:** `examples/test_data_sources.sh` (auto-generated)
**Created:** 2025-10-23
**Completed:** 2025-11-03

**Implementation Summary:**

Implemented a declarative test runner generator for data sources, following the pattern of `test_runner_inference.pl` but adapted for integration testing pipelines.

**Features Implemented:**
1. **Automatic Source Discovery**: Scans `examples/` directory for `.pl` files containing `:- source(...)` declarations
2. **Accurate Prolog Term Reading**: Uses Prolog's term reading instead of regex for robust parsing
3. **Source Type Detection**: Supports CSV, JSON, Python (inline/SQLite), HTTP, and AWK sources
4. **Arity Extraction**: Correctly extracts arity from config (`arity(N)`)
5. **Test Case Inference**: Generates appropriate test cases based on source type
6. **Pipeline Detection**: Identifies multi-stage data pipelines
7. **Test Data Setup**: Auto-generates sample CSV and JSON test data
8. **HTTP Filtering**: Excludes HTTP sources by default (use `include_http(true)` to enable)
9. **Clean Code**: No singleton warnings, deterministic clause matching

**Generated Output:**
- `examples/test_data_sources.sh` - 367 line auto-generated test runner
- Scans 7 example files, found 19 source declarations
- Excludes 3 HTTP sources by default
- Detects 6 pipelines for integration testing

**Usage:**
```prolog
?- use_module(unifyweaver(core/advanced/data_source_test_runner)).
?- generate_data_source_test_runner.
% Generates examples/test_data_sources.sh

% Or with options:
?- generate_data_source_test_runner('custom_path.sh', [
    examples_dir('examples'),
    include_http(true),
    test_data_dir('test_input'),
    output_dir('test_output')
]).
```

```bash
# Run the generated test runner
bash examples/test_data_sources.sh

# With test data preservation
KEEP_TEST_DATA=true bash examples/test_data_sources.sh
```

**Architectural Decisions (from original design questions):**
1. **Module Organization**: `core/advanced/data_source_test_runner.pl` (alongside `test_runner_inference.pl`)
2. **Abstraction Level**: Data-source-aware with extensible clause-based dispatch
3. **Test Discovery**: Hybrid - discovers sources via Prolog term reading, annotates with inferred tests
4. **Metadata Storage**: Extracted from source declarations at generation time
5. **Test Execution Model**: Self-contained bash script (portable, independent)
6. **Reusability**: Specialized for integration tests (complementary to unit test generator)
7. **Relationship to Compiler**: Separate workflow, invoked independently
8. **Extensibility**: Simple - add new `infer_data_source_tests/2` clause for new source types

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

---

## Bookmark Filing: Dual-Objective Model Experiments

**Status:** 📋 FUTURE ENHANCEMENT
**Created:** 2025-12-29
**Priority:** Low (current setup works well)

### Background

The dual-objective scoring system uses two embedding models:
- **Input Objective**: Semantic title matching (currently MiniLM 384-dim)
- **Output Objective**: Structural/functional matching (currently Nomic 768-dim)

MiniLM is a fallback for ModernBERT (requires Transformers 4.48+). Once ModernBERT is available, we should test different model arrangements.

### Model Arrangements to Test

| Arrangement | Input Objective | Output Objective | Notes |
|-------------|-----------------|------------------|-------|
| **Current** | MiniLM (384d) | Nomic (768d) | Fast, working |
| **ModernBERT** | ModernBERT (768d) | Nomic (768d) | Requires TF 4.48+ |
| **Short-context Input** | BGE-small (384d) | Nomic (768d) | Optimized for titles |
| **GTE** | GTE-small (384d) | Nomic (768d) | Strong MTEB scores |
| **Unified** | Nomic (768d) | Nomic (768d) | Simpler, one model |
| **Lightweight** | MiniLM (384d) | MiniLM (384d) | Fastest, ~180MB total |

### Evaluation Criteria

1. **Accuracy**: Hit rate on test queries (Hyphanet → Freenet & Hyphanet)
2. **Speed**: Query embedding time (important for phone)
3. **Size**: Total model size (important for phone)
4. **Disambiguation**: Handling ambiguous queries (wave, uncertainty)

### Test Queries

- "Feynman Lectures" → The Feynman Lectures on Physics (Physics)
- "Hyphanet" → Freenet & Hyphanet (Privacy/Darknet)
- "wave" → Waves (physics) vs Matter Wave (QM)
- "uncertainty" → Uncertainty Principle (QM)
- "neural network" → test disambiguation

### How to Test

```bash
# Regenerate embeddings with alternative model
python3 scripts/generate_dual_embeddings.py \
    --alt-model <MODEL_NAME> \
    --data reports/pearltrees_targets_full_pearls.jsonl \
    --output models/dual_embeddings_test.npz

# Test
python3 scripts/test_dual_objective.py \
    --embeddings models/dual_embeddings_test.npz \
    --query "Hyphanet" --top-k 5
```

### Recommended Priority

1. **ModernBERT** - When Transformers 4.48+ available
2. **BGE-small** - If phone performance is slow
3. **Unified Nomic** - If simplicity is preferred

