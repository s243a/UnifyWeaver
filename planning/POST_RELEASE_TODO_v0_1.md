# Post-Release TODO List (v0.1)

**Created:** 2026-02-17  
**Target Release:** v0.1.x and beyond

This document tracks follow-up work identified after the v0.1.0 release.

---

## Priority 1: C# Target Parity & Runtime Enhancements

- [ ] **Bash/C# feature parity audit**  
  Map existing Bash capabilities (partitioning, constraint handling, recursion patterns, data-source integrations) against the C# query runtime and streaming targets. Produce a checklist and highlight gaps that require runtime or codegen support.

- [ ] **Dynamic facts ingestion**  
  - Detect dynamic predicates in the source program and generate ingestion hooks for the C# runtime.  
  - Design converters that transform piped data (JSON, TSV, null-delimited records, etc.) into in-memory relations (`IEnumerable<(...)>`).  
  - Mirror the Bash behaviour for streaming JSON with null-separated records, extending support to other formats.

- [ ] **Iterable representations**  
  Establish helper classes/utilities in the C# runtime for converting incoming dynamic facts into tuples, arrays, or structs that align with the generated plans.

- [ ] **Pure C# target recursion strategy**  
  Extend `csharp_codegen` to optionally mimic the Bash recursion templates (memoization, BFS loops).  
  Emphasize readability and hackability of generated code, even if the approach is less efficient than the query runtime.

---

## Priority 2: Data Pipeline & Interop

- [ ] **Unified piping conventions**  
  Document and implement consistent conventions for streaming data between components (null separators, field delimiters) so both Bash and C# targets can interoperate.

- [ ] **Type conversion utilities**  
  Provide reusable converters for common data shapes (JSON arrays, CSV rows, key/value streams) targeting both Bash scripts and C# runtime ingestion.

---

## Priority 3: Tooling & Automation

- [ ] **Parity test suite**  
  Create automated tests that compare Bash and C# outputs for the same predicates (including dynamic fact scenarios), ensuring future changes keep the targets aligned.

- [ ] **Generator ergonomics**  
  Explore a configuration switch or preference (`target(csharp_codegen, recursion_strategy(bash_style))`) that chooses the more readable recursion template.

---

## Priority 4: Multi-Call Linear Recursion Completion

**Status:** Pattern detection complete, strategy selection working, direct code generation WIP

**Completed:**
- ✅ Extended `is_linear_recursive_streamable/1` to detect 2+ independent recursive calls
- ✅ Added `recursive_calls_have_distinct_args/2` for independence verification
- ✅ Implemented `get_recursive_call_count/2` and `get_multi_call_info/2` query API
- ✅ Added `recursion_strategy/2` directive for choosing fold vs direct strategies
- ✅ Integrated strategy selection into `advanced_recursive_compiler.pl`
- ✅ Created comprehensive test suite (10 tests passing)
- ✅ Fold-based approach working correctly with memoization

**Remaining Work:**

- [ ] **Debug direct_multi_call_recursion.pl**
  Currently falls back to fold pattern due to silent failures. Need to:
  - Add better error messages to identify where compilation fails
  - Debug `compile_direct_binary_recursion/3` execution path
  - Fix `partition(is_recursive_clause(...))` - predicate not exported
  - Verify base case and recursive case extraction logic

- [ ] **Performance comparison tests**
  Create benchmarks comparing fold-based vs direct recursive strategies:
  - Fibonacci with different input sizes
  - Tribonacci performance characteristics
  - Memory usage comparison
  - Identify which strategy is better for which cases

- [ ] **Documentation updates**
  - Update `ADVANCED_RECURSION.md` with multi-call pattern
  - Document strategy selection mechanism
  - Add examples of when to use fold vs direct
  - Update POST_RELEASE_TODO.md (mark item 16 as complete)

- [ ] **Edge case handling**
  - Test with predicates that have >3 recursive calls
  - Test with mixed scalar/structural arguments
  - Verify behavior with non-numeric base cases

**Reference Files:**
- `src/unifyweaver/core/advanced/pattern_matchers.pl` (lines 155-387)
- `src/unifyweaver/core/advanced/direct_multi_call_recursion.pl` (WIP)
- `examples/test_multicall_fibonacci.pl` (test suite)
- `examples/fibonacci_direct.pl` (example with strategy directive)

---

## Priority 5: Performance & Cross-Platform Optimization

- [ ] **PowerShell integration test performance**
  - Test 3b (integration_test.pl) takes significantly longer in PowerShell compared to native Linux
  - Delays were added to prevent hanging (SIGPIPE, stdin/stdout buffering, subprocess cleanup)
  - Consider investigating whether these delays can be reduced or made conditional
  - Potential areas: Python subprocess management, WSL communication buffering, SQLite locking
  - Trade-off: Current implementation prioritizes stability over speed
  - Source: PowerShell test plan section 3b

- [ ] **Improve PowerShell compatibility layer stdin piping**
  - **Current Issue:** `write_and_execute_bash/3` deadlocks in PowerShell→WSL mode when piping data to stdin
  - **Root Cause:** Buffering issues in SWI-Prolog → PowerShell → WSL → bash pipe chain
  - **Current Workaround:** `integration_test.pl` uses file-based input instead of stdin piping
  - **Locations of workarounds:**
    - `examples/integration_test.pl` line 250-254: Runs saved script instead of re-compiling
    - `examples/integration_test.pl` line 263-275: Uses temp file + bash redirection instead of stdin pipe
  - **Proper Fix Options:**
    1. Improve `bash_executor.pl` to detect PowerShell mode and auto-use file-based stdin
    2. Add timeout/non-blocking reads to prevent deadlocks
    3. Use named pipes (FIFOs) in WSL for better buffering control
    4. Investigate if `flush_output` needs additional sync mechanisms
  - **Current Fix Applied:** Added `flush_output(In)` before closing stdin in `bash_executor.pl:237`
  - **Status:** Workarounds functional but not ideal; proper fix would eliminate test-specific code

---

## Priority 6: Code Quality & Warnings Cleanup

- [ ] **Module export inconsistencies**
  Some source modules don't export predicates that integration tests try to import:
  - `json_source`: missing exports for `validate_config/1`, `source_info/1`, `compile_source/4`
  - `python_source`: same missing exports
  - Determine if these should be public API or if test code should use different approach
  - Source: Integration test warnings (test_env9/examples/integration_test.pl)

- [ ] **Test module import warnings**
  - `test_regressions.pl`: imports `recursive_compiler:is_transitive_closure/5` which is not exported
  - Either export the predicate or remove the import

- [ ] **Legacy code cleanup** (low priority)
  - `legacy/pragmatic_purity.pl`: Multiple singleton variable warnings (Report, Inputs, Args, Mode)
  - `legacy/purity_control.pl`: Singleton Pred variables
  - `legacy/llm_example.pl`: Deprecated newline syntax warnings
  - Consider moving to examples or documenting as deprecated

---

Feel free to add new sections as follow-up items are discovered.
