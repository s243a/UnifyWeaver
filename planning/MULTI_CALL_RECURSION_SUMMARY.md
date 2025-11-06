# Multi-Call Linear Recursion Implementation Summary

**Date:** 2025-11-04
**Branch:** `multi-call-linear-recursion`
**Status:** Pattern detection complete, infrastructure ready, direct compiler WIP

---

## Overview

Implemented infrastructure for detecting and compiling recursive predicates with 2+ independent recursive calls (e.g., fibonacci, tribonacci). This extends the existing linear recursion pattern to handle multi-call scenarios.

---

## What Was Accomplished

### 1. Pattern Detection Enhancement ✅

**File:** `src/unifyweaver/core/advanced/pattern_matchers.pl`

Extended `is_linear_recursive_streamable/1` to detect and validate multi-call patterns:

```prolog
% Now allows Count >= 1 (was Count =:= 1)
is_linear_recursive_streamable(Pred/Arity) :-
    % ...
    count_recursive_calls(RecBody, Pred, Count),
    Count >= 1,
    % If multiple calls, verify they use distinct argument variables
    (   Count =:= 1
    ->  true
    ;   recursive_calls_have_distinct_args(RecBody, Pred)
    ).
```

**Key Features:**
- Detects 2+ independent recursive calls
- Verifies arguments are computed scalars (via `is` expressions)
- Ensures each call uses distinct variables (N1 ≠ N2)
- Rejects structural decomposition (tree patterns)

**Test Results:**
```
✓ fib/2 detected as linear recursive (2 calls)
✓ trib/2 detected as linear recursive (3 calls)
✓ tree_sum/2 correctly NOT detected (structural args)
```

### 2. Call Count Query API ✅

**New Predicates:**

```prolog
get_recursive_call_count(fib/2, Count).
% Count = 2

get_multi_call_info(fib/2, Info).
% Info = multi_call(2, true, true, true)
%        (Count, IsLinear, HasDistinctArgs, IsPrecomputed)
```

**Usage:** Allows code generators to query recursion characteristics and make informed decisions about compilation strategies.

### 3. Strategy Selection Mechanism ✅

**New Directives:**

```prolog
% In source file:
:- use_module(unifyweaver(core/advanced/pattern_matchers)).
:- recursion_strategy(fib/2, direct).  % or 'fold' or 'auto'

% Query at runtime:
get_recursion_strategy(fib/2, Strategy).
% Strategy = direct
```

**Supported Strategies:**
- `fold` - Build dependency graph, then fold (existing approach)
- `direct` - Generate actual recursive bash functions with memoization (WIP)
- `auto` - Let compiler choose (default: fold)

### 4. Compiler Integration ✅

**File:** `src/unifyweaver/core/advanced/advanced_recursive_compiler.pl`

Modified `try_linear_recursion/3` to check strategy and route appropriately:

```prolog
try_linear_recursion(Pred/Arity, Options, BashCode) :-
    % ...
    get_recursive_call_count(Pred/Arity, CallCount),
    get_recursion_strategy(Pred/Arity, Strategy),
    (   CallCount >= 2, Strategy = direct ->
        compile_direct_multi_call(Pred/Arity, Options, BashCode)
    ;   % Fall back to fold-based approach
        compile_linear_recursion(Pred/Arity, Options, BashCode)
    ).
```

### 5. Comprehensive Test Suite ✅

**File:** `examples/test_multicall_fibonacci.pl`

**10 Tests, All Passing:**
- Fibonacci (2 calls) detected as linear
- Tribonacci (3 calls) detected as linear
- Tree sum correctly NOT detected
- Call count extraction (2, 3, 2)
- Multi-call info structures verified
- Distinct args check working

**Run Tests:**
```bash
swipl -q -l examples/test_multicall_fibonacci.pl -g main -t halt
# All 10 tests pass ✓
```

### 6. Example Files ✅

**Created:**
- `examples/fibonacci_linear.pl` - Clean fibonacci without directives
- `examples/fibonacci_direct.pl` - Demonstrates strategy selection
- Shows how to request direct vs fold compilation

---

## Current Status

### What Works ✅

1. **Pattern Detection:** Fibonacci and tribonacci correctly identified as multi-call linear recursive
2. **Strategy Selection:** Directive system working, preferences stored and queried correctly
3. **Compiler Routing:** Strategy preference checked, routes to appropriate compiler
4. **Fold-Based Compilation:** Existing approach works, generates correct memoized bash code
5. **Test Coverage:** Comprehensive test suite validates all detection logic

### What's WIP ⏳

**Direct Recursive Code Generator** (`direct_multi_call_recursion.pl`):
- Module structure complete
- Variable name conversion helper implemented
- Currently falls back to fold pattern due to silent failures
- Need to debug why `compile_direct_binary_recursion/3` fails

**Known Issues:**
- `partition(is_recursive_clause(...))` fails - predicate not exported
- Base case/recursive case extraction needs verification
- Missing error messages make debugging difficult

---

## Code Generation Comparison

### Fold-Based Approach (Currently Works)

```bash
#!/bin/bash
# Build dependency graph
fib_graph() {
    case "$input" in
        0|1) echo "leaf:$input" ;;
        *)  local n1=$((input - 1))
            local n2=$((input - 2))
            echo "node:$input:[$left,$right]" ;;
    esac
}

# Fold over graph
fold_fib() { ... }

fib() { fib_fold "$@"; }
```

### Direct Recursive Approach (Target, WIP)

```bash
#!/bin/bash
declare -gA fib_memo

fib() {
    # Check memo
    [[ -n "${fib_memo[$n]}" ]] && echo "$n:${fib_memo[$n]}" && return

    # Base cases
    [[ "$n" -eq 0 ]] && fib_memo[0]=0 && echo "0:0" && return
    [[ "$n" -eq 1 ]] && fib_memo[1]=1 && echo "1:1" && return

    # Recursive calls
    local n1=$((n - 1))
    local n2=$((n - 2))
    local f1=$(fib "$n1" | cut -d: -f2)
    local f2=$(fib "$n2" | cut -d: -f2)

    # Aggregate and memoize
    local result=$((f1 + f2))
    fib_memo[$n]=$result
    echo "$n:$result"
}
```

**Benefits of Direct Approach:**
- Simpler, more intuitive code
- More closely mirrors the Prolog source
- Potentially faster (no graph building phase)
- Easier to debug and understand

**Benefits of Fold Approach:**
- Already working and tested
- More functional style
- Separates graph building from computation

---

## File Changes Summary

**Modified Files:**
- `src/unifyweaver/core/advanced/pattern_matchers.pl` (+200 lines)
  - Extended pattern detection
  - Added query API
  - Added strategy selection

- `src/unifyweaver/core/advanced/advanced_recursive_compiler.pl` (+23 lines)
  - Integrated strategy routing

**New Files:**
- `src/unifyweaver/core/advanced/direct_multi_call_recursion.pl` (251 lines)
  - Direct recursive code generator (WIP)

- `examples/test_multicall_fibonacci.pl` (172 lines)
  - Comprehensive test suite

- `examples/fibonacci_linear.pl` (25 lines)
- `examples/fibonacci_direct.pl` (29 lines)
  - Example files demonstrating features

- `planning/POST_RELEASE_TODO_v0_1.md`
  - Documented remaining work

**Total:** 694 lines added across 6 files

---

## Next Steps (Post v0.1)

See `planning/POST_RELEASE_TODO_v0_1.md` Priority 4 for detailed task breakdown.

**High Priority:**
1. Debug direct_multi_call_recursion.pl
2. Add error messages for better debugging
3. Performance benchmarks (fold vs direct)

**Medium Priority:**
4. Documentation updates
5. Edge case testing
6. Mark item 16 in POST_RELEASE_TODO.md as complete

---

## Branch & Commits

**Branch:** `multi-call-linear-recursion`

**Commits:**
1. `c9ce57e` - Initial implementation (694 lines)
2. `148caa9` - WIP bug fixes for variable conversion
3. `2354d1a` - Documentation of remaining work

**Ready for:** Pull request and merge after v0.1 testing

---

## References

- **Theory:** POST_RELEASE_TODO.md item 16 (lines 667-760)
- **Tests:** `examples/test_multicall_fibonacci.pl`
- **Examples:** `examples/fibonacci_direct.pl`
