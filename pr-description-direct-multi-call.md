# Fix: Implement and Fix `direct_multi_call_recursion` Module

## Summary

- Fix 5 bugs in `direct_multi_call_recursion.pl` that prevented the module from working: missing predicates, broken detection logic, findall variable-copy bug in aggregation detection, singleton variable, and SWI-Prolog `N-1` → `N+(-1)` clause storage mismatch
- Fix 3 bugs in the R target's direct multicall section: hardcoded memo env name, missing negative-K expression handling, and invalid R identifiers starting with `_`
- Integrate the module into the `advanced_recursive_compiler.pl` pipeline as a new `try_direct_multi_call` step
- Update `ARCHITECTURE.md` with full compilation priority chain

## Before

```
$ swipl -g "direct_multi_call_recursion:test_direct_multi_call" -t halt
Test 1: Detect fibonacci as direct multi-call
  ✗ FAIL - should detect fibonacci
Test 2: Compile fibonacci to bash
  ✗ FAIL - bash compilation failed
Test 3: Compile fibonacci to R
  ✗ FAIL - R compilation failed
```

## After

```
$ swipl -g "direct_multi_call_recursion:test_direct_multi_call" -t halt
Test 1: Detect fibonacci as direct multi-call
  ✓ PASS - fibonacci detected
Test 2: Compile fibonacci to bash
  ✓ Compiled to output/advanced/fib_direct.sh (bash)
Test 3: Compile fibonacci to R
  ✓ Compiled to output/advanced/fib_direct.R (r)

$ Rscript output/advanced/fib_direct.R 10
55

$ Rscript output/advanced/fib_direct.R 20
6765
```

## Test plan

- [x] `fib_direct.R 0` → 0
- [x] `fib_direct.R 1` → 1
- [x] `fib_direct.R 7` → 13
- [x] `fib_direct.R 10` → 55
- [x] `fib_direct.R 20` → 6765
- [x] Existing scripts unaffected: `factorial.R 6` → 720, `sum_list.R 1,2,3,4,5` → 15

🤖 Generated with [Claude Code](https://claude.com/claude-code)
