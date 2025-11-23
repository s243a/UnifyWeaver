# feat(python): Add arity 3 tail recursion support (accumulator pattern)

## Summary
Extends Python tail recursion optimization to handle **arity 3 predicates with accumulator patterns**, completing parity with Bash target's tail recursion capabilities.

## What's New

### Arity 3 Accumulator Pattern Support
Tail-recursive predicates with accumulator (like `sum/3`, `product/3`) now generate efficient `while` loops instead of recursion.

**Example - Sum with Accumulator**:
```prolog
sum(0, Acc, Acc).
sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S).
```

**Generated Python**:
```python
def _sum_worker(n, acc):
    # Tail recursion (arity 3) optimized to while loop
    current = n
    result = acc
    
    if current == 0:
        return result
    
    # Iterative loop (tail recursion optimization)
    while current > 0:
        result = result + current  # Extracted from: Acc1 is Acc + N
        current = current - 1
    
    return result
```

### Smart Accumulator Update Extraction
- Detects second `is` expression as accumulator update
- Supports operators: `+`, `*`, `-`
- Handles both `Acc + N` and `N + Acc` argument orders
- Example: `Acc1 is Acc * N` → `result = result * current`

### Updated Wrapper Generation
Extended streaming wrapper to handle arity 3:
- Extracts `input`, `acc`, and `result` from dict
- Defaults accumulator to 0 if not provided
- Returns updated dict with all three values

## Testing

✅ All **11 tests passing** (7 generation + 4 execution):

**New Tests**:
- `tail_recursive_sum` - Verifies while loop generation (not ERROR)
- `sum_execution` - End-to-end test: `sum(5, 0)` = 15

**Existing Tests Still Pass**:
- `factorial_execution` - Linear recursion with memoization (5! = 120)
- All multi-clause and streaming tests

## Comparison: Before vs. After

### Before
```prolog
sum(0, Acc, Acc).
sum(N, Acc, S) :- ... sum(N1, Acc1, S).
```
**Output**: `# ERROR: Tail recursion only supported for arity 2, got arity 3`

### After
**Output**: Efficient while loop (shown above) ✅

## Design Decisions

### Why Arity 3 Matters
Most practical tail-recursive predicates use accumulators:
- `sum_list/3`, `product/3`, `reverse/3`
- Classic functional programming pattern
- Essential for avoiding stack overflow

### Learning from Bash Target
Studied `src/unifyweaver/core/advanced/tail_recursion.pl`:
- `generate_ternary_tail_loop` pattern
- Accumulator position detection
- Step operation extraction

Applied same principles to Python with appropriate idioms.

## Recursion Support Status

| Pattern | Arity 2 | Arity 3 | Status |
|---------|---------|---------|--------|
| **Tail Recursion** | ✅ While loop | ✅ While loop | Complete |
| **Linear Recursion** | ✅ Memoization | ❌ N/A | Supported |
| **Mutual Recursion** | ❌ Future | ❌ Future | Planned |

## Files Modified
- `src/unifyweaver/targets/python_target.pl`:
  - `generate_tail_recursive_code` - Added arity 3 branch
  - `generate_ternary_tail_loop` - New function
  - `extract_accumulator_update` - Smart expression extraction
  - `generate_recursive_wrapper` - Arity 3 support
- `tests/core/test_python_target.pl` - Updated `tail_recursive_sum` test
- `tests/core/test_python_execution.pl` - Added `sum_execution` test

## Next Steps (Future PRs)
1. **Mutual Recursion** - Separate branch (complex, requires SCC detection)
2. **Generator-Based Mode** - Alternative to procedural (C#-style semi-naive)
3. **Transitive Closure** - Specialized optimization for graph queries

## Related PRs
- Builds on PR #71 (Tail Recursion Detection)
- Builds on PR #70 (Multi-Clause & Streaming)
- Builds on PR #68 (Initial Python Target)
