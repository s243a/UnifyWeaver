# feat(python): Add recursion support with tail optimization

## Summary
This PR adds comprehensive recursion support to the Python target, including both general linear recursion (with memoization) and tail recursion optimization (with iterative `while` loops), learning from the Bash target's implementation.

## Features

### 1. Recursive Predicate Detection
- Automatically detects whether predicates are recursive
- Distinguishes between tail and non-tail recursion patterns
- Routes to appropriate code generation strategy

### 2. Tail Recursion → While Loops
Inspired by Bash target's `tail_recursion.pl`:
- Detects tail-recursive patterns (recursive call in tail position)
- Generates iterative `while` loops instead of recursion
- Avoids Python's recursion limit (~1000 calls)
- **Currently supports arity 2** (binary predicates)

### 3. Linear Recursion → Memoization
For non-tail recursive predicates:
- Generates worker functions with `@functools.cache` decorator
- Extracts arithmetic operators from `is` expressions
- Supports factorial-style patterns (`F is N * F1`)

## Examples

### Linear Recursion (Factorial)
**Prolog**:
```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
```

**Generated Python**:
```python
@functools.cache
def _factorial_worker(arg):
    if arg == 0:
        return 1
    return arg * _factorial_worker(arg - 1)
```

### Tail Recursion (Detection Only - Arity 3 Not Yet Implemented)
**Prolog**:
```prolog
sum(0, Acc, Acc).
sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S).
```

Currently returns ERROR for arity 3. Future work will generate:
```python
def _sum_worker(n, acc):
    while n > 0:
        acc = acc + n
        n = n - 1
    return acc
```

## Testing
✅ All 10 tests passing:
- **7 code generation tests** (`test_python_target.pl`):
  - Existing: module_exports, structure, filter, projection, multiple_clauses
  - New: `recursive_factorial`, `tail_recursive_sum`
- **3 execution tests** (`test_python_execution.pl`):
  - Existing: `streaming_behavior`, `multi_clause_execution`
  - New: `factorial_execution` (verifies 5! = 120)

## Design Decisions

### Why Two Strategies?
1. **Tail recursion → loops**: Eliminates recursion depth limit, more efficient
2. **Non-tail → memoization**: Necessary when work happens AFTER recursive call

### Learning from Bash Target
Studied `src/unifyweaver/core/advanced/tail_recursion.pl` which:
- Detects accumulator patterns
- Generates iterative bash loops
- Handles arity 2 and arity 3

Applied similar pattern detection to Python, starting with arity 2 support.

## Limitations & Future Work
- **Arity 3 tail recursion** not yet implemented (returns ERROR)
- **Mutual recursion** not supported
- **Tree recursion** (multiple recursive calls like Fibonacci) uses memoization only

## Files Modified
- `src/unifyweaver/targets/python_target.pl` - Core recursion logic
- `tests/core/test_python_target.pl` - Added recursion tests
- `tests/core/test_python_execution.pl` - Added factorial execution test

## Related PRs
- Builds on #70 (Multi-clause & Streaming)
- Builds on #68 (Initial Python Target)
