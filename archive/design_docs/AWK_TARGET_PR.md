# Add AWK Target for Fast Text Processing

## Summary

Adds a new AWK compilation target to UnifyWeaver for fast, streaming data processing. AWK is exceptionally fast for processing large text files (millions of lines) with pattern matching, filtering, and aggregation operations.

## Features Implemented

### Phase 1: Core Streaming & Constraints ✅

- **Facts → AWK associative arrays** - O(1) hash-based lookups
- **Single rules with constraints** - Pattern matching with predicates
- **Multiple rules (OR pattern)** - Combines alternative rule bodies
- **All constraint types** - Comparison (`>`, `<`, `>=`, `=<`), equality (`==`, `!=`), inequality (`\=`), arithmetic (`is/2`)
- **Proper variable mapping** - Correct field position mapping with identity checking

### Phase 2: Recursion Patterns ✅

#### 1. Aggregation Operations (Fold/Reduce)
Compiles to efficient AWK END blocks:
- `sum` - Total of values
- `count` - Number of records
- `max` - Maximum value
- `min` - Minimum value
- `avg` - Average value

Usage:
```prolog
compile_predicate_to_awk(total/1, [aggregation(sum)], AwkCode)
```

Generated AWK:
```awk
BEGIN { FS = "\t" }
{ sum += $1 }
END { print sum }
```

#### 2. Tail Recursion → While Loops
Automatically detects tail-recursive patterns and converts to efficient loops:

**Input (Prolog):**
```prolog
factorial(0, Acc, Acc).
factorial(N, Acc, F) :- N > 0, N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, F).
```

**Output (AWK):**
```awk
BEGIN { FS = "\t" }
{
    n = $1; acc = $2
    while (n > 0) {
        new_n = (n - 1)
        new_acc = (acc * n)
        n = new_n
        acc = new_acc
    }
    print acc
}
```

## Documentation

Complete documentation added:

- **[AWK_TARGET_STATUS.md](docs/AWK_TARGET_STATUS.md)** - Implementation status and technical details
- **[AWK_TARGET_EXAMPLES.md](docs/AWK_TARGET_EXAMPLES.md)** - 11 practical examples:
  - Log file analysis
  - Data aggregation
  - Constraint-based filtering
  - Tail-recursive computations
  - CSV/TSV processing
- **[AWK_TARGET_FUTURE_WORK.md](docs/AWK_TARGET_FUTURE_WORK.md)** - Future enhancements with feasibility analysis

## Testing

All features tested with real data:

### Aggregation Tests
```bash
# Input: 10, 20, 30, 40, 50
Sum: 150 ✓
Count: 5 ✓
Max: 50 ✓
Min: 10 ✓
Avg: 30 ✓
```

### Tail Recursion Tests
```bash
factorial(5, 1) = 120 ✓
sum_to_zero(10, 0) = 55 ✓
```

Test files:
- `test_awk_target.pl` - Basic facts
- `test_awk_single_rule.pl` - Single rule with predicates
- `test_awk_multiple_rules.pl` - Multiple rules (OR pattern)
- `test_awk_constraints.pl` - All constraint types
- `test_awk_aggregation.pl` - Aggregation operations
- `test_awk_tail_recursion.pl` - Tail-recursive patterns

## Performance Characteristics

- **O(1) fact lookups** using AWK associative arrays
- **Efficient filtering** with combined hash lookup + constraints
- **Streaming processing** - handles files too large for memory
- **No external dependencies** - self-contained AWK scripts
- **Exceptionally fast** - AWK processes millions of lines per second

## Example Usage

### Compile Prolog to AWK

```prolog
:- use_module('src/unifyweaver/targets/awk_target').

% Define facts
person(alice, 25).
person(bob, 30).

% Define rule
adult(Name, Age) :- person(Name, Age), Age >= 18.

% Compile to AWK
?- awk_target:compile_predicate_to_awk(adult/2, [], AwkCode),
   write(AwkCode).
```

### Run Generated AWK Script

```bash
echo -e "alice\t25\nbob\t17" | awk -f adult.awk
# Output: alice 25
```

## Integration

AWK target is integrated into the recursive compiler dispatcher:

```prolog
:- use_module('src/unifyweaver/core/recursive_compiler').

?- recursive_compiler:compile_to_target(adult/2, awk, [], AwkCode).
```

## Files Changed

- **New:** `src/unifyweaver/targets/awk_target.pl` (~1000 lines)
- **Modified:** `src/unifyweaver/core/recursive_compiler.pl` (AWK dispatcher integration)
- **New:** `docs/AWK_TARGET_STATUS.md`
- **New:** `docs/AWK_TARGET_EXAMPLES.md`
- **New:** `docs/AWK_TARGET_FUTURE_WORK.md`
- **New:** Test files (6 files)

## Limitations (By Design)

AWK target is designed for AWK's strengths. Not supported:
- Deep/unbounded recursion (AWK stack limits)
- Complex nested data structures (AWK is single-level)
- List processing (AWK is line-oriented)
- Mutual recursion (too complex)
- Backtracking (AWK is single-pass)

For these cases, use Python, Bash, or Prolog targets.

## Future Work

See [AWK_TARGET_FUTURE_WORK.md](docs/AWK_TARGET_FUTURE_WORK.md) for detailed feasibility analysis.

**High priority enhancements:**
- Better variable naming in generated code
- String operations (atom_concat, sub_atom, upcase, etc.)
- Regex pattern matching (AWK's strength)
- More aggregation operations (median, stddev, etc.)

## Breaking Changes

None - this is a new target.

## Checklist

- [x] Implementation complete
- [x] All features tested with real data
- [x] Documentation written
- [x] Examples provided
- [x] Integration with recursive_compiler
- [x] Future work documented with feasibility
- [x] No breaking changes

## Branch

`feature/awk-target`

## Commits

1. `e0ffdd8` - Add AWK target for fast text processing
2. `14c92a3` - Fix singleton variable warnings in AWK target
3. `0d4cded` - Implement streaming for single rules in AWK target
4. `bce8e51` - Implement multiple rules (OR pattern) for AWK target
5. `68a69bc` - Fix constraint variable mapping in AWK target
6. `85a8a36` - Fix variable identity checking in constraint mapping
7. `51064a0` - Add Phase 2 recursion patterns for AWK target
8. `303e49b` - Add comprehensive documentation for AWK target

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
