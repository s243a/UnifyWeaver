# Add Aggregation Support to Go Generator Mode

## Summary

Extends the Go generator mode with aggregation support, implementing `aggregate_all/3` (ungrouped) and `aggregate_all/4` (grouped) for fixpoint evaluation. Also adds backward-compatible aliasing from `aggregate/3`.

## What's included

### New Features
- **Ungrouped aggregation** (`aggregate_all/3`): count, sum, min, max, avg
- **Grouped aggregation** (`aggregate_all/4`): aggregate by key column
- **Syntax aliasing**: `aggregate/3` normalized to `aggregate_all/3`
- **Dependency closure**: Extracts inner goal predicates from aggregates

### Files Changed
- `src/unifyweaver/targets/go_target.pl` - Aggregation rule compilation
- `tests/core/test_go_generator_aggregates.pl` - Test suite (3 tests)

## Verification

Grouped sum aggregation produces correct results:

```bash
$ go run dept_total_gen.go
{"relation":"dept_total","args":{"arg0":"eng","arg1":2500}}
{"relation":"dept_total","args":{"arg0":"sales","arg1":2500}}
```

All tests pass.

## Usage

```prolog
% Ungrouped count
item_count(N) :- aggregate_all(count, item(_, _), N).

% Grouped sum
dept_total(Dept, Total) :- aggregate_all(sum(S), salary(Dept, S), Dept, Total).
```

## TODO (Future PRs)

- [ ] Indexing (arg0/arg1 buckets for faster joins)
- [ ] I/O integration (json_input for initial facts)
