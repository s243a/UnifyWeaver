# Refactor: Multifile Dispatch for All Recursion Modules

## Summary

- Extract all R code generators from the 5 remaining core recursion modules into `r_target.pl` using Prolog's multifile predicate mechanism (matching the pattern established for `tree_recursion.pl`)
- Core modules now declare multifile predicates and register only the bash clause; target plugins register their own clauses independently
- Fix `ExitAfterResult` singleton variable warning in `tail_recursion.pl`
- Update `ARCHITECTURE.md` with multifile delegation pattern, complete module tree, and updated future extensions

## Modules Refactored

| Module | Multifile Predicate | R Predicates Moved |
|--------|--------------------|--------------------|
| `tail_recursion.pl` | `compile_tail_pattern/9` | 5 |
| `linear_recursion.pl` | `compile_linear_pattern/8` | 5 |
| `mutual_recursion.pl` | `compile_mutual_pattern/5` | 15 |
| `multicall_linear_recursion.pl` | `compile_multicall_pattern/6` | 3 |
| `direct_multi_call_recursion.pl` | `compile_direct_multicall_pattern/5` | 8 |

## Design

Each core module follows the same pattern:

```prolog
% Core module declares multifile and registers bash clause:
:- multifile compile_<name>_pattern/N.
compile_<name>_pattern(bash, ...) :- generate_bash_code(...).

% r_target.pl registers R clause:
:- multifile <module>:compile_<name>_pattern/N.
<module>:compile_<name>_pattern(r, ...) :- generate_r_code(...).
```

Shared analysis predicates (e.g., `parse_recursive_body/5`, `extract_body_components/5`) are exported from core modules for use by target plugins. R helpers are renamed with module prefixes (`mutual_`, `direct_`, `multicall_`, etc.) to avoid name collisions within `r_target.pl`.

## Test plan

- [x] All 6 Prolog test suites pass (tail, linear, tree, mutual, multicall, direct_multicall)
- [x] Rscript verification: tree_sum→6, even_odd→TRUE, fib_multicall(10)→55, factorial(6)→720, sum_list→15, count_items→3
- [x] Bash verification: even_odd.sh still works
- [x] No regressions — core modules generate identical bash output

🤖 Generated with [Claude Code](https://claude.com/claude-code)
