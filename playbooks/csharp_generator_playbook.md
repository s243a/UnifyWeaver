# C# Generator Mode Playbook (Fibonacci & Derived Columns)

This playbook shows how to use **generator mode** to emit standalone C# with a local fixpoint solver. It pairs with `playbooks/csharp_query_playbook.md`, which is limited to datalog-style joins/filters. Recursive arithmetic (e.g., Fibonacci) works here but not in query mode.

## Prerequisites
- SWI-Prolog with Janus (`library(janus)`).
- .NET SDK in PATH (`dotnet`).
- From the repo root: `src/unifyweaver/targets/csharp_target.pl` available.

## Quick patterns

### Derived column (works in both query & generator)
```prolog
num_pair(1,2).
num_pair(3,4).
sum_pair(X, Y, Sum) :- num_pair(X, Y), Sum is X + Y.
```
Generator call:
```prolog
?- csharp_target:compile_predicate_to_csharp(sum_pair/3, [mode(generator)], Code).
```

### Recursive arithmetic (Fibonacci) — generator OK, query mode will reject
```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
```
Generator call:
```prolog
?- csharp_target:compile_predicate_to_csharp(fib/2, [mode(generator)], Code).
% Write Code to a .cs file and build with dotnet; or use the Janus helper:
?- use_module(library(janus)), py_call(csharp_test_helper:compile_and_run(Code, 'fib_gen'), Result).
```
Expected: `fib` compiles and runs in generator mode. In query mode, it fails because arguments (N1/N2) must be computed before the recursive calls.

## Notes
- Aggregates in generator mode: `aggregate_all/3` (count/sum/min/max/set/bag) and grouped `aggregate_all/4` (sum/min/max/set/bag/count) are supported.
- Indexing defaults ON (per-relation, arg0/arg1 buckets); disable with `enable_indexing(false)` if needed.
- Builtins/negation that reference only bound vars are evaluated early to prune work; order is otherwise preserved.
