# PR: Add multifile dispatch for tail/linear recursion to all targets

**Title:** `feat: add multifile dispatch for tail and linear recursion to all 11 remaining targets`

---

## Summary

- Adds `compile_tail_pattern/9` and `compile_linear_pattern/8` multifile clauses to all 11 targets that were missing them: F#, Haskell, Ruby, Perl, Scala, Kotlin, Java, Clojure, C++, C, and Jython
- F# additionally gets `compile_mutual_pattern/5` (mutual recursion with `let rec ... and ...` blocks)
- All 13 targets now support the multifile dispatch pattern, enabling the core analyzer to dispatch target-specific code generation without coupling
- Updates `docs/TODO_MULTIFILE_DISPATCH.md` to reflect completed status

## Idiomatic Code Generation per Target

| Target | Tail Recursion | Linear Recursion (with memoization) |
|--------|---------------|-------------------------------------|
| **Scala** | `@tailrec def loop(...)` | `foldLeft` + `mutable.Map` |
| **Kotlin** | `tailrec fun` | `fold` + `mutableMapOf` with `getOrPut` |
| **Java** | `for (int item : items)` | `HashMap<Integer, Integer>` |
| **Clojure** | `(loop [... ] (recur ...))` | `(reduce ...)` + `(atom {})` with `swap!` |
| **Haskell** | `{-# LANGUAGE BangPatterns #-}` + `!acc` | `foldl` |
| **F#** | `let rec loop` | `Dictionary<int,int>` memo |
| **Ruby** | `.each` loop | `.reduce` + `@memo` hash |
| **Perl** | `for my $item (@$items_ref)` | `List::Util::reduce` + `%memo` |
| **C** | `for` loop | `static int memo[MAX_MEMO]` |
| **C++** | range-based `for` | `std::unordered_map<int,int>` |
| **Jython** | `for item in items:` | `reduce()` + `dict` (Python 2/3 compatible) |

## How It Works

Each target file registers multifile clauses with the core analysis modules:

```prolog
:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(scala, PredStr, Arity, ..., Code) :-
    % Scala-specific code generation with @tailrec annotation
    ...
```

When the analyzer detects a tail or linear recursion pattern and a `target(T)` option is set, it calls `compile_tail_pattern(T, ...)` and Prolog's multifile dispatch routes to the correct target-specific clause.

## Test Plan

- [x] 26/26 Prolog compilation tests pass (13 targets x 2 patterns)
- [x] 26/26 end-to-end execution tests pass across all 13 targets:
  - Termux: R, Perl, C, C++, Python/Jython, Ruby, Java, Kotlin
  - proot debian: Haskell (ghc 9.0.2), F# (dotnet 9), Scala (scalac 3.7.4), Clojure, Elixir
- [x] All targets load cleanly with no discontiguous or undefined predicate warnings
- [x] Existing target functionality unaffected (multifile clauses are additive)

## Files Changed (12 files, +2,379 / -63)

- `src/unifyweaver/targets/fsharp_target.pl` — tail + linear + mutual dispatch (+420)
- `src/unifyweaver/targets/cpp_target.pl` — tail + linear dispatch (+212)
- `src/unifyweaver/targets/perl_target.pl` — tail + linear dispatch (+209)
- `src/unifyweaver/targets/c_target.pl` — tail + linear dispatch (+208)
- `src/unifyweaver/targets/java_target.pl` — tail + linear dispatch (+208)
- `src/unifyweaver/targets/ruby_target.pl` — tail + linear dispatch (+193)
- `src/unifyweaver/targets/haskell_target.pl` — tail + linear dispatch (+192)
- `src/unifyweaver/targets/jython_target.pl` — tail + linear dispatch (+186)
- `src/unifyweaver/targets/scala_target.pl` — tail + linear dispatch (+185)
- `src/unifyweaver/targets/kotlin_target.pl` — tail + linear dispatch (+171)
- `src/unifyweaver/targets/clojure_target.pl` — tail + linear dispatch (+164)
- `docs/TODO_MULTIFILE_DISPATCH.md` — updated status table and task list
