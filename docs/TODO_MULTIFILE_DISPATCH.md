# Multifile Dispatch TODO

## Overview

The multifile dispatch pattern allows recursion analysis modules (`core/advanced/tail_recursion.pl`,
`linear_recursion.pl`, `mutual_recursion.pl`) to call target-specific code generators without
coupling. **All 13 targets** now implement this pattern for tail and linear recursion.
Elixir, R, and F# also support mutual recursion dispatch.

## Current State

| Target | Tail Dispatch | Linear Dispatch | Mutual Dispatch | Transitive Closure | Own Recursion Fns |
|--------|:------------:|:--------------:|:--------------:|:-----------------:|:-----------------:|
| **Elixir** | ✅ | ✅ | ✅ | ✅ (recursive_compiler) | ✅ |
| **R** | ✅ | ✅ | ✅ | — | ✅ |
| **F#** | ✅ | ✅ | ✅ | — | ✅ (compile_tail/linear/mutual_recursion_fsharp) |
| **Haskell** | ✅ | ✅ | — | ✅ (target_info) | ✅ (compile_recursion_to_haskell) |
| **Scala** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **Clojure** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **Kotlin** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **Java** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **Jython** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **C** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **C++** | ✅ | ✅ | — | ✅ (recursive_compiler) | — |
| **Ruby** | ✅ | ✅ | — | — | ✅ (can_compile_tail/linear) |
| **Perl** | ✅ | ✅ | — | — | ✅ (can_compile_tail/linear) |

## Tasks

### Completed (all targets)

- [x] **Scala**: Tail (`@tailrec` annotation), Linear (`foldLeft` + `mutable.Map` memoization)
- [x] **Clojure**: Tail (`loop/recur`), Linear (`reduce` + `atom` memoization)
- [x] **Kotlin**: Tail (`tailrec fun`), Linear (`fold` + `mutableMapOf` with `getOrPut`)
- [x] **Haskell**: Tail (`BangPatterns` + strict `!acc`), Linear (`foldl` for numeric)
- [x] **F#**: Tail (`let rec loop`), Linear (`Dictionary<int,int>` memo), Mutual (`let rec ... and ...`)
- [x] **Ruby**: Tail (`each` loop), Linear (`reduce` + `@memo` hash)
- [x] **Perl**: Tail (`for` loop), Linear (`List::Util::reduce` + `%memo` hash)
- [x] **C**: Tail (`for` loop), Linear (static array memoization)
- [x] **C++**: Tail (range-based `for`), Linear (`std::unordered_map` memoization)
- [x] **Java**: Tail (`for-each` loop), Linear (`HashMap<Integer,Integer>` memoization)
- [x] **Jython**: Tail (`for` loop), Linear (`reduce` + `dict` memoization, Python 2/3 compatible)

### Remaining Work

- [ ] Add mutual recursion dispatch to remaining targets (currently only Elixir, R, and F#)
- [ ] Add tree recursion dispatch pattern

## Missing Features by Target

### Elixir (current gaps)
- [ ] Tree recursion pattern
- [ ] Aggregation support
- [ ] `mix.exs` template generation for Jason dependency

### Ruby
- [ ] Transitive closure
- [ ] Mutual recursion
- [ ] Aggregations

### Perl
- [ ] Transitive closure
- [ ] Mutual recursion
- [ ] Aggregations

### Haskell
- [ ] Aggregation support

### C / C++
- [ ] Aggregation support

## Design Notes

The multifile dispatch pattern established by R (PR #753) and adopted by Elixir:

```prolog
% In target_file.pl:
:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(my_target, PredStr, Arity, ..., Code) :-
    % target-specific code generation
    ...
```

This allows `tail_recursion.pl` to stay target-agnostic while each target registers
its own clause. The analyzer calls `compile_tail_pattern(Target, ...)` and the
correct target-specific clause fires via Prolog's multifile dispatch.

All 13 targets now register multifile clauses for tail and linear recursion.
Targets that predate this pattern (Haskell, F#, Ruby, Perl) retain their own
recursion functions alongside the new multifile dispatch clauses.

## Python Family

UnifyWeaver has 10 Python-family targets (CPython, Jython, IronPython, Cython,
MypyC, Numba, Codon, Nuitka, Pyodide, Fuzzy). All variants currently wrap the
core `python_target` and add compile-time decorators or annotations. Several
areas need work:

### Runtime-Specific Package Validation

- [ ] **Package availability checking**: When a predicate uses bindings that
  require specific packages (e.g. `numpy`, `pandas`, `lxml`), the compiler
  should validate the package is available in the selected runtime:
  - IronPython: No C extension packages (no numpy, scipy, etc.)
  - Jython: No C extensions, but can `import java.*`
  - Codon: Only a static subset of Python — no dynamic imports
  - Pyodide: Only packages available in the Pyodide distribution
  - Numba: Only numba-compatible subset within `@jit` functions

- [ ] **Graceful error messages**: When a required package isn't available for
  the selected runtime, emit a clear compile-time error explaining why and
  suggesting an alternative runtime

### Code Generation Differences

- [ ] **IronPython**: Currently generates identical code to CPython. Should
  generate `.NET`-aware code where beneficial:
  - Use `clr.AddReference()` for .NET assembly access
  - Use `System.Collections.Generic` instead of Python collections where
    it improves .NET interop
  - Flag C-extension imports as errors at compile time

- [ ] **Codon**: Validate that generated code uses only the Codon-compatible
  subset — no `eval()`, no dynamic attribute access, restricted `import`

- [ ] **Numba**: Validate that `@jit`-decorated functions use only
  numba-supported types and operations

### Variant-Specific Optimizations

- [ ] **Cython**: Generate `cdef` type declarations for known-typed variables
- [ ] **MypyC**: Generate strict type annotations from Prolog type analysis
- [ ] **Numba**: Auto-detect vectorizable operations and use `@vectorize`
- [ ] **Codon**: Use `@par` for parallelizable loops
- [ ] **Nuitka**: Generate `--standalone` build scripts alongside the code
- [ ] **Pyodide**: Generate `micropip.install()` calls for required packages
