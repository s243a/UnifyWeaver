# Multifile Dispatch TODO

## Overview

The multifile dispatch pattern allows recursion analysis modules (`core/advanced/tail_recursion.pl`,
`linear_recursion.pl`, `mutual_recursion.pl`) to call target-specific code generators without
coupling. Currently, only **Elixir** and **R** implement this pattern. All other targets use
either their own recursion functions or rely solely on `recursive_compiler.pl` for transitive closure.

## Current State

| Target | Tail Dispatch | Linear Dispatch | Mutual Dispatch | Transitive Closure | Own Recursion Fns |
|--------|:------------:|:--------------:|:--------------:|:-----------------:|:-----------------:|
| **Elixir** | ✅ | ✅ | ✅ | ✅ (recursive_compiler) | ✅ |
| **R** | ✅ | ✅ | ✅ | — | ✅ |
| **Haskell** | — | — | — | ✅ (target_info) | ✅ (compile_recursion_to_haskell) |
| **F#** | — | — | — | — | ✅ (compile_tail/linear/mutual_recursion_fsharp) |
| **Scala** | — | — | — | ✅ (recursive_compiler) | — |
| **Clojure** | — | — | — | ✅ (recursive_compiler) | — |
| **Kotlin** | — | — | — | ✅ (recursive_compiler) | — |
| **Java** | — | — | — | ✅ (recursive_compiler) | — |
| **Jython** | — | — | — | ✅ (recursive_compiler) | — |
| **C** | — | — | — | ✅ (recursive_compiler) | — |
| **C++** | — | — | — | ✅ (recursive_compiler) | — |
| **Ruby** | — | — | — | — | ✅ (can_compile_tail/linear) |
| **Perl** | — | — | — | — | ✅ (can_compile_tail/linear) |

## Tasks

### High Priority

- [ ] **Scala**: Add multifile dispatch clauses for tail, linear, and mutual recursion
  - Has transitive closure in recursive_compiler.pl already
  - No multifile registration for compile_tail_pattern/9, compile_linear_pattern/8, compile_mutual_pattern/5
  - Would benefit from Scala 3 match types and tail recursion annotation

- [ ] **Clojure**: Add multifile dispatch clauses for tail, linear, and mutual recursion
  - Has transitive closure in recursive_compiler.pl already
  - Should generate `loop/recur` for tail recursion (Clojure doesn't have TCO without it)
  - Mutual recursion should use `declare` + `defn` pattern

- [ ] **Kotlin**: Add multifile dispatch clauses for tail, linear, and mutual recursion
  - Has transitive closure in recursive_compiler.pl already
  - Should use `tailrec` annotation for tail recursion
  - Could use sealed classes for mutual recursion

### Medium Priority

- [ ] **Haskell**: Register multifile dispatch clauses
  - Already has `compile_recursion_to_haskell/3` but uses its own dispatch, not multifile
  - Should register with tail_recursion, linear_recursion, mutual_recursion modules
  - Would use `BangPatterns` for tail recursion

- [ ] **F#**: Register multifile dispatch clauses
  - Already has `compile_tail_recursion_fsharp/3` etc. but not registered as multifile
  - Should register with the core/advanced modules for consistency

- [ ] **Ruby**: Add transitive closure and mutual recursion
  - Has tail/linear recursion detection but no transitive closure in recursive_compiler.pl
  - No mutual recursion support

- [ ] **Perl**: Add transitive closure and mutual recursion
  - Same situation as Ruby
  - Perl's hash-based data structures are well-suited for BFS transitive closure

### Low Priority

- [ ] **C/C++**: Add multifile dispatch clauses
  - Already have transitive closure in recursive_compiler.pl
  - No multifile registration for tail/linear/mutual
  - C would need iterative rewrites (no TCO guarantee without compiler flags)

- [ ] **Java**: Add multifile dispatch clauses
  - Has transitive closure only
  - Should use while-loops for tail recursion (JVM doesn't guarantee TCO)

- [ ] **Jython**: Add multifile dispatch clauses
  - Same as Java but generating Python syntax for JVM execution

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

Targets that predate this pattern (Haskell, F#, Ruby, Perl) have their own
recursion functions that work but aren't integrated with the core analysis pipeline.
Migrating them would improve consistency and allow the analyzer to automatically
select the right recursion strategy.

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
