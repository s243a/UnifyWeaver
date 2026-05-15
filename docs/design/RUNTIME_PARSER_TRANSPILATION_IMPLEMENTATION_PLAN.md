# Runtime Parser Transpilation Implementation Plan

This plan moves runtime Prolog term parsing from target-local parsers toward a
single portable Prolog parser compiled into targets.

## Phase 1: Document The Existing R Contract

R remains the reference target. Its inline parser already supports
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, runtime `op/3`,
and CLI argument parsing.

Keep the R inline parser and tests in place while the compiled parser is being
hardened. The inline parser is useful as an oracle for target behavior, not as
the long-term source of parser semantics.

## Phase 2: Harden The Portable Parser

Keep `src/unifyweaver/core/prolog_term_parser.pl` in the
cross-target-compilable subset. Add tests when target runtime consumers expose
new parser-sensitive behavior.

Maintain structural compile coverage for WAM-R:

```text
tests/test_prolog_term_parser_wam_r_compile.pl
```

This test should continue proving that every parser predicate emits a target
label and wrapper, even before the runtime parser is swapped in.

## Phase 3: Fix Runtime Requirements Exposed By The Parser

The current blocker for running the transpiled parser as the R runtime parser
is WAM-R cut-barrier behavior. The parser uses cuts inside multi-clause helper
predicates such as tokenizer and identifier readers. The runtime must discard
choice points created after the active cut barrier, not merely the most recent
choice point.

Treat this as a general WAM correctness fix. Other non-trivial library code
will eventually depend on the same behavior.

## Phase 4: Swap R Behind A Guard

After cut semantics are correct, wire R so `read/2`,
`read_term_from_atom/2,3`, reverse `term_to_atom/2`, and CLI parsing can use
the compiled parser.

During the transition, keep a selectable fallback to the inline parser. Compare
both paths with:

```text
tests/benchmarks/wam_r_parser_bench.pl
```

The swap is complete when the compiled parser matches the inline parser on
runtime tests and has acceptable performance for normal stream and atom parsing
workloads.

## Phase 5: Generalize The Target Hook

Add a target capability hook for runtime parser inclusion. The hook should
answer whether a generated runtime needs the parser library and how the target
should expose parser entry points to builtins.

The hook should be independent of the WAM items API. A target can skip WAM text
generation at build time and still need runtime source-term parsing.

## Phase 6: Port The Next Target

Choose the next target based on a real runtime consumer, not on parser
availability alone. Good candidates are targets adding:

- `read/2`;
- `read_term_from_atom/2,3`;
- reverse `term_to_atom/2`;
- user-facing CLI term arguments.

Python, R, Elixir, and Lua are reasonable candidates, but R should stay the
semantic reference until the compiled parser fully replaces its inline parser.

## Risks

Parser performance may matter for stream-heavy programs. Benchmarks should run
on stable machines before drawing hard conclusions; Termux phone runs are good
smoke tests but not authoritative performance data.

Compiled parser code size may be noticeable in small generated programs.
Targets should include it only when runtime parsing predicates are reachable or
when the user requests a runtime parser explicitly.

Operator table shape differs by target runtime. The parser contract should stay
data-oriented so each target can adapt its native table into the canonical
`op(Name, Prec, Type)` list without exposing target internals.
