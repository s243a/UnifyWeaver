# Runtime Parser Transpilation Implementation Plan

This plan moves runtime Prolog term parsing from target-local parsers toward a
single portable Prolog parser compiled into targets.

## Phase 1: Document The Existing R Contract

R remains the reference target. Its inline parser already supports
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, runtime `op/3`,
and CLI argument parsing.

Keep the R inline parser and tests in place. The inline parser is both a
runtime implementation and an oracle for target behavior; benchmark data shows
it should remain R's hot path for now.

## Phase 2: Harden The Portable Parser

Keep `src/unifyweaver/core/prolog_term_parser.pl` in the
cross-target-compilable subset. Add tests when target runtime consumers expose
new parser-sensitive behavior.

Maintain compile and runtime coverage for WAM-R:

```text
tests/test_prolog_term_parser_wam_r_compile.pl
```

This test should continue proving that every parser predicate emits a target
label and wrapper and that representative parser drivers run through generated
WAM-R.

## Phase 3: Preserve Runtime Requirements Exposed By The Parser

WAM-R cut-barrier behavior is now part of the parser proof surface. The parser
uses cuts inside multi-clause helper predicates such as tokenizer and
identifier readers, so the runtime must continue discarding choice points
created after the active cut barrier, not merely the most recent choice point.

Treat regressions here as general WAM correctness regressions. Other
non-trivial library code depends on the same behavior.

## Phase 4: Keep R Inline, Use The Compiled Parser As A Guard

Compare the inline parser and compiled parser with:

```text
tests/benchmarks/wam_r_parser_bench.pl
```

Current benchmark data shows the compiled parser is much slower than the inline
R parser, so R should not swap to it by default. A guarded experimental switch
is still useful for equivalence checks and future runtime-performance work, but
`read/2`, `read_term_from_atom/2,3`, reverse `term_to_atom/2`, and CLI parsing
should keep using the inline R parser until the compiled path is both
equivalent and performance-credible.

## Phase 5: Generalize The Target Hook

The target capability hook for runtime parser inclusion lives in
`src/unifyweaver/targets/wam_runtime_parser_capability.pl`:

```prolog
wam_target_runtime_parser(+Target, +Options, -Mode)
```

`Mode` is `none`, `native(Entry)`, or `compiled(prolog_term_parser)`.
`Options` accepts
`runtime_parser(auto|off|native|compiled)`.

The initial implementation is intentionally small:

- the shared predicate and tests cover option resolution;
- R is registered as `native(parse_term)` by default;
- targets without runtime source-term parsing resolve to `none`;
- `compiled(prolog_term_parser)` is available only for targets
  with compile and runtime proof tests.
- the R project writer records the resolved mode in the generated
  `shared_program$runtime_parser` metadata.

Do not immediately wire every target's `write_wam_*_project/3` through the new
hook. The hook is a stable contract; target writers can migrate one at a time.
For R, builtin routing still uses the native inline parser hot path while the
metadata provides a stable seam for later experiments.
When R resolves to `runtime_parser(off)`, the project writer rejects requested
predicates whose statically visible bodies call parser-dependent builtins, so
the disabled mode does not silently generate a runtime that still accepts
source-term parsing calls.

The hook should be independent of the WAM items API. A target can skip WAM text
generation at build time and still need runtime source-term parsing.

## Phase 6: Port The Next Target

Choose the next target based on a real runtime consumer, not on parser
availability alone. Good candidates are targets adding:

- `read/2`;
- `read_term_from_atom/2,3`;
- reverse `term_to_atom/2`;
- user-facing CLI term arguments.

Python, Elixir, Lua, and future R experiments are reasonable candidates, but R
should stay the semantic reference while keeping its inline parser as the
production path.

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
