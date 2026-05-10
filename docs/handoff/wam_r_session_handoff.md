# WAM-R Target Session Handoff

## Context

This document closes out the campaign that built the WAM R hybrid target
from an empty scaffold to feature parity with the Scala / Haskell / Rust
targets on every dimension we set out to cover. The work spans ~30 merged
PRs and lives entirely on `main`; no follow-up branch state is required
to pick up where the campaign left off.

If you're starting a new session and want to extend WAM-R, start here,
then read `docs/WAM_R_TARGET.md` for the user-facing reference.

## What's there now

### Numbers

- **57 tests** in `tests/test_wam_r_generator.pl` (40 e2e via Rscript,
  the rest structural). Full suite runs in ~6-10 minutes; bottleneck is
  Rscript startup, not the runtime.
- **3 main source files**:
  - `src/unifyweaver/targets/wam_r_target.pl` -- codegen + project writer.
  - `src/unifyweaver/targets/wam_r_lowered_emitter.pl` -- Phase-3 native
    R emission (small subset; most lowering work happens via the
    fact-table / kernel paths instead).
  - `templates/targets/r_wam/runtime.R.mustache` -- the runtime, 3000+
    lines of R covering values, registers, unification, choice points,
    builtins, library predicates, parser, kernels, and fact tables.
- **1 example project** at `examples/wam_r_demo/` (genealogy + foreign
  handler + findall).
- **1 benchmark** at `tests/benchmarks/wam_r_fact_source_bench.pl`
  (WAM-compiled vs foreign-handler dispatch).
- **1 design doc** at `docs/WAM_R_TARGET.md` (the user-facing reference).

### Feature catalogue

| Surface | What's in |
|---|---|
| **Control** | `true/0`, `fail/0`, `!/0`, `\+/1` (also `not/1`), `=/2`, `\=/2`, `==/2`, `\==/2`, `@</2`, `@>/2`, `@=</2`, `@>=/2`, `=../2`, `functor/3`, `arg/3`, `copy_term/2`, `compare/3`, `call/1..N`. |
| **Arithmetic** | `is/2`, all six numeric comparisons; `is`-expression operators include trig, log, bitwise, integer ops, `gcd`, `min`/`max`, `^`/`**`. Integer / float, no bigints / rationals. |
| **Type checks** | `var`, `nonvar`, `atom`, `integer`, `float`, `number`, `atomic`, `compound`, `is_list`, `ground`. |
| **Lists** | `member/2` (multi-sol via iter-CP), `memberchk/2`, `length/2` ((+,-) and (-,+)), `append/3` ((+,+,-)), `reverse/2`, `last/2`, `nth0/3`, `nth1/3`, `select/3`, `delete/3`, `permutation/2` (det check + identity), `numlist/3`, `sort/2`, `msort/2`, `maplist/2..3`. |
| **Higher-order / meta** | `call/1..N`, `\+/1`, `once/1`, `forall/2`, `findall/3`, `bagof/3`, `setof/3` (with `^/2` existential), `phrase/2,3`. |
| **Strings / atoms** | `atom_codes`, `atom_chars`, `atom_length`, `atom_concat`, `atom_number`, `atom_string`, `string_to_atom`, `number_codes`, `number_chars`, `string_upper`, `string_lower`, `string_code`, `split_string/4`, `sub_atom/5`, `char_type/2`, `tab/1`, `term_to_atom/2`, `read_term_from_atom/2,3`. |
| **I/O** | `write`, `writeln`, `print`, `nl`, `format/1,2,3`. Stream I/O: `open/3`, `close/1`, `read/2` (line-buffered), `write/2`, `writeln/2`, `nl/1`. |
| **Dynamic** | `assertz/1`, `asserta/1`, `retract/1` (multi-solution), `abolish/1`, `clause/2` (multi-solution introspection). |
| **Exceptions** | `throw/1`, `catch/3` (on R's `tryCatch`). |
| **DCG** | `-->` via SWI's term-expansion + `phrase/2,3`. |
| **Parser** | Operator-precedence parser for the standard Prolog operator table; lists, structs, integers, floats, vars. User-defined infix and prefix operators via runtime `op/3` / `current_op/3` and codegen `r_op_decls(...)`. |
| **Streams** | File I/O via R `file()` connections; line-buffered `read/2` parses through the term parser. |
| **Fact-table lowering** | Pure-fact predicates (only `get_constant + proceed`) emit a flat R tuple list + per-arg hash indexes (one env per arg position), dispatched via `WamRuntime$fact_table_dispatch`. Smallest matching bound-arg bucket wins; O(N · F) memory. |
| **External fact sources (CSV)** | `r_fact_sources([source(P/A, file('data.csv'))])` -- predicate has no Prolog clauses; loader runs at program init. |
| **Kernels (7/7)** | All seven patterns from `recursive_kernel_detection.pl`: `transitive_closure2`, `transitive_distance3`, `weighted_shortest_path3`, `transitive_parent_distance4`, `transitive_step_parent_distance5`, `category_ancestor`, `astar_shortest_path4`. Native R BFS / Dijkstra / A* implementations that bypass the WAM stepper. |
| **Lowered dispatch** | `program$lowered_dispatch` env consulted by `Call` / `Execute` / `dispatch_call` before label dispatch, so internal Prolog-to-Prolog calls reach the fast path (not just direct R-API). |

### Architecture in one paragraph

Tagged R lists for values (`list(tag = "atom" | "int" | "float" | "unbound" | "struct", ...)`); state lives in an R environment for pass-by-reference; A/X registers and Y registers are split (`regs2` for A/X, stack-frame `saved_ys` for Y, restored on `Deallocate`). Choice points are tagged `"iter"` / `"dynamic"` / `"aggregate"` plus the standard variant. Dispatch tier is: lowered-dispatch -> labels -> foreign-handlers -> dynamic-store -> library -> builtin. The `iterate_goal` helper has R-level fast paths for `member/2`, `between/3`, conjunctions, dynamic preds, and lowered-dispatch preds, falling back to a `call_goal + backtrack + run` loop.

## Following the campaign through the PR list

The work landed in roughly this order, with each PR scoped to one user-visible feature plus tests:

1. Initial scaffold + WAM bytecode interpreter.
2. Compound terms, foreign handlers, builtin call dispatch, Phase-2 lowered emitter.
3-13. Standard-library expansion (term inspection, list/atom library, extended arithmetic, negation, meta-call, higher-order, list utilities, I/O, strings, between/3, assert/retract, findall, bagof/setof/once/forall, enumerable member/between, catch/throw + dynamic-aggregator, stdlib polish + term parser).
14. Design doc.
15. Operator-precedence parser.
16. Multi-solution `retract/1` + put_structure cycle fix.
17. `^/2` existential scope for bagof/setof/findall.
18. Structured CLI args + genealogy demo + nested-read fix.
19. Fact-source benchmark + `--bench N` mode.
20. `pred_` prefix on wrappers (footgun fix).
21. `read_term_from_atom/2,3` and multi-solution `clause/2`.
22. DCG support via `phrase/2,3`.
23. File streams + Y-register save/restore on call frames.
24. Fact-table lowering with first-arg hash index.
25-31. Seven recursive kernels (`transitive_closure2`, `transitive_distance3`, `weighted_shortest_path3`, `transitive_parent_distance4`, `transitive_step_parent_distance5`, `category_ancestor`, `astar_shortest_path4`).
32. External fact sources via CSV files.

Each PR has a self-contained description; look up the merge commit in
`git log` if you need the rationale for a specific change.

## Known limitations

These are documented in `docs/WAM_R_TARGET.md` and called out in the
relevant feature sections. Not bugs -- intentional scope boundaries.

- **No bigints / rationals.** Float arithmetic is `double` only.
- **No multi-line term reads.** `read/2` is line-buffered.
- **No streams beyond file I/O.** No `current_input/1`, `current_output/1`,
  `set_input/1`, character I/O, binary streams.
- **Fact-table indexing covers any arg position.** Per-arg hash
  indexes (one env per arg, dispatcher picks smallest bound
  bucket). Storage is O(N · F) -- linear in arity, no `2^N`
  composite-key blowup. Range / interval indexes are still future
  work. Design rationale + alternatives + connection to the
  parameterized C# query runtime are documented in
  [`docs/design/WAM_R_FACT_INDEXING.md`](../design/WAM_R_FACT_INDEXING.md).
- **`bagof` / `setof` per-witness grouping.** `^/2` existential scope
  works; non-quantified free vars are silently aggregated rather than
  producing one bag per witness binding.
- **`length/2` (-,-)** generative mode unsupported.
- **`retract/1` snapshot semantics.** The snapshot is taken at the
  original call; clauses asserted during the iteration are not seen.
- **Postfix `xf` vs `yf` chaining.** Single postfix application is
  correct (e.g. `5!` parses to `'!'(5)`). Chained postfix accepts
  either type without enforcing the ISO `xf` rule that the operand
  precedence be strictly less than the operator precedence. Same
  simplification the existing prefix path already makes (`fx` and
  `fy` are treated identically). Infix/prefix/postfix operators
  work at runtime via `op/3` and at codegen time via `r_op_decls`.
- **`astar_shortest_path4`** is correct only when the user-supplied
  heuristic is admissible. Documented; the fallback (using the edge
  pred itself) is not generally admissible.
- **Mode analysis is not implemented.** Every register is treated as
  potentially unbound at every entry. Affects performance, not
  correctness.

## Follow-up ideas, ranked

If you're picking up this campaign, these are the obvious next steps in
roughly priority order:

1. **LMDB backend** for `r_fact_sources`. Grouped-by-first TSV landed
   alongside this entry (`grouped_by_first('path.tsv')` Spec; arity-2;
   exploded into per-value tuples by `WamRuntime$read_facts_grouped_tsv`,
   then through the same `build_fact_indexes` + `fact_table_dispatch`
   pipeline as the CSV backend). LMDB is harder: needs an R LMDB binding
   (e.g. `thor` / `lmdbr`, both system-LMDB-dependent) and a different
   dispatch path -- the Scala impl deliberately bypasses the
   load-everything `fact_table_dispatch` and probes by `lookupByArg1`
   for ground arg1, `streamAll` otherwise. To extend: add a
   `fact_source_loader_call/4` clause for the new Spec shape **plus** a
   parallel `WamRuntime$lmdb_fact_dispatch` to skip the in-memory tuple
   list, since materialising every key defeats the point.
2. **Performance profiling + targeted optimization.** First PR should
   add `Rprof` instrumentation around `wam_r_fact_source_bench.pl` and
   identify a single hot path; subsequent PRs each fix one bottleneck.
   Tagged-list `$tag` access and env-based register lookups are likely
   suspects.
3. **Inline `WamRuntime$parse_term` swap-in.** The cross-target Prolog
   term parser at `src/unifyweaver/core/prolog_term_parser.pl` now
   runs end-to-end through the WAM-R compiled path -- atoms, integers,
   simple compounds, multi-arg compounds, lists, infix operators with
   correct precedence, postfix, var sharing, nested. Coverage proven
   by `tests/test_prolog_term_parser_wam_r_compile.pl:full_parser_e2e_via_rscript`.
   Per-frame Y storage and the `dispatch_call` sub-call cleanup landed
   alongside the cut-barrier work; nothing in the parser source needs
   to change. The remaining work is wiring `WamRuntime$parse_term`
   (and `term_to_atom/2`, `read_term_from_atom/2,3`, `read/2`, the
   CLI arg parser) to dispatch into the compiled predicate instead of
   the inline R parser. Important: existing op-related tests
   (`op_3_runtime_e2e_rscript`, `op_3_decl_e2e_rscript`,
   `op_3_postfix_e2e_rscript`, `term_to_atom*`) all exercise the
   inline parser today; the swap needs to keep them green. Performance
   Performance is unmeasured but the compiled parser will be slower
   than the inline R version (WAM stepping engine vs. native R
   control flow); the swap is a correctness / cross-target win, not
   a speed win, so benchmark before pulling the trigger and consider
   keeping the inline path for the CLI arg parser if startup latency
   regresses too far.

4. **`CutIte` (soft cut) barrier scope.** `!/0` now uses a proper
   cut barrier (PR #1960); `CutIte` -- emitted for
   `( A -> B ; C )` -- still drops only the most-recent CP. The
   same Allocate-stamps-barrier model would work for soft cut, with
   the barrier saved at the start of A so CutIte truncates back to
   it on commit. Until then, predicates that depend on if-then-else
   commit semantics need to be written with clause-level dispatch
   (see `parse_op_loop`, `parse_args_continue`,
   `tokenize_loop_step`, etc. in the cross-target parser).
5. **Strict `xf` precedence rule** (small). Threading the precedence
   of the current `left` expression through `wam_parse_expr` so `xf`
   and `yf` differ correctly at chained-postfix sites. Fixes the
   simplification documented as a known limitation. Same fix would
   let `fx` / `fy` differ correctly in the prefix path.
6. **Mode analysis (start).** Big multi-PR effort. Phase 1 collects
   mode info per predicate (in/out per arg); Phase 2 wires it to
   specialised emission (skip unifications when the mode says the slot
   is unbound, etc.). Look at the Haskell target's
   `WAM_HASKELL_MODE_ANALYSIS_*.md` design docs for the precedent.
7. **Multi-line `read/2`.** Read until a `.` terminator across lines.
   Niche but completes the streams story.
8. **Multi-solution `retract/1` ignoring snapshot.** I.e. re-read the
   live clause list on each backtrack. Trickier semantics; document
   carefully if pursued.
9. **Phase-3 lowered emitter expansion.** Currently handles only
   `deterministic` and `multi_clause_1` shapes. Could grow N-clause
   general lowering; would compete with the kernel / fact-table paths
   so the value is unclear.
10. **Range / interval indexes** on fact tables. Per-arg hash indexes
    land queries with ground atom/int args; range queries (`X > 5`,
    `X between A B`) still go through the per-tuple scan. A sorted-
    per-arg index would route these. Niche but a natural extension.

## Things to know before you start a follow-up

### The `lowered_dispatch` tier is the hot path

When `Call("p", N)` runs, `dispatch_call` consults `program$lowered_dispatch`
**before** the label dispatch. Anything we want to fast-path -- fact tables,
kernels, future indexed predicates -- registers itself here. Phase-3
lowered functions deliberately do NOT register; their wrappers manage
their own state and the WAM array path is used for nested calls.

### The `pred_` prefix on wrappers is load-bearing

Without it, a Prolog predicate literally named `c/2` shadows `base::c`
and the runtime's own `c(...)` calls crash. PR #1912 fixed this by
prepending `pred_` to every wrapper. Keep it.

### The `Y`-register save/restore on `Allocate` / `Deallocate` is also load-bearing

The flat `regs2` env stores A, X, and Y registers under the same
namespace. Without the save/restore in `Allocate` / `Deallocate`,
nested calls like `rt(F) :- do_write(F), do_read(F).` stomp each
other's permanent vars. PR #1920 fixed this. The fix is conservative
on `Deallocate` -- it restores caller's saved Ys but leaves callee-
added Ys alone, because SWI's WAM emit reads Ys after `Deallocate`
in some cases.

### Module qualifiers in clause bodies

The kernel detector uses `clause/2` to read predicate bodies. Inside
`plunit`, those bodies come back wrapped as `plunit_<test>:user:Goal`.
`wam_r_kernel_detect` strips module qualifiers via
`strip_module_qualifiers/2` before passing clauses to the shared
detector. If you add a new detector or any other code that reads
clause bodies, do the same.

### Tests live or die on actual `Rscript` stdout

Generator-level structural tests have repeatedly missed real runtime
bugs in this target. Every new feature should ship with an
`*_e2e_rscript` test that runs the generated program and asserts on
stdout. If `Rscript` isn't on `PATH`, the tests auto-skip.

### Bench, but with the right bench

The `wam_r_fact_source_bench.pl` measures cold-start vs inner-loop
costs. Cold-start is dominated by Rscript startup (~500ms); inner-loop
amortises over `--bench N` iterations. Per-iter is ~60µs for a chain
fact lookup. If you're proposing a perf win, show the per-iter number
on at least N=100 with `--inner 1000`.

## Closing thought

The R target reaches the kernel-parity bar with the Scala / Haskell /
Rust targets but gives up an order of magnitude in raw runtime speed
because R is interpreted with no JIT and our value representation
(tagged R lists) does an env lookup per deref. That's a known structural
cost. The kernel fast paths (BFS / Dijkstra / A*) close most of the gap
on graph workloads where they apply; the remaining slowness is on
predicates that genuinely exercise the WAM interpreter loop. A
performance campaign focusing on the interpreter (#4 above) is the
single biggest lever left.
