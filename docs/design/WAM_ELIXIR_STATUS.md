# WAM Elixir Target — Status

Living summary of the WAM-Elixir target after the multi-PR parity series.
Captures what has shipped, the current performance state, and the
work that remains deferred.

Companion docs:

- `WAM_ELIXIR_PARITY_PHILOSOPHY.md` — the rationale and ordering for the
  parity work. Read this for *why*.
- `WAM_ELIXIR_GAPS_SPECIFICATION.md` — per-PR phasing and acceptance
  criteria. Read this for the contract each PR landed against.
- `WAM_ELIXIR_CORRECTNESS_GAPS.md` — bugs surfaced by benchmarking the
  pre-parity Elixir target.
- `WAM_ELIXIR_PERF_PHASE_A_PLAN.md` — container/data-structure perf
  baseline work (heap/trail/code as Map/tuple). Orthogonal to feature
  parity but referenced by the perf section below.
- `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` — the cross-target
  ISO-errors contract this target implements.
- `WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` — per-target perf comparison
  numbers; Elixir is now competitive on the chain workload (see
  Performance below).
- [`../WAM_HYBRID_TARGETS_COMPARISON.md`](../WAM_HYBRID_TARGETS_COMPARISON.md)
  — fleet comparison (Elixir is the architectural reference baseline;
  sibling STATUS docs for Haskell/Rust/LLVM/C++/F#).

## Series Overview

17 PRs landed in priority order. All reachable from `main`.

| # | Branch | Merged via | What it added |
|---|---|---|---|
| docs | `docs/wam-elixir-parity` | `e10b88e9` | PHILOSOPHY + SPECIFICATION |
| PR #1 | `feat/wam-elixir-call-n-audit` | PR #2130 | `call/N` meta-call |
| PR #2 | `feat/wam-elixir-catch-throw` | PR #2149 | `catch/3` + `throw/1` |
| PR #3 | `feat/wam-elixir-iso-errors-plumbing` | PR #2158 | ISO-errors plumbing (config, rewrite hook, runtime helpers) |
| PR #4 | `feat/wam-elixir-iso-is` | PR #2169 | `is_iso/2` + `is_lax/2` |
| PR #5 | `feat/wam-elixir-iso-sweep` | PR #2199 | ISO sweep — arith compares, `succ_iso` (IEEE-754 lax deferred) |
| follow-up | `feat/wam-elixir-runtime-parser-validation` | PR #2210 | Runtime parser mode validation |
| follow-up | `feat/wam-elixir-compound-unify` | PR #2206 | Compound-vs-compound unify + structured catch patterns |
| follow-up | `fix/wam-elixir-warning-cleanup` | PR #2216 | Clean lowered-emitter warnings |
| follow-up | `feat/wam-elixir-bagof-setof-basics` | PR #2219 | `bagof/3` + `setof/3` basics via inlining |
| follow-up | `feat/wam-elixir-member-enumerator` | PR #2235 | Cons-cell `./2`/`[|]/2` aliasing + user-source `em/2` for bagof tests |
| compiler | `feat/wam-args-first-default` | PR #2285 | Flip `args_first_emission` default-on after audit (prevents heap-flat nested-compound bugs) |
| perf | `feat/wam-elixir-y-regs-separate-field` | PR #2349 | Y-regs in separate state field — 30-55x bench speedup |
| cross-target | `test/wam-args-first-cross-target-audit` | PR #2464 | args-first bytecode property + Rust/Haskell codegen acceptance tests |
| follow-up | `feat/wam-elixir-witness-group-bagof` | PR #2492 | Witness-group backtracking for `bagof/setof` + empty-retry-body fix |
| follow-up | `feat/wam-elixir-metacall-aggregate-dispatch` | PR #2494 | Meta-call dispatch for `findall/bagof/setof` (removes `inline_bagof_setof` requirement) |
| follow-up | `feat/wam-elixir-switch-on-term` | PR #2535 | `switch_on_term` + `switch_on_constant_fallthrough` — all WAM indexed dispatch |

## What's Implemented

**Meta-call.** `call/N` for atomic and compound goals, with extras (partial-application form).

**Exceptions.** `catch/3` + `throw/1` with both atomic and structured catcher patterns. Side-stack catcher frames (Option A per `WAM_ELIXIR_PARITY_PHILOSOPHY.md` §5); BEAM-native catch (Option B) is deferred.

**ISO errors.** The three-form contract from `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md` is wired:

- Default forms (`is/2`, `>`, `<`, `>=`, `=<`, `=:=`, `=\=`, `succ/2`) are rewritten per the enclosing predicate's ISO mode.
- Explicit `*_iso/N` forms always throw structured ISO errors.
- Explicit `*_lax/N` forms always preserve lax/fail-style behavior.

The rewrite is text-level at project-write time (see `iso_errors_rewrite_text/4` in `wam_elixir_target.pl`). Interpreter-mode predicate modules now parse the rewritten text through the shared WAM-items bridge before emitting Elixir instruction tuples. The lowered emitter still parses WAM line-by-line and has no items intermediate; moving that path to items remains the larger Items API refactor (`WAM_ITEMS_API_SPECIFICATION.md`).

**Aggregates.** `findall/3`, `bagof/3`, `setof/3`, `aggregate_all(Template, Goal, Result)` with all spec-compliant templates (`bag`, `set`, `count`, `sum`, `max`, `min`). Both compiled-inline (`begin_aggregate`/`end_aggregate`) and meta-call dispatch (`execute findall/3` etc.) paths work. Full ISO witness-group backtracking for `bagof/setof` — free-variable bindings are grouped and iterable via `{:agg_next_group, ...}` choice-point frames. The `^/2` existential quantifier is supported.

**Compound unification.** `=/2` recurses through nested compounds; structured `catch` patterns like `error(K, _)` unify against thrown compound terms.

**Member enumeration.** User-source `em/2` (2-clause ISO member) backtracks via `try_me_else`/`retry_me_else`/`trust_me`. The builtin `member/2` remains deterministic (boolean check). Cons-cell representations `./2` and `[|]/2` are treated as aliases in `step_get_structure_ref`.

**Indexed dispatch.** All WAM first-argument and second-argument indexing instructions are lowered: `switch_on_constant`, `switch_on_constant_a2`, `switch_on_constant_fallthrough`, `switch_on_constant_a2_fallthrough`, `switch_on_term`, `switch_on_term_a2`, `switch_on_structure`, `switch_on_structure_a2`. Type-based dispatch for `switch_on_term` distinguishes unbound variables, constants, list cons-cells (`./2`/`[|]/2`), and structures via heap lookup.

**End-to-end coverage.** 50 tests in `tests/test_wam_elixir_classic_programs.pl`:
3 classic programs (fib, ackermann, pythagoras) + 4 call/N + 6 catch/throw + 5 is_iso/is_lax + 14 ISO sweep + 5 compound-eq + 6 bagof/setof + 3 witness/meta-call + 4 em. All 50 pass.

## Performance

**Baseline bench:** `tests/bench_wam_elixir_atom_lookups.pl` — a chain-style `parent/2` + recursive `ancestor/2` workload at N=50..1000. Wallclock per-invocation, captured under `:timer.tc`.

**Profiling driver:** `tests/elixir_e2e/run_eprof.exs` — runs one invocation under BEAM's `:eprof` and prints per-function call counts + cumulative time.

**Y-regs perf finding (PR #2349).** The bench initially showed O(N²) growth driven by `WamRuntime.split_y_regs/1`, which folded the entire register map every time `allocate`/`deallocate` ran (once per env frame). Moving Y-regs to their own `state.y_regs` field — with range-dispatched `put_reg`/`get_reg`/`trail_binding` (≥201 & <300 → y_regs) — eliminated the fold and the O(N²) behaviour.

Before vs after:

| N | Before | After | Speedup |
|---|---|---|---|
| 50  | 3.02 ms | 1.07 ms | 2.8× |
| 250 | 23.83 ms | 1.15 ms | 20.7× |
| 500 | 93.55 ms | 3.47 ms | 27.0× |
| 1000 | 379.65 ms | 6.96 ms | **54.6×** |

Growth shape: O(N²) → ~O(N). Post-fix `:eprof` top consumers are `Map.get/3` (16%), `trail_binding/2` (11%), generated clause functions (8%) — work spread healthily across many real-cost functions, no pathological dominator.

**Why atom interning was deprioritized.** The original Item 1 in the parity plan was atom interning, modeled on the Rust target's 7.9× win on `category_ancestor`. Atom interning addresses string-comparison cost — but the post-fix Elixir profile shows string comparisons at < 1% of runtime. The Rust win came from `HashMap<String, Vec<String>>` alloc/hash overhead, a bottleneck Elixir does not share thanks to BEAM's binary representation. Revisit interning only if a future bench surfaces it as a real cost.

## Architectural Anchors

These choices are load-bearing and worth understanding before working on the target. Full rationale is in `WAM_ELIXIR_PARITY_PHILOSOPHY.md`.

**Heap as Map keyed by integer address.** Compounds are `{:ref, addr}` where `state.heap[addr] = {:str, "name/arity"}` and args at `heap[addr+1..addr+arity]` — contiguous. This contiguity assumption was the source of the `bagof_with_quantifier` bug (interleaved nested-put_structure broke it); the `args_first_emission` flag in the WAM compiler (now default per PR #2285) ensures args are emitted before nested structures.

**Lowered mode is the production path in tests.** Per-predicate
`clause_<Name><N>` functions, no per-instruction fetch loop, BEAM TCO
via tail-calls. Codegen’s unresolved default is still
`emit_mode(interpreter)` (`wam_elixir_resolve_emit_mode/2`); classic
programs / conformance / e2e pass **`emit_mode(lowered)`**
explicitly. Do not confuse “tests use lowered” with “codegen default
is lowered.”

**Side-stack catcher frames.** `catch/3` snapshots regs/y_regs/stack/trail-mark/cp-count into `state.catcher_frames`. `throw/1` raises `{:wam_throw, term, heap, heap_len}` as a BEAM throw — bundling the heap is critical for compound thrown terms (deep_copy creates cells in a state that gets discarded before the throw fires).

**ISO error rewriting is text-level.** `iso_errors_rewrite_text/4` recognizes `builtin_call`/`put_structure`/`call`/`execute` line shapes and substitutes keys per the predicate's ISO mode. Interpreter mode then converts the rewritten text to shared WAM items; lowered mode still consumes the rewritten text directly.

## Deferred Work

In rough priority order:

1. **Atom interning** — deprioritized; see Performance section. Revisit only if a future bench shows it.
2. ~~**Audit Rust/Haskell for the nested-compound bug**~~ — DONE (PR #2464). `args_first_emission` bytecode property verified + Rust/Haskell codegen acceptance tests.
3. ~~**Witness-group backtracking for `bagof/setof`**~~ — DONE (PR #2492). Full ISO grouping with `{:agg_next_group, ...}` iterator CPs. Empty-retry-body fix included.
4. ~~**Meta-call dispatch for `findall`/`bagof`/`setof`**~~ — DONE (PR #2494). Runtime `execute_aggregate_meta/2` + `dispatch_call_meta` integration. `inline_bagof_setof(true)` no longer required.
5. ~~**switch_on_term + switch_on_constant_fallthrough**~~ — DONE (PR #2535). All WAM indexed dispatch instructions lowered. fibonacci and ackermann now pass.
6. **IEEE-754 lax float-divide** — BEAM raises on `1.0/0.0`. Would need atom sentinels (`:inf`, `:nan`) plumbed through the lax arithmetic path.
7. **BEAM-native catch (Option B from PHILOSOPHY §5)** — only if perf measurements show the side-stack walk dominates. Current bench does not exercise it.
8. **Items API for the lowered emitter** — interpreter mode now uses the shared items bridge after text rewrite; the lowered emitter still needs an items-level path before ISO rewriting can move fully out of text. Cross-cutting; not Elixir-specific.

## Lessons Learned

**Bench before implementing.** The Y-regs fix would never have been found without the chain bench + `:eprof`. The plan said "atom interning"; the data said "env-frame folding". Cross-target perf wins do not always transfer — the bottleneck depends on the runtime, not the workload alone.

**Apostrophes in the Prolog runtime template are deadly.** The Elixir runtime is embedded inside Prolog `format(string(...), '...', [])` calls; the format string is delimited by single quotes. Any apostrophe inside body comments (`don't`, `caller's`) terminates the Prolog string early and produces an unhelpful "Operator expected" error several columns past the offender. Rephrase to avoid contractions and possessives in every new comment block.

**args-first emission is the safer default.** The WAM compiler's pre-PR-#2285 nested-compound emission interleaved outer-arg `set_variable` instructions with nested `put_structure` emission, breaking heap contiguity for any target that assumes args follow the functor cell. Defaulting args-first prevents the class of bug prospectively on Rust and Haskell too — even though no existing test exercised the shape.

**The lowered emitter has two copies of `allocate`/`deallocate`.** The runtime template version (`wam_elixir_target.pl`) is used by the interpreter mode; the inline-emitted version (`wam_elixir_lowered_emitter.pl`) is what lowered mode actually runs. Changes to env-frame discipline must touch both.
