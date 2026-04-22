# WAM Tiered Lowering and Purity-Driven Routing

## Summary

UnifyWeaver's WAM-hybrid targets are not uniform machines. Each target
declares a **non-determinism strategy menu**, and the emitter selects
per predicate based on purity, arity, and clause count. WAM
interpretation is the **lowest tier**, not the reference architecture.

Under this framing, the purity certificate produced by
`src/unifyweaver/core/purity_certificate.pl` is not just an advisory
hint for specific optimizations — it is the **routing signal** that
determines which tier a predicate lands in across every target that
has a tier menu.

## Motivation

The original WAM-hybrid design treated each target as having two
possible fates for any given predicate: "lowered" (target-native code)
or "WAM interpreted" (fallback). The intra-query parallelism work on
Haskell (see `WAM_HASKELL_INTRA_QUERY_SPEC.md`) exposed a third
possibility: **emit native WAM but annotate choice points as parallel**
(`ParTryMeElse`), gated on a purity certificate.

Generalising: every target with a rich host language has a **strategy
menu**, and the interesting question is not "lowered or WAM?" but
"which strategy within this target's menu fits this predicate?"

| Target  | Current non-determinism strategy                     |
| ------- | ---------------------------------------------------- |
| Haskell | `async`/`forkOrSequential` on `ParTryMeElse` (pure)   |
| F#      | TPL `Parallel.map` stubs (sequential fallback)        |
| WAT/WASM| `$backtrack` dispatch loop in linear memory           |
| Go      | Clause-1 lowering; clause-2+ interpreter              |
| Python  | Interpreter loop with `_backtrack()`                  |
| Rust    | Interpreter with choice-point stack; Rayon unused     |
| Elixir  | CPS with `throw({:fail, state})` / `catch` chain      |

No two targets use the same non-determinism mechanism. That's not an
accident — each target exploits its language's strengths differently.

## Three-tier menu (template)

A target's non-determinism strategy menu typically has these layers.
Not every target implements every tier; a target that only offers
tier 3 is still internally consistent.

### Tier 1 — Pure functional lowering

Facts and fully-deterministic predicates compile to host-idiomatic
data + lookup code. No WAM machinery. No backtracking surface.

- **Elixir**: `inline_data` module attribute + `stream_facts/3` over a
  list (plus optional `@facts_by_arg1` index). Landed Phases A–E.
- **Haskell**: top-level `IntMap` or literal list consumed by the FFI
  kernel path. Mostly landed.
- **Rust / Go**: function table / map over fact literals. Partial.

Precondition: predicate is fact-only (or deterministic-with-cut).

### Tier 2 — Host-native parallel search

Pure, multi-clause, potentially non-deterministic predicates compile
to concurrent host-native code. Alternative clauses run in parallel;
the first successful result wins (for `once`-style queries) or all are
collected (for `findall`-style).

- **Haskell**: `forkOrSequential` ↔ `async`/`waitAny`/`cancel`. Landed
  via `ParTryMeElse` emission (P4 delivered 2026-04-15).
- **Elixir**: **not yet implemented.** The natural expression is
  `Task.async_stream` over clause-alternative continuations.
  BEAM processes are cheaper than OS threads, have no shared heap
  (data races structurally eliminated), and let OTP supervisors absorb
  individual-branch failure. Hook point: a new `par_wrap_segment/3`
  alongside the existing `wrap_segment/3` in
  `wam_elixir_lowered_emitter.pl`, selected when purity gate passes.
- **F#**: TPL `Parallel.map` stubs are already scaffolded.
- **Rust**: Rayon parallel iterators over choice points (future).

Precondition: purity certificate with `Verdict = pure` and
`Confidence >= 0.85`. The same threshold Haskell uses.

### Tier 3 — WAM interpretation / CPS fallback

Everything else — impure predicates, cut-heavy bodies, side-effecting
goals, predicates whose certificate is `unknown` or `impure`. Each
target has its own implementation:

- **Elixir**: lowered CPS with `try/catch` around each segment,
  continuation functions per non-tail `call`. Landed.
- **Haskell**: native `step`/`backtrack` WAM interpreter. Landed.
- **Go, Python, Rust**: instruction-array interpreter with an explicit
  choice-point stack. Varies by target.

Tier 3 is always correct and complete. The parallel tiers are
correctness-preserving optimizations layered on top.

## Purity certificate as routing signal

`purity_certificate.pl` produces:

```prolog
purity_cert(Verdict, Proof, Confidence, Reasons)
```

Each target's tier selector consults this certificate:

```
if Verdict == pure and Confidence >= 0.85 and (tier 2 exists for target):
    emit Tier 2
elif Verdict == pure and Verdict is not probe-demanding:
    prefer Tier 1  (fact-shape layout decisions)
else:
    emit Tier 3
```

The Haskell target has already wired this for Tier 2 (see
`PURITY_CERTIFICATE_IMPLEMENTATION_PLAN.md` Phase P4). Every other
target with a Tier 2 option in its menu needs the same consumer.

Tier 1 consumers (e.g., the Elixir `cost_aware` policy in
`wam_elixir_lowered_emitter.pl`) can *also* read the certificate — a
pure+probe-friendly predicate might bias toward `inline_data_indexed`
when call-frequency data appears, but this is a nice-to-have, not a
correctness gate.

## Target-specific appendix: Elixir Tier 2

### Current state

- Tier 1 (`inline_data`, `@facts_by_arg1`) — landed (Phases A–E).
- Tier 3 (CPS / `try-catch` chain) — landed; default for all
  non-Tier-1 predicates.
- Tier 2 — **missing.** This is the natural next step for the Elixir
  WAM target, conditional on purity certificates being consumed here.

### Hook point

`wam_elixir_lowered_emitter.pl` exports `wrap_segment/3` which wraps a
clause segment body with its choice-point push and `try/catch` block.
The emission decision for a segment happens at a single call site.

A Tier-2 variant `par_wrap_segment/3` would:

1. Check `purity_certificate:analyze_predicate_purity/2` for the
   current predicate.
2. If `Verdict = pure` ∧ `Confidence >= 0.85` ∧
   `option(intra_query_parallel(false), Options)` is absent, emit
   `Task.async_stream`-based code instead of `try/catch`.
3. Fall through to `wrap_segment/3` otherwise (current behaviour).

### Shape of generated Tier-2 code

Roughly (exact shape subject to prototype):

```elixir
defp clause_main(state) do
  branches = [&clause_main_impl/1, &clause_k1_impl/1, &clause_k2_impl/1]

  branches
  |> Task.async_stream(& &1.(state),
                       on_timeout: :kill_task,
                       ordered: false,
                       max_concurrency: System.schedulers_online())
  |> Enum.reduce_while(:fail, fn
    {:ok, {:ok, _} = ok}, _acc -> {:halt, ok}
    _, acc -> {:cont, acc}
  end)
end
```

`clause_main_impl/1`, `clause_k1_impl/1`, etc. are the
currently-generated per-clause `defp`s, unchanged. The outer wrapper
swaps from sequential `try/catch` to parallel fan-out + first-win.

### Why BEAM processes specifically

- Cheaper than OS threads; millions per node are routine.
- No shared heap → data races structurally impossible; the WAM
  `state` is copied per branch by value (immutable Elixir structs).
- A failed branch is literally a dead BEAM process — maps naturally
  onto Prolog's "this alternative didn't unify, try the next."
- OTP supervision can absorb a crash in one branch without nuking
  the whole query.

### Preconditions

The following must hold before Tier 2 is a reasonable next PR:

1. A concrete Elixir-side consumer of `purity_certificate.pl` exists
   (even if only the wrapper predicate). Any target that wants Tier 2
   needs to wire the certificate lookup.
2. The CPS throw/catch semantics interacting with `Task.async_stream`
   need verification: a branch that throws `{:fail, state}` must be
   captured as "this alternative failed" without killing sibling
   tasks. The `{:exit, reason}` vs `{:ok, value}` dichotomy of
   `async_stream` maps onto this cleanly but needs testing.
3. Cut (`!`) semantics under parallel evaluation need a design pass —
   the Haskell Tier-2 gates on purity which typically excludes cuts,
   but pure-with-green-cut is a valid niche. Probably: if any clause
   contains `!`, fall back to Tier 3.

### Deferred

- **Measurement.** Whether Tier 2 actually pays on realistic Elixir
  workloads requires runtime profiling — desktop-environment work.
- **Cost-model knob for "worth parallelizing."** A 3-clause predicate
  probed once doesn't need parallelism. Threshold goes into the same
  cost-aware policy hook as existing layout decisions.

## Relation to existing UnifyWeaver documents

- `WAM_FACT_SHAPE_PLAN.md` (Elixir fact-shape work): handles Tier 1 in
  full for Elixir. Cross-references the Haskell fact-access trilogy.
- `WAM_HASKELL_FACT_ACCESS_{PHILOSOPHY,SPEC,PLAN}.md`: Haskell Tier 1
  + pieces of Tier 2 via the FFI kernel path.
- `WAM_HASKELL_INTRA_QUERY_SPEC.md`: Haskell Tier 2 design.
- `PURITY_CERTIFICATE_{PROPOSAL,SPECIFICATION,IMPLEMENTATION_PLAN}.md`:
  the routing signal this doc rests on. Phase P4 delivered the first
  Tier-2 consumer (Haskell).
- `PREPROCESSED_PREDICATE_ARTIFACTS.md`: the C# side's parallel
  direction for externally-built Tier-1 artifacts.

## Open questions

- **Cross-target Tier-2 test harness.** Once two targets implement
  Tier 2, a shared test that verifies pure-parallel vs pure-sequential
  emission produces identical multisets of solutions would catch
  accidental divergence.
- **Where the tier decision lives.** Currently scattered — each
  target's emitter has its own policy. A shared `layout_policy/6`
  hook like Elixir's Phase E might generalize across tiers, not just
  within Tier 1.
- **Should Tier 2 gate on cost?** A 2-clause predicate called 10
  times per query probably doesn't benefit from spinning up BEAM
  processes. The cost-aware policy hook is the natural home for this,
  but we lack runtime data to calibrate.

## Why this matters

Two reasons to capture this now, even without an implementation
landing:

1. **The purity certificate work now has a clear cross-target
   purpose.** It's not just "Haskell's parallelism toggle" — it's the
   gate for every future Tier-2 implementation. This reframes the
   certificate work as central infrastructure, not a Haskell-specific
   optimization.
2. **The Elixir Tier-2 gap is concrete and defensible.** The earlier
   framing ("Elixir has no parallel-emission surface") conflated the
   current implementation with a language constraint. BEAM processes
   + `Task.async_stream` + a `par_wrap_segment` emitter hook is a
   real, well-shaped future PR — not a speculative architectural
   fantasy.

Neither reason justifies dropping everything to implement Tier 2
today. Both justify having the design recorded so the next engineer
arriving fresh doesn't have to rediscover it.
