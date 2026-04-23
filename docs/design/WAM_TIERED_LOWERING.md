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

A Tier-2 variant `par_wrap_segment/3` gates on three static conditions
derivable at emit time, plus one **runtime** condition baked into the
emitted wrapper:

```prolog
par_wrap_segment(Pred/Arity, Segments, Options) :-
    \+ option(intra_query_parallel(false), Options),
    purity_certificate:analyze_predicate_purity(
        Pred/Arity, purity_cert(pure, _, Conf, _)),
    Conf >= 0.85,
    length(Segments, N), N >= 3,          % min-branches cost gate
    !,
    emit_par_tier2_wrapper(Pred/Arity, Segments, Options).
par_wrap_segment(Pred/Arity, Segments, Options) :-
    wrap_segment(Pred/Arity, Segments, Options).   % Tier 3 fallback.
```

The static gates at emit time:

- Kill-switch (`intra_query_parallel(false)`) absent.
- Purity confidence ≥0.85 (matches Haskell).
- Clause count ≥3 (min-branches cost gate).

No `contains_cut/1` static check — cut safety comes from the runtime
barrier pinned in the emitted wrapper (see "Cut semantics" below).

The runtime gate the emitted wrapper performs on every call:

- **Forkable aggregate frame present.** Unless the current
  `state.choice_points` has an aggregate-frame marker on it (from a
  surrounding `findall` / `bagof` / `aggregate_all` / `sum`), the
  wrapper falls back to sequential `wrap_segment` behaviour.

See "Aggregate-frame gate" below for why this is a correctness
requirement, not an optimisation.

### Cost gate: `forkMinBranches = 3`

Matches the Haskell target's identical threshold. Rationale:
`Task.async_stream` has non-trivial setup cost per task. A 2-clause
predicate where one branch is trivial is reliably slower with
parallelism than without. Three branches is the minimum at which
amortisation starts to pay.

This is a conservative **static** cost hint — the same pattern the
C# query runtime uses in
`docs/proposals/PREPROCESSED_PREDICATE_ARTIFACTS.md`: don't wait for
measured data before making a sensible default. Measurement-driven
threshold tuning plugs into the existing `cost_aware` policy hook
(PR #1559) when desktop-environment profiling lands.

### Aggregate-frame gate (correctness, not optimisation)

Tier 2 only forks when the calling context is a forkable aggregate
frame — `findall`, `bagof`, `aggregate_all`, `sum`, `set`. Outside
such a frame, the emitted wrapper falls back to sequential
`wrap_segment` behaviour. Matches Haskell's
`currentAggMergeStrategy` check in `forkOrSequential` exactly.

This is **not an optimisation** — it is the correctness requirement
that keeps `next_solution/1` enumeration working. Consider a
≥3-clause pure predicate `parent/2`:

- Called via `parent("a", Y)` + `next_solution/1` loop: no aggregate
  frame on the CP stack → sequential `try_me_else` chain → every
  solution is reachable by backtracking, same as today.
- Called inside `findall(Y, parent("a", Y), Ys)`: aggregate frame
  present → branches run in parallel, each runs to full exhaustion,
  results merged by the aggregate. Caller gets the same solution
  multiset a sequential implementation would produce.

Without the gate, a direct-enumeration caller would hit
`reduce_while + halt` first-win semantics and silently lose
non-winning solutions. **That bug is the reason this gate is a
precondition, not deferred work.**

Requires runtime support — new predicates in
`wam_elixir_target.pl`:

- `WamRuntime.in_forkable_aggregate_frame?(state)` — CP-stack walk
  looking for an aggregate-frame marker.
- `WamRuntime.merge_into_aggregate(state, branch_results)` — feeds
  fully-exhausted branch outputs into the surrounding aggregate
  accumulator.

Both land in the Tier-2 implementation PR; the WAM target doesn't
have aggregate-frame markers yet, so building them is part of Tier-2
scope.

### Nested-fork suppression (prevent exponential spark explosion)

Haskell's `runBranchForFork` documents this explicitly:

> Suppress nested forks: redirect Par* to sequential equivalents
> inside a branch. Only the OUTERMOST `ParTryMeElse` actually forks;
> inner recursive calls use sequential choice points. Without this,
> recursion depth D with branching factor B creates B^D nested
> `parMap` sparks — exponential explosion.

Elixir equivalent: `branch_state` gets `parallel_depth:` (0 at the
outermost fork, +1 inside a branch). `par_wrap_segment`'s emitted
wrapper checks the field — if `parallel_depth > 0`, emit sequential
`wrap_segment` instead of forking again. Only the outermost
aggregate's branches actually parallelise; recursive child calls
stay sequential inside each branch.

### Cut semantics (decision: runtime barrier, not static filter)

Matches Haskell's approach. Cut (`!`) inside a parallel branch is
contained to that branch via a **cut barrier pinned at fork time**.
The emitted wrapper writes the parent's current `choice_points` into
each branch's `cut_point` before spawning:

```elixir
branch_state = %{state | cut_point: state.choice_points}
```

When `!/0` fires inside the branch, the runtime truncates
`choice_points` back to `cut_point` — removing only CPs the branch
itself pushed after fork. Sibling branches' CPs are unreachable by
construction; parent's aggregate frame is unreachable by
construction.

This is the same mechanism Elixir already uses for per-predicate cut
via `allocate` / `deallocate` (PR #1535). Fork pinning is structurally
a "virtual allocate" — save the barrier without pushing an env frame.

The CPS `throw({:fail, state})` propagation handles branch failure
cleanly: a branch that throws `{:fail, _}` exits its own `try/catch`
and the task returns the failure signal; sibling tasks continue
unaffected. No `Task.Supervisor` + manual cancellation needed for
the failure path — only `on_timeout: :kill_task` for runaway
branches.

**Semantic tradeoff stated honestly.** A pure predicate of ≥3
clauses that contains `!` *and* is called inside a findall-style
aggregate collection will produce solutions that cut would have
pruned under sequential evaluation. `forkMinBranches = 3` excludes
the common cases (2-clause greens like `max/3`); the remaining
divergence is accepted as the cost of Tier 2, same as Haskell.
Called outside an aggregate (i.e. via direct `run/1` +
`next_solution/1` enumeration), the predicate stays sequential via
the aggregate-frame gate and cut semantics are unchanged from
Tier 3.

### Shape of generated Tier-2 code

```elixir
defp clause_main(state) do
  # Gate 1: only fork inside a forkable aggregate frame. A direct
  # run/1 + next_solution/1 enumeration call has no aggregate frame
  # on the CP stack → falls back to sequential (Tier 3 behaviour).
  cond do
    not WamRuntime.in_forkable_aggregate_frame?(state) ->
      clause_main_sequential(state)

    # Gate 2: suppress nested forks inside an already-forked branch.
    # Recursion depth D × branching factor B otherwise creates B^D
    # concurrent tasks. Inner calls use sequential CPs via the same
    # par_wrap_segment check.
    Map.get(state, :parallel_depth, 0) > 0 ->
      clause_main_sequential(state)

    true ->
      # Pin cut barrier at fork — !/0 inside branches can only prune
      # branch-local CPs, not sibling branches or the parent.
      # Increment parallel_depth so child calls use sequential CPs.
      branch_state = %{state |
        cut_point: state.choice_points,
        parallel_depth: Map.get(state, :parallel_depth, 0) + 1
      }

      branches = [&clause_main_impl/1, &clause_k1_impl/1, &clause_k2_impl/1]

      # Each branch runs to full exhaustion, collecting every solution
      # it can produce. Results are merged into the surrounding
      # aggregate's accumulator (the caller was inside findall /
      # bagof / aggregate_all).
      branch_results =
        branches
        |> Task.async_stream(& &1.(branch_state),
                             on_timeout: :kill_task,
                             ordered: false,
                             max_concurrency: System.schedulers_online())
        |> Enum.flat_map(fn
          {:ok, solutions} when is_list(solutions) -> solutions
          _ -> []
        end)

      WamRuntime.merge_into_aggregate(state, branch_results)
  end
end
```

`clause_main_impl/1`, `clause_k1_impl/1`, etc. are the
currently-generated per-clause `defp`s, unchanged. `clause_main_sequential/1`
is the pre-Tier-2 `wrap_segment`-produced body — Tier 2 emits both
forms so the runtime gates can choose. The outer wrapper adds three
gates (aggregate frame, nesting depth, static preconditions already
checked at emit time) around the parallel fan-out + cut-barrier
pinning + full-exhaustion collection.

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

1. **Elixir-side purity-certificate consumer.** Import the module,
   resolve `analyze_predicate_purity/2` at classify time, thread the
   verdict into the `par_wrap_segment/3` gate.
2. **Aggregate-frame support in the runtime.**
   `WamRuntime.in_forkable_aggregate_frame?/1` (CP-stack walk looking
   for an aggregate marker) plus `WamRuntime.merge_into_aggregate/2`
   (feeds fully-exhausted branch outputs into the surrounding
   aggregate accumulator). The WAM target does not have aggregate
   markers yet; building them is part of Tier-2 scope.
3. **Nested-fork suppression plumbing.** `parallel_depth` field on
   `WamState` (or equivalent), incremented when fanning out,
   consulted by `par_wrap_segment`'s emitted wrapper.
4. **CPS `throw/catch` semantics verified under `Task.async_stream`.**
   A branch that throws `{:fail, state}` must be captured as "this
   alternative failed" without killing sibling tasks. The
   `{:exit, reason}` vs `{:ok, value}` dichotomy of `async_stream`
   maps onto this cleanly but needs a focused test.

### Deferred

- **Measurement.** Whether Tier 2 actually pays on realistic Elixir
  workloads requires runtime profiling — desktop-environment work.
- **Cost-model knob for "worth parallelizing" beyond min-branches.**
  A 3-clause predicate probed once doesn't need parallelism; a
  5-clause predicate probed 10k times clearly does. Secondary
  threshold plugs into the same `cost_aware` policy hook
  (PR #1559) once measurement infrastructure exists.
- **Richer aggregate strategies.** Haskell's `MergeFindall`,
  `MergeSum`, `MergeSet`, etc. are distinct merge modes. The initial
  Elixir implementation can ship with findall-style list collection
  only and add `sum`/`set` specialisations later without changing
  the gate shape.

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
  accidental divergence — with the caveat that Tier 2 explicitly
  accepts cut-in-aggregate divergence (see "Cut semantics" above), so
  the test corpus must carefully separate predicates that should
  match under the two tiers from predicates that are allowed to
  differ.
- **Where the tier decision lives.** Currently scattered — each
  target's emitter has its own policy. A shared `layout_policy/6`
  hook like Elixir's Phase E might generalize across tiers, not just
  within Tier 1.

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
