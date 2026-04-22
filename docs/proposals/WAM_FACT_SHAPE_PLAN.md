# WAM Fact Shape Implementation Plan

## Goal

Move the WAM lowered emitters from "every fact is a function" toward
"facts are data." Start with Elixir, where the scaling failure is
already observable (a 6009-clause `category_parent/2` that the host
compiler cannot load in ten minutes). Generalise to other WAM targets
once the contract stabilises.

The phased rollout is designed so each phase is shippable on its own
and produces observable results.

## Phase A: Classification infrastructure (no behaviour change)

Status: landed (PR #1511). `classify_predicate/4`, `clause_count/2`,
`fact_only/2`, `first_arg_groundness/3` exported from
`wam_elixir_lowered_emitter`. `fact_layout/2` and
`fact_count_threshold/1` options recognised. Every generated module
carries a fact-shape comment under `@moduledoc`. Tests cover small /
big / rule / variable-head / override / threshold-override /
comment-in-output / no-behaviour-change. Observation-only — every
predicate still compiles via the `compiled` path.

Work:

- Implement `fact_only/1`, `clause_count/2`, `first_arg_groundness/2`
  as emitter-internal predicates operating on the already-loaded clause
  list.
- Add option plumbing: `fact_layout/2`, `fact_count_threshold/1`,
  `fact_layout_defaults/1`.
- At `lower_predicate_to_elixir/4` entry, compute the layout for each
  predicate. Store it on a new fact that later phases consume.
- Emit layout choice as a comment in the generated module so it is
  observable without a debugger.

Acceptance:

- New unit tests assert correct classification for: empty predicate,
  one-clause rule, 100-clause facts, 1000-clause facts, user override.
- All existing WAM-Elixir tests still pass (behaviour unchanged —
  everything still emits as `compiled`).

## Phase B: `inline_data` emission for Elixir

Status: landed (PR #1519). `inline_data` predicates now emit an
`@facts [{...}, ...]` module attribute and a single `run/1` that
delegates to `WamRuntime.stream_facts/3`. New runtime pieces
`stream_facts/3` + `resume_fact_stream/3` plus a `{:fact_stream,
remaining, arity}` CP shape that `backtrack/1` dispatches on.
Compound head args fall back to `compiled`. Observable effect:
dev-scale `category_parent.ex` drops from many defps to ~9.5 KB;
host-compile time drops from seconds to milliseconds. Required two
adjuncts to actually work at 6000+ clauses: WAM atom quoting
(PR #1522) and CPS continuations (PR #1500).

Work:

- In `wam_elixir_lowered_emitter`, add an `emit_inline_data/4` branch
  selected by the Phase-A layout choice.
- Emit `@facts` module attribute as an Elixir list of tuples. Convert
  atoms/strings/numbers to host literals; variable head args become a
  sentinel. Structure/list args are deferred to Phase C.
- Emit a `run/1` that delegates to `WamRuntime.stream_facts/3`, and a
  `run(args)` wrapper that mirrors the current arg-prep code.
- In `wam_elixir_target`, implement `stream_facts/3`,
  `resume_fact_stream/2`, and make `backtrack/1` recognise the
  fact-stream CP shape.

Acceptance:

- `examples/debug_wam_elixir_ancestor.pl` still produces the expected
  4/1/4 solution counts with correct N values.
- `examples/benchmark/benchmark_wam_elixir_effective_distance.py`
  at dev scale completes without timing out (driver issue
  notwithstanding — a separate arithmetic fix may be needed).
- `category_parent/2` at scale 300 emits a module < 100 KB with one
  `defp run/1` instead of 6009 `defp clause_*/1`; the Elixir compiler
  loads it in seconds, not minutes.
- A new unit test asserts the shape of a generated `inline_data`
  module for a small fact set.

## Phase C: First-argument indexing

Status: landed (PR #1525). Predicates with all-ground arg1 emit an
`@facts_by_arg1 %{key => [tuple, ...]}` module attribute alongside
`@facts`; `run/1` derefs A1 and picks the indexed bucket when ground,
else falls back to the flat list. `fact_index_policy` option (`auto`
default, `first_arg`, `none`) gates it. Measured on scale-300
`category_parent/2` (6008 facts): seeded query first solution in
~8 µs vs ~626 µs for unbound-first-arg (80× speedup on seeded).
Module grows ~2× (336 KB flat → 716 KB flat + index) in exchange.

Work:

- When `first_arg_groundness = all_ground` and `fact_index_policy` is
  `first_arg` or `auto`, emit an additional `@facts_by_arg1`
  attribute: an Elixir map grouping facts by their first-arg value.
- Update `stream_facts/3` to accept an optional index hint and
  dispatch to the indexed bucket when `regs[1]` is bound at call
  time.
- Keep the flat-scan fallback as the default and as the
  mixed-groundness path.

Acceptance:

- A new unit test exercises a seeded call (first arg bound) on a
  >1000-clause fact-only predicate and measures a significant
  speed-up over the flat-scan path.
- The reproducer and ancestor-scale benchmark still produce correct
  answers.

## Phase D: `external_source` layout and `FactSource` behaviour

Status: landed (PR #1551). Generic `WamRuntime.FactSource` behaviour
with `open/3`, `stream_all/2`, `lookup_by_arg1/3`, optional `close/2`;
struct-based facade so callers don't need to know the concrete
adaptor module. One concrete implementation: `WamRuntime.FactSource.Tsv`
(two-column TSV, eager load, first-arg index, `header: :skip | :none`
option). `WamRuntime.FactSourceRegistry` uses `:persistent_term` for
O(1) cross-process reads. Emitter branch for
`fact_layout(P/A, external_source(SourceSpec))` produces a module
with no `@facts` — just a `@pred_indicator` string, a registry
lookup, and a facade dispatch. Verified byte-for-byte equivalent to
the inline_data layout at dev scale. Designed to extend to SQLite,
ETS, memory-mapped hash tables, and the preprocessed-artifact
direction already in motion on the C# side without touching the
emitter (see `PREPROCESSED_PREDICATE_ARTIFACTS.md`, PR #1548).

Work:

- Define the `FactSource` behaviour in `wam_elixir_target.pl`:
  `open/3`, `next/2`, `close/2`, `lookup_by_first_arg/3`.
- Ship one concrete implementation: `WamRuntime.FactSource.Tsv`
  backed by two-column TSV files, matching the formats already in
  `data/benchmark/*/`.
- Add the emitter path for `fact_layout(P/A, external_source(SourceSpec))`.
- Update the benchmark generator to route very large fact-only
  predicates through `external_source(tsv(...))` rather than
  inlining 6 MB of literals.

Acceptance:

- `benchmark_wam_elixir_effective_distance.py` at scale 300 runs
  end-to-end, producing output the reference TSV can be diffed
  against.
- An explicit `fact_layout(cat_parent/2, external_source(tsv(...)))`
  declaration in the benchmark generator overrides the default and
  is exercised by tests.

## Phase E: Cost-based default selection

Status: landed (PR #1555) — **pluggable-policy** part only. The
Phase-A `pick_layout/5` is now a dispatcher to named policies.
Three built-ins: `auto` (byte-identical to the pre-Phase-E rule),
`compiled_only`, `inline_eager`. A multifile
`user:wam_elixir_layout_policy/5` hook lets external tooling plug in
their own policy by name without patching the emitter. User
per-predicate `fact_layout/2` overrides still preempt any policy.

**Deferred to follow-up:** the *measurement-driven tuning* the
Phase-E acceptance criterion talks about. The policy surface is open
for a cost-aware selector; a separate PR would plug one in via the
hook once measurement tooling and a representative corpus are in
place. Existing `auto` behaviour is unchanged and remains safe, so
this deferral doesn't block any of Phases A–D's observable wins.

## Phase F: Port to other WAM targets

Status: **deferred as future work** (intentional scope limit).

Rationale for deferral:

- Phases A–E landed for Elixir and the end-to-end result is strong
  enough to call a stopping point: 9193 data rows byte-for-byte
  identical to native SWI across the scale-300, 1k, 5k, 10x, and 10k
  corpora (PR #1547 for reference generation; PR #1554 for the
  harness parity column that will catch any regression).
- Each target is substantive separate work — a new `inline_data`
  emitter branch, a new `stream_facts` / `resume_fact_stream`
  implementation in that target's host language, and its own test
  surface. Doing all of them in one change would be huge; doing one
  at a time is a natural per-target PR pattern.
- The Elixir spec + behaviour contract already documents the shape
  other targets should adopt, so future ports are well-defined.

Targets in rough priority order (unchanged from the original plan):

1. `wam_clojure_target` — Clojure is next most likely to see the same
   pathology at scale.
2. `wam_go_target`, `wam_rust_target` — both compile facts to
   functions today.
3. `wam_haskell_target` — has its own fact-heavy fast paths; fit may
   already be acceptable, worth measuring first. **Design trilogy
   already drafted** — see
   `WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md`,
   `WAM_HASKELL_FACT_ACCESS_SPEC.md`, and
   `WAM_HASKELL_FACT_ACCESS_PLAN.md` (commit `776c9c2`). Those docs
   reuse the `compiled` / `inline_data` / `external_source`
   vocabulary from this plan verbatim, confirming it as the
   cross-target standard. Phased rollout F1–F6 mirrors this one;
   F6 in particular specs a `MmapFactSource` that Elixir could
   parallel once the binary artifact format stabilises
   (see below).
4. `wam_csharp_native_target` — the C# side is already on the
   materialization-aware runtime, but the WAM-generated-code side may
   still compile facts as methods.
5. Other targets as measurement warrants.

For each target:

- Implement the `inline_data` path using the host's most natural
  literal shape.
- Implement `stream_facts` / `resume_fact_stream` / fact-stream CP
  handling in the target's runtime.
- Add a regression test against a fact-only predicate that would
  previously have triggered the host compiler limit.

Acceptance:

- Each ported target compiles a >5000-clause fact-only predicate in
  seconds.
- The target's existing correctness tests still pass.

## Risks and mitigations

- **Risk: correctness drift between `compiled` and `inline_data`.**
  Mitigation: a golden-output test suite that exercises a predicate
  with each layout and diffs the solution stream.
- **Risk: indexed and flat-scan paths diverge semantically when arg 1
  starts bound and becomes unbound mid-backtrack.**
  Mitigation: the CP snapshot records the list being scanned, so a
  restore always resumes the same list; indexed lookup only happens
  at the entry call, never mid-stream.
- **Risk: third-party drivers depend on the per-clause `defp`
  naming.**
  Mitigation: drivers talk to `run/1` and `next_solution/1` only; the
  spec explicitly keeps those stable. A note in the CHANGELOG flags
  the private-function-name change.

## Exit criteria

The project was considered done when:

1. **Met.** `category_parent/2` at scale 300 compiles and runs
   through the effective-distance benchmark to a correct reference
   row count on Elixir — 272 rows including header, byte-for-byte
   identical to native SWI (see `data/benchmark/300/reference_output.tsv`
   from PR #1547 and the harness parity check in PR #1554).
   End-to-end runtime at scale 300: ~2.9 s.
2. **Deferred.** "At least one other WAM target has the same layout
   available" — intentional scope limit (Phase F above). The
   design contract is documented; the port itself is future work.
3. **Met.** Documentation in
   `docs/design/WAM_ELIXIR_CORRECTNESS_GAPS.md` and the fact-shape
   proposal triplet (philosophy / spec / this plan) reflect the new
   default shape.

## Wrap-up summary

Beyond the five phases, several supporting changes landed to make
the Elixir end-to-end actually work at scale:

- PR #1500 — CPS continuations (non-tail recursion).
- PR #1505 — O(N²) codegen string-accumulator fix.
- PR #1522 — WAM atom quoting (separator-containing atoms).
- PR #1525 — integer-preserving arithmetic + Phase C.
- PR #1535 — category_ancestor cut barrier + heap-walking list
  builtins (length/2, member/2).
- PR #1547 — reference outputs for scales 300 / 1k / 5k / 10x / 10k
  + UTF-8 fix in the bench driver.
- PR #1554 — Python benchmark harness reports parity vs reference.

Aggregate result at end of Phase E: WAM-Elixir `effective_distance`
matches native SWI byte-for-byte at every scale except dev (where
18 rows show FP-associativity deltas of ~0.001–0.02 from a
summation-order difference — semantically correct, dominant on tiny
data because bigger data aggregates more paths). Total parity:
**9193 rows across 10x / 300 / 1k / 5k / 10k scales exact**.

## Cross-target alignment notes

Ideas surfaced in sibling target-design docs that would cross-apply
back to the Elixir side without changing Phases A–E's scope:

- **Binary-artifact (`Mmap`) `FactSource` adaptor.** Parallels the
  Haskell plan's Phase F6 and consumes the artifact format described
  in `PREPROCESSED_PREDICATE_ARTIFACTS.md`. Would slot alongside
  `Tsv` / `Ets` / `Sqlite` in the existing
  `WamRuntime.FactSource` behaviour without any emitter or registry
  change. Deferred until the artifact format stabilises on the C#
  side (ongoing via the `csharp-query` PRs).
- **Purity / order-independence feedback loop.** The Haskell
  philosophy doc (`WAM_HASKELL_FACT_ACCESS_PHILOSOPHY.md` §
  "Connection to purity, order-independence, and parallelism")
  observes that goal reordering affects which fact predicates get
  probed, which access patterns dominate, and therefore which
  layout is optimal. If `PURITY_CERTIFICATE_SPECIFICATION.md`-style
  certificates reach the Elixir target, the `cost_aware` policy
  (PR #1559) becomes a natural consumer — probe-count hints could
  feed the `user:wam_elixir_layout_policy/5` hook that Phase E
  exposed. See also
  [`docs/design/WAM_TIERED_LOWERING.md`](../design/WAM_TIERED_LOWERING.md),
  which argues the purity certificate is the **cross-target routing
  signal** deciding which tier (1 pure-functional, 2 host-native
  parallel, 3 WAM/CPS) a predicate lands in. This fact-shape work
  is the Elixir side of Tier 1.
- **Laziness trade-offs.** Haskell's default lazy semantics let its
  fact sources stream without special machinery; Elixir is strict
  by default, so our `inline_data` holds the full `@facts` list in
  the BEAM literal pool. A future `Stream`-based `inline_data`
  variant would only help if paired with a predicate that never
  backtracks past the first solution — the existing shape is fine
  for now and matches how other strict targets behave.
