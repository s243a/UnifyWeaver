# WAM Fact Shape Specification

## Cross-target status

This spec was originally written for the Elixir target (landed in
Phases A–E, PRs #1511/#1519/#1525/#1551/#1555). The Haskell target's
fact-access trilogy
(`WAM_HASKELL_FACT_ACCESS_{PHILOSOPHY,SPEC,PLAN}.md`, commit
`776c9c2`) adopts the same classification vocabulary
(`compiled` / `inline_data` / `external_source`) and the same
`FactSource` shape (`open/3`, `stream_all/2`, `lookup_by_arg1/3`,
`close/2`) verbatim — so the terms defined below are treated as the
cross-target contract, not just Elixir nomenclature.

## Scope

This document specifies the contract between the WAM lowered emitter
and the WAM runtime for representing fact-only predicates. It does not
redesign unification, backtracking, or the broader lowered-emitter
pipeline. It specifies:

1. how the emitter classifies a predicate as "fact-only" and chooses a
   layout for it;
2. the Prolog-side configuration predicates the user can set to
   override that choice;
3. the runtime interface that every fact layout must satisfy, so
   drivers and calling code remain unchanged.

## Terminology

- **fact-only predicate** — every clause is a head-only clause: body
  is `true`, or equivalently absent. Head arguments may be ground
  terms (atoms, strings, numbers, compounds of those) or variables.
- **layout** — the host-language representation the emitter picks for
  a predicate: `compiled`, `inline_data`, or `external_source`.
- **fact source** — the runtime-side adaptor that yields tuples for
  an `external_source` layout. Analogous to `IRetentionAwareRelationProvider`
  in the C# query runtime.
- **fact stream** — the per-call iteration state carried in
  `state.choice_points` while enumerating a fact-only predicate's
  solutions.

## Classification predicates

The emitter consults these Prolog-side predicates before lowering a
predicate. Users may declare any of them in their program or in
`Options`; none are required.

### Introspection (emitter-provided, read-only)

- `fact_only(+PredIndicator)` — true when every clause's body is
  `true` after normalisation. Computed by the emitter from the
  already-loaded clauses.
- `clause_count(+PredIndicator, -N)` — number of clauses for the
  predicate.
- `first_arg_groundness(+PredIndicator, -Status)` — `all_ground`,
  `all_variable`, or `mixed`.

### User-controllable (optional)

- `fact_layout(+PredIndicator, -Layout)` — user-supplied override.
  If set, the emitter uses it without consulting the default policy.
  Valid `Layout` values are `compiled`, `inline_data(Options)`,
  `external_source(SourceSpec)`.
- `fact_count_threshold(-N)` — threshold used by the default policy
  to pick `inline_data` over `compiled`. Default: 100.
- `fact_index_policy(+PredIndicator, -Policy)` — `none`,
  `first_arg`, `auto`. Default is `auto`.

### Default layout policy

In absence of a user `fact_layout/2` fact, the emitter picks:

```
if not fact_only(P/A):
    layout = compiled              % current behaviour, CPS-lowered
elif clause_count(P/A, N), N =< fact_count_threshold:
    layout = compiled              % small fact sets don't need it
else:
    layout = inline_data([])
```

The choice should favour shapes that scale well. `compiled` remains
the fallback only when the numbers are small enough that it does not
matter, or when the user overrides explicitly.

## Layout contracts

Every layout produces a host-language module that satisfies the same
public interface:

```
Mod.run(%WamState{} = state) :: {:ok, state} | :fail
Mod.run(args :: list) :: {:ok, state} | :fail
```

This is the existing contract. Drivers do not observe the layout.

The differences are internal — what `run/1` does under the hood, and
how `backtrack/1` is expected to re-enter it.

### Layout: `compiled`

No change from today. Each clause becomes a `defp` function; choice
points refer to those functions.

Applies when: non-fact-only, small fact-only, or explicit user
override.

### Layout: `inline_data`

The emitter emits the fact tuples as a host literal (an `@facts`
module attribute in Elixir; analogous literal in other targets).
`run/1` delegates to a single per-target helper that iterates the
literal.

Requirements on the emitter:

- `@facts` is a list (or similar) of host-native tuples in clause
  order; element shape matches the head arity.
- Arguments that were variables in the head become a sentinel
  (`:_var`) in the tuple; the helper treats them as "unify with
  anything."
- Ground terms are lowered to their host-native equivalents.

Requirements on the runtime:

- `WamRuntime.stream_facts(state, facts, arity)` advances through
  `facts` attempting unification at each index; pushes a single
  "fact-stream CP" carrying `{facts, next_index}` when a match
  succeeds, so `backtrack/1` resumes the scan without popping
  per-fact choice points.
- `backtrack/1` recognises the fact-stream CP shape and dispatches to
  `resume_fact_stream(state, cp)` instead of invoking a `cp.pc`
  function reference.
- Trailing/unwinding behaves as today: the fact-stream CP snapshot
  holds the usual `trail_len`, `heap_len`, `regs` fields.

Indexing (when `fact_index_policy = first_arg` and the caller binds
arg 1 before the call):

- The emitter emits `@facts_by_arg1 %{arg1_val => [tuple, tuple, ...]}`
  alongside `@facts`. The helper picks the indexed list when `regs[1]`
  is ground; otherwise falls back to the flat `@facts`. One extra CP
  slot records which list is being scanned, so backtracking within
  the indexed bucket works the same way.

### Layout: `external_source`

The emitter emits only a thin wrapper that delegates to a runtime
`WamRuntime.FactSource` registered at boot time. The caller-supplied
source may back onto a TSV reader, an ETS table, a database iterator,
or any other retained state owner.

Requirements on the emitter:

- `SourceSpec` in `fact_layout(P/A, external_source(SourceSpec))` is
  an opaque term passed through to the runtime registration.
  Minimum: `tsv(Path, ArityHeader)`.
- When a shared `preprocess/2` declaration exists for the predicate,
  the emitted module should preserve normalized preprocess metadata
  alongside that raw source spec so later runtime/provider code can
  inspect the chosen declaration intent without reparsing Prolog.
  The first Elixir seam for this is module metadata, not a live runtime
  provider contract.
- `run/1` / `run/0` bodies call `WamRuntime.FactSource.open(SourceSpec, state, P/A)`
  and then use the same `stream_facts` / `backtrack` contract as
  `inline_data`.

Requirements on the runtime:

- A `FactSource` behaviour with `open/3`, `next/2`, `close/2`,
  `lookup_by_first_arg/3` callbacks. The last is optional; when
  absent, the runtime falls back to linear scan.
- A registration helper so drivers can bind a `SourceSpec` atom to a
  concrete implementation before calling `Mod.run/1`.
- The runtime does not need to consume preprocess metadata yet, but the
  generated module should surface it in a stable shape so future TSV,
  ETS, SQLite, mmap, or manifest-backed providers can use the same
  declaration seam.
- Near-term backend choice: Elixir should keep using the existing
  `FactSource` adaptors while the shared artifact/provider boundary
  matures. A local Rust LMDB prototype now confirms that LMDB is viable
  in Termux for exact relation artifacts, but that should land first as
  a shared artifact/provider path rather than a direct Elixir binding
  requirement.

## Option plumbing

The lowered emitter's `Options` list gains one new recognised key:

- `fact_layout_defaults(+List)` — overrides the default policy per
  module. Example: `fact_layout_defaults([threshold(200), index(auto)])`.

Existing options — `module_name/1`, `emit_mode/1` — are unchanged.

## Backward-compatibility

- Drivers that only call `run/1` and `next_solution/1` are
  unaffected.
- Predicates not classified as `inline_data` or `external_source`
  still emit as `compiled`, producing byte-identical output.
- Existing tests that inspect emitted code for specific `defp` names
  may need updating if they happen to reference a predicate that
  crosses the threshold. Mitigation: keep the threshold low enough
  in the default-test scenarios that test-sized predicates stay
  `compiled`.

## Non-goals

- Join planning, cost-based selection between layouts, or an
  integrated cost model. The default policy is intentionally simple
  so the behaviour is predictable; richer planning is a follow-up.
- Arbitrary body goals. A predicate with any body goal is not
  fact-only; it keeps the existing `compiled` shape.
- Cross-target uniformity. Each target implements the `inline_data`
  layout with the literal shape that suits its host language
  (Elixir module attributes, Go package-level `var`, Clojure `def`,
  Haskell `let`, etc.). The specification above is written in Elixir
  terms because Elixir is the first target; other targets follow the
  same contract with host-appropriate syntax.
