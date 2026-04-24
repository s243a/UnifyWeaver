# Elixir WAM Tier-2 wiring proposal

**Status:** Draft for design review (pre-implementation).
**Target audience:** Perplexity review + future implementer.
**Depends on:** PR #1586 (infrastructure), PR #1607 (async_stream probe),
PR #1608 (`par_wrap_segment/4` emitter).

## 1. Context

Three PRs have landed, all as dead code:

1. **PR #1586** — `WamState.parallel_depth`, `in_forkable_aggregate_frame?/1`,
   `merge_into_aggregate/2`, `tier2_purity_eligible/3`.
2. **PR #1607** — CPS `throw/catch` × `Task.async_stream` interaction probe;
   design-doc emission template revised to wrap each branch in `try/catch`
   inside the task function (naked throws crash the parent on any BEAM
   node).
3. **PR #1608** — `par_wrap_segment(+Pred/Arity, +Segments, +Options, -Code)`
   emits the `cond`-based super-wrapper per the corrected template, with
   three static gates (kill-switch / purity ≥0.85 / clause count ≥3).

Nothing is wired into the main emission path yet. `lower_predicate_to_elixir/4`
never calls `par_wrap_segment/4`. No user-observable behaviour has
changed across the three PRs.

## 2. Goal

Thread `par_wrap_segment/4` output into `lower_predicate_to_elixir/4` so
that **eligible predicates actually run through the super-wrapper at
runtime**. Ineligible predicates must continue to emit byte-for-byte
identical code so the 63 green tests keep passing.

"Eligible" means the three static gates of `par_wrap_segment/4` pass
AND the layout is `compiled` (Tier 2 doesn't apply to `inline_data` or
`external_source` — those are fact tables, not rule bodies).

## 3. Current emission pipeline

```
lower_predicate_to_elixir(Pred/Arity, WamCode, Options, Code)   [line 46]
  → classify_predicate/4                                         [line 289]
      → pick_layout → {compiled | inline_data | external_source}
  → if inline_data     → render_inline_data_module                [~line 160]
  → if external_source → render_external_source_module            [~line 60]
  → else (compiled)    → render_compiled_module/8                 [line 94]
      → generate_all_segments/3                                   [line 666]
          → maplist(generate_one_segment/3, Segments, ...)
              → segment_func_name(Name, FuncName)                 [line 818]
              → split_body_at_calls/2 → SubSegs                   [line 691]
              → emit_sub_segments/5                               [line 708]
                  → wrap_segment(FuncName, HeadType, ..., Code)   [line 830-892]
                      (embeds `pc: &<FallbackFunc>/1` for try_me_else
                       and retry_me_else via segment_func_name(L, ...))
      → `def run(state) = <FirstFunc>(state)` via segment_func_name [line 99, 113]
      → switch_on_constant → build_switch_arm_group(Key-Labels)    [line 802]
          → `"k" -> throw({:return, <LocalFunc>(...)})` via
             segment_func_name(OnlyLabel, LocalFunc)                [line 806]
```

**Every clause-name reference in emitted code flows through
`segment_func_name/2`.** That's the chokepoint.

All occurrences of `_FuncName` in `wam_elixir_lower_instr/5` are unused
(prefixed with underscore), so instruction-level emission does not
embed the current clause name anywhere — only the segment/wrap layer
does.

## 4. Proposed wiring

### 4.1 Rename strategy: `segment_func_name/3` with suffix

Add a suffix-taking variant:

```prolog
segment_func_name(Label, Suffix, Name) :-
    segment_func_name(Label, BaseName),
    (   Suffix == "" -> Name = BaseName
    ;   format(string(Name), "~w~w", [BaseName, Suffix])
    ).
```

Thread `Suffix` through:
- `generate_one_segment/3` → `generate_one_segment/4` (add `Suffix`)
- `emit_sub_segments/5` → `/6`
- `wrap_segment/4` → `/5` (uses `Suffix` for `FallbackFunc`)
- `build_switch_arm_group/2` → `/3`

When Tier 2 is active, pass `"_impl"`. Otherwise pass `""` — zero
behaviour change.

**Pro:** single threading point; every emitted reference is consistent
by construction.
**Con:** arity bump on four predicates. Some of those are exported
(`wrap_segment` is internal, but `par_wrap_segment` is not — leaving
that one at /4 since it's the super-wrapper itself, not a renameable
segment).

### 4.2 Orchestrator shape

Current per-clause wrappers already implement the sequential
backtracking chain via try_me_else / retry_me_else / trust_me CPs —
each clause's failure pops its own CP and resumes at the next clause
via the CP's `pc` field. **There is no standalone sequential
orchestrator today** — the chain walk is distributed across the
wrappers.

Consequence: `clause_main_sequential/1` can be a one-line alias:

```elixir
defp clause_main_sequential(state), do: clause_main_impl(state)
```

`clause_main_impl/1` (the renamed first clause) carries the
try_me_else CP pointing at `clause_LB_impl/1`, and so on down the
chain. Tier 3's entire control flow is preserved by the renames; no
new orchestrator logic is needed.

### 4.3 `lower_predicate_to_elixir/4` changes

Inside `render_compiled_module/8`, after `generate_all_segments/3`:

```prolog
render_compiled_module(CamelMod, CamelPred, PredStr, Arity, ShapeComment,
                       Segments, Labels, Options, Code) :-
    % Decide Tier-2 eligibility.
    par_wrap_segment(PredStr/Arity, Segments, Options, Tier2Wrapper),
    (   Tier2Wrapper == ""
    ->  Suffix = "",
        Tier2Extras = ""
    ;   Suffix = "_impl",
        format(string(Tier2Extras),
               '~w~n~n  defp clause_main_sequential(state), do: clause_main_impl(state)~n',
               [Tier2Wrapper])
    ),
    generate_all_segments(Segments, Labels, Suffix, FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FuncsBody),
    Segments = [FirstSegName-_|_],
    segment_func_name(FirstSegName, Suffix, FirstFunc),
    % FirstFunc is the delegate target of `def run(state)`:
    %   - Suffix == ""     → FirstFunc = clause_main (Tier-3 path, unchanged)
    %   - Suffix == "_impl" → FirstFunc = clause_main  (the Tier-2 super-
    %     wrapper, which in turn delegates to clause_main_sequential =
    %     clause_main_impl on gate miss)
    ...
```

**Key:** when Tier 2 is active, `run/1` still delegates to
`clause_main` — but `clause_main` is now the super-wrapper (from
`Tier2Wrapper`), not the first clause's body. The first clause's body
is `clause_main_impl`, reachable via the super-wrapper's true-arm or
via `clause_main_sequential`.

**When Tier 2 is NOT active, Tier2Extras is `""` and Suffix is `""`:**
`segment_func_name(clause_start, "", clause_main)` — identical to the
current `segment_func_name(clause_start, clause_main)`. Every emitted
byte is unchanged.

### 4.4 Module assembly order

Final emission structure when Tier 2 active:

```elixir
defmodule WamPredLow.Foo do
  def run(state), do: clause_main(state)   # unchanged surface
  def run(args)  ...                       # unchanged

  # --- Tier-2 super-wrapper (from par_wrap_segment/4) ---
  defp clause_main(state) do
    cond do
      not WamRuntime.in_forkable_aggregate_frame?(state) ->
        clause_main_sequential(state)
      Map.get(state, :parallel_depth, 0) > 0 ->
        clause_main_sequential(state)
      true ->
        ... Task.async_stream ... merge_into_aggregate ...
    end
  end

  defp clause_main_sequential(state), do: clause_main_impl(state)

  # --- Renamed clause chain (Tier-3 mechanics, unchanged) ---
  defp clause_main_impl(state) do
    cp = %{pc: &clause_LB_impl/1, ...}   # renamed fallback target
    state = %{state | choice_points: [cp | state.choice_points]}
    try do
      ...
    catch
      {:fail, s} -> ...
    end
  end

  defp clause_LB_impl(state) do ... end
  defp clause_LC_impl(state) do ... end
end
```

When Tier 2 is **not** active, `clause_main` IS the renamed first
clause (Suffix = ""), no super-wrapper, no sequential alias —
byte-for-byte identical to current output.

## 5. Candidate approaches considered and rejected

### 5.1 Post-hoc string rename

Emit sequential output first, then `re_replace` every `defp clause_X` →
`defp clause_X_impl` (and every internal `&clause_X/1` ref). **Rejected**:
regex over emitted code is fragile, especially when clause names can
collide with `_k` continuation naming (`clause_main_k1`) or switch-arm
references. One missed reference silently breaks the chain.

### 5.2 Parallel emission (both name sets coexist)

Emit both `clause_main` (original) and `clause_main_impl` (Tier-2
variant) side-by-side, each a copy of the body. **Rejected**: doubles
the defp count for eligible predicates, and having two definitions of
the same logic is a maintenance hazard (bug fixed in one, not the
other).

### 5.3 Threaded suffix (proposed above)

Pass `Suffix` atom through the naming helpers. Zero duplication;
every name is derived from one source. **Chosen.**

## 6. Risks and open questions

1. **Is `clause_main_sequential/1` correctly a pure alias?**
   The distributed Tier-3 chain walk means the first `_impl` function
   already orchestrates sequential evaluation. Does that hold in edge
   cases like first-arg indexing (`switch_on_constant` arms)? The switch
   arms dispatch to specific clauses via `build_switch_arm_group`;
   those targets are renamed too under Suffix threading. Needs tracing
   through a multi-clause predicate with first-arg indexing to confirm.

2. **CPS sub-segments (`_k1`, `_k2`, ...) naming collision.**
   A body with `call`s produces `clause_main_k1`, `clause_main_k2`, ...
   Under Suffix = "_impl" they become `clause_main_impl_k1`, etc. Does
   `emit_cont_segments/5` correctly thread the BaseFunc (which now has
   `_impl` suffix)? A quick audit says yes — the `_kN` format uses
   `BaseFunc` which comes from the parent segment's FuncName, which is
   already suffixed. But one reviewer pass would help.

3. **Cut barrier (PR #1535 `cut_point`).** The existing `cut_point`
   mechanism saves/restores `state.choice_points` snapshots. The Tier-2
   super-wrapper pins `cut_point: state.choice_points` on fork. Inside
   a forked branch, `!` restores to that snapshot, pruning branch-local
   CPs. Does this interact correctly with the CPs pushed by
   `clause_main_impl` (the try_me_else at branch entry)? The design
   doc claims yes because each fork is a single clause alternative, not
   the whole chain — but verifying on a declared-pure 3-clause predicate
   with a cut in one clause is a worthwhile test.

4. **`intra_query_parallel(false)` option propagation.** The kill-switch
   is read from `Options` in `par_wrap_segment/4`. `lower_predicate_to_elixir/4`
   already threads `Options` through. Confirmed propagates correctly.
   No wiring concern.

5. **Does `render_compiled_module/8` need an arity bump for `Options`?**
   Current signature at line 94 takes 8 args but does not include
   `Options`. Adding the Tier-2 check requires threading `Options` in.
   Small edit.

6. **Interaction with inline_data / external_source paths.**
   Both bypass `render_compiled_module`. `par_wrap_segment/4` is never
   called for them. Clean.

## 7. Test matrix

Must pass after wiring:

- **Regression — ineligible predicates.**
  - Every existing test in `tests/test_wam_elixir_target.pl` (currently
    63 pass). The whole suite is an implicit regression check since
    no predicate under test has been declared pure.
  - `examples/debug_wam_elixir_ancestor.pl` — 4 solutions, integer N.
  - Parity harness (scales 300/1k/5k/10x/10k) — 9193 rows byte-for-byte.

- **New — eligible predicates.**
  - 3-clause declared-pure predicate in a module, emitted output
    contains `defp clause_main` (super-wrapper) + `defp clause_main_sequential`
    + `defp clause_main_impl` + renamed chain.
  - Same predicate, with `intra_query_parallel(false)` in Options →
    falls through to sequential-only emission (byte-for-byte identical
    to the ineligible case).
  - Integration: call it without an aggregate frame on the CP stack
    → super-wrapper's `in_forkable_aggregate_frame?/1` gate returns
    false → `clause_main_sequential(state) = clause_main_impl(state)`
    executes sequentially. Result identical to Tier-3-only emission.

- **Compilability.**
  - `elixir -r <tier2_module>.ex -e "WamPredLow.Foo.run([...])"` must
    load and run without compilation errors. Especially around
    `clause_main_sequential` alias and the renamed chain.

## 8. Rollback plan

`par_wrap_segment/4` always returns `""` when gates fail. Emitting
`""` for `Tier2Wrapper` causes `Suffix = ""` which disables all
renames and restores current byte-for-byte behaviour. If post-merge
a bug appears, reverting just `render_compiled_module` / `generate_all_segments`
to pre-wiring signatures is sufficient — the Tier-2 emitter code
(`par_wrap_segment/4`, `emit_par_tier2_wrapper/2`) remains dead code
and harmless.

## 9. Questions for design review

1. **Is the threaded-suffix approach the right structural choice**, or
   should the rename happen at a different layer (e.g., a post-process
   pass that rewrites the emitted module string)? Trade-off is
   upfront type-system cleanliness vs. keeping the emitter's internal
   API stable.

2. **Is `clause_main_sequential/1` correctly a pure alias for
   `clause_main_impl/1`**, or does Tier 2 need an explicit sequential
   orchestrator that bypasses the try_me_else CP machinery inside the
   `_impl` chain? (The design doc assumes alias suffices; the risk is
   that on a direct `run/1` call that doesn't enter via the aggregate
   context, the CP pushed by `clause_main_impl` is visible to the
   caller's backtrack stack — but that's the current Tier-3 behaviour
   preserved intact, so should be fine. Sanity check wanted.)

3. **Should `lower_predicate_to_elixir/4` require an `Options` arg
   (it already has one) that the wiring reads directly, or should it
   consult a module-scoped default config?** Proposed: stay with
   `Options`, matching existing patterns (e.g., `module_name`,
   `fact_count_threshold`).

4. **Is first-arg indexing (`switch_on_constant`) compatible with
   Tier-2 fan-out?** The switch arms dispatch to specific clauses
   synchronously; the Tier-2 super-wrapper fan-outs are parallel over
   ALL clauses. When both machinery are emitted together, does the
   switch short-circuit the fan-out correctly? Proposed answer: the
   switch is emitted inside `clause_main_impl` (Tier-3 mechanics,
   untouched); the super-wrapper that calls the branches from
   `Task.async_stream` bypasses the switch entirely (it fans each
   clause's `_impl` directly, no switch dispatch). Each branch then
   does its own head unification and may fail fast on arg mismatch —
   which is fine, just less efficient than the switch. Confirmation
   wanted.

5. **Should the wiring PR also emit a `@moduledoc` tag indicating
   Tier-2 activation?** E.g., `@moduledoc "Tier-2 parallel-aware
   module (purity: declared, confidence: 1.0)"`. Debugging aid.

## 10. What this proposal does NOT cover

- **findall/3 producer side.** Without findall pushing an aggregate
  frame, Tier 2's `in_forkable_aggregate_frame?/1` gate never opens
  and the super-wrapper always short-circuits to sequential. Separate
  proposal.
- **Measurement-driven threshold tuning.** `forkMinBranches = 3`
  stays hardcoded. Desktop follow-up work.
- **Cross-predicate dispatch performance.** `WamDispatcher.call/2`
  dispatches across module boundaries; Tier-2 fan-out inside a branch
  body that itself calls another predicate goes through this
  dispatcher. No wiring concern, but a perf note worth recording.
