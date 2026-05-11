# WAM-R Mode Analysis: Phase Roadmap

> **Companion docs:** [`WAM_HASKELL_MODE_ANALYSIS_PHILOSOPHY.md`](WAM_HASKELL_MODE_ANALYSIS_PHILOSOPHY.md)
> (the *why*) and [`WAM_HASKELL_MODE_ANALYSIS_SPEC.md`](WAM_HASKELL_MODE_ANALYSIS_SPEC.md)
> (the *what*) describe the shared analyser. This doc is the WAM-R-
> specific *how*: which downstream specialisations the WAM-R target
> consumes the analyser output for, in priority order.

## Starting point (already in place)

The shared analyser at `src/unifyweaver/core/binding_state_analysis.pl`
runs per-clause inside the WAM compiler. It's already consulted at
~15 call sites in `src/unifyweaver/targets/wam_target.pl` to choose
between specialised and generic codegen for:

- `=../2` compose-mode -> `PutStructureDyn` instruction
- `functor/3` compose-mode -> same `PutStructureDyn` shape
- `arg/3` -> dedicated `arg` WAM instruction
- `\+ member(X, V)` for visited-set vars -> `not_member_set` (O(log N))

WAM-R inherits all of these via the shared compiler. The gap this
plan closes: the **WAM-R lowered emitter** (`wam_r_lowered_emitter.pl`)
operates one level downstream (on WAM-text lines) and currently has
no link to the analyser. The lowered emitter is the hot path for
predicates that `wam_r_lowerable/3` accepts.

## Phase 1 — visibility (LANDED)

**PR**: see commit history on `claude/wam-r-mode-analysis-phase1`.

- Add `mode_comments(on)` option to `write_wam_r_project/3`.
- Wire `binding_state_analysis:analyse_clause_bindings/3` into
  `lower_predicate_to_r/4` via `gather_pred_mode_records/2`.
- When the option is on, prepend a `# Mode analysis:` block to each
  emitted lowered function summarising the per-clause head-binding
  states and any `:- mode/1` declaration.
- No runtime / codegen behaviour change; visibility only.
- Foundation for phase 2: the analyser records are now reachable
  from the emitter without re-running the analyser at WAM-text-
  parsing time.

## Phase 2 candidates — measured next

The fact-source-bench Rprof profile (see WAM_R_TARGET.md "Rprof
profile of the WAM stepping engine") flags the dominant costs:

| Function | self.s | %self | Mode-info lever? |
|---------:|-------:|------:|------------------|
| `WamRuntime$step` | 1.500 | 29.0% | **YES** -- inline more ops when modes prove the slot shape |
| `WamRuntime$run` | 0.760 | 14.7% | indirectly (fewer step calls = fewer run iterations) |
| `WamRuntime$run_predicate` | 0.690 | 13.3% | no -- per-call overhead, not mode-driven |
| `WamRuntime$deref` | 0.380 | 7.3% | **YES** -- skip deref at known-bound slots |
| `WamRuntime$put_reg` | 0.260 | 5.0% | no -- already trivial |

Top phase-2 candidates, in priority order:

1. **`get_constant` head match -- skip deref when slot is provably
   bound to a constant.** Currently delegates to `step`. With
   mode info, emit `if (state$regs2[[idx]]$id != C$id) return(FALSE)`
   inline. Expected: ~3-7% wall-clock on fact-source-bench
   (head-match heavy).
2. **`get_value` head match -- skip deref-then-unify when both
   sides are provably bound atomics.** Similar to #1 but for
   var-to-var matching.
3. **`is/2` arithmetic fast path -- skip per-call type-tag checks
   when all RHS vars are provably bound to numerics.** Lives in
   builtin handlers, not the lowered emitter. Independent project.
4. **Lowering more shapes.** When mode info proves a multi-clause
   predicate is deterministic at this call site, the emitter could
   pick `deterministic` instead of `multi_clause_1`, skipping the
   CP push. Requires call-site mode info, not just clause-level.

Phase 2 should pick **one** candidate (likely #1), implement +
bench, and only proceed to the next if the measured win justifies
it. The bench command is documented in WAM_R_TARGET.md ("Rprof
profile of the WAM stepping engine").

## Phase 3+ — adaptive specialisation

Once phase 2 demonstrates a measurable win, the next direction is
call-site-driven specialisation: the analyser tracks not just
"variables in this clause body" but "modes of every call site I
see," and the emitter picks specialised emission per call site.
This is a much bigger change (requires whole-program mode
propagation) and is deferred until phase 2 establishes the
pay-for-itself baseline.

## Risks / open questions

- **Analyser cost.** Running the analyser per clause at compile
  time is cheap (the analyser is single-pass + no fixpoint), but
  it's not free. The visibility phase doesn't show this cost
  because it only runs when the user opts in. Phase 2 will need
  to run the analyser unconditionally; if codegen time becomes
  noticeable, we'd memoise per-clause.
- **Mode declaration coverage.** Users today rarely declare
  modes. Without declarations, the analyser starts from
  "everything unknown" and proves very little. Phase 2's wins
  may be small until users start adding `:- mode/1` declarations
  or we add automatic mode inference from call sites.
- **Divergence from Haskell.** The Haskell target's analyser
  integration drives `PutStructureDyn` lowering. WAM-R doesn't
  have an equivalent specialised instruction; its leverage is
  in the lowered emitter, not in new WAM ops. The plans diverge
  here intentionally -- both target the same analyser, but
  consume its output differently.
