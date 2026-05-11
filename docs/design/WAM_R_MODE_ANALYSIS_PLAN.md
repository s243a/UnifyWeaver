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

## Phase 2 — get_constant inline specialisation (LANDED)

**PR**: see commit history on `claude/wam-r-mode-analysis-phase2`.

- Fix the visibility bug from phase 1 where `numbervars` ran before
  the analyser query, causing `get_binding_state(_, '$VAR'(N), bound)`
  to fire (treating numbervared vars as non-vars) and every head arg
  to render as `bound` regardless of declared mode. Phase 2 runs the
  analyser on plain Prolog vars and renders names positionally
  (`Arg1`, `Arg2`, ...) instead. Test extended to cover all four
  cases: `+`, `-`, `?`, undeclared.
- New `mode_specialise(off)` option to disable the specialisation
  (defaults to ON; off is for testing / regression bisection).
- Add a specialised `emit_line_parts(["get_constant", ...], I)` clause
  that fires when the declared mode of the target A-register is `+`
  (input). The clause emits inline R:

  ```r
  { val_ <- WamRuntime$deref(state, WamRuntime$get_reg(state, AIdx))
    if (is.null(val_) || !identical(val_, CTerm)) return(FALSE) }
  ```

  in place of `if (!isTRUE(WamRuntime$step(program, state,
  GetConstant(C, AIdx)))) return(FALSE)`. Saves one `step()` call
  (list construction + function call + switch dispatch) per
  get_constant on the inlined clause-1 path.
- The specialisation reads the mode declaration directly (via a
  `b_setval`-stashed `wam_r_lowered_mode_decl`), not the analyser's
  BeforeEnv -- BeforeEnv conflates head-pattern binding (a literal
  `alice` in the head pattern looks `bound`) with caller-side
  binding (whether A_k was bound when the caller invoked us). For
  head-match instructions like get_constant we need the latter, and
  only the mode declaration gives it to us.

### Measured impact

On a recursive predicate (`pn(0). pn(N) :- N > 0, N1 is N - 1,
pn(N1).` with `:- mode(pn(+))`), 200 iterations × recursion depth
2000:

| Run | WITH (spec) | WITHOUT |
|-----|-------------|---------|
| 1 | 93.64s | 100.98s |
| 2 | 95.03s | 94.69s |
| 3 | 95.28s | 94.14s |
| 4 | 97.14s | 96.75s |

Mean: WITH 95.3s, WITHOUT 96.6s. The difference is **within run-to-
run noise** (run 1's WITHOUT looks like an outlier).

This is the expected result: most of the work is in clause 2 (the
recursive case), which runs through the WAM array path / `step`
unchanged. The inline get_constant only fires for clause 1 (the
base case `pn(0)`), which is ~2000 calls per recursion chain --
saving ~1-2μs each → ~2-4ms per chain → ~0.5-1% of the 95s total.

**Conclusion:** the specialisation is *correct + sound + zero
regression*, but the measurable wall-clock impact on existing
benches is small. The infrastructure is the win. Phase 3+ extends
the impact by:
1. Adding more specialisations (so a larger fraction of step()
   calls disappear).
2. Lowering more shapes (so a larger fraction of clauses runs the
   lowered path instead of the array path).

Future bench design: a workload where the lowered emitter runs the
*entire* call sequence (no array fallback) and where get_constant
is a dominant fraction would show a clean win. The existing
fact-source bench doesn't fit because its fact predicates go through
the dedicated fact_table dispatch, not the lowered emitter.

## Phase 3 candidates

The fact-source-bench Rprof profile (see WAM_R_TARGET.md "Rprof
profile of the WAM stepping engine") flags the dominant costs:

| Function | self.s | %self | Mode-info lever? |
|---------:|-------:|------:|------------------|
| `WamRuntime$step` | 1.500 | 29.0% | **YES** -- inline more ops when modes prove the slot shape |
| `WamRuntime$run` | 0.760 | 14.7% | indirectly (fewer step calls = fewer run iterations) |
| `WamRuntime$run_predicate` | 0.690 | 13.3% | no -- per-call overhead, not mode-driven |
| `WamRuntime$deref` | 0.380 | 7.3% | **YES** -- skip deref at known-bound slots |
| `WamRuntime$put_reg` | 0.260 | 5.0% | no -- already trivial |

Top phase-3 candidates, in priority order:

1. **`get_value` head match -- skip deref-then-unify when both
   sides are provably bound atomics.** Same shape as the phase-2
   `get_constant` specialisation but for var-to-var matching.
2. **`is/2` arithmetic fast path -- skip per-call type-tag checks
   when all RHS vars are provably bound to numerics.** Lives in
   builtin handlers, not the lowered emitter. Independent project.
3. **Lowering more shapes.** When mode info proves a multi-clause
   predicate is deterministic at this call site, the emitter could
   pick `deterministic` instead of `multi_clause_1`, skipping the
   CP push. Requires call-site mode info, not just clause-level.
4. **Auto-mode-inference from call sites.** Today the analyser
   only knows what `:- mode/1` declarations tell it. A whole-
   program pass could infer modes from observed call sites,
   eliminating the user's burden of declaring modes (and
   broadening phase-2/3 specialisation coverage). Big project --
   probably a separate sub-campaign.

Phase 3 should pick **one** candidate, implement + bench on a
workload that's actually dominated by the targeted op (not the
fact-source bench, which goes through fact_table_dispatch). The
bench command is documented in WAM_R_TARGET.md ("Rprof profile
of the WAM stepping engine").

## Phase 4+ — adaptive specialisation

Call-site-driven specialisation: the analyser tracks not just
"variables in this clause body" but "modes of every call site I
see," and the emitter picks specialised emission per call site.
This is a much bigger change (requires whole-program mode
propagation) and is deferred until phase 3 establishes the
pay-for-itself baseline on a representative workload.

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
