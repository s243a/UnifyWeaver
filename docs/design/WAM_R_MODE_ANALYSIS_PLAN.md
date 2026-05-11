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

## Phase 3 — is/2 specialisation (LANDED)

**PR**: see commit history on `claude/wam-r-mode-analysis-phase3-is-fastpath`.

Two combined changes targeting the `is/2` arithmetic builtin, which
is a dominant cost on any predicate that recurses while doing
arithmetic (the most common shape outside fact tables):

**(a) Runtime fast-path in the is/2 handler.** When the expression
is a 2-arg arithmetic struct (`+`, `-`, `*`, `//`, `mod`) and both
operands deref to ints, bypass the recursive `eval_arith` walk +
`arith_to_term` dispatch. Additionally fast-bind the target when it
derefs to unbound, skipping the unify. Falls through to the original
slow path for anything that doesn't match the fast shape.

```r
# In WamRuntime$call_builtin, is/2 branch:
expr_d <- WamRuntime$deref(state, expr)
if (struct + 2 args + both int) {
  fast_val <- switch(op, "+" = a + b, ...)
  if (!is.null(fast_val)) {
    res <- IntTerm(as.integer(fast_val))
    if (target derefs to unbound) {
      WamRuntime$bind(state, name, res)
      return(TRUE)
    }
    return(WamRuntime$unify(state, target, res))
  }
}
# slow path: eval_arith + arith_to_term + unify
```

Saves 3-4 function calls per is/2 hit (top-level `eval_arith` + 2
recursive calls for args + `arith_to_term`).

**(b) Lowered-emitter inline for `builtin_call is/2 2`.** Bypasses
`WamRuntime$step` → `WamRuntime$call_builtin` → big switch dispatch:

```r
{
  is_target_ <- WamRuntime$get_reg(state, 1L)
  is_expr_   <- WamRuntime$get_reg(state, 2L)
  is_n_      <- WamRuntime$eval_arith(state, is_expr_, intern_table)
  if (is.null(is_n_)) return(FALSE)
  is_res_    <- WamRuntime$arith_to_term(is_n_)
  if (is.null(is_res_)) return(FALSE)
  is_target_d_ <- WamRuntime$deref(state, is_target_)
  if (target_d unbound) bind  else unify
}
```

Saves 2 function calls + 2 switch lookups per is/2 hit. The fast-bind
branch is gated by a runtime tag check (not mode info), so it works
without `:- mode/1` declarations. Mode info would only save us the
one cheap if/else; not worth the codegen complexity for this op.

### Measured impact

Same pn recursive bench as phase 2 (`pn(0). pn(N) :- N > 0, N1 is N -
1, pn(N1).` with `:- mode(pn(+))`, 200 iterations × recursion depth
2000). Phase-2 baseline built from `main` worktree (clean phase-2
codegen + runtime), phase-3 built from this branch.

Alternating runs to defeat time-correlated noise:

| Run | PHASE2 baseline | PHASE3 |
|-----|-----------------|--------|
| 1 | 73.37s | 71.91s |
| 2 | 73.73s | 69.60s |
| 3 | 73.83s | 70.43s |
| 4 | 73.66s | 71.66s |

| Stat | PHASE2 | PHASE3 |
|------|--------|--------|
| Mean | 73.6s | 70.9s |
| Min  | 73.37s | 69.60s |
| Max  | 73.83s | 71.91s |
| Range | 0.46s | 2.31s |

**Phase 3 is consistently ~3.7% faster** (max PHASE3 < min PHASE2 by
1.46s -- no overlap in distributions). This is the first clearly-
above-noise mode-analysis-campaign win on the recursive workload.

The win comes mostly from the runtime fast-path (a): clause 2 of pn
runs through the array path / step, so the lowered-emitter inline
(b) doesn't apply to it. Phase 4's multi_clause_n would bring (b)
into play for clause 2 as well, multiplying the win.

### Connection to mode analysis

Phase 3 (b) is only weakly mode-driven: the unbound check is at
runtime. Phase 2's `get_constant` was the clearer example of mode-
driven inline. The is/2 work is in the same campaign because it
extends the same "skip dispatch hop" pattern -- and once the user
adds `:- mode(p(-, +, +, ...))`-style annotations, future phases
can drop even the unbound runtime check for the dominant idiom
`X is Expr` (X unbound, Expr bound).

### Important: get_constant inline + is/2 inline only fire on the
### lowered emitter path

Both phase-2 and phase-3 (b) specialisations only fire inside
`lowered_*` functions. The lowered emitter currently handles
`deterministic` (single-clause) and `multi_clause_1` (multi-clause,
first clause inline + array fallback) shapes. Clauses 2+ of a
multi-clause predicate run through the WAM array path / `step` /
`call_builtin` unchanged. To unlock more of the win, the next
direction is to *broaden what gets lowered*: a `multi_clause_n`
emitter that handles ALL clauses inline (gated on every clause being
individually lowerable). Listed as the leading phase-4 candidate
below.

## Phase 4 — multi_clause_n lowered emission + lowered_dispatch self-jump (LANDED)

**PR**: see commit history on `claude/wam-r-mode-analysis-phase4-multi-clause-n`.

Two combined changes that compound the phase-3 wins on recursive
arith-heavy predicates:

**(a) `multi_clause_n` lowered emission.** When every clause is
individually `wam_r_lowerable`, all clauses are emitted inline within
one lowered function, with state snapshot/restore between attempts:

```r
lowered_<pred>(program, state) <- function() {
  saved_regs_      <- state$regs2
  saved_cp_        <- state$cp
  saved_trail_len_ <- length(state$trail)
  saved_var_count_ <- state$var_counter
  clause_ok_ <- (function() { <clause 1 inline>; invisible(FALSE) })()
  if (isTRUE(clause_ok_)) return(TRUE)
  state$regs2       <- saved_regs_         # restore
  state$cp          <- saved_cp_
  WamRuntime$undo_trail_to(state, saved_trail_len_)
  state$var_counter <- saved_var_count_
  clause_ok_ <- (function() { <clause 2 inline>; invisible(FALSE) })()
  if (isTRUE(clause_ok_)) return(TRUE)
  ... (recursively for clauses 3..N)
  return(FALSE)
}
```

The emitter helper is recursive over the clause list (base case:
no more clauses, fall through to `return(FALSE)`; recursive case:
emit inline-try-then-restore-and-recurse). When only clause 1 is
lowerable (clauses 2+ contain unsupported ops), the emitter
degrades to the existing `multi_clause_1` shape: clause 1 inline +
WAM-array fallback.

**(b) `lowered_dispatch` self-jump.** Phase-1 lowered fns were
deliberately NOT registered in `program$lowered_dispatch`, so
internal `call`/`execute` of one lowered pred from another (or
recursion) went through the WAM array via `WamRuntime$step`.
Phase 4 changes this for `multi_clause_n` preds (which are
self-contained -- no `state$pc` dependency on entry):

  - At codegen, register the lowered fn in
    `shared_program$lowered_dispatch` alongside the kernel /
    fact-table fns.
  - The lowered emitter's `emit_call` / `emit_execute` consult
    `program$lowered_dispatch` first, falling back to the WAM array
    only when no lowered fn is registered.

The combination means a recursive predicate like pn can run
end-to-end through the lowered path: every recursive call to
`pn/1` self-jumps into `lowered_pn_1` instead of step-iterating
the WAM array.

`deterministic` and `multi_clause_1` preds stay off
`lowered_dispatch` because their failure paths assume `state$pc`
points at a WAM-array instruction (they advance pc + drop into
`run`), which would not hold when invoked via the dispatch tier.

### Frame-shape fix

Pre-existing bug: the inline `allocate` / `deallocate` emission
used a frame field name `locals` while the runtime expects `ys`
and `cps_barrier`. The mismatch never surfaced because
`multi_clause_1` lowering only inlined clause 1 of multi-clause
predicates, and the only clauses that include `allocate` are
non-base recursive cases that stayed in the WAM array. Phase 4
puts every clause inline, so the inline `allocate` now actually
runs -- the fix brings the inline emission into exact agreement
with the runtime's `"Allocate"` / `"Deallocate"` handlers
(including `state$shadow_frame` retention so post-`deallocate` Y
reads still resolve).

### Tail-call trampoline

First-cut phase-4 hit an R C-stack overflow at pn depth 2000: the
self-jump from `execute pn/1` recursed *into* `lowered_pn_1`, so
each Prolog recursion level added an R stack frame. R's default
C stack (~7 MiB) overflowed around depth ~1500.

Fix: trampoline the self-recursion through a state-borne signal.

  - `state$tail_call` slot added to `WamRuntime$new_state`.
  - `emit_execute` (Prolog tail call) sets `state$tail_call <-
    "X/Y"` and returns `TRUE` instead of invoking the lowered fn
    directly. The closest enclosing trampoline consumes the
    signal.
  - `WamRuntime$invoke_lowered_with_tco(program, state, key)` is
    the trampoline -- a `repeat` loop that calls the lowered fn,
    inspects `state$tail_call`, and dispatches the next iteration
    in-place (no extra R frame).
  - Every entry point to a lowered fn is routed through the
    helper: the runtime's `"Call"` / `"Execute"` / `dispatch_call`
    handlers, the lowered emitter's `emit_call`, and the per-pred
    wrapper. `emit_execute` is the *only* call site that signals
    instead of invoking, because Prolog's `execute` is by
    definition the tail-call WAM op.

With the trampoline, pn at depth 2000 (the original bench
workload) runs to completion in bounded R stack.

### Measured impact

(_filled in after bench completes_)

## Phase 5 candidates

1. **`get_value` head match.** Skip deref-then-unify when both
   sides are provably bound atomics. Same shape as phase-2's
   `get_constant` specialisation but for var-to-var matching.
2. **Inline more step-delegated ops in `multi_clause_n` bodies.**
   The phase-4 lowered pn body still has 4 `WamRuntime$step(...)`
   calls (`BuiltinCall(">/2", 2)`, `PutStructure`, `SetValue`,
   `SetConstant`). Each is a function call + switch hop. Inlining
   these would yield smaller-but-additive wins.
3. **Auto-mode-inference from call sites.** Today the analyser
   only knows what `:- mode/1` declarations tell it. A whole-
   program pass could infer modes from observed call sites,
   eliminating the user's burden of declaring modes (and
   broadening phase-2/3 specialisation coverage). Big project --
   probably a separate sub-campaign.

The fact-source-bench Rprof profile (see WAM_R_TARGET.md "Rprof
profile of the WAM stepping engine") flags the dominant costs:

| Function | self.s | %self | Mode-info lever? |
|---------:|-------:|------:|------------------|
| `WamRuntime$step` | 1.500 | 29.0% | **YES** -- inline more ops when modes prove the slot shape |
| `WamRuntime$run` | 0.760 | 14.7% | indirectly (fewer step calls = fewer run iterations) |
| `WamRuntime$run_predicate` | 0.690 | 13.3% | no -- per-call overhead, not mode-driven |
| `WamRuntime$deref` | 0.380 | 7.3% | **YES** -- skip deref at known-bound slots |
| `WamRuntime$put_reg` | 0.260 | 5.0% | no -- already trivial |

Phase 4 should pick **one** candidate (likely #1, since it
multiplies the impact of every other specialisation), implement +
bench on a workload that's actually dominated by the targeted op.

## Phase 5+ — adaptive specialisation

Call-site-driven specialisation: the analyser tracks not just
"variables in this clause body" but "modes of every call site I
see," and the emitter picks specialised emission per call site.
This is a much bigger change (requires whole-program mode
propagation) and is deferred until earlier phases establish the
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
