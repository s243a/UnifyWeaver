# Cross-target sweep: PutStructure A-register bind-through + non-atom list heads

Date: 2026-06-12. Follow-up to the two latent bugs found in the Rust
target during the parity campaign (P4/P5): the M139/M140-class
PutStructure bind-through (cyclic `X = f(X)` when a goal structure
embedding a live head variable is built into the same A register) and
the put_list completion keyed on the head cell being an atom. All 15
other WAM targets were probed behaviorally through their real project
writers and toolchains (R=1/R=0 discriminator convention; hangs
time-boxed and counted as positive class-1 signals).

Probes:

```prolog
mk1(T, T).
bindthrough(X, W) :- mk1(f(X), W), X = 1.
p_bind(R) :- bindthrough(_X, W), W == f(1), R is 1.   % class 1
lhead([H|_], H).
p_list(R) :- lhead([7,8], V), V =:= 7, R is 1.        % class 2
p_listatom(R) :- lhead([a,b], V), V == a, R is 1.     % class-2 control
```

(Where a target lacks `==/2`, equivalent `=`/`nonvar` variants carried
the verdict; several targets ALSO turned out to be missing `==/2` —
see side findings.)

## Verdicts and actions

| Target | Class 1 (bind-through) | Class 2 (list heads) | Action |
|---|---|---|---|
| Go | **BUGGY** → fixed | clean | guard `i.Ai >= 100` in `wam_go_case('PutStructure')`; regression test `tests/test_wam_go_bindthrough.pl` |
| Scala | **BUGGY** → fixed | clean | guard `top.targetReg >= 100` in `finalizeBuild` (`runtime.scala.mustache`); verified via probe battery (8/8) |
| Kotlin | **BUGGY** → fixed (+ put-as-get defect → fixed) | clean | `bindTarget` A-guard; new `beginStructurePut` (put_structure/put_list were routed through get-style dispatch and wrong-failed on stale constants); `==/2`, `\==/2` added to the builtin table |
| Haskell | **BUGGY** → fixed | clean | guard `ai >= 100` in `addToBuilder` BuildStruct AND BuildList finalize arms (covers both pure and ST interpreters) |
| C | **BUGGY** (and UNTRAILED — backtracking hazard) → fixed | clean | guard `instr->as.functor.is_y_reg != 0` in `INSTR_PUT_STRUCTURE` |
| WAT | **BUGGY** → fixed | clean | guard `$ai >= 32` in `put_structure` and `put_list` cases (A = 0–31) |
| C++ | clean | clean | — |
| LLVM | clean (M140 fix regression-verified on this probe shape) | clean (distinct from ledger item 4) | — |
| F# | clean (cycle-check-guarded bind-through) | clean | see theoretical divergence below |
| Rust | fixed in P4/P5 (origin of the sweep) | fixed in P5 | — |
| ILasm | BLOCKED | BLOCKED | newly pinned structural defect: cross-predicate `call` self-loops (below) |
| Python | clean (A-reg guard already present, both interpreter loops) | clean | — |
| Lua | **BUGGY** (worst variant: no guard AND no deref — clobbered already-BOUND variables too) → fixed | clean | guard `top.target >= 101` + deref in `push_built_term` (`runtime.lua.mustache`); controls verified flipped; gates: lua generator suite (37/37) + lowered ite/t4/t5 |
| R | clean (F#-style cycle-check guard) | clean | inherits the F# theoretical divergence (side finding 4) |
| Clojure | clean (`a-slot?` register-class guard already present) | clean | side defect found in the opportunistic-lowering path (side finding 7) |
| Elixir | BLOCKED (interpreter mode) / clean (lowered mode) | BLOCKED / clean | interpreter mode structurally cannot cross-call (side finding 5) |

Verification: every fixed target re-probed through the same harness
(the bug-shape probe flips to the ISO answer; all controls — including
the legitimate X-register placeholder chaining the guard must preserve
— unchanged). Gates: cross-target conformance suite
(`CONFORMANCE_TARGETS=go,scala,haskell,c,wat` — all pass),
`test_wam_c_var_alias`, WAT lowered t4/t6, Kotlin target suite (9/9),
Go negcut + lowered t4/t5/t6/ite suites.

## The defect, once

Every buggy runtime shared the same ancestral pattern: put_structure
(and sometimes put_list) "helpfully" binds the target register's old
occupant when it derefs to an unbound variable, to support top-down
nested-term construction (`set_variable Xn` placeholder, then
`put_structure F, Xn`). That bind is correct ONLY for X/Y registers,
where placeholders live. A registers are argument staging: their old
occupant is an unrelated live variable (typically a clause-head
argument), and binding it to the structure being built — which often
embeds that same variable — creates a cyclic term. The canonical
trigger is any clause shaped like `p(X, ...) :- q(f(X), ...), use(X)`.
The fix is identical everywhere: condition the bind-through on the
register class.

## Side findings (filed here, not fixed in this sweep)

1. **`==/2` (and the `@<` family / `\==`) missing from several
   runtimes**: C (`wam_execute_builtin` falls through to false),
   Haskell (no `BuiltinCall "==/2"` arm; falls to `step _ _ _ =
   Nothing`), Kotlin (fixed for `==`/`\==` in this sweep). The shared
   compiler emits these as `builtin_call`; they fail closed — the same
   class-7 gap the Rust P3 sweep closed. A Rust-P3-style builtin
   parity sweep applies to each.
2. **Kotlin: callee bindings wiped on return** —
   `returnFromPredicate` restores the environment from the call-time
   snapshot, discarding bindings the callee made to caller variables
   (`mk1(7, V)` leaves V unbound). Structural; beyond the
   low-investment budget for Kotlin, needs its own campaign.
3. **ILasm: cross-predicate `call` self-loops** — each predicate gets
   its own code/label arrays, `call other/N` resolves against the
   LOCAL label table, and unknown labels default to index 0, so any
   cross-predicate call jumps to the calling predicate's own start
   (infinite loop at runtime, no generation-time warning). Distinct
   from the documented compound-term stubs; pin for the ILasm
   campaign resumption.
4. **F# theoretical divergence**: F# guards its bind-through with a
   cycle check (`containsVid`) rather than a register-class condition.
   A NON-cyclic variant — building a goal structure that does not
   contain X into an A register whose occupant is the live head var X
   — would still alias X to that structure. Not exercised by these
   probes; worth one targeted probe in the F# stream. **R shares this
   exact shape** (`wam_term_contains_var` cycle check in
   `runtime.R.mustache:426-443`).
5. **Elixir interpreter mode structurally cannot cross-call**: each
   predicate module embeds only its own code/labels, and the
   call/execute fallbacks resolve against the module-local map and
   fail on miss (`wam_elixir_target.pl:296/:322`). Any clause calling
   another predicate wrong-fails; only lowered mode (WamDispatcher) is
   multi-predicate-capable. ILasm-family finding (clean fail rather
   than self-loop).
6. **Lua Y registers are not environment-local**: `get_reg`/`put_reg`
   index a single flat `state.regs` while `Allocate` pushes a `locals`
   table nothing uses — a callee's Y1 clobbers the caller's Y1. This
   keeps Lua's cross-call probes (p_bind/p_bind2) failing even after
   the class-1 fix; the fix's verification rests on the single-call
   controls (p_bindctl2 flipped false→true). Kotlin-finding-#2 family;
   needs its own campaign.
7. **Clojure: numeric literals interned as atoms in the
   opportunistically-lowered path** — FIXED in the follow-up commit
   (`clj_lowered_literal` now emits bare numeric tokens as numbers;
   the WAM compiler quotes-and-marks atoms that merely look numeric,
   so an unquoted numeral is a real number; all 12 previously
   wrong-failing zero-arity probe wrappers verified correct) — default mode silently lowers
   eligible (incl. all zero-arity) predicates, and the lowered
   `put-constant` routes literals through `normalize-literal-atom`,
   which interns numerals as atoms; a subsequent `R is 1` compares
   atom-"1" to integer 1 and wrong-fails. The interpreter path is
   correct. (`wam_clojure_target.pl:208-210`,
   `wam_clojure_lowered_emitter.pl:1284`,
   `runtime.clj.mustache:52-54`.)
8. **Elixir lowered mode: `==/2` fails closed** — FIXED in the
   follow-up commit (`==/2`/`\==/2` arms via `deep_copy_value`
   structural comparison; the previously confounded probe battery now
   passes in full). The broader Rust-P3-style builtin parity sweep
   still applies to Elixir.
9. **Elixir target suite: 6 pre-existing failures on main**
   (unify/3 compound-clause check, LMDB e2e via :elmdb, three
   atom-interning literal checks, cons-tag aliasing) — verified
   identical on the unmodified parent tree; flag to the Elixir
   stream.
