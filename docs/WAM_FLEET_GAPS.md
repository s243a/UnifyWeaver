<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# WAM fleet — deficiency classes from the whole-program exercise

**Living document.** In September 2026 UnifyWeaver transpiled a real
~400-line production program — peerhailer's CLI argument parser,
`examples/cli_args/cli_args.pl`, oracle-verified in step A1 — through two
lanes of the JavaScript family: the `wam_javascript` interpreter (step A2)
and the `typescript`/`vanilla_js`/`annotated_js` pattern lane (step A3).
Both now pass the 17-test production corpus and a seeded 5067-line
differential against the vendored JS oracle with **0 divergences**.

Getting there surfaced **deficiency classes**, not one-off bugs. Every
class was found in a mature, conformance-green target; each one almost
certainly exists in sibling targets, because **no other target has ever
been asked to run a whole real program**. This document names the classes,
records per-target verification status, and says how to repeat the
exercise against any other target.

Related docs:

- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md) — the
  runtime conventions (now including §7/§8, adopted from these findings).
- [`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md) — hand-off task
  cards for the *pre-exercise* fleet gaps (kernels, conformance arms,
  parsers). This document is the *post-exercise* deficiency census; fixes
  for rows here should eventually become cards there.
- [`WAM_JAVASCRIPT_STATUS.md`](WAM_JAVASCRIPT_STATUS.md) — the exemplar
  per-target status doc (updated with the A2 results on its own branch).
- `docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md` — the full A3 design
  record (per-predicate compile matrix → named gap catalogue → oracle
  differential); `docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md` — the
  D1–D38 done-ledger.

---

## Why WAM targets rank above pattern-only targets — and what even they were missing

The hybrid WAM targets are the **completeness band** of the fleet: they
consume shared bytecode with real unification, backtracking, and cut, so
any first-order program the shared compiler can emit is *semantically*
within reach — no shape recognizers, no refusals. The pattern-only
targets are template engines with shape recognizers; step A3 measured
that honestly (before its fixes, the TypeScript lane could transpile
**2 of 43** predicates of the parser, and its flagship clause-body path
had emitted unparseable code for years without a test noticing).

The exercise's second finding cuts the other way: **the WAM band's
completeness had never been exercised either.** The JS WAM was
conformance-green (48/48) and still needed three real runtime fixes
before the parser ran:

1. a missing string builtin the conformance programs never call,
2. a register-discipline hole that only a *large* ground-fact predicate
   triggers,
3. a control-transfer bug only a *last-goal* builtin call triggers.

Synthetic conformance suites exercise conventions; a real program
exercises the composition of everything at once. That is Class C below,
and it is the reason `examples/cli_args/` is now a fleet-wide benchmark
rather than a one-off demo.

---

## The deficiency classes

### Class A — WAM runtime bugs (found in the JS runtime; siblings suspect until checked)

All three were found in `wam_javascript` while running the parser, fixed
in commit `3cbd16219` (probe-pinned: `probe_sub_string/0`,
`probe_y_preserve/0`, `probe_tail_builtin/0` in
`tests/test_wam_javascript_builtins.pl`).

**A1 — `sub_string/5` missing.** The JS runtime had only `sub_atom/5`.
Real programs use SWI string builtins that synthetic conformance queries
don't. Fleet grep: **no other runtime implements `sub_string/5` except
C++** (`wam_cpp_target.pl:7935,7998` — an alias of its `sub_atom/5`, so
results intern as atoms). Note `sub_string/5` is **not** in the shared
`is_builtin_pred/2` table (`wam_target.pl`), so it reaches every runtime
as `call`/`execute` of an unlabelled name — which is what makes A3 below
load-bearing.

**A2 — Y-register clobber across `Call` of a no-`Allocate` fact.** The
shared fact compiler (`wam_target.pl`) emits a large ground fact (e.g.
`default_registry/1`, a 7-entry nested registry) with **no `Allocate`**
and with **more than 99 X-register placeholders**. In every runtime that
maps registers numerically as `A_n→n, X_n→n+100, Y_n→n+200` (or the
0-based/128-slot variants), **X101 and up alias into the Y range** —
X101 ≡ Y1. Calling such a fact therefore overwrites the caller's
permanent (Y) registers, and `Allocate`-framing offers no protection
because the callee never allocates. The JS runtime's workaround:
`Call`/`CallPc` snapshot the Y range, `Proceed` restores
(`templates/targets/javascript_wam/runtime.js.mustache`, `push_y_save` /
`proceed_to_cp`). Runtimes with *string-named* registers or fully
segregated banks (`"X101"` can never collide with `"Y1"`) are structurally
immune to the aliasing form.

**A3 — `Execute` of a builtin does not return to the continuation.** The
shared emitter compiles a last goal as `deallocate` + `execute P/N`. When
`P/N` is a builtin the runtime implements but the program defines no
label for (anything outside `is_builtin_pred/2` — `sub_string/5`,
`catch/3`, runtime-extended builtins), the runtime's `Execute` arm must
dispatch the builtin **and then take `Proceed`'s return path** (jump to
CP; restore the caller). The JS runtime instead set `halt`, killing the
success path of every last-goal builtin. Sibling behaviors range from
"partially handled" to "silently fails" to "jumps to instruction 0" —
see the matrix.

**A4 — string-type fidelity ladder.** Since D34–D37, the shared emitter
spells Prolog string literals as double-quoted WAM tokens
(`wam_target:quote_wam_constant/2`); the shared classifier still returns
`atom(Name)` plus a `wam_constant_token_is_string/1` side signal, so
non-string-aware runtimes degrade gracefully (full-fleet conformance
stayed ALL_PASS). Each runtime sits on a rung:

- **rung 0** — no string term tag; strings intern as atoms;
- **rung 1** — distinct tag, but compiled `"foo"` literals still collapse
  to atoms (the runtime ignores the token signal);
- **rung 2** — full: distinct tag + literals + SWI standard order
  (`Var < Number < String < Atom < Compound`), as `wam_javascript`
  after D34–D37.

Today the fleet is: `wam_javascript` rung 2; **every other WAM runtime
rung 0** (verified by grep for string term constructors in all 17
sibling runtimes — none has one). No runtime currently sits on rung 1.

### Class B — pattern-path deficiencies (the G-A3 series)

Closed in the `typescript` family ONLY (commit `e90928a9b`; regression
suite `tests/core/test_typescript_cli_args_shapes.pl`, 116 tests). Any
WAM target whose language **also** has a pattern target is suspect for
the same absences:

- cross-predicate calls in clause bodies and guard positions, lowered by
  the callee's output count, with honest failure semantics (G-A3-6);
- multi-output accumulator loops via positional-tuple returns (G-A3-9);
- semidet-with-outputs: a failure sentinel plus a can-fail least
  fixpoint over the call graph (G-A3-18);
- compound terms as distinguishable runtime data — `some(V)`/`ok`/`err(M)`
  vs atoms/lists/booleans (G-A3-12);
- ground-fact tables inlined as match-not-call (G-A3-19);
- in-order guard placement (no hoisting over the declarations a guard
  reads) (G-A3-6);

and two **hazard** patterns worth checking everywhere:

- **B-H1 — a fact-compilation fallback that *executes the predicate
  being compiled*** (`findall(..., (functor(Goal,P,A), call(Goal), ...))`).
  TypeScript's did — it hung the compiler at 1.5 GB on ordinary input
  (G-A3-8, now guarded). Fleet grep: **`haskell_target.pl` has the same
  pattern** (`compile_facts_to_haskell`, lines ~122–132, reachable as the
  dispatch *fallback* at lines 104–114). Every other audited pattern
  target (python, rust, go, lua, elixir, scala, kotlin, fsharp, clojure,
  c, cpp, r) enumerates `user:clause(Head, true)` instead — safe from
  execution, though a rule predicate that falls to a facts path still
  yields a silently *empty/wrong* table rather than a refusal;
- **B-H2 — emitted-code test suites that assert substrings but never
  syntax-check or run the output.** TypeScript shipped unparseable code
  for years that way; the fix was a `node --check`-style gate. The WAM
  lane is partially defended by the run-based conformance harness — but
  `wam_lua` and `wam_clojure` **have no conformance arm at all**
  (CONF-LUA / CONF-CLOJURE still open in
  [`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md)), so their emitted
  output is never executed by the shared harness. Pattern-lane suites
  outside the TS family have not been audited for a syntax gate; treat
  them as suspect.

### Class C — the methodology finding

**A real program is a stronger test than pattern suites or conformance
fixtures.** `examples/cli_args/` is a reusable fleet-wide benchmark:

- `cli_args.pl` — frozen, oracle-exact Prolog reference (43 predicates;
  if-then-else, multi-output accumulator loops, compound results,
  semidet lookups, string builtins — no cuts, no exceptions);
- `oracle/` — vendored read-only JS oracle (peerhailer `08ad35e`);
- `test_cli_args.pl` — the 17-test contract corpus;
- `gen_cases.mjs` + `run_differential.sh` — seeded 5067-line
  differential with a semantic JSONL comparer.

Three builds pass it today: the typescript/vanilla/annotated pattern
lane (`examples/cli_args/patternjs/`) and the JS WAM interpreter
(`examples/cli_args/wamjs/`).

**To run it against a new target:**

1. Compile `examples/cli_args/cli_args.pl` whole through the target
   (module-strip + `compile_module`-equivalent for a pattern lane;
   ordinary WAM emission for a hybrid lane). Model the build on
   `examples/cli_args/wamjs/build.sh` / `patternjs/build.sh` — one
   command, reproducible, syntax-gated (`node --check` analogue).
2. Write a term↔host edge shim that converts argv in and
   `ok(Positional, Flags)` / `error(Message)` out, and **nothing else**
   — no parse logic in the shim (the A2/A3 shims are the reference).
3. Point the corpus at it (import swap only) — 17/17 required.
4. Run the differential with the same seed — 0 divergences, 0 message
   mismatches required.
5. While doing so, census what breaks the way
   `docs/proposals/A3_PATTERN_TRANSPILE_REPORT.md` does: a
   per-predicate compile matrix, a *named* gap catalogue with
   probe-pinned reproductions, then the oracle differential as the
   acceptance gate. Fold new runtime-bug classes into this document and
   the target's `WAM_<TARGET>_STATUS.md`.

---

## Fleet matrix — target × deficiency class

Status vocabulary: **fixed** (defect closed, probe-pinned) ·
**handled** (runtime already does the right thing; evidence cited) ·
**verified** (defect confirmed by source reading; file:line cited) ·
**suspected** (same emitter pattern reaches the runtime; not yet
source-confirmed) · **absent** (structurally immune; reason cited).

| Target | A1 `sub_string/5` | A2 Y-clobber (no-`Allocate` fact) | A3 `Execute` of a builtin | A4 rung | B-H1 pattern fact fallback executes |
|---|---|---|---|---|---|
| **javascript** | fixed | fixed (Call snapshots Y, Proceed restores) | fixed (proceeds to CP) | **2** | n/a (TS family lane; G-A3-8 closed) |
| **cpp** | **handled** — alias of `sub_atom/5` (`wam_cpp_target.pl:7935,7998`); results are atoms | absent — string-named regs, per-frame `y_regs` (`:3321-3336`) | **handled** — general builtin fallback then proceed-to-CP (`:8035-8040`) | 0 | absent — `clause(Head,true)` |
| **rust** | **verified missing** — no `sub_atom` either | absent (aliasing form) — string-named regs; Y in topmost env frame (`state.rs.mustache:2398-2406`) | **verified partial** — only `catch/3`,`throw/1`,`succ/2` route (`wam_rust_target.pl:795-838,6733-6735`); anything else silently fails | 0 | absent — `clause(Head,true)` |
| **haskell** | verified missing | **verified (aliasing form, open)** — `+100/+200` encoding (`wam_haskell_target.pl:6806-6807`); `putReg` treats id ≥ 200 as Y in the *topmost env frame*, so a big fact's X100+ writes land in the caller's frame; with no frame the write is *silently dropped*. **Frameless-Y form: verified broken → fixed 2026-09** (ITE barrier levels moved onto the choice point in BOTH interpreters; GHC verification pending) | verified partial — ISO meta subset only (`:2231-2245,3497-3516`); else silent fail | 0 | **verified PRESENT** — `haskell_target.pl:104-114` falls back to `compile_facts_to_haskell`, which `call(Goal)`s the predicate (`:127-132`) |
| **lua** | verified missing (no `sub_atom`; ~10-builtin dispatch) | **verified** — `+100/+200` (`wam_lua_target.pl:136-137`); flat `state.regs` (`runtime.lua.mustache:158-159`); `Allocate` frame stores only `cp` (`:1023`); no Y save on `Call` (`:1127-1133`) | **verified** — `Execute`/`Call` have *no* builtin fallback: unresolved label = silent goal failure (`:1141-1148`) | 0 | absent — `clause(Head,true)` |
| **python** | verified missing | **verified (aliasing form, open)** — X window 128 (`wam_python_target.pl:2205-2208`, X_k→128+k so **X173 ≡ Y1**); Y ≥ 301 written into the *current env frame* (`wam_python_runtime/WamRuntime.py:171-194`). **Frameless-Y form: REPRODUCED as a live wrong answer → fixed 2026-09** (`ChoicePoint.levels`; probe `tests/test_wam_python_frameless_ite_level.pl`) | **verified** — `execute` with no label returns `False`, no builtin/foreign fallback (`wam_python_target.pl:400-406`) | 0 | absent — `clause(Head,true)` |
| **go** | verified missing (has `sub_atom`, `state.go.mustache:2792`) | **verified** — X_n→n+99 / Y_n→n+199 (`wam_go_target.pl:2103-2105`, **X101 ≡ Y1**); flat `vm.Regs`; Y saved only across the *callee's own* `Allocate` (`:2558-2598`) | **handled for known builtins** — dedicated `BuiltinExecute` proceeds to CP (`instructions.go.mustache:130-143`, `wam_go_target.pl:2729-2738`); a builtin missed by translation-time classification silently fails (`:2661-2669`) | 0 | absent — `clause(Head,true)` |
| **scala** | verified missing (has `sub_atom` intercept) | **verified** — `+100/+200` (`wam_scala_target.pl:106-107`); flat `s.regs`; `Allocate` *copies* Y 201–299 into the frame but `Deallocate` never restores (`runtime.scala.mustache:1185-1196`) | **handled for its 14-builtin intercept** — `interceptedExecuteBuiltin` routed from both `Call` and `Execute` with return plumbing (`:1158-1175,1846-1870`); outside the set → silent backtrack | 0 | absent — `clause(Head,true)` |
| **r** | verified missing (has `sub_atom` in `call_library`) | **verified (aliasing form, open)** — `+100/+200` (`wam_r_target.pl:208-209`); Y ≥ 201 in the topmost frame's `ys` (`runtime.R.mustache:289-299`). **Frameless-Y form: unreachable 2026-09** — wam_r never passes `ite_use_y_level(true)` (all compiles use `[]`), and `wam_parts_to_r/2` has no `get_level` clause so the shape would `stop()` loudly as `Raw`. Routing immunity only: `put_reg` does write the topmost frame. Probe `tests/test_wam_r_frameless_ite_level.pl` | **handled** — `Execute` falls through label → dynamic → `call_library` with the Proceed protocol (`runtime.R.mustache:1424-1467`) | 0 | absent — `clause(Head,true)` |
| **elixir** | verified missing | **verified** — X→+100 / Y→+200 (`wam_elixir_utils.pl:46-52`, **X101 ≡ Y1** despite the "avoid aliasing" comment); `y_regs` swapped *only* at `Allocate` (`wam_elixir_target.pl:376-382`) | verified partial — `call/N`, `catch/3`, `throw/1` route; else `:fail` (`wam_elixir_target.pl:346-370`) | 0 | absent — `clause(Head,true)` |
| **clojure** | verified missing | suspected-low — registers are string-named, env-stack of maps (`runtime.clj.mustache:2846-2855`); aliasing form absent, frameless-Y-write form unverified | **verified** — `:execute` special-cases only `variant/2`; unresolved label → backtrack (`runtime.clj.mustache:3275-3283`) | 0 | absent — `clause(Head,true)` |
| **kotlin** | verified missing | absent (aliasing form) — string-named regs, per-frame env slots (`WamRuntime.kt.mustache:104-110`) | suspected — native/bytecode dispatch not audited for the tail-builtin shape | 0 | absent — `clause(Head,true)` |
| **fsharp** | verified missing | **verified (aliasing form, open — was "suspected")** — `+100/+200` (`wam_fsharp_target.pl:4190-4191`) + per-frame `EfYRegs` (`:1136-1143`); the missing threshold is in `bindings/fsharp_wam_bindings.pl:546-592` — `getReg`/`putReg` branch on `n >= 201` and hit `s.WsStack`'s head frame. **Frameless-Y form: unreachable 2026-09** — wam_fsharp never passes `ite_use_y_level(true)` (all compiles use `[]`), and the emitter has no `get_level` clause so the shape becomes a warned `(* UNKNOWN *) Proceed` stub. Routing immunity only. Probe `tests/test_wam_fsharp_frameless_ite_level.pl` | verified partial — `isIsoMetaBuiltin` (catch/throw/succ) with correct tail return (`:1041-1070`); else silent fail (`:1072-1083`) | 0 | absent — `clause(Head,true)` |
| **c** | verified missing | absent (aliasing form) — segregated A/X/Y banks via `is_y` flags (`wam_c_target.pl:1684-1710`); **suspected overflow hazard**: fixed 256-slot arrays (`wam_runtime.h.mustache:11,81-82`), a >256-X-register fact would index out of bounds (no bounds check found) | **verified** — `INSTR_EXECUTE` special-cases aggregates, else label or `return false` (`wam_c_target.pl:2257-2270`) | 0 | absent — `clause(Head,true)` |
| **wat** | verified missing | suspected — memory-banked registers not audited | **verified (worst form)** — `resolve_label` falls back to **PC = 0** for an unknown label (`wam_wat_target.pl:1204-1209,348-351`): `execute <unlabelled builtin>` silently jumps to instruction 0. Fused `builtin_proceed`/`deallocate_builtin_proceed` cover only `is_builtin_pred` last-goals (`:760-817`) | 0 | not audited |
| **llvm** | verified missing | suspected | suspected | 0 | not audited |
| **jvm** | verified missing | suspected | suspected | 0 | not audited |
| **ilasm** | verified missing | suspected | suspected | 0 | not audited |

Notes on the matrix:

- "verified missing" for A1 is a fleet-wide grep result: `sub_string/5`
  is dispatched nowhere except `wam_cpp_target.pl` and (post-A2) the JS
  runtime. Only six runtimes even have `sub_atom/5` (scala, r,
  javascript, go, clojure†, cpp). †clojure's `sub_atom` appears in its
  builtin table, not its Call/Execute arms.
- The A2 column distinguishes the **aliasing form** (numeric X window
  spills into the Y range — the exact cli_args failure) from the
  **frameless-Y-write form** (a callee without a frame writing Y names
  into the caller's frame). Named-register runtimes are immune to the
  first.

  **Correction (2026-09-03).** This document originally dismissed the
  second form on the grounds that it "requires bytecode that names Y
  registers outside an `Allocate` body, which the current fact compiler
  does not emit". The fact compiler indeed does not — but
  **`compile_if_then_else/7` does.** The shared emitter reserves a
  permanent Y for the if-then-else barrier *after* it has decided whether
  the clause needs an environment, so under `ite_use_y_level(true)` it
  emits `get_level Yn` … `cut Yn` into `Allocate`-less clauses. wam_rust
  hit it on `examples/pkg_resolver/resolver.pl` (15 differential
  divergences, ledger D50); this round reproduced it as a **live wrong
  answer on wam_python** and verified it from the emitted bytecode on
  **wam_haskell**. Both are fixed the way wam_rust was — the barrier level
  lives on the if-then-else's own choice point, never in a register.

  Which targets are exposed is decided by one flag: `ite_use_y_level(true)`
  is passed by **cpp, go, haskell, javascript, llvm, lua, python, rust**
  and *not* by **r, fsharp** (both compile with a literal `[]`), which is
  why those two are unreachable rather than immune — their runtimes do
  route Y writes to the topmost frame. A minimal reproduction of the shape
  (usable against any target) is:

  ```prolog
  lt(A, B) :- A < B.
  sat(_V, any).
  sat(V, gte(G)) :- \+ lt(V, G).            % Allocate-less, emits get_level Y1
  pick(Ver, C, Tag, Out) :- sat(Ver, C), Tag = Out.   % caller HAS a frame
  ```

  `pick(3, gte(1), tagX, Out)` must give `Out = tagX`; a runtime with the
  defect returns a small integer (the choice-point depth) or fails. A scan
  of every predicate in `examples/pkg_resolver/resolver.pl` (79) and
  `examples/cli_args/cli_args.pl` (43) finds **zero** `Allocate`-less
  clauses naming a Y register under `[]`, and exactly one under
  `ite_use_y_level(true)` — `satisfies/2`, the clause wam_rust tripped on.

  Still to check for this form — every one of these passes the flag and
  none has been re-audited against the ITE barrier specifically:
  **lua, llvm, go, cpp, javascript**. Two of them look worth doing first:
  `wam_lua` writes Y into a flat `state.regs` with no save on `Call` at
  all *and* has no conformance arm (B-H2), and `wam_go` looks positively
  exposed on a read: Y registers live in the flat `vm.Regs[200:300]`, and
  the only protection is `Allocate` copying that range into
  `env.SavedYRegs` and `Deallocate` copying it back
  (`wam_go_target.pl` `Allocate`/`Deallocate` cases) — which an
  `Allocate`-less if-then-else clause never executes, so its `GetLevel Yn`
  writes straight over the caller's live `vm.Regs[200+n]`. Not verified by
  execution here (wam_go is owned by a concurrent round); flagged for
  whoever holds it. (`wam_javascript` snapshots the Y range at `Call` and
  restores at `Proceed`, so the caller's frame is repaired on return even
  if the callee scribbles on it; `wam_rust` is fixed; `wam_cpp` uses
  string-named per-frame `y_regs`.) Beyond those, **clojure / kotlin / c /
  wat / jvm / ilasm** were scored on the aliasing form alone, though none
  of them enables the flag today.
- The B column records only the two grep-able hazards. The full G-A3
  machinery gap (cross-calls, multi-output tuples, semidet sentinel,
  compound tags) should be presumed **absent outside the TS family**
  until a target's pattern lane is put through the Class-C benchmark.
- Class B does not apply to the WAM lane itself: the WAM runtimes
  execute real unification, so those shapes work there by construction —
  that is the completeness-band argument above.

---

## Priority reading of the matrix

1. **A3 in wat** is silent control-flow corruption, not a failure — the
   most dangerous single cell. Cheapest fix: make `resolve_label`
   refusal loud (or emit a fail instruction), then add builtin routing.
2. **A2 in haskell / python / go / scala / r / elixir / lua** is a
   correctness landmine on exactly the program shape the fleet has never
   run: any caller holding Y registers across a call to a large ground
   fact. The JS Call-snapshot workaround ports directly to the
   flat-register runtimes; the frame-based ones (haskell, python, r,
   fsharp) instead need the X window widened or the encoding made
   non-aliasing (see `WAM_BACKEND_CONVENTIONS.md` §8). That is the
   **aliasing** form and it is still open everywhere it is marked
   verified.
2b. **A2's frameless-Y form is the one that has actually bitten**, three
   times now (rust D50, python, haskell) — and it needs *no* big ground
   fact, just an if-then-else or a `\+` in a clause with no permanent
   variables. Its fix is not the X window: it is to stop putting the
   barrier in a register at all (the wam_rust `ChoicePoint::levels`
   model). **Highest-value remaining audit: `wam_lua` and `wam_llvm`**,
   which pass `ite_use_y_level(true)` and have not been re-checked for
   this shape (lua also has no conformance arm at all — see B-H2). Use
   the four-line reproduction under the matrix notes.
3. **A3 elsewhere**: silent failure of last-goal builtins. Runtimes with
   partial routing (rust, haskell, fsharp, elixir) have the pattern to
   extend; go's `BuiltinExecute` and cpp's fallback-then-proceed are the
   reference implementations.
4. **A1/A4** matter the moment any real, string-using program is pushed
   through a target — i.e. the moment step 1 of the Class-C recipe is
   attempted.
5. **B-H1 in haskell_target.pl** should get the same ground-fact guard
   typescript got (G-A3-8): facts path only when every clause body is
   `true`, loud refusal otherwise.

## Document status

Created 2026-09-01 from the A2/A3 whole-program exercise (commits
`3cbd16219` wamjs, `e90928a9b` pattern lane). Class A/B statuses are
source-verified against the tree at that date where a file:line is
cited, and marked suspected otherwise; deep reads were done on the lua,
rust, haskell, python, go, scala, r, elixir, c, cpp and wat runtimes.
Update the matrix as targets are put through the Class-C benchmark.

2026-09-03 — **A2 frameless-Y re-audit (haskell / python / r / fsharp)**,
prompted by the wam_rust D50 finding that this document's
"unreachable" note was wrong. Verdicts: **python broken → fixed**
(reproduced as a live wrong answer against SWI), **haskell broken →
fixed** (verified from the emitted bytecode and both interpreter
sources; execution pending GHC, which the audit environment lacks),
**r unreachable** and **fsharp unreachable** (neither enables
`ite_use_y_level(true)`; both refuse the instruction loudly if it ever
arrives — routing immunity, not structural). Each verdict is pinned by
a probe: `tests/test_wam_{python,haskell,r,fsharp}_frameless_ite_level.pl`.
Two side findings recorded in the per-target docs: F#'s A2 aliasing cell
is upgraded from *suspected* to *verified* (the getReg/putReg threshold
the first audit could not find lives in
`src/unifyweaver/bindings/fsharp_wam_bindings.pl`, not in the target),
and wam_haskell's pure interpreter truncated its NEWEST-FIRST
choice-point list with `take n` instead of `drop (len - n)` — right only
when `n == 0` — which is fixed alongside.
