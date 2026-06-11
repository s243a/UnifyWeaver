# WAM Correctness Campaign Handoff (M137–M149)

Status as of 2026-06-11. This campaign ran across PRs #2929, #2964,
#2967, #2976, #2978, #2985, #2986, #2988, #2995, #2998 and swept all
16 WAM targets for a family of variable-binding and control-flow
correctness bugs. Every found bug is either fixed or precisely
documented below.

## The recurring bug classes

Probe-driven debugging found the same few classes independently
re-implemented across targets. If you add a target or a runtime path,
check these first:

1. **Shallow equality at depth (M137/M138).** Comparing values without
   dereferencing Ref/forwarding chains at every level — `[a,b] ==
   [a,b]` false, sort/2 dedup keeping duplicates. Watch every
   `value_equals`-style helper's recursive arg loop.
2. **Equality where unification is required (M138).** `get_value`,
   read-mode `unify_value`, `arg/3` with a bound third argument are
   UNIFY operations. A `p(X, X)` head called with `p(f(A), f(B))`
   must bind `A = B`, not compare.
3. **Bind-through of a register's old occupant (M139/M140).**
   Overwriting an argument register (put_value/put_variable/
   put_constant/...) must never bind the variable that *used to* be
   there. LLVM's bind-through helper produced self-referential cells
   and silent variable aliasing. Distinguish register classes: A-reg
   writes are staging; X/Y writes may legitimately participate in
   top-down structure chaining.
4. **Bindings not following alias chains to the end (M140).** After
   `X = Y` creates `cell1 -> Ref{cell2}`, a later `X = 42` must land
   in cell2 (which Y reads), not cell1. Check every bind helper.
5. **Var-var unification that copies instead of aliasing (M143).**
   Unifying two unbound cells by copying one unbound marker over the
   other leaves the variables independent — smoking gun:
   `X = Y, X = 1, Y = 2` succeeds. Found in C and C++; fixed with a
   VAL_REF link (C) and alias-pairs + write-through propagation
   (C++, whose cell model has no Ref tag).
6. **If-then-else commit missing (M144 class).** `cut_ite` translated
   to NoOp / fail-closed / null slots, so a successful condition
   backtracks into the else branch. Found in SIX targets: LLVM
   (pre-M17), Rust, Elixir, Kotlin, ILasm, and (as crash) Haskell ST.
   Probe: `( 1 =:= 1 -> R is 1 ; R is 0 )` with R=0 must FAIL.
7. **Missing builtin tables.** Type checks (`var/1`, `nonvar/1`,
   `atom/1`...) and arithmetic absent from a runtime's builtin
   dispatch, failing closed or crashing. Found in Elixir, Kotlin,
   ILasm.

## The probe battery

Pure-Prolog probes, each queried with R pre-bound to 1 (expect
success) AND 0 (expect failure) so wrong-branch results are
distinguishable from plain failure:

```prolog
p1(R) :- var(X), X = done, R is 1.
p2(R) :- var(X), var(X), R is 1.
p3(R) :- ( var(X), nonvar(X) -> R is 0 ; R is 1 ).
p4(R) :- var(X), atom(foo), ( var(X) -> R is 1 ; R is 0 ).
p5(R) :- X = Y, X = 42, ( Y =:= 42 -> R is 1 ; R is 0 ).
p6(R) :- ( 1 =:= 1 -> R is 1 ; R is 0 ).        % ITE commit
gun(R) :- X = Y, X = 1, Y = 2, R is 1.           % must FAIL
chain3(R) :- X = Y, Y = Z, X = 42, ( Z =:= 42 -> R is 1 ; R is 0 ).
```

Run them through each target's real project writer + toolchain in
DEFAULT emit mode (the interpreter is the semantic baseline); lowered
modes separately. Generation generally needs `LC_ALL=C.UTF-8`.

## Per-target status (sweep verdicts + fixes)

| Target | Sweep verdict | Fixes landed |
|---|---|---|
| LLVM | buggy (classes 1–4, 7) | M137–M142 (#2929/#2964/#2967/#2976/#2978); also is_list, root-spec dedup, exec-harness tag-checks |
| C | buggy (class 5) | M143 (#2985) |
| C++ | buggy (class 5) | M143 (#2985): alias pairs + write-through; all 13 inline trail unwinds centralized |
| Rust | buggy (class 6) | M144 (#2986): GetLevel/CutTo + ite_use_y_level default |
| Haskell | ST crash (class 6 adjacent) | M145 (#2986): env-frame-aware reg access in stepST GetLevel/Cut |
| Scala | lowered emitter syntax | M146 (#2986): `locally { }` |
| Elixir | buggy (classes 6, 7 + module names + allocate) | M147 (#2988) |
| Kotlin | buggy (classes 6, 7) | M148 (#2995) |
| ILasm | 14 defects incl. classes 5 (proactively), 6, 7 | M147 one-liner + M149 full repair (#2998) |
| Python, Go, F#, Lua, R, Clojure, WAT | CLEAN | — |
| JVM | structurally non-executable | documented only (see below) |

## Remaining ledger (priority-ordered suggestions)

1. **JVM target ground-up campaign.** The target emits method-less
   code fragments (no class headers, operands lost into `;` comments,
   harmful `ldc` catch-all). Krakatau v2 (`krak2`) builds from source
   with cargo and rejects everything. Not fixable mechanically — an
   engine has to be written (model on the ILasm M149 shape: types +
   state + step + run_loop + builtin table).
2. **ILasm compound terms.** get/put_structure + unify_* are
   pre-existing heap-marker stubs; get_value's bind path is
   register-level. The M149 exec smoke test is the regression floor.
3. **WAT read-mode compound argument unification** is a no-op (no
   S-register) — breaks `member/2`-shaped code. Documented in
   docs/WAM_SWITCH_INDEXING_CROSS_TARGET.md.
4. **LLVM put_list/get_list vs put_structure cons representations**
   differ (heap-marker vs Compound) — `is_list([a])` standalone was
   fixed in M141, but `L = [11,22,33], L = [H|_]` cross-representation
   unification remains the known limitation from the M104-era notes.
5. **Pre-existing failures verified unrelated to this campaign:**
   `test_wam_rust_runtime` e2e (output mismatch), Scala runtime
   smoke `builtin_format` (`wam_format_combo/2`), Haskell
   `lowered_phase4` exits false despite all-PASS output.
6. **LLVM `@value_equals` call-site audit follow-on:** sort dedup and
   strict-eq now deep-compare, but `==`-style var identity for bare
   (non-Ref) register sentinels remains identity-less (degenerate
   case; put_variable always heap-allocates, so rarely observable).

## Methodology notes that paid off repeatedly

- **Full-suite runs are the real gate.** Three regressions and two
  masked latent bugs were caught only by the 870-test LLVM suite (it
  aborts at first failure; run detached, ~30 min).
- **Pre-existing-failure discipline:** before chasing a failure,
  re-run it on the unmodified parent commit (`git checkout -q HEAD~1`
  or stash). Five session red herrings dissolved this way.
- **Harness conventions bite:** LLVM exec drivers read result
  registers (now tag-checked, 254 = wrong tag); results must flow
  through a final `is/2`. Generation templates contain UTF-8 —
  writers need `encoding(utf8)`, environments need `LC_ALL=C.UTF-8`.
- **Toolchain provisioning (this container):** swipl/apt, dotnet/apt,
  ghc + libghc-{hashable,unordered-containers,async,parallel}-dev,
  llvm (llc), lua5.4, r-base-core, elixir, wabt, clojure (run via
  `java -cp /usr/share/java/clojure-*.jar`), mono-devel (ilasm),
  Scala 2.13.16 (GitHub dist tarball; coursier's native launcher
  fails on TLS-intercepting proxies), kotlinc 2.1.0 (JetBrains
  GitHub release zip), lmdbjava + jnr/asm jars (Maven Central via
  curl).
