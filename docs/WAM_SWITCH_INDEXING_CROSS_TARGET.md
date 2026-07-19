# Cross-target audit: WAM first-argument-indexing instruction handlers

**Status:** findings / triage (no code changes in this document).
**Origin:** while fixing the WAM **Scala** target, two first-argument
indexing instructions (`switch_on_constant_fallthrough`, `switch_on_term_a2`)
were found to be unhandled and to break first-argument-indexed predicates
such as `member/2`. This document audits whether the **other** WAM targets
have the same gap, and — importantly — quantifies the *actual* correctness
impact, which is narrower than a naïve "unhandled instruction" count
suggests.

## Background

The shared WAM compiler (`src/unifyweaver/targets/wam_target.pl`) emits a
family of first-argument (A1) and second-argument (A2) indexing
instructions for multi-clause predicates:

```
switch_on_constant            switch_on_constant_a2
switch_on_constant_fallthrough  switch_on_constant_a2_fallthrough
switch_on_term                switch_on_term_a2
switch_on_structure           switch_on_structure_a2
```

These are an **optimization**: each is immediately followed by the
`try_me_else` / `retry_me_else` / `trust_me` clause chain. If a target
ignores the switch instruction but still runs the chain, the predicate
produces **correct results** — it just enumerates clauses linearly
instead of jumping to the indexed clause. `member/2`, for example,
compiles to:

```
mem/2:
    switch_on_term_a2 0  0  default      <- A2 (list) index; degenerate (all-default)
    try_me_else L_mem_2_2
    get_variable X1, A1
    get_list A2
    ...
```

The `switch_on_term_a2 0 0 default` here is **degenerate** (no const/struct
cases, list routes to `default` = fall through), so even a correct handler
just falls through to `try_me_else`.

## The key distinction: *drop* vs *harmful* fallback

A target only mis-handles these instructions if its codegen has **no
explicit handler** for them, in which case they hit the target's
catch-all. **The catch-all's behaviour decides whether this is a
correctness bug or merely a missed optimization:**

- **Drop-style catch-all** (emit a comment / skip the instruction, keeping
  PC/label alignment consistent): **correctness-safe.** Execution falls
  through to the clause chain. The only loss is the indexing speedup.
- **Harmful catch-all** (emit an instruction the step loop mis-executes —
  `fail`, `proceed`, an extra `allocate`, operand-stack pollution, etc.):
  **correctness bug.** This is what broke Scala (`Raw(...)` → the step
  loop failed) and what makes `member/2` return wrong answers.

### Empirical confirmation

- **Scala (harmful, now fixed):** `switch_on_term_a2` → `Raw(...)`; the
  step loop backtracked on the unknown `Raw`, so `member(a,[a,b,c])`
  returned `false`. Fixed by handling the a2/fallthrough variants
  (PR adding a `reg` field to `SwitchOnConstant`/`SwitchOnTerm`).
- **Python (drop, correctness-safe):** `switch_on_term_a2` →
  `# SKIP ITEM: ...` (instruction dropped). Running `member` directly
  against the generated Python runtime:
  `member(a,[a,b,c]) = True`, `member(c,[a,b,c]) = True`,
  `member(z,[a,b,c]) = False` — **correct**. The dropped degenerate
  switch is harmless; the clause chain does the work.

This is the central correction to a naïve audit: **most "unhandled" cases
are optimization gaps, not correctness bugs.** Only harmful-catch-all
targets are actually broken.

## Coverage matrix

Handler present (Y) / falls to catch-all (N), per target's real code
emitter (not the token-parser predicates several targets also define).
`const_ft` = `switch_on_constant_fallthrough`, `const_a2_ft` =
`switch_on_constant_a2_fallthrough`.

| Target | const_ft | const_a2 | const_a2_ft | term_a2 | struct_a2 | catch-all behaviour | impact |
|---|:-:|:-:|:-:|:-:|:-:|---|---|
| cpp | Y | Y | Y | Y | Y | — | full coverage |
| elixir (lowered emitter) | Y | Y | Y | Y | Y | — | full coverage |
| fsharp | Y | Y | Y | Y | N | `Proceed` stub | covers the common set |
| **scala** (fixed) | Y | Y | Y | Y | N | `Raw` (now unreachable for these) | fixed |
| **haskell** (fixed) | Y | Y | Y | Y | Y | `Proceed` (now unreachable for these) | **was HARMFUL — fixed + runtime-validated** |
| elixir (interpreter mode) | N | N | N | N | N | `{:raw,…}` → step `_ -> :fail` | **non-functional legacy path (see below)** |
| elixir (lowered — real path) | Y | Y | Y | Y | Y | — | full coverage; member runtime-correct |
| **wat** (switch fix) | Y | Y | Y | Y* | Y | `allocate` (now unreachable for these) | switch fix landed; blocked by separate read-mode-unify gap (see below) |
| jvm | N | N | N | N | N | `ldc "<text>"` (operand-stack push) | **HARMFUL (suspect; no assembler toolchain to validate)** |
| rust | N | Y | N | N | N | `/* unknown */` (drop) | safe; missed opt |
| go | N | N | N | N | N | `// TODO` (drop) | safe; missed opt |
| python | N | N | N | N | N | `# SKIP` (drop) | safe; missed opt (confirmed) |
| **r** (fixed) | Y | Y | Y | Y | Y | `Raw(...)` (now unreachable for these) | safe linear no-op for all A1/A2 hints; classic conformance fixed |
| lua | N | Y | N | N | N | `I.Raw(...)` (drop-style) | safe; missed opt (verify) |
| clojure | N | N | N | N | N | `{:op :raw}` stub | depends on step-loop `:raw` |
| llvm | N | Y | N | partial(nop) | N | `; TODO` (switch is design-nop) | safe (chain preserved) |

`*` WAT `term_a2`: handled by emitting an empty fall-through term header
(register 1), not full A2 type-dispatch — correct, just unindexed (its
operand format differs from the A1 `switch_on_term` parser).

`switch_on_structure_a2` (all-compound A2 index) is unhandled even by the
fixed Scala reference and by F#; it is rare and the accepted baseline
tolerates it via the chain. (C++, Haskell, and WAT now handle it.)

## Runtime validation (toolchains installed)

The four harmful-flagged targets were investigated with their real
toolchains (GHC/cabal, Elixir/Erlang, wabt + node). Results refine the
triage above:

- **Haskell — confirmed harmful, fixed, validated.** Generated `member/2`
  to a cabal project and ran it. *Before:* `member(z,[a,b,c]) = true`
  (the `Proceed` catch-all returned from the predicate immediately, so
  any first-arg-indexed predicate succeeded unconditionally). *After*
  adding handlers for the `_fallthrough` / `term`/`structure` variants
  (empty `SwitchOnConstant` falls through to the clause chain):
  `member(a/c,…) = true`, `member(z,…) = false` — correct. No new test
  regressions (the 5 pre-existing failures are unrelated:
  async/unsafe imports, F1 fact classification).

- **Elixir — the "harmful" path is a non-functional legacy mode; the
  real path is correct.** The `:raw → :fail` catch-all lives in
  `emit_mode(interpreter)`, but interpreter mode is **not** the path any
  caller uses: every example, benchmark, and test passes
  `emit_mode(lowered)`, and interpreter mode is additionally
  non-compilable — it emits invalid module names
  (`defmodule WamPred.em`, a lowercase alias segment) and lacks emitter
  handlers for core instructions (`get_list`, `get_nil`, bare
  `allocate`), so a list predicate never even reaches a switch. The
  canonical **lowered** emitter has full switch coverage; the classic
  suite (50 tests, incl. `em_no_match → "false"`) passes, confirming
  `member` is runtime-correct on the real path. No fix applied — the
  matrix's earlier "elixir (main path) HARMFUL" row was misleading;
  reclassified as a non-functional legacy mode.

- **WAT — switch fix landed, but the actual blocker is a separate,
  deeper gap.** `switch_on_term_a2` did fall to the `allocate` catch-all;
  the switch fix removes that (parity with the other targets). *However,
  this produces no observable change for `member` yet,* because WAT's
  runtime **read-mode structure/list argument unification is
  unimplemented**: the read-mode branches of
  `unify_variable`/`unify_value`/`unify_constant` are nops that advance
  PC without walking the matched structure's args (there is no
  S-register). So `get_structure`/`get_list` match only the *functor*,
  and element mismatches go undetected — `member(a,[b]) = true` and
  `s2(x,k(3)) = true` (a pure-structure case with no switch involved,
  proving the gap is independent of indexing). Fixing this needs an
  S-register + heap-walking read-mode unify; flagged for follow-up.
  Existing WAT suite unchanged by the switch fix (49 pass / 8
  pre-existing env-path failures).

- **JVM — not validated.** No JVM-bytecode assembler is packaged
  (krakatau unavailable), and the JVM tests are emit-and-grep only, so
  the `ldc` catch-all could not be runtime-confirmed. Remains suspect.

(The catch-alls of note, with file:line. Haskell's and WAT's are now
unreachable for the indexing variants — handlers were added above them.)
- `wam_haskell_target.pl:6623` — `-- UNKNOWN: ~w` + `Proceed`
  (now unreachable for the switch variants)
- `wam_elixir_target.pl:5238` — interpreter-mode line emitter
  `{:raw, "~w"}`; step-loop default `wam_elixir_target.pl:107` —
  `_ -> :fail` (legacy mode only)
- `wam_wat_target.pl:1421` — unrecognized instruction → `allocate`
  (+stderr) (now unreachable for the switch variants)
- `wam_jvm_target.pl:664` — `ldc "<text>"`
- drop-style: `wam_python_target.pl:1681` `# SKIP`,
  `wam_go_target.pl:1297` `// TODO`, `wam_rust_target.pl:2837`
  `/* unknown */`, `wam_r_target.pl:449` `Raw(...)`,
  `wam_lua_target.pl:644` `I.Raw(...)`, `wam_clojure_target.pl:650`
  `{:op :raw}`.)

## Recommended action

Ranked by *correctness* impact (not raw coverage). **Status updated
after runtime validation — see "Runtime validation" above.**

1. **Harmful-catch-all targets.**
   - **haskell — DONE.** Handlers added (empty `SwitchOnConstant` falls
     through to the chain); `member/2` runtime-validated correct.
   - **elixir — NO ACTION NEEDED.** The `:raw → :fail` catch-all is only
     in the non-functional `interpreter` legacy mode (invalid module
     names, missing core handlers, used by nothing). The real `lowered`
     path has full coverage and is runtime-correct.
   - **wat — switch fix DONE; deeper blocker remains.** The four switch
     variants no longer expand to `allocate`. But `member/2` still
     returns wrong answers because **read-mode structure/list argument
     unification is unimplemented** (`unify_*` read-mode branches are
     nops; no S-register). That is the real fix for WAT and is a
     separate, larger runtime change — **highest-value WAT follow-up.**
   - **jvm — still suspect, unvalidated.** No bytecode assembler
     available; needs a maintainer with a JVM-asm toolchain (or a
     krakatau install) to confirm the `ldc` catch-all and apply the same
     minimal-drop / full-handler fix.

   Two safe options per target (when fixing the catch-all):
   - **Minimal:** make the catch-all *drop* the instruction
     (correctness-safe, like Python/Go/Rust) instead of emitting a
     harmful one — restores correctness, keeps the (already-absent)
     indexing as a follow-up.
   - **Full:** add proper `const_ft` / `const_a2` / `const_a2_ft` /
     `term_a2` handlers, mirroring the C++ / F# / Scala / Haskell
     implementations (for the constant family, the `_fallthrough`
     variant is shape-compatible with the plain variant and can delegate
     to it; `term_a2` / `const_a2` need A2-register dispatch).
   Each fix should be validated by running a first-argument-indexed
   predicate (`member/2`, or a mixed fact+rule like factorial) against
   that target's toolchain.

2. **Drop-style targets — optimization gaps, not bugs.** rust, go,
   python, lua (and clojure/llvm pending step-loop confirmation)
   produce correct results today; adding the a2/fallthrough handlers
   only restores the indexing speedup. Lower priority; do alongside
   per-target perf work.

3. **`switch_on_structure_a2`** everywhere — lowest priority (rare shape;
   chain fallback is correct).

## Reference implementations

- **Haskell** — `src/unifyweaver/targets/wam_haskell_target.pl`
  (constant `_fallthrough` variants reuse `switch_on_constant`'s
  translation; `term`/`structure` variants emit an empty
  `SwitchOnConstant` that falls through to the clause chain).
- **Scala** — `src/unifyweaver/targets/wam_scala_target.pl`
  (`switch_on_constant_fallthrough` reuses `switch_on_constant`;
  `switch_on_term_a2` / `switch_on_constant_a2(_fallthrough)` dispatch on
  register 2 via a `reg` field on the `SwitchOnConstant` / `SwitchOnTerm`
  runtime instructions, defaulting to 1).
- **C++** — `src/unifyweaver/targets/wam_cpp_target.pl` (full coverage incl. `struct_a2`).
- **Elixir lowered emitter** — `src/unifyweaver/targets/wam_elixir_lowered_emitter.pl` (full coverage).
- **R** — `src/unifyweaver/targets/wam_r_target.pl` maps
  `switch_on_constant_fallthrough`, `switch_on_term_a2`,
  `switch_on_constant_a2`, `switch_on_constant_a2_fallthrough`, and
  `switch_on_structure_a2` to the existing `SwitchOnTerm()` no-op
  (safe linear fallback — not optimized A2 dispatch), preserving the
  complete try/retry/trust choice chain.

## Toolchain note

Validating per-target fixes requires the respective toolchain. The
follow-up pass installed **GHC/cabal, Elixir/Erlang, and wabt + node**,
which let haskell, elixir, and wat be runtime-validated (see "Runtime
validation"). A **JVM-bytecode assembler** was still unavailable
(krakatau not packaged), so jvm remains the only flagged target not yet
runtime-confirmed. Rust, Go, Python, Node, Java, and Scala were available
from the original audit (Python confirmed drop-safe).
