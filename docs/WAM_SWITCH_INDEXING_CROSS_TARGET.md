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
| haskell | N | Y | N | N | N | `-- UNKNOWN` + **`Proceed`** | **HARMFUL** |
| elixir (main path) | N | N | N | N | N | `{:raw,…}` → step `_ -> :fail` | **HARMFUL** |
| wat | N | Y | N | N | N | rewrite to **`allocate`** (+stderr) | **HARMFUL (suspect)** |
| jvm | N | N | N | N | N | `ldc "<text>"` (operand-stack push) | **HARMFUL (suspect)** |
| rust | N | Y | N | N | N | `/* unknown */` (drop) | safe; missed opt |
| go | N | N | N | N | N | `// TODO` (drop) | safe; missed opt |
| python | N | N | N | N | N | `# SKIP` (drop) | safe; missed opt (confirmed) |
| r | N | N | N | N | N | `Raw(...)` (drop-style) | safe; missed opt (verify) |
| lua | N | Y | N | N | N | `I.Raw(...)` (drop-style) | safe; missed opt (verify) |
| clojure | N | N | N | N | N | `{:op :raw}` stub | depends on step-loop `:raw` |
| llvm | N | Y | N | partial(nop) | N | `; TODO` (switch is design-nop) | safe (chain preserved) |

`switch_on_structure_a2` (all-compound A2 index) is unhandled even by the
fixed Scala reference and by F#; it is rare and the accepted baseline
tolerates it via the chain.

(The catch-alls of note, with file:line:
- `wam_haskell_target.pl:6594` — `-- UNKNOWN: ~w` + `Proceed`
- `wam_elixir_target.pl:5236` — line emitter `{:raw, "~w"}`; step-loop
  default `wam_elixir_target.pl:107` — `_ -> :fail`
- `wam_wat_target.pl:1388` — unrecognized instruction → `allocate` (+stderr)
- `wam_jvm_target.pl:664` — `ldc "<text>"`
- drop-style: `wam_python_target.pl:1681` `# SKIP`,
  `wam_go_target.pl:1297` `// TODO`, `wam_rust_target.pl:2837`
  `/* unknown */`, `wam_r_target.pl:449` `Raw(...)`,
  `wam_lua_target.pl:644` `I.Raw(...)`, `wam_clojure_target.pl:650`
  `{:op :raw}`.)

## Recommended action

Ranked by *correctness* impact (not raw coverage):

1. **Harmful-catch-all targets — real correctness bugs.** Fix
   **haskell** (catch-all emits `Proceed`, which skips the predicate
   body), **elixir main path** (`:raw` → `:fail`), and verify
   **wat** (`allocate`) and **jvm** (`ldc`). Two safe options per target:
   - **Minimal:** make the catch-all *drop* the instruction
     (correctness-safe, like Python/Go/Rust) instead of emitting a
     harmful one — restores correctness, keeps the (already-absent)
     indexing as a follow-up.
   - **Full:** add proper `const_ft` / `const_a2` / `const_a2_ft` /
     `term_a2` handlers, mirroring the C++ / F# / Scala implementations
     (for the constant family, the `_fallthrough` variant is
     shape-compatible with the plain variant and can delegate to it;
     `term_a2` / `const_a2` need A2-register dispatch).
   Each fix should be validated by running a first-argument-indexed
   predicate (`member/2`, or a mixed fact+rule like factorial) against
   that target's toolchain.

2. **Drop-style targets — optimization gaps, not bugs.** rust, go,
   python, r, lua (and clojure/llvm pending step-loop confirmation)
   produce correct results today; adding the a2/fallthrough handlers
   only restores the indexing speedup. Lower priority; do alongside
   per-target perf work.

3. **`switch_on_structure_a2`** everywhere — lowest priority (rare shape;
   chain fallback is correct).

## Reference implementations

- **Scala** — `src/unifyweaver/targets/wam_scala_target.pl`
  (`switch_on_constant_fallthrough` reuses `switch_on_constant`;
  `switch_on_term_a2` / `switch_on_constant_a2(_fallthrough)` dispatch on
  register 2 via a `reg` field on the `SwitchOnConstant` / `SwitchOnTerm`
  runtime instructions, defaulting to 1).
- **C++** — `src/unifyweaver/targets/wam_cpp_target.pl` (full coverage incl. `struct_a2`).
- **Elixir lowered emitter** — `src/unifyweaver/targets/wam_elixir_lowered_emitter.pl` (full coverage).

## Toolchain note

Validating per-target fixes requires the respective toolchain. In the
audit environment, Rust, Go, Python, Node, Java, and Scala were available
(Python confirmed drop-safe); GHC, Elixir, Lua, R, a wasm runtime, and a
JVM-bytecode assembler were not, so the harmful-catch-all targets
(haskell/elixir/wat/jvm) could not be runtime-validated here and are
flagged for follow-up by maintainers who can run them.
