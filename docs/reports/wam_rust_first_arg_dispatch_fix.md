<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: first-argument dispatch with a shared index key (D138)

**Date:** 2026-10-11. **Ledger:** D138 in
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`.
**Regression harness:** `tests/test_wam_rust_first_arg_dispatch.pl`, plus the
Rust unit tests `d138_first_arg_dispatch_tests` in
`templates/targets/rust_wam/state.rs.mustache`.

## Symptom

The Rust WAM target returned no solutions when a first-argument index key
was shared by several clauses, none of them clause 1:

```prolog
dp(1, p). dp(2, q). dp(2, r).            % findall(R, dp(2,R), L): SWI [q,r], Rust []
da(a, 1). da(b, 2). da(c, 3). da(b, 4).  % findall(R, da(b,R), L): SWI [2,4], Rust []
```

Both emit modes (`interpreter` and `functions`). A key matched by one clause,
or by a group that includes clause 1, worked.

## Root cause

The shared compiler is correct. For a multi-clause predicate whose first
arguments are all constants (or all structures, or mixed), `wam_target.pl`
emits a `switch_on_constant` / `switch_on_structure` / `switch_on_term`
header. A key matched by several clauses (none of them clause 1) jumps to a
dedicated dispatch chain that `format_dispatch_chain/5`
(`wam_target.pl:951`) appends after the linear clause chain:

```
dp/2:
    switch_on_constant 1:default 2:L_dp_2_const_2_dispatch
    try_me_else L_dp_2_2
    ...                                  % clauses 1..3, linear chain
L_dp_2_const_2_dispatch:
    try   L_dp_2_2_body
    trust L_dp_2_3_body
```

`try L` / `retry L` / `trust L` are the standard WAM indexing instructions:
the choice point's alternative is the NEXT chain instruction and control
jumps to `L`. The Rust target translated them as the linear-chain
instructions (`wam_rust_target.pl:8633-8643` on main):

```prolog
wam_line_to_rust_instr(["try", Label], _, _, Rust) :-   % -> TryMeElse(Label)
wam_line_to_rust_instr(["trust", _Label], _, _, "Instruction::TrustMe").
wam_line_to_rust_instr(["retry", Label], _, _, Rust) :- % -> RetryMeElse(Label)
```

`TryMeElse(L)` stores `L` as the alternative and FALLS THROUGH; `TrustMe`
pops and falls through. So `try L_dp_2_2_body` pushed a choice point and ran
the next instruction, `trust` popped it and ran past the end of the chain
(the next predicate's code, or the end of the program): no clause body was
ever entered. Go, C, C++, JS, Scala, Clojure and F# implement `try`/`retry`/
`trust` as jumps; Rust was the outlier.

## Related defects found by the new harness (same code path)

Checking the other key kinds and the other index forms turned up five more
Rust-only defects in the same switch/chain path, fixed in the same change:

1. **Linear-chain targets rewrote the caller's choice point.** The
   `switch_on_constant_fallthrough` form (constant clauses with variable-
   first-argument clauses after them) and every second-argument form
   (`switch_on_constant_a2`) list one entry per clause, and a non-first
   clause's entry is its linear-chain label `L_p_n_i` -- the `retry_me_else`
   / `trust_me` line in front of clause i, meaning "clauses i..N". The
   switch arm jumped there with `self.pc = pc`, so `RetryMeElse` rewrote
   whatever choice point was on top (the CALLER's: it later resumed at
   clause i+1 with the caller's registers) and `TrustMe` popped it.
   `dv(b,R)` over `dv(a,1). dv(b,2). dv(X,v(X)). dv(b,3).` printed `2` and
   the driver failed (SWI `2, v(b), 3`); `dcr(2,R)` with a cut in the retry
   position looped forever.
2. **A key whose first entry is `default` lost clause 1.** In those per-
   clause tables a key can repeat (`a:default ... a:L_p_3_3`). The
   fallthrough emitter dropped only the `default` entry, and the A2 arm
   skipped it at run time, so the later entry won and clause 1 was never
   tried (`ds2(x,a,R)`: Rust `3`, SWI `1, 3`).
3. **Float keys did not compile.** `parse_index_entry_constant/2` emitted
   every number as `Value::Integer(..)`: `Value::Integer(2.5)`, a Rust type
   error that broke the whole crate.
4. **`true` / `false` keys never matched.** They were emitted as
   `Value::Bool`, but clause heads and callers build the atom
   (`rust_const_value/2`), so `SwitchOnConstant` found no key and failed.
5. **Quoted keys never matched.** `wam_tokenize_line/2` ends a quoted token
   at its closing quote, so `'Hello World':L_..` arrives as two tokens; the
   key kept its quotes and the label became a separate bogus entry. Also,
   an empty list reached as a list tail (`Value::List([])`) is the atom `[]`
   for `GetConstant` but was not for `SwitchOnConstant`, which then failed.

## Fix (Rust target only; `wam_target.pl` unchanged)

- `instructions.rs.mustache`: new `Instruction::Try(String)`,
  `Retry(String)`, `Trust(String)`.
- `wam_rust_target.pl`: `try` / `retry` / `trust` lines map to them, and
  their step arms have standard semantics -- `Try` pushes a choice point
  resuming at `pc + 1` and jumps to the body; `Retry` sets the top choice
  point's alternative to `pc + 1` and jumps; `Trust` pops and jumps. `Try` /
  `Retry` park the clause's cut barrier (below the chain's choice point) for
  the body's `Allocate`, exactly as `TryMeElse` / `RetryMeElse` do, so a `!`
  in a dispatched clause removes the chain's alternatives.
- `state.rs.mustache`: `enter_indexed_clause(target)`; every switch arm
  (label and `Pc` forms) jumps through it. A plain target is a jump. A
  `retry_me_else L` target is entered as `try_me_else L` would be (a choice
  point of the call's own, alternative `L`); a `trust_me` target runs its
  clause with no choice point. `index_key(v)` reads an empty `Value::List`
  as the atom `[]` in the constant arms.
- Table emission: `rust_index_entries/2` re-joins a quoted key with its
  `:Label`; keys are split at the LAST colon and classified with
  `rust_const_value/2` (the `get_constant` classifier); structure keys are
  unquoted; `rust_drop_default_keys/2` drops every entry of a key whose first
  entry is `default` in the fallthrough and A2 tables.
- The runtime text parser (`parse_single_instruction`) accepts
  `try`/`retry`/`trust`.

The benchmark generators' `resolve_targets` passes `Try`/`Retry`/`Trust`
through unchanged (label form); they never emit them, and their own group
tables target chain heads, for which `enter_indexed_clause` is a plain jump.

## Tests

`tests/test_wam_rust_first_arg_dispatch.pl` (SWI-comparison harness, same
shape as `test_wam_rust_baseline_fixes.pl`; every Rust run capped at 60 s):
68 program queries x {interpreter, functions} all-solutions checks, the
direct lowered entry (first solution, functions mode) where lowered and
callable, and a check that the generated code really contains the
`Try`/`Retry`/`Trust`, structure, fallthrough and A2 shapes. Coverage:
duplicate keys at first/middle/last positions, three clauses on one key
(try/retry/trust), all clauses on one key, unbound first argument; atom,
negative/positive integer, float (and `1` vs `1.0`), `true`/`false`, quoted
(`'Hello World'`, `'a:b'`, `'Up'`), `[]` (atom and list tail), structure
(incl. same name/different arity and a quoted functor), list and mixed
first arguments; variable-first-argument clauses interleaved (including a
repeated clause-1 key and the caller-choice-point cases); second-argument
indexing; cut in the try and retry positions, neck cut and cut after a call,
with an outer `member/2` choice point; dispatch inside `findall`,
`aggregate_all(count)`, `aggregate_all(bag)` and nested `findall`; deep
backtracking into a chain from a later goal; recursion over a duplicate-key
edge relation; predicates with 10 and 12 clauses.

Before the fix (origin/main): the three float-key queries make the generated
crate fail to compile (`Value::Integer(2.5)`), so they were run separately;
of the other 65 queries, 39 diverged from SWI in `interpreter` mode and the
same 39 in `functions` mode:

`a2_dup a2_dup_a a2_last_outer a2_outer big_dup big_int bool_keys
bool_keys_f bt_deep bt_filter bt_last bt_pairs cut_after_call cut_neck
cut_neck_outer cut_retry cut_retry_outer dup_last dup_mid dup_three in_bag
in_count in_findall in_nested int_keys int_keys_2 neg_keys nil_tail
quoted_colon quoted_keys reach rule_bodies struct_from_findall
struct_from_list struct_from_univ var_mix_b var_mix_dup var_mix_first
var_mix_outer`

(`cut_retry` looped until the 60 s cap.) Eight direct lowered entries
diverged (`bt_pairs cut_neck_outer cut_retry_outer nil_tail rule_bodies
struct_from_findall struct_from_list struct_from_univ`; these lowered bodies
call an interpreted, dispatched predicate), and the shape check failed (no
`Try`). After the fix every check passes in both modes.

The structure-key programs whose argument is built by `put_structure` in the
caller (`struct_keys`, `struct_arity`) passed even before: the argument is a
heap reference, `SwitchOnStructure` does not dereference it and falls
through to the linear chain. The `struct_from_*` programs pass a term value
(a list element, a `findall` copy, a `=..` result) and do reach the chain.

Direct lowered entries not checked (reasons unrelated to indexing; the
lowered parser drops `switch_on_*` and the chains altogether, and the
interpreted entry of each is checked in both modes): the four aggregate
programs (lowered emitter drops `begin_aggregate`, as in the baseline
harness), three deep-backtracking programs (a lowered body's call is first-
solution inside a raised backtrack floor; the interpreted entry's
`lowered_dispatch` guard declines such calls), and the quoted-atom programs:
**pre-existing lowered-emitter bug, not fixed here** -- `parse_lines/2` in
`wam_rust_lowered_emitter.pl` splits on blanks and commas, so
`get_constant 'Hello World', A1` becomes four tokens and is silently dropped,
and the lowered clause then matches any first argument.

## Other WAM targets (static review)

| target | `try`/`retry`/`trust` | verdict |
| --- | --- | --- |
| Go, C, C++, JS, Scala, Clojure, F# | dedicated ops: push/rewrite CP at `pc + 1`, jump to label | correct (as reference) |
| LLVM | `try`/`retry`/`trust` and `switch_on_*` are no-ops | correct (no indexing) |
| Haskell | `try` falls to the unknown-instruction fallback, which emits `Proceed`; `SwitchOnConstantPc` jumps to the `_dispatch` label | **suspect**: a shared key would return success once with no head unification; `TrustMe`/`RetryMeElse` at a mid-chain target also act on the caller's CP |
| Lua | `try` becomes `I.Raw(..)`; switch jumps to the `_dispatch` label | **suspect**: same no-solution shape as Rust |
| Python, Elixir, R, Kotlin, WAT, ILASM, JVM | not reviewed in depth | unverified |

No GHC or Lua toolchain is installed in this environment, so the two
suspect targets were not run.
