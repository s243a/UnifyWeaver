<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: numeric instruction encoding (R1, D139)

**Date:** 2026-10-17. **Ledger:** D139 in
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`. **Design:** phase R1 of
`docs/proposals/wam_rust_heap_cell_rewrite_design.md` (§4, §12), done on
today's `Value` runtime. **Base:** main `6568f52` (D138 plus the
label-uniqueness fix in `wam_target.pl`).

**What:** the Rust WAM generator stops encoding instructions with names that
the runtime decodes on every execution. Register operands are decoded
`Reg` values, branch targets are PCs, builtin calls carry a dispatch route,
and functor operands and the step arms' constant atoms are worked out once.
Nothing about `Value`, the trail, choice points or the binding table changed.
Every program prints exactly what it printed before.

**Result:** **−2.82 M Ir per warm resolve at N=40 (20.06 M → 17.24 M,
−14.0%)** and −3.22 M at N=5000 (389.02 M → 385.80 M, −0.8%). The saving is
roughly flat in N, like the D123–D127 levers it continues.
The warm in-process N=40 resolve drops from 2.90 to 2.56 ms (−12%, median of
41 interleaved rounds, twice); N=5000 wall is inside the noise.

## 1. What changed

Four commits, each green on the term differential, the corpus, byte
identity and the five SWI-comparison harnesses.

### 1.1 Register operands (`1a411c8`)

Every `Instruction` field that held a register name (`"A1"`, `"X3"`,
`"Y2"`) now holds a `Reg` (`templates/targets/rust_wam/instructions.rs.mustache`),
a `u16`:

* bit 15 set: `Y<n>`, `n` in the low bits. It lives in slot `n-1` of the
  topmost environment frame.
* bit 15 clear: an `A`/`X` register; the low 14 bits are the flat
  register-file index `WamState::reg_index(name)` gave (A1 → 0, X1 → 100);
  an index at or past `MAX_REGS` is stored as one out-of-range value, which
  every access already treated alike. Bit 14 records whether the name
  started with `A`, because `PutStructure` skips its bind-through for
  argument registers by testing that first byte.

The encoding keeps every property the runtime derived from a name, so an
access through a `Reg` is the same access through its name. `Reg::from_name`
is the exact decoder (it calls `reg_index` itself); the generator does the
same decode in Prolog (`src/unifyweaver/targets/wam_rust_reg.pl`) and emits
`Reg::a(1)`, `Reg::x(3)`, `Reg::y(2)`.

**Surprise:** the shared compiler emits register names outside the
`A/X/Y<n>` pattern. The findall/bagof template builder
(`wam_target.pl` `construct_template/7`) names its scratch registers
`_XT0`, `_XT1`, …, and the runtime decodes those with the generic
`reg_index_parse` rule: any prefix other than A/X/Y is flat slot 0, the
slot of `A1`. That aliasing is today's behaviour, so `wam_rust_reg.pl`
reproduces the rule and emits the raw encoding (`Reg(0)`) for such names.
The only names it refuses are the `Y` spellings the slot frames key by text
(`Y0`, `Y01`, `Y1x`); the compiler never emits them.

The step arms use `get_reg_r`, `put_reg_r`, `set_reg_r`, `get_reg_raw_r`,
`get_reg_ref_r` and `trail_reg` (`state.rs.mustache`): each is the `&str`
method of the same name with the decode taken out. The A/X paths are
inline; the Y paths (`get_y`, `put_y`, `trail_y`) are out of line, so the
remaining X/Y test is one bit of the operand. A Y trail entry gets the same
name text as before (`TrailEntry::register_reg`: a `REG_NAMES` borrow for
Y1..Y99, an owned `Y<n>` beyond). The `&str` accessors stay, unchanged, for
builtins, lowered code, fact sources, the shims and the benchmark drivers.
The lowered emitter's delegated `vm.step(&Instruction::…)` calls and the
runtime text parser (`parse_instructions`) build `Reg`s too.

New Rust tests `mod r1_reg_operand_tests`: the decoder agrees with
`reg_index`/`is_y_reg_name`/`YRegs::slot` on ~4.4K names (7 prefixes ×
0..620 plus odd spellings), the constructors agree with the decoder, and
20,000 random register operations through names on one machine and
through `Reg`s on another leave every register and trail entry equal.

### 1.2 Branch targets as PCs (`8c40006`)

`collect_wam_entries/7` first numbers every label of the shared table (the
numbering the `labels.insert` lines get; a repeated label keeps its last PC,
as `HashMap::insert` does) and then emits with that table:

| WAM | before | after |
| --- | --- | --- |
| `try_me_else L` (clause entry, R0a checked) | `TryMeElseArgs("L", n)` | `TryMeElsePcArgs(pc, n)` |
| `try_me_else L` (other clause) | `TryMeElse("L")` | `TryMeElsePc(pc)` |
| `try_me_else L_ite_else_N` | `TryMeElse("L_ite_else_N")` | `TryMeElseItePc(pc)` (new) |
| `retry_me_else L` | `RetryMeElse("L")` | `RetryMeElsePc(pc)` |
| `try/retry/trust L` (D138 chains) | `Try/Retry/Trust("L")` | `TryPc/RetryPc/TrustPc(pc)` (new) |
| `jump L` | `Jump("L")` | `JumpPc(pc)` (new) |
| `switch_on_constant` | `SwitchOnConstant` | `SwitchOnConstantPc` (PC 0 = `default`) |
| `switch_on_constant_fallthrough` | by name | `SwitchOnConstantFallthroughPc` (new) |
| `switch_on_structure`, `switch_on_constant_a2` | by name | the existing `*Pc` forms |
| `call P, N` / `execute P` | `Call("P", N)` / `Execute("P")` | `CallPc(pc, N)` / `ExecutePc(pc)` |

Exactness notes:

* `TryMeElse(label)` parks no clause barrier for an `L_ite_else_` label;
  `TryMeElsePc` always does. Hence the new `TryMeElseItePc`. `GetLevel`'s
  look-ahead for the guard's try instruction includes it.
* `SwitchOnConstant` fails on a key whose label is unknown and falls through
  on `default`. The Pc table encodes `default` as PC 0; the form resolves
  only when every entry does, otherwise the by-name form stays. The other
  three switch forms skip a matching entry with an unknown label and keep
  scanning, so such entries are dropped.
* The `Call`/`Execute` arms try `crate::lowered_call` before the label. A
  call to a predicate that `lowered_call` routes (the Stage-2 regions and
  the D55/F11 banks: the generator reads the keys from the emitted
  dispatch function) keeps the by-name form, as does a target with no label
  (builtins, foreign, dynamic, fact tables), with all their fallbacks. For
  every other labelled target the arm's first live branch was the jump.
* The `labels` map is still emitted in full: entry lookup (`call_pred`),
  meta-calls (`call/N`), `current_predicate/1` and the dynamic DB read it.
* A standalone predicate (`compile_wam_predicate_to_rust/4`, its own
  program) resolves only its local targets.
* `rust_named_calls(true)` keeps call/execute/builtin_call by name. The
  benchmark generators set it, because their `main.rs` rewrites and fuses
  those forms itself.

In the resolver's table this replaces 234 `Call`, 71 `Execute`, 98
`TryMeElse`, 63 `TryMeElseArgs`, 98 `Jump`, 43 `RetryMeElse`, 3+3 `Try`/`Trust`
and 21 switch tables; 15 `Call`s and 8 `Execute`s to `lowered_call`
targets and one unlabelled `predsort/3` stay by name.

### 1.3 Builtin routes (`d84c0d4`)

`builtin_call` becomes `BuiltinCallR(route, name, arity)`. Routes 0–3 are
the four core builtins (`true/0`, `fail/0`, `!/0`, `=/2`), run inline in the
step arm with `execute_builtin`'s own core bodies. Routes 4–9 enter the
family cascade (arith, io, type, term, ext, meta) at the first family whose
generated code contains the name as a string literal
(`execute_builtin_from`). That is the full cascade's result: the families
before it do not claim the name, and a family that does not claim a name
returns `false` without touching the machine (D124). A name no family names
takes the full `execute_builtin` (route 10). The route table is built from
the same `compile_execute_*_builtin_to_rust/1` text the runtime is generated
from, not from a hand list, and `tests/test_wam_rust_r1_encoding.pl` pins
that every family claims names through literal match arms only. The
in-family dispatch still matches the name string: turning ~200 arms into
numeric ids would rewrite every family for the ≤0.9% D124 measured as the
whole dispatch cost, and the measurement below agrees.

### 1.4 Functor operands and step constants (`1e221d9`)

`GetStructure`/`PutStructure` carry a `Functor`: the `"name/arity"` text
plus what the arms recomputed per execution, i.e. GetStructure's arity
(`split('/').nth(1)`) and PutStructure's (`split('/').last()`), which differ
for `"//2"`, the `"str(...)"` second spelling, and the interned `Sym`,
filled on the instruction's first use. The placeholder atoms the builders
push (`__struct_arg__` per argument, `__list_head__`, `__list_tail__`) and
`index_key`'s `[]` are interned once instead of building and interning a
`String` on every push. Both are filled on first use, which is when the old
code first interned them, so interner ids are assigned in the same order.
Constant operands (`GetConstant`/`PutConstant`/`SetConstant`/
`UnifyConstant` values and switch keys) were already `Value`s built once
when the program loads; R1 leaves them as they are.

### 1.5 Deferred, and why

* **Separate X/Y instruction variants** (design §4). After 1.1 the only
  run-time register-kind test left is one bit of the operand, with the Y
  side out of line. Splitting the register-carrying variants would double
  them in the generator, the lowered emitter, the text parser and the
  benchmark matchers to save about one predicted branch per register
  operand (~22K instructions per N=40 resolve), well under 0.1 M Ir.
* **`LevelId` operands for `GetLevel`/`CutTo`.** The level is keyed by the
  register name on the choice point (`ChoicePoint::levels:
  Vec<(String, usize)>`), so a numeric key changes the choice point, which
  is R2's. The `String` clone it costs per `GetLevel` is ~0.04 M Ir at N=40.
* **`FunctorId` switch keys, numeric in-family builtin ids, the
  Y-without-Allocate check:** R2 work (they need the functor table and the
  level ids), or below the noise (1.3).

## 2. Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8`, final commit
`1e221d9`. The base outputs are a build of a `git archive` export of
`6568f52`.

| gate | result |
| --- | --- |
| `rust/build.sh` + `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/build.sh` + `run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout at N=40, 250, 1000 and 5000. Also identical after each of the four commits (term lane) |
| generated crate `cargo test --release --lib` | `273 passed; 0 failed` (base 270, plus `r1_reg_operand_tests`) |
| `tests/test_wam_rust_baseline_fixes.pl` | 92/92 |
| `tests/test_wam_rust_r0a_clause_cps.pl` | 90/90 |
| `tests/test_wam_rust_register_restore.pl` | 120/120 |
| `tests/test_wam_rust_first_arg_dispatch.pl` | 217/217 |
| `tests/test_wam_rust_stream_swi_parity.pl` | 9 passed, 12 blocked (as base) |
| `tests/test_wam_rust_r1_encoding.pl` (new), `test_wam_rust_builtin_dispatch.pl` | 13/13, 3/3 |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, 57 files on base + the new one) | the same 31 failing test lines as base (all pre-existing: crates that already failed `cargo check` on base, and the stream-parity G4/G5 cases); per-file rc as base except `test_wam_rust_par_aggregate.pl`, rc 1 → 0 (its two wall-clock gate tests, flaky on a loaded box) |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, unsampled and with `CONFORMANCE_SAMPLE=2` |
| `python3 -m unittest tests.test_wam_rust_matrix_demand_flag` | 6/6 |
| benchmark generators | both generate (`gen rc=0`, matrix: 4 emit/kernel modes; effective-distance: seeded and accumulated). Their crates already failed `cargo check` on base (matrix: 7 `E0308` in the lowered `lib.rs`; effective-distance: 102 errors, mostly modules and methods its hand-written crate layout no longer provides, e.g. `boundary_cache`, `seek_fact_source`); after R1 the error sets are identical to base's |
| core quick suite (AGENTS.md: constraint analyzer, stream, recursive, advanced, constraints) | pass (`template_system.pl` changed) |

### Updated test expectations (generated-text assertions)

* `tests/test_wam_rust_r0a_clause_cps.pl`: `safe_group_uses_arity_only_clause_cps`
  (`TryMeElsePcArgs(<pc of L_F_A_2>, A)`, not `TryMeElseArgs("L_F_A_2", A)`),
  `ite_guards_keep_the_full_save` (`TryMeElseItePc(`, and no
  `TryMeElsePcArgs` at an `L_ite_else_` PC), `full_save_group_keeps_the_full_save`
  (`TryMeElsePc(<pc>)`, and no `TryMeElsePcArgs(`),
  `unchecked_arity_only_breaks_the_full_save_shapes` (`TryMeElsePcArgs(`).
* `tests/test_wam_rust_first_arg_dispatch.pl`: `dispatch_shapes_emitted`
  (`TryPc/RetryPc/TrustPc(<pc of the body label>)` and the `*Pc` switch forms).
* `tests/test_wam_rust_target.pl`: the aggregate instruction parser test
  expects `Instruction::EndAggregate(Reg::y(1))`.

New: `tests/test_wam_rust_r1_encoding.pl` (13), `mod r1_reg_operand_tests`
(3).

## 3. Measurements

Method as in `docs/reports/wam_rust_register_trail_static_names.md`: a
scratch copy of each generated crate (release, line tables on) with a shim
that repeats `call_pred` in-process and checks every answer is equal; Ir
per warm resolve is `(Ir(1+k) − Ir(1)) / k` under callgrind, k=10 at N=40
and k=2 at N=5000.

| N | base | R1 | Δ |
| ---: | ---: | ---: | ---: |
| 40 | 20.06 M | **17.24 M** | **−2.82 M (−14.0%)** |
| 5000 | 389.02 M | **385.80 M** | −3.22 M (−0.8%) |

### 3.1 The N=40 change by bucket (one commit each)

| commit | bucket | N=40 Ir | Δ |
| --- | --- | ---: | ---: |
| base `6568f52` | | 20.06 M | |
| `1a411c8` | register access | 18.95 M | −1.11 M |
| `8c40006` | labels (PC-resolved control) | 18.08 M | −0.87 M |
| `d84c0d4` | builtin dispatch | 18.06 M | −0.02 M |
| `1e221d9` | interning and functor decode | 17.24 M | −0.81 M |

Where it came from (exclusive Ir per warm resolve, N=40):

* **Register access.** `get_reg` 0.54 M, `put_reg` 0.46 M, `trail_binding`
  0.35 M, `set_reg` 0.35 M, `reg_index` 0.21 M and `get_reg_raw` 0.20 M (2.10 M)
  become `get_y` 0.30 M, `put_y` 0.23 M and `trail_y` 0.12 M (0.66 M, the Y
  frame walk), plus ~0.15 M more in `step` where the A/X paths now inline.
* **Labels.** The `labels: HashMap<String, usize>` probes, `sip::Hasher::write`
  0.33 M + `hash_one` 0.27 M + `HashMap::get` 0.20 M, drop to ~0.
* **Builtin dispatch.** `execute_builtin` self 0.09 M becomes
  `execute_builtin_from` 0.02 M; the core four no longer leave the step arm.
  As D124 found, dispatch was never a large cost.
* **Interning and functor decode.** `intern` 0.81 M → 0.38 M (the placeholder
  atoms and the per-execution functor intern); the `split('/')` arity parse
  (`CharSearcher::next_match`) 0.30 M → 0.20 M; fewer allocations.

What remains in the mechanics bucket: the Y frame walk (0.66 M), `step`
self 1.78 M, `set_heap_or_list` (structure construction, register scan),
`deref_heap`/`decomp` and the `Value` clones/drops that R2's cells replace.

### 3.2 Wall clock (shared 4-core container, noisy)

The two perf binaries, interleaved; 11 in-process resolves per run; warm is
the median of resolves 2..11, then the median over rounds. Cold is the first
resolve of each run (program load excluded).

| N | rounds | cold base | cold R1 | warm base | warm R1 | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 41 | 4.15 ms | 3.40 ms | 2.93 ms | **2.56 ms** | −13% |
| 40 | 41 | 4.25 ms | 3.44 ms | 2.90 ms | **2.56 ms** | −12% |
| 5000 | 15 | 56.3 ms | 61.6 ms | 53.7 ms | 53.4 ms | (noise) |

A first 21-round N=40 run, taken while other builds were finishing, gave the
opposite order (2.96 vs 3.24 ms warm); the two quiet 41-round runs agree
with each other and with Ir. Ir is the reliable metric.

## 4. Files

* `templates/targets/rust_wam/instructions.rs.mustache`: `Reg`,
  `builtin_route`, `Functor`; register fields → `Reg`, `Functor` operands,
  new `TryMeElseItePc`, `TryPc`, `RetryPc`, `TrustPc`, `JumpPc`,
  `SwitchOnConstantFallthroughPc`, `BuiltinCallR`.
* `templates/targets/rust_wam/state.rs.mustache`: the `Reg` accessors,
  `YRegs::get_n/insert_n`, `TrailEntry::register_reg`, the static step atoms,
  the text parser, `mod r1_reg_operand_tests`.
* `src/unifyweaver/targets/wam_rust_target.pl`: step arms, emission of
  `Reg`/`Functor`/PC/route operands, the label pass in
  `collect_wam_entries/7`, `rust_builtin_route/2`, `execute_builtin_from`.
* `src/unifyweaver/targets/wam_rust_reg.pl` (new): register decoding.
* `src/unifyweaver/targets/wam_rust_lowered_emitter.pl`: `Reg`/`Functor`
  in delegated step calls.
* `src/unifyweaver/core/template_system.pl`, `templates/targets/rust_wam/lib.rs.mustache`:
  `use instructions::{Functor, Reg}` in the generated `lib.rs`.
* `examples/benchmark/generate_wam_effective_distance_benchmark.pl`,
  `generate_wam_rust_matrix_benchmark.pl`: `rust_named_calls(true)`, `Reg`
  operands.
* Tests: see §2.

`wam_target.pl`, `resolver.pl` and `resolver_store.pl` are unmodified. The
regenerated checked-in crate is not committed.
