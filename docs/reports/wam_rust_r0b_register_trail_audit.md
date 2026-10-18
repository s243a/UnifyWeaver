<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: R0b audit, register trail entries and their consumers

**Date:** 2026-10-09. **Code:** `origin/main` at `5f08375`. **Phase:** R0b of
`docs/proposals/wam_rust_heap_cell_rewrite_design.md` (§5.1, §6.3, §12):
remove A/X register entries from the trail; Y entries stay until R2.
**What this is:** the audit and the test net that R0b requires before the
removal. No runtime code or template changed. The new test file is
`tests/test_wam_rust_register_restore.pl`. §7 records the implementation
that followed (D136).

Citations: `T` = `src/unifyweaver/targets/wam_rust_target.pl`, `ST` =
`templates/targets/rust_wam/state.rs.mustache`, `DB` =
`templates/targets/rust_wam/dynamic_db_methods.rs.mustache`, `LE` =
`src/unifyweaver/targets/wam_rust_lowered_emitter.pl`, `RT` =
`src/unifyweaver/targets/rust_target.pl`, `PA` =
`templates/targets/rust_wam/par_aggregate.rs.mustache`, `WT` =
`src/unifyweaver/targets/wam_target.pl`. Line numbers are at `5f08375` (the
design's citations are at `e095a00` and have shifted).

## Summary

1. **Exactly two rollback sites read an A/X register that only the register
   trail gives back:** `include/3`/`exclude/3` (`T:7204`, read at `T:7208`)
   and `partition/4` (`T:7228`, read at `T:7232-7233`). Each element's test is
   meta-called and rolled back with `unwind_trail_to`, and the builtin then
   reads its output argument from `A3` (and `A4`). The element goal's
   callees load those registers through instructions that push register
   entries, so today the unwind restores them. The fix is local: read `A3`
   and `A4` at builtin entry, before the loop, as `predsort/3` already does
   (`T:7162`).
2. **Every other site is safe without A/X entries.** 97 runtime rollback
   sites plus 39 helper call sites were checked (§2). Apart from the two
   above: 75 unwind a window that contains no register entries (trial
   unifications, region declines), 10 restore the registers explicitly
   afterwards (`restore_regs` from a `save_regs` snapshot), 4 are followed
   only by code that reloads A/X before reading them, and 6 never apply
   register entries at all (choice-point backtracking drops them).
3. **The test net bites exactly there.** `tests/test_wam_rust_register_restore.pl`
   has 35 programs, 120 plunit checks, all passing on `5f08375`. With the A/X
   register pushes disabled in a scratch build (the R0b emulation), the same
   3 checks fail and 117 pass: interpreter `include`, `exclude`, `partition`.
   With all register pushes disabled (the spike's s3), the result is the
   same: no other check changes.
4. **The register trail is already incomplete, and functions mode shows the
   consequence today.** Lowered code writes registers with `put_reg`, which
   does not trail, and several interpreter arms (`PutStructure`, `PutList`,
   `SetVariable`, write-mode `UnifyVariable`, `call_goal_key`'s argument
   loads) do not trail either. In `emit_mode(functions)`, `include/3`,
   `exclude/3` and `partition/4` over a goal whose callees are lowered
   already return the wrong answer (they fail where SWI succeeds). The R0b
   fix in item 1 fixes these too.
5. **Streams and other external state never pass through the register
   trail** (stream handles are integers held in terms; the stream table is a
   process-wide static; the term-input buffer and position are machine
   fields). Backtracking does not undo I/O, which matches SWI-Prolog on every
   comparison program, with one exception unrelated to the trail: a
   dynamic (asserted) rule body is re-executed from its start on each redo,
   so its output and its reads from `stream_open/2` handles are replayed.

## 1. How register entries are produced and consumed

A register entry is pushed by `trail_binding(name)` (`ST:6180`), which
records `get_reg(name)` (the dereferenced old value, `None` for an `Uninit`
A/X slot or a missing Y slot) as `TrailEntry::register`. It is consumed in
exactly one place, `unwind_trail_to` (`ST:4275`), which writes the old value
back with `put_reg` (`None` becomes `Uninit`). `register_name()` is read by
one runtime site, the Y filter in `call_goal_once` (`T:7347`), and by lib
tests. `classify()` is used only by `unwind_trail_to`.

Choice-point backtracking never applies register entries:
`backtrack` (`T:1443`) calls `unwind_trail_bindings_only` (`T:1509`, binding
entries only) and then truncates the trail, so register entries above the CP
are dropped; A/X come back from the CP's `saved_args`, Y from its stack
snapshot. That is why the spike saw 10,790 register entries per resolve and 0
consumed: the resolver never reaches an `unwind_trail_to` with a register
entry in its window.

So a register entry matters only if (a) it lands inside the window
`(mark, trail.len()]` of an `unwind_trail_to(mark)`, and (b) the restored
register is read afterwards before being written again or restored by some
other means. (a) needs instructions to run inside the window (a nested
`run()`, `vm.step(..)` in lowered code, or one of the builtins that call
`trail_binding` directly). Trial unifications only push binding entries.

### 1.1 Push sites

Runtime `trail_binding` calls (excluding lib tests at `ST:1849`, `ST:2107`,
`ST:9848`, `ST:9878`):

| site | instruction or builtin | register | why it exists |
| --- | --- | --- | --- |
| `T:128` | `GetConstant` on an unbound argument | `ai` | mirrors the register write; the binding is trailed separately |
| `T:140` | `GetVariable xn, ai` | `xn` (X or Y) | Y: the slot write is needed by trail-only sites (§6.2 of the design); X: no consumer |
| `T:188` | `GetStructure` write mode | `ai` | register write mirror |
| `T:239` | `GetList` write mode | `ai` | register write mirror |
| `T:281` | `UnifyVariable` read mode | `xn` (X or Y) | as `GetVariable` |
| `T:360` | `PutConstant` | `ai` | register write mirror |
| `T:367`, `T:368` | `PutVariable xn, ai` | `xn`, `ai` | as above |
| `T:375` | `PutValue` | `ai` | register write mirror |
| `T:453` | `LoadRegisterConstant` | `reg` | register write mirror |
| `T:636-640` | `RecurseCategoryAncestorPc` | child-hops reg, `A1`-`A4` | call-argument setup |
| `T:1138` | embedded aggregate result | result reg (often Y) | the §6.2 item 2 live-Y overwrite |
| `T:1584` | `is/2` result into `A1` | `A1` | register write mirror; `A1` is a dead builtin argument afterwards |
| `T:5177` | aggregate-frame finalization | result reg (often Y) | the §6.2 item 2 live-Y overwrite |
| `ST:4217` | `head_constant` (lowered `get_constant`) | `ai` | mirrors `T:128` |

None of these exists for a non-CP consumer. All are "trail every register
write the interpreter makes", added when choice points did not save
registers. Register writes that are **not** trailed: `PutStructure`,
`PutList`, `SetVariable`, write-mode `UnifyVariable`, `Cons`, the register
patch in `set_heap_or_list`, `call_goal_key`'s `set_reg` of `A1..An`
(`T:7505-7507`), `\+/1`'s argument loads (`T:6953-6955`), and every direct
lowered write (`LE:942-985`, `put_reg` / `set_reg_str`). Any consumer that
relied on the register trail was therefore already relying on an incomplete
record.

## 2. Consumer sites

### 2.1 Counts

| group | sites | class |
| --- | ---: | --- |
| `T` builtin trial unifications (`unwind_trail_to` after a failed `unify`) | 50 | no register entries in the window |
| `T` `catch/3` | 3 | 2 explicit restore, 1 no register entries |
| `T` `include/3`/`exclude/3`, `partition/4` | 2 | **needs explicit save** |
| `T` `predsort_order` | 3 | explicit restore |
| `T` `call_goal_value` (`->`, `;`, `\+`) | 3 | registers dead after rollback |
| `T` = 61 `unwind_trail_to` calls | **61** | |
| templates, executable `unwind_trail_to` calls: 22 builtin-family trial unifications, 5 `DB`, `lo_restore_clause`, `region_decline` | **29** | 23 no entries, 4 explicit restore (`DB`), 1 explicit restore (`lo_restore_clause`), 1 no entries (`region_decline`) |
| templates, `unwind_trail_to` definition + 5 lib-test calls | 6 | not runtime |
| = the design's "35 in templates" | 35 | |
| other trail consumers: `backtrack` (2), `apply_foreign_result` (2), `call_goal_once` Y filter (1), `reset_query` clear (1) | 6 | never apply register entries |
| callers of `lo_restore_clause`: `lowered_dispatch`, F11 (2), T4 | 4 | explicit restore |
| lowered ITE else-branch unwind | 1 (per emitted ITE) | registers dead after rollback |
| callers of `region_decline` (9 region dispatchers) | 35 | no register entries |

Totals over the runtime sites (61 + 29 + 6 + lowered ITE): **97**; plus the
39 helper call sites. **Needs explicit save: 2.** None needed: 95 (no
register entries in the window 75, explicit restore already 10, registers
dead after the rollback 4, not a consumer of register entries 6), and the
39 call sites inherit their helper's class.

### 2.2 Every row

`T` builtin trial unifications. In each, the window between `let mark =
self.trail.len()` and the unwind contains only `unify` calls, which push
binding entries; nothing runs instructions. Rolls back to: the builtin's
entry. A/X read afterwards: none restored by the trail, because the window
has no register entries. Replacement: none.

| builtin | `T` lines |
| --- | --- |
| `format/1-3` (`atom(_)`/`string(_)`/`codes(_)` sink) | 1834 |
| `file_base_name/2`, `file_directory_name/2` | 1947 |
| `file_name_extension/3` | 1974, 1980, 2002 |
| `path_join/3` | 2037 |
| `realpath/2`, `read_link/2` | 2062 |
| `read_file_to_atom/2` | 2136 |
| `directory_files/2` | 2268 |
| `size_file/2` | 2288 |
| `time_file/2` | 2311 |
| `get_time/1` | 2326 |
| `getenv/2` | 2345 |
| `getpid/1` family | 2390 |
| `length/2` | 2499, 2520 |
| `append/3` | 2549, 2564, 2577 |
| `functor/3` | 2638, 2659 |
| `arg/3` | 2700 |
| `=../2` | 2743, 2773 |
| `copy_term/2` | 2801 |
| `term_variables/2` | 2836 |
| `numbervars/3` | 2866, 2873, 2881 |
| `unifiable/3` (also reads the window's binding names, `T:2927`; register entries are skipped by `binding_name()`) | 2922, 2937, 2941 |
| `reverse/2` (`execute_reverse_builtin`) | 3047 |
| `fact_table_attempt` (per-row trial; the success path pushes a CP with `save_regs`) | 5043 |
| `builtin_unify_member` | 5765 |
| `\=/2` | 5893 |
| `sort/4` | 5995 |
| `pairs_keys/2`, `pairs_values/2` | 6031 |
| `pairs_keys_values/3` | 6056, 6066, 6072 |
| `memberchk/2` | 6088 |
| `delete/3` | 6146 |
| `intersection/3` | 6190 |
| `union/3` | 6217 |
| `list_to_set/2` | 6238 |
| `split_string/4` | 6387 |
| `atom_split/3` | 6417 |
| `sub_atom/5` | 6499, 6505 |

Template builtin families, same pattern (window = one or two `unify`
calls; class: no register entries; replacement: none):
`os_error_builtin` 38, 57; `os_utility_builtin` 90; `process_builtin` 118,
132, 163, 189; `process_context_builtin` 34, 44, 52, 73;
`process_metrics_builtin` 82; `process_resource_builtin` 143, 179, 210;
`random_builtin` 60, 84; `stream_builtin` 77, 98; `time_builtin` 170, 189,
215.

The sites that run code inside the window:

| site | rolls back to | A/X restored by the trail and read afterwards? | replacement |
| --- | --- | --- | --- |
| `T:7052` `catch/3`, goal failed | catch entry (trail, heap, stack, CPs, `cp`, B0) | no: `restore_regs(&saved_regs)` (`T:7055`) runs after the unwind and wins | none: explicit restore |
| `T:7064` `catch/3`, ball caught | catch entry, then the catcher is unified | no: `restore_regs` at `T:7067` | none: explicit restore |
| `T:7085` `catch/3`, catcher does not unify | `mark2`, after the catcher trial | no register entries (window = one `unify`) | none |
| `T:7204` `include/3`, `exclude/3`, per element | before the element call | **yes**: `A3` read at `T:7208` after the loop; the element goal's callees load `A3` through trailed instructions | **read `A3` at entry**, before the loop |
| `T:7228` `partition/4`, per element | before the element call | **yes**: `A3`, `A4` read at `T:7232-7233` | **read `A3`, `A4` at entry** |
| `T:7373`, `T:7379`, `T:7388` `predsort_order` | before the comparator call | no: `restore_regs(&saved)` at `T:7371` runs before the unwind. Today the unwind then re-applies the comparator body's register entries, whose oldest old values for `A1..A3` are the comparator's own arguments (loaded untrailed by `call_goal_key`), so after the call `A1..A3` can hold the comparator arguments rather than predsort's; nothing reads them (`predsort/3` captured `A3` at `T:7162`). Without A/X entries the explicit restore stands | none (R0b makes the explicit restore authoritative) |
| `T:7463` `call_goal_value`, `( C -> T ; E )` with `C` failed | before the condition | no: the next step is `call_goal_value(&E)`, which reloads `A1..An` from `E`'s term (`call_goal_key`, `T:7504-7507`) or recurses structurally; the callers of `call_goal_value` read no A/X afterwards (below) | none: registers dead after rollback |
| `T:7469` `call_goal_value`, `( A ; B )` with `A` failed | before `A` | no: as above, `B` reloads its arguments | none |
| `T:7486` `call_goal_value`, `\+ G` | before `G` | no: returns to its caller (below) | none |
| `T:7347` `call_goal_once`, trail tail filter | (not a rollback) drops the nested run's Y entries, keeps binding and A/X entries | n/a. After R0b there are no A/X entries to keep. Its comment (`T:7331-7345`: "Ai/Xi entries (restored explicitly by the caller's save/restore)") is wrong for `include`/`exclude`/`partition` and should be updated | none |
| `T:1466`, `T:1470` `backtrack` | the top CP | never applies register entries (bindings only, then truncate); A/X from `saved_args` | none |
| `T:4326`, `T:4327` `apply_foreign_result` | before the tuple | bindings only, then truncate (drops register entries; the window has none), then `restore_regs` | none |
| `ST:4301` `reset_query` | (clears the trail) | n/a | none |
| `ST:4539` `lo_restore_clause` | clause-entry snapshot | no: `restore_regs(&snap.saved_args)` (`ST:4540`) runs after the unwind and wins | none: explicit restore |
| `ST:4599` `lowered_dispatch` decline | entry snapshot | via `lo_restore_clause` | none |
| `LE:598`, `LE:607` F11 loop, `LE:815` T4 clause attempts | `_f11` / `_t4` snapshots | via `lo_restore_clause` | none |
| `LE:855` lowered ITE, condition failed | `_ite_mark` | no A/X read before being reloaded. The condition can push A/X entries (delegated `vm.step` arms, `is/2`, nested `run()` of a lowered `call`) and the else branch runs right after the unwind, but every variable live across an ITE is permanent: the clause allocates (`get_level` needs a Y, `WT:1335`) and the variables are pre-assigned Y registers (`WT:1302`, R−1c), so the else branch reads Y registers and constants and loads A/X with `put_*` first. Checked in every ITE the test file lowers (`ite_temps`, `ite_is`, `ifthen`, `multi_ite`, `naf_temp`). Y entries keep working (R0b keeps them) | none; add the emit-time check in §5 |
| `ST:4736` `region_decline` (35 call sites in 9 region dispatchers, `ST:4665`-`ST:6118`) | P2 snapshot (trail, heap, `var_counter`) | no register entries: regions write no register (G-1) and call no interpreted code; checked by grep over `ST:4600-6130` (no `put_reg`/`set_reg`/`step`/`run`/`trail_binding`) | none |
| `DB:540` `dynamic_clause_matches_pattern` | before the trial | no register entries (one `unify`) | none |
| `DB:600` `dynamic_call_attempt`, clause failed | before the clause | no: the head is unified in `self` (bindings only) and the body runs on clones (`dynamic_body_solutions` takes `&self`); `restore_regs(&cp_regs)` and the stack restore follow anyway | none |
| `DB:868` `current_predicate/1` | before the candidate | no register entries; `restore_regs` follows | none |
| `DB:957` `clause/2` | before the candidate | no register entries; `restore_regs` follows | none |
| `DB:1028` `retract/1` | before the candidate | no register entries; `restore_regs` follows | none |

The callers of `call_goal_value`/`call_goal_once` and what they read after
the call: `call/N` (`T:6874`) sets `pc` and continues interpreted code;
`maplist/2..5` (`T:7092`) reads only Rust locals; `foldl/4,5` (`T:7238`)
reads `out_reg` at `T:7275` (see §4, not a rollback consumer);
`predsort/3` captured its output before; `catch/3` restores explicitly;
the dynamic body solver (`DB:657`) works on a clone that becomes `*self`.
Interpreted code past a `call`/`execute` reads no A/X it loaded before the
call: a variable used in a later goal is permanent (`WT:1641-1690`). The
hybrid wrappers' `vm.backtrack()` loops (`RT:4573`, `4723`, `4933`, `5056`,
`5206`, `5341`) and `par_aggregate`'s `backtrack()/run()` (`PA:40`, `59`,
`171`, `196`) are choice-point based and never read trail entries.

## 3. Test net and the disabled-trail matrix

`tests/test_wam_rust_register_restore.pl` follows
`tests/test_wam_rust_baseline_fixes.pl`: one source of truth, SWI-Prolog as
the oracle, a failure-driven driver run by WAM label in
`emit_mode(interpreter)` and `emit_mode(functions)`, and a direct call of the
lowered function when the functions-mode emitter lowered the query
predicate. A shared helper `rr_w/5` is called with five arguments so that the
callers' argument registers are overwritten through `put_constant`,
`put_value` and `get_variable` (the trailed instructions), and then the goal
fails or its scope is rolled back.

Programs, by audited kind: `\+` inline, meta-called (structural `\+` in
`call_goal_value`) and followed by a temporary read; `call/1` of a failed
disjunction (left branch clobbers `A1`), of an if-then-else whose condition
fails, `call/2`, `once/1`; `catch/3` with a failing goal, a throwing goal and
a failing recovery; `findall/3` over a failing and a clobbering goal, a
`findall/3` or `aggregate_all/3` nested in `findall/3` (also over a goal
that always fails); `forall/2` holding and failing; `predsort/3` with a
clobbering comparator and with one that fails partway; `include/3`,
`exclude/3`, `partition/4`, `maplist/2` with a failing element; dynamic rule
bodies (clobber then fail; `\+` in the body); if-then-else whose condition
writes temporaries (a call, `is/2`) then fails, nested, and a bare
if-then in a disjunction; multi-clause predicates whose first clause
clobbers and fails (call, pure builtins, and a lowered ITE whose condition
calls a predicate and whose else branch reads both arguments). 35 programs,
120 checks.

Matrix (scratch builds; the template change was reverted, never committed):

| variant | plunit | checks that change vs main |
| --- | --- | --- |
| main `5f08375` | **120 passed** | — |
| A/X pushes disabled (`trail_binding` returns unless the name starts with `Y`): the R0b emulation | 117 passed, **3 failed** | interpreter `include` (`[2,3]` → no solution), `exclude` (`[1]` → no solution), `partition` (`parts([2,3],[1])` → no solution) |
| all register pushes disabled (the spike's s3) | not run as plunit; the per-check matrix below shows the same 3 failing checks | same; the Y entries change no check in this file |

The per-check matrix (both variants, every program, both modes, plus the
lowered entries) was produced by a scratch driver over the test module's
own predicates; the only differing rows are the three above. All lowered
direct entries (`ite_temps`, `ite_is`, `ifthen`, `multi_call`, `multi_pure`,
`multi_ite`, `maplist_fail`) pass in both variants, which confirms the
lowered-ITE and `lo_restore_clause` rows of §2.2.

**Out of scope, recorded in the test file.** Nine `rr_known_divergence/3`
entries are not compared with SWI; `known_divergences_still_present`
asserts that each still differs, so the fix that removes one must delete its
line:

- functions mode `include`, `exclude`, `partition`: wrong today (item 4 of
  the summary). The R0b fix (read `A3`/`A4` at entry) fixes them.
- `foldl` (both modes) and `include_wide` (`include/3` with an extended goal
  of arity 3 or more, both modes): §4.
- `nested_agg_inline` (both modes): §4.

Twenty-one `rr_lowered_out_of_scope/2` entries skip a direct lowered entry
that is wrong, or never reaches its rollback, for a lowered-tier reason
(§4); their interpreted entries in functions mode are still checked.

## 4. Other defects found (pre-existing, outside R0b)

1. **Output-register reads after nested calls (same family as the R0b
   sites).** `foldl/4,5` reads `A4`/`A5` at `T:7275` after the element
   calls, on the success path where nothing restores it; an element goal
   whose callees load `A4` makes `foldl` fail (`rr_foldl`: SWI `6`, Rust no
   solution). `include/3` with an extended goal of arity ≥ 3 reads `A3`
   after `call_goal_key` loaded it with the last element
   (`include(rr_between(1,2), [1,5,2], R)`: SWI `[1,2]`, Rust no solution).
   Same fix: read the output registers at builtin entry.
2. **Lowered ITE condition with a failing `call`.** The lowered `call`
   (`LE:1049-1063`) restores neither the stack nor `E` when the callee
   fails, so after a failed condition the else branch reads Y registers from
   the callee's abandoned frame (`rr_naf_temp/3` lowered: SWI `got(a,b)`,
   lowered no solution; the interpreted entry is correct). Related to the
   design's §6.6 "no protection of the current frame".
3. **`\+` written as a term in a compiled body does not compile.** The
   functor is emitted unescaped as the Rust literal `"\+/1"` (`T:8527` in the
   instruction table, `LE:992` in lowered code), which `rustc` rejects
   (`unknown character escape`). The test builds `\+(G)` at run time to
   avoid it.
4. **`ignore/1` is missing** (no builtin, not inlined): `call ignore/1`
   fails. **`aggregate_all/3` inside a `findall/3` goal** is compiled as
   `call aggregate_all/3`, which has no builtin or label, so the inner goal
   fails (`nested_agg_inline`: SWI `r(a,[c(p,2),c(q,2)])`, Rust
   `r(a,[])`). As a clause-body goal it is inlined and works (`nested_agg`).
5. **The `\+/1` builtin arm** (`T:6906`) pushes a CP named `naf_succeed`
   that `resume_builtin` does not handle and sets no backtrack floor. It is
   unreachable with the default `inline_not_as_failure(true)` (compiled `\+`
   is inlined, and `call_goal_value` handles `\+` structurally).
6. `write/1` prints operators in canonical form (`-(a,b)`) and lists with a
   space after commas (`[a, b]`); the tests avoid both.
7. A lowered `call`/`execute` of a builtin with no label fails (already noted
   by `tests/test_wam_rust_baseline_fixes.pl`).

## 5. R0b implementation plan

1. **`trail_binding`** (`ST:6180`): push only for Y names
   (`if key.as_bytes().first() != Some(&b'Y') { return; }` before the
   `get_reg`). `xn` operands can be X or Y, so the class check stays inside
   the function. The call sites that always name an argument register can
   drop the call entirely: `T:128`, `T:188`, `T:239`, `T:360`, `T:368`,
   `T:375`, `T:637-640`, `T:1584`, `ST:4217` (keep `T:140`, `T:281`,
   `T:367`, `T:453`, `T:636`, `T:1138`, `T:5177`, whose register can be Y).
   Check the generated code for any `get_constant`/`put_*` with a Y in the
   `ai` position before dropping a call; the class check in the function is
   the safe default.
2. **`include/3`/`exclude/3`** (`T:7184`): `let a3 =
   self.get_reg_raw("A3")...` before the element loop. **`partition/4`**
   (`T:7211`): `a3` and `a4` before the loop. Recommended in the same change
   (they fix known divergences, so `known_divergences_still_present` will
   ask for their lines to be removed): `foldl/4,5` reads `out_reg` (and keeps
   `acc0_reg` as today) before the loop. After this, delete the
   functions-mode `include`/`exclude`/`partition`, `foldl` and
   `include_wide` lines from `rr_known_divergence/3` and the matching
   `rr_lowered_out_of_scope/2` lines.
3. **Lowered ITE** (`LE:838-855`): no register save. Add an emit-time
   check that the else branch (and the code after the ITE) reads no A/X
   register before writing it, so a future compiler change cannot silently
   start relying on the removed entries.
4. **Comments:** `call_goal_once` (`T:7331-7345`) and `unwind_trail_to`
   (`ST:4268-4274`): register entries are Y-only.
5. **Lib tests** that push A/X entries and expect them back:
   `d125_bindings_tests` (register entries in the model, `ST:1849`),
   `d127_register_trail_tests` (`ST:2107` drives `trail_binding` over A/X/Y
   names), the trail tests at `ST:9848-9888`. Switch their register names to
   Y, or assert that an A/X push leaves the trail unchanged.
6. **Gates:** this test file (120/120, with the moved lines now checked),
   `tests/test_wam_rust_baseline_fixes.pl` (92/92),
   `tests/test_wam_rust_lowered_ite_exec.pl`, `tests/test_wam_rust_lowered_t4.pl`,
   the D127 gate set (term and store differentials, corpora, byte identity of
   the outputs), generated-crate lib tests.
7. **Interaction with R0a.** Independent: R0a narrows what clause CPs
   restore, and CP backtracking never applied register entries. Run this
   file again once both have landed.

**Expected gain, to re-measure.** The spike's s2→s3 step removed every
register entry (A/X and Y) for −1.88 M Ir per warm resolve at N=40 (−8.1%
of base) and −0.7% at N=5000. R0b removes only the A/X share, plus the
`get_reg` that each A/X push performs. Before the change, count
`trail_binding` calls by register class on the N=40 resolve (a temporary
counter, as in the spike's `spike_stats` build): the A/X fraction of the
10,790 pushes, times ~170 Ir per push (1.88 M / 10,790), is the expected
saving. Then measure Ir per warm resolve at N=40 and N=5000 against main.

## 6. Streams and other non-term state

### 6.1 What exists

| state | where it lives | backtracking today |
| --- | --- | --- |
| line streams: `stream_open/2`, `read_line/2`, `stream_close/1` | process-wide static table of `BufReader<File>` (`stream_builtin.rs.mustache`, `line_reader_table`); a stream is an `Integer` handle in the term | never restored |
| `read/1`, `read_term/1,2` input | `WamState.term_input` (all of stdin, read once) and `term_input_pos` (`ST:2214-2215`, `DB:9-43`) | not saved in CPs or scopes: never restored; machine clones carry their own copy |
| output: `write/1`, `print/1`, `nl/0`, `format/1-3`, `tab/1`, `put_char/1` | `stdout`/`stderr`, written immediately | never restored |
| `read_file_to_atom/2`, file system, `getenv/setenv`, `working_directory/2`, `umask/2`, process/time/metrics | the OS | never restored. `working_directory/2` changes directory only after its unifications succeeded, so a failed call leaves the directory unchanged |
| `random/1`, `random_between/3`, `set_random/1` | libc `drand48`/`lrand48` state | never restored (SWI does not either) |
| dynamic database | `WamState.dynamic_db` | not restored on backtracking (logical update view, like SWI); clones in the dynamic body solver carry it |
| fact sources: `seek_fact_source`, `csr_fact_source`, LMDB, `lazy_lookups`/`lazy_category_parents`, `boundary_cache` | read-only stores behind `Arc`; `Mutex` caches (opened store, decoded-row L1 cache, I/O counters) | no rollback needed: each call materializes its rows into a `Vec`, and the remaining rows travel in the CP's `builtin_state` |
| global variables (`b_setval`/`nb_setval`) | not implemented in this runtime | — |

There is no `peek_char`/`peek_code`, `get_char/2` on streams,
`stream_property/2`, `set_stream_position/2`, `seek/4`, `open/3,4` or
`phrase_from_file/2` in the Rust runtime, so those could only be run on the
SWI side.

### 6.2 Answers

**(a) Does any register entry carry stream or external state?** No. A
register entry stores a `Value` (`ST:6180-6183`). A stream is an `Integer`
handle inside a term; restoring a register can only restore which handle a
register names, never the stream's position, buffer or open/closed state.
The term-input position and the stream table are not registers. Removing
A/X entries cannot change stream behaviour.

**(b) Today's behaviour, and SWI.** Backtracking does not undo input
consumption, stream position or output, as in standard Prolog. Comparison
programs (one driver each; the Rust programs use the runtime's
`stream_open/2`/`read_line/2`/`stream_close/1`, and SWI runs the same text
with three shim clauses over `open/3`, `read_line_to_string/2`, `close/1`;
input file lines `l1`..`l4`; stdin `t1. t2. t3. t4.`):

| program | SWI-Prolog 9.0.4 | Rust interpreter | Rust functions mode |
| --- | --- | --- | --- |
| `( read_line(S,L), L == nomatch -> true ; read_line(S,L2), write(L2) )` | `l2` | `l2` | `l2` |
| `( read_line(S,L), write(L), nl, fail ; read_line(S,L2), write(L2) )` | `l1`, `l2` | same | same |
| `( member(_,[1,2,3]), read_line(S,L), L == l3 -> write(found(L)) ; true ), read_line(S,L4)` | `found(l3)`, `next(l4)` | same | same |
| `\+ \+ read_line(S,_), read_line(S,L)` | `l2` | `l2` | `l2` |
| `findall(L, (member(_,[1,2]), read_line(S,L)), Ls), read_line(S,L3)` | `[l1,l2]`, `l3` | same (prints `[l1, l2]`) | same |
| `catch((read_line(S,_), throw(x)), x, true), read_line(S,L)` | `l2` | `l2` | `l2` |
| `( write(a), fail ; write(b) )` | `ab` | `ab` | `ab` |
| `findall(X, (member(X,[1,2]), write(X)), _)` | `12` | `12` | `12` |
| `( read(X), X == nomatch -> true ; read(Y), write(Y) )` (stdin) | `t2` | `t2` | `t2` |
| `( read(X), write(X), fail ; read(Y), write(Y) )` | `t1t2` | `t1t2` | `t1t2` |
| `( assertz(st_f(1)), fail ; true ), findall(X, st_f(X), L)` | `[1]` | `[1]` | `[1]` |
| asserted `st_dw(X) :- member(X,[a,b]), write(X)`; `( st_dw(_), fail ; true )` | `ab` | **`aabab`** | **`aabab`** |
| asserted `st_dr(S,L) :- member(_,[1,2]), read_line(S,L)`; `( st_dr(S,L), write(L), nl, fail ; true ), read_line(S,A)` | `l1`, `l2`, `after(l3)` | **`l1`, `l3`, `after(end_of_file)`** | same as interpreter |
| asserted `st_drt(T) :- member(_,[1,2]), read(T)`; `( st_drt(T), write(T), nl, fail ; true ), read(U)` | `t1`, `t2`, `after(t3)` | `t1`, `t2`, `after(t3)` | same |
| asserted `st_da(X) :- member(X,[a,b]), assertz(st_log(X))`; all solutions, then `findall(L, st_log(L), Ls)` | `[a,b]` | `[a, b]` | same |

SWI-only, for the operations the Rust runtime lacks: `( peek_char(S,_),
fail ; true ), peek_char(S,C1), get_char(S,C2)` gives `l`, `l` (peek does
not consume; failure does not matter); `line_count` before and after
`( read_line_to_string(S,_), fail ; true )` is 1 and 2 (the position is not
restored); `set_stream_position/2` is the explicit way to re-read;
`phrase_from_file/2` with a first DCG alternative that reads and fails
re-reads the same input in the second alternative (`second`): the input is
a lazy list, so the replay comes from the term, not from the stream;
`b_setval/2` is undone on backtracking and `nb_setval/2` is not.

The one divergence: **dynamic rule bodies are re-run from their start on
each redo.** `dynamic_eval_body_solution` (`DB:624-630`) computes solution
`idx` by running the body again on clones with `limit = idx + 1`
(`dynamic_body_solutions`, `DB:632-668`) and adopts it with
`*self = solutions.remove(idx)` (`DB:628`), and the `dynamic_rule_body` CP asks for `idx + 1` on the
next redo. Pure bodies give the same answers, but each redo repeats the
body's side effects: output is written again (`aabab`), reads from the
process-wide stream table consume more lines (`l3` instead of `l2`). Reads
through `read/1` happen to match SWI, because the input position is a
machine field and the discarded clones' positions are discarded with them.
Asserts happen to match for the same reason. This is independent of the
register trail and of R0b.

**(c) Does any lazy or stream-backed structure rely on rollback?** No.
Fact sources materialize each call's rows into a `Vec` and carry the
unconsumed rows in the CP; their caches are memoization over read-only
data. `read/1` reads all of stdin into a `String` on first use and keeps an
index into it; that index is not rolled back. There are no lazy lists.

**(d) Proposal (not implemented).** Keep the SWI semantics as the default:
no I/O rollback.

1. *Fix the replay.* Make the dynamic body solver resumable instead of
   re-running: keep the suspended body machine (today's `vm` clone after it
   yields solution `k`, with its CPs) in the `dynamic_rule_body` CP's
   state, and on redo call `backtrack()` + `run()` on it to get solution
   `k+1`. Each side effect then happens once, as in SWI. Cost: one stored
   machine per pending dynamic redo (it is already cloned today), and the
   generation/scope rules of the rewrite apply to it as to a forked
   machine (§9.4 of the design).
2. *One home for stream state.* Move `term_input`/`term_input_pos` into the
   process-wide stream table as the `user_input` handle, so clones share it.
   Then every machine copy (dynamic body solver, `par_aggregate` workers,
   `lowered_dispatch` retries) sees one consumption order, which is SWI's
   model. Today a clone that reads and is discarded gives its input back.
3. *Opt-in replayable input.* Provide the SWI shape:
   `phrase_from_stream/2` (and `phrase_from_file/2`) over a lazy list. A
   lazy list is a normal list whose last tail is a *lazy cell* bound to
   `(handle, offset)`. Forcing it reads the next chunk into a per-handle
   append-only buffer, at `offset`, and binds the tail to the chunk's cells
   plus a new lazy tail at the next offset. The binding is an ordinary
   trailed binding, so backtracking unbinds the tail; the buffer keeps the
   chunk, and re-forcing the same `(handle, offset)` reads from the buffer
   (the "rollback index"), so the same input comes back. The stream itself
   only moves forward. On the heap-cell design this is one more cell kind
   (or a `BOX` payload) checked by `deref` in the list paths, and it needs
   no register-trail support. Work: the cell kind and its forcing hook in
   `get_list`/`unify`/`deref`, the buffer in the stream table, the two
   builtins, and tests that compare with SWI's `pure_input`.

## 7. Results: the R0b implementation (D136)

**Code:** this branch, on top of the audit commit. Ledger row D136 in
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`. It follows the plan of §5;
no choice-point code (`save_regs`, `ChoicePoint`, `backtrack`) changed, so it
is independent of R0a.

### 7.1 What changed

| item of §5 | change |
| --- | --- |
| 1. `trail_binding` | Returns before the `get_reg` unless `is_y_reg_name(key)` (the test `get_reg`/`put_reg` route on, so an entry is always undone into the storage it was read from); marked `#[inline]`. The calls that can only name an argument register are dropped: `RecurseCategoryAncestorPc`'s `A1`-`A4` and `is/2`'s `A1`. Every other call site keeps its call, because its operand can be a Y register (`xn`, `reg`, `result_reg`) or comes from the instruction text (`ai`); the in-function test makes those no-ops for A/X. |
| 2. read outputs at entry | `include/3`, `exclude/3` read `A3`, and `partition/4` reads `A3` and `A4`, before the element loop. `foldl/4,5` reads its output register (`A4`/`A5`) before the loop. The include change also fixes `include/3` with an extended goal of arity ≥ 3 (`include_wide`). |
| 3. lowered ITE | `emit_instrs/3` calls `rust_ite_else_ax_check/2` for every ITE it emits: along the else branch followed by the rest of the enclosing instruction list, every A/X register must be written before it is read. A `call`/`execute`/`proceed`/`fail` ends the path. A nested ITE continues from the registers its condition and then branch wrote, restarts its else branch from the set before it, and passes on the intersection. A violation throws `error(lowered_ite_else_reads_ax_before_write(Reg, Instr), _)` at generation time. No generated program in the test suites, the conformance smoke or the resolver builds triggers it. |
| 4. comments | `call_goal_once` (the Y filter: after R0b the kept tail is binding entries only), `unwind_trail_to` (restores no A/X register), `trail_binding`, and the lowered ITE emitter. |
| 5. lib tests | `d125_bindings_tests`: the register step pushes `A1` (no entry) and `Y1` (one entry). `d127_register_trail_tests`: the reference trail records Y names only, and every entry is checked to be Y. `trail_enum_tests`: the round trips use `Y1` in an Env frame and check that an `A1` write survives the unwind; new `trail_binding_records_y_registers_only` (A/X names, including `A01`, `X007`, `A100`, push nothing). |

Test file: the functions-mode `include`, `exclude`, `partition` rows and
both modes of `foldl` and `include_wide` now match SWI-Prolog, so their
`rr_known_divergence/3` lines and the matching `rr_lowered_out_of_scope/2`
lines are removed (the direct lowered entries of those five programs are
now checked and pass). A new block `wam_rust_r0b_ite_else_check` (no cargo
needed) checks the emit-time check on seven instruction lists.

**Sites the audit missed:** none. No gate below changed relative to the
base, and the two classes the audit predicted (the 2 sites of §2.2 and the
`foldl`/`include_wide` defects of §4) are the only behaviour changes.

### 7.2 Register trail pushes by class (base, before the change)

A temporary counter build of the base crate (relaxed atomic counter per
first byte of the name in `trail_binding`; never used for Ir or wall
numbers), one `resolve_layered([p30])` resolve:

| class | `trail_binding` calls, N=40 | share | after R0b |
| --- | ---: | ---: | --- |
| A | 4,824 | 44.7% | no entry |
| X | 3,052 | 28.3% | no entry |
| Y | 2,924 | 27.1% | one entry each, as before |
| total | 10,800 | | 2,924 register entries |

The counts are identical at N=5000 and identical for every in-process
repeat. The audit's estimate was 7,876 A/X pushes × ~170 Ir ≈ 1.34 M Ir.

### 7.3 Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8`. Base is the audit
commit (`5f08375` code): the term crate built in the worktree before the
change, the store crate and plunit in a `git archive` export of it.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout at N=40 and N=5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 261 passed; 0 failed` (260 + the new test) |
| `tests/test_wam_rust_register_restore.pl` | 113/113 (was 120 with 9 known-divergence checks; 7 of those programs are now compared with SWI) + 7/7 for the emit-time check |
| `tests/test_wam_rust_baseline_fixes.pl` | 92/92 |
| Rust WAM plunit (58 files: `tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`) | same per-file rc as base (36 rc=0, 22 rc≠0) and the same 33 failing test names |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, unsampled and with `CONFORMANCE_SAMPLE=2` |

### 7.4 Measurements

Method as in `docs/reports/wam_rust_register_trail_static_names.md`: a
scratch copy of the generated crate with a shim that repeats `call_pred`
in-process and checks that every answer is equal; Ir per warm resolve is
`(Ir(1+k) − Ir(1)) / k` under callgrind, k=10 at N=40 and k=2 at N=5000.

| N | base | R0b | Δ |
| ---: | ---: | ---: | ---: |
| 40 | 23.10 M | **21.90 M** | **−1.20 M (−5.2%)** |
| 5000 | 388.25 M | **387.07 M** | −1.18 M (−0.3%) |

The saving is flat across N, as the push counts are. At N=40,
`trail_binding` (inclusive) per warm resolve goes from 1.55 M to 0.54 M,
and `backtrack` (inclusive), which drops the truncated trail tail, from
about 3.32 M to 3.17 M. In all, ~152 Ir per avoided push. A whole `--bench` process at N=40 (one
cold resolve, load included) goes from 34.29 M to 32.94 M Ir (−3.9%).

Wall clock (shared 4-core container), base and new interleaved, 11
in-process resolves per run; warm = median of resolves 2..11, then the
median over rounds:

| N | rounds | cold base | cold R0b | warm base | warm R0b | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 21 | 5.22 ms | 4.49 ms | 3.50 ms | **3.36 ms** | −4% |
| 40 | 41 | 4.91 ms | 4.27 ms | 3.46 ms | **3.25 ms** | −6% |
| 5000 | 9 | 59.31 ms | 50.98 ms | 53.69 ms | 52.71 ms | (noise) |

Ir is the reliable metric.
