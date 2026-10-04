<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# PLAN: plawk / LLVM-WAM template refactor

Base: feature line `claude/plawk-llvm-wam-hybrid-p9ujut` (HEAD 7abb77cea, pattern_stache ported). All counts below were measured on that tree on 2026-10-02; `gh/main` was consulted for the template corpus and the one production `.stache` consumer.

## 0. Recommendations in one screen

1. **The hazard is one atom.** 116 of the 183 embedded LLVM `define`s live in a single quoted atom, `compile_execute_builtin_to_llvm/2`, `src/unifyweaver/targets/wam_llvm_target.pl:4537-21336` (16,799 lines, 131 lines containing `''`, exactly one hole `__M113_DIRENT_NAME_PTR__`, spliced at :21327-21329). Move it first, as one file, byte-verified. That removes ~63% of the exposure in one mechanical PR.
2. **Static IR goes to `.ll.mustache` libraries parsed once per process**, not rendered per function. Rendering `state.ll.mustache` (6,389 lines) costs 11 ms; a per-function whole-file render of a 17k-line library would be ~100x that. Use the Go target's library idiom (`gh/main:src/unifyweaver/targets/wam_go_templates.pl:139-160, 377-430`: read once, scan `{{case NAME}}`, validate case-set == order).
3. **`.stache` is for families, not just for AST dispatch.** The decision rule (§3) picks `.stache` whenever the case set is a family over a term shape: bodies identical modulo names/types/tags bound by the match. The existing `.mustache` corpus holds almost no such families (§2.1, honest finding); the families live in the Prolog-embedded IR and in plawk's emitters (§2.2-2.3), which is exactly where they would become literal `.mustache` cases if moved naively.
4. **Pilot:** `assoc_elem.ll.stache` for plawk -- three `@wam_assoc_{i64,f64,str}_print` call-site strings + eleven `@wam_assoc_i64_inc` + three `@wam_assoc_f64_add` strings -> 3 structural cases. Validated during planning with a prototype template (load 0.4 ms, render 0.15 ms, output byte-equal to today's lines); the template itself lands with PR 4.
5. **Byte-identity has one known exception:** the runtime atom contains Prolog escapes that evaluate to raw NUL bytes (`c"end_of_file\00"` emits a literal 0x00 into the `.ll`, verified at `probe/count.ll:8592`). A template file must either carry raw NULs (git sees binary) or spell them as LLVM `\00` -- the latter is a one-time deliberate re-baseline verified with `llvm-as | llvm-dis`.

## 1. Inventory

### 1.1 `src/unifyweaver/targets/wam_llvm_target.pl` (28,302 lines)

| area | where | defines | holes | kind |
|---|---|---|---|---|
| builtin dispatch + plawk runtime (assoc table, fields, split, strnum, awk number fmt, regex, streams/getline, cache, term reader, string builder) | `:4537-21336` one atom | 116 | 1 (`__M113_DIRENT_NAME_PTR__`) | static |
| backtrack / unwind | `:4340-4536` | 2 | 0 | static |
| eval_arith family | `:21337-21722` | 3 | 0 | static |
| copy_term family | `:22409-22954` | 8 | 0 | static |
| term_cmp, ssp_emit_segment | `:22955`, `:23088` | 2 | 0 | static |
| atom string globals + intern/hash | `:23429-23870` | 14 | per-module data | parameterised |
| driver `main`s (text/binary/varlen x argv/fixed-path) | `:24278-24811` | 6 | `~w` holes | parameterised, **2x3 grid** |
| dirent probe, external decls (wasm libc stubs), meta-call dispatch, step | `:25079`, `:1498`, `:2438`, `:2875` | 14 | few | static |
| `.wamo` loader + `wam_object_call_*` | `:26834-28258` | 17 | 0 | static, **kind family** |
| already in templates | `templates/targets/llvm_wam/{state,value}.ll.mustache` | 115 + 15 | 0 | static |

Rendering today: `write_wam_llvm_project_/3` (`:1186-1386`) reads five templates via `read_template_file/2` (`:1886-1899`, cwd then module-relative), `render_template/3` for parts, and `stream_render_mustache/3` (`:1471`) for the flat module; the WASM variant (`:1400-1440`) shares `compile_wam_helpers_to_llvm/2` (`:4340`). The module file is opened `encoding(utf8)` (`:1380-1384`) because templates contain UTF-8 comment glyphs.

### 1.2 `examples/plawk/codegen/llvm/plawk_native_codegen.pl` (24,703 lines)

- 1,744 `format(` calls, ~1,101 of them `format(atom|string(..), '<IR>', ...)`; 1,359 lines contain IR mnemonics; 214 lines carry `''`; 0 `~~`; 36 `c"` globals; 33 shared `llvm_emit_*` helpers all live in `wam_llvm_target.pl` (0 in the codegen).
- 37 clauses of `plawk_program_native_driver_ir/3` (plus 5 multipass) -- the "drivers".
- Shape-dispatched emitter families (clause counts): `plawk_rule_body_print_field` 13 (`:16758-16855`, capability declarations), `plawk_emit_print_expr_for_context` 11+ (`:23734-24275`), `plawk_assoc_print_field_spec` 9 (`:10712-10750`) + `plawk_assoc_print_one_field` 11 (`:10895-11046`), `plawk_end_printf_arg` 11 (`:21179-21270`), `plawk_end_field_print_lines` 9 (`:21749-21821`), `plawk_assoc_end_print_lines` 6.
- Same leaf emitted thrice: `NR` as `plawk_i64_expr_ir(nr, ...)` `:19971`, `plawk_end_nr_print_lines` `:21833`, `plawk_end_printf_arg(special('NR'), ...)` `:21209`.
- Element value printers by kind: `:11003-11024` (f64/i64/str as three format strings), selector `plawk_assoc_kind_value_print_line` `:11030`; `plawk_forin_rule_value_print(i64|str)` `:12463/:12472`; 37 inline `wam_atom_to_string` + printf sequences.
- Element increments: eleven `@wam_assoc_i64_inc` format strings (`:12169, :12515, :12536, :12578, :12605, :12668, :12849, :20858, :20912`...), three `@wam_assoc_f64_add` (`:12665, :12846`...); the bodies differ only in result name, table index, key IR, delta, and the type word.
- Dyncall by result kind: 66 `@wam_object_call_*` call sites over 9 kinds; ~20 `plawk_dyncall_*shim*_ir` predicates, one per kind.

### 1.3 Tests / corpus

219 `tests/test_plawk_*.pl`; ~1,568 literal program call sites (`run(`, `build_only(`, `run_sorted(`...), 552 `plawk_parse_string("...")` literals; 85 test files call `plawk_program_native_driver_ir` directly. One trivial build (`{ c[$1]++ } END {...}`) takes 5.65 s: clang ~3.2 s on the 28,319-line / 280-define `.ll`; module loads 0.3 s + 0.13 s; template render 0.011 s.

## 2. The owner's point: literal cases that collapse into one structural case

### 2.1 Survey of existing `{{match}}` templates (lead finding)

On this feature line only two `.mustache` files use `{{match}}` (`fsharp_wam/program.fs.mustache` 5 cases, `rust_wam/materialisation_setup.rs.mustache` 2) plus the fixture `tests/fixtures/kernel_match_dispatch.hs.mustache`. `gh/main` has 18 `.mustache` files with 79 cases and one `.stache` (`cpp_wam/lowered/head_constant.cpp.stache`, 2 cases). The TODO's "13 files" is stale in both directions.

| family | cases | collapsible? | why |
|---|---|---|---|
| 11 x `rust_wam/*_builtin.rs.mustache` `{{match part}}{{case helpers}}/{{case arms}}` | 22 | **no** | a part selector: two different texts, not two values of one shape |
| 5 x `go_wam/step/*.go.mustache` `{{match instr}}` (8+8+20+8+6) | 50 | **no** | one body per instruction with distinct semantics; the only regular part (`case *Type:`) is already Prolog-owned (`go_step_case_order/1`) |
| `fsharp_wam/program.fs.mustache:185-257`, `rust_wam/materialisation_setup.rs.mustache:6-149` eager/lazy/default | 6 | **no** | three different algorithms |
| `fsharp_wam/program.fs.mustache:375-396` `kernel_kind` benchmark loop | 2 (+default) | **yes, 2 -> 1** | both bodies are `for cat in seedInts do let hits = <Fn> <Args> ... repSolutions + 1`; a `.stache` case `kernel_loop(Fn, Args)` with Prolog supplying `Args` text per kind; a third kernel = a Prolog fact, no new case. The setup block `:341-371` is bidirectional-only with nested `{{#has_csr}}` -- stays a per-kind case |
| `r_wam/run_effective_distance.R.mustache:77-103` emit_mode | 3 | no | distinct |
| `head_constant.cpp.stache` `atom(Esc)` / `value(CppVal)` | 2 | no | already structural; bodies differ (match_reg_atom vs get_reg/is_unbound) |

Honest conclusion: the current `.mustache` corpus is dominated by part-selection and one-body-per-instruction; there is one small 2->1 collapse. The families that justify `.stache` are the ones that have not been written as templates yet -- they are still Prolog strings -- and §2.2-2.3 is where the owner's generalization pays.

### 2.2 Families inside the embedded runtime (would-be literal cases on a naive move)

| family | evidence | shape | result |
|---|---|---|---|
| `@wam_object_call_posarray` vs `_posarray_str` | `:27885-27964` vs `:27972-28051`, 79 lines each, **5 differing lines**: name suffix, tag constant `1`/`0`, two SSA names | `object_call_posarray(Kind, Tag)`; Prolog resolves `i64->1`, `str->0` | 2 -> 1 |
| `@wam_object_call_assoc` vs `_assoc_str` | `:27653-27763` vs `:27774-27876`, 16/110 lines differ | same tag/name substitution (verify the 16 lines before committing) | 2 -> 1 |
| `@wam_object_call_{i64,f64,bytes}` | 47/46/54 lines; i64 vs f64 differ in 5 hunks: return type, extraction tail | shared 40-line shell + per-kind extraction tail | shell as one case, tails as 3 cases, composed by the caller (not one case: the tails differ structurally) |
| six driver `main`s | `:24278-24811`; text-argv vs binary-argv prologues differ only by the `malloc(%rec)` block; the argv/`-`/stdin prologue (~25 lines) is repeated 3x, the fixed-path prologue 3x | `input_source(argv \| path(Global, Len))` x `record_loop(text\|binary\|varlen)` | 6 -> 2 + 3 |
| `@wam_assoc_{i64,f64,str}_print` definitions | `:4841, :5936, :4864` | printf i64 / bitcast + `@wam_print_awk_number` / resolve atom + null check | **not** collapsible -- three different bodies; stay static; only their *call sites* collapse (§2.3) |
| `getline_{file,pipe}` (82 vs 188 lines), `cache_load` vs `_str` (41 vs 103, schema header), `split_into` vs `_re`, `fields_new` vs `_re` | `:8843-9321`, `:6159-6442` | different algorithms | **not** families despite the names; static |

### 2.3 Families in plawk's emitters (the Prolog side)

**A. Element printer / updater by value kind (recommended pilot).** Today: three printer strings (`:11007/:11011/:11024`) behind two deciders (`plawk_assoc_value_print_line` decides f64 via `plawk_f64_table_index/1`; `plawk_assoc_kind_value_print_line` decides str), plus 14 increment strings. Pilot `.stache` (prototyped during planning; renders byte-equal):

```
{{! dialect(pattern_stache, 1) }}
{{match op}}
{{case elem_print(T, Key, Kind)}}  call void @wam_assoc_{{Kind}}_print(%WamAssocI64Table* %plawk_assoc_table_{{T}}, i64 {{Key}})
{{case elem_add(Res, T, Key, Delta, i64)}}  {{Res}} = call i64 @wam_assoc_i64_inc(... i64 {{Key}}, i64 {{Delta}})
{{case elem_add(Res, T, Key, Delta, f64)}}  {{Res}} = call double @wam_assoc_f64_add(... i64 {{Key}}, double {{Delta}})
{{/match}}
```

Eliminates: 3 + 14 format strings -> 3 cases; the eleven inc sites become one predicate call with a ground term. The kind decision (`PLAWK_ARRAY_VALUE_MODEL.md` §4-5: encoding follows semantic kind; mixing rules decline) stays in Prolog, as does the result-name scheme. The runtime naming is already uniform (`wam_assoc_<kind>_print`), which is what makes `{{Kind}}` mid-token legal. A fourth encoding would be zero new cases for print and one for add (the LLVM type word cannot be derived in-template -- no computation, by spec).

**B. Dyncall result kind.** 66 call sites of `@wam_object_call_{i64,f64,bytes,record,assoc,assoc_str,posarray,posarray_str}` and ~20 one-per-kind `plawk_dyncall_*shim*_ir` predicates. The runtime side collapses per §2.2; the emitter side is a candidate `object_call(Kind, RetTy, Args...)` family. **Premature today**: the shim bodies also differ in argument marshalling (`plawk_dyncall_source_ir/8` has 5 shape clauses, `:` compile_src recursion), so first collapse in Prolog per PHILOSOPHY §6.6, then template.

**C. END print-field vocabulary** (PHILOSOPHY §6.4: six kinds x three walkers, ~11 of 18 cells filled). The natural target is one `{{match field}}` over the resolved term with one case per kind rendered by every walker. **Not yet**: the three walkers do not agree on inputs (`StatePlan`, `EndRecord`, `PrintIndex` threading differ: `:21749` vs `:21179` vs `:23734`), and the spec's threaded-index shadowing hazard (RECORD §2) means the term vocabulary must be fixed first. This is §6.6 step 1-2 work; the template is step 3.

**D. Scalar/str print sequence.** 37 inline `wam_atom_to_string` + `llvm_emit_printf_string` sequences and 14 `llvm_emit_printf_i64` sites: a `print_value(PtrOrVal, Kind, Base)` family, mostly already funnelled through the two helpers; low value as a template until A lands.

## 3. Decision rule

1. **Static text, 0-2 holes, whole functions or globals -> `.ll.mustache` library.** Example: everything in §1.1 marked static. Holes use `{{name}}`; libraries are parsed once (§4).
2. **A family over a term shape -> `.stache`.** Test: two or more bodies identical after substituting names/types/tags that a pattern can bind, where a new member should be a new *value*, not a new *case*. Examples: §2.2 posarray/assoc object calls, driver prologue x loop, §2.3-A. Also pick `.stache` when the caller dispatches on an AST term (`var/field/special/assoc/concat`) **after** the Prolog emitters agree (§6.6).
3. **Stays in Prolog:** anything that decides -- key space (interned vs positional vs raw, `ARRAY_VALUE_MODEL` §2), kind inference and mixing rules, plan-time sets, SSA naming schemes and index threading, label flow, lists whose length is decided at emit time (SPEC exclusion "list patterns"), case *priority* (RECORD §1: never order two cases to choose an interpretation; functor-distinct tags only).
4. **Tie-break:** if a family has two members and no third is planned, `.ll.mustache` with two cases is acceptable; the revisit condition is the third member (same spirit as `docs/TODO_STACHE_CONVERTER.md`).

## 4. Library granularity, layout, composition

Layout (owner decision #1-2 below):

```
templates/targets/llvm_wam/
  types|value|state|runtime|module.ll.mustache        (unchanged)
  runtime/builtin_dispatch.ll.mustache                 @execute_builtin (monolith, 1 hole {{dirent_name_ptr}})
  runtime/assoc_table.ll.mustache                      wam_assoc_i64_* (13), f64 wrappers (3), str_print
  runtime/fields.ll.mustache                           wam_fields_*, subsep, split_into(_re), atom_field_*, slice_*
  runtime/strnum.ll.mustache                           looks_numeric, strnum_cmp*, awk_num_*, print_awk_number, strtod
  runtime/regex.ll.mustache                            regex_*, fs_regex_*, rs_regex_*, rt_*
  runtime/streams.ll.mustache                          stream_*, getline_*, environ/argc/argv, cmdline, redirect
  runtime/cache.ll.mustache                            wam_cache_*
  runtime/term_reader.ll.mustache                      wam_parse_*, wam_sb_*, wam_build_*, term_to_sb
  runtime/{arith,copy_term,term_cmp,ssp,backtrack,meta_call,dirent,wasm_libc}.ll.mustache
  runtime/wamo_loader.ll.mustache                      the 17 object-support defines (already optional)
  runtime/object_call.ll.stache                        posarray/assoc kind family (§2.2)
  runtime/driver_main.ll.stache                        input_source x record_loop (§2.2)
examples/plawk/codegen/llvm/templates/
  assoc_elem.ll.stache                                 pilot (§2.3-A)
```

Each `.ll.mustache` library is `{{match fn}}` + one `{{case <symbol>}}` per function (the Go idiom), so a function is addressable. A new small module (template_system.pl is frozen) -- `src/unifyweaver/core/template_library.pl` or private to `wam_llvm_target.pl` -- does: read with `encoding(utf8)`, require a single final LF, scan `{{case NAME}}` into `case(Name, Body)` pairs, validate the set against a Prolog `runtime_fn(Name, Library)` table (duplicate/missing/unknown are errors, as `go_validate_step_case_set/1`), cache per path keyed by sha256 (`gh/main:src/unifyweaver/targets/wam_cpp_templates.pl:207-221` precedent; the CLI is one process per build so load-once and digest-keyed cost the same). Assembly = `atomic_list_concat` of selected bodies in the table's order (today's order, for byte identity).

Pay-per-use: inclusion is a Prolog decision (`emit_wamo_loader(true)` at `:1346-1352` is the existing precedent). Add `runtime_dep(Fn, Dep)` so including `wam_assoc_f64_print` pulls `wam_assoc_i64_{exists,get}` and `wam_print_awk_number`. Default stays "include all" until a measured clang-time win justifies re-baselining (clang is ~3.2 s of 5.6 s; dead functions are already stripped at link).

`.stache` consumers get a thin `plawk_render_stache/3` with the same digest cache plus a preflight that renders every case once at load (`cpp_preflight_stache/0` precedent).

## 5. Verification

- **Runtime moves (PR1-3, 5): byte-identical full `.ll`.** Golden corpus built with `--keep-ll` before/after and `cmp` (handoff `PLAWK_CAMPAIGN_HANDOFF.md:411`); key off the binary and record build status, never `[ -f X.ll ]` (`:422` trap). Include a non-plawk `write_wam_llvm_project/3` module and the WASM variant, since both share `compile_wam_helpers_to_llvm/2`.
- **NUL exception (PR1 only):** spell the 7 `\00` / 3 `\0` evaluated NULs as LLVM `\00` text in the template, compare with `llvm-as | llvm-dis` round-trip for that PR, then byte-compare thereafter. Alternatively generate the template file by `write/1`-ing the evaluated atom (keeps bytes, but commits raw NULs). The `\\0A\\00` and `\n\n` escapes in the atom also change spelling on extraction; generating from the evaluated atom is the only cut-and-paste-proof route -- do that, then post-edit NULs.
- **Per-program driver IR corpus (every PR touching the codegen):** a script under `examples/plawk/scripts/` that extracts every literal program from `tests/test_plawk_*.pl` (552 `plawk_parse_string` literals + the `run(`/`build_only(` sites), dumps `plawk_program_native_driver_ir/3` (and multipass) text + `variant_sha1` per program, run under `LC_ALL=C` with the dump opened `encoding(utf8)` -- a locale stream silently truncated an earlier dump. Diff before/after; declines recorded as `decline`, not skipped.
- **Full sweeps:** all 219 plawk suites, `tests/core/test_pattern_stache*.pl` (76 + 33 pass here), the wam_llvm suites.
- **`.stache` whitespace (SPEC Whitespace):** `{{case P}}` shares the line with the first emitted character; `{{/match}}` directly after the last; the file's trailing LF after `{{/match}}` is rendered (the pilot showed a doubled newline until trimmed) -- adopt the required-single-final-LF rule and strip it. Never normalise body text: bytes inside `c"..."` are data (OFS, printf formats). Threaded indices must be quoted keys (`'Idx'=3`) and case variables must not reuse their spelling (RECORD §2 shadowing).

## 6. Phased PRs

| PR | content | touches | verification | notes |
|---|---|---|---|---|
| 0 | corpus tooling: IR-dump script, golden `.ll` builder, baseline capture | new scripts only | self | unblocks everything; no conflict with arrays series |
| 1 | extract the 4537-21336 atom to `runtime/builtin_dispatch.ll.mustache`, generated from the evaluated atom; `{{dirent_name_ptr}}` hole; library loader + digest cache | `wam_llvm_target.pl` | round-trip diff (NUL exception), then byte | kills the apostrophe class for 63% of defines |
| 2 | split into concern libraries; move the other static atoms (backtrack, arith, copy_term, term_cmp, ssp, meta_call, dirent, wasm stubs, wamo loader); `runtime_fn/2` table + case-set validation | `wam_llvm_target.pl` | byte-identical, native + WASM | one PR per 2-3 libraries if review size demands |
| 3 | `.stache` runtime families: `object_call.ll.stache` (posarray, assoc), `driver_main.ll.stache` (2x3) | `wam_llvm_target.pl` | byte-identical | small win; could be deferred (decision #5) |
| 4 | plawk pilot `assoc_elem.ll.stache` + `plawk_render_stache/3` + preflight; replace the 3 printer and 14 inc/add strings | `plawk_native_codegen.pl` | IR corpus byte-identical | **lands after arrays PR 2** (numeric element writes touch the same lines) or arrays PR 2 adopts it directly |
| 5 | pay-per-use inclusion with `runtime_dep/2` (opt-in flag first) | `wam_llvm_target.pl` | behavioural re-baseline + clang timing | only if the measured win is real |
| 6+ | END print-field `.stache` after §6.6 steps 1-2 (Prolog collapse, capability matrix as data) | codegen | IR corpus | gated by the campaign, not by this plan |

Ordering: PR0-3 and 5 never touch `plawk_native_codegen.pl`, so they run in parallel with the arrays series. PR4 is the only collision; sequence it behind arrays PR 2 and keep it mechanical.

Risks and traps: `''` and `\`-escapes change spelling on extraction (generate, don't paste); `~` in any text that later passes through `format/2` as the *format* argument must be `~~` (the move removes that hazard by never formatting template text); LLVM aggregate types `{ i64, i1 }` are single braces and safe for `{{` scanners; templates hold UTF-8 glyphs -- read with `encoding(utf8)` regardless of locale (the writer already does, `:1380`); `.stache` case values with hyphens or capitals must be quoted (SPEC migration checklist); `read_template_file/2`'s cwd-then-module-relative lookup must survive (the CLI runs from arbitrary cwd); never pin tests on common mnemonics (handoff); do not render a 17k-line library per selected function.

## 7. Open decisions for the owner

1. Library granularity: ~15 concern files with `{{case}}`-addressable functions (recommended) vs one file per function (~180 files) vs keep `@execute_builtin` monolithic forever.
2. Home for plawk-specific templates: `examples/plawk/codegen/llvm/templates/` (recommended: next to its consumer) vs `templates/targets/llvm_wam/plawk/`.
3. NUL handling in PR1: LLVM `\00` text + one round-trip-verified re-baseline (recommended) vs raw NUL bytes for strict byte identity.
4. Engine for static libraries: a small new `template_library.pl` (template_system.pl is frozen) vs private predicates in `wam_llvm_target.pl`.
5. Ratify `.stache` inside the core LLVM target now (PR3: two 2->1 collapses plus the 2x3 main grid) or keep those as `.ll.mustache` until a third kind appears, and let plawk's PR4 be the first `.stache` consumer on this line.
6. Pay-per-use: pursue (changes IR, needs re-baseline, saves clang time only) or not.
7. Golden-diff policy: strict `cmp` everywhere except the declared NUL re-baseline, vs `llvm-as | llvm-dis` normalisation as the standing oracle.
8. Whether to fix the stale "13 files" count in `docs/TODO_STACHE_CONVERTER.md` and record §2.1's finding (no literal families in the corpus) as that document's measured answer.

## 8. Decisions (owner, 2026-10-03)

The owner approved the recommendations; they are binding for the phased PRs of §6.

| # | decision |
|---|---|
| 1 | **~15 concern libraries** of `{{case}}`-addressable functions (not one file per function, not the monolith). |
| 2 | plawk-specific templates live in **`examples/plawk/codegen/llvm/templates/`**, next to their consumer. |
| 3 | NULs are written as LLVM **`\00` text**; PR 1 carries one round-trip-verified (`llvm-as \| llvm-dis`) re-baseline. Raw NUL bytes in a text template are a hazard of their own. |
| 4 | A small new **`template_library.pl`** (parse once per process, cached; case lookup) -- `template_system.pl` stays frozen. |
| 5 | **plawk's pilot (PR 4) is the first `.stache` consumer on this line**; the core LLVM target's `.stache` families (PR 3) wait until the dialect has proven itself here. PR 3's moves land as `.ll.mustache` in the meantime. |
| 6 | **Pay-per-use is deferred** (it changes IR and saves only clang time). |
| 7 | Golden diffs are a **strict byte compare**, except the one declared `\00` re-baseline of PR 1. |
| 8 | `docs/TODO_STACHE_CONVERTER.md` records §2.1's re-measurement. |

Guiding principle, from the owner: templates should be as general and composable as practically
possible (often meaning smaller templates), and **many literal cases that share a shape should
collapse into one structural `.stache` case** -- §2's survey is where that pays: in the code that is
still embedded in Prolog, not in today's `.mustache` corpus.

## 9. Status

- **PR 1 (merged):** the monolith atom became `runtime/builtin_dispatch.ll.mustache`.
- **PR 2a:** that file is split into eight case libraries -- `assoc_table`, `fields`, `strnum`,
  `regex`, `streams`, `cache`, `term_reader`, `builtin_dispatch` (now `@execute_builtin` alone).
  A chunk is a define with the comments and globals before it (118 chunks: 117 defines plus the
  stream-handle globals). `src/unifyweaver/targets/wam_llvm_runtime_libs.pl` holds the
  `wam_llvm_runtime_chunk(Name, Library)` table: which chunks exist, where each lives, and the
  assembly order (the monolith's, so the module is byte-identical). Each library's case set is
  validated against the table once per process; `template_library_cases/2` parses a library
  once and caches it. Pinned by `tests/test_template_library.pl`.
- **PR 2b (next):** the other static atoms (backtrack, arith, copy_term, term_cmp, ssp,
  meta_call, dirent, wasm stubs, wamo loader).
