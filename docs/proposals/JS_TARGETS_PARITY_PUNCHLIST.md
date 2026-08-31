<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# JS-family targets — parity punchlist

**Living document.** Tracks the gap between the JavaScript-family targets
(TypeScript, annotated JS, vanilla JS, ClojureScript, and the `wam_javascript`
hybrid) and the mature reference targets (python/rust/go for pattern; rust/haskell/
lua/cpp for WAM). Built to be worked down item by item and to double as a status board.

**Last updated:** 2026-08-30 (some OPEN rows are pending confirmation from two
in-flight parity analyses — see §6).

## Legend

| Field | Values |
|---|---|
| **Status** | ✅ done · 🔄 in-progress · ⬜ open · ❓ suspected (needs analysis to confirm) |
| **Owner** | `grok` (JS WAM runtime/emitter lane) · `opus` (disjoint pattern work) · `main` (INT-0 wiring/merges) · `—` (unassigned) |
| **Size** | S (hours) · M (day) · L (multi-day / research) |
| **Collision** | which hot files it edits — `runtime.js.mustache`/emitter are actively edited by Grok's indexing work, so ⚠️ marks WAM items that must serialize behind it |

---

## 1. Completed this session ✅ (done ledger)

| ID | Target | Item | Merged |
|---|---|---|---|
| D1 | typescript | Bindings 154 → 179 (Map/Set, Number formatting) | PR #4177 |
| D2 | typescript | Fixed 3 codegen bugs (empty fact array, leaked `generate_match_expr`, `transitive_closure` fallthrough) | #4177 |
| D3 | annotated_js | New pattern target (inherits TS, TS→JSDoc rewrite; `tsc --checkJs` clean) | #4177 |
| D4 | vanilla_js | New pattern target (inherits TS, centralized type-strip; node-verified) | #4177 |
| D5 | clojurescript | `runtime(scittle\|nbb\|bb)` variants; 67-binding `clojurescript_bindings.pl` | #4177 |
| D6 | clojurescript / clojure base | Numeric recursion codegen fix (base-level; also improves JVM Clojure) | #4177 |
| D7 | (infra) | PAR-1 pattern-target parity harness | #4177 |
| D8 | wam_javascript | Interpreter tier + six WAM conventions + 48/48 conformance arm | #4177 |
| D9 | wam_javascript | Builtins: findall/functor/arg/=../copy_term/\+/call/aggregate_all | #4177 |
| D10 | wam_javascript | Full ISO bagof/3 + setof/3 (witness grouping, `^`, empty-fail, std order) | PR #4178 |
| D11 | (infra) | Registry wiring + BINDING_MATRIX rows for the new targets | #4177/#4178 |
| D12 | wam_javascript | First-argument indexing (`switch_on_constant/structure/term`; ground dispatch leaves no CP, var-first-arg falls back to try/retry/trust) | Grok `grok/wamjs-indexing` |
| D13 | vanilla_js | Registered all 7 advanced-pattern hooks (delegate to TS + type-strip); `target(vanilla_js)` now dispatches (was: no clause → fail). G-P10. | (this branch) |
| D14 | typescript (→AJS/VJS) | Real recursion (G-P1): tree/multicall/direct/tail hooks now derive base cases + offsets + aggregation from the actual clause (was: hardcoded fib for any predicate). Structural recursion (G-P2): native list lowering for member/append/reverse/length, node-verified vs SWI. | PR #4182 |
| D15 | wam_javascript | Switch-indexing cross-target matrix row added (G-W5): full coverage of all 5 tracked columns (const_ft/const_a2/const_a2_ft/term_a2/struct_a2), residuals documented. | (this branch) |
| D16 | typescript (→AJS/VJS) | Component emission (G-P4): `compile_module` now calls `compile_component` (was collected-but-dropped); revived orphaned `custom_chart` (+ fixed its unevaluated-`format` bug). Component-free modules unchanged. | (this branch) |
| D17 | clojurescript | Wired the `clojurescript` binding key (G-P11): `resolve_binding([clojurescript, clojure], …)` fallback-with-override; loaded both catalogues; call-head rewrite after interop (no double-transform); bb skips it. `parse_double→js/parseFloat` proof, nbb-verified, JVM regression green. Bindings 67→68. | (this branch) |
| D18 | typescript (→AJS/VJS) | Aggregate compilation (G-P3): `aggregate_all` (count/sum/max/min/bag/set) + `findall` as goals (was 0 refs → 50); node-verified vs SWI. Follow-up: bagof/setof at pattern level + non-extensional inner goals. | PR #4184 |
| D19 | clojure + clojurescript | Component pattern wired into the clojure base (G-P5): loads `component_registry` + revived orphaned `custom_clojure`, emits declared components (both JVM Clojure and CLJS via interop rewrite). CLJS `compile_module/3` added (G-P6). nbb-verified (module + component); JVM regression green. | PR #4185 |
| D20 | wam_javascript | **Tier-2 lowered emitter (G-W1, Grok):** new `wam_javascript_lowered_emitter.pl` + `functions`/`mixed` emit modes. Lowers single-clause det bodies, T4 inline, T5 first-arg dispatch, T6 hash dispatch (≥8 keys), ITE/negation/once; falls back to interpreter for aggregates/cuts/unsupported (never wrong code). Interpreter mode unchanged → conformance still 48/48. JS is no longer one of the 4 WAM targets without a lowered emitter. | PR #4186 |
| D21 | typescript (→AJS/VJS) | Streaming/generator emit mode (G-P8): `mode(generator)`/`mode(pipeline)` (+ clojure-style aliases) via Node built-in `readline` (no deps); incremental stdin→stdout. Batch mode unchanged; falls back to batch for non-qualifying shapes. node-verified across TS/AJS/VJS vs SWI. | PR #4188 |
| D22 | annotated_js, vanilla_js | Rewriter hardening (G-P14): vanilla strips union types (incl. `\| null`); annotated keeps `import * as`/named-alias imports intact + strips inline arrow-param annotations; union return → `@returns` JSDoc. node + `tsc --checkJs` verified; base TS suite unaffected. | PR #4189 |
| D23 | wam_javascript | ISO/library builtin breadth (G-W3, Grok): sort family (sort/2,4, msort, keysort, compare, predsort), list lib (append/reverse/nth0/nth1/last/sum_list/max_list/min_list/list_to_set/select/include/exclude), atom/string (atom_concat/length/chars/codes, char_code, sub_atom, up/downcase, atom_string/split_string/string_concat), format/2,3 (~w~a~d~p~q~n~s~t~~), tab, writeln, assoc (list-based). node-verified vs SWI; conformance 48/48. Follow-up: term_variables/numbervars/=@=, ~f/~r, real string tag, AVL assoc. | PR #4190 |
| D24 | typescript + clojure (→AJS/VJS/CLJS) | Guard-goal codegen (G-P7): added negation (`\+`/`not`) and type-check preds (integer/float/number/atom/is_list/compound/var/nonvar/ground) to `ts_guard_condition/3` + `clojure_guard_condition/3` (+ `member/2` under negation). Was: classified as guard upstream then render-failed. node+nbb verified vs SWI; 5 suites green (TS 57/AJS 17/VJS 22/CLJS 26/JVM 12). Regex `match` = follow-up. | PR #4192 |
| D25 | wam_javascript | Richer runtime term parser (G-W2, Grok): Pratt reader for ints/floats/atoms (bare+quoted)/vars/lists (`[H\|T]`)/compounds, ISO default op table (`1+2`→`+(1,2)`); CLI pads missing args with unbound vars; `read_term_from_atom`/`atom_to_term`/`term_to_atom` builtins. Registered `native(parse_term)` in `wam_runtime_parser_capability.pl` (INT-0). node-vs-SWI CLI (list→S=6, compound→X=a, float 3.14>3.0); conformance 48/48. Follow-up: user `op/3`, postfix ops. | PR #4193 |
| D26 | typescript (→AJS/VJS) | Data-source consumer (G-P9, Option A): TS now consumes JSON + CSV sources via `_typescript` templates (`json_source.pl`/`csv_source.pl`) + new `typescript_source_compiler.pl` wrapper + one routing clause on `is_dynamic_source/1`. Self-contained Node (fs, no npm). node output matches jq/awk; existing bash/PowerShell-pure paths untouched. Follow-up: non-comma CSV delimiters, columns/schema projection. | PR #4194 |
| D27 | wam_javascript | External fact sources (G-W4, Grok): `javascript_wam_fact_sources([source(P/2, file(Path))])` (alias `js_fact_sources/1`) emits `CallFactStream` instead of inline WAM; Node runtime reads TSV/CSV/JSONL via `fs` (no npm), first-arg indexed when A1 bound; cells parsed via `parse_term`. Inline-facts unchanged. node-vs-SWI on shared triples; conformance 48/48. LMDB/CSR out of scope (matches Lua peer). | PR #4196 |
| D28 | typescript + clojure (→AJS/VJS/CLJS) | Regex `match/2,3` guard codegen (G-P7 completion): `match(Subject,Pattern)` → TS `new RegExp(pat).test(x)`, Clojure `(re-find (re-pattern pat) x)`; `match/3` regex-type arg advisory (uses host engine). String-input typing for match-on-arg1; `\+ match` composes. node/nbb-vs-SWI (pcre oracle). Optional INT-0 follow-up (in patch): a 2-line `is_guard_goal` addition to make bare positive `match` a batch guard. | PR #4198 |
| D29 | wam_javascript | Term-meta builtins (G-W3 follow-up, Grok): `term_variables/2` (distinct unbound, first-occurrence order), `numbervars/3` (`'$VAR'(N)` from Start, End=Start+count), `=@=/2`/`\=@=/2` (variant equality). Reuses `copy_term`'s deref/seen-struct walk. node-vs-SWI matched; conformance 48/48. Residual: `write` prints `'$VAR'(N)` literally (no A/B letter-render), no `numbervars/4` options/attvars. | PR #4199 |
| D30 | typescript + clojure (→AJS/VJS/CLJS) | Uniqueness/order constraints (G-P-dedup): `compile_facts` now consults `constraint_analyzer:get_constraints/2` — `unique(true)` emits order-preserving/sorted dedup (TS `Set`+`JSON.stringify`; Clojure `distinct`/`sort`), `unique(false)` raw; mirrors rust/go. Covers the facts/query-helper shape (inherited by AJS/VJS/CLJS); recursion/aggregate/streaming shapes deferred (no single materialized site). node/nbb-vs-SWI `setof`; 5 suites green. | PR #4201 |
| D31 | typescript (→AJS/VJS) | Data-source polish (G-P9 follow-up): non-comma CSV delimiters (tab/pipe/… via `String.split` string-literal, `js_escape_delimiter/2` — no regex pitfalls; bash/PS `delimiter` unchanged) + JSON `columns()` projection with reorder (dotted/`jsonpath` paths). node-vs-`awk` verified. Deferred (blocked-upstream, documented): CSV subset/reorder projection + JSON `schema` (need `sources.pl`/`compile_source` changes that'd alter bash/PS output). | PR #4202 |
| D32 | wam_javascript | User-defined operators `op/3` (G-W2 follow-up, Grok): live Pratt operator table = ISO defaults + user declarations. Infix (xfx/xfy/yfx), prefix (fx/fy), postfix (xf/yf); priority 0 removes; atom or list of names; ISO coexistence rules. Compile-time `:- op/3` threaded via `javascript_wam_ops/1` (alias `js_op_decls/1`) → `install_declared_ops` at startup; runtime `op/3` builtin. node-vs-SWI (custom infix/prefix/postfix + emit-time decls); conformance 48/48. Residual: no `current_op/3`; process-global (like SWI). | (this branch) |
| D33 | typescript (→AJS/VJS) | CSV subset/reorder projection (G-P9 residual, unblocks D31): new additive `project_columns([Name,…])` option — distinct from arity-defining `columns` — resolves names→0-based indices against the file's detected header at compile time (`resolve_projection/5`) and selects/reorders exactly those fields in the `_typescript` templates (`[fields[i],fields[j]].join(":")`; single-col arity-1 too). Pass-through defaults keep no-projection TS **byte-identical**; bash/PS never reference the vars → unchanged. Unknown name / arity mismatch fails with a clear error. node-vs-expected verified; 15 TS-source tests green (4 new). | (this branch) |
| D34 | wam_javascript | Real string type tag (G-W3 residual, Grok): distinct `V.String` term tag in the runtime term model — `unify`/`==`/`copy_term`/`compare`/`sort`/`write` all handle it; a string unifies only with an equal string, not an atom of the same text. String-producing builtins now box strings (`atom_string/2`, `string_concat/3`, `string_chars/2`, `string_to_atom/2` A1, `number_string/2`, `split_string/4`); `atom_concat`/`atom_chars` stay atoms; `string/1` true only for the tag, `atom/1` false. Standard order matches **SWI 9.0.4** (`Var<Number<String<Atom<Compound`; `"foo" @< foo`). node-vs-SWI probes + `probe_string_tag` green; conformance still 48/48; GitHub CI 16/16. Residuals: compiled `"foo"` literals collapse to atom (shared WAM constant tokens — needs `wam_target.pl`); `writeq`/`~q` nested-string quoting; fact-source cells stay atoms; no `string_length/2`. | (this branch) |
| D35 | wam_javascript | String polish (D34 residuals, Grok): `string_length/2` (string/atom/number → char length) + `writeq/1` and a recursive quoted writer reused by `~q` — strings render `"…"`, atoms single-quoted only when needed. Quoting subset: unquoted `[]`/`!`/`;`/`{}`/`^[a-z][A-Za-z0-9_]*$`/ISO graphic atoms (except lone `.`), quoted otherwise; escapes `\\`/`\'`. `write/1` unchanged. node-vs-SWI + `probe_string_polish` green; builtins/lowered/fact-sources green; conformance 48/48. Residuals: compiled `"foo"` literals still collapse (out of scope, `wam_target.pl`); fact-source cells stay atoms; no `\n`/`\t` escapes in `writeq`; brace terms `{a}` not special-cased. | (this branch) |
| D36 | typescript (→AJS/VJS) | JSON `schema` mode for the Node consumer (G-P9 "Option-B" residual, Opus): the TS/Node JSON consumer now honors a declared `schema(...)` — reads `dynamic_source_metadata/2`'s `schema_fields` (per-field `name`/`column_type`/`path`) and emits a **typed record object** per JSON record (`JSON.stringify` of `{field:value,…}` in declared order, coerced string→String / int→trunc Number / float→Number / boolean→bool / json→passthrough / null→null), the parity the C# target already had. New `_typescript_schema` templates + `json_schema_projection_js/2`; wrapper routes via non-empty `schema_fields`, else falls through to flat-columns unchanged. Only non-additive edit = one render-var line; bash/PS byte-identical; flat-columns TS untouched (2 control tests). New isolated suite `test_typescript_json_schema` 6/6 green (2 node-exec); `test_typescript_source`/`json_source_validation`/`csv_source` green. | (this branch) |
| D37 | wam_javascript (+ shared WAM) | Compiled string literals → JS string tag (D34/D35 residual, Grok; **2 shared files**): `wam_target:quote_wam_constant/2` now spells a Prolog `string` value as a double-quoted `"…"` token (atoms/numbers unchanged); `wam_text_parser:wam_classify_constant_token/2` strips the quotes and **still returns `atom(Name)`** (no new `string(_)` Class — every other runtime interns the atom exactly as before) plus a new exported `wam_constant_token_is_string/1` signal. JS-only consumers (`wam_javascript_target:constant_to_js_term/2`, lowered emitter T5/T6) build a `V.String` when the signal is true, so a compiled `lit("hi")`/`X = "hi"` yields a real string. `probe_string_literal` green; **full-fleet conformance ALL_PASS** (compiled+ran python/rust/c/cpp/go through the changed shared code, no regression) + JS arm 48/48. Residual: fact-source JSON/TSV cells still intern as atoms. | (this branch) |
| D38 | wam_javascript | Runtime profiling (GP-PROF, Grok): opt-in instrumentation, off by default (`Runtime._prof=null`; unprofiled loop byte-identical to pre-GP-PROF). `UW_PROFILE=1` → compact stderr table, `UW_PROFILE=json` → stable-schema JSON on stderr, `Runtime.profile(...)` programmatic. Per-predicate (exclusive): calls/instr/ns/cps/max_cp_depth/lowered; global: instructions/unify_calls/trail_pushes/heap_cells/backtracks/trail_undos/wall_ns. Lowered tier = call counts only (marked `*`). stdout byte-identical on AND off (report → stderr). Overhead: off ≈ noise (±1%), on ≈ 6–10%. 3 probes + lowered call-count check green; suites + JS-arm conformance green. | (this branch) |

---

## 2. In progress 🔄

| ID | Target | Item | Owner | Branch |
|---|---|---|---|---|
| A1 ✅ | (SWI spec) | **peerhailer argparser in Prolog — DONE** (`examples/cli_args/`): `cli_args.pl` mirrors the oracle exactly — schemas as `Name-schema(Options,Positionals)\|group(Actions)` data, strict + verbatim lenient + leading globals, `ok(Positional,Flags)`/`err(Message)` tagged results (no throw; the JS edge maps `err→CliError`), flags as insertion-ordered KV pairs. Even the JS prototype-chain lookup quirks (`--toString` as a truthy global, `--__proto__` no-op assignment) are modelled. **17/17 plunit** + **5067-line seeded differential vs node oracle: 0 divergences, 0 message mismatches**. Pure/det/tail-recursive, no cuts/throws/pcre — transpilation-ready. A2/A3 re-point `run_differential.sh`'s Prolog leg at the transpiled build (same seed, same bar). | opus | merged |

### Phase 2 — the argparser maturity demonstration (A-series)

Goal: transpile a REAL program — peerhailer's CLI arg parser — through the JS-family
targets, as the maturity proof. Oracle = `examples/cli_args/oracle/cliArgs.js` +
its 17-test corpus `oracle/cliArgs.test.mjs` (copied from peerhailer @ `08ad35e`,
verified 17/17 green under `node --test`). A usage error is a thrown `CliError` at
the JS boundary; the Prolog spec returns `error(Message)` and the transpile maps it
to `throw new CliError(Message)` at the edge, so the same corpus drives both.

| ID | Step | Depends | Owner |
|---|---|---|---|
| A1 | Prolog reference implementation + plunit corpus port + differential harness vs node oracle | — | opus (running) |
| A2 | Compile via **wam_javascript**; emitted module passes `cliArgs.test.mjs` with only the import swapped (thin `parseArgs`/`CliError` shim over the WAM entry) | A1 | grok |
| A3 | Compile via **typescript → AJS/VJS** pattern targets; every shape the compiler can't lower becomes a NAMED new gap item here (expected: this surfaces the next real punchlist) | A1 | opus |
| A4 | Differential fuzz (seeded, len 2–7 alphabet lines): transpiled JS vs oracle agree on `{positional, flags}` / both-throw with matching messages; any deliberate delta named in a contract table | A2/A3 | main |

### Phase 2 — further gap items (user-named)

| ID | Target(s) | Gap | Status | Notes |
|---|---|---|---|---|
| GP-PROF | wam_javascript | Runtime profiling/instrumentation | ✅ D38 | landed (Grok) — baseline for GP-PAR |
| GP-PAR | wam_javascript (+pattern) | Parallelism — WAM-level (haskell/elixir/rust precedent) and/or pattern-level | ⬜ | deferred earlier as G-W6; revisit after profiling gives a baseline |
| GP-LMDB | wam_javascript | LMDB-backed fact sources | ⬜ | needs a native/npm dep (`lmdb`), conflicting with the no-deps ethos peer targets follow (Lua skipped it) — needs a decision: allow an optional dep tier, or an mmap-free file index instead |

**Remaining tail** (all low-priority polish / out-of-scope — no active work):
- **Pattern:** bagof/setof-with-witness-grouping at pattern level (G-P3 follow-up — the minimal slice is marginal over aggregate_all(bag/set)); (JSON `schema` mode now landed for the Node consumer, D36; CSV subset projection via `project_columns`, D33); bare positive `match` batch-guard (**not** a clean 2-liner after all — adding `match` to the shared `clause_body_analysis:is_guard_goal/2` regresses two targets: `python_guard_condition/3` has no `match` clause and no catch-all so the guard `maplist` fails (python renders `match` via the output-goal path, not as a guard), and wam's `is_builtin_pred/2` derives builtin-ness from `is_guard_goal` so `match` would emit `builtin_call match/2` that no WAM runtime implements. Only clojure+ts have `match` guard renderers. Proper fix = add `match` guard renderers to python + gate wam's builtin path, i.e. a cross-target change, not polish. D28's deferral stands.); test depth (G-P13).
- **WAM (lane exhausted for term model/builtins):** AVL assoc (list-based today); LMDB/CSR fact sources (out of scope, matches Lua peer); G-W6 parallelism (deferred); fact-source JSON/TSV cells still intern as atoms (D37 residual). Cosmetic only: `\n`/`\t` escapes in `writeq` and brace-term `{a}` special-casing. `op/3`/postfix (D32), real string tag (D34), `string_length/2`+`writeq`/`~q` quoting (D35), and compiled string literals → string tag (D37, the shared-file `wam_target.pl` item) now all landed.

---

## 3. Open gaps — PATTERN targets ⬜ — CONFIRMED by analysis A1

Two extension mechanisms apply: **bindings** (`declare_binding/6`) and the **component
pattern** (`component_registry` + per-target `custom_*.pl`). A1 corrected two of my
suspicions (see §3b N/A): bindings are **not** a gap (JS leads), and TS/AJS register
**all 7** recursion hooks (more than python's 4) — the gap is hook *quality*, not count.

| ID | Target(s) | Gap | Status | Copy-from | Size | Owner |
|---|---|---|---|---|---|---|
| G-P1 | typescript (→ AJS/VJS) | **"Supported" recursion hooks are canned Fibonacci** — `compile_tree_pattern`/`compile_multicall_pattern`/`compile_direct_multicall_pattern(typescript,…)` all emit `const result = pred(n-1)+pred(n-2)` regardless of the real clause (`typescript_target.pl:~1402,1435,1468`; tail arity-2 stubs `return items.length` `:1228`). | ⬜ | extend TS's own partial `native_ts_clause_body` (`:736`); generality from python/go native lowering | L | opus |
| G-P2 | all JS pattern targets | **No structural recursion** (member/append/reverse) — none compile it (`docs/JS_PATTERN_CONFORMANCE.md`); python/rust/go do via native clause lowering. | ⬜ | python_target / go_target native lowering | L | opus |
| G-P3 | all JS pattern targets | **No aggregate compilation** — 0 `aggregate_all/findall/bagof/setof` goal refs in TS/CLJS vs go 36, python 10, rust 10. Core datalog surface. | ⬜ | `go_target.pl` (richest) | L | opus |
| G-P4 | typescript (→ AJS/VJS) | **Components collected but never emitted** — `compile_module/3` (`:414-436`) never calls `component_registry:compile_component` (python `:198`/rust `:378`/go `:14951` do). Also `custom_chart.pl` is orphaned (self-registers at `:219` but no module `use_module`s it → dead). | ⬜ | python_target.pl:198 emit loop | M | opus |
| G-P5 | clojure + clojurescript | **No component support at all** — `clojure_target.pl` has zero `component_registry` refs; `custom_clojure.pl:79` self-registers but is never loaded; no CLJS component. | ⬜ | mirror python_target.pl:96-97,198 | M | opus |
| G-P6 | clojurescript | **DONE (D19).** CLJS `compile_module/3` added. (Structural recursion still tracked under G-P2.) | ✅ | — | — | done |
| G-P7 | typescript (→AJS/VJS) + clojure/CLJS | **Minimal slice DONE (this branch).** Added negation (`\+`/`not`) + type-check (integer/atom/number/is_list/compound/var/nonvar/ground/float, + inert `atomic`) + `member` guard clauses to `ts_guard_condition/3` and `clojure_guard_condition/3` (AJS/VJS inherit TS; CLJS inherits clojure). Negation-of-non-guard-goal handles `member` specially and otherwise fails cleanly (no wrong code). node + nbb verified vs SWI; all 5 target suites green (TS 57, AJS 17, VJS 22, CLJS 26, clj-native 12). See `integration_patches/GP7_INTEGRATION_PATCH.md`. **Still open (separate follow-ups):** regex `match/2,3` (python `translate_match/6` model); `constraint_analyzer.pl` unique/unordered dedup (rust/go call `get_constraints/2`; TS/CLJS don't). | 🔄 (slice done) | python_target guard/negation; clause_body_analysis | S–M | opus |
| G-P8 | typescript (→ AJS/VJS) | **No streaming/pipeline/generator mode** — 0 refs vs python 432/go 390/rust 325. CLJS inherits clojure's `generator_mode`/`pipeline_mode` best-effort. | ⬜ | clojure_target (closest paradigm) or python | M | opus |
| G-P9 | typescript (→AJS/VJS) | **Data-source consumer (RE-SCOPED by recon).** python/rust/go do NOT consume csv/json sources (the old `semantic_source.pl` dispatch claim was wrong). Only PowerShell-pure consumes them (via `_suffix` templates + its own path; `powershell_compiler.pl:225-239`), and C# via a query-plan/metadata model. **Minimal slice = Option A: mirror PowerShell-pure** — add `_typescript` templates to `csv_source.pl`/`json_source.pl`, a thin wrapper module, + one routing clause in `compile_predicate_to_typescript/3` (before the fallback) checking `dynamic_source_compiler:is_dynamic_source/1`. Independent path; no native-clause/guard changes. Option B (C#-style relation model) = separate L task, not "catching up" since python/rust lack it too. **DONE (this branch, Option A):** added `json_file_source_typescript`/`json_stdin_source_typescript` + `csv_source_unary_typescript`/`csv_source_binary_plus_typescript` templates (additive only), new wrapper `targets/typescript_source_compiler.pl` (`compile_to_typescript_source/3`), and one routing clause in `compile_predicate_to_typescript/3` before the fallback. Emits self-contained Node (`fs` + `JSON.parse`, no npm/jq). JSON + CSV both land; node-verified vs jq/awk; new suite `tests/core/test_typescript_source.pl` green; TS/AJS/VJS suites unaffected. CSV non-comma delimiters + `columns`/`schema` projection noted as follow-ups. See `integration_patches/GP9_INTEGRATION_PATCH.md`. | ✅ (this branch) | powershell-pure (`powershell_compiler.pl`) | S–M | opus |
| G-P10 | vanilla_js | **Inheritance asymmetry** — registers **0** `compile_*_pattern` clauses (annotated_js registered all 7); a caller driving the advanced compiler with `target(vanilla_js)` gets no clause and fails. **Verified.** | ⬜ | copy annotated_js:886-911 (delegate to TS then `vanilla_js_type_strip/2`) | S | opus |
| G-P11 | clojurescript | **FIXED.** Wired the `clojurescript` binding key via `resolve_binding([clojurescript, clojure], …)` — CLJS bindings override, clojure is fallback. Loaded both catalogues (`ensure_cljs_bindings/0`), added a call-head rewrite ordered after the interop rewrite (disjoint tokens, no double-transform); bb runtime skips it. Proven with `parse_double/2 → js/parseFloat`, nbb-verified, JVM Clojure regression green. Bindings 67→68. | ✅ D17 | — | — | done |
| G-P12 | annotated_js, vanilla_js | **PAR-1 arm activation** — arms exist but skip until the harness loads the targets; needs a real tsc/node run (fine in an agent env). | ⬜ | — | S | opus |
| G-P13 | JS pattern targets | **Thin test depth** — 8/3/3/5 referencing files (TS/AJS/VJS/CLJS) vs python 204/rust 155/go 78; no aggregate/negation/service/data-source tests. Grow as G-P1..G-P9 land. | ⬜ | those suites | M (ongoing) | opus |
| G-P14 | annotated_js, vanilla_js | **FIXED (D22).** Hardened both rewriters: vanilla strips union types incl. `\| null`/`\| undefined`; annotated leaves `import * as`/`{a as b}` intact (scanning `as`-cast detector) and strips inline arrow-param annotations; union return → `@returns {A \| B}`. node + `tsc --checkJs` verified. (Left, pre-existing/out-of-scope: `const x = (fact as any).foo;` misdetected as a signature by the pre-existing detector.) | ✅ D22 | — | — | done |

### 3b. Pattern-side N/A / not-a-gap (per A1)
- **Bindings** — JS family LEADS: TS **179** (most in the repo), CLJS 67, vs python 106, go 161, rust 29. Not a gap. (Earlier G-P8 "bindings depth" dropped.)
- **Recursion-hook count** — TS/AJS register all 7 multifile hooks (python registers 4). Coverage is fine; quality is the issue (G-P1/G-P2).
- **Unix-socket / TCP service generation** — mostly N/A for a browser/Node JS target; TS's Express HTTP server/router/client (`typescript_target.pl:517-669`) is the right analogue and is present.
- **AJS/VJS lacking own bindings/components** — by design they inherit TS; not a gap except the concrete bugs G-P4 (emission) and G-P10 (hooks).

---

### Fleet denominators (census — how common each capability is, out of 18 WAM target modules)
- **Lowered emitter:** 14/18 have one; the only 4 without are `wam_c`, `wam_ilasm`, `wam_jvm`, **`wam_javascript`** → G-W1 is the fleet norm, not exotic.
- **External fact sources (LMDB/CSR/TSV):** 14/18; the 4 without are `wam_ilasm`, `wam_jvm`, `wam_kotlin`, **`wam_javascript`** (JS has an empty `fact_sources:{}` scaffold only) → G-W4.
- **Runtime-parser capability:** 7/18 registered (`wam_r/cpp/python/fsharp/haskell/rust/go`); JS not among them → G-W2.
- **Dedicated kernel templates:** only 2/18 (Haskell, F#, 8 each); 8 more drive kernels via the shared `recursive_kernel_detection.pl`; **JS does neither** → kernels are the deferred tail (see §5b), the reference peer `wam_lua` also has neither.
- **WAM bindings file:** 7/18 have none at all; among those that use `declare_binding`, counts run 19–49. JS has a 39-entry catalogue in a different scheme → not a gap.
- **Conformance arm:** JS is registered (16 arms total); `wam_lua` and `wam_clojure` are NOT.

## 4. Open gaps — WAM (`wam_javascript`) ⬜ — CONFIRMED by analyses A2 + census

Mature WAM targets have up to three tiers (interpreter → lowered emitter → FFI kernels);
`wam_javascript` is interpreter-tier only. Reference peer is `wam_lua` (interpreter-only,
dynamically typed); `wam_rust`/`wam_haskell` are the full-stack exemplars. Evidence
cited is from A2 (file:line in the analysis).

| ID | Gap | Status | Copy-from | Size | Collision | Owner |
|---|---|---|---|---|---|---|
| G-W1 | **Tier-2 lowered emitter + `functions`/`mixed` emit modes** — every peer incl. `wam_lua` has `wam_*_lowered_emitter.pl`; JS is interpreter-only (`wam_javascript_target.pl:48-51` throws on other modes). Subsumes the old "emit modes" row. | ⬜ | `wam_lua_lowered_emitter.pl` (closest) | L | ⚠️ emitter + runtime | grok — **after P1 indexing lands** |
| G-W2 | **Runtime term parser** — CLI parses **ints + atoms only** (`runtime.js.mustache:1862-1866`) — no floats/lists/compounds; JS absent from `wam_runtime_parser_capability.pl`. Blocks structured-arg queries + `read_term`. | ⬜ | register `runtime_parser(compiled)` like go/haskell/rust; or extend `parse_cli_atom_or_int` | S–M | low (distinct runtime region) — **parallel-safe** | grok or opus |
| G-W3 | **ISO/library builtin breadth** — missing `term_variables/2`, `numbervars/3`, `=@=/2`; sort family (`sort/2,4`, `msort`, `keysort`, `predsort`); structural list lib (`append`, `reverse`, `nth0/1`, `last`, `sum_list/max_list/min_list`, `select`); atom/string ops (`atom_concat`, `sub_atom`, `atom_length`, `atom_chars/codes`, `char_code`, `split_string`); `format/2,3`, `tab/1`; assoc lib. (NB: JS is *ahead* of Lua on aggregates/findall — not a peer gap there.) | ⬜ | `rust_wam_bindings.pl` for breadth | M | bindings file disjoint; runtime `builtin_*` shares file (low-mod) | grok |
| G-W4 | **External fact-source / data tier** — no LMDB/CSR/TSV/JSON consumption; JS has no fact-source options. | ⬜ | `wam_lua_target.pl:475,545` (lightweight `lua_fact_sources`); rust/haskell for full LMDB+CSR | L | new templates disjoint; some `_target.pl` options | grok — **premature until G-W1** |
| G-W5 | **Switch-indexing conformance row** — runtime wires `SwitchOn*` (`runtime.js.mustache:70-73,1146-1149`) but JS is absent from `WAM_SWITCH_INDEXING_CROSS_TARGET.md:86-99` and its harness. | ⬜ | that harness (rust/haskell/lua rows) | S | disjoint (docs + test) — **follow-on to P1** | main/opus |
| G-W6 | **Parallelism + cost model** — a real fleet capability, present in **≥3 backends** each with its own mechanism: `wam_rust` (rayon / `parallel_gate.pl`, 86 hits), `wam_haskell` (`parMap` + `cost_function.hs.mustache`, 76 hits), `wam_elixir` (actor/Task-based, 23 hits + its own lowered emitter). `wam_javascript` has none. Node analogue = Worker threads. **DEFERRED — low priority: comes after G-W1 (lowered emitter), P1 (indexing), and other single-thread perf.** | ⬜ (deferred) | wam_haskell cost model; wam_elixir for actor style | L | ⚠️ runtime + emitter | — (later) |

---

## 5. Sequencing / collision rules

- **Grok owns `runtime.js.mustache` + `wam_javascript_target.pl`** while indexing (P1) is live. All ⚠️ WAM items serialize behind it — do NOT hand them to a parallel agent.
- **Opus takes disjoint pattern-target gaps** (recursion patterns, components, aggregates, PAR-1 activation) which don't touch the WAM files.
- **INT-0 (main)** applies shared-file wiring centrally (registry, BINDING_MATRIX, harness arms) from each agent's integration patch.
- Worktree agents branch from `main`; for extend-not-rebuild tasks, tell them the target files already exist on `main` and to return net deltas.

---

## 5b. Explicitly N/A or premature for `wam_javascript` (do NOT file as near-term gaps)

Per A2 — recorded so they aren't re-raised:
- **Tier-3 FFI graph kernels** — the reference peer `wam_lua` has none either; not a peer-parity gap. Only after Tier-2 (G-W1).
- **Parallelism gate / cost model** (`parallel_gate.pl`, cost templates) — used by **only** `wam_rust` (haskell uses parMap); not a peer gap.
- **Perf cross-target row** (`WAM_PERF_CROSS_TARGET.md`) — meaningless until a lowered/kernel path exists; add after G-W1.
- **Distinct string type tag / String-vs-Atom standard order** — low-value ISO corner (`WAM_JAVASCRIPT_STATUS.md:93`).

## 6. Pending analysis (populate on completion)

- **A1 (pattern parity + component axis)** → ✅ landed; folded into §3 (rewritten as 13 confirmed gaps G-P1..G-P13) and §3b (N/A). Headline: canned-fib recursion hooks, no structural recursion, no aggregates, component collect-but-never-emit, clojure/CLJS component void, vanilla_js hook asymmetry. Corrected: bindings are not a gap (JS leads), hook *count* is fine.
- **A2 (WAM parity)** → ✅ landed; folded into §4 and §5b (WAM rows collapsed 10→5 confirmed gaps; Tier-3/parallelism/perf reclassified N/A; builtin-surface reframed as ISO breadth since JS is ahead of Lua on aggregates).
- **Census (Sonnet)** → ✅ landed; denominators added above §4. Key: lowered emitter 14/18 (JS one of 4 without), external fact sources 14/18, runtime-parser 7/18, dedicated kernels 2/18. Also confirmed annotated_js/vanilla_js have NO own bindings/runtime/template dirs — by design (inherit TS); not a gap. And 10 WAM modules exist on disk but aren't registry-registered (invoked directly by tests) — an architecture note, not a gap.

All three analyses (A1, A2, census) are now folded in. Next: schedule the wave (see §7).

## 7. Proposed next wave (post-analysis)

Grok stays in the WAM runtime/emitter lane; Opus takes disjoint pattern work; both avoid the files the other is editing.

**Highest impact first:**
1. **G-P1 + G-P2 (real recursion)** — replace canned-fib hooks + add structural recursion. Opus. Large. The most important pattern fix (current output is partly fake).
2. **G-P10 (vanilla_js hooks)** — quick correctness fix. Opus/main. Small.
3. **G-P4 (component emission)** — wire `compile_component` into TS `compile_module`; load orphaned `custom_chart`. Opus. Medium.
4. **G-W5 (switch-indexing conformance row)** — after Grok's P1 indexing lands. Small.
5. **G-W1 (lowered emitter)** — Grok, after indexing. Large; fleet norm (14/18).
6. Then: G-P3 aggregates, G-W2 parser, G-W3 builtin breadth, G-P5/G-P6/G-P7/G-P8/G-P9, G-W4 fact sources.
**Deferred (low priority):** G-W6 parallelism/cost, dedicated kernels, perf row.
