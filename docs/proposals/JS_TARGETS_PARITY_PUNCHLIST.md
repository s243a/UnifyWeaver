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

---

## 2. In progress 🔄

| ID | Target | Item | Owner | Branch |
|---|---|---|---|---|
| P1 | wam_javascript | First-argument indexing (`switch_on_constant/structure/term` → real indexed dispatch, var-first-arg fallback) | grok | `grok/wamjs-indexing` |
| A1 | pattern targets | Parity gap analysis (recursion patterns, bindings, components, compilation surface, tests) | opus (analysis) | — |
| A2 | wam_javascript | WAM parity gap analysis (tiers, emit modes, runtime-parser, fact sources, builtins, parallelism) | opus (analysis) | — |

---

## 3. Open gaps — PATTERN targets ⬜/❓

Two extension mechanisms apply here: **bindings** (`declare_binding/6`) and the
**component pattern** (`component_registry` + per-target `custom_*.pl`).

| ID | Target(s) | Gap | Status | Copy-from | Size | Owner |
|---|---|---|---|---|---|---|
| G-P1 | typescript, annotated_js, vanilla_js, clojurescript | **Missing recursion patterns** — target_info declares only `[tail_recursion, linear_recursion, list_fold, transitive_closure]`; theory defines ~6. Likely missing: **tree recursion**, **mutual recursion (SCC)**, **multi-call linear (memoized fib)**. | ❓ | python/rust/go multifile `target(...)` clauses in `core/advanced/*_recursion.pl` | M–L | opus |
| G-P2 | clojurescript | **Stream / pipeline / generator modes** — module header calls these "best-effort" (assume nbb Node runtime, not browser). | ❓ | python/go stream compilation | M | opus |
| G-P3 | annotated_js, vanilla_js | **Component pattern coverage** — do they inherit TS's `collect_declared_component/2` integration correctly? Any JS-specific component types? | ❓ | `typescript_runtime/custom_typescript.pl`, `custom_chart.pl` | S–M | opus |
| G-P4 | clojurescript | **Component pattern** — CLJS-specific component, or only inherited `custom_clojure`? | ❓ | `clojure_runtime/custom_clojure.pl` | S | opus |
| G-P5 | typescript, annotated_js, vanilla_js, clojurescript | **Aggregate compilation** (aggregate_all/findall/bagof at the pattern level) — do the mature targets compile these where the JS ones don't? | ❓ | `docs/proposals/AGGREGATE_ALL_*`, python/go | M | opus |
| G-P6 | JS pattern targets | **Constraint compilation** (`docs/CONSTRAINT_SYSTEM.md`, `PROLOG_CONSTRAINTS.md`). | ❓ | python/rust | M–L | opus |
| G-P7 | JS pattern targets | **External / data-source consumption** (`src/unifyweaver/sources/`). | ❓ | python/go | M | opus |
| G-P8 | annotated_js, vanilla_js, clojurescript | **Bindings depth** — TS 179; CLJS 67; annotated/vanilla inherit TS. Any category holes vs python(106)? | ❓ | python/go bindings | S–M | grok-friendly |
| G-P9 | annotated_js, vanilla_js | **PAR-1 arm activation** — arms exist but skip until the harness loads the targets; needs a real tsc/node run (killed in the local shell, fine in an agent env). | ⬜ | — | S | opus |
| G-P10 | JS pattern targets | **Test-coverage depth** vs python/rust/go suites. | ❓ | those suites | S–M | opus |

---

## 4. Open gaps — WAM (`wam_javascript`) ⬜ — CONFIRMED by analysis A2

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

- **A1 (pattern parity + component axis)** → 🔄 running. Will confirm/refute G-P1…G-P10 with file:line evidence, add a capability matrix incl. the component pattern, and a ranked list.
- **A2 (WAM parity)** → ✅ landed; folded into §4 and §5b (WAM rows collapsed 10→5 confirmed gaps; Tier-3/parallelism/perf reclassified N/A; builtin-surface reframed as ISO breadth since JS is ahead of Lua on aggregates).
- **Census (Sonnet)** → 🔄 running. Fleet-wide capability denominator (per-target tier/binding/component counts) to firm up "copy-from" references and headline numbers.

Once A1 + census land: reconcile IDs, set sizes/owners firmly, and schedule the next wave.
