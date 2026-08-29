<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# JS-family targets: build/improve plan & multi-agent delegation

**Status:** proposal / for review. Nothing here is committed to `main`.
**Scope:** the JavaScript-family compilation targets — TypeScript, ClojureScript
(incl. the sci/Babashka runtime variants), annotated (JSDoc) JS, and vanilla JS —
across the three capability axes UnifyWeaver targets carry: **recursion patterns**,
**bindings**, and **WAM (with hybrid variants)**.

This document is the delegation spec: it records current state, the per-target
checklist, the model assignments (Grok vs. Opus subagents), the collision-avoidance
strategy, and the acceptance bars. The companion file
`GROK_ANNOTATED_JS_TARGET_PROMPT.md` is the self-contained prompt for the Grok slice.

---

## 1. Current state (source-verified)

| Target | Exists today | Kind | Verdict |
|---|---|---|---|
| **TypeScript** | `targets/typescript_target.pl` (~59 KB), `bindings/typescript_bindings.pl`, `targets/typescript_runtime/`, `docs/TYPESCRIPT_TARGET.md` | Pattern/direct (no WAM) | **Improve** |
| **Clojure (JVM)** | `targets/clojure_target.pl` (~59 KB) + `wam_clojure_target.pl` (hybrid, narrow LMDB niche) | Pattern + a hybrid WAM | (base for CLJS) |
| **ClojureScript** | `targets/clojurescript_target.pl` (279 lines, *inherits* `clojure_target`); Scittle/**SCI** in browser, **nbb** on Node; WAM→CLJS path exists | Pattern/direct | **Improve/extend** |
| **Annotated (JSDoc) JS** | — does not exist | — | **Build new** |
| **Vanilla JS** | — does not exist | — | **Build new** |
| **Babashka (bb) / squint** | — do not exist | — | **Build (part of CLJS work)** |

`node` / `deno` / `bun` / `browser` are registered target *names* (family
`javascript`) but have **no `target_module/2`** — they are runtime labels selected by
`glue/js_glue.pl:js_runtime_choice/2`, not emitters. All JS-family emission currently
routes through the TypeScript target.

---

## 2. Architecture primer (what every target agent must know)

### 2.1 Two kinds of target
- **Pattern / direct target** — translates Prolog clauses straight to idiomatic host
  code, implementing each recursion pattern itself. Cheap; **no WAM conformance bar**.
  (TS, ClojureScript, python, bash are these.)
- **Hybrid WAM target** — Prolog → shared WAM bytecode (`targets/wam_target.pl`) → a
  host-language WAM interpreter + optional per-predicate "lowered" fast paths +
  optional native FFI kernels. Must pass the 15-arm cross-target conformance suite.

**Decision (yours): run both routes in parallel.** The JS family is pattern-based
today; we keep the new/improved JS targets pattern-based for shipping (matches the
peerhailer "no build step, debuggability first" thesis) **and** stand up a
`wam_javascript` hybrid as a separate research track.

### 2.2 The module contract (pattern target)
From `typescript_target.pl` — a target module exports:
`target_info/1`, `compile_predicate/3`, `compile_facts/3`, `compile_recursion/3`,
`compile_module/3`, `write_<lang>_module/2`, `init_<lang>_target/0`, plus the binding
hooks `clear_binding_imports/0`, `collect_binding_import/1`, `get_collected_imports/1`.

`target_info/1` is a dict, e.g.:
```prolog
target_info(info{ name:"TypeScript", family:javascript, file_extension:".ts",
  runtime:auto, features:[types,generics,async,modules,interfaces],
  recursion_patterns:[tail_recursion,linear_recursion,list_fold,transitive_closure],
  compile_command:"npx tsc" }).
```

### 2.3 Inheritance precedent (use it)
`clojurescript_target.pl` does `:- use_module(clojure_target)` and overrides **only**
the JVM→JS differences (interop rewrite, deps file, build artifact). Same shape as
`python_cython_target : python_target`. **The annotated-JS and vanilla-JS targets
should inherit from `typescript_target`** and vary only type-annotation emission.

### 2.4 The three capability axes
- **Recursion patterns** (`docs/RECURSION_PATTERN_THEORY.md`, `docs/ADVANCED_RECURSION.md`):
  six patterns — tail, linear, multi-call linear, direct multi-call, tree, mutual —
  dispatched by `core/advanced/advanced_recursive_compiler.pl`. Pattern targets emit
  loop/memo/structure code per pattern (multifile dispatch keyed on `target(T)` in
  Options); a target is exercised in the `tests/test_advanced.pl` matrix. WAM targets
  get recursion "for free" from choice points. **JS targets that inherit from TS
  inherit TS's patterns** (`tail_recursion, linear_recursion, list_fold, transitive_closure`).
- **Bindings** (`docs/BINDING_MATRIX.md`): map Prolog builtins → host functions via
  `declare_binding/6` in `bindings/<lang>_bindings.pl`; registry
  `core/binding_registry.pl` + `binding_codegen.pl`; imports flow through
  `collect_binding_import/1`. The matrix is a living scoreboard (Python 106, Go 50+,
  Clojure 64, C++ 45…). **There is no TypeScript or ClojureScript row yet** — a gap.
- **WAM + hybrid** (`docs/WAM_BACKEND_CONVENTIONS.md`,
  `docs/WAM_HYBRID_TARGETS_COMPARISON.md`, `docs/WAM_TARGET_ROADMAP.md`): the parity
  bar for a WAM target is the six conventions + the new-backend checklist
  (WAM_BACKEND_CONVENTIONS.md lines 231-257) + a green run of the conformance suite.

### 2.5 Registration
`core/target_registry.pl`: add a `register_target(Name, Family, Caps)` line in
`register_builtin_targets/0` (js family lives at lines 218-223) and a `target_module(Name, Module)`
fact (js family at lines 260-276). `compile_to_target/4` is the universal dispatch.

### 2.6 Conformance / parity
- **WAM route:** `tests/wam_conformance_fixtures.pl` (shared spec) +
  `tests/test_wam_cross_target_conformance.pl` (harness). 15 arms currently green with
  **zero** `ct_xfail`/`ct_skip`. Programs: `member, append, reverse, fib, ack,
  builtins` + head-shape `wide, nested, buildnest, repeatvar, emptylist`. Wire a
  backend with `ct_build/4`, `ct_run/5`, `ct_teardown/2`, `conformance_target/1`,
  `ct_toolchain/2`. Run:
  ```
  CONFORMANCE_TARGETS=<t> swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl
  ```
- **Pattern route:** there is **no** cross-target parity harness for pattern targets
  today — each is validated by its own plunit suite + the `test_advanced.pl` matrix.
  We add one (**PAR-1** below) so the four JS pattern targets are checked for parity by
  running generated code under Node/nbb against the Prolog oracle.

---

## 3. Work items (task cards)

Each card is self-contained (files, reference impl, steps, acceptance) so it can be
handed to an agent. **Shared-file rule:** no agent edits
`target_registry.pl`, `BINDING_MATRIX.md`, `test_advanced.pl`, or `js_glue.pl`
directly. Instead each agent emits a small **"integration patch" snippet** in its
handoff, and **INT-0 (main session)** applies all of them centrally. This keeps
worktrees conflict-free on the hot shared files.

### JS-1 — Annotated (JSDoc) JS pattern target  ·  Owner: **Grok**
- **Goal:** new pattern target emitting plain `.js` + JSDoc type comments, checkable
  with `tsc --checkJs` (no emit, no runtime dep). The peerhailer-ethos "best" target.
- **Create:** `src/unifyweaver/targets/annotated_js_target.pl` (inherits
  `typescript_target`), `src/unifyweaver/targets/annotated_js_runtime/` (if needed),
  `tests/core/test_annotated_js_target.pl`, `docs/ANNOTATED_JS_TARGET.md`.
- **Reference:** `clojurescript_target.pl` (inheritance shape), `typescript_target.pl`
  (contract + JSDoc emission already present in `typescript_runtime/custom_typescript.pl`).
- **Integration patch (hand to INT-0):** `register_target(annotated_js, javascript, [jsdoc, tsc_checked, modules, async])` + `target_module(annotated_js, annotated_js_target)`; `BINDING_MATRIX.md` row.
- **Acceptance:** `swipl -q -g test_annotated_js_target -t halt tests/core/test_annotated_js_target.pl` green; a sample predicate's output passes `npx tsc --checkJs --noEmit --allowJs` clean.
- **Full brief:** `GROK_ANNOTATED_JS_TARGET_PROMPT.md`.

### JS-2 — Vanilla JS pattern target  ·  Owner: **Opus subagent**  ·  depends on JS-1
- **Goal:** new pattern target = annotated-JS minus the JSDoc/type comments (or a
  strip pass over JS-1's output). `family: javascript`, runtime via `js_runtime_choice/2`.
- **Create:** `src/unifyweaver/targets/vanilla_js_target.pl` (inherits
  `annotated_js_target` or `typescript_target`), `tests/core/test_vanilla_js_target.pl`,
  `docs/VANILLA_JS_TARGET.md`.
- **Integration patch:** `register_target(vanilla_js, javascript, [plain, modules, async])` + `target_module(vanilla_js, vanilla_js_target)`.
- **Acceptance:** plunit suite green; sample output runs under `node` and produces the
  oracle result.

### TS-1 — TypeScript hardening  ·  Owner: **Opus subagent**
- **Goal:** close the two known TS gaps — no `BINDING_MATRIX` row, thin tests, not in
  any parity harness.
- **Touch/create:** count/declare TS bindings in `bindings/typescript_bindings.pl` and
  add the matrix row (integration patch); expand `tests/core/test_typescript_target.pl`
  (facts, each recursion pattern, module compile, services); register TS as an arm in
  **PAR-1**.
- **Acceptance:** TS plunit suite green; PAR-1 green for `typescript`.

### CLJS-1 — ClojureScript sci / bb / nbb variants  ·  Owner: **Opus subagent**
- **Goal:** extend the existing CLJS target with an **nbb / Babashka (bb)** Node
  runtime path (sci-based), and (optional) a **squint** build-based path; the browser
  Scittle/SCI path already exists.
- **Touch/create:** `targets/clojurescript_target.pl` (runtime-variant option +
  bb/nbb entrypoint emission), `bindings/clojurescript_bindings.pl` (+ matrix row),
  extend `tests/core/test_clojurescript_runtime_smoke.pl` to also run under `bb`/`nbb`.
- **Reference:** `docs/WAM_CLOJURE_STATUS.md`, `docs/handoff/scirepl-clojurescript-kernel/`,
  the peerhailer doc's `shell:bb` idea (bb as an external binary, no npm dep).
- **Acceptance:** CLJS plunit suites green; runtime smoke green under `nbb` (and `bb`
  when present; skip cleanly when absent, per the existing gating convention).

### PAR-1 — Pattern-target JS parity harness  ·  Owner: **Opus subagent** / INT-0
- **Goal:** the missing cross-target parity harness for pattern JS targets. Shared
  fixtures + Prolog oracle; run each target's generated code under its runtime
  (node / nbb) and diff against expected.
- **Create:** `tests/js_pattern_conformance_fixtures.pl`,
  `tests/test_js_pattern_cross_target_conformance.pl` (model on
  `tests/test_wam_cross_target_conformance.pl` — reuse its env-knob + skip-on-missing-toolchain shape).
- **Arms:** `typescript, annotated_js, vanilla_js, clojurescript`.
- **Acceptance:** green for all present arms; missing runtimes skip, never fail.

### WAMJS-1 — `wam_javascript` hybrid target (research)  ·  Owner: **Opus subagent (worktree)**
- **Goal:** Prolog → `wam_target.pl` bytecode → a JS (Node) WAM interpreter, following
  the six WAM conventions; register a conformance adapter.
- **Create:** `src/unifyweaver/targets/wam_javascript_target.pl` (+ optional
  `wam_javascript_lowered_emitter.pl`), runtime template dir
  `templates/targets/javascript_wam/`, `bindings/javascript_wam_bindings.pl`,
  `docs/WAM_JAVASCRIPT_STATUS.md`.
- **Reference:** `wam_rust_target.pl`, `wam_haskell_target.pl`, `wam_cpp_target.pl`
  (contrast: full template dir → partial → inline emit), `docs/WAM_BACKEND_CONVENTIONS.md`.
- **Integration patch:** register arm via `ct_build/4`, `ct_run/5`, `ct_teardown/2`,
  `conformance_target(javascript)`, `ct_toolchain(javascript, node)`.
- **Acceptance:** `CONFORMANCE_TARGETS=javascript swipl -q -g run_tests -t halt tests/test_wam_cross_target_conformance.pl`
  green on the full spec (classics + head-shape), **no** `ct_xfail`/`ct_skip`.

### INT-0 — Shared-file wiring & merge coordination  ·  Owner: **main session (me)**
- Own all edits to `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
  `tests/test_advanced.pl` matrix, `glue/js_glue.pl`. Collect each agent's integration
  patch, apply centrally, merge worktree branches, run the full suite, resolve conflicts.

---

## 4. Model assignments & sequencing

| Item | Owner | Depends on | Route |
|---|---|---|---|
| JS-1 Annotated JS | **Grok** (prompt handoff) | — | pattern |
| JS-2 Vanilla JS | Opus subagent | JS-1 | pattern |
| TS-1 TS hardening | Opus subagent | (PAR-1 for the harness arm) | pattern |
| CLJS-1 sci/bb/nbb | Opus subagent | — | pattern |
| PAR-1 parity harness | Opus subagent / INT-0 | — | pattern |
| WAMJS-1 hybrid | Opus subagent (worktree) | — | WAM research |
| INT-0 wiring/merges | main session | all | — |

**Sequencing:** JS-1 (Grok) and the independent Opus items (TS-1, CLJS-1, PAR-1,
WAMJS-1) start together. JS-2 begins once JS-1 lands. INT-0 runs continuously.

**Isolation:** every Opus subagent runs in its own git worktree and touches only its
own new files + emits an integration patch; INT-0 serializes the shared-file edits.
Grok works from the companion prompt and returns files + an integration patch.

---

## 5. Acceptance bars (summary)

- **Pattern targets:** own plunit suite green **and** a `test_advanced.pl` matrix
  entry green **and** PAR-1 parity green for that arm. Annotated JS additionally:
  output passes `tsc --checkJs --noEmit`.
- **WAM target:** the six conventions honored + the new-backend checklist +
  conformance suite green on the full spec with no tolerated divergences.
- **Every target:** SPDX headers on new files; no new runtime npm/crate dependency
  (dev-only toolchains — tsc, node, nbb, bb — are fine); a `docs/<LANG>_TARGET.md`.

---

## 6. Open questions to settle at review

1. Target names: `annotated_js` / `vanilla_js` vs. reusing `node`/`browser` labels —
   proposal keeps them distinct emitter names. OK?
2. CLJS squint path: in-scope now, or defer (bb/nbb only for v1)?
3. WAMJS-1 is genuinely heavy (Rust/C++/Haskell WAM modules are 7k–11k lines). Treat
   as a longer research spike behind its own worktree, not gated to the pattern work?
