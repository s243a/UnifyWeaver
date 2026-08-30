<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# CLJS-1 Integration Patch

Shared-file edits for the **CLJS-1 (ClojureScript sci / bb / nbb variants)**
card. Per the delegation plan's shared-file rule, the CLJS-1 worktree does **not**
touch `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
`src/unifyweaver/core/advanced/test_advanced.pl`, or `glue/js_glue.pl`. INT-0
(main session) applies the snippets below centrally.

Everything else for CLJS-1 lands in the worktree itself:

- `src/unifyweaver/targets/clojurescript_target.pl` — `runtime(Kind)` option
  (`scittle` / `nbb` / `bb` / default), `clojurescript_from_clojure/3`,
  runtime-specific banner + shebang; JVM→JS rewrite kept centralized and applied
  only for the JS-host runtimes (bb keeps JVM interop).
- `src/unifyweaver/bindings/clojurescript_bindings.pl` — **new**, 67 bindings.
- `src/unifyweaver/core/recursive_compiler.pl` — TC path now threads `TcOptions`
  into `clojurescript_from_clojure/3` (additive, module-qualified; no import or
  API change). *Not a shared hot file — applied in the worktree.*
- `tests/core/test_clojurescript_target.pl` — runtime-variant unit tests.
- `tests/core/test_clojurescript_runtime_smoke.pl` — bb + nbb runtime arms,
  gated/skipped when the binary is absent.
- `docs/CLOJURESCRIPT_TARGET.md` — **new**, runtime variants + squint note.

---

## 1. `core/target_registry.pl` — capability additions

Add `nbb, babashka` to the clojurescript capability list.

**Before** (js family, `register_builtin_targets/0`):
```prolog
register_target(clojurescript, javascript, [streaming, functional, lisp, browser, scittle, interpreted]),
```

**After:**
```prolog
register_target(clojurescript, javascript, [streaming, functional, lisp, browser, scittle, nbb, babashka, interpreted]),
```

No `target_module/2` change needed — `target_module(clojurescript, clojurescript_target)`
already exists.

---

## 2. `docs/BINDING_MATRIX.md` — new row

Add a **ClojureScript** row to the "Target | Bindings | Categories" table
(alongside the existing **Clojure** row). The new
`bindings/clojurescript_bindings.pl` registers **67** bindings across the same
five categories as Clojure:

```markdown
| **ClojureScript** | 67 | 5 (Core, Collections, Sequences, Strings, Threading Macros) |
```

(For reference, the sibling **Clojure** row reads `| **Clojure** | 64 | 5 (...)`.
ClojureScript adds `sort`, `distinct`, `range` to the sequence category.)

Optionally add a **TypeScript**/**ClojureScript** presence note in the
per-binding detail tables — not required for this card.

---

## 3. `src/unifyweaver/core/advanced/test_advanced.pl` — optional module load

The advanced-recursion suite loads each target module for side effects. Add
clojurescript alongside the existing clojure load (line ~46) so the CLJS variant
is exercised in the advanced matrix:

**After** the existing:
```prolog
:- use_module('../../targets/clojure_target', []).
```
add:
```prolog
:- use_module('../../targets/clojurescript_target', []).
```

CLJS inherits the base Clojure recursion patterns unchanged, so no new
pattern-specific assertions are required; the runtime behavior is covered by
`tests/core/test_clojurescript_runtime_smoke.pl` (bb + nbb).

---

## 4. `glue/js_glue.pl` — no change required

`clojurescript` already routes through its own `target_module`, not through the
`js_runtime_choice/2` label mechanism. The bb/nbb runtimes are selected by the
`runtime(Kind)` **compile option** on the ClojureScript target, not by a JS
runtime label, so `js_glue.pl` needs no edit for CLJS-1. (A future `squint`
build path — see `docs/CLOJURESCRIPT_TARGET.md` — may revisit this.)

---

## Verification performed in the worktree

- `swipl -q -g test_clojurescript_target -t halt tests/core/test_clojurescript_target.pl` → all pass (incl. 10 new runtime-variant tests).
- `NBB=... BB=... swipl -q -g test_clojurescript_runtime_smoke -t halt tests/core/test_clojurescript_runtime_smoke.pl` → **All 9 tests passed** (5 nbb + 4 bb) with both runtimes present; **No tests to run** (all skipped, still passes) when both binaries are absent.
- `swipl -q -g test_clojurescript_bindings ... clojurescript_bindings.pl` → 67 bindings.
