<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P5 / G-P6 Integration Patch (for INT-0)

**Task:** G-P5 — wire the component pattern into the Clojure target (and thereby
ClojureScript), reviving the orphaned `custom_clojure` component; G-P6 — add
`compile_module/3` to ClojureScript.
**Worktree:** `agent-a158f0174162bb010`
**Shared-file rule:** this agent did **NOT** edit `core/target_registry.pl`,
`docs/BINDING_MATRIX.md`, `tests/test_advanced.pl`, `glue/js_glue.pl`,
`typescript_target.pl`, `annotated_js_target.pl`, `vanilla_js_target.pl`, or any
`wam_*` file.

## Files changed (all inside the allowed set)

| File | Change |
|------|--------|
| `src/unifyweaver/targets/clojure_target.pl` | **G-P5:** `use_module('../core/component_registry')` + `use_module('clojure_runtime/custom_clojure', [])` (triggers its `:- initialization(...,now)` self-registration — the module was dead before). Added `:- dynamic collected_component/2`, `collect_declared_component/2`, `compile_collected_components/1` (mirrors `python_target.pl:~187,195`). `init_clojure_target/0` now clears `collected_component/2`. **G-P6:** added `clojure_predicate_defn/3` (emits just a predicate's `(defn ...)` — no file header, no CLI entry — reusing the existing native clause lowering) and `compile_module/3` (base multi-predicate module: `(ns ...)` form + each defn + any collected components). All exported. |
| `src/unifyweaver/targets/clojurescript_target.pl` | **G-P6:** added and exported `compile_module/3` — reuses `clojure_target:compile_module/3` then applies the existing `clojurescript_from_clojure/3` (JVM→JS interop rewrite + CLJS binding-name rewrite + runtime banner/shebang). Changed the base import to `use_module(clojure_target, except([compile_module/3]))` so the CLJS variant is the one in scope (see caveat below). |
| `src/unifyweaver/targets/clojure_runtime/custom_clojure.pl` | Narrowed the `component_registry` import to `[register_component_type/4]` (silences three "local definition overrides weak import" load warnings), matching `custom_typescript.pl`'s style. No behavior change. |
| `tests/core/test_clojurescript_target.pl` | Added `compile_module_multi_predicate`, `compile_module_accepts_pred_terms`, `cljs_component_emission_includes_declared` (+ setup/cleanup), and `cljs_component_free_module_unchanged`. |

**No central wiring is required.** `compile_module/3` is a directly-called API
(the same way `annotated_js`/`vanilla_js` call `typescript_target:compile_module/3`),
not a `target_registry` dispatch entry — grep confirms `target_registry.pl` never
references `compile_module`. The registry's single-predicate dispatch
(`compile_predicate/3`) is unchanged. No capability/registry change warranted.

## One coordination caveat (no action needed unless a new consumer appears)

`clojure_target.pl` now **exports** `compile_module/3` (useful for JVM Clojure
module compilation, and matching the TS/python precedent). Consequently, any
module that `use_module`s **both** `clojure_target` and `clojurescript_target`
would get a `compile_module/3` import clash. The two current such consumers are
already safe:

- `clojurescript_target.pl` imports the base with `except([compile_module/3])`.
- `core/recursive_compiler.pl` and `core/advanced/test_advanced.pl` import each
  target with explicit/empty import lists, so neither pulls in `compile_module/3`.

If INT-0 later adds a consumer importing both with a bare `use_module`, use an
`except([compile_module/3])` (or a selective import) on one of them. Verified:
`recursive_compiler.pl`, `advanced/test_advanced.pl`, and
`tests/test_js_pattern_cross_target_conformance.pl` all still load clean.

## Behavior preservation

`compile_collected_components/1` returns `''` when nothing was collected, so
`compile_module/3` emits exactly `(ns …)` + defns with no component markers —
verified by `component_free_module_unchanged` / `cljs_component_free_module_unchanged`.
The single-predicate path (`compile_predicate_to_clojure/3` /
`compile_predicate_to_clojurescript/3`) is untouched: it keeps its file header and
per-predicate CLI entry. JVM regression (`test_clojure_native_lowering`),
CLJS target/recursive/runtime-smoke suites, and the G-P11 binding wiring are all
still green.
