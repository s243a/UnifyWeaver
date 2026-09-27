<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P11 Integration Patch

Shared-file notes for the **G-P11 (wire the `clojurescript` binding key into the
CLJS compile path)** work. Per the delegation plan's shared-file rule, this
worktree does **not** touch `core/target_registry.pl`,
`src/unifyweaver/core/advanced/test_advanced.pl`, or `glue/js_glue.pl`. Only one
shared doc needs a follow-up count bump (item 1 below) for INT-0.

## Root cause (confirmed)

`bindings/clojurescript_bindings.pl` (the 67-binding catalogue, key
`clojurescript`) was **inert on two counts**:

1. **Never loaded.** No module `use_module`'d it, and its bindings register
   through `init_clojurescript_bindings/0` (direct `declare_binding/6` calls) --
   the `:- initialization(declare_binding(...))` term-expansion only fires for
   the `:- cljs_binding(...)` *directive* form, which the file does not use. So
   `stored_binding(clojurescript, ...)` was empty at runtime. (The JVM
   `clojure` catalogue, `bindings/clojure_bindings.pl`, was equally unloaded --
   64 bindings, also dead.)
2. **Never consulted.** The CLJS compile path is
   `compile_predicate_to_clojure/3` (base codegen) + `clojurescript_interop_rewrite/2`.
   The base Clojure codegen emits **raw Prolog functors** as call heads
   (verified: `run/2 :- myfn(X,Y)` → `(myfn arg1)`); it performs **no** binding
   lookup at all -- not on `clojure`, not on `clojurescript`. The interop
   rewrite only translates fixed JVM host tokens (`Integer/parseInt`, `Math/`,
   …). So the `clojurescript` key was never read.

The shared clojure bindings "worked" only because Prolog functors were chosen to
match Clojure core names (`map`→`map`, `+`→`+`), making the absent lookup a
silent no-op.

## Fix (all in-worktree, no hot shared files)

- `src/unifyweaver/targets/clojurescript_target.pl`
  - `use_module`s `../core/binding_registry`, `../bindings/clojure_bindings`,
    and `../bindings/clojurescript_bindings`.
  - `ensure_cljs_bindings/0` — idempotent (dynamic `cljs_bindings_loaded/0`
    guard) loader that runs `init_clojure_bindings/0` then
    `init_clojurescript_bindings/0`, populating both the fallback (`clojure`) and
    override (`clojurescript`) keys. Called from `init_clojurescript_target/0`
    and from the rewrite entry point, so a bare
    `compile_predicate_to_clojurescript/3` (as the tests call it) works without a
    separate init step.
  - `cljs_binding_name_rewrite/2` — the routing. Enumerates every predicate with
    a `clojurescript` or `clojure` binding, resolves each through the **existing**
    `binding_registry:resolve_binding([clojurescript, clojure], Pred, _, Binding)`
    (the coordinator-specified fallback: a `clojurescript` binding overrides; the
    64 shared `clojure` bindings are the fallback), and rewrites the predicate's
    **call head** to the resolved target name. Boundary-anchored on `(fn ` and
    `(fn)` (name immediately after an open paren), with both the raw-functor and
    underscore→hyphen spelling as the source, so `(defn fn …)` headers and
    substrings of other tokens are never touched.
  - Wired into `clojurescript_from_clojure/3`: for JS-host runtimes
    (scittle/nbb/default) the pass runs **after** `clojurescript_interop_rewrite/2`;
    for `bb` (Clojure-on-JVM) it is **not** applied, so bb keeps JVM names and
    JVM interop unchanged.
- `src/unifyweaver/bindings/clojurescript_bindings.pl`
  - New `register_divergent_bindings/0` section (called from
    `init_clojurescript_bindings/0`) with one host-divergent binding:
    `parse_double/2 → 'js/parseFloat'`. ClojureScript's `cljs.core` has no
    `parse-double` (a JVM Clojure 1.11 addition); the CLJS idiom is the JS global
    `parseFloat`, and this difference is **not** expressible by the interop
    rewrite -- so it can only take effect via the `clojurescript` key. Catalogue
    count is now **68** (was 67).

## Avoiding double-transformation with the interop rewrite

The two text passes act on **disjoint tokens** and run in a fixed, documented
order:

- **Interop rewrite (first):** translates JVM host tokens on the base output
  (`Integer/parseInt` → `js/parseInt`, `Math/` → `js/Math.`, …).
- **Binding rewrite (second):** substitutes call heads with the resolved CLJS
  names, which may themselves be `js/...` host calls (e.g. `js/parseFloat`).

Because the binding rewrite runs last, the `js/...` names it introduces are never
re-seen by the interop pass and cannot be double-mangled. Conversely its *source*
tokens are the raw functors the base codegen emits, which the interop pass leaves
alone. The `bb` path applies neither transform's JS half, keeping JVM output
byte-identical.

## Concrete before/after proof (`toflo/2 :- parse_double(S,X)`)

| Path | Emitted call head |
|---|---|
| JVM Clojure base (`compile_predicate_to_clojure`) | `(parse_double arg1)` — unchanged |
| CLJS **before** (interop-only; interop doesn't match `parse_double`) | `(parse_double arg1)` |
| CLJS **after** (this fix) | `(js/parseFloat arg1)` |
| CLJS `runtime(bb)` (JVM host) | `(parse_double arg1)` — override correctly not applied |

`resolve_binding([clojurescript, clojure], parse_double/2, K, …)` → `K=clojurescript`,
name `js/parseFloat` (override). A `clojure`-only predicate resolves with
`K=clojure` (fallback). The emitted CLJS runs under `nbb`
(`(toflo "3.14")` → `3.14`, `(toflo "42")` → `42`), matching the SWI oracle.
The CLI entry in the same output shows `Integer/parseInt` → `js/parseInt` still
applied, with `js/parseFloat` untouched (no double-transform).

## 1. `docs/BINDING_MATRIX.md` — INT-0 only (count bump)

Row `**ClojureScript** | 67 | …` should become `68` (added `parse_double →
js/parseFloat`). Not edited here (hot shared doc). No code depends on the literal
67/68; no test hardcodes it.

## Acceptance (nbb present)

- `test_clojurescript_target`, `test_clojure_native_lowering` (JVM base, no
  regression), `test_clojurescript_recursive`, `test_clojurescript_runtime_smoke`
  (nbb arms run, not skipped) all green.
- `test_clojurescript_bindings` registers **68** bindings.
- Divergent `clojurescript` binding proven live: CLJS emits `js/parseFloat`
  (base/JVM emits `parse_double`); runs under nbb, matches SWI.
