<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — build the Annotated (JSDoc) JavaScript target for UnifyWeaver

> This file is the self-contained brief to paste into Grok. Everything Grok needs is
> inline; the "read these files" list is for grounding, not a prerequisite.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that translates Prolog predicates into many host languages ("targets"). Work in a
clean checkout on a branch named `grok/annotated-js-target`. Prolog dialect is
**SWI-Prolog** (`swipl`).

### Your task
Build a new **pattern/direct compilation target that emits plain JavaScript annotated
with JSDoc type comments** — code that runs unmodified on Node/browser AND type-checks
under `tsc --checkJs --noEmit` with **no build step and no runtime dependency**. This
is the highest-value JS target: the shipped artifact stays the exact file you read,
edit, and debug (the "the file you edit is the file that runs" principle).

Do **not** write a WAM interpreter. This is a pattern target: it translates clauses
directly, exactly like the existing TypeScript target, and reuses that target's logic
by inheritance.

### How targets work here (essential facts)
- A target is a Prolog module in `src/unifyweaver/targets/<name>_target.pl`.
- It exports this contract (see `typescript_target.pl`):
  `target_info/1`, `compile_predicate/3`, `compile_facts/3`, `compile_recursion/3`,
  `compile_module/3`, `write_<lang>_module/2`, `init_<lang>_target/0`, and the binding
  hooks `clear_binding_imports/0`, `collect_binding_import/1`, `get_collected_imports/1`.
- `target_info/1` is a dict: `info{name, family, file_extension, runtime, features,
  recursion_patterns, compile_command}`.
- **Inheritance is the intended pattern.** `clojurescript_target.pl` does
  `:- use_module(clojure_target)` and overrides ONLY the JVM→JS differences; the base
  target's clause lowering, recursion patterns, and expression translation are reused
  unchanged. Do the same: **inherit from `typescript_target` and override only the
  type-annotation emission** (TypeScript inline types → JSDoc comment blocks; `.ts` →
  `.js`; drop `interface`/generic syntax in favor of `@typedef`/`@param`/`@returns`).
- The recursion patterns you inherit from TS are:
  `tail_recursion, linear_recursion, list_fold, transitive_closure`. You do not need
  to reimplement them — verify they still emit valid JS after your annotation changes.
- JSDoc emission already exists in the repo: `typescript_runtime/custom_typescript.pl`
  emits a JSDoc block from a `description(...)` option. Reuse that style.

### Files to read first (for grounding)
- `src/unifyweaver/targets/typescript_target.pl` — the contract + all the codegen you inherit.
- `src/unifyweaver/targets/clojurescript_target.pl` — the exact inheritance/override shape to copy.
- `src/unifyweaver/targets/typescript_runtime/custom_typescript.pl` — JSDoc emission style.
- `src/unifyweaver/bindings/typescript_bindings.pl` — how bindings/imports are declared.
- `src/unifyweaver/core/target_registry.pl` (lines ~200-280) — registration format.
- `docs/TYPESCRIPT_TARGET.md`, `docs/BINDING_MATRIX.md`, `docs/RECURSION_PATTERN_THEORY.md`.
- `tests/core/test_clojurescript_target.pl` — the plunit test-file shape to copy.

### Deliverables (create these files; SPDX header on each)
1. `src/unifyweaver/targets/annotated_js_target.pl`
   - `:- module(annotated_js_target, [...contract...]).`
   - `:- use_module(typescript_target).`
   - `target_info(info{ name:"AnnotatedJS", family:javascript, file_extension:".js",
     runtime:auto, features:[jsdoc, tsc_checked, modules, async],
     recursion_patterns:[tail_recursion,linear_recursion,list_fold,transitive_closure],
     compile_command:"npx tsc --checkJs --noEmit --allowJs" }).`
   - `compile_predicate/3` delegates to the TS base, then post-processes the output:
     move inline TS types into `/** @param {T} x */`-style JSDoc, convert `interface`
     blocks to `@typedef`, strip type-only syntax so the result is valid ES module JS.
     Keep the transformation in ONE place (a `ts_to_annotated_js/2` rewrite predicate),
     mirroring how `clojurescript_interop_rewrite/2` centralizes JVM→JS rewrites.
   - Implement the binding hooks by delegating to the TS ones (or collect into the same
     import mechanism, emitting `import`/`require` per `js_runtime_choice`).
2. `tests/core/test_annotated_js_target.pl`
   - plunit, module `test_annotated_js_target`, exports `test_annotated_js_target/0`
     calling `run_tests([annotated_js_target])`. Cover: a fact predicate, each of the
     four recursion patterns, a multi-predicate module, and an assertion that the
     emitted text contains JSDoc (`/**`) and NO TypeScript-only syntax (`: number`,
     `interface `, `<T>`).
3. `docs/ANNOTATED_JS_TARGET.md` — usage + a feature table like `docs/TYPESCRIPT_TARGET.md`.
4. **`INTEGRATION_PATCH.md`** (do NOT edit the shared files yourself — a coordinator
   applies these centrally to avoid merge conflicts). List exactly:
   - the line to add to `register_builtin_targets/0` in `core/target_registry.pl`:
     `register_target(annotated_js, javascript, [jsdoc, tsc_checked, modules, async]),`
   - the fact to add near the other `target_module/2` js entries:
     `target_module(annotated_js, annotated_js_target).`
   - the new row for `docs/BINDING_MATRIX.md` (target `AnnotatedJS`, binding count you
     wired, categories).
   - the entry to add to the `tests/test_advanced.pl` multi-target recursion matrix.

### Constraints
- **No runtime dependency** — generated code must run on stock Node/browser; `tsc` is a
  dev-only *checker* invoked with `--noEmit` (never an emit/build step).
- Every new file starts with:
  ```
  % SPDX-License-Identifier: MIT OR Apache-2.0
  % Copyright (c) 2026 John William Creighton (s243a)
  ```
  (`.md` files use the `<!-- ... -->` comment form.)
- Match surrounding code style; keep the override surface minimal (inherit, don't fork).
- Do not modify `target_registry.pl`, `BINDING_MATRIX.md`, `test_advanced.pl`, or
  `js_glue.pl` — put those changes in `INTEGRATION_PATCH.md`.

### Acceptance (must pass before handoff)
1. `swipl -q -g test_annotated_js_target -t halt tests/core/test_annotated_js_target.pl`
   → all tests pass.
2. Compile a sample predicate (e.g. `factorial/2` and a `member/2`-style transitive
   closure), write the `.js`, then:
   `npx tsc --checkJs --noEmit --allowJs <file>.js` → **zero** type errors.
3. `node <file>.js` (or an ES-module smoke) produces the same result the Prolog query
   would (spot-check factorial and one list predicate).

### Handoff format
Return: the new files' full contents, `INTEGRATION_PATCH.md`, and a short note listing
which recursion patterns you verified and any TS→JSDoc rewrite edge cases you hit.

## ↑↑↑ Copy to here
