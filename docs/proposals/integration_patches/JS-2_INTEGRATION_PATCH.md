<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2025 John William Creighton (@s243a) -->

# JS-2 Integration Patch — Vanilla JS Pattern Target

This card (JS-2) adds a self-contained target module
(`src/unifyweaver/targets/vanilla_js_target.pl`), its tests, and its docs
without touching any shared coordination file. The edits below are the ones a
coordinator must apply to wire the target into the registry and the shared test
matrices. Each is a small, additive change — apply them verbatim.

> **Shared-file collision rule:** the four files below are shared with other JS
> cards. Apply these lines; do not reformat the surrounding blocks.

---

## 1. `src/unifyweaver/core/target_registry.pl`

### 1a. Register the target in `register_builtin_targets/0`

In the **JavaScript family** block (the run of `register_target(...)` calls that
begins with `register_target(typescript, javascript, ...)`), add:

```prolog
    register_target(vanilla_js, javascript, [plain, modules, async]),
```

Place it directly after the `typescript` / `clojurescript` lines. Mind the
comma: it must match the surrounding conjunction (every entry except the final
`register_target(sql, ...)` line ends with a comma).

### 1b. Link the target module

In the `target_module/2` facts block, add:

```prolog
target_module(vanilla_js, vanilla_js_target).
```

Place it next to `target_module(typescript, typescript_target).`.

---

## 2. `src/unifyweaver/core/advanced/test_advanced.pl`

Add the target module to the load block (around lines 30–46), so its advanced
multifile recursion dispatch clauses are registered for the advanced-recursion
suite. The clauses themselves are inherited from `typescript_target`, which is
already loaded — loading `vanilla_js_target` (which `use_module`s the TS base)
keeps the matrix complete and lets the suite compile through the vanilla target:

```prolog
:- use_module('../../targets/vanilla_js_target', []).
```

Place it after the `:- use_module('../../targets/typescript_target', []).` line.

---

## 3. `docs/BINDING_MATRIX.md`

The Vanilla JS target adds **no new bindings**: it reuses the TypeScript base
target's binding system unchanged (`typescript_bindings`), and its binding hooks
(`collect_binding_import/1`, `clear_binding_imports/0`, `get_collected_imports/1`)
delegate straight to `typescript_target`. TypeScript is not currently a row in
the BINDING_MATRIX summary table, so no new row is required.

If/when a JavaScript-family row is added, note that Vanilla JS shares it with
TypeScript. Suggested one-line note under the target summary:

```markdown
> **Vanilla JS** and **TypeScript** share one binding set (`typescript_bindings`);
> Vanilla JS emits the same imports with the type annotations stripped.
```

---

## Verification after applying

```bash
# Target module + tests (self-contained; already green in the worktree)
swipl -q -g test_vanilla_js_target -t halt tests/core/test_vanilla_js_target.pl

# Registry dispatch (after patch 1)
swipl -q -g "use_module('src/unifyweaver/core/target_registry'), \
    target_module(vanilla_js, M), writeln(M), \
    target_has_capability(vanilla_js, plain), writeln(ok), halt" \
    src/unifyweaver/core/target_registry.pl
```
