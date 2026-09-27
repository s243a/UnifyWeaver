<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# Integration patch — AnnotatedJS target

Coordinator applies these edits to shared files. This branch does **not**
modify `target_registry.pl`, `BINDING_MATRIX.md`, `test_advanced.pl`, or
`js_glue.pl`.

New files already on this branch:

- `src/unifyweaver/targets/annotated_js_target.pl`
- `tests/core/test_annotated_js_target.pl`
- `docs/ANNOTATED_JS_TARGET.md`

## 1. `src/unifyweaver/core/target_registry.pl`

### `register_builtin_targets/0` (JavaScript family, ~line 218)

Insert next to the existing `typescript` registration:

```prolog
    register_target(annotated_js, javascript, [jsdoc, tsc_checked, modules, async]),
```

Suggested context:

```prolog
    register_target(typescript, javascript, [types, async, modules, generics]),
    register_target(annotated_js, javascript, [jsdoc, tsc_checked, modules, async]),
    register_target(clojurescript, javascript, [streaming, functional, lisp, browser, scittle, interpreted]),
```

### `target_module/2` (near the other JS entries, ~line 276)

```prolog
target_module(annotated_js, annotated_js_target).
```

Suggested context:

```prolog
target_module(typescript, typescript_target).
target_module(annotated_js, annotated_js_target).
target_module(haskell, haskell_target).
```

## 2. `docs/BINDING_MATRIX.md` (Summary table)

AnnotatedJS inherits the TypeScript binding table
(`src/unifyweaver/bindings/typescript_bindings.pl`): **155**
`declare_binding/6` entries in **11** categories.

Insert this row in the Summary table (after **Clojure** is a reasonable
home; anywhere in the target list is fine):

```markdown
| **AnnotatedJS** | 155 (via TypeScript) | 11 (Built-ins, String, Math, Array, Object, JSON, Console, Node fs, Node path, Promise, Date) |
```

## 3. `src/unifyweaver/core/advanced/test_advanced.pl`

Multi-target recursion matrix: load the target so its multifile
`compile_*_pattern` clauses (which wrap the TypeScript clauses and run
`ts_to_annotated_js/2`) are registered.

Insert next to the TypeScript `use_module` (~line 39):

```prolog
:- use_module('../../targets/annotated_js_target', []).
```

Suggested context:

```prolog
:- use_module('../../targets/typescript_target', []).
:- use_module('../../targets/annotated_js_target', []).
:- use_module('../../targets/jython_target', []).
```

## 4. Optional (recommended) — `recursive_compiler.pl`

Not required for the target contract, but needed for
`compile_recursive(..., [target(annotated_js)])` and
`compile_to_target(annotated_js, ...)`.

`compile_non_recursive/4` (~line 201):

```prolog
compile_non_recursive(annotated_js, Pred/Arity, FinalOptions, GeneratedCode) :-
    annotated_js_target:compile_predicate(Pred/Arity, FinalOptions, GeneratedCode).
```

`compile_transitive_closure/6` (next to the TypeScript clause ~line 887):

```prolog
compile_transitive_closure(annotated_js, Pred, Arity, BasePred, Options, GeneratedCode) :-
    compile_transitive_closure(typescript, Pred, Arity, BasePred, Options, TSCode),
    annotated_js_target:ts_to_annotated_js(TSCode, GeneratedCode),
    !.
```

## 5. `js_glue.pl`

No change. Runtime selection (`js_runtime_choice/2`) already covers
`node` / `deno` / `bun` / `browser`. AnnotatedJS is a codegen variant of
TypeScript, not a new JS host runtime.

## 6. Smoke after apply

```bash
swipl -q -g test_annotated_js_target -t halt tests/core/test_annotated_js_target.pl
```
