<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# AnnotatedJS Target

Compile Prolog predicates to **plain JavaScript** annotated with JSDoc type
comments. The shipped artifact is the exact `.js` file you read, edit, and
debug: it runs unmodified on Node or in the browser, and type-checks under
`tsc --checkJs --noEmit` with **no build step** and **no runtime dependency**.

This is a pattern/direct target (not a WAM interpreter). It inherits
[typescript_target](TYPESCRIPT_TARGET.md) and only overrides type-annotation
emission: TypeScript inline types become JSDoc (`@param`, `@returns`,
`@typedef`, `@type`, `@template`); `interface` / generic syntax is stripped
so the result is valid ES-module JavaScript.

## Overview

| Feature | Status |
|---------|--------|
| Facts | ✅ Typed arrays + `@typedef` (from TS `interface`) |
| Recursion | ✅ `tail_recursion`, `linear_recursion`, `list_fold`, `transitive_closure` |
| Modules | ✅ Multi-predicate compilation |
| JSDoc | ✅ `@param` / `@returns` / `@typedef` / `@type` / `@template` |
| `tsc --checkJs` | ✅ Dev-only checker (`--noEmit --allowJs`); never a build step |
| Runtime | ✅ Stock Node / browser (no UnifyWeaver runtime) |

## Quick Start

```prolog
?- use_module('src/unifyweaver/targets/annotated_js_target').

?- compile_module(
       [pred(sum, 2, tail_recursion),
        pred(factorial, 1, factorial)],
       [module_name('PrologMath')],
       Code),
   write_annotated_js_module(Code, 'PrologMath.js').
```

```bash
# Type-check (optional, no emit)
npx tsc --checkJs --noEmit --allowJs PrologMath.js

# Run (ES module)
node --input-type=module -e "import { factorial } from './PrologMath.js'; console.log(factorial(5))"
```

## API Reference

### `target_info/1`

```prolog
target_info(Info).
% Info.name            = "AnnotatedJS"
% Info.family          = javascript
% Info.file_extension  = ".js"
% Info.features        = [jsdoc, tsc_checked, modules, async]
% Info.recursion_patterns = [tail_recursion, linear_recursion, list_fold, transitive_closure]
% Info.compile_command = "npx tsc --checkJs --noEmit --allowJs"
```

### `compile_predicate/3`, `compile_facts/3`, `compile_recursion/3`, `compile_module/3`

Same contract as the TypeScript target. Each predicate delegates to
`typescript_target` and post-processes the result with `ts_to_annotated_js/2`.

```prolog
compile_recursion(sum/2, [pattern(tail_recursion)], Code).
compile_recursion(listSum/2, [pattern(list_fold)], Code).
compile_recursion(fib/2, [pattern(linear_recursion)], Code).
compile_recursion(ancestor/2, [pattern(transitive_closure), base_pred(parent)], Code).
```

### `ts_to_annotated_js/2`

Single rewrite predicate (mirrors `clojurescript_interop_rewrite/2`):

- Move `(n: number): number` into `/** @param {number} n` / `@returns {number} */`
- Convert `export interface Foo { x: string }` into `@typedef` / `@property`
- Convert `<T, R>(...)` into `@template T` / `@template R`
- Strip `!` non-null assertions, `as T` casts, and `new Map<K,V>()` generics

### Binding hooks

`clear_binding_imports/0`, `collect_binding_import/1`, and
`get_collected_imports/1` delegate to the TypeScript binding collector.
AnnotatedJS inherits the TypeScript binding table (155 bindings).

## Recursion Patterns

| Pattern | Description | Emitted JavaScript |
|---------|-------------|--------------------|
| `tail_recursion` | O(1) stack / TCO loop | `(n, acc = 0) => ...` + `*Strict` while-loop |
| `list_fold` | Array reduce | `items.reduce(...)` + generic fold helper |
| `linear_recursion` | Memoized fib | `Map` + `@type {Map<number, number>}` |
| `transitive_closure` | BFS reachability | Reuses the TS `tc_definitions` template |
| `factorial` | Simple recursion (module) | `n * f(n - 1)` |

## Generated Code Examples

### Tail Recursion

```javascript
/**
 * @param {number} n
 * @param {number} [acc=0]
 * @returns {number}
 */
export const sum = (n, acc = 0) => {
  if (n <= 0) return acc;
  return sum(n - 1, acc + n);
};
```

### List Fold

```javascript
/**
 * @param {number[]} items
 * @returns {number}
 */
export const listSum = (items) => {
  return items.reduce((acc, item) => acc + item, 0);
};
```

### Memoized Fibonacci

```javascript
/** @type {Map<number, number>} */
const fibMemo = new Map();

/**
 * @param {number} n
 * @returns {number}
 */
export const fib = (n) => {
  if (n <= 0) return 0;
  if (n === 1) return 1;
  if (fibMemo.has(n)) {
    return /** @type {*} */ (fibMemo.get(n));
  }
  const result = fib(n - 1) + fib(n - 2);
  fibMemo.set(n, result);
  return result;
};
```

### Facts → `@typedef`

```javascript
/**
 * @typedef {Object} PersonFact
 * @property {string} arg1
 * @property {string} arg2
 */
```

## Inheritance

```
annotated_js_target : typescript_target
    ::  clojurescript_target : clojure_target
```

Do not fork the TypeScript codegen. Override only annotation emission.
Advanced recursion multifile hooks (`compile_tail_pattern/9`, etc.) call
the TypeScript clauses and then `ts_to_annotated_js/2`.

## Type-checking

`tsc` is a **checker**, not a compiler. The command is always `--noEmit`:

```bash
npx tsc --checkJs --noEmit --allowJs generated.js
```

There is no transpile step. Generated `.js` is the source of truth.

If you run that command from a checkout that already has a root `tsconfig.json`, TypeScript 5.8+/7 reports `TS5112` (config present but unused when files are listed). Add `--ignoreConfig`, or run `tsc` from the directory that holds the generated file.

## See Also

- [TYPESCRIPT_TARGET.md](TYPESCRIPT_TARGET.md) — base target this inherits
- [BINDING_MATRIX.md](BINDING_MATRIX.md) — binding coverage (AnnotatedJS row)
- [RECURSION_PATTERN_THEORY.md](RECURSION_PATTERN_THEORY.md) — pattern taxonomy
- [src/unifyweaver/targets/annotated_js_target.pl](../src/unifyweaver/targets/annotated_js_target.pl)
