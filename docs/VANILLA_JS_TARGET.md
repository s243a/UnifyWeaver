<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2025 John William Creighton (@s243a) -->

# Vanilla JS Target

Compile Prolog predicates to **plain, untyped JavaScript** — valid ES-module JS
that runs on stock Node.js or in the browser with **no build step and no runtime
dependency**.

## Overview

The Vanilla JS target is a *variant* of the [TypeScript target](TYPESCRIPT_TARGET.md).
It inherits all of the TypeScript target's code generation (native clause
lowering, recursion patterns, expression translation, and every advanced
recursion dispatch hook) and overrides exactly one thing: it strips
TypeScript's compile-time-only type syntax so the result is plain JavaScript.

```
vanilla_js_target : typescript_target  ::  clojurescript_target : clojure_target
```

| Feature | Status |
|---------|--------|
| Facts | ✅ Object arrays (inherited, types stripped) |
| Recursion | ✅ tail / linear / list_fold / transitive_closure |
| Modules | ✅ Multi-predicate compilation |
| Runtime | ✅ Node / browser / Deno / Bun (plain ES modules) |
| Build step | ❌ None required — runs as-is |
| Runtime dependency | ❌ None |

## What gets stripped

The single centralized rewrite predicate `vanilla_js_type_strip/2` (mirroring
`clojurescript_target:clojurescript_interop_rewrite/2`) removes:

| TypeScript syntax | Vanilla JS result |
|-------------------|-------------------|
| `: number`, `: string[]`, `: Set<string>` (parameter/variable annotations) | removed |
| `): number =>`, `): string[] {` (return-type annotations) | `) =>`, `) {` |
| `interface Foo { ... }` (declaration blocks) | dropped entirely |
| `new Map<number, number>()`, `<T, R>` (generic type arguments) | `new Map()`, removed |
| `x.get(n)!` (non-null assertions) | `x.get(n)` |
| `(fact as any)` (type assertions) | `(fact)` |

## Quick Start

```prolog
?- use_module('src/unifyweaver/targets/vanilla_js_target').

% A single predicate (native clause lowering)
?- compile_predicate_to_vanilla_js(double/2, [], Code).

% Recursion patterns
?- compile_recursion(sum/2,   [pattern(tail_recursion)], Code).
?- compile_recursion(total/2, [pattern(list_fold)],      Code).
?- compile_recursion(fib/2,   [pattern(linear_recursion)], Code).

% A module of several predicates
?- compile_module(
       [pred(sum, 2, tail_recursion), pred(factorial, 1, factorial)],
       [module_name('PrologMath')],
       Code),
   write_vanilla_js_module(Code, 'PrologMath.mjs').
```

```bash
# No compile step — run directly.
node PrologMath.mjs
```

> The emitted code uses `export`/`import` (ES modules). Run it as `.mjs`, or as
> `.js` inside a package with `{"type": "module"}`. Native-clause-lowering output
> (plain `function` + `console.log` CLI, no `import`/`export`) also runs as a
> plain CommonJS `.js` script.

## API Reference

| Predicate | Description |
|-----------|-------------|
| `target_info/1` | Target metadata (see below) |
| `compile_predicate/3` | Registry dispatch entry point |
| `compile_predicate_to_vanilla_js/3` | Compile one predicate to vanilla JS |
| `compile_facts/3` | Facts → object array + query helpers |
| `compile_recursion/3` | Compile a recursion pattern |
| `compile_module/3` | Compile several predicates into one module |
| `vanilla_js_type_strip/2` | The centralized TS → JS type-stripping rewrite |
| `write_vanilla_js_module/2` | Write generated code to a file |
| `init_vanilla_js_target/0` | Initialize (delegates to TypeScript base) |
| `clear_binding_imports/0`, `collect_binding_import/1`, `get_collected_imports/1` | Binding hooks (delegated to the TypeScript base) |

### `target_info/1`

```prolog
target_info(Info).
% Info.name              = "VanillaJS"
% Info.family            = javascript
% Info.file_extension    = ".js"
% Info.runtime           = auto
% Info.features          = [plain, modules, async]
% Info.recursion_patterns = [tail_recursion, linear_recursion, list_fold, transitive_closure]
% Info.compile_command   = "node"
```

## Transitive Closure

Transitive closure is produced by the recursive compiler's TypeScript template
path, then post-processed with `vanilla_js_type_strip/2`:

```prolog
?- recursive_compiler:compile_transitive_closure(
       typescript, path, 2, edge, [input(embedded)], TsCode),
   vanilla_js_target:vanilla_js_type_strip(TsCode, JsCode).
```

The result is a BFS over a `Map`/`Set`-backed relation, in plain JavaScript.

## Generated Code Examples

### Tail Recursion

```javascript
export const sum = (n, acc = 0) => {
  if (n <= 0) return acc;
  return sum(n - 1, acc + n);
};
```

### Memoized Fibonacci (linear recursion)

```javascript
const fibMemo = new Map();

export const fib = (n) => {
  if (n <= 0) return 0;
  if (n === 1) return 1;
  if (fibMemo.has(n)) {
    return fibMemo.get(n);
  }
  const result = fib(n - 1) + fib(n - 2);
  fibMemo.set(n, result);
  return result;
};
```

### Transitive Closure (BFS)

```javascript
const baseRelation = new Map();

const findAll = (start) => {
  const results = [];
  const visited = new Set([start]);
  const queue = [start];
  while (queue.length > 0) {
    const current = queue.shift();
    for (const next of baseRelation.get(current) || []) {
      if (!visited.has(next)) {
        visited.add(next);
        queue.push(next);
        results.push(next);
      }
    }
  }
  return results;
};
```

## Testing

```bash
swipl -q -g test_vanilla_js_target -t halt tests/core/test_vanilla_js_target.pl
```

## See Also

- [TYPESCRIPT_TARGET.md](TYPESCRIPT_TARGET.md) — the base target this one inherits from
- [target_registry.pl](../src/unifyweaver/core/target_registry.pl) — family definitions
- `docs/proposals/integration_patches/JS-2_INTEGRATION_PATCH.md` — coordinator wiring
