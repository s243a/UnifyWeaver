# TypeScript Target

Compile Prolog predicates to TypeScript for type-safe JavaScript integration.

## Overview

| Feature | Status |
|---------|--------|
| Facts | ✅ Typed arrays + interfaces |
| Recursion | ✅ Multiple patterns |
| Modules | ✅ Multi-predicate compilation |
| Runtime Selection | ✅ Node/Deno/Bun/Browser |

## Quick Start

```prolog
?- use_module('src/unifyweaver/targets/typescript_target').

?- compile_module(
       [pred(sum, 2, tail_recursion),
        pred(factorial, 1, factorial)],
       [module_name('PrologMath')],
       Code),
   write_typescript_module(Code, 'PrologMath.ts').
```

```bash
npx tsc PrologMath.ts
node PrologMath.js
```

## API Reference

### `target_info/1`
Returns target metadata:
```prolog
target_info(Info).
% Info.name = "TypeScript"
% Info.family = javascript
% Info.features = [types, generics, async, modules]
```

### `compile_recursion/3`
Compile recursive predicates:
```prolog
compile_recursion(sum/2, [pattern(tail_recursion)], Code).
compile_recursion(listSum/2, [pattern(list_fold)], Code).
compile_recursion(fib/2, [pattern(linear_recursion)], Code).
```

### `compile_module/3`
Multiple predicates in one module:
```prolog
compile_module(
    [pred(sum, 2, tail_recursion),
     pred(factorial, 1, factorial),
     pred(fib, 2, linear_recursion),
     pred(listSum, 2, list_fold)],
    [module_name('PrologMath')],
    Code).
```

## Recursion Patterns

| Pattern | Description | TypeScript |
|---------|-------------|------------|
| `tail_recursion` | O(1) stack | `(n, acc) => ...` |
| `list_fold` | Array reduce | `items.reduce()` |
| `linear_recursion` | Memoized fib | `Map<number, number>` |
| `factorial` | Simple recursion | `n * f(n-1)` |

## Generated Code Examples

### Tail Recursion
```typescript
export const sum = (n: number, acc: number = 0): number => {
  if (n <= 0) return acc;
  return sum(n - 1, acc + n);
};
```

### List Fold
```typescript
export const listSum = (items: number[]): number => {
  return items.reduce((acc, item) => acc + item, 0);
};
```

### Memoized Fibonacci
```typescript
const fibMemo = new Map<number, number>();

export const fib = (n: number): number => {
  if (n <= 0) return 0;
  if (n === 1) return 1;
  
  if (fibMemo.has(n)) return fibMemo.get(n)!;
  
  const result = fib(n - 1) + fib(n - 2);
  fibMemo.set(n, result);
  return result;
};
```

## Runtime Selection

The `js_glue.pl` module provides variant selection:

```prolog
?- js_runtime_choice([typescript, secure], Runtime).
Runtime = deno.

?- js_runtime_choice([npm], Runtime).
Runtime = node.

?- js_runtime_choice([dom], Runtime).
Runtime = browser.
```

### Supported Runtimes

| Runtime | Features |
|---------|----------|
| Node.js | npm, filesystem, streaming |
| Deno | TypeScript native, permissions |
| Bun | Fast, npm compatible |
| Browser | DOM, fetch, localStorage |

## Dependencies

```bash
# Node.js (most common)
npm install -g typescript

# Or Deno (no install needed for TS)
deno --version

# Or Bun
curl -fsSL https://bun.sh/install | bash
```

## Bindings

Prolog builtins are mapped to TypeScript/JavaScript stdlib functions in
[`bindings/typescript_bindings.pl`](../src/unifyweaver/bindings/typescript_bindings.pl)
via `declare_binding/6`, and queried through the convenience predicates
`ts_binding/5` and `ts_binding_import/2`. Call `init_typescript_bindings/0`
(invoked by `init_typescript_target/0`) to register them.

**179 bindings across 13 categories:**

| Category | Examples | Notes |
|----------|----------|-------|
| Core built-ins | `typeof/2`, `to_number/2`, `parse_int/3`, `is_nan/2` | Coercions, type checks |
| String | `string_length/2`, `string_slice/4`, `string_replace_all/4`, `string_split/3` | `pattern(method_call)` / `property_access` |
| Math | `abs/2`, `sqrt/2`, `pow/3`, `sin/2`, `log10/2`, `math_pi/1` | `Math.*` |
| Array | `array_map/3`, `array_filter/3`, `array_reduce/4`, `array_push/3` | Pure vs. `effect(mutation)` split |
| Object | `object_keys/2`, `object_entries/2`, `object_freeze/2` | `Object.*` |
| JSON | `json_parse/2`, `json_stringify/2`, `json_stringify_pretty/3` | `parse` is `partial`/`effect(throws)` |
| Console / I/O | `console_log/1`, `console_error/1`, `console_table/1` | `effect(io)` |
| Node `fs` | `read_file_sync/2`, `write_file/3`, `exists_sync/2` | `import('fs')`, `async` on promises |
| Node `path` | `path_join/3`, `path_resolve/2`, `path_dirname/2` | `import('path')` |
| Promise / async | `promise_all/2`, `promise_race/2`, `promise_resolve/2` | `Promise.*` |
| Date | `date_now/1`, `date_to_iso_string/2`, `date_get_time/2` | `effect(clock)` on constructors |
| **Collections (Map/Set)** | `map_get/3`, `map_set/4`, `map_has/3`, `set_add/3`, `set_has/3` | Constructors/reads pure; `set/add/delete/clear` are `effect(mutation)` |
| **Number formatting** | `number_to_fixed/3`, `number_to_precision/3`, `number_is_integer/2`, `number_parse_int/3` | Instance/`Number.*` methods, all pure |

Node built-in bindings carry an `import(...)` option; the target's
`collect_binding_import/1` / `get_collected_imports/1` API accumulates and
dedups these for the generated module's import header.

```prolog
?- init_typescript_bindings.
?- ts_binding(map_get/3, Target, In, Out, Opts).
Target = '.get', In = [map, any], Out = [any], Opts = [pure, deterministic, total, pattern(method_call)].

?- ts_binding_import(read_file_sync/2, Import).
Import = fs.
```

## Test Coverage

The plunit suite
[`tests/core/test_typescript_target.pl`](../tests/core/test_typescript_target.pl)
(`swipl -q -g test_typescript_target -t halt tests/core/test_typescript_target.pl`)
covers the public compilation surface:

- **`target_info/1`** — `family: javascript`, `.ts` extension, and the four
  declared recursion patterns.
- **Facts** — binary and unary predicates produce the typed interface, fact
  array scaffold, and `query`/`is` helpers (`compile_facts/3`).
- **Recursion** — each pattern from `target_info/1`: `tail_recursion`,
  `list_fold`, `linear_recursion` (memoized), and `transitive_closure`. Note:
  `transitive_closure` has no dedicated `compile_recursion/3` generator and
  currently routes through the `tail_recursion` fallback; genuine
  transitive-closure lowering is exercised via the general-recursive multifile
  hook (`compile_general_recursive_pattern/6`), which emits a recursive
  traversal.
- **Modules** — multi-predicate `compile_module/3` mixing tail/fold/linear/factorial.
- **Bindings** — registration of stdlib + the new Map/Set/Number bindings, and
  `import` lookup / collection round-trip.

The lower-level native clause-body lowering is covered separately by
[`tests/core/test_typescript_native_lowering.pl`](../tests/core/test_typescript_native_lowering.pl).

## See Also

- [js_glue.pl](../src/unifyweaver/glue/js_glue.pl) - Runtime selection
- [target_registry.pl](../src/unifyweaver/core/target_registry.pl) - Family definitions
- [BINDING_MATRIX.md](BINDING_MATRIX.md) - Cross-target binding scoreboard
