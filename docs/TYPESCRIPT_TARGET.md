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

## See Also

- [js_glue.pl](../src/unifyweaver/glue/js_glue.pl) - Runtime selection
- [target_registry.pl](../src/unifyweaver/core/target_registry.pl) - Family definitions
