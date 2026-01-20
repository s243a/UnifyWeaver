# Skill: TypeScript Target

Generate TypeScript code from Prolog specifications with type-safe interfaces and runtime targeting.

## When to Use

- User asks "how do I compile Prolog to TypeScript?"
- User wants type-safe JavaScript generation
- User needs to target Node.js, Deno, Bun, or browser
- User wants Express.js service generation
- User asks about TypeScript bindings for stdlib functions

## Overview

The TypeScript target compiles Prolog predicates to TypeScript code:

```prolog
:- use_module('src/unifyweaver/targets/typescript_target').

?- compile_predicate(my_pred/2, [type(facts)], Code).
```

**Output:** `.ts` files with typed interfaces, query functions, and exports.

## Supported Runtimes

| Runtime | Environment | Use Case |
|---------|-------------|----------|
| `node` | Node.js | Backend services, CLI tools |
| `deno` | Deno | Secure, TypeScript-first runtime |
| `bun` | Bun | Fast JavaScript runtime |
| `browser` | Browser | Web applications |

Runtime is auto-detected or can be specified.

## Target Features

| Feature | Description |
|---------|-------------|
| `types` | TypeScript type annotations |
| `generics` | Generic type parameters |
| `async` | Async/await support |
| `modules` | ES module imports/exports |
| `interfaces` | Interface generation for data |

## Compilation Types

### Facts to Typed Arrays

```prolog
% Define facts
parent(john, mary).
parent(john, bob).
parent(mary, alice).

% Compile to TypeScript
?- compile_facts(parent, 2, Code).
```

**Output:**
```typescript
export interface ParentFact {
  arg1: string;
  arg2: string;
}

export const parentFacts: ParentFact[] = [
  { arg1: "john", arg2: "mary" },
  { arg1: "john", arg2: "bob" },
  { arg1: "mary", arg2: "alice" }
];

export const queryParent = (criteria: Partial<ParentFact>): ParentFact[] => {
  return parentFacts.filter(fact => {
    return Object.entries(criteria).every(([key, value]) =>
      (fact as any)[key] === value
    );
  });
};

export const isParent = (...args: string[]): boolean => {
  const [arg1, arg2] = args;
  return parentFacts.some(f => f.arg1 === arg1 && f.arg2 === arg2);
};
```

### Recursion Patterns

| Pattern | Description | TypeScript Strategy |
|---------|-------------|---------------------|
| `tail_recursion` | Accumulator-based | While loop for TCO |
| `linear_recursion` | Fibonacci-style | Memoization |
| `list_fold` | Reduce over list | Array.reduce |
| `transitive_closure` | Graph traversal | Iteration with Set |

```prolog
?- compile_recursion(factorial/2, [pattern(tail_recursion)], Code).
```

**Output:**
```typescript
export const factorial = (n: number, acc: number = 1): number => {
  if (n <= 1) return acc;
  return factorial(n - 1, acc * n);
};

// Strict version for guaranteed TCO
export const factorialStrict = (n: number, acc: number = 1): number => {
  let current = n;
  let result = acc;
  while (current > 1) {
    result *= current;
    current--;
  }
  return result;
};
```

### Module Compilation

```prolog
?- compile_module([
       pred(factorial, 2, tail_recursion),
       pred(fibonacci, 2, linear_recursion),
       pred(sum_list, 2, list_fold)
   ], [module_name('MathUtils')], Code).
```

## TypeScript Bindings

The binding system maps Prolog predicates to TypeScript/JavaScript standard library:

### Core Built-ins

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `typeof/2` | `typeof` | Runtime type |
| `instanceof/3` | `instanceof` | Prototype check |
| `to_boolean/2` | `Boolean()` | Boolean coercion |
| `to_number/2` | `Number()` | Number coercion |
| `to_string/2` | `String()` | String coercion |
| `parse_int/3` | `parseInt()` | Parse integer |
| `parse_float/2` | `parseFloat()` | Parse float |
| `is_nan/2` | `isNaN()` | NaN check |

### String Operations

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `string_length/2` | `.length` | String length |
| `string_includes/3` | `.includes()` | Substring test |
| `string_starts_with/3` | `.startsWith()` | Prefix test |
| `string_ends_with/3` | `.endsWith()` | Suffix test |
| `string_upper/2` | `.toUpperCase()` | Uppercase |
| `string_lower/2` | `.toLowerCase()` | Lowercase |
| `string_trim/2` | `.trim()` | Trim whitespace |
| `string_split/3` | `.split()` | Split to array |
| `string_replace/4` | `.replace()` | Replace substring |
| `string_slice/4` | `.slice()` | Extract substring |

### Math Operations

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `abs/2` | `Math.abs()` | Absolute value |
| `sqrt/2` | `Math.sqrt()` | Square root |
| `pow/3` | `Math.pow()` | Power |
| `ceil/2` | `Math.ceil()` | Ceiling |
| `floor/2` | `Math.floor()` | Floor |
| `round/2` | `Math.round()` | Round |
| `sin/2`, `cos/2`, `tan/2` | `Math.sin()`, etc. | Trigonometric |
| `log/2`, `log10/2` | `Math.log()`, etc. | Logarithmic |
| `random/1` | `Math.random()` | Random number |
| `math_pi/1` | `Math.PI` | Pi constant |

### Array Operations

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `array_length/2` | `.length` | Array length |
| `array_map/3` | `.map()` | Transform elements |
| `array_filter/3` | `.filter()` | Filter elements |
| `array_reduce/4` | `.reduce()` | Fold/accumulate |
| `array_find/3` | `.find()` | Find element |
| `array_includes/3` | `.includes()` | Element test |
| `array_push/3` | `.push()` | Add element |
| `array_slice/4` | `.slice()` | Extract subarray |
| `array_concat/3` | `.concat()` | Join arrays |
| `array_join/3` | `.join()` | Join to string |

### Node.js File System

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `read_file_sync/2` | `fs.readFileSync()` | Read file (sync) |
| `write_file_sync/3` | `fs.writeFileSync()` | Write file (sync) |
| `exists_sync/2` | `fs.existsSync()` | File exists |
| `mkdir_sync/2` | `fs.mkdirSync()` | Create directory |
| `readdir_sync/2` | `fs.readdirSync()` | List directory |
| `read_file/2` | `fs.promises.readFile()` | Read file (async) |
| `write_file/3` | `fs.promises.writeFile()` | Write file (async) |

### Node.js Path

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `path_join/3` | `path.join()` | Join paths |
| `path_resolve/2` | `path.resolve()` | Resolve path |
| `path_dirname/2` | `path.dirname()` | Directory name |
| `path_basename/2` | `path.basename()` | File name |
| `path_extname/2` | `path.extname()` | Extension |

### Promise/Async

| Prolog | TypeScript | Description |
|--------|------------|-------------|
| `promise_resolve/2` | `Promise.resolve()` | Resolved promise |
| `promise_reject/2` | `Promise.reject()` | Rejected promise |
| `promise_all/2` | `Promise.all()` | All promises |
| `promise_race/2` | `Promise.race()` | First promise |

## Express Service Generation

Generate Express.js REST services from specifications:

```prolog
?- compile_express_service(
       service(api, [
           port(3000),
           endpoints([
               endpoint('/users', get, getUsers),
               endpoint('/users', post, createUser),
               endpoint('/users/:id', get, getUserById)
           ]),
           middleware([cors, json])
       ]),
       Code
   ).
```

**Output:**
```typescript
import express, { Request, Response } from "express";
import cors from "cors";

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Endpoints
app.get("/users", async (req: Request, res: Response) => {
  try {
    const result = await getUsers(req);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({ success: false, error: String(error) });
  }
});

app.post("/users", async (req: Request, res: Response) => {
  try {
    const result = await createUser(req);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({ success: false, error: String(error) });
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`api service running on port ${PORT}`);
});

export default app;
```

## HTTP Client Generation

Generate typed API clients:

```prolog
?- compile_http_client(
       client(myapi, [
           base_url('https://api.example.com'),
           endpoints([
               endpoint('/users', get, fetchUsers),
               endpoint('/users', post, createUser)
           ])
       ]),
       Code
   ).
```

## Import Management

Bindings automatically track required imports:

```prolog
% Collect imports as you use bindings
?- collect_binding_import(fs).
?- collect_binding_import(path).

% Get all collected imports
?- get_collected_imports(Imports).
% Imports = [fs, path]
```

Generated code includes proper import statements:

```typescript
import * as fs from 'fs';
import * as path from 'path';
```

## File Output

```prolog
?- compile_predicate(my_pred/2, [type(facts)], Code),
   write_typescript_module(Code, 'output/my_pred.ts').
```

**Console output:**
```
TypeScript module written to: output/my_pred.ts
Compile with: npx tsc output/my_pred.ts
```

## Common Workflows

### Generate Data Module

```prolog
% 1. Load facts
?- [my_data].

% 2. Compile to TypeScript
?- compile_facts(my_data, 3, Code).

% 3. Write output
?- write_typescript_module(Code, 'src/data.ts').

% 4. Run TypeScript compiler
% $ npx tsc src/data.ts
```

### Generate REST Service

```prolog
% 1. Define service
?- compile_express_service(
       service(myapp, [
           port(8080),
           endpoints([...]),
           middleware([cors, json])
       ]),
       Code
   ).

% 2. Write output
?- write_typescript_module(Code, 'src/server.ts').

% 3. Run
% $ npx ts-node src/server.ts
```

## Related

**Parent Skill:**
- `skill_gui_generation.md` - GUI generation sub-master

**Sibling Skills:**
- `skill_app_generation.md` - Full app generation

**Code:**
- `src/unifyweaver/targets/typescript_target.pl` - Main compiler
- `src/unifyweaver/bindings/typescript_bindings.pl` - Stdlib bindings
- `src/unifyweaver/core/binding_registry.pl` - Binding registry
