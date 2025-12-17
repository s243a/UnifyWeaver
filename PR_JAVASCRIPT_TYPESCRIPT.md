# feat: Add JavaScript Family and TypeScript Target

## Summary

Adds a complete JavaScript family to the target registry with TypeScript as the first fully-implemented target. Uses the Python variant pattern for runtime selection.

## New Features

### JavaScript Family (5 targets)
- `typescript` - Type-safe JavaScript
- `node` - Server-side JavaScript
- `deno` - Secure TypeScript runtime
- `bun` - Fast JavaScript runtime
- `browser` - DOM-based JavaScript

### TypeScript Target
- **target_info/1** - Metadata and capabilities
- **compile_facts/3** - Generates typed arrays + interfaces
- **compile_recursion/3** - Multiple patterns:
  - `tail_recursion` - Accumulator pattern
  - `list_fold` - Array.reduce()
  - `linear_recursion` - Memoized (Map<number, number>)
  - `factorial` - Simple recursion
- **compile_module/3** - Multiple predicates in one file

### Runtime Selection (js_glue.pl)
```prolog
js_runtime_choice([typescript, secure], deno).
js_runtime_choice([npm], node).
js_runtime_choice([dom], browser).
```

## Tests: 5/5 Pass
```
[PASS] target_info
[PASS] tail_recursion
[PASS] list_fold
[PASS] linear_recursion (fibonacci)
[PASS] compile_module
```

## Files Changed

### Implementation
```
src/unifyweaver/core/target_registry.pl    [MODIFIED] +64 lines
- JavaScript family (5 targets)
- target_module/2 linking
- compile_to_target/4 dispatch

src/unifyweaver/targets/typescript_target.pl [NEW] 275 lines
src/unifyweaver/glue/js_glue.pl              [NEW] 243 lines
tests/test_typescript_target.pl              [NEW] 83 lines
```

### Documentation
```
docs/TYPESCRIPT_TARGET.md                    [NEW]
README.md                                    [MODIFIED] TypeScript in Extended Targets

education/other-books/book-typescript-target/ [NEW]
- README.md
- 01_introduction.md
- 02_recursion.md
- 03_runtimes.md
```

## Architecture Notes

This follows the Python variant pattern from `dotnet_glue.pl`:
- `python_runtime_choice/2` → `js_runtime_choice/2`
- `ironpython_compatible/1` → `node_supports/1`, `deno_supports/1`, etc.

The `target_module/2` predicate enables unified dispatch:
```prolog
compile_to_target(typescript, ancestor/2, [], Code).
```

## Commits

| Commit | Description |
|--------|-------------|
| `27342ce` | feat: Add JavaScript family and TypeScript target |
| `9adc9b1` | docs: Add TypeScript target documentation |
| `eb2cb9b` | (education) docs: Add TypeScript education book |
