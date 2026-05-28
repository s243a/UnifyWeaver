# LLVM Target

The LLVM target compiles Prolog predicates directly to LLVM IR for native code generation with guaranteed optimizations.

## Overview

| Feature | Status |
|---------|--------|
| Fact Compilation | ✅ |
| Tail Recursion | ✅ `musttail` |
| Linear Recursion | ✅ Memo table |
| Mutual Recursion | ✅ `musttail` |
| Transitive Closure | ✅ BFS worklist |
| C ABI Export | ✅ `dllexport` |

## Quick Start

```prolog
?- use_module('src/unifyweaver/targets/llvm_target').
?- init_llvm_target.

% Generate LLVM IR for tail recursion
?- compile_tail_recursion_llvm(sum/2, [], Code),
   write_llvm_program(Code, 'sum.ll').
```

```bash
# Fix escaping and compile
sed -i 's/%%/%/g' sum.ll
llc -filetype=obj sum.ll -o sum.o
clang sum.o -o sum
./sum
```

## API Reference

### `compile_predicate_to_llvm/3`
```prolog
compile_predicate_to_llvm(+Pred/Arity, +Options, -LLVMCode)
```

### `compile_tail_recursion_llvm/3`
Guaranteed O(1) stack via `musttail`:
```prolog
compile_tail_recursion_llvm(+Pred/Arity, +Options, -LLVMCode)
% Options: export(true) generates extern "C" wrapper
```

### `compile_linear_recursion_llvm/3`
With static memoization table:
```prolog
compile_linear_recursion_llvm(+Pred/Arity, +Options, -LLVMCode)
```

### `compile_mutual_recursion_llvm/3`
For is_even/is_odd patterns:
```prolog
compile_mutual_recursion_llvm(+Predicates, +Options, -LLVMCode)
```

### `compile_transitive_closure_llvm/3`
BFS graph reachability:
```prolog
compile_transitive_closure_llvm(+Pred/Arity, +Options, -LLVMCode)
```

## Generated Code Patterns

### Tail Recursion → `musttail`
```llvm
define i64 @sum(i64 %n, i64 %acc) {
  ...
  %result = musttail call i64 @sum(i64 %n1, i64 %acc1)
  ret i64 %result
}
```

### Transitive Closure → BFS
```llvm
@edges = internal global [100 x i64] zeroinitializer
@visited = internal global [1000 x i1] zeroinitializer
@queue = internal global [1000 x i64] zeroinitializer
```

## Dependencies

- LLVM toolchain (`llc`, `clang`)

```bash
# Ubuntu/Debian
sudo apt install llvm clang
```

---

## Phase 2: C ABI Integration

### `compile_shared_library_llvm/3`
Compile multiple functions to shared library:
```prolog
compile_shared_library_llvm(
    [func(sum, 2, tail_recursion), func(factorial, 1, factorial)],
    [library_name(prolog_math)],
    Code).
```

### `generate_c_header/2`
Generate C header file:
```prolog
generate_c_header(Functions, HeaderCode).
% → prolog_math.h with int64_t declarations
```

### `generate_cgo_bindings/2`
Generate Go cgo package:
```prolog
generate_cgo_bindings(Functions, GoCode).
% → package prologmath with C.function() calls
```

### `generate_rust_ffi/2`
Generate Rust FFI module:
```prolog
generate_rust_ffi(Functions, RustCode).
% → mod ffi { extern "C" { ... } }
```

### Building
```bash
# Compile to shared library
llc -filetype=obj -relocation-model=pic lib.ll -o lib.o
clang -shared lib.o -o libprolog_math.so

# Verify exports
nm -D libprolog_math.so | grep "T sum"
```

---

## Phase 5: WASM String Support

Generate WASM modules with string handling for graph visualization.

### `compile_wasm_string_module/3`
Compile with string and edge storage support:
```prolog
compile_wasm_string_module(
    [func(ancestor, 2, transitive_closure)],
    [module_name(family_graph)],
    Code).
```

### `generate_ts_string_bindings/2`
Generate TypeScript bindings with `GraphWasm` class:
```prolog
generate_ts_string_bindings(Functions, TSCode).
```

Generated API:
```typescript
const graph = await GraphWasm.load('family_graph.wasm');
graph.addEdge('tom', 'bob');
graph.getEdges(); // [['tom', 'bob']]
```

### Example
See [examples/wasm-graph/](../examples/wasm-graph/) for complete Cytoscape.js demo.

---

## Phase 6: Curve Plotting (Mathematical Visualization)

Generate WASM modules for mathematical curve evaluation and Chart.js visualization.

### `generate_curve_wasm/2`
Compile mathematical curves to WASM with float operations:
```prolog
generate_curve_wasm([
    curve_def(wave, sine),
    curve_def(parabola, quadratic),
    curve_def(growth, exponential)
], LLVMCode).
```

### Supported Curve Types

| Type | Equation | Parameters |
|------|----------|------------|
| `linear` | y = mx + b | m, b |
| `quadratic` | y = ax² + bx + c | a, b, c |
| `sine` | y = amp × sin(freq × x + phase) | amp, freq, phase |
| `cosine` | y = amp × cos(freq × x + phase) | amp, freq, phase |
| `exponential` | y = scale × e^(base × x) | base, scale |

### `generate_ts_chart_bindings/2`
Generate TypeScript bindings with Chart.js integration:
```prolog
generate_ts_chart_bindings(CurveSpecs, TSCode).
```

Generated API:
```typescript
const curves = await CurveWasm.load('curve_plot.wasm');
curves.set_wave_params(1.0, 2.0, 0.0);  // amp, freq, phase
curves.generate_wave(-10, 10, 200);      // xMin, xMax, numPoints
const chartData = curves.toChartData('sin(2x)', '#00d4ff');
```

### Pipeline Architecture

```
Prolog (curve_module.pl)
    ↓ generate_curve_wasm/2
LLVM IR (curve_plot.ll)
    ↓ llc -march=wasm32 + wasm-ld
WebAssembly (curve_plot.wasm)
    ↓ TypeScript bindings
Chart.js (browser visualization)
```

### Example
See [examples/curve-plot/](../examples/curve-plot/) for complete Chart.js demo with:
- Interactive parameter controls
- Multiple curve overlays
- Preset mathematical functions (sin, cos, parabola)

### Custom Chart Component

The `custom_chart` component type allows declarative Chart.js configuration:

```prolog
declare_component(source, my_chart, custom_chart, [
    chart_type(line),
    title("Mathematical Functions"),
    x_axis([label("X"), type(linear)]),
    y_axis([label("Y")]),
    datasets([
        dataset([label("sin(x)"), color("#00d4ff")])
    ])
]).
```

See [custom_chart.pl](../src/unifyweaver/targets/typescript_runtime/custom_chart.pl) for implementation.

## Lowered Emit Mode (WAM-LLVM hybrid target)

The WAM-LLVM hybrid target (`wam_llvm_target`, distinct from the legacy
`llvm_target` documented above) compiles every predicate to WAM
bytecode and runs it through the `@step` switch dispatcher by default.
With `emit_mode(functions)` or `emit_mode(mixed([P/A, ...]))`, eligible
predicates are instead compiled to standalone LLVM functions whose
bodies inline the WAM instruction sequence as straight-line basic
blocks. No bytecode array, no `@step` switch, no `@run_loop`
trampoline for those predicates.

```prolog
:- use_module('src/unifyweaver/targets/wam_llvm_target',
              [write_wam_llvm_project/3]).

:- dynamic add1/2.
add1(X, R) :- R is X + 1.

:- initialization((
    write_wam_llvm_project(
        [user:add1/2],
        [ module_name('add1_mod'),
          target_triple('x86_64-pc-linux-gnu'),
          emit_mode(functions)    %% <-- opt in to lowered emission
        ],
        '/tmp/add1.ll'),
    halt
)).
```

The generated module defines `@lowered_add1_2(%Value %a1, %Value %a2)`
whose body is `entry → pc_0 → pc_1 → ... → lowered_succeed`, one block
per WAM instruction.

### Eligibility gate

Mirrors `wam_fsharp_lowered_emitter.pl` / `wam_rust_lowered_emitter.pl`:

1. **Clause-1 body must be deterministic** — `wam_llvm_lowerable/3`
   strips any `try_me_else` prefix and the leading `switch_on_constant`
   indexing, then takes the body through the first `proceed`/`fail`
   as "clause 1". That body must contain no choice-point instructions
   (`try_me_else` / `retry_me_else` / `trust_me`).
2. **Supported instructions only** — the supported set is
   `get_constant`, `get_variable`, `get_value`, `get_structure`,
   `get_list`, `unify_variable`, `unify_value`, `unify_constant`,
   `put_constant`, `put_variable`, `put_value`, `put_structure`,
   `set_constant`, `set_variable`, `set_value`, `allocate`,
   `deallocate`, `builtin_call`, `call`, `execute`, `proceed`,
   `fail`. Anything else (`jump`, `cut_ite`, etc.) makes the gate
   fail silently and the predicate falls back to the bytecode path.
3. **Closure-closed under `call`/`execute`** — `call <pred>/<N>` and
   `execute <pred>/<N>` instructions are lowered as direct
   `call i1 @lowered_<callee>_<N>(%WamState* %vm)` so the callee
   symbol must exist. Before classification, M4 runs a fixpoint
   closure analysis on the input predicate list: a predicate joins
   the lowered closure iff every `call`/`execute` target in its
   clause-1 body is also in the closure. Anything left outside the
   closure falls back to the bytecode path even if its instruction
   set is otherwise supported. (Future: cross-module closure when
   the linker can resolve symbols across compile units.)

The lowerability check returns a *shape* the pipeline acts on:

- `single_clause` — bytecode contains no choice-point instrs anywhere;
  the lowered fn IS the whole predicate. Emitted as a `native` record
  plus a thin `@<pred>` wrapper that just delegates to
  `@lowered_<pred>_<arity>`.
- `multi_clause_c1` — clause 1 is lowerable but the bytecode contains
  `try_me_else` / `retry_me_else` / `trust_me` for additional clauses.
  Emitted as a `hybrid` record: the lowered clause-1 fast path AND the
  full bytecode (all clauses) ship in the module, and `@<pred>` is a
  dispatcher that tries the fast path first; on failure it falls back
  to running the full bytecode through `@run_loop`.

Predicates that fail the gate emit a `<pred>/<arity>: WAM fallback`
log line instead of `<pred>/<arity>: lowered LLVM emission`, and the
generated module still works — `emit_mode(functions)` is "try to
lower, fall back to bytecode if we can't", not "fail compilation if
some predicate isn't lowerable".

### Calling convention

The lowered function has the *same* signature as the WAM-fallback
entry function — `define i1 @<name>(%Value %a1, %Value %a2, ...)`
— and allocates its own `%WamState` internally. External drivers
that invoke either form by name don't need to know which path the
predicate took.

### Status

- **M1**: single-clause deterministic predicates over simple
  register/arithmetic ops.
- **M2**: pattern matching — `get_structure`, `get_list`,
  `unify_variable`, `unify_value`, `unify_constant`. Read-mode
  (matching against a bound compound from the caller) and write-mode
  (allocating a fresh compound on the arena) both supported, mirroring
  the bytecode `@step` cases. `get_list` read-mode currently returns
  the same failure sentinel the bytecode interpreter does — once the
  interpreter gains ground-list read support, this path follows.
- **M3**: multi-clause via lowered clause-1 + bytecode
  fallback. Predicates whose clause 1 is deterministic + supported
  but whose full body has additional clauses with `try_me_else` /
  `retry_me_else` / `trust_me` lower as a `hybrid` record. The
  emitted module contains BOTH `@lowered_<pred>_<arity>` (the
  clause-1 fast path) AND `@<pred>` (a dispatcher). Single-clause
  predicates also gain a `@<pred>` wrapper around their lowered
  function so external callers can use the same name across emit
  modes.
- **M4 (this release): kernel signature refactor + `call`/`execute`
  with shared state + closure analysis**. The lowered kernel
  signature is now `define i1 @lowered_<pred>_<arity>(%WamState*
  %vm)` — no `%Value` parameters. Args live in the caller's `%vm`
  registers; the kernel reads them from there. Public-entry wrappers
  (`@<pred>(%Value %a1, ...)`) allocate state, copy args into
  registers, call the kernel, free state.
  Lowered-to-lowered calls now emit
  `call i1 @lowered_<callee>_<N>(%WamState* %vm)` for `call` and
  `musttail call i1 @lowered_<callee>_<N>(%WamState* %vm)` for
  `execute` — proper tail calls when the caller's last instruction
  is a tail-call. The hybrid dispatcher (M3) also now shares state
  between its fast and slow paths.
  Fixpoint closure analysis (`compute_lowered_closure/3`) computes
  the set of mutually-call-resolvable preds before classification,
  so the emitter can confidently emit direct calls knowing the
  symbols will resolve at link time.
- **M5**: growable trail, stack, and choice-point allocators.
  `wam_state_new` still mallocs the initial buffers (trail 65536
  entries, CPs 1024, stack 1024) but `wam_trail_binding`,
  `wam_trail_heap_binding`, the stack push helpers
  (`wam_push_unify_ctx`, `wam_push_write_ctx`, the `allocate` @step
  case, the lowered emitter's allocate), and the CP push sites
  (`try_me_else`, `begin_aggregate`, `wam_push_foreign_choice_point`)
  now route through `wam_<area>_ensure_capacity` helpers that double
  the buffer via `realloc` when at cap. Choice points hold trail
  marks as indices (not pointers), so trail / CP grow is transparent
  to backtrack.
- **M6 (this release): growable heap with WriteCtx fixup**.
  Closes the last fixed-size allocator. `wam_heap_push` now routes
  through `wam_heap_grow` on overflow, which doubles the heap buffer
  via `realloc` and runs `wam_fixup_writectx_after_heap_grow` — a
  one-pass stack walk that translates any WriteCtx entry's `data`
  pointer that falls in the old heap range to the equivalent offset
  in the new heap. Refs are unaffected because they store heap
  INDICES (`i32`), not pointers, so `@wam_heap_get` / `@wam_heap_set`
  / `@wam_deref_value` all work post-grow without changes. Pre-M6
  the heap aborted via `exit(2)` on overflow at ~22k iterations of a
  `make_list/2`-style heap-allocating predicate; post-M6 a 50k-iter
  stress test completes in ~10 ms while the heap grows from 65k →
  131k → 262k cells transparently.
- Future: trail-rollback between fast and slow paths in the hybrid
  dispatcher so clause-1 partial bindings don't leak into the slow
  path; branch-weight `!prof` metadata on the `@step` switch;
  cross-module closure when LTO is available.

### Tests

- [`tests/core/test_wam_llvm_lowered_emitter.pl`](../tests/core/test_wam_llvm_lowered_emitter.pl)
  — lowerability gate, IR structure, `llvm-as` validation, and
  `llc -O2 + clang -O2 + run` execution tests.

## See Also

- [llvm_target_design.md](./proposals/llvm_target_design.md) - Design doc
- [wasm_string_support.md](./proposals/wasm_string_support.md) - WASM string proposal
- [Cross-Target Glue Book](../education/book-07-cross-target-glue/) - FFI examples
- [GO_TARGET.md](./GO_TARGET.md) - Go target
- [RUST_TARGET.md](./RUST_TARGET.md) - Rust target
- [WAM_FSHARP_TARGET.md](./WAM_FSHARP_TARGET.md#emit-modes) - F# emit-mode reference
