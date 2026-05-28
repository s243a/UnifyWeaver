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

1. **Single-clause only** — predicates with `try_me_else` /
   `retry_me_else` / `trust_me` in the bytecode (multi-clause + WAM
   indexing) are rejected. They stay on the bytecode path where the
   choice-point machinery handles backtracking.
2. **Supported instructions only** — the supported set is
   `get_constant`, `get_variable`, `get_value`, `get_structure`,
   `get_list`, `unify_variable`, `unify_value`, `unify_constant`,
   `put_constant`, `put_variable`, `put_value`, `put_structure`,
   `set_constant`, `set_variable`, `set_value`, `allocate`,
   `deallocate`, `builtin_call`, `proceed`, `fail`. Anything else
   (`call`, `execute`, `jump`, `cut_ite`, etc.) makes the gate fail
   silently and the predicate falls back to the bytecode path.

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

- M1: single-clause deterministic predicates over simple
  register/arithmetic ops.
- M2 (this release): pattern matching — `get_structure`, `get_list`,
  `unify_variable`, `unify_value`, `unify_constant`. Read-mode
  (matching against a bound compound from the caller) and write-mode
  (allocating a fresh compound on the arena) both supported, mirroring
  the bytecode `@step` cases. `get_list` read-mode currently returns
  the same failure sentinel the bytecode interpreter does — once the
  interpreter gains ground-list read support, this path follows.
- Future: multi-clause via "lowered clause-1 + bytecode clause-2+"
  (mirrors the F# emit-mode pattern), `call`/`execute` lowered to
  direct `call i1 @<other_pred>`, indirect dispatch through a
  caller-supplied `%WamState*` to avoid the per-call state alloc.

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
