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

## See Also

- [llvm_target_design.md](./proposals/llvm_target_design.md) - Design doc
- [wasm_string_support.md](./proposals/wasm_string_support.md) - WASM string proposal
- [Cross-Target Glue Book](../education/book-07-cross-target-glue/) - FFI examples
- [GO_TARGET.md](./GO_TARGET.md) - Go target
- [RUST_TARGET.md](./RUST_TARGET.md) - Rust target
