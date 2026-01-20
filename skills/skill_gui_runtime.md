# Skill: GUI Runtime

Sub-master skill for client-side code execution, including WebAssembly, browser Python, and reactive data binding.

## When to Use

- User asks "how do I run Python in the browser?"
- User wants WebAssembly for performance-critical code
- User needs reactive data binding for UI components
- User asks about LLVM compilation or FFI

## Overview

UnifyWeaver supports multiple client-side runtimes:

| Technology | Purpose | Performance |
|------------|---------|-------------|
| **Pyodide** | Python in browser | NumPy, pandas, SciPy |
| **WebAssembly** | Native performance | LLVM-compiled code |
| **Reactive Binding** | UI state management | React hooks |

## Individual Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `skill_data_binding.md` | React hooks, providers | Reactive UI state |
| `skill_webassembly.md` | LLVM to WASM | Performance-critical code |
| `skill_browser_python.md` | Pyodide integration | NumPy/pandas in browser |

## Data Binding

Generate React hooks and providers for Prolog data:

```prolog
:- use_module('src/unifyweaver/glue/data_binding_generator').

% Define data source
data_source(sales_data, [
    predicate(sales_record/4),
    fields([date, product, quantity, amount]),
    refresh_interval(5000)
]).

% Generate React hook
generate_binding_hook(line_chart, Hook).
```

**Features:**
- One-way and two-way bindings
- WebSocket real-time sync
- Computed/derived sources
- Polling with configurable intervals
- TypeScript type generation

See `skill_data_binding.md` for details.

## WebAssembly

Compile Prolog to LLVM IR, then to WebAssembly:

```prolog
:- use_module('src/unifyweaver/targets/llvm_target').

% Compile with musttail for guaranteed tail-call optimization
compile_tail_recursion_llvm(factorial/2, [export(true)], LLVMCode).

% Generate WASM module
compile_wasm_module([factorial], [], WASMCode).

% Generate TypeScript bindings
generate_ts_bindings([factorial], TSCode).
```

**Features:**
- O(1) stack space via musttail
- FFI for Go, Rust, C
- TypeScript/JavaScript bindings
- String support in WASM

See `skill_webassembly.md` for details.

## Browser Python (Pyodide)

Run full Python with scientific libraries in browser:

```prolog
:- use_module('src/unifyweaver/targets/python_pyodide_target').

% Generate Pyodide module
compile_pyodide_module(data_analysis, [
    packages([numpy, pandas, scipy]),
    exports([analyze, transform])
], PythonCode).

% Generate complete HTML page
generate_pyodide_html('Data Analysis', [
    packages([numpy]),
    chart(true)
], HTML).
```

**Features:**
- NumPy, SciPy, pandas in browser
- No server required
- Web Worker for background execution
- Chart.js integration

See `skill_browser_python.md` for details.

## Technology Selection Guide

### Use Data Binding When:
- Building interactive UIs
- Need real-time updates
- Working with React/Vue
- Data changes frequently

### Use WebAssembly When:
- Performance is critical
- Compute-intensive algorithms
- Need native-speed execution
- Working with existing C/Rust code

### Use Pyodide When:
- Need scientific Python (NumPy, pandas)
- Client-side data analysis
- Educational/demo applications
- Rapid prototyping with Python

## Common Workflows

### Real-Time Dashboard

```prolog
% 1. Define data source with polling
data_source(metrics, [
    predicate(metric/3),
    fields([timestamp, name, value]),
    refresh_interval(1000)
]).

% 2. Create binding for chart
binding(metrics_chart, metrics, [
    x_axis(timestamp),
    y_axis(value),
    series(name)
]).

% 3. Generate React hook
generate_binding_hook(metrics_chart, Hook).

% 4. Generate WebSocket sync for real-time
generate_websocket_sync(metrics, SyncCode).
```

### High-Performance Computation

```prolog
% 1. Compile algorithm to LLVM
compile_tail_recursion_llvm(compute/2, [export(true)], LLVMCode).

% 2. Generate WASM module
compile_wasm_module([compute], [], WASMCode).

% 3. Generate TypeScript bindings
generate_ts_bindings([compute], TSCode).

% 4. Build WASM
build_wasm_module('compute.ll', 'compute', Commands).
```

### In-Browser Data Science

```prolog
% 1. Define Python analysis module
compile_pyodide_module(analysis, [
    packages([numpy, pandas, matplotlib]),
    exports([process_data, visualize])
], PythonCode).

% 2. Generate Web Worker for background execution
generate_pyodide_worker([packages([numpy, pandas])], WorkerCode).

% 3. Generate complete HTML demo
generate_pyodide_html('Data Analysis Demo', [
    packages([numpy, pandas]),
    python_module(PythonCode),
    chart(true)
], HTML).
```

## Integration Between Runtimes

### WASM + React Binding

```prolog
% Use WASM for heavy computation
compile_wasm_module([matrix_multiply], [], WASMCode).

% Bind result to React component
data_source(computation_result, [
    fields([input, output, time])
]).

binding(result_display, computation_result, [
    x_axis(input),
    y_axis(output)
]).
```

### Pyodide + Data Binding

```prolog
% Python processes data
compile_pyodide_module(processor, [...], PythonCode).

% Results flow to React via binding
data_source(processed_data, [
    fields([id, value, category])
]).

binding(data_chart, processed_data, [
    x_axis(category),
    y_axis(value)
]).
```

## Related

**Parent Skill:**
- `skill_gui_tools.md` - GUI master skill

**Individual Skills:**
- `skill_data_binding.md` - Reactive state, React hooks
- `skill_webassembly.md` - LLVM to WASM, FFI
- `skill_browser_python.md` - Pyodide, NumPy in browser

**Sibling Sub-Masters:**
- `skill_gui_generation.md` - App generation
- `skill_gui_design.md` - UI design

**Code:**
- `src/unifyweaver/glue/data_binding_generator.pl` - React binding
- `src/unifyweaver/targets/llvm_target.pl` - LLVM/WASM
- `src/unifyweaver/targets/python_pyodide_target.pl` - Pyodide
