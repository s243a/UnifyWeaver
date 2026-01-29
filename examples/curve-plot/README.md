# Curve Plotting Demo

This example demonstrates the UnifyWeaver cross-target pipeline for mathematical curve visualization:

**Pipeline:** Prolog -> LLVM IR -> WebAssembly -> TypeScript -> Chart.js

**Fully Generated:** All output files (LLVM IR, TypeScript, HTML) are generated from `curve_module.pl`.

## Overview

The demo shows how to:
1. Define mathematical curves in Prolog (`curve_module.pl`)
2. Compile curve evaluation to WASM via LLVM
3. Generate TypeScript bindings for the WASM module
4. Generate complete HTML/CSS/JavaScript web application
5. Visualize curves using Chart.js in the browser

## Files

| File | Description |
|------|-------------|
| `curve_module.pl` | Prolog module - source of truth for all generation |
| `index.html` | (Generated) Complete web app with Chart.js visualization |
| `curve_wasm.ts` | (Generated) TypeScript bindings for WASM |
| `curve_plot.ll` | (Generated) LLVM IR for curve evaluation |
| `curve_plot.wasm` | (Generated) WebAssembly module (after compilation) |

## Supported Curve Types

- **Linear**: `y = mx + b`
- **Quadratic**: `y = ax^2 + bx + c`
- **Sine**: `y = amp * sin(freq * x + phase)`
- **Cosine**: `y = amp * cos(freq * x + phase)`
- **Exponential**: `y = scale * e^(base * x)`

## Quick Start

### Generate All Files

```bash
cd examples/curve-plot
swipl -g "consult('curve_module.pl'), curve_module:generate_all" -t halt
```

This generates:
- `curve_plot.ll` - LLVM IR for curve evaluation
- `curve_wasm.ts` - TypeScript bindings
- `index.html` - Complete web application

### Browser Demo (JavaScript fallback)

Simply open `index.html` in a browser. The app uses a JavaScript fallback if WASM isn't compiled.

```bash
python3 -m http.server 8080
# Open http://localhost:8080
```

### Full WASM Pipeline (optional)

1. **Generate files (if not already done):**
   ```bash
   swipl -g "consult('curve_module.pl'), curve_module:generate_all" -t halt
   ```

2. **Build WebAssembly module:**
   ```bash
   # Fix Prolog escaping
   sed -i 's/%%/%/g' curve_plot.ll

   # Compile to WASM object
   llc -march=wasm32 -filetype=obj curve_plot.ll -o curve_plot.o

   # Link to WASM module
   wasm-ld --no-entry --export-all curve_plot.o -o curve_plot.wasm
   ```

3. **Serve and view:**
   ```bash
   python3 -m http.server 8080
   # Open http://localhost:8080 in browser
   ```

## Prolog API

### Curve Evaluation

```prolog
% Evaluate a curve at a point
?- evaluate_curve(sine(1, 1, 0), 3.14159, Y).
Y = 0.0.

% Generate sample points
?- curve_points(quadratic(1, 0, 0), -5, 5, Points).
Points = [[-5, 25], [-4, 16], ...].
```

### Code Generation

```prolog
% Generate WASM-compatible LLVM IR
?- generate_curve_wasm([
       curve_def(wave, sine),
       curve_def(parabola, quadratic)
   ], LLVMCode).

% Generate TypeScript bindings
?- generate_ts_chart_bindings([
       curve_def(wave, sine)
   ], TSCode).
```

## Architecture

```
┌─────────────────┐
│  Prolog         │
│  curve_module.pl│
└────────┬────────┘
         │ generate_curve_wasm/2
         ▼
┌─────────────────┐
│  LLVM IR        │
│  curve_plot.ll  │
└────────┬────────┘
         │ llc + wasm-ld
         ▼
┌─────────────────┐
│  WebAssembly    │
│  curve_plot.wasm│
└────────┬────────┘
         │ TypeScript bindings
         ▼
┌─────────────────┐
│  Chart.js       │
│  Browser UI     │
└─────────────────┘
```

## Custom Chart Component

This demo also introduces the `custom_chart` component type for TypeScript. Define charts declaratively in Prolog:

```prolog
declare_component(source, my_chart, custom_chart, [
    chart_type(line),
    title("Mathematical Functions"),
    x_axis([label("X"), min(-10), max(10)]),
    y_axis([label("Y")]),
    datasets([
        dataset([label("sin(x)"), color("#00d4ff")]),
        dataset([label("cos(x)"), color("#7c3aed")])
    ])
]).
```

## Dependencies

- SWI-Prolog (for code generation)
- LLVM toolchain (`llc`, `wasm-ld`) - for full WASM compilation
- Modern browser with WebAssembly support

## See Also

- [wasm-graph example](../wasm-graph/) - Graph visualization with Cytoscape.js
- [LLVM Target documentation](../../docs/LLVM_TARGET.md)
- [Cross-Target Glue guide](../../docs/guides/cross-target-glue.md)

## Reference

Original hand-written index.html preserved in git history:
https://github.com/s243a/UnifyWeaver/blob/8de1dfd/examples/curve-plot/index.html
