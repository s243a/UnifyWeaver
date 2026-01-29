# Pyodide Matrix Visualization

Interactive matrix operations using NumPy running in your browser via WebAssembly.

## Features

- **Matrix Inverse** - Compute A⁻¹
- **Eigenvalues/Eigenvectors** - Spectral decomposition
- **SVD** - Singular Value Decomposition
- **Matrix Transformation** - Visualize how matrices transform the unit circle
- **Linear Regression** - Least squares fitting with NumPy

## Quick Start

Just open `index.html` in a browser. No server required!

```bash
# Option 1: Direct file open
firefox index.html
# or
google-chrome index.html

# Option 2: Local server (for development)
python3 -m http.server 8080
# Then open http://localhost:8080
```

## Architecture

```
Browser
├── TypeScript/JavaScript (UI, Charts)
├── Pyodide (Python → WebAssembly)
│   └── NumPy (matrix operations)
└── Chart.js (visualization)
```

All computation runs **client-side** in the browser sandbox. No data is sent to any server.

## Security

This example demonstrates the security advantages of Pyodide:

| Aspect | This Demo | Server-side Python |
|--------|-----------|-------------------|
| Code injection | ✅ Impossible | ⚠️ Risk if using eval() |
| Data privacy | ✅ Never leaves browser | ⚠️ Sent to server |
| Filesystem access | ✅ None | ⚠️ Full access |

## Generated Files

All files in this example are generated from `matrix_module.pl`:

```bash
swipl -g "consult('matrix_module.pl'), generate_all" -t halt
```

This creates:
- `matrix_lib.py` - Python module with NumPy operations
- `matrix_wasm.ts` - TypeScript bindings
- `index.html` - Complete web app with Chart.js visualization

The original hand-written index.html (before generation was added) can be found at:
https://github.com/s243a/UnifyWeaver/blob/92ea1592a4205ddba1f1c94953abde8eca2913f4/examples/pyodide-matrix/index.html

## Matrix Presets

| Preset | Matrix | Effect |
|--------|--------|--------|
| Rotation 45° | [0.707, -0.707; 0.707, 0.707] | Rotates vectors |
| Scale 2x | [2, 0; 0, 2] | Doubles size |
| Shear | [1, 1; 0, 1] | Horizontal shear |
| Reflection | [-1, 0; 0, 1] | Flip across y-axis |
| Singular | [1, 2; 2, 4] | Non-invertible |
| Identity | [1, 0; 0, 1] | No change |

## How It Works

1. **Pyodide loads** - CPython compiled to WebAssembly
2. **NumPy loads** - Full NumPy package in browser
3. **Python code runs** - Matrix operations execute in WASM
4. **Results return to JS** - JSON serialization bridges Python↔JavaScript
5. **Chart.js visualizes** - Interactive charts update

## Performance

First load takes 5-10 seconds (downloading Pyodide + NumPy).
After that, computations are fast (~3-5x slower than native Python).

## See Also

- [Pyodide Documentation](https://pyodide.org/)
- [UnifyWeaver Python Variants](../../../docs/PYTHON_VARIANTS.md)
- [Curve Plot Example](../curve-plot/) - LLVM→WASM approach
