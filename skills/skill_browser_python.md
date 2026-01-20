# Skill: Browser Python (Pyodide)

Run full Python with NumPy, pandas, and SciPy directly in the browser via WebAssembly.

## When to Use

- User asks "how do I run Python in the browser?"
- User needs NumPy/pandas for client-side data analysis
- User wants interactive data visualization without a server
- User needs educational/demo Python applications

## Quick Start

```prolog
:- use_module('src/unifyweaver/targets/python_pyodide_target').

% Generate a Python module for browser
compile_pyodide_module(analysis, [
    packages([numpy, pandas]),
    exports([analyze, transform])
], PythonCode).

% Generate complete HTML page
generate_pyodide_html('Data Analysis', [
    packages([numpy]),
    chart(true)
], HTML).
```

## What is Pyodide?

Pyodide compiles CPython to WebAssembly, enabling:
- Full Python 3 in browser
- NumPy, SciPy, pandas, matplotlib
- No server required
- Browser sandbox security
- Async JavaScript integration

## Available Packages

```prolog
pyodide_packages(Packages).
% [numpy, scipy, pandas, matplotlib, scikit_learn,
%  sympy, networkx, pillow, opencv_python, statsmodels]
```

## Compilation Functions

### Compile Function

```prolog
compile_pyodide_function(Pred/Arity, Options, Code).
```

**Options:**
- `packages(List)` - Required packages
- `async(true)` - Generate async function

```prolog
compile_pyodide_function(matrix_multiply/3, [
    packages([numpy]),
    async(false)
], Code).
```

**Output:**
```python
# Pyodide-compatible Python
# Runs in browser via WebAssembly
import numpy as np

def matrix_multiply(a, b):
    """Generated from Prolog predicate matrix_multiply/3.

    Runs in Pyodide (browser-based Python).
    """
    return np.matmul(a, b)
```

### Compile Module

```prolog
compile_pyodide_module(ModuleName, Options, Code).
```

**Options:**
- `predicates(List)` - Predicates to compile
- `packages(List)` - Required packages
- `exports(List)` - Functions to expose to JavaScript

```prolog
compile_pyodide_module(data_processor, [
    packages([numpy, pandas]),
    exports([load_data, process, summarize])
], Code).
```

**Output:**
```python
# Pyodide Module: data_processor
# Runs entirely in browser via WebAssembly
# No server required - secure client-side execution

import numpy as np
import pandas as pd

# ... compiled functions ...

# Registry for JavaScript access
EXPORTS = {
    'load_data': load_data,
    'process': process,
    'summarize': summarize
}

def call_function(name, *args):
    """Call exported function by name (for JS interop)."""
    if name in EXPORTS:
        return EXPORTS[name](*args)
    raise ValueError(f'Unknown function: {name}')
```

## JavaScript Integration

### Generate Loader

```prolog
generate_pyodide_loader(Options, JSCode).
```

**Output:**
```javascript
class PyodideRunner {
    constructor() {
        this.pyodide = null;
        this.ready = false;
    }

    async init() {
        if (this.ready) return;
        console.log('Loading Pyodide...');
        this.pyodide = await loadPyodide();
        console.log('Loading packages...');
        await this.pyodide.loadPackage(['numpy', 'pandas']);
        this.ready = true;
    }

    async runPython(code) {
        if (!this.ready) await this.init();
        return await this.pyodide.runPythonAsync(code);
    }

    async callFunction(pythonCode, funcName, ...args) {
        if (!this.ready) await this.init();
        await this.pyodide.runPythonAsync(pythonCode);
        const argsJson = JSON.stringify(args);
        const result = await this.pyodide.runPythonAsync(`
            import json
            args = json.loads('${argsJson}')
            result = ${funcName}(*args)
            json.dumps(result.tolist() if hasattr(result, 'tolist') else result)
        `);
        return JSON.parse(result);
    }

    async numpy(operation) {
        return await this.runPython(`
import numpy as np
result = ${operation}
result.tolist() if hasattr(result, 'tolist') else result
        `);
    }
}

const pyodide = new PyodideRunner();
```

## HTML Generation

### Generate Complete Page

```prolog
generate_pyodide_html(Title, Options, HTML).
```

**Options:**
- `packages(List)` - Packages to load
- `python_module(Code)` - Python code to include
- `chart(true)` - Include Chart.js

```prolog
generate_pyodide_html('NumPy Demo', [
    packages([numpy]),
    python_module(MyPythonCode),
    chart(true)
], HTML).
```

**Output includes:**
- Pyodide CDN script
- Chart.js (if enabled)
- Styled dark theme
- Status indicator
- Output area
- Canvas for charts

## Web Worker

For background execution without blocking UI:

```prolog
generate_pyodide_worker(Options, WorkerCode).
```

**Output:**
```javascript
// Pyodide Web Worker
// Runs Python in background thread, won't block UI

importScripts('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');

let pyodide = null;

async function init() {
    pyodide = await loadPyodide();
    await pyodide.loadPackage(['numpy']);
    postMessage({ type: 'ready' });
}

onmessage = async (event) => {
    const { id, code } = event.data;
    try {
        if (!pyodide) await init();
        const result = await pyodide.runPythonAsync(code);
        postMessage({ id, type: 'result', data: result });
    } catch (error) {
        postMessage({ id, type: 'error', error: error.message });
    }
};

init();
```

**Usage in main thread:**
```javascript
const worker = new Worker('pyodide-worker.js');

worker.onmessage = (event) => {
    if (event.data.type === 'ready') {
        console.log('Pyodide ready in worker');
    } else if (event.data.type === 'result') {
        console.log('Result:', event.data.data);
    }
};

worker.postMessage({
    id: 1,
    code: 'import numpy as np; np.array([1,2,3]).tolist()'
});
```

## Common Workflows

### Interactive Data Visualization

```prolog
% 1. Create Python analysis module
compile_pyodide_module(viz, [
    packages([numpy, matplotlib]),
    exports([plot_histogram, plot_scatter])
], PythonCode).

% 2. Generate HTML page with Chart.js
generate_pyodide_html('Interactive Visualization', [
    packages([numpy]),
    python_module(PythonCode),
    chart(true)
], HTML).
```

### Client-Side Data Processing

```prolog
% 1. Create processing module
compile_pyodide_module(processor, [
    packages([pandas, numpy]),
    exports([load_csv, clean_data, aggregate])
], PythonCode).

% 2. Generate loader for integration
generate_pyodide_loader([packages([pandas, numpy])], LoaderJS).
```

### Educational Demo

```prolog
% 1. Create math demonstration
compile_pyodide_function(matrix_ops/2, [
    packages([numpy]),
    async(false)
], MathCode).

% 2. Generate complete demo page
generate_pyodide_html('Linear Algebra Demo', [
    packages([numpy]),
    python_module(MathCode),
    chart(false)
], HTML).
```

### Heavy Computation in Worker

```prolog
% 1. Create computation module
compile_pyodide_module(compute, [
    packages([numpy, scipy]),
    exports([fft, optimize, integrate])
], ComputeCode).

% 2. Generate Web Worker
generate_pyodide_worker([packages([numpy, scipy])], WorkerCode).
```

## Performance Considerations

### Initial Load Time

Pyodide + NumPy: ~10-15 seconds on first load (cached afterward)

**Mitigation:**
- Show loading indicator
- Pre-load during idle time
- Use Web Worker to not block UI

### Memory Usage

Browser WASM has memory limits:
- ~2GB on 64-bit browsers
- Consider data chunking for large datasets

### Computation Speed

NumPy in Pyodide: ~60-80% of native Python speed
- Fast for vectorized operations
- Use Web Worker for CPU-intensive tasks

## Use Cases

| Use Case | Packages | Example |
|----------|----------|---------|
| Data analysis | pandas, numpy | CSV processing |
| Scientific computing | scipy, numpy | Signal processing |
| Machine learning | scikit-learn | Simple models |
| Data visualization | matplotlib | Charts, plots |
| Symbolic math | sympy | Equation solving |
| Image processing | pillow, opencv | Filters, transforms |
| Graph algorithms | networkx | Path finding |

## Related

**Parent Skill:**
- `skill_gui_runtime.md` - GUI runtime sub-master

**Sibling Skills:**
- `skill_data_binding.md` - React binding
- `skill_webassembly.md` - LLVM/WASM

**Code:**
- `src/unifyweaver/targets/python_pyodide_target.pl` - Pyodide target
- `src/unifyweaver/targets/python_target.pl` - Base Python target
