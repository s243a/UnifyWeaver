# Python Variant Targets

UnifyWeaver supports multiple Python compilation variants for different performance and deployment needs.

## Overview

| Variant | License | NumPy | Use Case | Compilation |
|---------|---------|-------|----------|-------------|
| **python** | - | ✅ | Standard Python | Interpreted |
| **numba** | BSD 2-Clause | ✅ | Numerical computing | JIT → LLVM |
| **cython** | Apache 2.0 | ✅ | C extensions | AOT → C |
| **nuitka** | Apache 2.0 | ✅ | Deployment | AOT → C |
| **codon** | BSL 1.1* | ⚠️ | High-performance | AOT → LLVM |
| **mypyc** | MIT | ✅ | Type-checked | AOT → C |
| **pyodide** | MPL 2.0 | ✅ | Browser/client-side | WASM |

\* Codon has commercial restrictions; becomes Apache 2.0 after 3 years.

## Quick Start

```prolog
% Load a variant target
?- use_module('src/unifyweaver/targets/python_numba_target').
?- init_numba_target.

% Compile with Numba JIT
?- compile_numba_function(factorial/1, [cache(true)], Code).
```

---

## Numba Target

Numba compiles Python to LLVM at runtime for near-native speed on numerical code.

### Module: `python_numba_target`

```prolog
% JIT-compiled function
compile_numba_function(+Pred/Arity, +Options, -Code)

% Vectorized ufunc
compile_numba_vectorized(+Pred/Arity, +Options, -Code)

% Parallel execution
compile_numba_parallel(+Pred/Arity, +Options, -Code)
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `nopython` | true/false | true | Use @njit (faster, no Python objects) |
| `cache` | true/false | false | Cache compiled code |
| `parallel` | true/false | false | Enable prange parallelism |
| `fastmath` | true/false | false | Relax IEEE 754 compliance |

### Example

```prolog
?- compile_numba_function(matrix_mult/3, [
       parallel(true),
       cache(true)
   ], Code).
```

Generated:
```python
from numba import njit, prange
import numpy as np

@njit(parallel=True, cache=True)
def matrix_mult(a, b, result):
    ...
```

### Custom Component

```prolog
declare_component(source, fast_sum, custom_numba, [
    code("    return np.sum(input)"),
    decorator(njit),
    parallel(true)
]).
```

---

## Cython Target

Cython generates C extensions with static typing for maximum performance.

### Module: `python_cython_target`

```prolog
% Typed function
compile_cython_function(+Pred/Arity, +Options, -Code)

% Full module
compile_cython_module(+Functions, +Options, -Code)

% Build files
generate_setup_py(+ModuleName, -SetupCode)
generate_pyproject_toml(+ModuleName, -TomlCode)
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `mode` | cpdef/cdef/def | cpdef | Function visibility |
| `types` | [Type, ...] | - | Argument types |
| `return_type` | Type | object | Return type |
| `nogil` | true/false | false | Release GIL |
| `inline` | true/false | false | Inline function |

### Type Mappings

```prolog
cython_type(integer, 'long long').
cython_type(float, 'double').
cython_type(list(float), 'double[:]').
cython_type(array(float), 'np.ndarray[np.float64_t, ndim=2]').
```

### Build Pipeline

```bash
# Generate .pyx file
swipl -g "generate_code" -t halt my_module.pl > my_module.pyx

# Compile
python setup.py build_ext --inplace

# Or with pyproject.toml
pip install .
```

---

## Nuitka Target

Nuitka compiles entire Python programs to C for deployment.

### Module: `python_nuitka_target`

```prolog
% Standard module
compile_nuitka_module(+ModuleName, +Options, -Code)

% Web application
compile_nuitka_webapp(+AppName, +Options, -Code)

% Build script
generate_nuitka_build_script(+ModuleName, +Options, -Script)

% Config file
generate_nuitka_config(+Options, -Config)
```

### Build Options

| Option | Values | Description |
|--------|--------|-------------|
| `standalone` | true/false | Create standalone executable |
| `onefile` | true/false | Single file output |
| `plugins` | [flask, qt, ...] | Framework plugins |
| `windows` | true/false | Windows-specific options |

### Web Framework Support

```prolog
?- compile_nuitka_webapp(my_api, [
       framework(flask),
       routes([
           route('/', get, index),
           route('/api/data', post, handle_data)
       ])
   ], Code).
```

### Build Commands

```bash
# Module
nuitka --module my_module.py

# Standalone executable
nuitka --standalone my_app.py

# Single file distribution
nuitka --onefile my_app.py

# With Flask plugin
nuitka --standalone --enable-plugin=flask my_app.py
```

---

## Codon Target

Codon compiles a subset of Python directly to native code via LLVM.

### Module: `python_codon_target`

```prolog
% Typed function
compile_codon_function(+Pred/Arity, +Options, -Code)

% Full module
compile_codon_module(+ModuleName, +Options, -Code)

% Parallel execution
compile_codon_parallel(+Pred/Arity, +Options, -Code)
```

### Options

| Option | Values | Description |
|--------|--------|-------------|
| `types` | [Type, ...] | Argument types (required for best performance) |
| `return_type` | Type | Return type |
| `jit` | true/false | Use @codon.jit for hybrid mode |

### Parallel Execution

Codon supports `@par` decorator for GPU/OpenMP parallelism:

```python
@par(schedule='dynamic', num_threads=8)
for i in range(len(data)):
    result[i] = process(data[i])
```

### Build Commands

```bash
# Debug build
codon build my_program.py

# Release build
codon build -release my_program.py

# Run directly (JIT)
codon run my_program.py

# Shared library
codon build -release -lib my_module.py
```

### Limitations

- Subset of Python (no dynamic features)
- Limited library support
- Static typing required for best performance
- BSL license has commercial restrictions

---

## mypyc Target

mypyc compiles type-annotated Python to C extensions using the mypy type checker.

### Module: `python_mypyc_target`

```prolog
% Typed function
compile_mypyc_function(+Pred/Arity, +Options, -Code)

% Full module
compile_mypyc_module(+ModuleName, +Options, -Code)

% Typed class
compile_mypyc_class(+ClassName, +Options, -Code)

% Build files
generate_mypyc_build_command(+ModuleName, -Command)
generate_mypyc_setup_py(+ModuleName, -SetupCode)
```

### Type Mappings

```prolog
mypyc_type(integer, 'int').
mypyc_type(list(float), 'List[float]').
mypyc_type(dict(string, integer), 'Dict[str, int]').
mypyc_type(optional(string), 'Optional[str]').
mypyc_type(callable([integer], boolean), 'Callable[[int], bool]').
```

### Class Generation

```prolog
?- compile_mypyc_class(person, [
       fields([
           field(name, string),
           field(age, integer)
       ]),
       methods([
           method(greet, [], string)
       ])
   ], Code).
```

### Build Pipeline

```bash
# Type check first
mypy --strict my_module.py

# Compile
mypyc my_module.py

# Or via setup.py
python setup.py build_ext --inplace
```

---

## Pyodide Target

Pyodide runs CPython in the browser via WebAssembly. **No server required** - completely client-side.

### Module: `python_pyodide_target`

```prolog
% Browser-compatible function
compile_pyodide_function(+Pred/Arity, +Options, -Code)

% Full module with exports
compile_pyodide_module(+ModuleName, +Options, -Code)

% JavaScript loader
generate_pyodide_loader(+Options, -JSCode)

% Complete HTML page
generate_pyodide_html(+Title, +Options, -HTML)

% Web Worker for background execution
generate_pyodide_worker(+Options, -WorkerCode)
```

### Available Packages

Pyodide includes many scientific Python packages:

```prolog
pyodide_packages([
    numpy, scipy, pandas, matplotlib, scikit_learn,
    sympy, networkx, pillow, opencv_python, statsmodels
]).
```

### Security Advantages

| Aspect | Pyodide | Server-side Python |
|--------|---------|-------------------|
| Code injection risk | ✅ None (browser sandbox) | ⚠️ Requires careful handling |
| Filesystem access | ✅ None | ⚠️ Full access |
| Network access | ✅ Restricted (CORS) | ⚠️ Full access |
| Resource limits | ✅ Browser enforced | ⚠️ Must configure |

### Example: Matrix Operations in Browser

```prolog
?- compile_pyodide_module(matrix_ops, [
       packages([numpy]),
       exports([inverse, eigenvalues])
   ], Code).
```

Generated Python:
```python
import numpy as np

def inverse(matrix):
    return np.linalg.inv(matrix)

def eigenvalues(matrix):
    return np.linalg.eigvals(matrix)

EXPORTS = {
    'inverse': inverse,
    'eigenvalues': eigenvalues
}
```

### HTML Generation

```prolog
?- generate_pyodide_html('Matrix Calculator', [
       packages([numpy]),
       chart(true)
   ], HTML).
```

Generates a complete HTML page with:
- Pyodide loader
- Chart.js integration
- Dark theme UI
- Python execution environment

### JavaScript Integration

```javascript
// Using the generated PyodideRunner class
const pyodide = new PyodideRunner();
await pyodide.init();

// Run NumPy operations directly
const result = await pyodide.numpy('np.linalg.inv([[1,2],[3,4]])');
console.log(result);  // [[-2, 1], [1.5, -0.5]]
```

### Custom Component

```prolog
declare_component(source, matrix_inverse, custom_pyodide, [
    code("    return np.linalg.inv(input)"),
    packages([numpy]),
    js_wrapper(true)
]).
```

### Performance Notes

| Operation | Pyodide | Native Python |
|-----------|---------|---------------|
| NumPy matrix ops | ~3-5x slower | Baseline |
| Pure Python loops | ~10x slower | Baseline |
| First load | ~5-10 seconds | Instant |

Pyodide is best for:
- Interactive demos
- Client-side data processing
- Educational applications
- Security-sensitive contexts

---

## Choosing a Variant

### Performance Comparison

| Workload | Recommended | Why |
|----------|-------------|-----|
| Numerical loops | Numba | JIT optimizes NumPy operations |
| C interop | Cython | Direct C/C++ integration |
| Deployment | Nuitka | Full Python, single executable |
| Maximum speed | Codon | Native LLVM, near C++ |
| Type safety | mypyc | mypy ecosystem integration |
| Browser/client-side | Pyodide | No server, browser sandbox |

### Decision Tree

1. **Runs in browser?** → Pyodide (only option for client-side)
2. **Need NumPy performance?** → Numba
3. **Building C extensions?** → Cython
4. **Distributing executables?** → Nuitka
5. **Maximum performance, limited Python?** → Codon
6. **Want type checking + compilation?** → mypyc

---

## Custom Components

Each variant has a corresponding custom component type:

| Component Type | Variant | Description |
|---------------|---------|-------------|
| `custom_python` | Standard | Raw Python code |
| `custom_numba` | Numba | JIT-compiled code |
| `custom_cython` | Cython | C extension code |
| `custom_nuitka` | Nuitka | Deployment-ready code |
| `custom_codon` | Codon | Native-compiled code |
| `custom_mypyc` | mypyc | Type-annotated code |
| `custom_pyodide` | Pyodide | Browser/WASM code |

### Example

```prolog
% Numba component for fast matrix operations
declare_component(source, matrix_ops, custom_numba, [
    code("    return np.dot(input, input.T)"),
    decorator(njit),
    cache(true)
]).

% Cython component for C-level speed
declare_component(source, c_ops, custom_cython, [
    code("    return arg0 * arg1"),
    mode(cpdef),
    types([double, double]),
    return_type(double),
    nogil(true)
]).
```

---

## See Also

- [PYTHON_TARGET.md](./PYTHON_TARGET.md) - Base Python target
- [LLVM_TARGET.md](./LLVM_TARGET.md) - LLVM compilation
- [Cross-Target Glue Book](../education/book-07-cross-target-glue/) - FFI examples
