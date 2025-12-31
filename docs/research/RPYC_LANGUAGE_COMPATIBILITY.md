# RPyC Language Compatibility Matrix

**Status:** Research Document
**Date:** 2025-12-26
**Author:** Claude Code (Opus 4.5)

## Key Insight

RPyC is **Python-only**, but any language that can embed a real CPython runtime can use RPyC. The critical requirements are:

1. **Embed CPython** - Not a reimplementation (like Jython), but actual CPython
2. **RPyC available** - The rpyc module importable (see installation options below)
3. **Call Python functions** - Invoke RPyC's connect/call APIs
4. **Handle Python objects** - Pass RPyC proxy objects back to host language

## Installing RPyC Without pip

pip is convenient but **not required**. RPyC is pure Python with one dependency (plumbum), so you have options:

### Option 1: Wheel Files (Recommended for Embedded)

```bash
# Download wheels (no pip needed to use them)
wget https://files.pythonhosted.org/packages/.../rpyc-6.0.2-py3-none-any.whl
wget https://files.pythonhosted.org/packages/.../plumbum-1.9.0-py3-none-any.whl

# Extract to your embedded Python's site-packages
unzip rpyc-6.0.2-py3-none-any.whl -d /path/to/embedded/python/site-packages/
unzip plumbum-1.9.0-py3-none-any.whl -d /path/to/embedded/python/site-packages/
```

### Option 2: Vendoring (Bundle with Your App)

```
your_app/
├── vendor/
│   ├── rpyc/           # Copy from rpyc source
│   └── plumbum/        # Copy from plumbum source
└── main.py
```

```python
import sys
sys.path.insert(0, 'vendor')
import rpyc  # Works!
```

### Option 3: Source Installation

```bash
# No pip, just Python
tar xzf rpyc-6.0.2.tar.gz
cd rpyc-6.0.2
python setup.py install --prefix=/path/to/embedded/python
```

### Option 4: Single-File Bundling

For truly minimal deployments, tools like [PyInstaller](https://pyinstaller.org/) or [Nuitka](https://nuitka.net/) can bundle RPyC into a single executable or library.

### Dependency Note

RPyC depends on **plumbum** (shell command toolkit). For RPyC classic mode (which we use), plumbum is required but only uses its basic features. Both are pure Python - no C extensions.

## Compatibility Matrix

| Language | Bridge Technology | CPython? | pip? | RPyC Feasibility | Notes |
|----------|------------------|----------|------|------------------|-------|
| **Prolog (SWI)** | [Janus](https://www.swi-prolog.org/pldoc/man?section=janus) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | This project |
| **C#/F#** | [Python.NET](https://pythonnet.github.io/) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | .NET Core 9.0, math + NumPy |
| **C#** | [CSnakes](https://tonybaloney.github.io/posts/embedding-python-in-dot-net-with-csnakes.html) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | .NET 9, FromRedistributable |
| **Java** | [JPype](https://github.com/jpype-project/jpype) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | Java 11, math + NumPy |
| **Java** | [jpy](https://jpy.readthedocs.io/) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | Java 11, bi-directional |
| **Java** | [GraalPy](https://github.com/oracle/graalpython) | ⚠️ GraalVM | ⚠️ Limited | ⚠️ Medium | May have C extension issues |
| **Rust** | [PyO3](https://pyo3.rs/) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | PyO3 0.22, math + NumPy |
| **Ruby** | [PyCall.rb](https://github.com/red-data-tools/pycall.rb) | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | v1.5.2, math + NumPy |
| **Go** | [go-python3](https://github.com/DataDog/go-python3) | ✅ Yes | ✅ Yes | ⚠️ Medium | GIL management complex, archived 2021 |
| **Go** | [go-embed-python](https://github.com/kluctl/go-embed-python) | ✅ Yes | ✅ Yes | ⚠️ Medium | No CGO, uses subprocess |
| **Go** | Rust FFI Bridge | ✅ Yes | ✅ Yes | ✅ **Tested & Working** | Go → Rust cdylib → PyO3 → RPyC |
| **Lua** | [lupa](https://github.com/scoder/lupa) | ❌ No* | ❌ No | ❌ Low | Lua embeds in Python, not reverse |
| **Lua** | [lunatic-python](https://github.com/bastibe/lunatic-python) | ⚠️ Partial | ⚠️ Partial | ⚠️ Low | Bi-directional but limited |
| **Node.js** | python-shell | ❌ Subprocess | ✅ Yes | ⚠️ Low | IPC overhead defeats purpose |
| **JavaScript** | [GraalJS + GraalPy](https://www.graalvm.org/latest/reference-manual/polyglot-programming/) | ⚠️ GraalVM | ⚠️ Limited | ⚠️ Medium | Polyglot context |

*lupa embeds Lua in Python, not Python in Lua

## Tier 1: High Confidence (Should Work)

### Prolog (SWI-Prolog + Janus) ✅ TESTED

```prolog
% Already implemented and tested in this project
rpyc_connect(localhost, [security(unsecured), acknowledge_risk(true)], Proxy),
rpyc_import(Proxy, math, Math),
py_call(Math:sqrt(16), Result).
% Result = 4.0
```

### C# / F# (Python.NET)

```csharp
using Python.Runtime;

// Initialize Python
PythonEngine.Initialize();

using (Py.GIL())
{
    // Import RPyC
    dynamic rpyc = Py.Import("rpyc");

    // Connect to server
    dynamic conn = rpyc.classic.connect("localhost", 18812);

    // Use remote modules
    dynamic math = conn.modules.math;
    var result = math.sqrt(16);
    Console.WriteLine($"sqrt(16) = {result}");  // 4.0

    conn.close();
}
```

### Java (JPype)

```java
import org.jpype.*;

public class RPyCExample {
    public static void main(String[] args) {
        // Start JVM with Python
        JPype.startJVM();

        // Import RPyC
        PyObject rpyc = PyModule.import_("rpyc");
        PyObject classic = rpyc.getAttr("classic");

        // Connect
        PyObject conn = classic.invoke("connect", "localhost", 18812);

        // Use remote math
        PyObject modules = conn.getAttr("modules");
        PyObject math = modules.getAttr("math");
        PyObject result = math.invoke("sqrt", 16);

        System.out.println("sqrt(16) = " + result);  // 4.0

        conn.invoke("close");
        JPype.shutdownJVM();
    }
}
```

### Rust (PyO3)

```rust
use pyo3::prelude::*;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // Import rpyc
        let rpyc = py.import("rpyc")?;
        let classic = rpyc.getattr("classic")?;

        // Connect
        let conn = classic.call_method1("connect", ("localhost", 18812))?;

        // Access remote math
        let modules = conn.getattr("modules")?;
        let math = modules.getattr("math")?;
        let result: f64 = math.call_method1("sqrt", (16,))?.extract()?;

        println!("sqrt(16) = {}", result);  // 4.0

        conn.call_method0("close")?;
        Ok(())
    })
}
```

### Ruby (PyCall.rb)

```ruby
require 'pycall'

# Import rpyc
rpyc = PyCall.import_module('rpyc')

# Connect
conn = rpyc.classic.connect('localhost', 18812)

# Use remote math
math = conn.modules.math
result = math.sqrt(16)
puts "sqrt(16) = #{result}"  # 4.0

conn.close()
```

## Tier 2: Medium Confidence (Likely Works)

### Go (go-python3)

```go
package main

import (
    "fmt"
    python "github.com/DataDog/go-python3"
)

func main() {
    python.Py_Initialize()
    defer python.Py_Finalize()

    // Import rpyc
    rpyc := python.PyImport_ImportModule("rpyc")
    classic := rpyc.GetAttrString("classic")

    // Connect - Note: GIL management required
    args := python.PyTuple_New(2)
    python.PyTuple_SetItem(args, 0, python.PyUnicode_FromString("localhost"))
    python.PyTuple_SetItem(args, 1, python.PyLong_FromLong(18812))

    conn := classic.CallMethodObjArgs("connect", args)

    // Use remote math
    modules := conn.GetAttrString("modules")
    math := modules.GetAttrString("math")

    sqrtArgs := python.PyTuple_New(1)
    python.PyTuple_SetItem(sqrtArgs, 0, python.PyLong_FromLong(16))
    result := math.CallMethodObjArgs("sqrt", sqrtArgs)

    fmt.Printf("sqrt(16) = %v\n", python.PyFloat_AsDouble(result))

    conn.CallMethodObjArgs("close", nil)
}
```

### GraalVM Polyglot (Java + Python)

```java
import org.graalvm.polyglot.*;

public class GraalRPyCExample {
    public static void main(String[] args) {
        try (Context context = Context.newBuilder("python")
                .allowAllAccess(true)
                .build()) {

            // Execute Python code that uses RPyC
            Value result = context.eval("python", """
                import rpyc
                conn = rpyc.classic.connect('localhost', 18812)
                result = conn.modules.math.sqrt(16)
                conn.close()
                result
            """);

            System.out.println("sqrt(16) = " + result.asDouble());
        }
    }
}
```

## Tier 3: Low Confidence (Challenging)

### Lua

Lua bridges typically embed Lua in Python, not the reverse. To use RPyC from Lua, you'd need:

1. **lunatic-python** (limited bi-directional support)
2. Custom FFI to CPython (significant effort)
3. Subprocess approach (defeats purpose of live proxies)

### Node.js

Node.js options typically use subprocess communication:

```javascript
// This defeats the purpose of RPyC's live proxies
const { spawn } = require('child_process');
const python = spawn('python', ['-c', `
import rpyc
conn = rpyc.classic.connect('localhost', 18812)
print(conn.modules.math.sqrt(16))
`]);
```

For true integration, consider [edge-py](https://github.com/nicola/edge-py) but it has limitations.

## Architecture Decision

### When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Prolog + Python | Janus + RPyC (this project) |
| .NET + Python | Python.NET or CSnakes + RPyC |
| Java + Python | JPype or jpy + RPyC |
| Rust + Python | PyO3 + RPyC |
| Ruby + Python | PyCall.rb + RPyC |
| Go + Python | **Rust FFI bridge** (see below) |
| Node.js + Python | Rust FFI bridge or gRPC |
| Lua + Python | Rust FFI bridge or embed Lua in Python |
| Other FFI languages | Rust FFI bridge |

### Recommended: Rust FFI Bridge for Languages Without Mature CPython Embedding

For Go, Node.js, Lua, and other languages with C FFI but without stable CPython bindings,
use **Rust as a universal bridge layer**:

```
┌─────────────────────────────────────────────────────────┐
│              FFI-Capable Languages                       │
├──────┬──────┬──────┬──────┬──────┬──────────────────────┤
│  Go  │ Node │ Lua  │ PHP  │ Zig  │  Any C FFI lang...   │
│ (cgo)│(napi)│(ffi) │(ffi) │      │                      │
└──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──────────────────────┘
   └──────┴──────┴──┬───┴──────┘
                    ▼
         ┌──────────────────────┐
         │  Rust cdylib (.so)   │  ← Single bridge to maintain
         │  ┌────────────────┐  │
         │  │     PyO3       │  │  ← Tested & working
         │  │  ┌──────────┐  │  │
         │  │  │ CPython  │  │  │
         │  │  │  + RPyC  │  │  │
         │  │  └──────────┘  │  │
         │  └────────────────┘  │
         └──────────────────────┘
                    │
                    ▼ (TCP, live proxies)
         ┌──────────────────────┐
         │    RPyC Server       │
         │  NumPy, PyTorch,     │
         │  pandas, sklearn...  │
         └──────────────────────┘
```

**Why Rust FFI over direct bridges:**

| Aspect | Direct Go Bridges | Rust FFI Bridge |
|--------|-------------------|-----------------|
| GIL handling | Manual, error-prone | PyO3 handles it |
| Stability | go-python3 archived (2021) | PyO3 actively maintained |
| Maintenance | N bridges for N languages | One bridge for all |
| Live proxies | go-embed-python uses subprocess | Full RPyC proxy support |

**Why Rust FFI over gRPC:**

- **Live proxies**: RPyC proxies work across the bridge (gRPC serializes everything)
- **No schema files**: No `.proto` definitions needed
- **Direct access**: Call any Python module dynamically

### Alternative: gRPC for Stateless Services

For languages without good CPython embedding AND when live proxies aren't needed:

1. **Generate gRPC stubs** from Python services
2. **Polyglot support** out of the box
3. **Better tooling** for non-Python clients
4. **Trade-off**: No live proxies, requires proto definitions

Use gRPC when you only need request/response patterns, not interactive object manipulation.

## Implementation Priority

Based on ecosystem maturity and use cases:

1. **Prolog (Janus)** - ✅ Done
2. **C# (Python.NET/CSnakes)** - ✅ Done - .NET ecosystem
3. **Java (JPype/jpy)** - ✅ Done - Enterprise adoption
4. **Rust (PyO3)** - ✅ Done - Systems programming
5. **Ruby (PyCall.rb)** - ✅ Done - Rails/web development
6. **Go (Rust FFI)** - ✅ Done - Via Rust bridge for stable CPython access

## References

- [Python.NET Documentation](https://pythonnet.github.io/pythonnet/dotnet.html)
- [JPype User Guide](https://jpype.readthedocs.io/en/latest/userguide.html)
- [PyO3 User Guide](https://pyo3.rs/main/python-from-rust)
- [PyCall.rb GitHub](https://github.com/red-data-tools/pycall.rb)
- [GraalPy Interoperability](https://github.com/oracle/graalpython/blob/master/docs/user/Interoperability.md)
- [Cgo and Python - Datadog](https://www.datadoghq.com/blog/engineering/cgo-and-python/)
- [lupa PyPI](https://pypi.org/project/lupa/)

## Python Variant Compatibility

In addition to language bridges, RPyC works with various Python accelerators and compilers. Tests run on 2025-12-26 confirm:

| Variant | Version Tested | Status | Notes |
|---------|---------------|--------|-------|
| **CPython** | 3.8.10 | ✅ Works | Baseline - all RPyC features work |
| **Numba** | 0.58.1 | ✅ Works | JIT-compiled functions callable via RPyC |
| **Cython** | 3.2.3 | ✅ Works | Services run correctly (full .pyx compile is build step) |
| **mypyc** | (mypy 1.14.1) | ✅ Works | Type-annotated services work |
| **Nuitka** | 2.8.9 | ✅ Works | Service patterns verified |
| **Codon** | 0.19.4 | ⚠️ Partial | Native NumPy support; RPyC requires CPython bridge |
| **Pyodide** | N/A | ❌ Incompatible | Browser sandbox, no TCP sockets |

### Why These Work

All the working variants run on or with CPython:
- **Numba**: JIT compilation at runtime, runs on CPython
- **Cython**: Compiles to C extensions, loaded by CPython
- **mypyc**: Compiles typed code to C extensions for CPython
- **Nuitka**: Compiles Python to native code, includes CPython runtime
- **Codon**: Native compiler with Python interop; can call CPython via `from python import`

### Known Incompatible

- **Pyodide**: WebAssembly-based, runs in browser sandbox without real network access
- **Jython**: Python 2.7 on JVM, reimplementation (not CPython)
- **IronPython**: .NET CLR reimplementation (not CPython)

### Tested: Language Bridge Integration

These bridges have been tested with RPyC (2025-12-28):

| Language | Bridge | Status | Test Results |
|----------|--------|--------|--------------|
| **.NET** | Python.NET | ✅ Verified | .NET Core 9.0, math.sqrt + numpy.mean |
| **.NET** | CSnakes | ✅ Verified | .NET 9, FromRedistributable + venv |
| **Java** | JPype | ✅ Verified | Java 11, math.sqrt + numpy.mean |
| **Java** | jpy | ✅ Verified | Java 11, bi-directional ArrayList demo |
| **Rust** | PyO3 | ✅ Verified | PyO3 0.22, math.sqrt + numpy.mean |
| **Ruby** | PyCall.rb | ✅ Verified | v1.5.2, math.sqrt + numpy.mean |
| **Go** | Rust FFI | ✅ Verified | Go 1.18+ via Rust cdylib, math.sqrt + numpy.mean + math.pi |
| **Java** | GraalPy | ⏳ Untested | GraalVM-based, may have C extension issues |
| **Java** | JNI + CPython | ⏳ Untested | Manual integration via Java Native Interface |

**All 7 primary bridges now verified working.** See `examples/python-bridges/` for working examples.

### Test Suite

Run the variant compatibility tests:

```bash
python tests/rpyc_variants/test_rpyc_variants.py
```

Run the bridge integration tests:

```bash
# Start RPyC server first
python examples/rpyc-integration/rpyc_server.py &

# Run all bridge tests
python -m pytest tests/integration/python_bridges/ -v
```

## Prolog Glue Module

The `python_bridges_glue.pl` module provides automatic bridge detection and selection.

### Auto-Detection

```prolog
?- use_module('src/unifyweaver/glue/python_bridges_glue').

% Detect all available bridges
?- detect_all_bridges(Bridges).
% Bridges = [pythonnet, jpype, jpy]

% Check if a specific bridge is ready
?- check_bridge_ready(jpype, Status).
% Status = ready
% Status = missing_runtime('Java')
% Status = missing_package(jpype1)

% Get bridge requirements
?- bridge_requirements(jpype, Reqs).
% Reqs = [requirement(runtime, 'Java 11+'), ...]
```

### Auto-Selection with Fallback

```prolog
% Auto-select best bridge for target platform
?- auto_select_bridge(jvm, Bridge).
% Bridge = jpype  (first available in priority order)

?- auto_select_bridge(dotnet, Bridge).
% Bridge = pythonnet  (or csnakes, or none)

% With explicit preferences
?- auto_select_bridge(jvm, [prefer(jpy)], Bridge).
% Bridge = jpy

% With fallback chain
?- auto_select_bridge(jvm, [fallback([jpy, jpype])], Bridge).
% Bridge = jpy  (tries jpy first, then jpype)
```

### Preference and Firewall Integration

The auto-selection respects UnifyWeaver's preference and firewall systems:

```prolog
% Global bridge preferences
?- assertz(preferences:preferences_default([prefer_bridges([jpy, jpype])])).

% Firewall can deny specific bridges
?- assertz(firewall:rule_firewall(python_bridge/1, [denied([csnakes])])).

% Auto-select now filters by firewall before applying preferences
?- auto_select_bridge(any, Bridge).
```

### Auto-Generation

```prolog
% Generate code for best available bridge
?- generate_auto_client(jvm, [port(18812)], Code).
% Generates JPype code (or jpy if JPype unavailable)

?- generate_auto_client(dotnet, [host("server.local")], Code).
% Generates Python.NET code (or CSnakes if preferred)
```

## Next Steps

To test a new language bridge:

1. Install the bridge package (e.g., `pip install jpype1` for Java-side, Maven for Java)
2. Install RPyC: `pip install rpyc`
3. Start a test server: `python examples/rpyc-integration/rpyc_server.py`
4. Write bridge code using patterns above
5. Test connection and remote calls
6. Document any quirks or limitations
