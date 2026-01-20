# Skill: Python Bridges

Cross-runtime Python embedding for .NET, JVM, Rust, Ruby, and FFI-based integration.

## When to Use

- User asks "how do I embed Python in my .NET/Java/Rust app?"
- User needs lowest-latency Python integration
- User wants to call Python from Go or Node.js via FFI
- User needs to select the best bridge for their platform

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/python_bridges_glue').

% Detect available bridges on system
detect_all_bridges(Bridges).

% Auto-select best bridge for platform
auto_select_bridge(RuntimeEnv, Preferences, Bridge).

% Generate client code for specific bridge
generate_pythonnet_rpyc_client(Predicates, CSharpCode).
generate_jpype_rpyc_client(Predicates, JavaCode).
generate_pyo3_rpyc_client(Predicates, RustCode).
```

## Supported Bridges

### .NET Bridges

| Bridge | Description | Status |
|--------|-------------|--------|
| Python.NET | CPython embedding in .NET | Mature |
| CSnakes | Source-generated Python interop | Modern |

### JVM Bridges

| Bridge | Description | Status |
|--------|-------------|--------|
| JPype | Direct Python integration | Mature |
| jpy | Bidirectional Java-Python | Fast |

### Rust Bridge

| Bridge | Description | Status |
|--------|-------------|--------|
| PyO3 | Rust bindings for Python | Mature |

### Ruby Bridge

| Bridge | Description | Status |
|--------|-------------|--------|
| PyCall.rb | Python-Ruby interop | Stable |

### FFI Bridges

| Target | Library | Description |
|--------|---------|-------------|
| Go | cgo | C FFI to Python |
| Node.js | koffi | Modern FFI library |
| Node.js | ffi-napi | N-API based FFI |

## Bridge Detection

### Detect All Bridges

```prolog
detect_all_bridges(Bridges).
```

Returns list of available bridges on the system.

### Auto-Select Bridge

```prolog
auto_select_bridge(RuntimeEnv, Preferences, Bridge).
```

**RuntimeEnv:** Runtime environment (dotnet, jvm, rust, ruby, go, nodejs)

**Preferences:** Optional preferences list
- `prefer(BridgeName)` - Prefer specific bridge
- `avoid(BridgeName)` - Avoid specific bridge

**Example:**
```prolog
% Auto-select for JVM
auto_select_bridge(jvm, [], Bridge).
% Bridge = jpype or jpy

% Prefer specific bridge
auto_select_bridge(dotnet, [prefer(csnakes)], Bridge).
% Bridge = csnakes
```

## Code Generation

### Python.NET (.NET)

```prolog
generate_pythonnet_rpyc_client(Predicates, Code).
```

**Output (C#):**
```csharp
using Python.Runtime;

public class PythonBridge {
    public static void Initialize() {
        PythonEngine.Initialize();
    }

    public static dynamic CallFunction(string module, string func, params object[] args) {
        using (Py.GIL()) {
            dynamic py = Py.Import(module);
            return py.InvokeMethod(func, args);
        }
    }
}
```

### CSnakes (.NET)

```prolog
generate_csnakes_rpyc_client(Predicates, Code).
```

Modern source-generated approach with strong typing.

### JPype (JVM)

```prolog
generate_jpype_rpyc_client(Predicates, Code).
```

**Output (Java):**
```java
import org.jpype.*;

public class PythonBridge {
    public static void initialize() {
        JPype.startJVM();
    }

    public static Object callFunction(String module, String func, Object... args) {
        PyObject pyModule = PyModule.importModule(module);
        return pyModule.callMethod(func, args);
    }
}
```

### jpy (JVM)

```prolog
generate_jpy_rpyc_client(Predicates, Code).
```

### PyO3 (Rust)

```prolog
generate_pyo3_rpyc_client(Predicates, Code).
```

**Output (Rust):**
```rust
use pyo3::prelude::*;

fn call_python(module: &str, func: &str, args: Vec<&PyAny>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let py_module = PyModule::import(py, module)?;
        let result = py_module.call_method(func, args, None)?;
        Ok(result.into())
    })
}
```

### PyCall.rb (Ruby)

```prolog
generate_pycall_rb_rpyc_client(Predicates, Code).
```

**Output (Ruby):**
```ruby
require 'pycall'

def call_python(mod, func, *args)
  py_module = PyCall.import_module(mod)
  py_module.send(func, *args)
end
```

### Rust FFI Bridge

```prolog
generate_rust_ffi_bridge(Predicates, Code).
```

Generates Rust cdylib that can be called from Go and Node.js.

### Go FFI Client

```prolog
generate_go_ffi_client(Predicates, Code).
```

Uses cgo to call Rust cdylib.

### Node.js FFI Client

```prolog
generate_node_koffi_client(Predicates, Code).
generate_node_ffi_napi_client(Predicates, Code).
```

## Firewall Integration

Bridges integrate with UnifyWeaver's firewall system:

```prolog
% Check if bridge is allowed by firewall
validate_bridge(Bridge, Firewall).

% Auto-select respects firewall constraints
auto_select_bridge(Env, [firewall(Firewall)], Bridge).
```

## Performance Comparison

| Bridge | Startup | Call Latency | Memory | Best For |
|--------|---------|--------------|--------|----------|
| Python.NET | Medium | Low | Medium | .NET apps |
| CSnakes | Low | Very Low | Low | Modern .NET |
| JPype | Medium | Low | Medium | JVM apps |
| jpy | Low | Very Low | Low | High-perf JVM |
| PyO3 | Low | Very Low | Low | Rust apps |
| FFI | Medium | Medium | Low | Polyglot |

## Common Patterns

### .NET Application

```prolog
% Generate .NET client
generate_pythonnet_rpyc_client([
    numpy_mean/2,
    scipy_optimize/3
], CSharpCode).
```

### JVM Application

```prolog
% Auto-select and generate
auto_select_bridge(jvm, [], Bridge),
(Bridge = jpype ->
    generate_jpype_rpyc_client(Predicates, Code)
;
    generate_jpy_rpyc_client(Predicates, Code)
).
```

### Rust Application

```prolog
% Generate PyO3 bindings
generate_pyo3_rpyc_client([
    process_data/2,
    train_model/3
], RustCode).
```

### Go Application via FFI

```prolog
% Generate Rust FFI bridge
generate_rust_ffi_bridge(Predicates, RustCode).

% Generate Go client
generate_go_ffi_client(Predicates, GoCode).
```

## Related

**Parent Skill:**
- `skill_ipc.md` - IPC sub-master

**Sibling Skills:**
- `skill_pipe_communication.md` - Unix pipes
- `skill_rpyc.md` - Network-based RPC

**Code:**
- `src/unifyweaver/glue/python_bridges_glue.pl`
