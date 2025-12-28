# Python Bridge Examples

Examples for embedding CPython in .NET and JVM runtimes to use RPyC.

## Overview

These examples demonstrate how to use RPyC (Remote Python Call) from:
- **.NET** via Python.NET or CSnakes
- **JVM** via JPype or jpy

Each bridge embeds CPython and provides access to RPyC's live object proxies.

## Bridges

| Bridge | Runtime | Status | Best For |
|--------|---------|--------|----------|
| [Python.NET](pythonnet/) | .NET 6+ | ✅ Tested | Dynamic Python execution, F# interop |
| [CSnakes](csnakes/) | .NET 8+ | ⚠️ Different | Source generators, compile-time wrappers |
| [JPype](jpype/) | JVM | ✅ Tested | Java projects needing NumPy/SciPy |
| [jpy](jpy/) | JVM | ✅ Tested | Bi-directional Java↔Python |

**Note:** CSnakes uses compile-time source generators rather than dynamic execution. See [csnakes/README.md](csnakes/) for details.

## Quick Start

### Prerequisites

1. **RPyC server running:**
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. **Bridge-specific requirements:**
   - Python.NET: `pip install pythonnet` + .NET SDK 6+ (tested with 9.0)
   - CSnakes: .NET SDK 8.0+ + `dotnet new install CSnakes.Templates`
   - JPype: `pip install jpype1` + Java JDK 11+ (tested with 11.0.27)
   - jpy: `pip install jpy` + Java JDK 11+ + Maven

   **Note:** Python.NET defaults to .NET Core (not Mono) on modern systems.

### Running Examples

**.NET (Python.NET):**
```bash
cd examples/python-bridges/pythonnet
dotnet run
```

**.NET (CSnakes):**
```bash
cd examples/python-bridges/csnakes
dotnet run
```

**Java (JPype):**
```bash
cd examples/python-bridges/jpype
./gradlew run
```

**Java (jpy):**
```bash
cd examples/python-bridges/jpy
./gradlew run
```

## Code Generation

Generate bridge code from Prolog:

```prolog
?- use_module('src/unifyweaver/glue/python_bridges_glue').

% Generate Python.NET client
?- generate_pythonnet_rpyc_client([host("myserver"), port(18812)], Code).

% Generate JPype client
?- generate_jpype_rpyc_client([package("com.example")], Code).

% Or use generic interface
?- generate_python_bridge_client(pythonnet, [port(18900)], Code).
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Host Application (.NET or JVM)                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Bridge (Python.NET / CSnakes / JPype / jpy)             │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ Embedded CPython                                    │ │ │
│ │ │ ┌─────────────────────────────────────────────────┐ │ │ │
│ │ │ │ RPyC Client                                     │ │ │ │
│ │ │ │                                                 │ │ │ │
│ │ │ │  conn = rpyc.connect("server", 18812)           │ │ │ │
│ │ │ │  result = conn.modules.numpy.array([1,2,3])     │ │ │ │
│ │ │ └─────────────────────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ TCP/IP (live proxies)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ RPyC Server (Python)                                        │
│ - NumPy, SciPy, pandas available                            │
│ - Custom services                                           │
│ - Live object proxies                                       │
└─────────────────────────────────────────────────────────────┘
```

## When to Use Each Bridge

### Python.NET
- Mature, stable (since 2003)
- Good for existing .NET Framework and .NET Core projects
- F# interoperability
- Larger community

### CSnakes
- Modern .NET 8+ only
- **Source generator approach** - generates C# wrappers from Python files at compile time
- Better async support
- Type-safe but requires wrapper functions (not dynamic execution)
- Best when you have well-defined Python functions to wrap

### JPype
- Mature, well-maintained
- Shared memory for NumPy arrays (fast!)
- Good documentation
- Active community

### jpy
- True bi-directional calling
- Java can call Python, Python can call Java
- Used by JetBrains tools
- Good for mixed Java/Python codebases

## Related Documentation

- [RPyC Integration](../rpyc-integration/)
- [Python Bridges Glue](../../src/unifyweaver/glue/python_bridges_glue.pl)
- [Cross-Runtime Pipeline](../../src/unifyweaver/glue/cross_runtime_pipeline.pl)
