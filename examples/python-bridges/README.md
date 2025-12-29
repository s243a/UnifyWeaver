# Python Bridge Examples

Examples for embedding CPython in various runtimes to use RPyC.

## Overview

These examples demonstrate how to use RPyC (Remote Python Call) from:
- **.NET** via Python.NET or CSnakes
- **JVM** via JPype or jpy
- **Rust** via PyO3
- **Ruby** via PyCall.rb

Each bridge embeds CPython and provides access to RPyC's live object proxies.

## Bridges

| Bridge | Runtime | Status | Best For |
|--------|---------|--------|----------|
| [Python.NET](pythonnet/) | .NET 6+ | ✅ Tested | Dynamic Python execution, F# interop |
| [CSnakes](csnakes/) | .NET 9+ | ✅ Tested | Source generators, compile-time wrappers |
| [JPype](jpype/) | JVM | ✅ Tested | Java projects needing NumPy/SciPy |
| [jpy](jpy/) | JVM | ✅ Tested | Bi-directional Java↔Python |
| [PyO3](pyo3/) | Rust | ✅ Tested | Systems programming with Python ML |
| [PyCall.rb](pycall/) | Ruby | ✅ Tested | Ruby/Rails with Python data science |

**Legend:** ✅ Tested and working | ⚠️ Partial (see notes) | 🔬 Documented (high confidence)

**Notes:**
- CSnakes: Uses `FromRedistributable` for cross-platform Python (auto-downloads ~60MB)
- PyCall.rb: Requires `ruby-dev` package for native extension compilation

## Quick Start

### Prerequisites

1. **RPyC server running:**
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. **Bridge-specific requirements:**
   - Python.NET: `pip install pythonnet` + .NET SDK 6+ (tested with 9.0)
   - CSnakes: .NET SDK 9.0+ (auto-downloads Python via redistributable)
   - JPype: `pip install jpype1` + Java JDK 11+ (tested with 11.0.27)
   - jpy: `pip install jpy` + Java JDK 11+ + Maven
   - PyO3: Rust toolchain (`rustup`) + Python dev headers
   - PyCall.rb: Ruby 2.7+ + `gem install pycall`

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

## Auto-Detection and Selection

The glue module can automatically detect and select the best available bridge:

```prolog
?- use_module('src/unifyweaver/glue/python_bridges_glue').

% Detect all available bridges
?- detect_all_bridges(Bridges).
% Bridges = [jpype, jpy]  % (depends on what's installed)

% Auto-select best bridge for a target platform
?- auto_select_bridge(jvm, Bridge).
% Bridge = jpype

?- auto_select_bridge(dotnet, Bridge).
% Bridge = pythonnet  % (or csnakes, or none if unavailable)

% Auto-select with explicit preferences
?- auto_select_bridge(jvm, [prefer(jpy)], Bridge).
% Bridge = jpy

% Auto-generate code for best available bridge
?- generate_auto_client(jvm, [port(18812)], Code).
% Generates code for jpype (or jpy if jpype unavailable)
```

### Bridge Requirements and Status

```prolog
% List requirements for a bridge
?- bridge_requirements(jpype, Reqs).
% Reqs = [requirement(runtime, 'Java 11+'),
%         requirement(python_package, jpype1), ...]

% Check if a bridge is ready to use
?- check_bridge_ready(jpype, Status).
% Status = ready
% Status = missing_runtime('Java')
% Status = missing_package(jpype1)

% Validate configuration options
?- validate_bridge_config(jpype, [host(localhost), port(18812)]).
% true (valid options)

?- validate_bridge_config(jpype, [port(99999)]).
% false (invalid port)
```

### Integration with Preferences and Firewall

The auto-selection integrates with UnifyWeaver's preference and firewall systems:

```prolog
% Set bridge preferences globally
?- assertz(preferences:preferences_default([prefer_bridges([jpy, jpype])])).

% Set bridge preferences for specific predicates
?- assertz(preferences:rule_preferences(my_bridge/1, [prefer([jpype])])).

% Firewall can deny specific bridges
?- assertz(firewall:rule_firewall(python_bridge/1, [denied([csnakes])])).

% Auto-select now respects these constraints
?- auto_select_bridge(jvm, Bridge).
% Returns bridge allowed by firewall, ordered by preferences
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
