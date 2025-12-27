# Janus In-Process Python↔Prolog Integration

This example demonstrates using Janus for direct in-process communication between Prolog and Python.

> **Note:** Janus is **SWI-Prolog specific**. It is not available in other Prolog implementations (GNU Prolog, XSB, etc.). For cross-Prolog compatibility, use the pipe transport.

## What is Janus?

Janus is SWI-Prolog's embedded Python interface that allows:
- **Direct function calls** between Prolog and Python
- **Zero serialization overhead** for compatible types
- **Shared memory** for large data structures
- **Bidirectional calling** (Prolog→Python and Python→Prolog)

## Comparison with Other Transports

| Transport | Overhead | Use Case |
|-----------|----------|----------|
| **Janus** | Minimal (in-process) | Tight integration, NumPy, ML |
| **Pipe** | Medium (serialization) | Process isolation, streaming |
| **HTTP** | High (network) | Distributed, microservices |

## Requirements

- **SWI-Prolog 9.0+** with Janus support
- **Python 3.8+**
- **janus-swi** Python package (for bidirectional calling)

```bash
# Install Python package for bidirectional calling
pip install janus-swi

# Optional: NumPy for numerical examples
pip install numpy
```

## Quick Start

```bash
# Run the demo
cd examples/janus-integration
swipl janus_demo.pl
?- run_demo.
```

## Examples

### Basic Python Calls

```prolog
?- use_module(library(janus)).

% Call math.sqrt
?- py_call(math:sqrt(16), R).
R = 4.0.

% Import and use module
?- py_call(importlib:import_module(json), Json),
   py_call(Json:dumps([1,2,3]), S).
S = "[1, 2, 3]".
```

### Using the Glue Module

```prolog
?- use_module('src/unifyweaver/glue/janus_glue').

% Higher-level interface
?- janus_call_python(math, sqrt, [16], R).
R = 4.0.

% NumPy integration
?- janus_numpy_array([1,2,3,4,5], Arr),
   janus_numpy_call(mean, [Arr], Mean).
Mean = 3.0.
```

### Code Generation

```prolog
% Generate wrapper for compiled Python predicate
?- generate_janus_wrapper(matrix_inverse/1,
       [module(numpy_linalg), function(inv)],
       Code).
```

Generated:
```prolog
% Janus wrapper for matrix_inverse/1
% Calls Python function numpy_linalg.inv in-process
matrix_inverse(Arg1, Result) :-
    janus_glue:janus_call_python(numpy_linalg, inv, [Arg1], Result).
```

## Integration with UnifyWeaver

Janus adds a new transport type to the glue system:

```prolog
% Register Janus as transport option
compile_pipeline(Steps, [transport(janus)], Code).
```

### When to Use Janus

| Scenario | Recommended Transport |
|----------|----------------------|
| NumPy/SciPy heavy computation | **Janus** |
| ML model inference | **Janus** |
| Large array processing | **Janus** |
| Streaming data | Pipe |
| Process isolation needed | Pipe |
| Distributed system | HTTP |

## Architecture

```
┌─────────────────────────────────────────────┐
│ SWI-Prolog Process                          │
│ ┌─────────────────┐  ┌───────────────────┐  │
│ │ Prolog Code     │←→│ Embedded Python   │  │
│ │                 │  │ (NumPy, etc.)     │  │
│ │ - UnifyWeaver   │  │                   │  │
│ │ - janus_glue.pl │  │ - Shared memory   │  │
│ │ - Your code     │  │ - Zero-copy arrays│  │
│ └─────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────┘
```

## Performance

Janus vs Pipe for 1000 calls:

| Operation | Janus | Pipe (subprocess) |
|-----------|-------|-------------------|
| Simple function | ~0.1s | ~10s |
| NumPy array (1000 elements) | ~0.2s | ~15s |
| Large matrix (1M elements) | ~0.5s | ~60s+ |

Janus is **50-100x faster** for repeated calls because it avoids:
- Process spawning
- Serialization/deserialization
- Inter-process communication

## See Also

- [SWI-Prolog Janus Documentation](https://www.swi-prolog.org/pldoc/man?section=janus)
- [JanusBridge Tutorial](../../context/projects/JanusBridge/)
- [Python Variants](../../docs/PYTHON_VARIANTS.md)
- [Cross-Target Glue](../../education/book-07-cross-target-glue/)
