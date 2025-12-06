# Cross-Target Glue Design

This directory contains design documentation for UnifyWeaver's cross-target communication system.

## Status: Phases 1-5 Complete ✅

The cross-target glue system is now functionally complete with ~6,600 lines of code across 7 modules and 294+ test assertions.

## Overview

Cross-target glue enables UnifyWeaver targets to communicate and compose, allowing predicates compiled to different languages to work together seamlessly.

## Documents

| Document | Description |
|----------|-------------|
| [01-philosophy.md](01-philosophy.md) | Core principles, design goals, use cases |
| [02-specification.md](02-specification.md) | API definitions, protocols, data formats |
| [03-implementation-plan.md](03-implementation-plan.md) | Phased implementation with status |
| [04-api-reference.md](04-api-reference.md) | Complete API documentation |

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (registry, mapping, pipe glue) | ✅ Complete |
| 2 | Shell Integration (AWK ↔ Python ↔ Bash) | ✅ Complete |
| 3 | .NET Integration (C# ↔ PowerShell ↔ IronPython) | ✅ Complete |
| 4 | Native Targets (Go ↔ Rust binaries) | ✅ Complete |
| 5 | Network Layer (HTTP, sockets) | ✅ Complete |
| 6 | Advanced Features (error handling, monitoring) | Planned |

## Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `target_registry.pl` | `src/unifyweaver/core/` | Target metadata management |
| `target_mapping.pl` | `src/unifyweaver/core/` | Predicate-to-target declarations |
| `pipe_glue.pl` | `src/unifyweaver/glue/` | Basic pipe templates |
| `shell_glue.pl` | `src/unifyweaver/glue/` | AWK/Python/Bash scripts |
| `dotnet_glue.pl` | `src/unifyweaver/glue/` | .NET bridge generation |
| `native_glue.pl` | `src/unifyweaver/glue/` | Go/Rust binary orchestration |
| `network_glue.pl` | `src/unifyweaver/glue/` | HTTP/socket communication |

## Key Concepts

### Location Types (Arity 1)

```
in_process      → Same runtime (e.g., C# ↔ PowerShell in .NET)
local_process   → Separate process, same machine (pipes)
remote(Host)    → Different machine (network)
```

### Transport Types (Arity 2)

```
direct          → Function call (in-process only)
pipe            → Unix pipes with TSV/JSON
socket          → TCP socket streaming
http            → REST API calls
```

### Runtime Families

| Family | Targets | Default Communication |
|--------|---------|----------------------|
| .NET | C#, PowerShell, IronPython | In-process |
| JVM | Java, Scala, Jython | In-process |
| Shell | Bash, AWK, sed | Pipes |
| Native | Go, Rust, C | Pipes or shared memory |
| Python | CPython | Pipes |

### Sensible Defaults

- Same runtime family → in-process (zero serialization)
- Different runtimes → process pipes with TSV
- Remote → HTTP with JSON

## Quick Example

```prolog
:- use_module('src/unifyweaver/glue/shell_glue').

% Generate a 3-stage pipeline
generate_pipeline(
    [
        step(filter, awk, 'filter.awk', []),
        step(analyze, python, 'analyze.py', []),
        step(summarize, awk, 'summarize.awk', [])
    ],
    [],
    Script
).
```

Generated output:
```bash
#!/bin/bash
set -euo pipefail

cat \
    | awk -f "filter.awk" \
    | python3 "analyze.py" \
    | awk -f "summarize.awk"
```

## Examples

| Directory | Description |
|-----------|-------------|
| `examples/cross-target-glue/` | AWK ↔ Python log analysis |
| `examples/dotnet-glue/` | C# ↔ PowerShell ↔ Python |
| `examples/native-glue/` | Go ↔ Rust high-performance |
| `examples/network-glue/` | Distributed microservices |

## User Guide

See [Cross-Target Glue Guide](../../guides/cross-target-glue.md) for practical usage patterns and API reference.

## Related

- [Target documentation](../../targets/)
- [AWK target examples](../../AWK_TARGET_EXAMPLES.md)
- [SQL target documentation](../../targets/sql.md)
