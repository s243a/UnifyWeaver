# Cross-Target Glue Design

This directory contains design documentation for UnifyWeaver's cross-target communication system.

## Overview

Cross-target glue enables UnifyWeaver targets to communicate and compose, allowing predicates compiled to different languages to work together seamlessly.

## Documents

| Document | Description |
|----------|-------------|
| [01-philosophy.md](01-philosophy.md) | Core principles, design goals, use cases |
| [02-specification.md](02-specification.md) | API definitions, protocols, data formats |
| [03-implementation-plan.md](03-implementation-plan.md) | Phased implementation approach |

## Key Concepts

### Location Types

```
in_process      → Same runtime (e.g., C# ↔ PowerShell in .NET)
local_process   → Separate process, same machine (pipes)
remote          → Different machine (network)
```

### Runtime Families

| Family | Targets | Default Communication |
|--------|---------|----------------------|
| .NET | C#, PowerShell, IronPython | In-process |
| JVM | Java, Scala, Jython | In-process |
| Shell | Bash, AWK | Pipes |
| Native | Go, Rust | Pipes or shared memory |
| Python | CPython | Pipes |

### Sensible Defaults

- Same runtime family → in-process (zero serialization)
- Different runtimes → process pipes with TSV
- Remote → network sockets with JSON

## Quick Example

```prolog
% Define which target compiles each predicate
:- target(filter_logs/2, awk).
:- target(analyze/2, python).
:- target(store/2, sql).

% UnifyWeaver automatically generates:
% - AWK script with TSV output
% - Python script with TSV input/output
% - SQL insert from TSV input
% - Bash orchestrator connecting them via pipes
```

## Status

**Current:** Design phase

**Next:** Phase 1 implementation (target registry, basic pipe glue)

## Related

- [Target documentation](../../targets/)
- [AWK target examples](../../AWK_TARGET_EXAMPLES.md)
- [SQL target documentation](../../targets/sql.md)
