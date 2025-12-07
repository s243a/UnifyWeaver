# PR: Cross-Target Glue Phase 3 - .NET Integration

## Title
feat: Add cross-target glue Phase 3 - .NET integration

## Summary

Phase 3 of the cross-target glue system implements in-process communication between C#, PowerShell, and Python within the .NET ecosystem. This enables high-performance data pipelines that minimize process spawn overhead while maintaining compatibility with the full Python ecosystem.

## Changes

### New Module: `src/unifyweaver/glue/dotnet_glue.pl`

**.NET Runtime Detection:**
- `detect_dotnet_runtime/1` - Detect .NET Core/5+/6+, Mono, or none
- `detect_ironpython/1` - Check IronPython availability (ipy/ipy64)
- `detect_powershell/1` - Detect PowerShell Core (pwsh) or Windows PowerShell

**IronPython Compatibility:**
- `ironpython_compatible/1` - Module compatibility predicate
- `can_use_ironpython/1` - Check if import list is fully compatible
- `python_runtime_choice/2` - Auto-select optimal Python runtime

Compatible modules (40+):
- Core: sys, os, json, re, collections, math, datetime, etc.
- File: pathlib, glob, shutil, tempfile
- Format: csv, xml, configparser
- .NET: clr (IronPython special)

Incompatible (require CPython):
- numpy, pandas, scipy, tensorflow, torch, etc. (C extensions)

**Bridge Generation:**

| Bridge | Transport | Use Case |
|--------|-----------|----------|
| PowerShellBridge | In-process | System.Management.Automation SDK |
| IronPythonBridge | In-process | IronPython.Hosting |
| CPythonBridge | Subprocess pipe | Fallback for C-extension modules |

**PowerShell Bridge Features:**
- `Invoke<TInput, TOutput>` - Execute script with input
- `InvokeStream<TInput, TOutput>` - Streaming record processing
- `InvokeCommand<TOutput>` - Call cmdlet by name
- `SetVariable/GetVariable` - Runspace variable access

**IronPython Bridge Features:**
- `Execute` - Run arbitrary Python code
- `ExecuteWithInput<TInput>` - Execute with input variable
- `ExecuteStream<TInput>` - Streaming record processing
- `CallFunction/DefineFunction` - Function management
- `ToPythonDict/FromPythonDict` - Collection conversion

**CPython Bridge Features:**
- JSON-based stdin/stdout communication
- Streaming support via line-based JSON
- Configurable Python path
- Error propagation from subprocess

### Integration Tests: `tests/integration/glue/test_dotnet_glue.pl`

72 test assertions covering:
- IronPython compatibility checking (17 tests)
- PowerShell bridge generation (11 tests)
- IronPython bridge generation (15 tests)
- CPython bridge generation (13 tests)
- C# host generation (9 tests)
- .NET pipeline generation (11 tests)

### Example: `examples/dotnet-glue/`

Four-stage data pipeline demonstrating cross-target communication:

```
C# (validate) → PowerShell (filter) → IronPython (enrich) → CPython (score)
```

**Architecture:**
```
┌──────────────────────────────────────────────────────────────┐
│                    .NET Host Process                          │
├──────────────────────────────────────────────────────────────┤
│  C# Validate → PowerShell Filter → IronPython Enrich ──┐    │
│                    (in-process)         (in-process)    │    │
└─────────────────────────────────────────────────────────┼────┘
                                                          │
                                            ┌─────────────▼────┐
                                            │  CPython Score   │
                                            │  (numpy via pipe)│
                                            └──────────────────┘
```

**Files:**
- `data_pipeline.pl` - Pipeline definition and code generator
- `README.md` - Architecture documentation and usage

## Test Results

```
=== .NET Glue Integration Tests ===

Test: IronPython compatibility checking
  ✓ sys is compatible
  ✓ numpy is NOT compatible
  ✓ chooses ironpython for [sys, json]
  ✓ chooses cpython_pipe for [numpy, sys]
  ... (17 tests)

Test: PowerShell bridge generation
  ✓ uses PowerShell SDK
  ✓ has generic Invoke method
  ✓ has streaming method
  ... (11 tests)

Test: IronPython bridge generation
  ✓ uses IronPython hosting
  ✓ creates Python engine
  ✓ has ExecuteStream
  ... (15 tests)

Test: CPython bridge generation
  ✓ uses JSON serialization
  ✓ has streaming method
  ... (13 tests)

Test: C# host generation
  ✓ handles powershell target
  ✓ handles ironpython target
  ✓ handles cpython target
  ... (9 tests)

Test: .NET pipeline generation
  ✓ step 1 uses PowerShell
  ✓ step 2 uses IronPython
  ✓ step 3 uses CPython
  ... (11 tests)

All tests passed!
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    dotnet_glue.pl                        │
├─────────────────────────────────────────────────────────┤
│  Runtime Detection                                       │
│  ├── detect_dotnet_runtime/1                            │
│  ├── detect_ironpython/1                                │
│  └── detect_powershell/1                                │
├─────────────────────────────────────────────────────────┤
│  Compatibility Checking                                  │
│  ├── ironpython_compatible/1  (40+ modules)             │
│  ├── can_use_ironpython/1                               │
│  └── python_runtime_choice/2                            │
├─────────────────────────────────────────────────────────┤
│  Bridge Generation                                       │
│  ├── generate_powershell_bridge/2                       │
│  ├── generate_ironpython_bridge/2                       │
│  └── generate_cpython_bridge/2                          │
├─────────────────────────────────────────────────────────┤
│  Host/Pipeline Generation                                │
│  ├── generate_csharp_host/3                             │
│  └── generate_dotnet_pipeline/3                         │
└─────────────────────────────────────────────────────────┘
```

## Relationship to Previous Phases

- **Phase 1**: Target registry and mapping (foundation)
- **Phase 2**: Shell integration (AWK ↔ Python ↔ Bash pipes)
- **Phase 3**: .NET integration (C# ↔ PowerShell ↔ Python in-process)

## Key Design Decisions

1. **In-process preference**: IronPython/PowerShell run in the same .NET process for performance
2. **Automatic fallback**: CPython used when C-extension modules detected
3. **JSON protocol**: Universal data exchange format for pipe communication
4. **Streaming support**: All bridges support streaming for large datasets

## Next Steps (Phase 4+)

- Go/Rust binary orchestration
- Network layer (sockets, HTTP)
- Error handling and retry mechanisms
