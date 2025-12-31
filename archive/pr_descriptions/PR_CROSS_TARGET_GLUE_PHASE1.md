# Add Cross-Target Glue Phase 1: Target Registry and Mapping

## Summary

Implements the foundation for cross-target communication, enabling predicates compiled to different targets to communicate seamlessly.

This is Phase 1 of the cross-target glue system as described in `docs/design/cross-target-glue/`.

## Changes

### Core Modules

**`src/unifyweaver/core/target_registry.pl`** - Central registry for target metadata
- 15+ built-in targets: bash, awk, python, go, rust, csharp, powershell, sql, etc.
- Runtime families: shell, python, dotnet, jvm, native, database
- Capability tracking (streaming, regex, ml, aggregation, etc.)
- Default location/transport resolution based on runtime family

**`src/unifyweaver/core/target_mapping.pl`** - Predicate-to-target declarations
- `declare_target/2,3` - Map predicates to compilation targets
- `declare_location/2` - Specify where predicates execute
- `declare_connection/3` - Specify transport between predicates
- Automatic resolution with sensible defaults
- Validation for mappings

**`src/unifyweaver/glue/pipe_glue.pl`** - Glue code generation for pipes
- TSV and JSON reader/writer templates
- Support for AWK, Python, Bash, Go, Rust
- Pipeline orchestrator script generation

### Tests

- `tests/core/test_target_registry.pl` - 40+ assertions
- `tests/core/test_target_mapping.pl` - 30+ assertions

All tests pass.

## Usage Example

```prolog
:- use_module('src/unifyweaver/core/target_registry').
:- use_module('src/unifyweaver/core/target_mapping').

% Declare which target compiles each predicate
:- declare_target(filter_logs/2, awk).
:- declare_target(analyze_data/2, python).
:- declare_target(store_results/2, sql).

% Query resolved transport (defaults to pipe for different runtimes)
?- resolve_transport(filter_logs/2, analyze_data/2, Transport).
Transport = pipe.

% Same runtime family uses direct (in-process)
:- declare_target(cs_logic/2, csharp).
:- declare_target(ps_script/2, powershell).
?- resolve_transport(cs_logic/2, ps_script/2, Transport).
Transport = direct.
```

## Design Decisions

1. **Location vs Transport** - Locations (arity 1) describe where things run; transports (arity 2) describe how they connect
2. **Runtime Family Affinity** - Same family defaults to in-process/direct; different families default to local_process/pipe
3. **Sensible Defaults** - Works out of the box; explicit declarations override defaults

## Next Steps

- Phase 2: Shell Integration (AWK ↔ Python pipe communication)
- Phase 3: .NET Integration (C# ↔ PowerShell in-process)

## Test Plan

```bash
# Run registry tests
swipl tests/core/test_target_registry.pl

# Run mapping tests
swipl tests/core/test_target_mapping.pl
```

## Files Changed

- `src/unifyweaver/core/target_registry.pl` (new)
- `src/unifyweaver/core/target_mapping.pl` (new)
- `src/unifyweaver/glue/pipe_glue.pl` (new)
- `tests/core/test_target_registry.pl` (new)
- `tests/core/test_target_mapping.pl` (new)

**Total: 1,428 insertions(+)**
