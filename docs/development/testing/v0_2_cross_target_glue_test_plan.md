# Cross-Target Glue Test Plan - v0.2

**Version**: 0.2
**Date**: December 2025
**Status**: Draft
**Scope**: Cross-target integration and glue layer testing

## Overview

This test plan covers the cross-target glue system that enables interoperability between different compilation targets (Bash, Python, Go, Rust, C#, .NET) and platform-specific integrations.

## Prerequisites

### System Requirements

- SWI-Prolog 9.0+
- Multiple targets installed (varies by test):
  - Bash (default on Linux/macOS)
  - Python 3.8+
  - Go 1.21+
  - Rust 1.70+
  - .NET 8.0+ (for C# tests)
  - AWK (gawk preferred)

### Verification

```bash
# Verify available targets
bash --version
python3 --version
go version
rustc --version
dotnet --version
awk --version
```

## Test Categories

### 1. Shell Glue Tests

The shell glue module enables cross-language pipelines in shell environments.

```bash
# Run shell glue integration tests
swipl -g "use_module('tests/integration/glue/test_shell_glue'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `test_awk_script_generation` | Generate AWK scripts | Proper field separator, header handling |
| `test_python_script_generation` | Generate Python scripts | Correct imports, heredoc patterns |
| `test_bash_script_generation` | Generate Bash scripts | Safe quoting, error handling |
| `test_pipeline_generation` | Multi-stage pipelines | Proper pipe chaining |
| `test_format_options` | TSV/JSON/CSV output | Correct formatting |

#### 1.1 AWK Script Generation

```prolog
generate_awk_script(
    '# Filter high salary\n    if (salary > 50000) {',
    [name, dept, salary],
    [format(tsv)],
    AwkScript
).
```

**Verification**:
- Sets correct field separator (`FS = "\t"`)
- Assigns field variables correctly (`name = $1`, etc.)
- Handles headers when specified

#### 1.2 Python Script Generation

```prolog
generate_python_script(
    '# Process data',
    [id, value],
    [format(json)],
    PyScript
).
```

**Verification**:
- Uses `/dev/fd/3` heredoc pattern
- Proper import statements
- Error handling

#### 1.3 Pipeline Generation

```prolog
generate_pipeline([
    stage(parse, awk, [format(tsv)]),
    stage(transform, python, []),
    stage(aggregate, bash, [])
], Options, Pipeline).
```

### 2. .NET Glue Tests

The .NET glue module enables C#/PowerShell interoperability.

```bash
# Run .NET glue integration tests
swipl -g "use_module('tests/integration/glue/test_dotnet_glue'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `test_ironpython_compatibility` | IronPython module check | Correct compatibility detection |
| `test_powershell_bridge_generation` | PowerShell bridges | Proper PSObject handling |
| `test_ironpython_bridge_generation` | IronPython bridges | clr module usage |
| `test_cpython_bridge_generation` | CPython via .NET | Process execution |
| `test_csharp_host_generation` | C# host programs | Proper compilation |
| `test_dotnet_pipeline_generation` | .NET pipelines | Inter-language data flow |

#### 2.1 IronPython Compatibility

```prolog
% These should be compatible
ironpython_compatible(sys).
ironpython_compatible(json).
ironpython_compatible(clr).

% These should NOT be compatible
\+ ironpython_compatible(numpy).
\+ ironpython_compatible(pandas).
```

#### 2.2 Bridge Generation

```prolog
generate_powershell_bridge(
    'Process-Data',
    [param1, param2],
    BridgeCode
).
```

### 3. Native Glue Tests

The native glue module handles compiled language interoperability.

```bash
# Run native glue integration tests
swipl -g "use_module('tests/integration/glue/test_native_glue'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Go compilation | Build Go binaries | Successful compilation |
| Rust compilation | Build Rust binaries | Successful compilation |
| Cross-call | Go calls Rust via FFI | Data exchange works |
| Shared memory | Memory-mapped communication | Correct synchronization |

### 4. Network Glue Tests

The network glue module handles service communication.

```bash
# Run network glue integration tests
swipl -g "use_module('tests/integration/glue/test_network_glue'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| HTTP client generation | curl/wget scripts | Proper headers, auth |
| gRPC stubs | Protocol buffer handling | Correct serialization |
| Service discovery | Endpoint resolution | DNS/registry lookup |

### 5. Deployment Glue Tests

```bash
# Run deployment glue tests
swipl -g "use_module('tests/glue/test_deployment_glue'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Service declaration | `declare_service/2` | Configuration stored |
| Deployment method | SSH/rsync generation | Proper scripts |
| Lifecycle hooks | Pre/post deploy hooks | Hook execution |
| Health checks | Endpoint monitoring | Status detection |

## Cross-Target Pipeline Tests

### 5.1 AWK -> Python -> Go Pipeline

```prolog
% Define multi-stage pipeline
pipeline([
    stage(extract, awk, [format(tsv), fields([id, name, value])]),
    stage(transform, python, [mode(generator)]),
    stage(aggregate, go, [mode(generator)])
]).
```

**Test Execution**:
```bash
# Generate pipeline
swipl -l init.pl -g "
    generate_cross_target_pipeline([
        stage(extract, awk, []),
        stage(transform, python, []),
        stage(aggregate, go, [])
    ], '/tmp/pipeline_test'),
    halt
" -t halt

# Execute pipeline
cat input.tsv | /tmp/pipeline_test/run_pipeline.sh
```

### 5.2 Bash -> C# -> Rust Pipeline

```prolog
pipeline([
    stage(preprocess, bash, []),
    stage(query, csharp, [mode(query)]),
    stage(output, rust, [json_output(true)])
]).
```

## Test Matrix

### Glue Module Coverage

| Module | Unit Tests | Integration | E2E |
|--------|------------|-------------|-----|
| shell_glue | ✓ | ✓ | ⚠ |
| dotnet_glue | ✓ | ✓ | ⚠ |
| native_glue | ✓ | ⚠ | ⚠ |
| network_glue | ✓ | ⚠ | ⚠ |
| deployment_glue | ✓ | ✓ | ⚠ |

### Cross-Target Combinations

| Source | Target | Data Format | Status |
|--------|--------|-------------|--------|
| Bash | AWK | TSV | ✓ Stable |
| AWK | Python | TSV/JSON | ✓ Stable |
| Python | Go | JSON | ✓ Stable |
| Python | Rust | JSON | ✓ Stable |
| Go | Rust | Binary | ⚠ Testing |
| C# | Python | JSON | ✓ Stable |
| PowerShell | Python | JSON | ⚠ Platform |

### Platform Coverage

| Platform | Shell | .NET | Native |
|----------|-------|------|--------|
| Linux | ✓ | ✓ | ✓ |
| macOS | ✓ | ✓ | ✓ |
| Windows (WSL) | ✓ | ✓ | ✓ |
| Windows (native) | ⚠ | ✓ | ✓ |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_DOTNET_GLUE` | `0` | Skip .NET glue tests |
| `SKIP_NATIVE_GLUE` | `0` | Skip native glue tests |
| `SKIP_NETWORK_GLUE` | `0` | Skip network glue tests |
| `GLUE_OUTPUT_DIR` | `/tmp/unifyweaver_glue` | Output directory |
| `GLUE_KEEP_ARTIFACTS` | `0` | Keep generated files |

## Quick Test Commands

### Fast Verification (Shell Glue Only)

```bash
swipl -g "use_module('tests/integration/glue/test_shell_glue'), run_tests" -t halt
```

### Full Glue Test Suite

```bash
# All glue tests
swipl -g "use_module('tests/integration/glue/test_shell_glue'), run_tests" -t halt && \
swipl -g "use_module('tests/integration/glue/test_dotnet_glue'), run_tests" -t halt && \
swipl -g "use_module('tests/integration/glue/test_native_glue'), run_tests" -t halt
```

### Platform-Specific Tests

```bash
# Linux-only tests
swipl -g "use_module('tests/integration/glue/test_shell_glue'), run_tests" -t halt

# Windows/.NET tests (requires .NET SDK)
swipl -g "use_module('tests/integration/glue/test_dotnet_glue'), run_tests" -t halt
```

## Integration with Firewall

Cross-target glue respects firewall policies:

```prolog
% .firewall
:- firewall_mode(enforce).

% Only allow approved pipeline patterns
:- allow(pipeline([stage(_, bash, _), stage(_, go, _)])).
:- deny(pipeline([stage(_, python, _), stage(_, bash, _)])).  % Insecure

% Require validation for all cross-target calls
:- implies(cross_target_call(_, _), validation(passed)).
```

## Known Issues

1. **Windows path handling**: Backslash vs forward slash normalization
2. **IronPython limitations**: C extension modules not available
3. **Network glue**: Requires network access for full testing
4. **Binary protocols**: Endianness considerations for cross-platform

## Related Documentation

- [Book 8: Security & Firewall](../../../education/book-08-security-firewall/README.md)
- [Deployment Glue Implementation](../../../src/unifyweaver/glue/deployment_glue.pl)
- [Shell Glue Implementation](../../../src/unifyweaver/glue/shell_glue.pl)
- [.NET Glue Implementation](../../../src/unifyweaver/glue/dotnet_glue.pl)
- [Quick Testing Guide](quick_testing.md)

## Changelog

- **v0.2** (Dec 2025): Initial test plan creation
