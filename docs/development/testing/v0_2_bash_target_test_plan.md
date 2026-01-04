# Bash Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Bash script generation target testing

## Overview

This test plan covers the Bash target for UnifyWeaver, which generates portable shell scripts for system automation, pipeline orchestration, and cross-platform execution.

## Prerequisites

### System Requirements

- Bash 4.0+ (5.0+ recommended)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned
- Common Unix utilities (grep, sed, cut, sort, etc.)

### Verification

```bash
# Verify Bash version
bash --version

# Verify Prolog
swipl --version

# Verify platform detection
uname -s  # Linux, Darwin, MINGW64_NT-*, etc.
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify Bash code generation without executing the generated scripts.

#### 1.1 Basic Generator Tests

```bash
# Run Bash generator tests
swipl -g "use_module('tests/core/test_bash_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `variable_declaration` | Variable handling | Proper quoting, `local` keyword |
| `function_generation` | Function definitions | Correct `function name() {}` syntax |
| `array_handling` | Bash arrays | Proper array syntax `arr=()` |
| `conditional_logic` | If/else/case | Correct bracket syntax `[[ ]]` |
| `loop_generation` | For/while loops | Proper loop constructs |
| `pipeline_construction` | Pipe chains | Correct `|` chaining |

#### 1.2 Platform Detection Tests

```bash
swipl -g "use_module('tests/core/test_bash_platform_detection'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `detect_linux` | Linux detection | `uname` check works |
| `detect_macos` | macOS detection | Darwin detection |
| `detect_wsl` | WSL detection | WSL environment vars |
| `detect_mingw` | Git Bash/MSYS | MINGW detection |

#### 1.3 Enhanced Chaining Tests

```bash
swipl -g "use_module('tests/core/test_bash_enhanced_chaining'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `simple_chain` | A → B pipeline | Correct pipe syntax |
| `multi_chain` | A → B → C → D | Chain preserved |
| `error_handling` | `set -e` pipefail | Error propagation |
| `subshell_isolation` | Subshell grouping | `()` vs `{}` correct |

### 2. Integration Tests (Compilation + Execution)

These tests compile Prolog to Bash, execute the script, and verify output.

#### 2.1 Executor Tests

```bash
./tests/integration/test_bash_executor.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Simple execution | Basic script run | Exit code 0 |
| Stdin processing | Read from pipe | Correct input handling |
| Stdout capture | Output collection | Captured correctly |
| Stderr handling | Error output | Separated from stdout |
| Exit code propagation | Non-zero exit | Correct code returned |

#### 2.2 Enhanced Chaining Integration

```bash
KEEP_TEST_DATA=1 ./tests/integration/test_bash_enhanced_chaining.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Bash → AWK | Shell to AWK handoff | Data flows correctly |
| Bash → Python | Shell to Python | Correct subprocess call |
| Parallel execution | Background jobs | `&` and `wait` work |
| Named pipes | FIFO usage | mkfifo/cleanup works |

#### 2.3 Match Captures Tests

```bash
./tests/integration/test_bash_match_captures.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Regex capture | `=~` with groups | BASH_REMATCH populated |
| Multiple captures | Multiple groups | All groups accessible |
| No match handling | Pattern fails | Empty result, no error |

### 3. Cross-Platform Tests

#### 3.1 Platform Compatibility

Run on each supported platform:

```bash
# Linux
./tests/integration/test_bash_linux.sh

# macOS
./tests/integration/test_bash_macos.sh

# Windows (Git Bash/WSL)
./tests/integration/test_bash_windows.sh
```

**Platform Matrix**:
| Feature | Linux | macOS | Git Bash | WSL |
|---------|-------|-------|----------|-----|
| Arrays | Yes | Yes | Yes | Yes |
| `[[ ]]` | Yes | Yes | Yes | Yes |
| Process substitution | Yes | Yes | Limited | Yes |
| `/dev/fd/*` | Yes | Yes | No | Yes |
| `readarray` | Yes | 4.0+ | 4.0+ | Yes |

#### 3.2 Shell Compatibility

```bash
# Test with different shells (where compatible)
SHELL=/bin/bash ./tests/integration/test_bash_compat.sh
SHELL=/bin/zsh ./tests/integration/test_bash_compat.sh  # zsh compatibility mode
```

### 4. Performance Tests

#### 4.1 Script Startup Time

```bash
# Measure script overhead
time bash -c 'echo hello'

# Measure generated script overhead
time bash /tmp/generated_script.sh
```

**Benchmarks**:
| Test | Expected Time |
|------|---------------|
| Empty script | < 10ms |
| Simple pipeline | < 50ms |
| Complex script (100 lines) | < 100ms |

#### 4.2 Large Data Processing

```bash
# Process large input
seq 1 100000 | ./generated_script.sh | wc -l
```

## Test Commands Reference

### Quick Smoke Test

```bash
# Generate a simple Bash script
swipl -g "
    use_module('src/unifyweaver/targets/bash_target'),
    compile_to_bash(test_pipeline, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
# Run all Bash tests
./tests/run_bash_tests.sh

# Or individually:
swipl -g "use_module('tests/core/test_bash_generator'), run_tests" -t halt
swipl -g "use_module('tests/core/test_bash_executor'), run_tests" -t halt
./tests/integration/test_bash_enhanced_chaining.sh
./tests/integration/test_bash_match_captures.sh
```

## Known Issues

1. **Bash 3.x (macOS default)**: Some features require Bash 4.0+
2. **MINGW/Git Bash**: Process substitution limited
3. **Space in paths**: Requires careful quoting
4. **Large arrays**: Performance degrades >10000 elements

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UNIFYWEAVER_SHELL` | Shell interpreter | `/bin/bash` |
| `BASH_COMPAT` | Compatibility mode | (none) |
| `KEEP_TEST_DATA` | Preserve test artifacts | `0` |
| `DEBUG_BASH` | Enable `set -x` tracing | `0` |

## Script Safety Patterns

Generated scripts follow these safety patterns:

```bash
#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'        # Safer word splitting

# Cleanup on exit
cleanup() {
    rm -f "$TEMP_FILE" 2>/dev/null || true
}
trap cleanup EXIT
```

## Related Documentation

- [Bash Target Implementation](../../architecture/targets/bash_target.md)
- [Enhanced Chaining](../../architecture/enhanced_chaining.md)
- [Platform Detection](../../architecture/platform_detection.md)
- [Pipeline Generator Mode](../../architecture/pipeline_generator_mode.md)
