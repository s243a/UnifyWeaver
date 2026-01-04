# C Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: C code generation target testing

## Overview

This test plan covers the C target for UnifyWeaver, which generates portable C code for high-performance, embedded, and systems programming use cases.

## Prerequisites

### System Requirements

- GCC 11+ or Clang 14+
- Make or CMake 3.20+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify C compiler
gcc --version
# or
clang --version

# Verify build tools
make --version
cmake --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_c_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `struct_generation` | Struct definitions | Correct struct syntax |
| `function_generation` | Function prototypes | Header declarations |
| `pointer_handling` | Pointer syntax | Correct * and & usage |
| `memory_management` | malloc/free | Memory allocation |
| `array_handling` | Static/dynamic arrays | Array syntax |

#### 1.2 Header Generation Tests

```bash
swipl -g "use_module('tests/core/test_c_headers'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `include_guards` | #ifndef/#define | Header protection |
| `forward_declarations` | Type declarations | Proper ordering |
| `extern_declarations` | External linkage | extern keyword |

### 2. Compilation Tests

#### 2.1 GCC Compilation

```bash
./tests/integration/test_c_gcc.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `compile_success` | gcc -c passes | Object file created |
| `link_success` | gcc linking | Executable created |
| `warnings_clean` | -Wall -Wextra | No warnings |
| `pedantic` | -pedantic | Standards compliant |

#### 2.2 Clang Compilation

```bash
./tests/integration/test_c_clang.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `clang_compile` | clang -c | Compiles cleanly |
| `static_analysis` | --analyze | No issues found |
| `sanitizers` | -fsanitize | No runtime errors |

### 3. Integration Tests

#### 3.1 Execution Tests

```bash
./tests/integration/test_c_execution.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `basic_execution` | Run binary | Correct output |
| `stdin_input` | Read from stdin | Input processed |
| `exit_codes` | Return values | Proper exit codes |

### 4. Generated Code Structure

```c
#ifndef GENERATED_QUERY_H
#define GENERATED_QUERY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    char* relation;
    char** args;
    int arg_count;
} Fact;

typedef struct {
    Fact* facts;
    int count;
    int capacity;
} FactSet;

FactSet* create_fact_set(void);
void add_fact(FactSet* set, Fact fact);
bool contains_fact(FactSet* set, Fact fact);
void free_fact_set(FactSet* set);

void init_facts(FactSet* set);
void apply_rules(FactSet* total, FactSet* delta);
void solve(FactSet* result);

#endif
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/c_target'),
    compile_to_c(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Compile and Run

```bash
# Generate, compile, and run
swipl -g "compile_to_c(query, Code), format('~w', [Code])" -t halt > /tmp/query.c
gcc -O2 -o /tmp/query /tmp/query.c
/tmp/query
```

### Full Test Suite

```bash
./tests/run_c_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CC` | C compiler | `gcc` |
| `CFLAGS` | Compiler flags | `-O2 -Wall` |
| `SKIP_C_EXECUTION` | Skip runtime tests | `0` |
| `KEEP_C_ARTIFACTS` | Preserve generated code | `0` |

## Known Issues

1. **Memory management**: Manual malloc/free required
2. **String handling**: No built-in string type
3. **Platform differences**: sizeof varies by platform
4. **Undefined behavior**: Careful with pointer arithmetic
