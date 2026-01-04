# LLVM Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: LLVM IR code generation target testing

## Overview

This test plan covers the LLVM target for UnifyWeaver, which generates LLVM Intermediate Representation (IR) for compilation to native machine code across multiple architectures.

## Prerequisites

### System Requirements

- LLVM 15+ toolchain
- Clang 15+ (for linking)
- llc, opt, llvm-link utilities
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify LLVM installation
llvm-config --version
llc --version
opt --version
clang --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic IR Generator Tests

```bash
swipl -g "use_module('tests/core/test_llvm_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `module_structure` | Module declaration | Valid module |
| `function_definition` | define keyword | Function syntax |
| `basic_blocks` | Label blocks | BB structure |
| `type_definitions` | %struct types | Type syntax |
| `instructions` | add, load, store | Instruction syntax |

#### 1.2 IR Verification

```bash
swipl -g "use_module('tests/core/test_llvm_verify'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `ssa_form` | SSA compliance | Single assignment |
| `type_consistency` | Type checking | Types match |
| `terminator_check` | Block terminators | Proper terminators |

### 2. Compilation Tests

#### 2.1 LLVM Tools Pipeline

```bash
./tests/integration/test_llvm_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `llvm_as` | Assemble IR | .bc file created |
| `opt_passes` | Optimization | Optimized IR |
| `llc_codegen` | Code generation | .o or .s file |
| `clang_link` | Final linking | Executable |

#### 2.2 Verification Tests

```bash
./tests/integration/test_llvm_verify.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `llvm_verify` | IR verification | No errors |
| `undefined_refs` | Reference check | All defined |
| `memory_safety` | ASAN check | No violations |

### 3. Generated IR Structure

```llvm
; ModuleID = 'unifyweaver_generated'
source_filename = "generated.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Fact = type { i8*, [8 x i8*], i32 }
%struct.FactSet = type { %struct.Fact*, i32, i32 }

@.str.parent = private unnamed_addr constant [7 x i8] c"parent\00"
@.str.john = private unnamed_addr constant [5 x i8] c"john\00"
@.str.mary = private unnamed_addr constant [5 x i8] c"mary\00"

declare i8* @malloc(i64)
declare void @free(i8*)
declare i32 @printf(i8*, ...)

define void @init_facts(%struct.FactSet* %set) {
entry:
  ; Initialize facts
  ret void
}

define void @apply_rules(%struct.FactSet* %total, %struct.FactSet* %delta, %struct.FactSet* %result) {
entry:
  ; Rule application
  ret void
}

define void @solve(%struct.FactSet* %result) {
entry:
  %facts = alloca %struct.FactSet
  %delta = alloca %struct.FactSet
  call void @init_facts(%facts)
  ; Fixpoint loop
  ret void
}

define i32 @main() {
entry:
  %result = alloca %struct.FactSet
  call void @solve(%result)
  ; Print results
  ret i32 0
}
```

### 4. Architecture Tests

#### 4.1 Multi-Target Compilation

```bash
./tests/integration/test_llvm_targets.sh
```

**Test Cases**:
| Target | Description | Expected |
|--------|-------------|----------|
| x86_64 | 64-bit x86 | Compiles |
| aarch64 | ARM 64-bit | Compiles |
| wasm32 | WebAssembly | Compiles |

### 5. Optimization Tests

#### 5.1 Optimization Passes

```bash
./tests/integration/test_llvm_optimization.sh
```

**Optimization Levels**:
| Level | Passes | Expected |
|-------|--------|----------|
| -O0 | None | Debug build |
| -O1 | Basic | Some optimization |
| -O2 | Standard | Good optimization |
| -O3 | Aggressive | Maximum optimization |

### 6. Performance Tests

#### 6.1 Generated Code Performance

```bash
./tests/perf/test_llvm_performance.sh
```

**Benchmarks**:
| Test | -O0 | -O2 | -O3 |
|------|-----|-----|-----|
| Simple query | < 10ms | < 5ms | < 2ms |
| 1000 facts | < 100ms | < 50ms | < 20ms |

## Test Commands Reference

### Quick Smoke Test

```bash
# Generate LLVM IR
swipl -g "
    use_module('src/unifyweaver/targets/llvm_target'),
    compile_to_llvm(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Compile and Run

```bash
# Full pipeline
swipl -g "compile_to_llvm(query, IR), format('~w', [IR])" -t halt > /tmp/query.ll
llc -filetype=obj /tmp/query.ll -o /tmp/query.o
clang /tmp/query.o -o /tmp/query
/tmp/query
```

### Full Test Suite

```bash
./tests/run_llvm_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLVM_DIR` | LLVM installation | (system) |
| `LLVM_TARGET` | Target triple | (host) |
| `LLVM_OPT_LEVEL` | Optimization level | `-O2` |
| `SKIP_LLVM_EXECUTION` | Skip runtime tests | `0` |
| `KEEP_LLVM_ARTIFACTS` | Preserve IR files | `0` |

## Known Issues

1. **IR version**: LLVM IR changes between versions
2. **Target-specific**: Some features are target-dependent
3. **Debug info**: DWARF generation can be complex
4. **Memory management**: Manual memory management required
5. **Linking**: External functions need proper declarations
