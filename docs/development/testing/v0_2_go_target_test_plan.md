# Go Target Test Plan - v0.2

**Version**: 0.2
**Date**: December 2025
**Status**: Draft
**Scope**: Go code generation target testing

## Overview

This test plan covers the Go target for UnifyWeaver, which generates native Go programs from Prolog predicates using semi-naive fixpoint evaluation.

## Prerequisites

### System Requirements

- Go 1.21+ installed
- `go` command in PATH
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Go installation
go version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify Go code generation without executing the generated code.

#### 1.1 Basic Generator Tests

```bash
# Run all Go generator tests
swipl -g "use_module('tests/core/test_go_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `compile_transitive_closure` | Ancestor relationship | Generates fixpoint loop, rule functions |
| `compile_with_negation` | Negation-as-failure | Generates negation check code |
| `contains_fixpoint_loop` | Fixpoint structure | Contains `changed := true`, `for changed` |

#### 1.2 Semantic Compilation Tests

```bash
swipl -g "use_module('tests/core/test_go_semantic_compilation'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Fact compilation | Simple facts | Generates initial fact set |
| Rule compilation | Basic rules | Generates rule application functions |
| Join operations | Multi-body rules | Generates nested iteration |

#### 1.3 XML Integration Tests

```bash
swipl -g "use_module('tests/core/test_go_xml_integration'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| XML parsing | Read XML input | Generates proper unmarshaling |
| XPath-like access | Field extraction | Generates struct field access |

### 2. Integration Tests (Compilation + Execution)

These tests compile Prolog to Go, build the Go binary, and execute it.

#### 2.1 JSON Processing Tests

```bash
swipl -g "use_module('tests/test_go_json_advanced'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| JSON input parsing | Parse JSON records | Correct struct population |
| JSON output generation | Emit JSON results | Valid JSON output |
| Streaming JSON | Line-by-line processing | Handles large inputs |

#### 2.2 Validation Tests

```bash
swipl -g "use_module('tests/test_go_validation'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Input validation | Type checking | Proper error messages |
| Constraint validation | Range checks | Rejects invalid data |

#### 2.3 Parallel Execution Tests

```bash
swipl -g "use_module('tests/test_go_parallel'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Concurrent fact processing | Parallel iteration | Thread-safe execution |
| Worker pool generation | Goroutine pools | Proper synchronization |

#### 2.4 Aggregation Tests

```bash
swipl -g "use_module('tests/test_go_new_aggregation'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Sum aggregation | `sum(X)` | Correct totals |
| Count aggregation | `count(X)` | Correct counts |
| Group-by aggregation | `group_by(K, V)` | Proper grouping |

#### 2.5 Group-By Tests

```bash
swipl -g "use_module('tests/test_go_group_by'), run_tests" -t halt
```

### 3. End-to-End Tests

#### 3.1 Full Pipeline Test

```bash
# Generate Go code, compile, and execute
cd /root/UnifyWeaver
swipl -l init.pl -g "
    assertz(parent(john, mary)),
    assertz(parent(mary, sue)),
    assertz((ancestor(X, Y) :- parent(X, Y))),
    assertz((ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z))),
    compile_predicate_to_go(ancestor/2, [mode(generator), output_dir('/tmp/go_test')], _),
    halt
" -t halt

# Build and run
cd /tmp/go_test
go build -o ancestor_test
echo -e "john\tmary\nmary\tsue" | ./ancestor_test
```

**Expected Output**:
```
john	mary
mary	sue
john	sue
```

## Test Matrix

### Feature Coverage

| Feature | Code Gen | Build | Execute | Status |
|---------|----------|-------|---------|--------|
| Facts | ✓ | ✓ | ✓ | Stable |
| Rules | ✓ | ✓ | ✓ | Stable |
| Transitive closure | ✓ | ✓ | ✓ | Stable |
| Negation | ✓ | ✓ | ⚠ | Testing |
| Aggregation | ✓ | ✓ | ⚠ | Testing |
| JSON I/O | ✓ | ✓ | ✓ | Stable |
| Parallel | ✓ | ⚠ | ⚠ | Experimental |

### Platform Coverage

| Platform | Code Gen | Compilation | Execution |
|----------|----------|-------------|-----------|
| Linux x86_64 | ✓ | ✓ | ✓ |
| Linux ARM64 | ✓ | ✓ | ✓ |
| macOS x86_64 | ✓ | ✓ | ✓ |
| macOS ARM64 | ✓ | ✓ | ✓ |
| Windows (WSL) | ✓ | ✓ | ✓ |
| Windows (native) | ✓ | ⚠ | ⚠ |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_GO_EXECUTION` | `0` | Skip Go compilation/execution |
| `GO_OUTPUT_DIR` | `/tmp/unifyweaver_go` | Output directory for generated code |
| `GO_KEEP_ARTIFACTS` | `0` | Keep generated files after tests |

## Quick Test Commands

### Fast Verification (Code Generation Only)

```bash
SKIP_GO_EXECUTION=1 \
swipl -g "use_module('tests/core/test_go_generator'), run_tests" -t halt
```

### Full Test Suite

```bash
swipl -g "use_module('tests/core/test_go_generator'), run_tests" -t halt && \
swipl -g "use_module('tests/test_go_validation'), run_tests" -t halt && \
swipl -g "use_module('tests/test_go_json_advanced'), run_tests" -t halt
```

## Known Issues

1. **Windows native Go**: Path handling may require adjustments
2. **Large datasets**: Memory usage for fixpoint computation
3. **Parallel tests**: May have race conditions in certain edge cases

## Related Documentation

- [Book 6: Go Target](../../../education/book-06-go-target/README.md)
- [Go Target Implementation](../../../src/unifyweaver/targets/go_target.pl)
- [Quick Testing Guide](quick_testing.md)

## Changelog

- **v0.2** (Dec 2025): Initial test plan creation
