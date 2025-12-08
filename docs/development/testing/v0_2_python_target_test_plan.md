# Python Target Test Plan - v0.2

**Version**: 0.2
**Date**: December 2025
**Status**: Draft
**Scope**: Python code generation target testing

## Overview

This test plan covers the Python target for UnifyWeaver, which generates Python programs from Prolog predicates using semi-naive fixpoint evaluation with FrozenDict-based fact representation.

## Prerequisites

### System Requirements

- Python 3.8+ installed
- `python3` command in PATH
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Python installation
python3 --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify Python code generation without executing the generated code.

#### 1.1 Basic Generator Tests

```bash
# Run all Python generator tests
swipl -g "[tests/core/test_python_generator], run_tests(python_generator)" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `simple_facts_generator` | Basic fact compilation | Generates FrozenDict class, rule functions |
| `transitive_closure_generator` | Recursive rules | Generates fixpoint loop with `while delta:` |
| `disjunction_generator` | OR rules | Generates disjunctive rule handling |
| `complex_disjunction_generator` | Multi-way joins | Generates nested loops |
| `generator_execution` | End-to-end (conditional) | Full pipeline test |

#### 1.2 Python Target Tests

```bash
swipl -g "use_module('tests/core/test_python_target'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Mode selection | Query vs generator | Correct mode dispatch |
| Field delimiter | TSV/CSV output | Proper delimiter handling |

#### 1.3 Semantic Compilation Tests

```bash
swipl -g "use_module('tests/core/test_python_semantic_compilation'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Fact to Python | Simple facts | Generates initial set |
| Rule to Python | Basic rules | Generates apply functions |

#### 1.4 Execution Tests

```bash
swipl -g "use_module('tests/core/test_python_execution'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Script execution | Run generated Python | Correct output |
| Error handling | Invalid input | Proper error messages |

### 2. Integration Tests

#### 2.1 XML Integration Tests

```bash
swipl -g "use_module('tests/core/test_python_xml_integration'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| XML parsing | ElementTree usage | Generates proper imports |
| XPath queries | Node selection | Generates findall/find |

#### 2.2 LLM Integration Tests

```bash
swipl -g "use_module('tests/core/test_python_llm'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| LLM-compatible output | JSON format | Valid JSON for LLM consumption |

### 3. Data Source Tests (Python Source Plugin)

The Python source plugin allows embedding Python code in Prolog predicates.

```bash
swipl -g "use_module('tests/core/test_python_source'), test_python_source" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| Inline Python | Heredoc execution | Correct output capture |
| Python file | External script | File execution |
| SQLite via Python | Database access | Query results |

### 4. End-to-End Tests

#### 4.1 Full Pipeline Test

```bash
# Generate Python code and execute
cd /root/UnifyWeaver
swipl -l init.pl -g "
    assertz(edge(a, b)),
    assertz(edge(b, c)),
    assertz((path(X, Y) :- edge(X, Y))),
    assertz((path(X, Z) :- edge(X, Y), path(Y, Z))),
    compile_predicate_to_python(path/2, [mode(generator)], Code),
    open('/tmp/path_test.py', write, S),
    write(S, Code),
    close(S),
    halt
" -t halt

# Execute
python3 /tmp/path_test.py <<< $'a\tb\nb\tc'
```

**Expected Output**:
```
a	b
b	c
a	c
```

#### 4.2 Generator Mode Pipeline

```bash
# Test complete generator mode flow
echo -e "a\tb\nb\tc\nc\td" | python3 /tmp/path_test.py
```

**Expected Output**:
```
a	b
b	c
c	d
a	c
b	d
a	d
```

## Test Matrix

### Feature Coverage

| Feature | Code Gen | Execute | Status |
|---------|----------|---------|--------|
| Facts | ✓ | ✓ | Stable |
| Rules | ✓ | ✓ | Stable |
| Transitive closure | ✓ | ✓ | Stable |
| Negation | ✓ | ⚠ | Testing |
| Disjunction | ✓ | ✓ | Stable |
| N-way joins | ✓ | ✓ | Stable |
| Aggregation | ✓ | ⚠ | Experimental |
| FrozenDict | ✓ | ✓ | Stable |

### Python Version Compatibility

| Version | Code Gen | Execution | Notes |
|---------|----------|-----------|-------|
| 3.8 | ✓ | ✓ | Minimum supported |
| 3.9 | ✓ | ✓ | Full support |
| 3.10 | ✓ | ✓ | Full support |
| 3.11 | ✓ | ✓ | Full support |
| 3.12 | ✓ | ✓ | Recommended |

### Platform Coverage

| Platform | Code Gen | Execution |
|----------|----------|-----------|
| Linux | ✓ | ✓ |
| macOS | ✓ | ✓ |
| Windows (WSL) | ✓ | ✓ |
| Windows (native) | ✓ | ✓ |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_PYTHON_EXECUTION` | `0` | Skip Python execution tests |
| `PYTHON_OUTPUT_DIR` | `/tmp/unifyweaver_py` | Output directory |
| `PYTHON_KEEP_ARTIFACTS` | `0` | Keep generated files |
| `PYTHON_VERSION` | `python3` | Python interpreter |

## Quick Test Commands

### Fast Verification (Code Generation Only)

```bash
SKIP_PYTHON_EXECUTION=1 \
swipl -g "[tests/core/test_python_generator], run_tests(python_generator)" -t halt
```

### Full Test Suite

```bash
swipl -g "[tests/core/test_python_generator], run_tests(python_generator)" -t halt && \
swipl -g "use_module('tests/core/test_python_target'), run_tests" -t halt && \
swipl -g "use_module('tests/core/test_python_execution'), run_tests" -t halt
```

### Python Source Plugin Tests

```bash
swipl -g "use_module('tests/core/test_python_source'), test_python_source" -t halt
```

## Generated Code Structure

The Python generator produces code with this structure:

```python
#!/usr/bin/env python3
"""Generated by UnifyWeaver"""
import sys
from typing import FrozenSet

class FrozenDict:
    """Immutable dictionary for fact representation"""
    def __init__(self, items):
        self._items = tuple(sorted(items))
        self._hash = hash(self._items)
    ...

def _apply_rule_1(total, delta):
    """Rule: path(X, Y) :- edge(X, Y)"""
    ...

def _apply_rule_2(total, delta):
    """Rule: path(X, Z) :- edge(X, Y), path(Y, Z)"""
    ...

def process_stream_generator(input_facts):
    """Semi-naive fixpoint evaluation"""
    total = set(input_facts)
    delta = set(input_facts)
    while delta:
        new_facts = set()
        for rule in [_apply_rule_1, _apply_rule_2]:
            new_facts |= rule(total, delta)
        delta = new_facts - total
        total |= delta
    return total
```

## Known Issues

1. **Large transitive closures**: Memory intensive for deep recursion
2. **String handling**: Unicode edge cases in some scenarios
3. **Aggregation**: Limited support compared to Go/Rust targets

## Related Documentation

- [Book 5: Python Target](../../../education/book-05-python-target/README.md)
- [Python Target Implementation](../../../src/unifyweaver/targets/python_target.pl)
- [Python Source Plugin](../../../src/unifyweaver/sources/python_source.pl)
- [Quick Testing Guide](quick_testing.md)

## Changelog

- **v0.2** (Dec 2025): Initial test plan creation
