# C++ Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: C++ code generation target testing

## Overview

This test plan covers the C++ target for UnifyWeaver, which generates modern C++ code with STL containers, smart pointers, and RAII patterns.

## Prerequisites

### System Requirements

- GCC 12+ or Clang 15+ (C++20 support)
- CMake 3.20+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify C++ compiler
g++ --version
# or
clang++ --version

# Check C++20 support
echo 'int main() { return 0; }' | g++ -std=c++20 -x c++ -

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_cpp_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `class_generation` | Class definitions | Correct class syntax |
| `template_generation` | Template classes | Template syntax |
| `namespace_handling` | Namespace usage | namespace blocks |
| `smart_pointers` | unique_ptr/shared_ptr | RAII patterns |
| `stl_containers` | vector, set, map | STL usage |

#### 1.2 Modern C++ Features

```bash
swipl -g "use_module('tests/core/test_cpp_modern'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `auto_keyword` | Type inference | auto usage |
| `range_based_for` | for(auto& x : c) | Range loops |
| `lambda_expressions` | [] captures | Lambda syntax |
| `structured_bindings` | auto [a, b] | C++17 bindings |
| `concepts` | requires clauses | C++20 concepts |

### 2. Compilation Tests

#### 2.1 Standard Compliance

```bash
./tests/integration/test_cpp_standards.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `cpp17_compile` | -std=c++17 | Compiles |
| `cpp20_compile` | -std=c++20 | Compiles |
| `warnings_clean` | -Wall -Wextra | No warnings |

### 3. Generated Code Structure

```cpp
#pragma once

#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>
#include <memory>

namespace unifyweaver {

struct Fact {
    std::string relation;
    std::vector<std::string> args;

    bool operator==(const Fact& other) const = default;
};

struct FactHash {
    std::size_t operator()(const Fact& f) const;
};

class Query {
public:
    using FactSet = std::unordered_set<Fact, FactHash>;

    Query() { init_facts(); }
    FactSet solve();

private:
    FactSet facts_;
    FactSet delta_;

    void init_facts();
    FactSet apply_rules(const FactSet& total, const FactSet& delta);
};

} // namespace unifyweaver
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/cpp_target'),
    compile_to_cpp(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_cpp_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CXX` | C++ compiler | `g++` |
| `CXXFLAGS` | Compiler flags | `-std=c++20 -O2` |
| `SKIP_CPP_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Compilation time**: Templates increase compile time
2. **Binary size**: STL can increase binary size
3. **ABI compatibility**: Different compilers may have ABI issues
