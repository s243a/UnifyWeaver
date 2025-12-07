<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Test Runner for Education Examples

## Overview

The education examples include an automatic test runner that validates your compiled scripts. After compiling Prolog predicates to Bash, you can generate and run tests automatically.

## Quick Start

### 1. Compile Your Scripts

```prolog
% Initialize environment
?- ['education/init'].

% Load compilers
?- use_module(unifyweaver(core/recursive_compiler)).
?- use_module(unifyweaver(core/stream_compiler)).

% Load your program
?- ['education/family_tree'].

% Compile parent facts
?- stream_compiler:compile_facts(parent, 2, [], BashCode),
   open('education/output/advanced/parent.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).

% Compile ancestor rule
?- compile_recursive(ancestor/2, [], BashCode),
   open('education/output/advanced/ancestor.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).
```

### 2. Generate Test Runner

```prolog
% Load test runner generator
?- use_module('src/unifyweaver/core/advanced/test_runner_inference').

% Generate tests
?- generate_test_runner_inferred(
    'education/output/advanced/test_runner.sh',
    [output_dir('education/output/advanced')]
).
```

### 3. Run Tests

```bash
cd education/output/advanced
bash test_runner.sh
```

## Understanding Test Output

### PASS/FAIL Indicators

The test runner shows clear PASS/FAIL results:

```
Test 1: Check isaac is ancestor of judah
isaac:judah
    Result: PASS

Test 3: Check ishmael is NOT ancestor of jacob (should fail)
    Result: PASS (correctly failed)
```

### Exit Code Testing

Functions ending in `_check` return exit codes (0=success, 1=failure):
- `ancestor_check "isaac" "judah"` → exit code 0 (success) → **PASS**
- `ancestor_check "ishmael" "jacob"` → exit code 1 (failure) → **PASS (correctly failed)**

Regular functions return both output and exit codes:
- `ancestor "isaac" "judah"` → prints `isaac:judah` and exits 0 → **PASS**
- `ancestor "ishmael" "jacob"` → prints nothing and exits 1 → **PASS (correctly failed)**

## How Test Inference Works

The test runner automatically infers appropriate tests based on function patterns:

### Family Tree / Transitive Closure

For predicates like `ancestor/2`:
- **1 argument**: Find all descendants
  - `ancestor abraham` → lists all descendants of abraham
- **2 arguments**: Check relationship
  - `ancestor isaac judah` → checks if isaac is ancestor of judah

Tests use meaningful names from the family tree (`education/family_tree.pl`):
- abraham, isaac, sarah, jacob, etc.

### Pattern-Based Testing

The system recognizes different recursion patterns and generates appropriate tests:

- **Linear recursion**: Base cases and recursive cases
- **Mutual recursion**: Valid and invalid inputs (e.g., even/odd)
- **Tree recursion**: Empty, single-node, and complex trees
