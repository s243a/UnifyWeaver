# Rust Target Test Plan - v0.2

**Version**: 0.2
**Date**: December 2025
**Status**: Draft
**Scope**: Rust code generation target testing

## Overview

This test plan covers the Rust target for UnifyWeaver, which generates high-performance, memory-safe Rust programs from Prolog predicates.

## Prerequisites

### System Requirements

- Rust 1.70+ (stable channel)
- `rustc` and `cargo` in PATH
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Rust installation
rustc --version
cargo --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify Rust code generation without compiling the generated code.

#### 1.1 Basic Compilation Tests

```bash
# Run all Rust target tests
swipl -g run_tests -t halt tests/test_rust_target.pl
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `test_facts_compilation` | Simple facts | Generates struct definitions |
| `test_rule_compilation` | Basic rules | Generates rule functions |
| `test_aggregation_compilation` | Sum aggregation | Generates aggregation code |
| `test_regex_compilation` | Pattern matching | Generates regex crate usage |
| `test_json_input_compilation` | JSON parsing | Generates serde_json usage |
| `test_json_output_compilation` | JSON output | Generates serialization |

### 2. Code Generation Verification

#### 2.1 Generated Code Structure

The Rust generator produces code with this structure:

```rust
use std::io::{self, BufRead, Write};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Fact {
    relation: String,
    args: Vec<String>,
}

fn get_initial_facts() -> Vec<Fact> {
    vec![
        Fact { relation: "parent".into(), args: vec!["john".into(), "mary".into()] },
        // ...
    ]
}

fn apply_rule_1(total: &HashSet<Fact>, delta: &HashSet<Fact>) -> HashSet<Fact> {
    // Rule implementation
}

fn solve() -> HashSet<Fact> {
    let mut total: HashSet<Fact> = get_initial_facts().into_iter().collect();
    let mut delta = total.clone();

    while !delta.is_empty() {
        let mut new_facts = HashSet::new();
        new_facts.extend(apply_rule_1(&total, &delta));
        delta = new_facts.difference(&total).cloned().collect();
        total.extend(delta.clone());
    }

    total
}
```

#### 2.2 Cargo Project Generation

```bash
# Test full Cargo project generation
swipl -l init.pl -g "
    assertz(test_fact(a, 1)),
    assertz(test_fact(b, 2)),
    compile_predicate_to_rust(test_fact/2, [output_project('/tmp/rust_test')], _),
    halt
" -t halt

# Verify project structure
ls -la /tmp/rust_test/
# Expected: Cargo.toml, src/main.rs
```

### 3. Integration Tests (Compilation + Execution)

These tests require Rust toolchain and compile/execute the generated code.

#### 3.1 Facts Compilation and Execution

```bash
# Generate and compile
cd /tmp/rust_test
cargo build --release 2>&1

# Execute
echo -e "a\t1\nb\t2" | ./target/release/rust_test
```

#### 3.2 Transitive Closure Test

```bash
swipl -l init.pl -g "
    assertz(parent(john, mary)),
    assertz(parent(mary, sue)),
    assertz((ancestor(X, Y) :- parent(X, Y))),
    assertz((ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z))),
    compile_predicate_to_rust(ancestor/2, [output_project('/tmp/rust_ancestor')], _),
    halt
" -t halt

cd /tmp/rust_ancestor
cargo build --release
echo -e "john\tmary\nmary\tsue" | ./target/release/rust_ancestor
```

**Expected Output**:
```
john	mary
mary	sue
john	sue
```

### 4. Feature-Specific Tests

#### 4.1 JSON Schema Tests

```prolog
:- json_schema(user_schema, [
    field(name, string),
    field(age, integer)
]).
```

```bash
# Test JSON input handling
swipl -l init.pl -g "
    json_schema(user_schema, [field(name, string), field(age, integer)]),
    assertz((user_info(Name, Age) :- json_record([name-Name, age-Age]))),
    compile_predicate_to_rust(user_info/2, [
        json_input(true),
        json_schema(user_schema),
        output_project('/tmp/rust_json')
    ], _),
    halt
" -t halt
```

#### 4.2 Regex Tests

```bash
# Test regex pattern matching
swipl -l init.pl -g "
    assertz((error_line(Line) :- input(Line), match(Line, '^ERROR'))),
    compile_predicate_to_rust(error_line/1, [
        output_project('/tmp/rust_regex')
    ], _),
    halt
" -t halt
```

#### 4.3 Aggregation Tests

```bash
# Test aggregation functions
swipl -l init.pl -g "
    assertz((total_sum(S) :- aggregation(sum), value(S))),
    compile_predicate_to_rust(total_sum/1, [
        aggregation(sum),
        output_project('/tmp/rust_agg')
    ], _),
    halt
" -t halt
```

## Test Matrix

### Feature Coverage

| Feature | Code Gen | Compile | Execute | Status |
|---------|----------|---------|---------|--------|
| Facts | ✓ | ✓ | ✓ | Stable |
| Rules | ✓ | ✓ | ✓ | Stable |
| Transitive closure | ✓ | ✓ | ✓ | Stable |
| Negation | ✓ | ⚠ | ⚠ | Testing |
| Aggregation | ✓ | ✓ | ⚠ | Testing |
| JSON I/O | ✓ | ✓ | ✓ | Stable |
| Regex | ✓ | ✓ | ✓ | Stable |
| Memory safety | ✓ | ✓ | ✓ | Guaranteed |

### Platform Coverage

| Platform | Code Gen | Compilation | Execution |
|----------|----------|-------------|-----------|
| Linux x86_64 | ✓ | ✓ | ✓ |
| Linux ARM64 | ✓ | ✓ | ✓ |
| macOS x86_64 | ✓ | ✓ | ✓ |
| macOS ARM64 | ✓ | ✓ | ✓ |
| Windows (WSL) | ✓ | ✓ | ✓ |
| Windows (native) | ✓ | ✓ | ✓ |

### Rust Toolchain Compatibility

| Version | Code Gen | Compilation | Notes |
|---------|----------|-------------|-------|
| 1.70 | ✓ | ✓ | Minimum supported |
| 1.75 | ✓ | ✓ | Full support |
| 1.80+ | ✓ | ✓ | Recommended |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_RUST_EXECUTION` | `0` | Skip Rust compilation/execution |
| `RUST_OUTPUT_DIR` | `/tmp/unifyweaver_rust` | Output directory |
| `RUST_KEEP_ARTIFACTS` | `0` | Keep generated projects |
| `CARGO_RELEASE` | `1` | Build in release mode |

## Quick Test Commands

### Fast Verification (Code Generation Only)

```bash
SKIP_RUST_EXECUTION=1 \
swipl -g run_tests -t halt tests/test_rust_target.pl
```

### Full Test Suite

```bash
# Code generation tests
swipl -g run_tests -t halt tests/test_rust_target.pl

# Verify generated projects compile
for dir in /tmp/rust_test /tmp/rust_ancestor /tmp/rust_json; do
    if [ -d "$dir" ]; then
        cd "$dir" && cargo build --release
    fi
done
```

## Generated Cargo.toml

The generator creates a Cargo.toml with appropriate dependencies:

```toml
[package]
name = "generated_program"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
regex = "1.10"

[profile.release]
opt-level = 3
lto = true
```

## Security Considerations

Rust target provides:

1. **Memory safety**: No buffer overflows, use-after-free
2. **Thread safety**: Data race prevention
3. **Type safety**: Compile-time type checking
4. **No undefined behavior**: Safe Rust guarantees

The firewall policy can enforce Rust-only compilation for high-security applications:

```prolog
:- firewall_mode(enforce).
:- allow(target(rust)).
:- deny(target(bash)).
:- deny(target(python)).
```

## Known Issues

1. **Compilation time**: Rust compilation is slower than other targets
2. **Binary size**: Release binaries may be large without stripping
3. **Cross-compilation**: Requires target-specific toolchains

## Related Documentation

- [Book 7: Rust Target](../../../education/book-07-rust-target/README.md)
- [Rust Target Implementation](../../../src/unifyweaver/targets/rust_target.pl)
- [Target Security](../../../education/book-08-security-firewall/04_target_security.md)
- [Quick Testing Guide](quick_testing.md)

## Changelog

- **v0.2** (Dec 2025): Initial test plan creation
