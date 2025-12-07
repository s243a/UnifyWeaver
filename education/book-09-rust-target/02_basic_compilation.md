<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 2: Basic Compilation

In this chapter, we will compile simple Prolog predicates into single-file Rust programs.

## Compiling Facts

### Prolog Source (`facts.pl`)

```prolog
:- use_module('src/unifyweaver/targets/rust_target').

% Define facts
user(john, 25).
user(jane, 30).

% Compile to Rust
compile_facts :-
    compile_predicate_to_rust(user/2, [], Code),
    write_rust_program(Code, 'user.rs').
```

### Generated Rust Code

The compiler generates a Rust program using `HashSet` for facts.

```rust
use std::collections::HashSet;

fn main() {
    let mut facts = HashSet::new();
    facts.insert("john:25");
    facts.insert("jane:30");
    // ... iteration logic ...
}
```

### Running the Program

```bash
swipl -g compile_facts -t halt facts.pl
rustc user.rs
./user
# Output:
# john:25
# jane:30
```

## Compiling Rules

Rules are compiled into stream processors that read from stdin.

### Prolog Source (`rules.pl`)

```prolog
:- use_module('src/unifyweaver/targets/rust_target').

% Rule: Filter adults
adult(Name, Age) :- person(Name, Age), Age >= 18.

compile_rules :-
    compile_predicate_to_rust(adult/2, [field_delimiter(colon)], Code),
    write_rust_program(Code, 'adult.rs').
```

### Usage

```bash
swipl -g compile_rules -t halt rules.pl
rustc adult.rs
echo -e "alice:25\nbob:10" | ./adult
# Output:
# alice:25
```

---

## Navigation

**←** [Previous: Chapter 1: Introduction to the Rust Target](01_introduction) | [📖 Book 9: Rust Target](./) | [Next: Chapter 3: Project Generation →](03_project_generation)
