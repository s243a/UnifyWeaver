<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 3: Project Generation

For more complex applications involving external crates (libraries), UnifyWeaver can generate full Cargo projects.

## Why Project Generation?

Single-file compilation (`rustc`) works for standard library features. However, features like Regex matching and JSON processing require external crates (`regex`, `serde`). UnifyWeaver handles this by generating a `Cargo.toml` file.

## Generating a Project

Use `write_rust_project/2` instead of `write_rust_program/2`.

```prolog
:- use_module('src/unifyweaver/targets/rust_target').

% Rule using Regex
valid_email(Line) :-
    input(Line),
    match(Line, "^[a-z0-9]+@[a-z0-9]+\\.[a-z]+$").

compile_project :-
    compile_predicate_to_rust(valid_email/1, [], Code),
    write_rust_project(Code, 'output/email_validator').
```

This creates:
-   `output/email_validator/Cargo.toml`
-   `output/email_validator/src/main.rs`

## Building and Running

```bash
cd output/email_validator
cargo build --release
./target/release/email_validator < emails.txt
```

UnifyWeaver automatically detects the need for the `regex` crate and adds it to `Cargo.toml`.

---

## Navigation

**←** [Previous: Chapter 2: Basic Compilation](02_basic_compilation) | [📖 Book 9: Rust Target](README) | [Next: Chapter 4: Advanced Features →](04_advanced_features)
