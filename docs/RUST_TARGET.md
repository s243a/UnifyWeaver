# Rust Target

**Status:** Complete (Phase 1-3)
**Module:** `src/unifyweaver/targets/rust_target.pl`

The Rust target compiles Prolog predicates into standalone, performant, and safe Rust programs.

## Features

### 1. Core Compilation (Standard Lib Only)
Compile simple predicates to single-file Rust programs (`.rs`) that require no external dependencies (compile with `rustc`).

*   **Facts:** Embedded as `HashSet` lookups.
*   **Single Rules:** Stream processing (stdin -> stdout) with field reordering.
*   **Constraints:** Arithmetic comparisons (`>`, `<`, etc.) on numeric fields.
*   **Aggregations:** `sum`, `count`, `avg`, `min`, `max`.

### 2. Advanced Features (Project Generation)
Compile complex predicates into full Cargo projects (`Cargo.toml`, `src/main.rs`) with automatic dependency management.

*   **Regex Support:**
    *   Usage: `match(Var, "Pattern")`.
    *   Dependencies: `regex` crate.
*   **JSON Input:**
    *   Usage: `json_input(true)`, `json_record([...])`, `json_path([...])`.
    *   Dependencies: `serde`, `serde_json`.
    *   Parses JSON Lines (NDJSON) from stdin.
*   **JSON Output:**
    *   Usage: `json_output(true)`.
    *   Dependencies: `serde`, `serde_json`.
    *   Serializes output to JSON Lines.

## Usage

### Basic Usage (Single File)

For simple tasks, use `compile_predicate_to_rust/3` and `write_rust_program/2`.

```prolog
:- use_module('src/unifyweaver/targets/rust_target').

% Define rule
adult(Name, Age) :- person(Name, Age), Age >= 18.

% Compile
?- compile_predicate_to_rust(adult/2, [field_delimiter(colon)], Code),
   write_rust_program(Code, 'adult.rs').
```

**Run:**
```bash
echo "alice:25" | rustc adult.rs && ./adult
```

### Advanced Usage (Project Generation)

For Regex or JSON, use `write_rust_project/2` to generate a buildable Cargo project.

```prolog
% Define rule with Regex
valid_email(Line) :- 
    input(Line),
    match(Line, "^[a-z0-9]+@[a-z0-9]+\\.[a-z]+$").

% Compile
?- compile_predicate_to_rust(valid_email/1, [], Code),
   write_rust_project(Code, 'output/email_validator').
```

**Build & Run:**
```bash
cd output/email_validator
cargo build --release
./target/release/email_validator < emails.txt
```

### JSON Processing

**JSON Input:**
```prolog
:- json_schema(user, [field(name, string), field(age, integer)]).

user_info(Name, Age) :-
    json_record([name-Name, age-Age]).

?- compile_predicate_to_rust(user_info/2, [json_input(true), json_schema(user)], Code),
   write_rust_project(Code, 'output/json_processor').
```

**JSON Output:**
```prolog
output_user(Name, Age) :- input_data(Name, Age).

?- compile_predicate_to_rust(output_user/2, [json_output(true)], Code),
   write_rust_project(Code, 'output/json_writer').
```

## API Reference

### `compile_predicate_to_rust/3`
```prolog
compile_predicate_to_rust(+Predicate, +Options, -RustCode)
```
**Options:**
*   `json_input(true)`: Enable JSON input mode.
*   `json_output(true)`: Enable JSON output mode.
*   `json_schema(SchemaName)`: Use defined schema for typing.
*   `field_delimiter(Delim)`: Set delimiter (colon, tab, comma, pipe).
*   `aggregation(Op)`: Generate aggregation logic (sum, count, min, max, avg).

### `write_rust_project/2`
```prolog
write_rust_project(+RustCode, +ProjectDir)
```
Generates `Cargo.toml` and `src/main.rs` in `ProjectDir`. Automatically detects dependencies (`regex`, `serde`, `serde_json`) from the code.