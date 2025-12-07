<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 2: Basic Compilation

In this chapter, we will cover the fundamentals of compiling Prolog predicates to Go.

## Compiling Facts

The simplest use case is compiling a set of facts into a lookup table.

### Prolog Source (`facts.pl`)

```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Define facts
user(john, 25).
user(jane, 30).
user(bob, 28).

% Compile to Go
compile_facts :-
    compile_predicate_to_go(user/2, [], Code),
    write_go_program(Code, 'user.go').
```

### Generated Go Code

The compiler generates a Go program that stores these facts in a `map[string]bool` for efficient O(1) lookups.

```go
package main
import "fmt"

func main() {
    facts := map[string]bool{
        "john:25": true,
        "jane:30": true,
        "bob:28": true,
    }
    // ... iteration logic ...
}
```

### Running the Program

```bash
swipl -g compile_facts -t halt facts.pl
go build user.go
./user
# Output:
# john:25
# jane:30
# bob:28
```

## Compiling Rules

For rules, the Go target generates a stream processor that reads from standard input (stdin) and writes to standard output (stdout).

### Prolog Source (`rules.pl`)

```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Rule: Swap fields
swap(Y, X) :- input(X, Y).

compile_rules :-
    compile_predicate_to_go(swap/2, [], Code),
    write_go_program(Code, 'swap.go').
```

### Usage

The generated binary reads line-by-line from stdin.

```bash
swipl -g compile_rules -t halt rules.pl
go build swap.go

echo -e "alice:bob\ncharlie:david" | ./swap
# Output:
# bob:alice
# david:charlie
```

## Options

You can customize the compilation with options:

-   `field_delimiter(Delim)`: Change the separator (default: `colon`). Options: `tab`, `comma`, `pipe`.
-   `unique(Boolean)`: Enable/disable automatic deduplication (default: `true`).

```prolog
compile_predicate_to_go(swap/2, [field_delimiter(tab)], Code).
```

---

## Navigation

**←** [Previous: Chapter 1: Introduction to the Go Target](01_introduction) | [📖 Book 6: Go Target](./) | [Next: Chapter 3: Advanced Features →](03_advanced_features)
