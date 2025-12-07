<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# The AWK Target Book

A practical guide to compiling Prolog predicates to self-contained AWK scripts using UnifyWeaver's AWK target.

## What You'll Learn

This book teaches you how to use UnifyWeaver to generate portable AWK scripts from Prolog predicates. AWK is a powerful text-processing language available on virtually every Unix-like system, making it ideal for data pipelines and log analysis.

## Prerequisites

- Basic understanding of Prolog syntax
- Familiarity with command-line tools
- AWK installed on your system (usually pre-installed on Linux/macOS)
- UnifyWeaver set up and working

## Chapter Overview

### [Chapter 1: Getting Started](chapters/01-getting-started.md)
Introduction to the AWK target, installation verification, and your first compiled script.

### [Chapter 2: Facts and Filtering](chapters/02-facts-filtering.md)
Learn to compile Prolog facts to AWK associative arrays and filter data with constraints.

### [Chapter 3: Rules and Constraints](chapters/03-rules-constraints.md)
Compile single and multiple rules with arithmetic and logical constraints.

### [Chapter 4: Aggregations](chapters/04-aggregations.md)
Use built-in aggregation operations: sum, count, max, min, and avg.

### [Chapter 5: Tail Recursion](chapters/05-tail-recursion.md)
Compile tail-recursive predicates to efficient AWK while loops.

### [Chapter 6: Regex Pattern Matching](chapters/06-regex-matching.md)
Use the `match/2`, `match/3`, and `match/4` predicates for regex operations.

### [Chapter 7: Practical Applications](chapters/07-practical-applications.md)
Real-world examples: log analysis, data transformation, and pipeline integration.

## Key Features of the AWK Target

| Feature | Description |
|---------|-------------|
| Facts | Compiled to AWK associative arrays |
| Single Rules | Direct translation with constraint evaluation |
| Multiple Rules | Combined with union-style logic |
| Aggregations | sum, count, max, min, avg operations |
| Tail Recursion | Compiled to efficient while loops |
| Regex Matching | Full support via `match/2`, `match/3`, `match/4` |
| Input Formats | TSV, CSV, JSONL record formats |

## Input Format Options

The AWK target supports multiple input formats:

```prolog
% Tab-separated values (default)
compile_predicate_to_awk(pred/2, [record_format(tsv)], AWK)

% Comma-separated values
compile_predicate_to_awk(pred/2, [record_format(csv)], AWK)

% JSON Lines (one JSON object per line)
compile_predicate_to_awk(pred/2, [record_format(jsonl)], AWK)

% Custom field separator
compile_predicate_to_awk(pred/2, [field_separator(':')], AWK)
```

## Quick Example

**Prolog predicate:**
```prolog
high_salary(Name, Salary) :-
    employee(Name, _, Salary),
    Salary > 50000.
```

**Compile to AWK:**
```prolog
?- compile_predicate_to_awk(high_salary/2, [], AWK),
   write_awk_script('high_salary.awk', AWK).
```

**Run the script:**
```bash
awk -f high_salary.awk employees.tsv
```

## Limitations

The AWK target is designed for stream processing and has some constraints:

- **No deep recursion**: Only tail recursion is supported
- **No nested data structures**: Flat data only
- **No mutual recursion**: Single recursive predicates only
- **No backtracking**: First match semantics
- **No complex unification**: Simple variable binding only

These limitations are inherent to AWK's streaming model and are not bugs.

## Why AWK?

- **Ubiquitous**: Pre-installed on virtually all Unix-like systems
- **Fast**: Optimized for line-by-line text processing
- **Portable**: Same script works across platforms
- **Pipeline-friendly**: Designed for Unix pipes
- **No dependencies**: Self-contained scripts

## Getting Help

- Check the [UnifyWeaver documentation](../../../docs/)
- Review the [AWK target examples](../../../docs/AWK_TARGET_EXAMPLES.md)
- See the [AWK target status](../../../docs/AWK_TARGET_STATUS.md)

## License

This educational content is dual-licensed under MIT (code) and CC-BY-4.0 (documentation).
