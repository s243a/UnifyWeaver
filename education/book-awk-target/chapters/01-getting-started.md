<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 1: Getting Started with the AWK Target

Welcome to the AWK target! This chapter introduces you to compiling Prolog predicates into self-contained AWK scripts.

## What is the AWK Target?

The AWK target transforms Prolog predicates into portable AWK scripts. AWK is a text-processing language designed for:

- Line-by-line data processing
- Field extraction and manipulation
- Pattern matching and filtering
- Aggregating data from streams

Unlike SQL or Go targets that work with databases or compiled binaries, AWK scripts process text streams, making them perfect for log analysis, CSV transformation, and Unix pipelines.

## Prerequisites

Before starting, verify you have:

1. **UnifyWeaver installed and working**
2. **AWK available on your system**

Check AWK availability:

```bash
awk --version
# or
gawk --version
```

Most Unix-like systems (Linux, macOS, BSD) have AWK pre-installed. On Windows, install via WSL, Cygwin, or Git Bash.

## The AWK Target API

The AWK target exports two main predicates:

```prolog
% Compile a predicate to AWK code
compile_predicate_to_awk(Predicate/Arity, Options, AWKCode)

% Write AWK code to a file
write_awk_script(Filename, AWKCode)
```

### Loading the Module

```prolog
:- use_module('src/unifyweaver/targets/awk_target').
```

## Your First AWK Script

Let's compile a simple predicate that filters employees by department.

### Step 1: Define the Predicate

```prolog
% file: first_awk.pl
:- use_module('src/unifyweaver/targets/awk_target').

% Define a simple filter predicate
% Assumes input: Name<TAB>Department<TAB>Salary
engineering_staff(Name, Dept) :-
    employee(Name, Dept, _),
    Dept = "Engineering".
```

### Step 2: Compile to AWK

```prolog
?- compile_predicate_to_awk(engineering_staff/2, [], AWK),
   write_awk_script('engineering.awk', AWK).
```

### Step 3: Create Test Data

Create a file `employees.tsv`:

```
Alice	Engineering	75000
Bob	Sales	65000
Carol	Engineering	80000
Dave	Marketing	70000
Eve	Engineering	72000
```

### Step 4: Run the Script

```bash
awk -f engineering.awk employees.tsv
```

Output:
```
Alice	Engineering
Carol	Engineering
Eve	Engineering
```

## Understanding the Generated AWK

The generated script follows this structure:

```awk
#!/usr/bin/awk -f

BEGIN {
    # Initialization
    FS = "\t"  # Field separator (tab by default)
}

{
    # Main processing block
    # Runs for each input line

    # Extract fields
    Name = $1
    Dept = $2
    Salary = $3

    # Apply constraints
    if (Dept == "Engineering") {
        # Output matching records
        print Name "\t" Dept
    }
}

END {
    # Cleanup (if needed)
}
```

## Compilation Options

The `compile_predicate_to_awk/3` predicate accepts various options:

### Record Format

```prolog
% Tab-separated (default)
compile_predicate_to_awk(pred/2, [record_format(tsv)], AWK)

% Comma-separated
compile_predicate_to_awk(pred/2, [record_format(csv)], AWK)

% JSON Lines
compile_predicate_to_awk(pred/2, [record_format(jsonl)], AWK)
```

### Field Separator

```prolog
% Use colon as separator
compile_predicate_to_awk(pred/2, [field_separator(':')], AWK)

% Use pipe
compile_predicate_to_awk(pred/2, [field_separator('|')], AWK)
```

### Include Header

```prolog
% Skip first line (header row)
compile_predicate_to_awk(pred/2, [include_header(true)], AWK)
```

### Unique Results

```prolog
% Remove duplicates from output
compile_predicate_to_awk(pred/2, [unique(true)], AWK)
```

## Example: Complete Workflow

Here's a complete example demonstrating the full workflow:

```prolog
% file: salary_filter.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

% Employees earning above a threshold
high_earners(Name, Salary) :-
    employee(Name, _, Salary),
    Salary > 70000.

% Run this to generate the AWK script
generate :-
    compile_predicate_to_awk(high_earners/2, [
        record_format(tsv),
        include_header(true)
    ], AWK),
    write_awk_script('high_earners.awk', AWK),
    format('Generated high_earners.awk~n').

:- initialization(generate, main).
```

Run it:

```bash
swipl salary_filter.pl
awk -f high_earners.awk employees.tsv
```

## AWK Script Portability

The generated AWK scripts are:

1. **Self-contained**: No external dependencies
2. **POSIX-compatible**: Work with any AWK implementation
3. **Portable**: Run on Linux, macOS, BSD, Windows (with AWK)

You can distribute just the `.awk` file without needing Prolog or UnifyWeaver at runtime.

## Common Patterns

### Pattern 1: Simple Filter

```prolog
active_users(Name) :-
    user(Name, Status),
    Status = "active".
```

### Pattern 2: Field Selection

```prolog
name_and_email(Name, Email) :-
    contact(Name, _, Email, _).
```

### Pattern 3: Computed Values

```prolog
with_bonus(Name, Total) :-
    employee(Name, _, Salary),
    Total is Salary * 1.1.
```

## Exercises

1. **Basic Filter**: Create a predicate that filters products with price > 100 and compile it to AWK.

2. **Department Report**: Write a predicate that selects employees from the "Sales" department, showing name and salary.

3. **Custom Separator**: Compile a predicate to process colon-separated data (like `/etc/passwd`).

4. **Multiple Conditions**: Create a predicate that filters employees who are in "Engineering" AND earn more than 70000.

## Summary

In this chapter, you learned:

- What the AWK target does and when to use it
- The main API: `compile_predicate_to_awk/3` and `write_awk_script/2`
- Compilation options: `record_format`, `field_separator`, `include_header`, `unique`
- How to generate and run AWK scripts
- Basic filter patterns

## Next Chapter

In Chapter 2, we'll explore facts and filtering in detail, learning how Prolog facts compile to AWK associative arrays.
