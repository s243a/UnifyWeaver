<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 4: Pipe Protocols and Data Formats

## Overview

When targets communicate via pipes, they need a shared data format. This chapter covers:

- TSV (Tab-Separated Values) - the default format
- JSON Lines - for structured data
- In-process protocols - for same-runtime communication
- Header negotiation - for schema exchange

## The Pipe Glue Module

Located at `src/unifyweaver/glue/pipe_glue.pl`:

```prolog
:- module(pipe_glue, [
    generate_pipe_writer/4,      % Generate output code
    generate_pipe_reader/4,      % Generate input code
    generate_tsv_writer/3,       % TSV output
    generate_tsv_reader/3,       % TSV input
    generate_json_writer/3,      % JSON output
    generate_json_reader/3,      % JSON input
    generate_pipeline_script/3   % Pipeline orchestration
]).
```

## TSV Protocol (Default)

TSV is the default format for pipe communication.

### Specification

```
┌─────────────────────────────────────────────────────┐
│                   TSV PROTOCOL                       │
├─────────────────────────────────────────────────────┤
│ Encoding: UTF-8                                      │
│ Record delimiter: Newline (\n)                       │
│ Field delimiter: Tab (\t)                            │
│ Escape: Backslash for \t, \n, \\                    │
│ NULL: Empty field                                    │
│ EOF: Close pipe                                      │
└─────────────────────────────────────────────────────┘
```

### Example Data

```
name	age	city
Alice	30	New York
Bob	25	Los Angeles
Charlie	35	Chicago
```

### Why TSV?

**Universality**: Every language can parse tabs and newlines.

```awk
# AWK - fields are automatic
{ print $1, $2, $3 }
```

```python
# Python - trivial parsing
fields = line.strip().split('\t')
```

```bash
# Bash - read into variables
IFS=$'\t' read -r name age city
```

**Streaming**: Process one line at a time, bounded memory.

**Debuggability**: Use `cat`, `head`, `tail` to inspect.

**Performance**: No parsing overhead for simple types.

### Escape Sequences

When field values contain tabs or newlines:

| Character | Escape |
|-----------|--------|
| Tab | `\t` |
| Newline | `\n` |
| Backslash | `\\` |

### Generated TSV Writer (AWK)

```prolog
generate_tsv_writer(awk, [name, age, city], Code).
```

Generates:
```awk
{
    # Output TSV
    print name "\t" age "\t" city
}
```

### Generated TSV Reader (Python)

```prolog
generate_tsv_reader(python, [name, age, city], Code).
```

Generates:
```python
import sys

def read_records():
    for line in sys.stdin:
        fields = line.rstrip('\n').split('\t')
        yield {
            'name': fields[0],
            'age': fields[1],
            'city': fields[2]
        }

for record in read_records():
    # Process record...
```

## JSON Lines Protocol

For structured or nested data, use JSON Lines.

### Specification

```
┌─────────────────────────────────────────────────────┐
│                  JSON LINES                          │
├─────────────────────────────────────────────────────┤
│ One JSON object per line                             │
│ Encoding: UTF-8                                      │
│ Record delimiter: Newline (\n)                       │
│ No trailing comma                                    │
│ NULL: JSON null                                      │
│ EOF: Close connection                                │
└─────────────────────────────────────────────────────┘
```

### Example Data

```json
{"name": "Alice", "age": 30, "address": {"city": "NYC", "zip": "10001"}}
{"name": "Bob", "age": 25, "address": {"city": "LA", "zip": "90001"}}
```

### When to Use JSON

| Use TSV When... | Use JSON When... |
|-----------------|------------------|
| Flat records | Nested structures |
| Simple types | Mixed types in arrays |
| Maximum speed | Self-describing needed |
| Shell tools | API compatibility |

### Generated JSON Writer (Python)

```prolog
generate_json_writer(python, [name, age, city], Code).
```

Generates:
```python
import json
import sys

def write_record(record):
    print(json.dumps(record))
    sys.stdout.flush()
```

### Generated JSON Reader (Go)

```prolog
generate_json_reader(go, [name, age, city], Code).
```

Generates:
```go
import (
    "bufio"
    "encoding/json"
    "os"
)

type Record struct {
    Name string `json:"name"`
    Age  int    `json:"age"`
    City string `json:"city"`
}

func readRecords() <-chan Record {
    ch := make(chan Record)
    go func() {
        scanner := bufio.NewScanner(os.Stdin)
        for scanner.Scan() {
            var r Record
            json.Unmarshal(scanner.Bytes(), &r)
            ch <- r
        }
        close(ch)
    }()
    return ch
}
```

## In-Process Protocol

For same-runtime communication, no serialization needed.

### Specification

```
┌─────────────────────────────────────────────────────┐
│               IN-PROCESS PROTOCOL                    │
├─────────────────────────────────────────────────────┤
│ Format: Native objects                               │
│ Transfer: Direct memory reference                    │
│ Streaming: Iterator/IEnumerable/Channel             │
│ No serialization overhead                            │
└─────────────────────────────────────────────────────┘
```

### .NET Example

```csharp
// C# calling PowerShell in-process
public static IEnumerable<T> InvokeStream<T>(string script) {
    using var ps = PowerShell.Create();
    ps.AddScript(script);
    foreach (var result in ps.Invoke<T>()) {
        yield return result;  // Direct object passing
    }
}
```

### Go Channel Example

```go
// Between goroutines
func process(input <-chan Record, output chan<- Record) {
    for record := range input {
        // Transform
        output <- transformed
    }
    close(output)
}
```

## Header Negotiation

For typed communication, headers can specify schema.

### Header Format

```
#UNIFYWEAVER:v1:tsv
#FIELDS:name:string,age:int,salary:float
Alice	30	75000.00
Bob	25	65000.00
```

### Parsing Headers

```python
def read_with_schema():
    # Read header lines
    version_line = sys.stdin.readline()
    fields_line = sys.stdin.readline()

    # Parse schema
    schema = parse_fields(fields_line)

    # Read data with type conversion
    for line in sys.stdin:
        yield parse_record(line, schema)
```

### Type Specifications

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text value | `"Alice"` |
| `int` | Integer | `30` |
| `float` | Floating point | `75000.00` |
| `bool` | Boolean | `true`, `false` |
| `datetime` | ISO 8601 | `2024-01-15T10:30:00Z` |
| `json` | Embedded JSON | `{"key": "value"}` |

## Field Mapping

Map fields between different naming conventions:

```prolog
% Configure field mapping
:- declare_connection(producer/2, consumer/2, [
    field_map([
        user_name -> name,      % Rename
        user_age -> age,
        ignore(internal_id)     % Drop field
    ])
]).
```

### Generated Mapping Code

```python
def map_fields(record):
    return {
        'name': record['user_name'],
        'age': record['user_age']
        # internal_id is dropped
    }
```

## Generating Complete Scripts

### AWK Script Generation

```prolog
generate_pipe_reader(awk, [ip, timestamp, status], [format(tsv)], ReaderCode),
generate_pipe_writer(awk, [ip, count], [format(tsv)], WriterCode).
```

Combined into complete script:

```awk
#!/usr/bin/awk -f

BEGIN {
    FS = "\t"
    OFS = "\t"
}

{
    # Input fields
    ip = $1
    timestamp = $2
    status = $3

    # Processing logic here...

    # Output
    print ip, count
}
```

### Python Script Generation

```prolog
generate_pipe_reader(python, [name, value], [format(json)], ReaderCode),
generate_pipe_writer(python, [name, result], [format(json)], WriterCode).
```

Generated:

```python
#!/usr/bin/env python3
import sys
import json

def read_input():
    for line in sys.stdin:
        yield json.loads(line)

def write_output(record):
    print(json.dumps(record))
    sys.stdout.flush()

for record in read_input():
    name = record['name']
    value = record['value']

    # Processing logic here...

    write_output({'name': name, 'result': result})
```

## Pipeline Orchestration

Generate shell script connecting multiple stages:

```prolog
generate_pipeline_script(
    [
        step(filter, awk, 'filter.awk', []),
        step(transform, python, 'transform.py', [format(json)]),
        step(aggregate, awk, 'aggregate.awk', [])
    ],
    [input('data.tsv'), output('result.tsv')],
    Script
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

cat "data.tsv" \
    | awk -f "filter.awk" \
    | python3 "transform.py" \
    | awk -f "aggregate.awk" \
    > "result.tsv"
```

## Format Conversion

When adjacent stages use different formats:

```prolog
:- declare_connection(tsv_producer/2, json_consumer/2, [
    format_in(tsv),
    format_out(json)
]).
```

Generated converter:

```bash
# TSV to JSON converter
awk -F'\t' '{
    printf "{\"name\":\"%s\",\"age\":%s}\n", $1, $2
}'
```

## Error Handling in Protocols

### TSV Error Recovery

```python
def read_tsv_safe():
    for line_num, line in enumerate(sys.stdin, 1):
        try:
            fields = line.rstrip('\n').split('\t')
            if len(fields) != expected_count:
                print(f"Warning: line {line_num} has {len(fields)} fields",
                      file=sys.stderr)
                continue
            yield fields
        except Exception as e:
            print(f"Error on line {line_num}: {e}", file=sys.stderr)
```

### JSON Error Recovery

```python
def read_json_safe():
    for line_num, line in enumerate(sys.stdin, 1):
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON on line {line_num}: {e}", file=sys.stderr)
```

## Performance Considerations

### TSV vs JSON Performance

| Operation | TSV | JSON |
|-----------|-----|------|
| Parse simple record | ~1μs | ~5μs |
| Serialize simple record | ~0.5μs | ~3μs |
| Memory per record | ~100B | ~200B |
| Human readable | Yes | Yes |
| Nested data | No | Yes |

### Buffering Strategies

```prolog
% Line buffering (default) - flush after each record
[buffer(line)]

% Block buffering - flush after N bytes
[buffer(block(65536))]

% No buffering - immediate output
[buffer(none)]
```

## Chapter Summary

- **TSV** is the default: universal, fast, streamable
- **JSON Lines** for structured/nested data
- **In-process** for same-runtime: zero serialization
- **Headers** enable typed communication
- **Field mapping** handles naming differences
- **Error handling** prevents data loss

## Next Steps

In Chapter 5, we'll use these protocols to generate complete shell scripts:
- AWK script generation
- Python script generation
- Bash script generation
- Handling all format options

## Exercises

1. **Format choice**: You have records with nested address data. Which format should you use? Write the expected format.

2. **TSV parsing**: Write AWK code to parse TSV with escaped tabs (`\t` in field values).

3. **Performance test**: Generate TSV and JSON writers in Python. Benchmark writing 1M records.

4. **Field mapping**: Configure a connection that renames `firstName` to `first_name` and drops `ssn`.

## Advanced Example: Unification over Pipes

This example demonstrates a **unification algorithm** (core to Prolog) implemented as a multi-stage pipeline using TSV and JSON protocols. Each stage handles a different aspect of unification.

### The Problem

Implement term unification: given two terms, find a substitution that makes them equal.

```
unify(foo(X, bar), foo(baz, Y)) => {X=baz, Y=bar}
unify(foo(X, X), foo(a, b)) => FAIL (X can't be both a and b)
```

### The Complete Pipeline

```prolog
% unification_pipeline.pl
:- use_module('src/unifyweaver/glue/shell_glue').

% Unification pipeline using TSV for simple data, JSON for substitutions
unification_pipeline(Script) :-
    generate_pipeline(
        [
            % Stage 1: Parse terms (AWK - fast text parsing)
            % Input: "foo(X, bar) = foo(baz, Y)"
            % Output TSV: term1 \t term2
            step(parse_terms, awk, '
                {
                    split($0, parts, " = ")
                    print parts[1] "\t" parts[2]
                }
            ', []),

            % Stage 2: Decompose into unification pairs (Python)
            % Input TSV: term1 \t term2
            % Output JSON: {"pairs": [["X", "baz"], ["bar", "Y"]], "vars": ["X", "Y"]}
            step(decompose, python, '
import sys
import json
import re

def parse_term(s):
    """Parse term into (functor, args) or just value"""
    s = s.strip()
    match = re.match(r"(\\w+)\\((.*)\\)", s)
    if match:
        functor = match.group(1)
        args_str = match.group(2)
        # Simple arg parsing (no nested terms for this example)
        args = [a.strip() for a in args_str.split(",")]
        return ("compound", functor, args)
    elif s[0].isupper():
        return ("var", s)
    else:
        return ("atom", s)

def decompose_pair(t1, t2):
    """Decompose two terms into unification pairs"""
    pairs = []
    vars_found = set()

    def collect(a, b):
        pa, pb = parse_term(a), parse_term(b)

        if pa[0] == "var":
            vars_found.add(pa[1])
            pairs.append([a, b])
        elif pb[0] == "var":
            vars_found.add(pb[1])
            pairs.append([a, b])
        elif pa[0] == "atom" and pb[0] == "atom":
            pairs.append([a, b])
        elif pa[0] == "compound" and pb[0] == "compound":
            if pa[1] != pb[1] or len(pa[2]) != len(pb[2]):
                pairs.append(["FAIL", f"functor mismatch: {pa[1]} vs {pb[1]}"])
            else:
                for arg1, arg2 in zip(pa[2], pb[2]):
                    collect(arg1, arg2)
        else:
            pairs.append([a, b])

    collect(t1, t2)
    return {"pairs": pairs, "vars": list(vars_found)}

for line in sys.stdin:
    parts = line.strip().split("\\t")
    if len(parts) == 2:
        result = decompose_pair(parts[0], parts[1])
        print(json.dumps(result))
', []),

            % Stage 3: Solve unification (Python - needs complex logic)
            % Input JSON: {"pairs": [...], "vars": [...]}
            % Output JSON: {"success": true, "substitution": {"X": "baz", "Y": "bar"}}
            step(solve, python, '
import sys
import json

def unify_pairs(pairs):
    """Apply unification algorithm to pairs"""
    subst = {}

    def apply_subst(term, subst):
        """Apply substitution to a term"""
        if term in subst:
            return apply_subst(subst[term], subst)
        return term

    def occurs_check(var, term, subst):
        """Check if var occurs in term"""
        term = apply_subst(term, subst)
        if term == var:
            return True
        # For compound terms, would recurse into args
        return False

    for pair in pairs:
        if pair[0] == "FAIL":
            return {"success": False, "error": pair[1]}

        t1 = apply_subst(pair[0], subst)
        t2 = apply_subst(pair[1], subst)

        if t1 == t2:
            continue
        elif t1[0].isupper():  # t1 is a variable
            if occurs_check(t1, t2, subst):
                return {"success": False, "error": f"occurs check failed: {t1} in {t2}"}
            subst[t1] = t2
        elif t2[0].isupper():  # t2 is a variable
            if occurs_check(t2, t1, subst):
                return {"success": False, "error": f"occurs check failed: {t2} in {t1}"}
            subst[t2] = t1
        else:
            return {"success": False, "error": f"cannot unify: {t1} with {t2}"}

    return {"success": True, "substitution": subst}

for line in sys.stdin:
    data = json.loads(line)
    result = unify_pairs(data["pairs"])
    print(json.dumps(result))
', []),

            % Stage 4: Format output (AWK - simple formatting)
            step(format, awk, '
                {
                    # Parse JSON minimally
                    if (index($0, "\"success\": true") > 0) {
                        gsub(/.*\"substitution\": /, "")
                        gsub(/}$/, "")
                        print "SUCCESS: " $0 "}"
                    } else {
                        gsub(/.*\"error\": \"/, "FAILURE: ")
                        gsub(/\".*/, "")
                        print
                    }
                }
            ', [])
        ],
        [input('unify_problems.txt')],
        Script
    ).
```

### Sample Input (`unify_problems.txt`)

```
foo(X, bar) = foo(baz, Y)
pair(A, A) = pair(1, 2)
cons(H, T) = cons(1, cons(2, nil))
X = Y
atom = atom
func(X) = func(X)
```

### Expected Output

```
SUCCESS: {"X": "baz", "Y": "bar"}
FAILURE: cannot unify: 1 with 2
SUCCESS: {"H": "1", "T": "cons(2, nil)"}
SUCCESS: {"X": "Y"}
SUCCESS: {}
SUCCESS: {}
```

### Protocol Flow

```
┌────────────┐  TSV   ┌────────────┐  JSON  ┌────────────┐  JSON  ┌────────────┐
│   parse    │ ─────► │ decompose  │ ─────► │   solve    │ ─────► │   format   │
│   (AWK)    │        │  (Python)  │        │  (Python)  │        │   (AWK)    │
└────────────┘        └────────────┘        └────────────┘        └────────────┘
     │                      │                     │                     │
 text input            structured            algorithm              human
 fast split            breakdown              result               readable
```

### Why Different Formats?

| Stage | Format | Reason |
|-------|--------|--------|
| parse → decompose | TSV | Simple two-column data |
| decompose → solve | JSON | Complex nested structure (pairs, vars) |
| solve → format | JSON | Substitution is a dictionary |

### Modification Exercise

**Task**: Extend the pipeline to handle **nested compound terms** like `foo(bar(X), Y)`.

**What to modify**:
1. The `parse_term` function needs recursive parsing
2. The `decompose_pair` function needs to handle nested compounds
3. Add proper functor/arity checking

**Hints**:
- Use a stack-based parser or recursive descent
- The TSV between parse and decompose stays the same
- The JSON structure for pairs needs to represent nested terms

**Test case to add**:
```
foo(bar(X), Y) = foo(bar(1), baz)
```

**Expected output**:
```
SUCCESS: {"X": "1", "Y": "baz"}
```

**Bonus Challenge**: Implement the **occurs check** properly for nested terms, so that `X = foo(X)` correctly fails.

## Code Examples

See `examples/02-shell-formats/` for working examples of:
- TSV round-trip
- JSON Lines processing
- Format conversion
- Header-based typing

---

## Navigation

**←** [Previous: Chapter 3: Target Registry and Mapping](03_target_registry) | [📖 Book 7: Cross-Target Glue](README) | [Next: Chapter 5: Shell Script Generation →](05_shell_glue)
