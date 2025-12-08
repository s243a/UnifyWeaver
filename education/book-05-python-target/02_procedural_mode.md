<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 2: Procedural Mode

Procedural mode is the default compilation mode for the Python target. It translates Prolog predicates into Python generator functions that process records in a streaming fashion, yielding results as they're computed.

## How Procedural Mode Works

In procedural mode, UnifyWeaver translates Prolog rules into Python generators:

- **Facts** become constant yields
- **Rules** become nested generator loops
- **Constraints** become `if` conditions
- **Recursion** uses memoization or loop optimization

### Basic Translation Pattern

**Prolog:**
```prolog
process(X, Y) :- input(X), transform(X, Y).
```

**Python:**
```python
def _clause_0(record):
    x = record.get('arg0')
    if x is not None:
        y = transform(x)
        yield {'arg0': x, 'arg1': y}
```

## Streaming Architecture

Generated scripts follow a consistent streaming pattern:

```python
#!/usr/bin/env python3
import sys
import json
import functools
from typing import Dict, Iterator, Any

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    """Main entry point - processes records and yields results."""
    seen = set()
    for record in records:
        for result in _clause_0(record):
            key = json.dumps(result, sort_keys=True)
            if key not in seen:
                seen.add(key)
                yield result

def _clause_0(record: Dict) -> Iterator[Dict]:
    """Implements the first clause of the predicate."""
    # ... predicate logic ...
    yield result

def main():
    """Read JSONL from stdin, process, write JSONL to stdout."""
    def read_records():
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield json.loads(line)

    for result in process_stream(read_records()):
        print(json.dumps(result))

if __name__ == '__main__':
    main()
```

## Compiling Facts

Facts compile to simple constant yields:

**Prolog:**
```prolog
color(red).
color(green).
color(blue).
```

**Python:**
```python
def _clause_0(record: Dict) -> Iterator[Dict]:
    yield {'arg0': 'red'}

def _clause_1(record: Dict) -> Iterator[Dict]:
    yield {'arg0': 'green'}

def _clause_2(record: Dict) -> Iterator[Dict]:
    yield {'arg0': 'blue'}

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    seen = set()
    for record in records:
        for clause in [_clause_0, _clause_1, _clause_2]:
            for result in clause(record):
                key = json.dumps(result, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    yield result
```

## Compiling Rules with Constraints

Rules with arithmetic and comparison constraints:

**Prolog:**
```prolog
adult(Name, Age) :-
    person(Name, Age),
    Age >= 18.
```

**Python:**
```python
def _clause_0(record: Dict) -> Iterator[Dict]:
    name = record.get('name')
    age = record.get('age')
    if name is not None and age is not None:
        if age >= 18:
            yield {'name': name, 'age': age}
```

## Compiling Arithmetic

Arithmetic expressions using `is/2` are translated to Python expressions:

**Prolog:**
```prolog
double(X, Y) :- Y is X * 2.
```

**Python:**
```python
def _clause_0(record: Dict) -> Iterator[Dict]:
    x = record.get('arg0')
    if x is not None:
        y = x * 2
        yield {'arg0': x, 'arg1': y}
```

### Supported Arithmetic Operations

| Prolog | Python | Description |
|--------|--------|-------------|
| `X + Y` | `x + y` | Addition |
| `X - Y` | `x - y` | Subtraction |
| `X * Y` | `x * y` | Multiplication |
| `X / Y` | `x / y` | Division |
| `X // Y` | `x // y` | Integer division |
| `X mod Y` | `x % y` | Modulo |
| `X ** Y` | `x ** y` | Exponentiation |
| `abs(X)` | `abs(x)` | Absolute value |
| `min(X, Y)` | `min(x, y)` | Minimum |
| `max(X, Y)` | `max(x, y)` | Maximum |

## Multiple Clauses

Multiple clauses for a predicate are tried in order:

**Prolog:**
```prolog
grade(Score, 'A') :- Score >= 90.
grade(Score, 'B') :- Score >= 80, Score < 90.
grade(Score, 'C') :- Score >= 70, Score < 80.
grade(Score, 'F') :- Score < 70.
```

**Python:**
```python
def _clause_0(record: Dict) -> Iterator[Dict]:
    score = record.get('score')
    if score is not None and score >= 90:
        yield {'score': score, 'grade': 'A'}

def _clause_1(record: Dict) -> Iterator[Dict]:
    score = record.get('score')
    if score is not None and score >= 80 and score < 90:
        yield {'score': score, 'grade': 'B'}

def _clause_2(record: Dict) -> Iterator[Dict]:
    score = record.get('score')
    if score is not None and score >= 70 and score < 80:
        yield {'score': score, 'grade': 'C'}

def _clause_3(record: Dict) -> Iterator[Dict]:
    score = record.get('score')
    if score is not None and score < 70:
        yield {'score': score, 'grade': 'F'}

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    seen = set()
    for record in records:
        for clause in [_clause_0, _clause_1, _clause_2, _clause_3]:
            for result in clause(record):
                key = json.dumps(result, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    yield result
```

## Disjunction (`;`)

Disjunction in rule bodies compiles to separate branches:

**Prolog:**
```prolog
category(X, small) :- (X < 10 ; X =:= 0).
category(X, large) :- X >= 10.
```

**Python:**
```python
def _clause_0(record: Dict) -> Iterator[Dict]:
    x = record.get('arg0')
    if x is not None:
        # First disjunct
        if x < 10:
            yield {'arg0': x, 'arg1': 'small'}
        # Second disjunct
        if x == 0:
            yield {'arg0': x, 'arg1': 'small'}
```

## Input/Output Formats

### JSONL (Default)

JSON Lines format - one JSON object per line:

```bash
echo '{"x": 5}
{"x": 10}
{"x": 15}' | python3 double.py
```

Output:
```json
{"x": 5, "result": 10}
{"x": 10, "result": 20}
{"x": 15, "result": 30}
```

### NUL-Delimited JSON

For binary-safe streaming:

```prolog
compile_predicate_to_python(pred/2, [record_format(nul_json)], Code).
```

```python
def read_records():
    buffer = b''
    while True:
        chunk = sys.stdin.buffer.read(4096)
        if not chunk:
            break
        buffer += chunk
        while b'\0' in buffer:
            record, buffer = buffer.split(b'\0', 1)
            if record:
                yield json.loads(record.decode('utf-8'))
```

## Native XML Input

For processing XML files directly without external piping:

**Prolog:**
```prolog
compile_predicate_to_python(process_products/1, [
    input_source(xml('data.xml', ['product']))
], Code).
```

**Generated Python:**
```python
from lxml import etree

def read_xml_records(filename, tags):
    """Stream XML elements as dictionaries."""
    for event, elem in etree.iterparse(filename, events=['end'], tag=tags):
        record = {}
        # Extract attributes
        for key, value in elem.attrib.items():
            record[f'@{key}'] = value
        # Extract text
        if elem.text:
            record['text'] = elem.text.strip()
        # Extract children
        for child in elem:
            record[child.tag] = child.text
        yield record
        elem.clear()  # Free memory

def main():
    for result in process_stream(read_xml_records('data.xml', 'product')):
        print(json.dumps(result))
```

## Deduplication

Generated scripts automatically deduplicate results using a seen set:

```python
def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    seen = set()
    for record in records:
        for result in _clause_0(record):
            # Create hashable key
            key = json.dumps(result, sort_keys=True)
            if key not in seen:
                seen.add(key)
                yield result
```

This ensures each unique result is yielded exactly once, matching Prolog's set semantics.

## Example: Complete Pipeline

Let's build a complete example that filters, transforms, and aggregates data:

**Prolog:**
```prolog
% Filter adults
adult(Name, Age) :- person(Name, Age), Age >= 18.

% Categorize by age
category(Name, 'young') :- adult(Name, Age), Age < 30.
category(Name, 'middle') :- adult(Name, Age), Age >= 30, Age < 50.
category(Name, 'senior') :- adult(Name, Age), Age >= 50.
```

**Compilation:**
```prolog
?- compile_predicate_to_python(category/2, [], Code),
   open('categorize.py', write, S),
   write(S, Code),
   close(S).
```

**Usage:**
```bash
echo '{"name": "Alice", "age": 25}
{"name": "Bob", "age": 45}
{"name": "Carol", "age": 17}
{"name": "Dave", "age": 62}' | python3 categorize.py
```

**Output:**
```json
{"name": "Alice", "category": "young"}
{"name": "Bob", "category": "middle"}
{"name": "Dave", "category": "senior"}
```

Note: Carol (age 17) is filtered out because she doesn't match `adult/2`.

## Performance Considerations

### When Procedural Mode Excels

- Simple transformations and filters
- Arithmetic computations
- Shallow recursion (with memoization)
- Streaming large datasets

### When to Use Generator Mode Instead

- Deep recursion (avoid stack overflow)
- Transitive closure queries
- Graph algorithms
- Unknown recursion depth

## Summary

Procedural mode provides:

- **Streaming semantics** - Process records one at a time
- **Generator-based** - Memory-efficient iteration
- **Automatic deduplication** - Set semantics
- **Constraint support** - Arithmetic and comparisons
- **Multiple formats** - JSONL, NUL-JSON, XML

In the next chapter, we'll explore Generator Mode for fixpoint evaluation of recursive queries.

---

## Navigation

**←** [Previous: Chapter 1: Introduction to the Python Target](01_introduction) | [📖 Book 5: Python Target](./) | [Next: Chapter 3: Generator Mode - Fixpoint Evaluation →](03_generator_mode)
