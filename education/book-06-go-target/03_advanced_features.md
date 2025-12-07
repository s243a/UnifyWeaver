<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 3: Advanced Features

The Go target supports advanced features for data extraction, filtering, and aggregation.

## Regex Matching

You can use the `match/2` predicate to filter records using Regular Expressions.

```prolog
% Filter lines containing 'ERROR'
error_log(Line) :-
    log(Line),
    match(Line, 'ERROR').
```

### Capture Groups

Extract data using regex capture groups with `match/4`.

```prolog
% Extract date and level from log
% Input: "2025-01-01 ERROR: Something happened"
parse_log(Date, Level) :-
    log(Line),
    match(Line, '^([0-9-]+) ([A-Z]+):', auto, [Date, Level]).
```

The compiler translates this into efficient Go `regexp` calls.

## Constraints

UnifyWeaver supports numeric constraints that are compiled into Go comparison operators.

```prolog
% Filter adults
adult(Name, Age) :-
    person(Name, Age),
    Age >= 18.
```

Supported operators: `>`, `<`, `>=`, `=<`, `==`, `\=`.

## Aggregations

You can perform aggregations on numeric fields using the `aggregation/1` predicate.

```prolog
% Calculate sum of values
total_score(Sum) :-
    aggregation(sum),
    score(Sum).

% Count records
record_count(Count) :-
    aggregation(count),
    record(Count).
```

Supported aggregations: `sum`, `count`, `avg`, `min`, `max`.

### Example: Log Analysis Pipeline

Combine these features to build a powerful log analysis tool:

```prolog
% Extract response time from web logs and find slow requests (> 500ms)
slow_request(Url, Time) :-
    log(Line),
    match(Line, 'GET (.+) ([0-9]+)ms', auto, [Url, TimeStr]),
    Time is TimeStr,  % Convert to number
    Time > 500.
```

Compile and run:
```bash
go build slow_request.go
cat access.log | ./slow_request
```

---

## Navigation

**←** [Previous: Chapter 2: Basic Compilation](02_basic_compilation) | [📖 Book 6: Go Target](./) | [Next: Chapter 4: JSON Processing →](04_json_processing)
