<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 4: Advanced Features

## JSON Processing

The Rust target supports high-performance JSON processing using `serde` and `serde_json`.

### JSON Input

```prolog
:- json_schema(user, [field(name, string), field(age, integer)]).

user_info(Name, Age) :-
    json_record([name-Name, age-Age]).

compile_json :-
    compile_predicate_to_rust(user_info/2, [
        json_input(true),
        json_schema(user)
    ], Code),
    write_rust_project(Code, 'output/json_processor').
```

### JSON Output

```prolog
output_user(Name, Age) :- input_data(Name, Age).

compile_output :-
    compile_predicate_to_rust(output_user/2, [json_output(true)], Code),
    write_rust_project(Code, 'output/json_writer').
```

## Aggregations

Rust target supports efficient aggregations.

```prolog
total_sales(Sum) :-
    aggregation(sum),
    sale(Sum).
```

Supported: `sum`, `count`, `min`, `max`, `avg`.

## Constraints

Numeric constraints are compiled to native Rust comparisons.

```prolog
high_value(Val) :-
    input(Val),
    Val > 1000,
    Val =< 5000.
```

---

## Navigation

**←** [Previous: Chapter 3: Project Generation](03_project_generation) | [📖 Book 9: Rust Target](README)
