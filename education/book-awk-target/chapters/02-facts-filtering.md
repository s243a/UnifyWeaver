<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 2: Facts and Filtering

This chapter covers how Prolog facts compile to AWK associative arrays and how to filter data with various constraints.

## How Facts Compile to AWK

In Prolog, facts represent static data. The AWK target compiles these to associative arrays, enabling efficient lookups during stream processing.

### Simple Facts

```prolog
% Prolog facts
department(1, "Engineering").
department(2, "Sales").
department(3, "Marketing").
```

Compiles to AWK:

```awk
BEGIN {
    department[1] = "Engineering"
    department[2] = "Sales"
    department[3] = "Marketing"
}
```

### Multi-Argument Facts

Facts with multiple arguments use composite keys:

```prolog
% Employee facts: id, name, dept_id, salary
employee(101, "Alice", 1, 75000).
employee(102, "Bob", 2, 65000).
employee(103, "Carol", 1, 80000).
```

Compiles to:

```awk
BEGIN {
    employee[101, "name"] = "Alice"
    employee[101, "dept_id"] = 1
    employee[101, "salary"] = 75000

    employee[102, "name"] = "Bob"
    employee[102, "dept_id"] = 2
    employee[102, "salary"] = 65000

    employee[103, "name"] = "Carol"
    employee[103, "dept_id"] = 1
    employee[103, "salary"] = 80000
}
```

## Combining Facts with Input Data

The power of AWK comes from combining static facts with streaming input.

### Example: Department Lookup

```prolog
% Static facts
department(1, "Engineering").
department(2, "Sales").
department(3, "Marketing").

% Rule using facts and input
employee_with_dept(Name, DeptName) :-
    employee_record(Id, Name, DeptId, _),  % From input stream
    department(DeptId, DeptName).           % From facts
```

Input (`employees.tsv`):
```
101	Alice	1	75000
102	Bob	2	65000
103	Carol	1	80000
```

Output:
```
Alice	Engineering
Bob	Sales
Carol	Engineering
```

## Filtering with Constraints

### Equality Constraints

```prolog
% Filter by exact match
engineering_only(Name) :-
    employee(Name, Dept, _),
    Dept = "Engineering".
```

Generated AWK:

```awk
{
    Name = $1
    Dept = $2

    if (Dept == "Engineering") {
        print Name
    }
}
```

### Numeric Comparisons

```prolog
% Salary above threshold
high_salary(Name, Salary) :-
    employee(Name, _, Salary),
    Salary > 70000.

% Salary in range
mid_range(Name, Salary) :-
    employee(Name, _, Salary),
    Salary >= 50000,
    Salary =< 80000.
```

### String Comparisons

```prolog
% Names starting before 'M' alphabetically
early_names(Name) :-
    employee(Name, _, _),
    Name < "M".
```

### Inequality

```prolog
% Not in sales
not_sales(Name, Dept) :-
    employee(Name, Dept, _),
    Dept \= "Sales".
```

## Multiple Conditions

### Conjunction (AND)

Multiple conditions in a rule body are combined with AND:

```prolog
% Engineering AND high salary
senior_engineer(Name, Salary) :-
    employee(Name, Dept, Salary),
    Dept = "Engineering",
    Salary > 75000.
```

### Using Facts for Membership

```prolog
% Define valid departments
valid_dept("Engineering").
valid_dept("Sales").
valid_dept("Marketing").

% Filter to valid departments only
validated(Name, Dept) :-
    employee(Name, Dept, _),
    valid_dept(Dept).
```

## Field Selection

### Selecting Specific Fields

The arguments in the head determine output columns:

```prolog
% Select only name and salary (skip department)
name_salary(Name, Salary) :-
    employee(Name, _, Salary).
```

### Reordering Fields

```prolog
% Salary first, then name
salary_name(Salary, Name) :-
    employee(Name, _, Salary).
```

### Computed Fields

```prolog
% Add bonus calculation
with_bonus(Name, Bonus) :-
    employee(Name, _, Salary),
    Bonus is Salary * 0.1.

% Total compensation
total_comp(Name, Total) :-
    employee(Name, _, Salary),
    Total is Salary * 1.15.  % Salary + 15% bonus
```

## Working with Multiple Input Columns

### Mapping Input Fields

When processing input, fields map to predicate arguments:

```prolog
% Input: Name<TAB>Dept<TAB>Salary<TAB>HireYear
% Arguments map to $1, $2, $3, $4

tenure_report(Name, Years) :-
    employee_record(Name, _, _, HireYear),
    Years is 2025 - HireYear.
```

### Ignoring Fields

Use `_` to ignore fields you don't need:

```prolog
% Only care about name and salary
simple_report(Name, Salary) :-
    employee(Name, _, Salary, _, _, _).
```

## Unique Results

Remove duplicates from output:

```prolog
?- compile_predicate_to_awk(departments/1, [unique(true)], AWK).
```

This generates AWK that tracks seen values:

```awk
{
    Dept = $2
    if (!(Dept in seen)) {
        seen[Dept] = 1
        print Dept
    }
}
```

## Complete Example

```prolog
% file: facts_filtering.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

% Static department facts
department(1, "Engineering", "Building A").
department(2, "Sales", "Building B").
department(3, "Marketing", "Building B").
department(4, "HR", "Building A").

% Filter: Engineering employees earning > 70k
qualified(Name, Salary, Location) :-
    employee_record(_, Name, DeptId, Salary),
    Salary > 70000,
    department(DeptId, "Engineering", Location).

% Generate script
generate :-
    compile_predicate_to_awk(qualified/3, [
        record_format(tsv),
        include_header(true)
    ], AWK),
    write_awk_script('qualified.awk', AWK),
    format('Generated qualified.awk~n').

:- initialization(generate, main).
```

Test data (`employees.tsv`):
```
Id	Name	DeptId	Salary
101	Alice	1	75000
102	Bob	2	85000
103	Carol	1	80000
104	Dave	3	70000
105	Eve	1	65000
```

Run:
```bash
swipl facts_filtering.pl
awk -f qualified.awk employees.tsv
```

Output:
```
Alice	75000	Building A
Carol	80000	Building A
```

## Filtering Patterns Summary

| Pattern | Example | AWK Condition |
|---------|---------|---------------|
| Equality | `Dept = "Sales"` | `Dept == "Sales"` |
| Greater than | `Salary > 50000` | `Salary > 50000` |
| Less than | `Age < 30` | `Age < 30` |
| Greater or equal | `Score >= 80` | `Score >= 80` |
| Less or equal | `Price =< 100` | `Price <= 100` |
| Not equal | `Status \= "inactive"` | `Status != "inactive"` |
| Range | `X >= 10, X =< 20` | `X >= 10 && X <= 20` |

## Exercises

1. **Basic Filter**: Create a predicate that filters employees with salary between 60000 and 80000.

2. **Department Lookup**: Define department facts and create a predicate that joins employee input with department names.

3. **Multiple Conditions**: Write a predicate that finds employees who are NOT in "HR" AND earn more than 65000.

4. **Computed Field**: Create a predicate that shows employee name and their tax (salary * 0.25).

5. **Unique Departments**: Generate a script that outputs unique department names from employee data.

## Summary

In this chapter, you learned:

- How Prolog facts compile to AWK associative arrays
- Combining static facts with streaming input
- Filtering with equality, numeric, and string constraints
- Using multiple conditions (conjunction)
- Field selection and reordering
- Computing derived fields
- Removing duplicates with `unique(true)`

## Next Chapter

In Chapter 3, we'll explore rules and constraints in more depth, including arithmetic expressions and complex condition handling.
