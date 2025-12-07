<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 4: Aggregations

This chapter covers the built-in aggregation operations in the AWK target: sum, count, max, min, and average.

## Overview

Aggregations process all input records and produce summary statistics. Unlike filtering (which outputs per-record), aggregations output in AWK's END block after all input is processed.

```
Input Records → Accumulation → END Block → Output
```

## Aggregation Option

Enable aggregations with the `aggregation` option:

```prolog
compile_predicate_to_awk(pred/N, [aggregation(Operation)], AWK)
```

Available operations:
- `sum` - Sum of values
- `count` - Number of records
- `max` - Maximum value
- `min` - Minimum value
- `avg` - Average (mean) value

## Sum Aggregation

Calculate the total of numeric values.

### Basic Sum

```prolog
total_salary(Total) :-
    employee(_, _, Salary),
    Total = Salary.
```

Compile with aggregation:

```prolog
?- compile_predicate_to_awk(total_salary/1, [aggregation(sum)], AWK).
```

Generated AWK:

```awk
BEGIN {
    sum_Total = 0
}

{
    Salary = $3
    sum_Total += Salary
}

END {
    print sum_Total
}
```

### Filtered Sum

```prolog
engineering_budget(Total) :-
    employee(_, Dept, Salary),
    Dept = "Engineering",
    Total = Salary.
```

Only sums salaries where department is Engineering.

## Count Aggregation

Count the number of matching records.

### Basic Count

```prolog
employee_count(N) :-
    employee(_, _, _),
    N = 1.  % Each record counts as 1
```

Compile:

```prolog
?- compile_predicate_to_awk(employee_count/1, [aggregation(count)], AWK).
```

Generated AWK:

```awk
BEGIN {
    count_N = 0
}

{
    count_N++
}

END {
    print count_N
}
```

### Filtered Count

```prolog
high_earner_count(N) :-
    employee(_, _, Salary),
    Salary > 70000,
    N = 1.
```

Counts only employees with salary above 70000.

## Max Aggregation

Find the maximum value.

### Basic Max

```prolog
highest_salary(Max) :-
    employee(_, _, Salary),
    Max = Salary.
```

Compile:

```prolog
?- compile_predicate_to_awk(highest_salary/1, [aggregation(max)], AWK).
```

Generated AWK:

```awk
BEGIN {
    max_Max = ""
    max_initialized = 0
}

{
    Salary = $3
    if (!max_initialized || Salary > max_Max) {
        max_Max = Salary
        max_initialized = 1
    }
}

END {
    print max_Max
}
```

### Filtered Max

```prolog
max_sales_salary(Max) :-
    employee(_, Dept, Salary),
    Dept = "Sales",
    Max = Salary.
```

Finds maximum salary in Sales department only.

## Min Aggregation

Find the minimum value.

### Basic Min

```prolog
lowest_salary(Min) :-
    employee(_, _, Salary),
    Min = Salary.
```

Compile:

```prolog
?- compile_predicate_to_awk(lowest_salary/1, [aggregation(min)], AWK).
```

Generated AWK:

```awk
BEGIN {
    min_Min = ""
    min_initialized = 0
}

{
    Salary = $3
    if (!min_initialized || Salary < min_Min) {
        min_Min = Salary
        min_initialized = 1
    }
}

END {
    print min_Min
}
```

## Average Aggregation

Calculate the arithmetic mean.

### Basic Average

```prolog
avg_salary(Avg) :-
    employee(_, _, Salary),
    Avg = Salary.
```

Compile:

```prolog
?- compile_predicate_to_awk(avg_salary/1, [aggregation(avg)], AWK).
```

Generated AWK:

```awk
BEGIN {
    sum_Avg = 0
    count_Avg = 0
}

{
    Salary = $3
    sum_Avg += Salary
    count_Avg++
}

END {
    if (count_Avg > 0) {
        print sum_Avg / count_Avg
    }
}
```

## Grouped Aggregations

Aggregate by groups (like SQL GROUP BY).

### Sum by Group

```prolog
dept_total(Dept, Total) :-
    employee(_, Dept, Salary),
    Total = Salary.
```

This creates per-department totals. The first argument becomes the group key.

Generated AWK:

```awk
BEGIN {
    # Nothing special needed
}

{
    Dept = $2
    Salary = $3
    sum_by_dept[Dept] += Salary
}

END {
    for (Dept in sum_by_dept) {
        print Dept "\t" sum_by_dept[Dept]
    }
}
```

### Count by Group

```prolog
dept_count(Dept, Count) :-
    employee(_, Dept, _),
    Count = 1.
```

### Multiple Aggregations

Different predicates can compute different aggregations on the same data:

```prolog
% Total salary
total(Sum) :-
    employee(_, _, Salary),
    Sum = Salary.

% Number of employees
count(N) :-
    employee(_, _, _),
    N = 1.

% Average salary
average(Avg) :-
    employee(_, _, Salary),
    Avg = Salary.
```

Compile each separately:

```prolog
compile_predicate_to_awk(total/1, [aggregation(sum)], AWK1),
compile_predicate_to_awk(count/1, [aggregation(count)], AWK2),
compile_predicate_to_awk(average/1, [aggregation(avg)], AWK3).
```

## Aggregation with Headers

Handle header rows properly:

```prolog
?- compile_predicate_to_awk(total_salary/1, [
    aggregation(sum),
    include_header(true)
], AWK).
```

Generated AWK:

```awk
BEGIN {
    sum_Total = 0
}

NR == 1 { next }  # Skip header

{
    Salary = $3
    sum_Total += Salary
}

END {
    print sum_Total
}
```

## Complete Example

```prolog
% file: aggregations.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

% Basic aggregations
total_payroll(Total) :-
    employee(_, _, Salary),
    Total = Salary.

employee_count(N) :-
    employee(_, _, _),
    N = 1.

max_salary(Max) :-
    employee(_, _, Salary),
    Max = Salary.

min_salary(Min) :-
    employee(_, _, Salary),
    Min = Salary.

avg_salary(Avg) :-
    employee(_, _, Salary),
    Avg = Salary.

% Grouped aggregation
dept_payroll(Dept, Total) :-
    employee(_, Dept, Salary),
    Total = Salary.

% Filtered aggregation
engineering_total(Total) :-
    employee(_, Dept, Salary),
    Dept = "Engineering",
    Total = Salary.

% Generate all scripts
generate :-
    compile_predicate_to_awk(total_payroll/1, [aggregation(sum)], AWK1),
    write_awk_script('total_payroll.awk', AWK1),

    compile_predicate_to_awk(employee_count/1, [aggregation(count)], AWK2),
    write_awk_script('employee_count.awk', AWK2),

    compile_predicate_to_awk(max_salary/1, [aggregation(max)], AWK3),
    write_awk_script('max_salary.awk', AWK3),

    compile_predicate_to_awk(min_salary/1, [aggregation(min)], AWK4),
    write_awk_script('min_salary.awk', AWK4),

    compile_predicate_to_awk(avg_salary/1, [aggregation(avg)], AWK5),
    write_awk_script('avg_salary.awk', AWK5),

    compile_predicate_to_awk(dept_payroll/2, [aggregation(sum)], AWK6),
    write_awk_script('dept_payroll.awk', AWK6),

    compile_predicate_to_awk(engineering_total/1, [
        aggregation(sum),
        include_header(true)
    ], AWK7),
    write_awk_script('engineering_total.awk', AWK7),

    format('Generated aggregation scripts~n').

:- initialization(generate, main).
```

Test data (`employees.tsv`):
```
Name	Dept	Salary
Alice	Engineering	75000
Bob	Sales	65000
Carol	Engineering	80000
Dave	Marketing	70000
Eve	Sales	55000
```

Run:
```bash
swipl aggregations.pl

echo "Total Payroll:"
awk -f total_payroll.awk employees.tsv

echo "Employee Count:"
awk -f employee_count.awk employees.tsv

echo "Max Salary:"
awk -f max_salary.awk employees.tsv

echo "Avg Salary:"
awk -f avg_salary.awk employees.tsv

echo "Payroll by Department:"
awk -f dept_payroll.awk employees.tsv

echo "Engineering Total (with header):"
awk -f engineering_total.awk employees.tsv
```

Expected output:
```
Total Payroll:
345000

Employee Count:
5

Max Salary:
80000

Avg Salary:
69000

Payroll by Department:
Engineering	155000
Sales	120000
Marketing	70000

Engineering Total (with header):
155000
```

## Aggregation Summary

| Operation | Prolog Pattern | AWK Implementation |
|-----------|---------------|-------------------|
| Sum | `aggregation(sum)` | `sum_var += value` |
| Count | `aggregation(count)` | `count_var++` |
| Max | `aggregation(max)` | `if (v > max) max = v` |
| Min | `aggregation(min)` | `if (v < min) min = v` |
| Avg | `aggregation(avg)` | `sum/count` in END |

## Exercises

1. **Basic Sum**: Create a predicate that sums all order amounts from an orders file.

2. **Filtered Count**: Count how many products have a price above $50.

3. **Group Statistics**: Calculate the average salary per department.

4. **Min/Max**: Find the minimum and maximum prices in a product catalog.

5. **Combined Analysis**: Create scripts to show: total revenue, number of orders, average order value, highest order, and lowest order.

## Summary

In this chapter, you learned:

- The `aggregation` option for compile_predicate_to_awk
- Sum aggregation for totals
- Count aggregation for record counts
- Max and min for extreme values
- Avg for arithmetic mean
- Grouped aggregations for per-category statistics
- Combining aggregations with filters
- Handling headers with aggregations

## Next Chapter

In Chapter 5, we'll explore tail recursion and how it compiles to efficient AWK while loops.
