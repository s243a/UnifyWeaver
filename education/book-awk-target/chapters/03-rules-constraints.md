<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 3: Rules and Constraints

This chapter covers single and multiple rules, arithmetic expressions, and complex constraint handling in the AWK target.

## Single Rules

A single rule is a predicate with one clause. These compile directly to AWK's main processing block.

### Basic Rule Structure

```prolog
% Prolog rule
output(X, Y) :-
    input(A, B, C),
    constraint(A, B),
    X = A,
    Y is B + C.
```

Compiles to:

```awk
{
    A = $1
    B = $2
    C = $3

    if (constraint_check) {
        X = A
        Y = B + C
        print X "\t" Y
    }
}
```

### Example: Tax Calculation

```prolog
tax_report(Name, Salary, Tax) :-
    employee(Name, _, Salary),
    Salary > 30000,
    Tax is Salary * 0.22.
```

Generated AWK:

```awk
{
    Name = $1
    Salary = $3

    if (Salary > 30000) {
        Tax = Salary * 0.22
        print Name "\t" Salary "\t" Tax
    }
}
```

## Multiple Rules

Multiple rules for the same predicate create a union of results. Each matching rule contributes to the output.

### Union Semantics

```prolog
% Rule 1: High earners
bonus(Name, Bonus) :-
    employee(Name, _, Salary),
    Salary > 80000,
    Bonus is Salary * 0.15.

% Rule 2: Long tenure
bonus(Name, Bonus) :-
    employee(Name, _, Salary, Years),
    Years > 10,
    Bonus is Salary * 0.10.
```

Both rules are evaluated, and matching records from either rule appear in output.

### Example: Priority Classification

```prolog
% High priority
priority(Id, "high") :-
    ticket(Id, _, Severity, _),
    Severity = "critical".

% Medium priority
priority(Id, "medium") :-
    ticket(Id, _, Severity, Age),
    Severity = "major",
    Age > 7.

% Low priority (default)
priority(Id, "low") :-
    ticket(Id, _, Severity, _),
    Severity \= "critical",
    Severity \= "major".
```

## Arithmetic Expressions

The AWK target supports standard arithmetic operations.

### Basic Operations

| Prolog | AWK | Description |
|--------|-----|-------------|
| `X is A + B` | `X = A + B` | Addition |
| `X is A - B` | `X = A - B` | Subtraction |
| `X is A * B` | `X = A * B` | Multiplication |
| `X is A / B` | `X = A / B` | Division |
| `X is A mod B` | `X = A % B` | Modulo |
| `X is A ** B` | `X = A ^ B` | Exponentiation |

### Complex Expressions

```prolog
% Compound interest
future_value(Principal, Rate, Years, FV) :-
    investment(Principal, Rate, Years),
    FV is Principal * ((1 + Rate) ** Years).

% Weighted average
weighted_score(Name, Score) :-
    grades(Name, Midterm, Final, Homework),
    Score is Midterm * 0.3 + Final * 0.5 + Homework * 0.2.
```

### Parentheses for Precedence

```prolog
% Order of operations
result(X) :-
    data(A, B, C),
    X is (A + B) * C.  % Explicit grouping
```

## Comparison Constraints

### Numeric Comparisons

```prolog
in_range(X) :-
    value(X),
    X >= 10,
    X =< 100.

outside_range(X) :-
    value(X),
    (X < 10 ; X > 100).  % Note: disjunction may not compile directly
```

### String Comparisons

```prolog
% Alphabetical ordering
before_m(Name) :-
    person(Name),
    Name < "M".

% Exact match
is_admin(User) :-
    account(User, Role),
    Role = "admin".
```

## Logical Constraints

### Conjunction (AND)

All constraints in a rule body are implicitly ANDed:

```prolog
qualified(Name) :-
    employee(Name, Dept, Salary, Years),
    Dept = "Engineering",
    Salary > 70000,
    Years >= 3.
```

Generates:

```awk
if (Dept == "Engineering" && Salary > 70000 && Years >= 3) {
    print Name
}
```

### Negation

Use `\=` for not-equal or `\+` for negation-as-failure (limited support):

```prolog
% Not equal
not_sales(Name, Dept) :-
    employee(Name, Dept, _),
    Dept \= "Sales".

% Negation (when supported)
no_manager(Name) :-
    employee(Name, _, _),
    \+ has_manager(Name).
```

## Variable Binding

### Direct Binding

```prolog
formatted(Label, Value) :-
    data(X, Y),
    Label = "Result",
    Value is X + Y.
```

### Chained Binding

```prolog
processed(Output) :-
    raw(Input),
    Temp is Input * 2,
    Output is Temp + 10.
```

## Constraint Order

Constraints are evaluated in order. Place cheap constraints first:

```prolog
% Good: Filter by department first (string compare is fast)
efficient(Name, Salary) :-
    employee(Name, Dept, Salary),
    Dept = "Engineering",      % Fast check first
    Salary > complex_calc().   % Expensive check last

% Less efficient: Expensive check first
inefficient(Name, Salary) :-
    employee(Name, Dept, Salary),
    Salary > complex_calc(),   % Expensive check first
    Dept = "Engineering".      % Fast check last
```

## Type Coercion

AWK handles type coercion automatically, but be aware:

```prolog
% String to number
parse_amount(Id, Amount) :-
    record(Id, AmountStr),
    Amount is AmountStr + 0.  % Force numeric

% Number to string
format_id(Prefix, Id, Formatted) :-
    data(Id),
    Formatted = Prefix "" Id.  % Concatenation forces string
```

## Complete Example

```prolog
% file: rules_constraints.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

% Tax brackets (multiple rules)
tax(Name, Salary, Tax, Bracket) :-
    employee(Name, _, Salary),
    Salary > 100000,
    Tax is Salary * 0.35,
    Bracket = "high".

tax(Name, Salary, Tax, Bracket) :-
    employee(Name, _, Salary),
    Salary > 50000,
    Salary =< 100000,
    Tax is Salary * 0.25,
    Bracket = "medium".

tax(Name, Salary, Tax, Bracket) :-
    employee(Name, _, Salary),
    Salary =< 50000,
    Tax is Salary * 0.15,
    Bracket = "low".

% Complex calculation
performance_bonus(Name, Bonus) :-
    employee(Name, _, Salary, Rating, Years),
    Rating >= 4,
    Years >= 2,
    BaseBonus is Salary * 0.1,
    TenureMultiplier is 1 + (Years * 0.02),
    Bonus is BaseBonus * TenureMultiplier.

% Generate scripts
generate :-
    compile_predicate_to_awk(tax/4, [], AWK1),
    write_awk_script('tax.awk', AWK1),

    compile_predicate_to_awk(performance_bonus/2, [], AWK2),
    write_awk_script('bonus.awk', AWK2),

    format('Generated tax.awk and bonus.awk~n').

:- initialization(generate, main).
```

Test data (`employees.tsv`):
```
Alice	Engineering	120000	5	8
Bob	Sales	65000	4	3
Carol	Marketing	45000	3	5
Dave	Engineering	85000	4	6
Eve	HR	52000	5	2
```

Run:
```bash
swipl rules_constraints.pl
awk -f tax.awk employees.tsv
awk -f bonus.awk employees.tsv
```

## Constraint Patterns

| Pattern | Prolog | Use Case |
|---------|--------|----------|
| Equality | `X = Y` | Exact match |
| Inequality | `X \= Y` | Exclusion |
| Range | `X >= L, X =< H` | Bounded values |
| Threshold | `X > T` | Minimum requirement |
| Computed | `R is Expr` | Calculated values |
| Chained | `T is A+B, R is T*2` | Multi-step calculation |

## Exercises

1. **Tax Brackets**: Create a predicate with three rules for different tax brackets (0-30000: 10%, 30001-70000: 20%, 70001+: 30%).

2. **Grade Classification**: Write rules that classify scores: A (90+), B (80-89), C (70-79), D (60-69), F (<60).

3. **Commission Calculator**: Create a predicate that calculates sales commission: 5% for sales under 10000, 8% for 10000-50000, 12% for over 50000.

4. **Complex Eligibility**: Write a predicate that checks eligibility based on multiple criteria: age >= 18, income >= 30000, credit_score >= 650.

5. **Derived Fields**: Create a predicate that computes BMI from height and weight, then classifies as underweight/normal/overweight/obese.

## Summary

In this chapter, you learned:

- Single rule compilation to AWK main block
- Multiple rules creating union semantics
- Arithmetic expressions and operators
- Numeric and string comparisons
- Conjunction (AND) of constraints
- Variable binding and chaining
- Constraint ordering for efficiency
- Type coercion in AWK

## Next Chapter

In Chapter 4, we'll explore aggregation operations: sum, count, max, min, and avg.
