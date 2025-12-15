---
file_type: UnifyWeaver Example Library
---
# AWK Advanced Patterns Examples

This file contains executable records for the AWK advanced patterns playbook.

## `unifyweaver.execution.awk_aggregation`

> [!example-record]
> id: unifyweaver.execution.awk_aggregation
> name: AWK Aggregation Operations
> platform: bash

This record demonstrates sum, count, max, min, avg compiled to AWK.

```bash
#!/bin/bash
# AWK Advanced Demo - Aggregation Operations
# Demonstrates sum, count, max, min, avg compiled to AWK

set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Advanced Demo: Aggregation Patterns ==="

# Create test directory and data
mkdir -p tmp/awk_demo

# Create test sales data
cat > tmp/awk_demo/sales.tsv << 'TSV'
region	product	quantity	price
North	Widget	100	25.50
South	Gadget	50	45.00
North	Gadget	75	45.00
East	Widget	120	25.50
West	Gizmo	30	89.99
South	Widget	90	25.50
TSV

echo "Created test data: tmp/awk_demo/sales.tsv"

# Create Prolog script for aggregation patterns
cat > tmp/awk_demo/demo_aggregation.pl << 'PROLOG'
:- use_module('src/unifyweaver/targets/awk_target').

% Define test facts that AWK will load
:- dynamic user:sales/4.

% Load sales data as facts
user:sales('North', 'Widget', 100, 25.50).
user:sales('South', 'Gadget', 50, 45.00).
user:sales('North', 'Gadget', 75, 45.00).
user:sales('East', 'Widget', 120, 25.50).
user:sales('West', 'Gizmo', 30, 89.99).
user:sales('South', 'Widget', 90, 25.50).

main :-
    format("~n=== Generating AWK Aggregation Scripts ===~n~n"),

    % Example 1: SUM aggregation
    format("1. Generating SUM aggregation...~n"),
    compile_predicate_to_awk(sales/4,
        [aggregation(sum), field_separator('\t')],
        SumCode),
    write_awk_script(SumCode, 'tmp/awk_demo/sum_sales.awk'),

    % Example 2: COUNT aggregation
    format("~n2. Generating COUNT aggregation...~n"),
    compile_predicate_to_awk(sales/4,
        [aggregation(count), field_separator('\t')],
        CountCode),
    write_awk_script(CountCode, 'tmp/awk_demo/count_sales.awk'),

    % Example 3: MAX aggregation
    format("~n3. Generating MAX aggregation...~n"),
    compile_predicate_to_awk(sales/4,
        [aggregation(max), field_separator('\t')],
        MaxCode),
    write_awk_script(MaxCode, 'tmp/awk_demo/max_sales.awk'),

    % Example 4: AVG aggregation
    format("~n4. Generating AVG aggregation...~n"),
    compile_predicate_to_awk(sales/4,
        [aggregation(avg), field_separator('\t')],
        AvgCode),
    write_awk_script(AvgCode, 'tmp/awk_demo/avg_sales.awk'),

    format("~n=== All aggregation scripts generated ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate AWK aggregation scripts..."
swipl tmp/awk_demo/demo_aggregation.pl

echo ""
echo "=== Generated AWK Scripts ==="

echo ""
echo "--- sum_sales.awk ---"
cat tmp/awk_demo/sum_sales.awk

echo ""
echo "=== Testing Aggregation Scripts ==="

# Prepare numeric test data (just quantities)
tail -n +2 tmp/awk_demo/sales.tsv | cut -f3 > tmp/awk_demo/quantities.txt

echo ""
echo "Input quantities:"
cat tmp/awk_demo/quantities.txt

echo ""
echo "SUM of quantities:"
awk -f tmp/awk_demo/sum_sales.awk tmp/awk_demo/quantities.txt

echo ""
echo "COUNT of records:"
awk -f tmp/awk_demo/count_sales.awk tmp/awk_demo/quantities.txt

echo ""
echo "MAX quantity:"
awk -f tmp/awk_demo/max_sales.awk tmp/awk_demo/quantities.txt

echo ""
echo "AVG quantity:"
awk -f tmp/awk_demo/avg_sales.awk tmp/awk_demo/quantities.txt

echo ""
echo "Success: AWK aggregation demo complete"
```

## `unifyweaver.execution.awk_tail_recursion`

> [!example-record]
> id: unifyweaver.execution.awk_tail_recursion
> name: AWK Tail Recursion to While Loop
> platform: bash

This record demonstrates compiling tail-recursive predicates to AWK while loops.

```bash
#!/bin/bash
# AWK Advanced Demo - Tail Recursion Compilation
# Demonstrates compiling tail-recursive predicates to AWK while loops

set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Advanced Demo: Tail Recursion to While Loop ==="

mkdir -p tmp/awk_demo

# Create Prolog script with tail-recursive predicates
cat > tmp/awk_demo/demo_tail_rec.pl << 'PROLOG'
:- use_module('src/unifyweaver/targets/awk_target').

% Factorial with accumulator (tail-recursive)
% factorial(N, Acc, Result) - computes N! using accumulator
:- dynamic user:factorial/3.

user:factorial(0, Acc, Acc).
user:factorial(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    user:factorial(N1, Acc1, Result).

% Sum from 1 to N (tail-recursive)
% sum_to(N, Acc, Result)
:- dynamic user:sum_to/3.

user:sum_to(0, Acc, Acc).
user:sum_to(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    user:sum_to(N1, Acc1, Result).

main :-
    format("~n=== Compiling Tail-Recursive Predicates to AWK ===~n~n"),

    % Compile factorial
    format("1. Compiling factorial/3 (N! with accumulator)...~n"),
    compile_predicate_to_awk(factorial/3,
        [field_separator('\t')],
        FactorialCode),
    write_awk_script(FactorialCode, 'tmp/awk_demo/factorial.awk'),
    format("   Generated: tmp/awk_demo/factorial.awk~n"),

    % Compile sum_to
    format("~n2. Compiling sum_to/3 (sum from 1 to N)...~n"),
    compile_predicate_to_awk(sum_to/3,
        [field_separator('\t')],
        SumCode),
    write_awk_script(SumCode, 'tmp/awk_demo/sum_to.awk'),
    format("   Generated: tmp/awk_demo/sum_to.awk~n"),

    format("~n=== Tail recursion compilation complete ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to compile tail-recursive predicates..."
swipl tmp/awk_demo/demo_tail_rec.pl

echo ""
echo "=== Generated AWK While Loop Scripts ==="

echo ""
echo "--- factorial.awk ---"
cat tmp/awk_demo/factorial.awk

echo ""
echo "--- sum_to.awk ---"
cat tmp/awk_demo/sum_to.awk

echo ""
echo "=== Testing Compiled Scripts ==="

# Test factorial: input is "N  Acc" (tab-separated)
echo ""
echo "Testing factorial(5, 1, Result) - expect 120:"
echo -e "5\t1" | awk -f tmp/awk_demo/factorial.awk

echo ""
echo "Testing factorial(10, 1, Result) - expect 3628800:"
echo -e "10\t1" | awk -f tmp/awk_demo/factorial.awk

echo ""
echo "Testing sum_to(10, 0, Result) - expect 55:"
echo -e "10\t0" | awk -f tmp/awk_demo/sum_to.awk

echo ""
echo "Testing sum_to(100, 0, Result) - expect 5050:"
echo -e "100\t0" | awk -f tmp/awk_demo/sum_to.awk

echo ""
echo "Success: Tail recursion compilation demo complete"
```

## `unifyweaver.execution.awk_constraints`

> [!example-record]
> id: unifyweaver.execution.awk_constraints
> name: AWK Fact Lookup with Constraints
> platform: bash

This record demonstrates compiling predicates with constraints to AWK.

```bash
#!/bin/bash
# AWK Advanced Demo - Fact Lookup with Constraints
# Demonstrates compiling predicates with constraints to AWK

set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Advanced Demo: Fact Lookup with Constraints ==="

mkdir -p tmp/awk_demo

# Create Prolog script with constrained predicates
cat > tmp/awk_demo/demo_constraints.pl << 'PROLOG'
:- use_module('src/unifyweaver/targets/awk_target').

% Employee facts
:- dynamic user:employee/4.

user:employee(alice, engineering, 85000, senior).
user:employee(bob, sales, 65000, junior).
user:employee(charlie, engineering, 95000, senior).
user:employee(diana, marketing, 72000, mid).
user:employee(eve, engineering, 78000, mid).

% High earners: employees earning > 75000
:- dynamic user:high_earner/4.

user:high_earner(Name, Dept, Salary, Level) :-
    user:employee(Name, Dept, Salary, Level),
    Salary > 75000.

main :-
    format("~n=== Compiling Constrained Predicates to AWK ===~n~n"),

    % Compile employee facts
    format("1. Compiling employee/4 facts...~n"),
    compile_predicate_to_awk(employee/4,
        [field_separator('\t'), unique(true)],
        EmpCode),
    write_awk_script(EmpCode, 'tmp/awk_demo/employee.awk'),

    % Compile high_earner with constraint
    format("~n2. Compiling high_earner/4 with salary constraint...~n"),
    compile_predicate_to_awk(high_earner/4,
        [field_separator('\t'), unique(true)],
        HighCode),
    write_awk_script(HighCode, 'tmp/awk_demo/high_earner.awk'),

    format("~n=== Constraint compilation complete ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to compile constrained predicates..."
swipl tmp/awk_demo/demo_constraints.pl

echo ""
echo "=== Generated AWK Scripts ==="

echo ""
echo "--- employee.awk (fact lookup) ---"
cat tmp/awk_demo/employee.awk

echo ""
echo "--- high_earner.awk (with salary > 75000 constraint) ---"
cat tmp/awk_demo/high_earner.awk

echo ""
echo "=== Testing Constrained Scripts ==="

# Create test input (all employees)
cat > tmp/awk_demo/all_employees.tsv << 'TSV'
alice	engineering	85000	senior
bob	sales	65000	junior
charlie	engineering	95000	senior
diana	marketing	72000	mid
eve	engineering	78000	mid
frank	sales	55000	junior
TSV

echo ""
echo "Input data (all_employees.tsv):"
cat tmp/awk_demo/all_employees.tsv

echo ""
echo "Employees matching facts (known employees):"
awk -f tmp/awk_demo/employee.awk tmp/awk_demo/all_employees.tsv

echo ""
echo "High earners (salary > 75000):"
awk -f tmp/awk_demo/high_earner.awk tmp/awk_demo/all_employees.tsv

echo ""
echo "Success: Constraint compilation demo complete"
```

## `unifyweaver.execution.awk_module_info`

> [!example-record]
> id: unifyweaver.execution.awk_module_info
> name: AWK Target Module Info
> platform: bash

This record displays AWK target capabilities and compilation options.

```bash
#!/bin/bash
# AWK Target Module Information
# Shows AWK target capabilities

set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Target Module Information ==="

echo ""
echo "=== Public API ==="
echo ""
echo "compile_predicate_to_awk(+Predicate, +Options, -AwkCode)"
echo "  Compile a Prolog predicate to AWK code"
echo ""
echo "write_awk_script(+AwkCode, +FilePath)"
echo "  Write AWK script to file and make executable"

echo ""
echo "=== Compilation Options ==="
echo ""
echo "record_format(jsonl|tsv|csv) - Input format (default: tsv)"
echo "field_separator(Char)       - Field separator (default: tab)"
echo "include_header(true|false)  - Include shebang (default: true)"
echo "unique(true|false)          - Deduplicate results (default: true)"
echo "unordered(true|false)       - Allow unordered output (default: true)"
echo "aggregation(sum|count|max|min|avg) - Aggregation operation"

echo ""
echo "=== Supported Compilation Patterns ==="
echo ""
echo "1. Facts: Prolog facts -> AWK associative array lookups"
echo "2. Single Rules: Predicates with body -> AWK conditions"
echo "3. Multiple Rules (OR): UNION of alternatives"
echo "4. Aggregation: GROUP BY operations -> AWK BEGIN/END blocks"
echo "5. Tail Recursion: Recursive predicates -> AWK while loops"
echo "6. Constraints: Comparison operators -> AWK conditionals"
echo "7. Regex Matching: match/4 -> AWK regex patterns"

echo ""
echo "=== AWK Regex Support ==="
echo ""
echo "Supported regex types for AWK target:"
echo "  - auto: Auto-detect (default)"
echo "  - ere:  Extended Regular Expressions"
echo "  - bre:  Basic Regular Expressions"
echo "  - awk:  AWK-native regex"
echo ""
echo "NOT supported (will error):"
echo "  - pcre: Perl-Compatible Regex"
echo "  - python: Python regex"
echo "  - dotnet: .NET regex"

echo ""
echo "=== Example: Tail Recursion Pattern ==="
echo ""
echo "Prolog (factorial):"
echo "  factorial(0, Acc, Acc)."
echo "  factorial(N, Acc, R) :- N > 0, N1 is N-1, Acc1 is Acc*N, factorial(N1, Acc1, R)."
echo ""
echo "Compiled AWK (while loop):"
echo "  BEGIN { FS = \"\\t\" }"
echo "  {"
echo "      n = \$1; acc = \$2"
echo "      while (n > 0) {"
echo "          acc1 = (acc * n)"
echo "          n1 = (n - 1)"
echo "          acc = acc1"
echo "          n = n1"
echo "      }"
echo "      print acc"
echo "  }"

echo ""
echo "Success: AWK module info displayed"
```

