# AWK Target Examples

This document shows practical examples of using the AWK target for real-world data processing tasks.

## Table of Contents

1. [Log File Analysis](#log-file-analysis)
2. [Data Aggregation](#data-aggregation)
3. [Filtering and Constraints](#filtering-and-constraints)
4. [Tail-Recursive Computations](#tail-recursive-computations)
5. [CSV/TSV Processing](#csvtsv-processing)

---

## Log File Analysis

### Example 1: Count Error Lines

**Prolog:**
```prolog
:- use_module('src/unifyweaver/core/recursive_compiler').

% Count all lines (using aggregation)
compile_log_counter :-
    recursive_compiler:compile_to_target(
        count_lines/0,
        awk,
        [aggregation(count)],
        AwkCode
    ),
    write(AwkCode).
```

**Generated AWK:**
```awk
BEGIN { FS = "\t" }
{ count++ }
END { print count }
```

**Usage:**
```bash
awk -f log_counter.awk /var/log/syslog
```

### Example 2: Sum Response Times

**Given log format:** `timestamp response_time status`

**Prolog:**
```prolog
% Sum response times
compile_response_time_sum :-
    recursive_compiler:compile_to_target(
        total_response_time/1,
        awk,
        [aggregation(sum)],
        AwkCode
    ),
    write(AwkCode).
```

**Usage:**
```bash
# Log format: timestamp, response_time, status
echo -e "2025-01-01\t120\t200\n2025-01-01\t85\t200\n2025-01-01\t340\t500" | \
awk 'BEGIN { FS = "\t" } { sum += $2 } END { print sum }'
# Output: 545
```

---

## Data Aggregation

### Example 3: Sales Analytics

**Given sales.tsv:** `product price quantity`

**Prolog:**
```prolog
% Calculate statistics on sales data
:- use_module('src/unifyweaver/targets/awk_target').

% Total revenue (assuming $2 = price, $3 = quantity)
% Note: Current implementation sums single field
% This example shows the concept
compile_revenue_sum :-
    awk_target:compile_predicate_to_awk(
        revenue/1,
        [aggregation(sum)],
        AwkCode
    ),
    write(AwkCode).
```

**Manual AWK for revenue:**
```awk
BEGIN { FS = "\t" }
{
    revenue = $2 * $3
    total += revenue
}
END { print total }
```

**Test data (sales.tsv):**
```
Widget	10.50	100
Gadget	25.00	50
Doohickey	5.75	200
```

**Output:** `2800` (1050 + 1250 + 1150)

### Example 4: Multiple Aggregations

**Prolog:**
```prolog
% Get min, max, avg price from products
compile_price_stats :-
    % Max price
    awk_target:compile_predicate_to_awk(max_price/1, [aggregation(max)], MaxCode),
    write('=== MAX ==='), nl, write(MaxCode), nl,

    % Min price
    awk_target:compile_predicate_to_awk(min_price/1, [aggregation(min)], MinCode),
    write('=== MIN ==='), nl, write(MinCode), nl,

    % Avg price
    awk_target:compile_predicate_to_awk(avg_price/1, [aggregation(avg)], AvgCode),
    write('=== AVG ==='), nl, write(AvgCode), nl.
```

---

## Filtering and Constraints

### Example 5: Filter High-Value Transactions

**Prolog:**
```prolog
% Facts: transactions with amount
transaction(tx001, 1500).
transaction(tx002, 250).
transaction(tx003, 5000).
transaction(tx004, 750).

% Rule: high-value transactions (> 1000)
high_value(TxId, Amount) :-
    transaction(TxId, Amount),
    Amount > 1000.

% Compile to AWK
compile_high_value :-
    awk_target:compile_predicate_to_awk(high_value/2, [], AwkCode),
    write(AwkCode).
```

**Generated AWK:**
```awk
#!/usr/bin/awk -f
BEGIN {
    facts["tx001:1500"] = 1
    facts["tx002:250"] = 1
    facts["tx003:5000"] = 1
    facts["tx004:750"] = 1
}
{
    key = $1":"$2
    if ((key in facts) && ($2 > 1000)) print $0
}
```

**Usage:**
```bash
# Input: transaction_id amount
echo -e "tx001\t1500\ntx002\t250\ntx003\t5000" | awk -f filter.awk
# Output:
# tx001  1500
# tx003  5000
```

### Example 6: Multiple Constraints

**Prolog:**
```prolog
% Users within age range
user(alice, 25, premium).
user(bob, 17, free).
user(charlie, 35, premium).
user(dave, 42, free).

% Adults with premium accounts
premium_adult(Name, Age) :-
    user(Name, Age, premium),
    Age >= 18,
    Age =< 65.

compile_premium_adults :-
    awk_target:compile_predicate_to_awk(premium_adult/2, [], AwkCode),
    write(AwkCode).
```

---

## Tail-Recursive Computations

### Example 7: Factorial Calculator

**Prolog:**
```prolog
% Tail-recursive factorial
factorial(0, Acc, Acc).
factorial(N, Acc, F) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, F).

compile_factorial :-
    awk_target:compile_predicate_to_awk(factorial/3, [], AwkCode),
    write(AwkCode).
```

**Generated AWK:**
```awk
BEGIN { FS = "\t" }
{
    n = $1
    acc = $2
    result = $3
    while (n > 0) {
        new_n = (n - 1)
        new_acc = (acc * n)
        n = new_n
        acc = new_acc
    }
    print acc
}
```

**Usage:**
```bash
# Calculate 5! (factorial of 5)
echo -e "5\t1\t0" | awk -f factorial.awk
# Output: 120

# Calculate 10!
echo -e "10\t1\t0" | awk -f factorial.awk
# Output: 3628800
```

### Example 8: Fibonacci Numbers

**Prolog:**
```prolog
% Tail-recursive fibonacci
% fib(N, Current, Next, Result)
fibonacci(0, Acc, _, Acc).
fibonacci(N, Current, Next, F) :-
    N > 0,
    N1 is N - 1,
    NewNext is Current + Next,
    fibonacci(N1, Next, NewNext, F).

compile_fibonacci :-
    awk_target:compile_predicate_to_awk(fibonacci/4, [], AwkCode),
    write(AwkCode).
```

**Usage:**
```bash
# Calculate 10th fibonacci number
# Input: n, fib(0)=0, fib(1)=1, result
echo -e "10\t0\t1\t0" | awk -f fibonacci.awk
# Output: 55
```

### Example 9: Sum from N to 0

**Prolog:**
```prolog
% Sum from N down to 0
sum_to_zero(0, Acc, Acc).
sum_to_zero(N, Acc, Sum) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum_to_zero(N1, Acc1, Sum).

compile_sum :-
    awk_target:compile_predicate_to_awk(sum_to_zero/3, [], AwkCode),
    write(AwkCode).
```

**Usage:**
```bash
# Sum from 100 to 1
echo -e "100\t0\t0" | awk -f sum.awk
# Output: 5050
```

---

## CSV/TSV Processing

### Example 10: Process Tab-Separated Data

**Given employees.tsv:** `name department salary`

**Prolog:**
```prolog
% Calculate average salary
compile_avg_salary :-
    awk_target:compile_predicate_to_awk(
        avg_salary/1,
        [aggregation(avg), field_separator('\t')],
        AwkCode
    ),
    write(AwkCode).
```

**Test data:**
```tsv
Alice	Engineering	95000
Bob	Sales	75000
Charlie	Engineering	105000
Dave	Marketing	68000
Eve	Sales	82000
```

**Usage:**
```bash
# Extract salary column (field 3) and calculate average
cut -f3 employees.tsv | awk -f avg_salary.awk
# Output: 85000
```

### Example 11: Multi-Field Analysis

**Prolog:**
```prolog
% Facts about employees
employee(alice, engineering, 95000).
employee(bob, sales, 75000).
employee(charlie, engineering, 105000).

% High-paid engineers (salary > 90k in engineering)
high_paid_engineer(Name, Salary) :-
    employee(Name, engineering, Salary),
    Salary > 90000.

compile_high_paid_engineers :-
    awk_target:compile_predicate_to_awk(high_paid_engineer/2, [], AwkCode),
    write(AwkCode).
```

---

## Performance Tips

### Tip 1: Use Aggregation for Large Files

For large files (millions of lines), aggregation operations are extremely fast:

```bash
# Count 10M lines - nearly instant
seq 1 10000000 | awk '{ count++ } END { print count }'
```

### Tip 2: Hash Lookups are O(1)

Fact-based filtering uses AWK associative arrays (hash tables):

```awk
BEGIN { facts["key1"] = 1; facts["key2"] = 1 }
{ if ($1 in facts) print }  # O(1) lookup per line
```

### Tip 3: Combine Multiple Operations

AWK can efficiently combine filtering and aggregation:

```awk
BEGIN { FS = "\t" }
$3 > 50000 {  # Filter: salary > 50k
    sum += $3
    count++
}
END {
    print "Average high salary:", sum/count
}
```

---

## Limitations

Current AWK target limitations:

1. **List processing**: AWK is line-oriented, not designed for recursive list operations
2. **Deep recursion**: Stack limitations prevent deep recursive calls
3. **Complex structures**: Nested data structures are not well-supported
4. **Mutual recursion**: Too complex for AWK's execution model

For these cases, use Python, Bash, or Prolog targets instead.

---

## See Also

- [AWK Target Status](AWK_TARGET_STATUS.md) - Implementation status and technical details
- [AWK Target Future Work](AWK_TARGET_FUTURE_WORK.md) - Planned enhancements and feasibility
