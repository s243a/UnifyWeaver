<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 7: Practical Applications

This chapter brings together all AWK target concepts with complete, real-world applications.

## Application 1: Log File Analyzer

A comprehensive log analysis toolkit for server logs.

### Log Format

Assuming Apache Combined Log Format:
```
127.0.0.1 - - [15/Jan/2025:10:30:00 +0000] "GET /api/users HTTP/1.1" 200 1234
```

### Prolog Definitions

```prolog
% file: log_analyzer.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

%% ============================================
%% Parse log entry
%% ============================================

% Extract key fields from log line
parse_log(Line, IP, Method, Path, Status, Size) :-
    log_line(Line),
    match(Line,
        "([0-9.]+) .* \"([A-Z]+) ([^ ]+) .*\" ([0-9]+) ([0-9]+)",
        [IP, Method, Path, Status, Size]).

%% ============================================
%% Error Analysis
%% ============================================

% Find 5xx server errors
server_errors(IP, Path, Status) :-
    log_line(Line),
    match(Line, "([0-9.]+) .* \"[A-Z]+ ([^ ]+) .*\" (5[0-9]{2})", [IP, Path, Status]).

% Find 4xx client errors
client_errors(IP, Path, Status) :-
    log_line(Line),
    match(Line, "([0-9.]+) .* \"[A-Z]+ ([^ ]+) .*\" (4[0-9]{2})", [IP, Path, Status]).

%% ============================================
%% Traffic Analysis
%% ============================================

% Count requests by IP
requests_by_ip(IP, Count) :-
    log_line(Line),
    match(Line, "^([0-9.]+)", [IP]),
    Count = 1.

% Count requests by path
requests_by_path(Path, Count) :-
    log_line(Line),
    match(Line, "\"[A-Z]+ ([^ ]+)", [Path]),
    Count = 1.

% Total bytes transferred
total_bytes(Bytes) :-
    log_line(Line),
    match(Line, "\" [0-9]+ ([0-9]+)$", [Bytes]).

%% ============================================
%% Slow Endpoints (assuming response time in extended format)
%% ============================================

% If log includes response time: ... 200 1234 0.523
slow_requests(Path, ResponseTime) :-
    log_line(Line),
    match(Line, "\"[A-Z]+ ([^ ]+) .*\" [0-9]+ [0-9]+ ([0-9.]+)", [Path, ResponseTime]),
    ResponseTime > 1.0.

%% ============================================
%% Generate all scripts
%% ============================================

generate :-
    compile_predicate_to_awk(parse_log/6, [], AWK1),
    write_awk_script('parse_log.awk', AWK1),

    compile_predicate_to_awk(server_errors/3, [], AWK2),
    write_awk_script('server_errors.awk', AWK2),

    compile_predicate_to_awk(client_errors/3, [], AWK3),
    write_awk_script('client_errors.awk', AWK3),

    compile_predicate_to_awk(requests_by_ip/2, [aggregation(count)], AWK4),
    write_awk_script('requests_by_ip.awk', AWK4),

    compile_predicate_to_awk(requests_by_path/2, [aggregation(count)], AWK5),
    write_awk_script('requests_by_path.awk', AWK5),

    compile_predicate_to_awk(total_bytes/1, [aggregation(sum)], AWK6),
    write_awk_script('total_bytes.awk', AWK6),

    format('Generated log analysis scripts~n').

:- initialization(generate, main).
```

### Usage

```bash
# Generate all scripts
swipl log_analyzer.pl

# Find server errors
awk -f server_errors.awk access.log

# Count requests per IP (top 10)
awk -f requests_by_ip.awk access.log | sort -t$'\t' -k2 -rn | head -10

# Total bandwidth
awk -f total_bytes.awk access.log
```

## Application 2: CSV Data Processor

Process and transform CSV files.

### Prolog Definitions

```prolog
% file: csv_processor.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

%% ============================================
%% Data Cleaning
%% ============================================

% Remove rows with missing required fields
valid_rows(Id, Name, Email, Amount) :-
    record(Id, Name, Email, Amount),
    Id \= "",
    Name \= "",
    Email \= "",
    Amount \= "".

% Normalize email to lowercase
normalized_email(Id, Name, LowerEmail) :-
    record(Id, Name, Email, _),
    LowerEmail = sql_lower(Email).  % Will use AWK's tolower()

%% ============================================
%% Data Transformation
%% ============================================

% Convert cents to dollars
amount_dollars(Id, Name, Dollars) :-
    record(Id, Name, _, AmountCents),
    Dollars is AmountCents / 100.

% Add calculated field
with_tax(Id, Name, Amount, Tax, Total) :-
    record(Id, Name, _, Amount),
    Tax is Amount * 0.08,
    Total is Amount + Tax.

%% ============================================
%% Filtering
%% ============================================

% High value transactions
high_value(Id, Name, Amount) :-
    record(Id, Name, _, Amount),
    Amount > 1000.

% Filter by date range (assuming YYYY-MM-DD format)
date_range(Id, Name, Date, Amount) :-
    record(Id, Name, Date, Amount),
    Date >= "2025-01-01",
    Date =< "2025-01-31".

%% ============================================
%% Aggregations
%% ============================================

% Total by customer
customer_total(Name, Total) :-
    record(_, Name, _, Amount),
    Total = Amount.

% Statistics
total_amount(Total) :-
    record(_, _, _, Amount),
    Total = Amount.

count_records(N) :-
    record(_, _, _, _),
    N = 1.

avg_amount(Avg) :-
    record(_, _, _, Amount),
    Avg = Amount.

%% ============================================
%% Generate scripts
%% ============================================

generate :-
    compile_predicate_to_awk(valid_rows/4, [
        record_format(csv),
        include_header(true)
    ], AWK1),
    write_awk_script('valid_rows.awk', AWK1),

    compile_predicate_to_awk(with_tax/5, [
        record_format(csv),
        include_header(true)
    ], AWK2),
    write_awk_script('with_tax.awk', AWK2),

    compile_predicate_to_awk(high_value/3, [
        record_format(csv),
        include_header(true)
    ], AWK3),
    write_awk_script('high_value.awk', AWK3),

    compile_predicate_to_awk(customer_total/2, [
        record_format(csv),
        include_header(true),
        aggregation(sum)
    ], AWK4),
    write_awk_script('customer_total.awk', AWK4),

    compile_predicate_to_awk(total_amount/1, [
        record_format(csv),
        include_header(true),
        aggregation(sum)
    ], AWK5),
    write_awk_script('total_amount.awk', AWK5),

    format('Generated CSV processing scripts~n').

:- initialization(generate, main).
```

### Usage

```bash
# Generate scripts
swipl csv_processor.pl

# Clean data
awk -f valid_rows.awk data.csv > clean_data.csv

# Add tax calculations
awk -f with_tax.awk orders.csv > orders_with_tax.csv

# Get customer totals
awk -f customer_total.awk orders.csv

# High value only
awk -f high_value.awk orders.csv
```

## Application 3: System Monitoring

Monitor system metrics from various sources.

### Prolog Definitions

```prolog
% file: sys_monitor.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

%% ============================================
%% Process /etc/passwd analysis
%% ============================================

% List shell users (not system accounts)
shell_users(User, Shell) :-
    passwd_entry(User, _, UID, _, _, _, Shell),
    UID >= 1000,
    Shell \= "/usr/sbin/nologin",
    Shell \= "/bin/false".

% Find users in specific group
group_members(User, GID) :-
    passwd_entry(User, _, _, GID, _, _, _).

%% ============================================
%% Disk usage (from df -h output)
%% ============================================

% Find filesystems above threshold
disk_warning(Filesystem, UsedPercent, MountPoint) :-
    df_line(Filesystem, _, _, _, UsedPercent, MountPoint),
    UsedPercent > 80.

%% ============================================
%% Network connections (from netstat/ss)
%% ============================================

% Count connections by state
conn_by_state(State, Count) :-
    netstat_line(_, _, _, _, State),
    Count = 1.

% Find established connections to specific port
port_connections(LocalAddr, RemoteAddr) :-
    netstat_line(_, LocalAddr, RemoteAddr, _, State),
    match(LocalAddr, ":22$"),
    State = "ESTABLISHED".

%% ============================================
%% Memory usage (from /proc/meminfo)
%% ============================================

% Extract memory values
mem_value(Key, Value) :-
    meminfo_line(Line),
    match(Line, "^([A-Za-z_]+):\\s+([0-9]+)", [Key, Value]).

%% ============================================
%% Generate scripts
%% ============================================

generate :-
    compile_predicate_to_awk(shell_users/2, [field_separator(':')], AWK1),
    write_awk_script('shell_users.awk', AWK1),

    compile_predicate_to_awk(disk_warning/3, [], AWK2),
    write_awk_script('disk_warning.awk', AWK2),

    compile_predicate_to_awk(conn_by_state/2, [aggregation(count)], AWK3),
    write_awk_script('conn_by_state.awk', AWK3),

    format('Generated system monitoring scripts~n').

:- initialization(generate, main).
```

### Usage

```bash
# Generate scripts
swipl sys_monitor.pl

# List interactive users
awk -f shell_users.awk /etc/passwd

# Check disk usage
df -h | awk -f disk_warning.awk

# Connection states
ss -ta | awk -f conn_by_state.awk
```

## Application 4: Data Pipeline

Combine multiple AWK scripts into a processing pipeline.

### Pipeline Components

```prolog
% file: pipeline.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

%% ============================================
%% Stage 1: Parse raw input
%% ============================================

parse_input(Timestamp, Level, Message) :-
    raw_line(Line),
    match(Line, "([0-9-T:]+) \\[([A-Z]+)\\] (.*)", [Timestamp, Level, Message]).

%% ============================================
%% Stage 2: Filter relevant entries
%% ============================================

filter_errors(Timestamp, Message) :-
    parsed(Timestamp, Level, Message),
    Level = "ERROR".

filter_warnings(Timestamp, Message) :-
    parsed(Timestamp, Level, Message),
    Level = "WARN".

%% ============================================
%% Stage 3: Enrich with context
%% ============================================

% Add severity score
with_severity(Timestamp, Message, Severity) :-
    error_entry(Timestamp, Message),
    Severity = 10.

with_severity(Timestamp, Message, Severity) :-
    warning_entry(Timestamp, Message),
    Severity = 5.

%% ============================================
%% Stage 4: Aggregate
%% ============================================

error_count(N) :-
    error_entry(_, _),
    N = 1.

%% ============================================
%% Generate pipeline scripts
%% ============================================

generate :-
    compile_predicate_to_awk(parse_input/3, [], AWK1),
    write_awk_script('stage1_parse.awk', AWK1),

    compile_predicate_to_awk(filter_errors/2, [], AWK2),
    write_awk_script('stage2_errors.awk', AWK2),

    compile_predicate_to_awk(error_count/1, [aggregation(count)], AWK3),
    write_awk_script('stage3_count.awk', AWK3),

    format('Generated pipeline scripts~n').

:- initialization(generate, main).
```

### Pipeline Usage

```bash
# Full pipeline
cat raw_logs.txt \
  | awk -f stage1_parse.awk \
  | awk -f stage2_errors.awk \
  | awk -f stage3_count.awk

# Save intermediate results
awk -f stage1_parse.awk raw_logs.txt > parsed.tsv
awk -f stage2_errors.awk parsed.tsv > errors.tsv
awk -f stage3_count.awk errors.tsv
```

## Application 5: Mathematical Computations

Combine tail recursion with practical math.

### Prolog Definitions

```prolog
% file: math_tools.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

%% ============================================
%% Compound Interest Calculator
%% ============================================

compound_interest(Principal, 0, _, Acc, Acc).
compound_interest(Principal, Years, Rate, Acc, Result) :-
    Years > 0,
    Years1 is Years - 1,
    Acc1 is Acc * (1 + Rate),
    compound_interest(Principal, Years1, Rate, Acc1, Result).

calc_compound(Principal, Years, Rate, FinalValue) :-
    compound_interest(Principal, Years, Rate, Principal, FinalValue).

%% ============================================
%% Loan Payment Calculator
%% ============================================

% Monthly payment = P * (r(1+r)^n) / ((1+r)^n - 1)
% Using tail recursion to compute (1+r)^n

power_acc(_, 0, Acc, Acc).
power_acc(Base, N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * Base,
    power_acc(Base, N1, Acc1, Result).

monthly_payment(Principal, AnnualRate, Years, Payment) :-
    input(Principal, AnnualRate, Years),
    MonthlyRate is AnnualRate / 12,
    Months is Years * 12,
    power_acc(1 + MonthlyRate, Months, 1, Factor),
    Payment is Principal * (MonthlyRate * Factor) / (Factor - 1).

%% ============================================
%% Statistical Functions
%% ============================================

% Running variance (Welford's algorithm)
running_variance(0, _, M2, _, M2).
running_variance(N, Mean, M2, Values, Result) :-
    N > 0,
    % Implementation would be complex for AWK
    Result = M2.

%% ============================================
%% Numeric Integration (Trapezoidal)
%% ============================================

% Sum for trapezoidal rule
trap_sum(0, _, _, Acc, Acc).
trap_sum(N, A, H, Acc, Result) :-
    N > 0,
    X is A + N * H,
    % F = X * X for example function f(x) = x^2
    F is X * X,
    Acc1 is Acc + F,
    N1 is N - 1,
    trap_sum(N1, A, H, Acc1, Result).

%% ============================================
%% Generate scripts
%% ============================================

generate :-
    compile_predicate_to_awk(calc_compound/4, [], AWK1),
    write_awk_script('compound.awk', AWK1),

    format('Generated math scripts~n').

:- initialization(generate, main).
```

### Usage

```bash
# Calculate compound interest
# Input: Principal Years Rate
echo "10000 5 0.05" | awk -f compound.awk
# Output: 12762.82 (approximately)
```

## Best Practices Summary

### 1. Script Organization

```bash
project/
├── predicates/           # Prolog source files
│   ├── log_analysis.pl
│   ├── data_cleaning.pl
│   └── aggregations.pl
├── scripts/              # Generated AWK scripts
│   ├── parse_log.awk
│   ├── clean_data.awk
│   └── summary.awk
├── data/                 # Input data
│   ├── logs/
│   └── csv/
└── Makefile             # Build automation
```

### 2. Makefile for Automation

```makefile
SWIPL = swipl
SCRIPTS = scripts/parse_log.awk scripts/clean_data.awk

all: $(SCRIPTS)

scripts/%.awk: predicates/%.pl
	$(SWIPL) $<

clean:
	rm -f scripts/*.awk
```

### 3. Testing

```bash
# Test with sample data
echo -e "Alice\tEngineering\t75000" | awk -f filter.awk

# Compare with expected output
awk -f script.awk test_input.txt | diff - expected_output.txt
```

### 4. Performance Tips

- Use `include_header(true)` to skip headers
- Use `unique(true)` sparingly (requires memory)
- Prefer simple regex patterns
- Use grouped aggregations instead of multiple passes

## Exercises

1. **Log Analyzer**: Create a complete log analysis suite for your application's logs.

2. **ETL Pipeline**: Build a 3-stage ETL pipeline: extract, transform, load for CSV data.

3. **System Reporter**: Create scripts to generate a daily system health report.

4. **Financial Calculator**: Build a suite of financial calculators: compound interest, loan payments, depreciation.

5. **Text Processor**: Create a text analysis toolkit: word count, frequency analysis, pattern extraction.

## Summary

In this chapter, you learned:

- Building complete applications with multiple AWK scripts
- Log file analysis with regex and aggregations
- CSV data processing and transformation
- System monitoring scripts
- Creating data processing pipelines
- Mathematical computations with tail recursion
- Best practices for organization and testing

## Conclusion

Congratulations on completing the AWK Target book! You now know how to:

- Generate portable AWK scripts from Prolog predicates
- Use facts, rules, constraints, and filters
- Apply aggregation operations
- Implement tail-recursive algorithms
- Use regex pattern matching
- Build complete data processing applications

AWK scripts generated by UnifyWeaver are self-contained, portable, and integrate seamlessly with Unix pipelines. Use them for log analysis, data transformation, and any text processing task.
