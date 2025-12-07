<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 6: Regex Pattern Matching

This chapter covers the `match/2`, `match/3`, and `match/4` predicates for regular expression operations in the AWK target.

## Overview

The AWK target provides regex pattern matching through dedicated predicates:

| Predicate | Purpose |
|-----------|---------|
| `match(String, Pattern)` | Boolean match test |
| `match(String, Pattern, Captures)` | Match with capture groups |
| `match(String, Pattern, Captures, Type)` | Match with capture and regex type |

## Boolean Matching with match/2

The simplest form tests whether a string matches a pattern.

### Basic Usage

```prolog
% Match email pattern
valid_email(Email) :-
    user(_, Email),
    match(Email, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}").

% Match phone number
valid_phone(Phone) :-
    contact(_, Phone),
    match(Phone, "^[0-9]{3}-[0-9]{3}-[0-9]{4}$").
```

Generated AWK:

```awk
{
    Email = $2
    if (Email ~ /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/) {
        print Email
    }
}
```

### Common Patterns

```prolog
% Starts with
starts_with_a(Name) :-
    person(Name),
    match(Name, "^A").

% Ends with
ends_with_son(Name) :-
    person(Name),
    match(Name, "son$").

% Contains
contains_error(Line) :-
    log_line(Line),
    match(Line, "ERROR").

% Word boundary (approximate in AWK)
word_match(Text) :-
    document(Text),
    match(Text, "(^|[^a-zA-Z])test([^a-zA-Z]|$)").
```

## Capture Groups with match/3

Extract matched substrings using capture groups.

### Basic Capture

```prolog
% Extract username from email
extract_user(Email, Username) :-
    contact(_, Email),
    match(Email, "([^@]+)@", [Username]).
```

Generated AWK (using `match()` function):

```awk
{
    Email = $2
    if (match(Email, /([^@]+)@/, captures)) {
        Username = captures[1]
        print Username
    }
}
```

### Multiple Captures

```prolog
% Parse date: YYYY-MM-DD
parse_date(DateStr, Year, Month, Day) :-
    record(_, DateStr),
    match(DateStr, "([0-9]{4})-([0-9]{2})-([0-9]{2})", [Year, Month, Day]).

% Parse log entry: [LEVEL] message
parse_log(Line, Level, Message) :-
    log(Line),
    match(Line, "\\[([A-Z]+)\\] (.*)", [Level, Message]).
```

### Capture with Optional Groups

```prolog
% Parse version: major.minor[.patch]
parse_version(Ver, Major, Minor, Patch) :-
    software(_, Ver),
    match(Ver, "([0-9]+)\\.([0-9]+)(\\.([0-9]+))?", [Major, Minor, _, Patch]).
```

## Regex Types with match/4

Specify the regex dialect explicitly.

### Available Types

| Type | Description |
|------|-------------|
| `auto` | AWK's native regex (default) |
| `ere` | Extended Regular Expressions |
| `bre` | Basic Regular Expressions |
| `awk` | AWK-specific regex |

### Using Different Types

```prolog
% Explicit ERE
match_ere(Text, Pattern, Captures) :-
    input(Text),
    match(Text, Pattern, Captures, ere).

% Explicit BRE (requires different escaping)
match_bre(Text, Pattern, Captures) :-
    input(Text),
    match(Text, Pattern, Captures, bre).
```

### When to Use Each Type

```prolog
% ERE: Standard extended regex (most common)
email_ere(Email, User, Domain) :-
    record(Email),
    match(Email, "([^@]+)@(.+)", [User, Domain], ere).

% AWK: AWK's native (similar to ERE)
email_awk(Email, User, Domain) :-
    record(Email),
    match(Email, "([^@]+)@(.+)", [User, Domain], awk).
```

## Pattern Escaping

### Special Characters

In Prolog strings, backslashes need double escaping:

```prolog
% Match literal dot
dot_match(Text) :-
    input(Text),
    match(Text, "\\.").  % Matches literal "."

% Match digits
digit_match(Text) :-
    input(Text),
    match(Text, "[0-9]+").  % Character class

% Match whitespace
space_match(Text) :-
    input(Text),
    match(Text, "[ \\t]+").  % Space or tab
```

### Common Escape Sequences

| Prolog | Regex | Matches |
|--------|-------|---------|
| `\\.` | `\.` | Literal dot |
| `\\\\` | `\\` | Literal backslash |
| `\\[` | `\[` | Literal bracket |
| `\\(` | `\(` | Literal parenthesis |
| `\\t` | Tab | Tab character |
| `\\n` | Newline | Newline character |

## Practical Examples

### Log File Analysis

```prolog
% Parse Apache log format
apache_log(Line, IP, Method, Path, Status) :-
    log_line(Line),
    match(Line,
        "([0-9.]+) .* \"([A-Z]+) ([^ ]+) .*\" ([0-9]+)",
        [IP, Method, Path, Status]).

% Extract error messages
error_message(Line, ErrorType, Message) :-
    log_line(Line),
    match(Line, "ERROR: ([A-Za-z]+): (.*)", [ErrorType, Message]).
```

### Data Validation

```prolog
% Validate IP address (simplified)
valid_ip(IP) :-
    network(IP),
    match(IP, "^[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}$").

% Validate UUID
valid_uuid(UUID) :-
    record(UUID),
    match(UUID, "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$").

% Validate credit card (simplified)
valid_card(Card) :-
    payment(Card),
    match(Card, "^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}$").
```

### URL Parsing

```prolog
% Extract URL components
parse_url(URL, Protocol, Host, Path) :-
    link(URL),
    match(URL, "^([a-z]+)://([^/]+)(/.*)?$", [Protocol, Host, Path]).

% Extract domain from URL
domain(URL, Domain) :-
    link(URL),
    match(URL, "://([^/:]+)", [Domain]).
```

### Text Extraction

```prolog
% Extract quoted strings
quoted_text(Line, Text) :-
    document(Line),
    match(Line, "\"([^\"]*)\"", [Text]).

% Extract bracketed content
bracketed(Line, Content) :-
    document(Line),
    match(Line, "\\[([^\\]]*)\\]", [Content]).

% Extract key=value pairs
key_value(Line, Key, Value) :-
    config(Line),
    match(Line, "^([^=]+)=(.*)$", [Key, Value]).
```

## Negated Matching

To match lines that DON'T match a pattern:

```prolog
% Lines without errors
clean_lines(Line) :-
    log(Line),
    \+ match(Line, "ERROR|WARN|FATAL").
```

Generated AWK:

```awk
{
    Line = $0
    if (Line !~ /ERROR|WARN|FATAL/) {
        print Line
    }
}
```

## Combining with Other Features

### Regex + Filtering

```prolog
% Valid emails from active users
active_valid_email(Email) :-
    user(Name, Email, Status),
    Status = "active",
    match(Email, "[a-z]+@[a-z]+\\.[a-z]+").
```

### Regex + Aggregation

```prolog
% Count error types
error_count(ErrorType, N) :-
    log(Line),
    match(Line, "ERROR: ([A-Za-z]+):", [ErrorType]),
    N = 1.

?- compile_predicate_to_awk(error_count/2, [aggregation(count)], AWK).
```

## Complete Example

```prolog
% file: regex_matching.pl
:- encoding(utf8).
:- use_module('src/unifyweaver/targets/awk_target').

% Boolean match - valid emails
valid_email(Email) :-
    contact(_, Email, _),
    match(Email, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}").

% Capture - parse email
email_parts(Email, User, Domain) :-
    contact(_, Email, _),
    match(Email, "([^@]+)@(.*)", [User, Domain]).

% Parse date
date_parts(DateStr, Year, Month, Day) :-
    record(_, DateStr, _),
    match(DateStr, "([0-9]{4})-([0-9]{2})-([0-9]{2})", [Year, Month, Day]).

% Parse log entry
log_entry(Line, Level, Timestamp, Message) :-
    log_line(Line),
    match(Line, "\\[([A-Z]+)\\] ([0-9:T-]+) (.*)", [Level, Timestamp, Message]).

% Filter errors only
errors_only(Line, Message) :-
    log_line(Line),
    match(Line, "\\[ERROR\\] [^ ]+ (.*)", [Message]).

% Generate scripts
generate :-
    compile_predicate_to_awk(valid_email/1, [], AWK1),
    write_awk_script('valid_email.awk', AWK1),

    compile_predicate_to_awk(email_parts/3, [], AWK2),
    write_awk_script('email_parts.awk', AWK2),

    compile_predicate_to_awk(date_parts/4, [], AWK3),
    write_awk_script('date_parts.awk', AWK3),

    compile_predicate_to_awk(log_entry/4, [], AWK4),
    write_awk_script('log_entry.awk', AWK4),

    compile_predicate_to_awk(errors_only/2, [], AWK5),
    write_awk_script('errors_only.awk', AWK5),

    format('Generated regex AWK scripts~n').

:- initialization(generate, main).
```

Test data (`contacts.tsv`):
```
Alice	alice@example.com	active
Bob	bob.smith@company.org	active
Carol	invalid-email	inactive
Dave	dave@test.io	active
```

Test data (`logs.txt`):
```
[INFO] 2025-01-15T10:30:00 Server started
[ERROR] 2025-01-15T10:31:00 Connection failed
[WARN] 2025-01-15T10:32:00 High memory usage
[ERROR] 2025-01-15T10:33:00 Database timeout
```

Run:
```bash
swipl regex_matching.pl

echo "Filtering valid emails:"
awk -f valid_email.awk contacts.tsv

echo "Parsing email parts:"
awk -f email_parts.awk contacts.tsv

echo "Extracting errors:"
awk -f errors_only.awk logs.txt
```

## Summary

In this chapter, you learned:

- Boolean matching with `match/2`
- Capture groups with `match/3`
- Regex types (auto, ere, bre, awk) with `match/4`
- Pattern escaping in Prolog strings
- Practical patterns: logs, emails, dates, URLs
- Negated matching
- Combining regex with filtering and aggregation

## Next Chapter

In Chapter 7, we'll bring everything together with practical applications: log analysis, data transformation, and pipeline integration.
