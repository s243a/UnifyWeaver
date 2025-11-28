# AWK Match Predicate - Regex Pattern Matching

The `match` predicate provides regex pattern matching support for the AWK target, with full capture group extraction.

## Table of Contents

1. [API Overview](#api-overview)
2. [Regex Type Support](#regex-type-support)
3. [Boolean Matching](#boolean-matching)
4. [Capture Groups](#capture-groups)
5. [Examples](#examples)
6. [Limitations](#limitations)

---

## API Overview

The `match` predicate comes in multiple forms:

```prolog
% Boolean match with auto type detection (defaults to ERE)
match(String, Pattern)

% Boolean match with explicit regex type
match(String, Pattern, RegexType)

% Match with capture group extraction
match(String, Pattern, RegexType, CaptureList)
```

### Parameters

- **String**: Variable containing the text to match
- **Pattern**: Regex pattern (atom or string)
- **RegexType**: Type of regex (`auto`, `ere`, `bre`, `awk`)
- **CaptureList**: List of variables to receive captured groups

---

## Regex Type Support

### Supported Types for AWK

| Type | Description | Status |
|------|-------------|--------|
| `auto` | Auto-detect (uses ERE for AWK) | ✅ Supported |
| `ere` | POSIX Extended Regular Expressions | ✅ Supported |
| `bre` | POSIX Basic Regular Expressions | ✅ Supported |
| `awk` | AWK-specific regex | ✅ Supported |

### Unsupported Types

| Type | Description | Status |
|------|-------------|--------|
| `pcre` | Perl Compatible RE | ❌ Not supported by AWK |
| `python` | Python regex | ❌ Not supported by AWK |
| `dotnet` | .NET regex | ❌ Not supported by AWK |

Attempting to use an unsupported type will fail with a clear error message at compile time.

---

## Boolean Matching

Boolean matching checks if a string matches a pattern without extracting values.

### Prolog Code

```prolog
% Match ERROR lines
error_line(Line) :-
    log(error, Line),
    match(Line, 'ERROR').

% Match timeout errors with explicit ERE type
timeout_error(Line) :-
    log(error, Line),
    match(Line, 'ERROR.*timeout', ere).
```

### Generated AWK

```awk
# Boolean match with ~~ operator
{
    if (($1 ~ /ERROR/)) {
        print $0
    }
}

# With constraint
{
    key = $1
    if (key in log_data && ($1 ~ /ERROR.*timeout/)) {
        print $0
    }
}
```

### Features

- Uses AWK's `~` (match) operator
- O(1) performance for simple patterns
- No overhead from capture array allocation

---

## Capture Groups

Capture groups extract parts of matched strings using parentheses in the regex pattern.

### Prolog Code

```prolog
% Extract timestamp and level from log lines
parse_log(Line, Time, Level) :-
    log(Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', ere, [Time, Level]).

% Extract just the timestamp
parse_timestamp(Line, Time) :-
    log(Line),
    match(Line, '([0-9-]+ [0-9:]+)', ere, [Time]).
```

### Generated AWK

```awk
# Two capture groups
{
    key = $1
    if (key in log_data && match($1, /([0-9-]+ [0-9:]+) ([A-Z]+)/, __captures__)) {
        if (!(key in seen)) {
            seen[key] = 1
            print __captures__[1], __captures__[2]
        }
    }
}

# Single capture group
{
    if (match($1, /([0-9-]+ [0-9:]+)/, __captures__)) {
        print __captures__[1]
    }
}
```

### Features

- Uses AWK's `match()` function with capture array
- Automatically prints captured values
- Supports 1-N capture groups
- Capture indices start at 1 (AWK convention)

---

## Examples

### Example 1: Filter Log Levels

```prolog
:- use_module('src/unifyweaver/targets/awk_target').

% Log entries
log('2025-01-15 10:30:45 ERROR: connection timeout').
log('2025-01-15 10:31:22 WARNING: slow response').
log('2025-01-15 10:32:10 INFO: request completed').

% Match ERROR or WARNING
important_log(Line) :-
    log(Line),
    match(Line, 'ERROR|WARNING', ere).

% Compile
?- awk_target:compile_predicate_to_awk(important_log/1, [], AwkCode).
```

**Generated AWK:**
```awk
{
    if (($1 ~ /ERROR|WARNING/)) {
        print $0
    }
}
```

**Test:**
```bash
echo -e "ERROR: timeout\nINFO: ok\nWARNING: slow" | awk -f filter.awk
# Output:
# ERROR: timeout
# WARNING: slow
```

### Example 2: Extract IP Addresses

```prolog
% Extract IP from access log
parse_ip(Line, IP) :-
    access_log(Line),
    match(Line, '([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)', ere, [IP]).
```

**Generated AWK:**
```awk
{
    if (match($1, /([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)/, __captures__)) {
        print __captures__[1]
    }
}
```

**Test:**
```bash
echo "192.168.1.100 - GET /index.html" | awk -f extract_ip.awk
# Output: 192.168.1.100
```

### Example 3: Parse CSV Fields

```prolog
% Extract name and email from CSV
parse_user(Line, Name, Email) :-
    user_csv(Line),
    match(Line, '([^,]+),([^@]+@[^,]+)', ere, [Name, Email]).
```

**Generated AWK:**
```awk
{
    if (match($1, /([^,]+),([^@]+@[^,]+)/, __captures__)) {
        print __captures__[1], __captures__[2]
    }
}
```

**Test:**
```bash
echo "John Doe,john@example.com,developer" | awk -f parse_user.awk
# Output: John Doe john@example.com
```

### Example 4: Extract Date Components

```prolog
% Parse ISO date into year, month, day
parse_date(Line, Year, Month, Day) :-
    date_line(Line),
    match(Line, '([0-9]{4})-([0-9]{2})-([0-9]{2})', ere, [Year, Month, Day]).
```

**Generated AWK:**
```awk
{
    if (match($1, /([0-9]{4})-([0-9]{2})-([0-9]{2})/, __captures__)) {
        print __captures__[1], __captures__[2], __captures__[3]
    }
}
```

**Test:**
```bash
echo "2025-01-15" | awk -f parse_date.awk
# Output: 2025 01 15
```

### Example 5: Match with Word Boundaries

```prolog
% Find exact word matches
find_word(Line, Word) :-
    text(Line),
    Word = 'error',
    match(Line, '\\berror\\b', ere).  % \b = word boundary
```

**Generated AWK:**
```awk
{
    if (($1 ~ /\berror\b/)) {
        print $0
    }
}
```

---

## Limitations

### Current Limitations

1. **Output only**: Captured values are printed, not used in further constraints
   - ✅ Works: `match(Line, '(\\d+)', ere, [Num])` → prints Num
   - ❌ Doesn't work yet: Using Num in arithmetic constraints

2. **Single match per rule**: Only one match with captures per rule
   - Multiple boolean matches are fine
   - Multiple match/4 with captures in one rule not yet supported

3. **No nested captures**: Capture groups must be at the same level
   - ✅ Works: `'(\\d+) (\\w+)'`
   - ⚠️ Limited: `'(\\d+) ((\\w+) (\\w+))'` (nested groups)

### Workarounds

**For multiple matches:**
```prolog
% Instead of:
parse_both(Line, IP, Date) :-
    match(Line, '([0-9.]+)', ere, [IP]),    % Won't work
    match(Line, '([0-9-]+)', ere, [Date]).  % in same rule

% Do this:
parse_both(Line, IP, Date) :-
    match(Line, '([0-9.]+).*([0-9-]+)', ere, [IP, Date]).
```

**For arithmetic on captures:**
```prolog
% Not yet supported:
process_num(Line, Double) :-
    match(Line, '(\\d+)', ere, [Num]),
    Double is Num * 2.  % Can't use Num this way yet

% Workaround: Do math in AWK or post-process
```

---

## Performance Notes

### Boolean Match Performance

- **Simple patterns**: O(n) where n = string length
- **Uses native AWK `~` operator**: Very fast
- **No array allocation**: Minimal memory overhead

### Capture Group Performance

- **Pattern complexity**: O(n × m) where m = pattern complexity
- **Capture extraction**: Negligible overhead
- **Memory**: One array per match (reused)

### Best Practices

1. **Use boolean match when possible**: Faster than captures if you don't need extraction
2. **Combine captures**: One pattern with multiple groups is faster than multiple matches
3. **Anchor patterns**: Use `^` and `$` to avoid scanning entire string

```prolog
% Good: Anchored pattern
match(Line, '^ERROR', ere)  % Only checks start

% Less efficient: Unanchored
match(Line, 'ERROR', ere)   % Scans entire string
```

---

## Integration with Other Features

### With Aggregation

```prolog
% Count ERROR lines
error_count(Count) :-
    aggregation(count),
    log(Line),
    match(Line, 'ERROR', ere).
```

### With Constraints

```prolog
% ERROR logs from specific hour
morning_errors(Line) :-
    log(Line),
    match(Line, '10:[0-9]{2}:[0-9]{2} ERROR', ere).
```

### With Facts

```prolog
% Known error patterns
error_pattern('timeout').
error_pattern('connection failed').

% Match any known pattern
known_error(Line) :-
    log(Line),
    error_pattern(Pattern),
    match(Line, Pattern, ere).
```

---

## Future Enhancements

Planned improvements (see [AWK_TARGET_FUTURE_WORK.md](AWK_TARGET_FUTURE_WORK.md)):

1. **Use captures in constraints**: Allow arithmetic/comparison on captured values
2. **Named captures**: Support for named capture groups
3. **Regex translation**: Auto-translate PCRE to ERE where possible
4. **Multiple matches**: Support multiple match/4 in one rule
5. **Match flags**: Case-insensitive, multiline, etc.

---

## See Also

- [AWK_TARGET_EXAMPLES.md](AWK_TARGET_EXAMPLES.md) - General AWK target examples
- [AWK_TARGET_STATUS.md](AWK_TARGET_STATUS.md) - Implementation status
- [AWK_TARGET_FUTURE_WORK.md](AWK_TARGET_FUTURE_WORK.md) - Planned enhancements
