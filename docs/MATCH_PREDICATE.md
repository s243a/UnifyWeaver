# Match Predicate - Cross-Target Regex Pattern Matching

The `match` predicate provides unified regex pattern matching support across multiple compilation targets, with full capture group extraction.

## Table of Contents

1. [API Overview](#api-overview)
2. [Target Support](#target-support)
3. [Regex Type Support](#regex-type-support)
4. [Boolean Matching](#boolean-matching)
5. [Capture Groups](#capture-groups)
6. [Examples by Target](#examples-by-target)
7. [Limitations](#limitations)
8. [Best Practices](#best-practices)

---

## API Overview

The `match` predicate comes in multiple forms:

```prolog
% Boolean match with auto type detection (defaults per target)
match(String, Pattern)

% Boolean match with explicit regex type
match(String, Pattern, RegexType)

% Match with capture group extraction
match(String, Pattern, RegexType, CaptureList)
```

### Parameters

- **String**: Variable containing the text to match
- **Pattern**: Regex pattern (atom or string)
- **RegexType**: Type of regex (`auto`, `ere`, `bre`, `awk`, `python`, `pcre`)
- **CaptureList**: List of variables to receive captured groups

---

## Target Support

| Target | Boolean Match | Capture Groups | Status |
|--------|---------------|----------------|--------|
| **AWK** | ✅ | ✅ | Complete |
| **Python** | ✅ | ✅ | Complete |
| **Bash** | 🚧 | 🚧 | Planned |
| **C#** | ❌ | ❌ | Not yet |
| **Prolog** | ❌ | ❌ | Not yet |

---

## Regex Type Support

### AWK Target

| Type | Description | Status |
|------|-------------|--------|
| `auto` | Auto-detect (uses ERE) | ✅ Supported |
| `ere` | POSIX Extended RE | ✅ Supported |
| `bre` | POSIX Basic RE | ✅ Supported |
| `awk` | AWK-specific regex | ✅ Supported |
| `pcre` | Perl Compatible RE | ❌ Not supported |
| `python` | Python regex | ❌ Not supported |

### Python Target

| Type | Description | Status |
|------|-------------|--------|
| `auto` | Auto-detect (uses Python re) | ✅ Supported |
| `python` | Python regex | ✅ Supported |
| `pcre` | PCRE-like (Python re) | ✅ Supported |
| `ere` | POSIX ERE | ✅ Supported |
| `bre` | POSIX Basic RE | ❌ Not supported |
| `awk` | AWK-specific | ❌ Not supported |

**Note:** Attempting to use an unsupported type will fail with a clear error message at compile time.

---

## Boolean Matching

Boolean matching checks if a string matches a pattern without extracting values.

### Prolog Code

```prolog
% Match ERROR lines (works across all targets)
error_line(Line) :-
    log(error, Line),
    match(Line, 'ERROR').

% Match timeout errors with explicit regex type
timeout_error(Line) :-
    log(error, Line),
    match(Line, 'ERROR.*timeout', auto).
```

### Generated Code

**AWK:**
```awk
{
    if (($1 ~ /ERROR/)) {
        print $0
    }
}
```

**Python:**
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('message') != v_1: return
    if not re.search(r'ERROR', str(v_1)): return
    yield v_0
```

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

### Generated Code

**AWK:**
```awk
{
    key = $1
    if (key in log_data && match($1, /([0-9-]+ [0-9:]+) ([A-Z]+)/, __captures__)) {
        if (!(key in seen)) {
            seen[key] = 1
            print __captures__[1], __captures__[2]
        }
    }
}
```

**Python:**
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('line') != v_3: return
    __match__ = re.search(r'([0-9-]+ [0-9:]+) ([A-Z]+)', str(v_3))
    if not __match__: return
    v_1 = __match__.group(1)
    v_2 = __match__.group(2)
    # ... use captured values
    yield result
```

---

## Examples by Target

### AWK Target Examples

See [AWK_MATCH_PREDICATE.md](AWK_MATCH_PREDICATE.md) for detailed AWK-specific examples including:
- Log filtering
- IP address extraction
- CSV parsing
- Date component extraction
- Word boundary matching

### Python Target Examples

#### Example 1: Filter Error Records

```prolog
:- use_module('src/unifyweaver/targets/python_target').

filter_errors(Record) :-
    get_dict(message, Record, Line),
    match(Line, 'ERROR', python).

% Compile
?- python_target:compile_predicate_to_python(filter_errors/1, [], Code).
```

**Generated Python:**
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('message') != v_1: return
    if not re.search(r'ERROR', str(v_1)): return
    yield v_0
```

#### Example 2: Extract IP from Access Log

```prolog
parse_ip(Record, IP) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)', python, [IP]),
    Record = _{line: Line, ip: IP}.
```

**Generated Python:**
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('line') != v_2: return
    __match__ = re.search(r'([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', str(v_2))
    if not __match__: return
    v_1 = __match__.group(1)
    v_0 = {'ip': v_1, 'line': v_2}
    yield v_1
```

#### Example 3: Parse Timestamp and Level

```prolog
parse_log(Record, Time, Level) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', python, [Time, Level]),
    Record = _{line: Line, time: Time, level: Level}.
```

**Generated Python:**
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('line') != v_3: return
    __match__ = re.search(r'([0-9-]+ [0-9:]+) ([A-Z]+)', str(v_3))
    if not __match__: return
    v_1 = __match__.group(1)
    v_2 = __match__.group(2)
    v_0 = {'level': v_2, 'line': v_3, 'time': v_1}
    yield v_2
```

---

## Limitations

### Current Limitations

1. **AWK - Output only**: Captured values are printed, not used in further constraints
   - ✅ Works: `match(Line, '(\\d+)', ere, [Num])` → prints Num
   - ❌ Doesn't work yet: Using Num in arithmetic constraints

2. **Python - Procedural mode only**: Match currently works in procedural mode
   - ✅ Works: Procedural compilation
   - 🚧 Partial: Generator mode (builtin support added, needs testing)

3. **Cross-target regex syntax**: Different targets support different regex flavors
   - Use `auto` type for maximum portability
   - Or specify target-appropriate type explicitly

### Workarounds

**For multiple matches in AWK:**
```prolog
% Instead of:
parse_both(Line, IP, Date) :-
    match(Line, '([0-9.]+)', ere, [IP]),    % Won't work
    match(Line, '([0-9-]+)', ere, [Date]).  % in same rule

% Do this:
parse_both(Line, IP, Date) :-
    match(Line, '([0-9.]+).*([0-9-]+)', ere, [IP, Date]).
```

**For arithmetic on captures (AWK):**
```prolog
% Not yet supported:
process_num(Line, Double) :-
    match(Line, '(\\d+)', ere, [Num]),
    Double is Num * 2.  % Can't use Num this way yet

% Workaround: Post-process with AWK or pipe to another stage
```

---

## Best Practices

### 1. Choose the Right Regex Type

```prolog
% Good: Use auto for portability
match(Line, 'ERROR', auto)

% Also good: Use target-specific when needed
match(Line, '(?P<name>\\w+)', python)  % Named groups in Python
```

### 2. Anchor Patterns for Performance

```prolog
% Good: Anchored pattern
match(Line, '^ERROR', ere)  % Only checks start

% Less efficient: Unanchored
match(Line, 'ERROR', ere)   % Scans entire string
```

### 3. Combine Captures

```prolog
% Good: One pattern with multiple groups
match(Line, '([0-9.]+).*([0-9-]+)', ere, [IP, Date])

% Less efficient: Multiple match calls
match(Line, '([0-9.]+)', ere, [IP]),
match(Line, '([0-9-]+)', ere, [Date])  % Might not work in all contexts
```

### 4. Use Boolean Match When Possible

```prolog
% Good: Just checking if pattern exists
match(Line, 'ERROR', ere)  % Faster than captures

% Unnecessary: Extracting when you don't need the value
match(Line, '(ERROR)', ere, [_])  % Slower
```

### 5. Escape Special Characters

```prolog
% Good: Escaped dots for literal match
match(Line, '([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)', ere, [IP])

% Wrong: Unescaped dots match any character
match(Line, '([0-9]+.[0-9]+.[0-9]+.[0-9]+)', ere, [IP])
```

---

## Performance Notes

### AWK Target

- **Boolean match**: O(n) where n = string length
- **Uses native `~` operator**: Very fast
- **Capture extraction**: Negligible overhead
- **Best for**: Large file processing, log analysis

### Python Target

- **Boolean match**: O(n) with Python `re` module
- **Capture extraction**: Minimal overhead
- **Dict-based**: Works with record-oriented data
- **Best for**: Complex data transformations, integration with Python ecosystem

---

## Integration with Other Features

### With Aggregation (AWK)

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

### With Dict Operations (Python)

```prolog
% Extract and transform
process_log(Record) :-
    get_dict(message, Record, Line),
    match(Line, 'ERROR: (.+)', python, [Msg]),
    Record = _{message: Line, error_msg: Msg}.
```

---

## Future Enhancements

Planned improvements:

1. **Use captures in constraints**: Allow arithmetic/comparison on captured values
2. **Named captures**: Support for named capture groups
3. **Regex translation**: Auto-translate between compatible types
4. **More targets**: C#, Bash, Prolog native
5. **Match flags**: Case-insensitive, multiline, etc.

---

## See Also

- [AWK_MATCH_PREDICATE.md](AWK_MATCH_PREDICATE.md) - AWK-specific details and examples
- [AWK_TARGET_EXAMPLES.md](AWK_TARGET_EXAMPLES.md) - General AWK target examples
- [AWK_TARGET_STATUS.md](AWK_TARGET_STATUS.md) - AWK target implementation status
- [Python Target Documentation](../src/unifyweaver/targets/python_target.pl) - Python target source

---

## Quick Reference

### Syntax Summary

```prolog
% Boolean match (auto type)
match(Var, Pattern)

% Boolean match (explicit type)
match(Var, Pattern, Type)

% With captures
match(Var, Pattern, Type, [Cap1, Cap2, ...])
```

### Supported Types by Target

| Type | AWK | Python | Bash |
|------|-----|--------|------|
| `auto` | ✅ ERE | ✅ Python re | 🚧 POSIX |
| `ere` | ✅ | ✅ | 🚧 |
| `bre` | ✅ | ❌ | 🚧 |
| `awk` | ✅ | ❌ | ❌ |
| `python` | ❌ | ✅ | ❌ |
| `pcre` | ❌ | ✅ | ❌ |

✅ = Supported | ❌ = Not supported | 🚧 = Planned
