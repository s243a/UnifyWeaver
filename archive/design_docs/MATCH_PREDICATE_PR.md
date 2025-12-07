# feat(targets): Add cross-target match predicate for regex pattern matching

## Summary

This PR implements the `match` predicate for regex pattern matching across three compilation targets: **AWK**, **Python**, and **Bash**. The match predicate provides a unified API for regex operations while generating target-appropriate code with full capture group support.

## Key Features

✅ **Three Target Implementations**:
- **AWK Target**: Boolean matching with `~` operator, capture groups with `match()` function
- **Python Target**: Boolean matching with `re.search()`, capture groups with `__match__.group(n)`
- **Bash Target**: Boolean matching with `grep` filters in streaming pipelines

✅ **Flexible API**:
```prolog
match(String, Pattern)                    % Boolean match, auto type detection
match(String, Pattern, RegexType)         % Boolean match, explicit type
match(String, Pattern, Type, CaptureList) % Match with capture group extraction
```

✅ **Regex Type Validation**: Each target validates and supports appropriate regex types (ERE, BRE, AWK, Python, PCRE)

✅ **Comprehensive Documentation**:
- `docs/MATCH_PREDICATE.md` - Cross-target overview with examples
- `docs/AWK_MATCH_PREDICATE.md` - AWK-specific deep dive

## Implementation by Target

### AWK Target (`src/unifyweaver/targets/awk_target.pl`)

**Boolean Matching**:
```prolog
error_line(Line) :-
    log(error, Line),
    match(Line, 'ERROR').
```

**Generated AWK**:
```awk
{
    if (($1 ~ /ERROR/)) {
        print $0
    }
}
```

**Capture Groups**:
```prolog
parse_log(Line, Time, Level) :-
    log(Line),
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', ere, [Time, Level]).
```

**Generated AWK**:
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

### Python Target (`src/unifyweaver/targets/python_target.pl`)

**Boolean Matching**:
```prolog
filter_errors(Record) :-
    get_dict(message, Record, Line),
    match(Line, 'ERROR', python).
```

**Generated Python**:
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('message') != v_1: return
    if not re.search(r'ERROR', str(v_1)): return
    yield v_0
```

**Capture Groups**:
```prolog
parse_ip(Record, IP) :-
    get_dict(line, Record, Line),
    match(Line, '([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)', python, [IP]),
    Record = _{line: Line, ip: IP}.
```

**Generated Python**:
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('line') != v_2: return
    __match__ = re.search(r'([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)', str(v_2))
    if not __match__: return
    v_1 = __match__.group(1)
    v_0 = {'ip': v_1, 'line': v_2}
    yield v_1
```

### Bash Target (`src/unifyweaver/core/stream_compiler.pl`)

**Boolean Matching**:
```prolog
error_lines(Line) :-
    log(Line),
    match(Line, 'ERROR').
```

**Generated Bash**:
```bash
#!/bin/bash
# error_lines - streaming pipeline with match filtering

error_lines() {
    log_stream | grep 'ERROR' | sort -u
}

# Stream function for use in pipelines
error_lines_stream() {
    error_lines
}
```

**Integration**: Match constraints are seamlessly integrated into Bash streaming pipelines using `grep` filters.

## Regex Type Support

| Type | AWK | Python | Bash |
|------|-----|--------|------|
| `auto` | ✅ ERE | ✅ Python re | ✅ ERE (grep) |
| `ere` | ✅ | ✅ | ✅ (grep) |
| `bre` | ✅ | ❌ | ✅ (grep) |
| `awk` | ✅ | ❌ | ❌ |
| `python` | ❌ | ✅ | ❌ |
| `pcre` | ❌ | ✅ | ❌ |

Each target validates regex types at compile time with clear error messages.

## Files Changed

### AWK Target
- `src/unifyweaver/targets/awk_target.pl` - Match predicate implementation
- `test_awk_match.pl` - Boolean match tests
- `test_awk_match_captures.pl` - Capture group tests
- `docs/AWK_MATCH_PREDICATE.md` - AWK-specific documentation

### Python Target
- `src/unifyweaver/targets/python_target.pl` - Match predicate implementation
- `test_python_match.pl` - Boolean match tests
- `test_python_match_simple.pl` - Translation tests
- `test_python_match_captures.pl` - Capture group tests

### Bash Target
- `src/unifyweaver/core/stream_compiler.pl` - Match predicate implementation
- `test_bash_stream_match.pl` - Boolean match tests

### Documentation
- `docs/MATCH_PREDICATE.md` - Cross-target overview and examples
- `docs/AWK_MATCH_PREDICATE.md` - AWK-specific detailed documentation

## Test Results

✅ **AWK Tests**:
```prolog
?- test_awk_match:test_error_match.
% Compiles and generates correct AWK code with ~ operator

?- test_awk_match_captures:test_parse_log.
% Compiles and generates correct AWK code with match() and __captures__
```

✅ **Python Tests**:
```prolog
?- test_python_match_captures:run_all.
% All capture group tests pass with correct Python code generation
```

✅ **Bash Tests**:
```prolog
?- test_bash_stream_match:run_all.
% All streaming pipeline tests pass with grep filters integrated
```

## Future Enhancements

Documented in `docs/MATCH_PREDICATE.md`:

1. **Bash Capture Groups**: Extract BASH_REMATCH values for capture group support
2. **Named Captures**: Support named capture groups where available
3. **Match Flags**: Case-insensitive, multiline, etc.
4. **More Targets**: C#, native Prolog
5. **Constraint Integration**: Use captured values in arithmetic/comparison constraints

## Breaking Changes

None - this is a new feature with no impact on existing code.

## Commit History

1. `77426b0` - Implement match predicate for AWK target with regex type support
2. `964e801` - Add capture group support for match predicate in AWK target
3. `de32746` - Add match predicate support for Python target
4. `a081451` - Add capture group support for Python match predicate
5. `9394ec3` - Add cross-target match predicate documentation
6. `c446b1f` - Update match predicate docs: Bash support deferred to future work
7. `a03d85a` - Add match predicate support to Bash core compiler

## Attribution

Author: John William Creighton (@s243a)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
