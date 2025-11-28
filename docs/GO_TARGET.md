# Go Target - Cross-Platform Record Processing

The Go target compiles Prolog predicates to standalone Go programs for efficient, cross-platform record and field processing.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Compilation Modes](#compilation-modes)
6. [Examples](#examples)
7. [Options](#options)
8. [Limitations](#limitations)
9. [Future Enhancements](#future-enhancements)

---

## Overview

The Go target generates standalone Go executables from Prolog predicates, providing:
- **Cross-platform compatibility**: Runs on any platform with Go support
- **High performance**: Compiled binaries with low memory footprint
- **Stream processing**: Efficient stdin/stdout pipeline integration
- **Easy deployment**: Single binary with no runtime dependencies

---

## Features

### Current Features

- **Facts compilation**: Generates map-based lookups for fact predicates
- **Single rule compilation**: Compiles simple transformations with stdin I/O
- **Configurable delimiters**: Support for colon, tab, comma, and custom field delimiters
- **Automatic deduplication**: Built-in `map[string]bool` for unique results
- **Field reordering**: Correctly maps variables between head and body arguments

### Planned Features

- Match predicate support (regex filtering)
- Multiple rules compilation (OR patterns)
- Constraints and arithmetic
- JSON input/output
- Aggregations (count, sum, etc.)

---

## Quick Start

```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Define facts
parent(alice, bob).
parent(bob, charlie).

% Define rules
child(C, P) :- parent(P, C).

% Compile to Go
?- compile_predicate_to_go(parent/2, [], Code),
   write_go_program(Code, 'parent.go').

?- compile_predicate_to_go(child/2, [], Code),
   write_go_program(Code, 'child.go').
```

**Build and run:**

```bash
# Facts program
go build parent.go
./parent
# Output:
# alice:bob
# bob:charlie

# Rules program (reads from stdin)
go build child.go
echo "alice:bob" | ./child
# Output:
# bob:alice
```

---

## API Reference

### `compile_predicate_to_go/3`

Compile a Prolog predicate to Go code.

```prolog
compile_predicate_to_go(+Predicate, +Options, -GoCode)
```

**Parameters:**
- `Predicate`: Predicate indicator (e.g., `parent/2`)
- `Options`: List of compilation options
- `GoCode`: Generated Go code as atom

**Example:**
```prolog
?- compile_predicate_to_go(child/2, [field_delimiter(tab)], Code).
```

### `write_go_program/2`

Write Go code to a file.

```prolog
write_go_program(+GoCode, +FilePath)
```

**Parameters:**
- `GoCode`: Go code atom from `compile_predicate_to_go/3`
- `FilePath`: Output file path

**Example:**
```prolog
?- write_go_program(Code, 'output/child.go').
```

---

## Compilation Modes

### Facts Compilation

Generates a Go program with embedded facts in a `map[string]bool`.

**Prolog:**
```prolog
user(john, 25).
user(jane, 30).
user(bob, 28).
```

**Generated Go:**
```go
package main

import (
	"fmt"
)

func main() {
	facts := map[string]bool{
		"john:25": true,
		"jane:30": true,
		"bob:28": true,
	}

	for key := range facts {
		fmt.Println(key)
	}
}
```

### Single Rule Compilation

Generates a Go program that reads from stdin, processes records, and outputs results.

**Prolog:**
```prolog
child(C, P) :- parent(P, C).
```

**Generated Go:**
```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	// Read from stdin and process parent records
	scanner := bufio.NewScanner(os.Stdin)
	seen := make(map[string]bool)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ":")
		if len(parts) == 2 {
			field1 := parts[0]
			field2 := parts[1]
			result := field2 + ":" + field1
			if !seen[result] {
				seen[result] = true
				fmt.Println(result)
			}
		}
	}
}
```

---

## Examples

### Example 1: Field Reordering

Swap fields in user records:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/go_target').

user(john, 25).
user(jane, 30).

% Reverse to age:name
age_user(Age, Name) :- user(Name, Age).

test :-
    compile_predicate_to_go(age_user/2, [], Code),
    write_go_program(Code, 'age_user.go').
```

**Usage:**
```bash
go build age_user.go
echo -e "john:25\njane:30" | ./age_user
# Output:
# 25:john
# 30:jane
```

### Example 2: Different Delimiters

Use tab delimiters:

**Prolog:**
```prolog
test_tab :-
    compile_predicate_to_go(child/2, [field_delimiter(tab)], Code),
    write_go_program(Code, 'child_tab.go').
```

**Usage:**
```bash
go build child_tab.go
echo -e "alice\tbob" | ./child_tab
# Output:
# bob	alice
```

### Example 3: Extract Single Field

Get just user names:

**Prolog:**
```prolog
user_name(Name) :- user(Name, _).

test :-
    compile_predicate_to_go(user_name/1, [], Code),
    write_go_program(Code, 'user_name.go').
```

**Usage:**
```bash
go build user_name.go
echo -e "john:25\njane:30" | ./user_name
# Output:
# john
# jane
```

---

## Options

### `field_delimiter(Delimiter)`

Specify the field separator character.

**Values:**
- `colon` - Use `:` (default)
- `tab` - Use tab character
- `comma` - Use `,`
- `pipe` - Use `|`
- `Atom` - Any single character atom

**Example:**
```prolog
compile_predicate_to_go(child/2, [field_delimiter(comma)], Code)
```

### `record_delimiter(Delimiter)`

Specify the record separator (not yet fully implemented).

**Values:**
- `newline` - Use newline (default)
- `null` - Use null character

### `include_package(Boolean)`

Include `package main` and imports (default: `true`).

**Example:**
```prolog
compile_predicate_to_go(child/2, [include_package(false)], CodeOnly)
```

### `unique(Boolean)`

Deduplicate results using a seen map (default: `true`).

---

## Limitations

### Current Limitations

1. **Single rule only**: Only handles predicates with one rule in the body
   - ✅ Works: `child(C, P) :- parent(P, C).`
   - ❌ Not yet: Multiple rules (OR patterns)

2. **No constraints**: Arithmetic and comparison not yet supported
   - ❌ Not yet: `adult(Name) :- user(Name, Age), Age >= 18.`

3. **No match predicate**: Regex filtering not yet implemented
   - ❌ Not yet: `error_log(Line) :- log(Line), match(Line, 'ERROR').`

4. **No aggregations**: Count, sum, etc. not yet supported
   - ❌ Not yet: `total(Sum) :- aggregation(sum), values(X).`

5. **Simple variable mapping only**: Complex unification not supported
   - ✅ Works: Variable reordering
   - ❌ Not yet: Nested structures, partial instantiation

### Workarounds

**For multiple rules:**
Compile each rule separately and combine externally:
```bash
./rule1 < input.txt > intermediate.txt
./rule2 < intermediate.txt > output.txt
```

**For filtering:**
Use grep before/after processing:
```bash
grep "ERROR" input.txt | ./process > output.txt
```

---

## Future Enhancements

Planned additions (in priority order):

1. **Match predicate support** - Regex filtering with capture groups
2. **Multiple rules compilation** - OR patterns
3. **Constraints** - Arithmetic and comparison operators
4. **Record structures** - Nested field access
5. **Aggregations** - count, sum, avg, min, max
6. **JSON I/O** - Parse and generate JSON
7. **Custom functions** - User-defined Go helpers
8. **Optimizations** - Eliminate unnecessary allocations

---

## Performance Notes

- **Compilation speed**: Fast (< 1 second for typical predicates)
- **Runtime performance**: Comparable to hand-written Go
- **Memory footprint**: Low (map-based deduplication only)
- **Binary size**: ~2MB (typical Go binary overhead)
- **Startup time**: Instant (compiled binaries)

---

## Comparison with Other Targets

| Feature | Go | AWK | Bash | Python | C# |
|---------|-----|-----|------|--------|-----|
| Cross-platform | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Single binary | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| No runtime needed | ✅ | ❌ | ❌ | ❌ | ❌ |
| Low memory | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| Fast compilation | ✅ | ✅ | ✅ | ✅ | ❌ |
| Match predicate | ⏳ | ✅ | ✅ | ✅ | ❌ |

✅ = Full support | ⚠️ = Partial support | ❌ = Not supported | ⏳ = Planned

---

## Contributing

The Go target is under active development. Current priorities:

1. Match predicate implementation
2. Multiple rules compilation
3. Constraint support
4. Comprehensive test suite

See `test_go_target.pl` and `test_go_target_comprehensive.pl` for examples.

---

## See Also

- [AWK Target](AWK_TARGET_STATUS.md) - Similar record/field processing in AWK
- [Match Predicate](MATCH_PREDICATE.md) - Regex matching across targets
- [Python Target](../src/unifyweaver/targets/python_target.pl) - Alternative target

---

## Quick Reference

### Syntax

```prolog
% Compile predicate
compile_predicate_to_go(Predicate/Arity, Options, Code)

% Write to file
write_go_program(Code, 'output.go')

% Options
[field_delimiter(colon|tab|comma|pipe|Char)]
[record_delimiter(newline|null)]
[include_package(true|false)]
[unique(true|false)]
```

### Supported Patterns

```prolog
% Facts
fact(arg1, arg2, ...).

% Single rules (variable reordering)
head(X, Y) :- body(Y, X).

% Single rules (projection)
head(X) :- body(X, _).
```
