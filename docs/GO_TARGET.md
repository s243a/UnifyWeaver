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
- **Match predicate support**: Regex filtering with boolean matching
- **Match predicate capture groups**: Extract substrings using regex capture groups
- **Match predicates with body predicates**: Combine regex captures with source predicates
- **Multiple rules compilation**: OR patterns with combined regex matching
- **Multiple rules with different bodies**: Sequential rule matching for different source arities
- **Configurable delimiters**: Support for colon, tab, comma, and custom field delimiters
- **Automatic deduplication**: Built-in `map[string]bool` for unique results
- **Field reordering**: Correctly maps variables between head and body arguments
- **Selective field assignment**: Only assigns fields actually used in output
- **Constraints and arithmetic**: Support for >, <, >=, =<, ==, !=, and is/2
- **Aggregations**: count, sum, avg, max, min
- **Advanced Aggregations**:
    - **Statistical**: `stddev`, `median`, `percentile`
    - **Array**: `collect_list` (duplicates), `collect_set` (unique)
    - **Window Functions**: `row_number`, `rank`, `dense_rank`
- **JSON Input/Output**: Full JSONL stream processing with nested field extraction
- **Schema Validation**: Compile-time schema-based type checking and validation
- **XML Input**: Streaming XML parsing and flattening
- **Database Support**: Embedded `bbolt` database support with:
    - **Key Strategies**: single field, composite, hash
    - **Secondary Indexes**: `:- index(predicate/arity, field).` for optimized lookups
    - **Query Optimization**: Automatic direct lookup and prefix scan selection
- **Stream Processing Observability**:
    - **Error Aggregation**: `error_file(Path)` to capture failures
    - **Progress Reporting**: `progress(interval(N))` for throughput monitoring
    - **Error Thresholds**: `error_threshold(count(N))` to fail fast
    - **Metrics Export**: `metrics_file(Path)` for performance statistics
- **Smart imports**: Only includes necessary Go packages when needed

### Planned Features

- **Cost-Based Optimization**: Using statistics for join ordering
- **Custom Go functions**: User-defined Go helpers

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

### `compile_facts_to_go/3`

Export Prolog facts as a standalone Go program with struct-based data.

```prolog
compile_facts_to_go(+Pred, +Arity, -GoCode)
```

**Parameters:**
- `Pred`: Predicate name (atom)
- `Arity`: Number of arguments
- `GoCode`: Generated Go code as string

**Features:**
- Generates `struct` with typed `ArgN` fields
- `GetAllPRED() []PRED` - Returns all facts
- `StreamPRED(fn func(PRED))` - Iterator with callback
- `ContainsPRED(target PRED) bool` - Membership check

**Example:**
```prolog
?- ['examples/family_tree'].
?- go_target:compile_facts_to_go(parent, 2, Code).
```

**Generated Go:**
```go
type PARENT struct {
    Arg1 string
    Arg2 string
}

func GetAllPARENT() []PARENT { ... }
func StreamPARENT(fn func(PARENT)) { ... }
func ContainsPARENT(target PARENT) bool { ... }
```

### `compile_tail_recursion_go/3`

Compile tail recursive predicates to O(1) space for loops.

```prolog
compile_tail_recursion_go(+Pred/Arity, +Options, -GoCode)
```

### `compile_linear_recursion_go/3`

Compile linear recursive predicates with map-based memoization.

```prolog
compile_linear_recursion_go(+Pred/Arity, +Options, -GoCode)
```

### `compile_mutual_recursion_go/3`

Compile mutually recursive predicates (is_even/is_odd) with shared memo.

```prolog
compile_mutual_recursion_go(+Predicates, +Options, -GoCode)
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

### XML Input Mode

Compiles a predicate to read and flatten XML data.

**Prolog:**
```prolog
compile_predicate_to_go(item/2, [
    xml_input(true),
    xml_file('data.xml'), % or stdin
    tags(['item'])
], Code).
```

**Features:**
- Streams XML using `encoding/xml`
- Flattens elements into a map (Attributes -> `@attr`, Text -> `text`)
- Supports `bbolt` database storage via `db_backend(bbolt)`
- Compatible with existing field extraction (`json_get`)

**Generated Go:**
```go
// ... imports encoding/xml ...
decoder := xml.NewDecoder(f)
for {
    t, _ := decoder.Token()
    // ... match start element ...
    var node XmlNode
    decoder.DecodeElement(&node, &se)
    data := FlattenXML(node)
    // ... process data map ...
}
```
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

### Example 4: Match Predicate - Filter Logs

Use regex patterns to filter records:

**Prolog:**
```prolog
log('ERROR: timeout occurred').
log('WARNING: slow response').
log('INFO: operation successful').
log('ERROR: connection failed').

% Filter error logs
error_log(Line) :-
    log(Line),
    match(Line, 'ERROR').

% Filter specific errors with pattern
timeout_error(Line) :-
    log(Line),
    match(Line, 'ERROR.*timeout').

test :-
    compile_predicate_to_go(error_log/1, [], Code),
    write_go_program(Code, 'error_log.go').
```

**Usage:**
```bash
go build error_log.go
echo -e "ERROR: timeout occurred\nWARNING: slow response\nERROR: connection failed" | ./error_log
# Output:
# ERROR: timeout occurred
# ERROR: connection failed
```

### Example 5: Multiple Rules - Combined Filters

Compile multiple rules with different match patterns into a single OR regex:

**Prolog:**
```prolog
log('ERROR: connection timeout').
log('WARNING: slow response').
log('CRITICAL: database down').

% Multiple rules for different alert levels
alert(Line) :- log(Line), match(Line, 'ERROR').
alert(Line) :- log(Line), match(Line, 'WARNING').
alert(Line) :- log(Line), match(Line, 'CRITICAL').

test :-
    compile_predicate_to_go(alert/1, [], Code),
    write_go_program(Code, 'alert.go').
```

**Generated Regex:** `ERROR|WARNING|CRITICAL`

**Usage:**
```bash
go build alert.go
echo -e "ERROR: timeout\nWARNING: slow\nINFO: ok\nCRITICAL: down" | ./alert
# Output:
# ERROR: timeout
# WARNING: slow
# CRITICAL: down
```

### Example 6: Capture Groups - Extract Log Components

Extract specific parts of log messages using regex capture groups:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Extract timestamp and log level from log lines
parse_log(Line, Time, Level) :-
    match(Line, '([0-9-]+ [0-9:]+) ([A-Z]+)', auto, [Time, Level]).

% Extract date, time, and level separately
parse_detailed(Line, Date, Time, Level) :-
    match(Line, '([0-9-]+) ([0-9:]+) ([A-Z]+)', auto, [Date, Time, Level]).

test :-
    compile_predicate_to_go(parse_log/3, [], Code),
    write_go_program(Code, 'parse_log.go').
```

**Usage:**
```bash
go build parse_log.go
cat logs.txt | ./parse_log
# Input: 2025-01-15 10:30:45 ERROR timeout occurred
# Output: 2025-01-15 10:30:45 ERROR timeout occurred:2025-01-15 10:30:45:ERROR
```

### Example 7: Constraints - Filter by Numeric Conditions

Use constraints to filter records based on numeric comparisons:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/go_target').

person(alice, 25).
person(bob, 17).
person(charlie, 45).

% Filter adults (age > 18)
adult(Name, Age) :- person(Name, Age), Age > 18.

% Filter working age (18 <= age <= 65)
working_age(Name, Age) :- person(Name, Age), Age >= 18, Age =< 65.

test :-
    compile_predicate_to_go(adult/2, [], Code),
    write_go_program(Code, 'adult.go').
```

**Usage:**
```bash
go build adult.go
echo -e "alice:25\nbob:17\ncharlie:45" | ./adult
# Output:
# alice:25
# charlie:45
```

### Example 8: Aggregations - Sum, Count, Average

Perform aggregation operations on numeric fields:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/go_target').

value(10).
value(20).
value(30).
value(40).
value(50).

% Compute sum of all values
total(Sum) :- aggregation(sum), value(Sum).

% Count number of values
num_values(Count) :- aggregation(count), value(Count).

% Compute average
average(Avg) :- aggregation(avg), value(Avg).

test :-
    compile_predicate_to_go(total/1, [], SumCode),
    write_go_program(SumCode, 'sum.go'),
    compile_predicate_to_go(average/1, [], AvgCode),
    write_go_program(AvgCode, 'avg.go').
```

**Usage:**
```bash
go build sum.go && go build avg.go
echo -e "10\n20\n30\n40\n50" | ./sum
# Output: 150

echo -e "10\n20\n30\n40\n50" | ./avg
# Output: 30
```

### Example 9: Match + Body Predicates - Parse Log Entries

Combine regex pattern matching with source predicates to extract structured data:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Source predicate with log entries
log_entry(alice, '2025-01-15 ERROR: timeout occurred').
log_entry(bob, '2025-01-15 INFO: operation successful').
log_entry(charlie, '2025-01-15 WARNING: slow response').

% Extract name, level, and message using match + body
parsed(Name, Level, Message) :-
    log_entry(Name, Line),
    match(Line, '([A-Z]+): (.+)', auto, [Level, Message]).

test :-
    compile_predicate_to_go(parsed/3, [field_delimiter(tab)], Code),
    write_go_program(Code, 'parsed.go').
```

**Usage:**
```bash
go build parsed.go
cat log_entries.txt | ./parsed
# Input: alice	2025-01-15 ERROR: timeout occurred
# Output: alice	ERROR	timeout occurred
```

### Example 10: Multiple Rules with Different Bodies - Unified Person View

Compile multiple rules with different source predicates into a single program:

**Prolog:**
```prolog
:- use_module('src/unifyweaver/targets/go_target').

% Different source predicates with different arities
user(alice).
employee(bob, engineering).
contractor(charlie, design, hourly).

% Unify them into a single person/1 predicate
person(Name) :- user(Name).
person(Name) :- employee(Name, _).
person(Name) :- contractor(Name, _, _).

test :-
    compile_predicate_to_go(person/1, [], Code),
    write_go_program(Code, 'person.go').
```

**Generated Strategy:**
The compiler generates sequential if-continue blocks that try each rule pattern based on field count:
- If 1 field → try user/1
- If 2 fields → try employee/2
- If 3 fields → try contractor/3

**Usage:**
```bash
go build person.go
cat input.txt | ./person
# Input:
# alice
# bob:engineering
# charlie:design:hourly
# Output:
# alice
# bob
# charlie
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

1. **Multiple rules**: Full support for OR patterns and different body predicates
   - ✅ Works: Multiple rules with different match patterns (combined into OR regex)
   - ✅ Works: Multiple rules with different body predicates (sequential matching)

2. **Match predicate**: Fully supported with capture groups and body predicates
   - ✅ Works: `error_log(Line) :- log(Line), match(Line, 'ERROR').`
   - ✅ Works: `timeout(Line) :- log(Line), match(Line, 'ERROR.*timeout').`
   - ✅ Works: `parse(Line, Time, Level) :- match(Line, '([0-9:]+) ([A-Z]+)', auto, [Time, Level]).`
   - ✅ Works: `parsed(Name, Level, Msg) :- log_entry(Name, Line), match(Line, '([A-Z]+): (.+)', auto, [Level, Msg]).`

3. **Simple variable mapping only**: Complex unification not supported
   - ✅ Works: Variable reordering
   - ✅ Works: Constraints with type conversion (strconv.Atoi)
   - ✅ Works: Aggregations on numeric fields
   - ❌ Not yet: Nested structures, partial instantiation

### Workarounds

**For multiple rules:**
Compile each rule separately and combine externally:
```bash
./rule1 < input.txt > intermediate.txt
./rule2 < intermediate.txt > output.txt
```

**For capture groups:**
Use match/2 or match/3 for boolean filtering, then process matches with additional rules or external tools:
```bash
./filter_errors < input.txt | cut -d: -f2 > error_codes.txt
```

---

## Future Enhancements

Completed features (moved to Current Features):
- ✅ Match predicate capture groups
- ✅ Constraints (arithmetic and comparison)
- ✅ Aggregations (count, sum, avg, min, max)
- ✅ Advanced Aggregations (Stats, Arrays, Window)
- ✅ Match predicates with body predicates
- ✅ Multiple rules with different bodies
- ✅ JSON Input/Output (JSONL)
- ✅ XML Input (Streaming/Flattening)
- ✅ Database Support (BoltDB)
- ✅ Secondary Indexes
- ✅ Stream Processing Observability (Error file, Progress, Thresholds, Metrics)


Planned additions (in priority order):

1. **Cost-Based Optimization** - Using table statistics for join ordering
2. **Custom functions** - User-defined Go helpers
3. **Optimizations** - Eliminate unnecessary allocations
4. **Semantic Runtime** - Vector embeddings and search (ONNX integration)

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
| Match predicate | ⚠️ | ✅ | ✅ | ✅ | ❌ |

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

- [Go Embedder Backends](GO_EMBEDDER_BACKENDS.md) - Semantic search with Pure Go, Candle, ORT, and XLA backends
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

% Match predicates (boolean filtering)
filtered(X) :- source(X), match(X, 'pattern').

% Match predicates (with options)
filtered(X) :- source(X), match(X, 'pattern', auto).

% Match predicates (with capture groups)
parsed(Line, Field1, Field2) :- match(Line, '(\\w+) (\\d+)', auto, [Field1, Field2]).

% Constraints (numeric comparisons)
adult(Name, Age) :- person(Name, Age), Age > 18.
working_age(Name, Age) :- person(Name, Age), Age >= 18, Age =< 65.

% Aggregations
total(Sum) :- aggregation(sum), value(Sum).
count_records(N) :- aggregation(count), value(N).
average_value(Avg) :- aggregation(avg), value(Avg).
maximum(Max) :- aggregation(max), value(Max).
minimum(Min) :- aggregation(min), value(Min).

% Multiple rules (OR pattern with match)
result(X) :- source(X), match(X, 'pattern1').
result(X) :- source(X), match(X, 'pattern2').

% Match + body predicates (capture with source)
parsed(Name, Level, Msg) :- log_entry(Name, Line), match(Line, '([A-Z]+): (.+)', auto, [Level, Msg]).

% Multiple rules with different bodies (sequential matching)
person(Name) :- user(Name).
person(Name) :- employee(Name, _).
person(Name) :- contractor(Name, _, _).
```
