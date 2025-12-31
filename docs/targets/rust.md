# Rust Target

The Rust target (`target(rust)`) generates safe, high-performance Rust programs from Prolog predicates. It produces standalone binaries with strong type safety and memory efficiency.

## Overview

Rust programs compile to optimized native binaries with zero-cost abstractions, making them ideal for performance-critical applications, systems programming, and scenarios requiring memory safety guarantees.

```prolog
% Compile to Rust
?- compile_to_rust(my_predicate/3, [json_input(true)], RustCode).
```

## Features

### Core Capabilities

| Feature | Status | Description |
|---------|--------|-------------|
| Facts | ✅ | Direct fact compilation |
| Single Rules | ✅ | Body-to-code translation |
| Multiple Rules (OR) | ✅ | Union of clause results |
| Recursion | ✅ | Tail, linear, mutual recursion |
| Joins | ✅ | Inner, left, right, full outer |

### Aggregations

**Basic Aggregations:**
- `count` - Count records
- `sum` - Sum numeric values
- `avg` - Average (mean)
- `min` - Minimum value
- `max` - Maximum value

**Statistical Aggregations (NEW - 2025-12-25):**
- `stddev` - Sample standard deviation
- `median` - Median value
- `percentile(N)` - Nth percentile

```prolog
% Example: Calculate standard deviation
?- compile_to_rust(scores/1, [aggregation(stddev)], Code).
```

**Collection Aggregations (NEW - 2025-12-25):**
- `collect_list` - Aggregate into JSON array (preserves duplicates)
- `collect_set` - Aggregate into sorted JSON array (unique values)

```prolog
% Example: Collect unique values
?- compile_to_rust(categories/1, [aggregation(collect_set)], Code).
% Output: ["alpha","beta","gamma"]
```

### JSON Processing

- **Input:** Parse JSONL streams with serde_json
- **Output:** Generate JSON/JSONL output
- **Field Extraction:** Type-safe field access
- **Schema Validation:** Comprehensive schema enforcement

### Schema Validation (NEW - 2025-12-25)

Define schemas with field constraints and generate validation code:

```prolog
% Define schema with constraints
:- json_schema(user, [
    field(name, string, [required, min(1), max(100)]),
    field(age, integer, [required, min(0), max(150)]),
    field(email, string, [required, format(email)]),
    field(bio, string, [optional, max(500)]),
    field(address, object(address_schema), [optional])
]).

:- json_schema(address_schema, [
    field(street, string, [required]),
    field(city, string, [required]),
    field(zip, string, [format(date)])  % or other format
]).
```

**Supported Types:**
- `string`, `integer`, `float`, `boolean`, `any`
- `array`, `array(Type)` - Arrays with optional element type
- `object`, `object(SchemaName)` - Nested schema validation

**Field Options:**
- `required` - Field must be present
- `optional` - Field may be absent
- `min(N)` - Minimum value (numbers) or length (strings)
- `max(N)` - Maximum value (numbers) or length (strings)
- `format(email)` - Email format validation
- `format(date)` - Date format validation (YYYY-MM-DD)

**Generate Validator:**
```prolog
?- generate_rust_schema_validator(user, Code).
% Generates: fn validate_user(data: &serde_json::Value) -> bool { ... }
```

### Observability (NEW - 2025-12-25)

**Progress Reporting:**
```prolog
% Report every 1000 records
?- compile_to_rust(process/2, [progress(interval(1000))], Code).
```

**Error Logging:**
```prolog
% Log errors to JSON file
?- compile_to_rust(process/2, [error_file("errors.jsonl")], Code).
```
Output format:
```json
{"timestamp": "2025-12-25T10:30:00Z", "error": "parse error", "line": "invalid data"}
```

**Error Threshold:**
```prolog
% Exit after 100 errors
?- compile_to_rust(process/2, [error_threshold(count(100))], Code).
```

**Metrics Export:**
```prolog
% Write performance metrics
?- compile_to_rust(process/2, [metrics_file("metrics.json")], Code).
```
Output format:
```json
{"total_records": 50000, "error_records": 5, "duration_secs": 2.345, "records_per_sec": 21322.6}
```

**Combined Observability:**
```prolog
?- compile_to_rust(etl_pipeline/3, [
    progress(interval(10000)),
    error_file("errors.jsonl"),
    error_threshold(count(1000)),
    metrics_file("metrics.json")
], Code).
```

### XML Processing (NEW - 2025-12-25)

Stream and process XML files with `quick-xml`:

```prolog
% Define XML field extraction
book_info(Title, Author) :-
    xml_field(title, Title),
    xml_field(author, Author).

% Compile with XML input
?- compile_rust_xml_mode(book_info, 2, [
    xml_file("books.xml"),
    tags([book]),
    unique(true)
], Code).
```

**Input XML:**
```xml
<library>
  <book id="1">
    <title>The Rust Book</title>
    <author>Steve Klabnik</author>
  </book>
  <book id="2">
    <title>Programming Rust</title>
    <author>Jim Blandy</author>
  </book>
</library>
```

**Features:**
- Streaming parsing (memory-efficient for large files)
- Attribute extraction with `@` prefix (`@id`, `@class`, etc.)
- Text content extraction
- Tag filtering with `tags([tag1, tag2])`
- Stdin or file input
- Deduplication with `unique(true)`

**Cargo.toml dependency:**
```toml
[dependencies]
quick-xml = "0.31"
```

### Window Functions (NEW - 2025-12-25)

**Ranking Functions:**
- `row_number/2` - Sequential row numbering within partition
- `rank/2` - Rank with gaps for ties
- `dense_rank/2` - Rank without gaps

**Value Access Functions:**
- `lag/3` - Access previous row: `lag(SortField, ValueField, Result)`
- `lag/4` - With offset: `lag(SortField, ValueField, Offset, Result)`
- `lag/5` - With default: `lag(SortField, ValueField, Offset, Default, Result)`
- `lead/3` - Access next row: `lead(SortField, ValueField, Result)`
- `lead/4` - With offset: `lead(SortField, ValueField, Offset, Result)`
- `lead/5` - With default: `lead(SortField, ValueField, Offset, Default, Result)`
- `first_value/3` - First value in window partition
- `last_value/3` - Last value in window partition

**Example:**
```prolog
% Calculate day-over-day price change
daily_change(Date, Price, PrevPrice, Change) :-
    stock_price(Date, Price),
    lag(Date, Price, 1, 0, PrevPrice),
    Change is Price - PrevPrice.
```

### Database Integration (NEW - 2025-12-25)

Embedded database support using the `sled` pure-Rust database.

**Basic Usage:**
```prolog
% Read mode - query from database
?- compile_to_rust(user/3, [
    db_backend(sled),
    db_file('users.db'),
    db_key_field(name),
    db_mode(read)
], Code).

% Write mode - store JSONL to database
?- compile_to_rust(user/3, [
    db_backend(sled),
    db_file('users.db'),
    db_key_field(name),
    db_mode(write),
    json_input(true)
], Code).

% Analyze mode - collect statistics
?- compile_to_rust(user/3, [
    db_backend(sled),
    db_file('users.db'),
    db_mode(analyze)
], Code).
```

**Key Strategies:**
- `db_key_field(Field)` - Use single field as key
- `db_key_strategy(field(F))` - Same as above
- `db_key_strategy(composite([field(a), field(b)]))` - Composite key
- `db_key_strategy(hash(field(F)))` - Hash-based key
- `db_key_strategy(hash(field(F), sha256))` - SHA-256 hash
- `db_key_strategy(uuid)` - UUID v4 key

**Secondary Indexes:**
```prolog
% Declare index
:- index(user/3, email).

% Compile with index support
?- compile_to_rust(user/3, [
    db_backend(sled),
    db_key_field(name)
], Code).
```

**Predicate Pushdown:**
The compiler automatically optimizes queries:
- **Direct lookup** - O(log n) when filtering on key field
- **Prefix scan** - O(k log n) for composite key prefix match
- **Index scan** - O(k log n) using secondary indexes

**Cargo.toml dependency:**
```toml
[dependencies]
sled = "0.34"
serde_json = "1.0"
```

### Pipeline Support

- **Sequential Pipelines:** Chain processing stages
- **Enhanced Pipelines:** Complex multi-stage workflows
- **Parallel Mode:** Thread-based parallelism

### Project Generation

Generate complete Cargo projects:
```prolog
?- generate_rust_project(my_app, [my_pred/2, other_pred/3], "output_dir").
```

Creates:
```
output_dir/
├── Cargo.toml
├── src/
│   └── main.rs
└── README.md
```

## Compilation Options

```prolog
compile_to_rust(Pred/Arity, Options, RustCode)
```

| Option | Description | Default |
|--------|-------------|---------|
| `json_input(Bool)` | Parse JSONL input | `false` |
| `json_output(Bool)` | Generate JSON output | `false` |
| `aggregation(Op)` | Aggregation operation | `none` |
| `progress(interval(N))` | Progress every N records | - |
| `error_file(Path)` | Error log file path | - |
| `error_threshold(count(N))` | Exit after N errors | - |
| `metrics_file(Path)` | Metrics output file | - |
| `field_delimiter(Delim)` | Output field separator | `colon` |
| `unique(Bool)` | Deduplicate output | `true` |
| `include_main(Bool)` | Include main function | `true` |
| `pipeline_name(Name)` | Pipeline identifier | `pipeline` |
| `pipeline_mode(Mode)` | `sequential` or `parallel` | `sequential` |
| `db_backend(Backend)` | Database backend (sled) | - |
| `db_file(Path)` | Database file path | `data.db` |
| `db_tree(Name)` | Database tree name | predicate name |
| `db_key_field(Field)` | Primary key field | - |
| `db_key_strategy(S)` | Key generation strategy | `auto` |
| `db_mode(Mode)` | `read`, `write`, or `analyze` | `read` |

## Generated Code Structure

```rust
use std::io::{self, BufRead};
use serde_json::{self, Value};

fn main() {
    // Optional: observability setup
    let start_time = std::time::Instant::now();
    let mut record_count: u64 = 0;

    for line in io::stdin().lock().lines() {
        if let Ok(l) = line {
            // Parse and process
            record_count += 1;

            // Progress reporting
            if record_count % 1000 == 0 {
                eprintln!("Processed {} records", record_count);
            }
        }
    }

    // Optional: metrics output
    let elapsed = start_time.elapsed();
    eprintln!("Completed: {} records in {:.2}s", record_count, elapsed.as_secs_f64());
}
```

## Dependencies

Generated code uses standard Rust crates:
- `serde` / `serde_json` - JSON serialization
- `std::collections::HashSet` - Deduplication

Add to `Cargo.toml`:
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## Best Use Cases

1. **Performance-Critical ETL** - Zero-cost abstractions
2. **Memory-Constrained Environments** - Efficient memory usage
3. **Safety-Critical Applications** - Compile-time guarantees
4. **Standalone Tools** - Single binary distribution
5. **Systems Integration** - C ABI compatibility

## Comparison with Other Targets

| Aspect | Rust | Go | Python | Bash |
|--------|------|-----|--------|------|
| Memory Safety | Compile-time | Runtime | Runtime | N/A |
| Performance | Excellent | Very Good | Good | Variable |
| Binary Size | Small | Small | N/A | N/A |
| Compile Time | Slow | Fast | N/A | N/A |
| Statistical Aggs | ✅ | ✅ | ✅ | ❌ |
| Observability | ✅ | ✅ | ✅ | ❌ |
| Window Functions | ✅ | ✅ | ✅ | ❌ |
| Database | sled | BoltDB | SQLite | ❌ |

## Limitations

- No current known limitations for core features

## See Also

- [Target Overview](overview.md)
- [Target Comparison](comparison.md)
- [Rust Runtime](../../src/runtime/rust_runtime/)
