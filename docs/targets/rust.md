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
- **Schema Validation:** Optional schema enforcement

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
| Window Functions | ❌ | ✅ | ✅ | ❌ |
| Database | ❌ | BoltDB | SQLite | ❌ |

## Limitations

- **Window Functions:** Not yet implemented (use Go or Python)
- **Database Integration:** Not yet implemented (planned: sled/rocksdb)
- **XML Processing:** Not yet implemented

## See Also

- [Target Overview](overview.md)
- [Target Comparison](comparison.md)
- [Rust Runtime](../../src/runtime/rust_runtime/)
