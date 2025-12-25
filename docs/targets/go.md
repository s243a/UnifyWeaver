# Go Target

The Go target (`target(go)`) generates standalone Go programs from Prolog predicates. It is the most feature-complete procedural target, offering embedded database support, advanced aggregations, window functions, and comprehensive observability.

## Overview

Go programs compile to single static binaries with no runtime dependencies, making them ideal for deployment in containerized environments, edge computing, and systems where minimal footprint matters.

```prolog
% Compile to Go
?- compile_to_go(my_predicate/3, [json_input(true)], GoCode).
```

## Features

### Core Capabilities

| Feature | Status | Description |
|---------|--------|-------------|
| Facts | ✅ | Direct fact compilation |
| Single Rules | ✅ | Body-to-code translation |
| Multiple Rules (OR) | ✅ | Union of clause results |
| Recursion | ✅ | Tail, linear, mutual, transitive closure |
| Joins | ✅ | Inner, left, right, full outer, N-way |

### Aggregations

**Basic Aggregations:**
- `count/1`, `sum/2`, `avg/2`, `min/2`, `max/2`

**Statistical Aggregations:**
- `stddev/2` - Sample standard deviation
- `median/2` - Median value
- `percentile/3` - Nth percentile (e.g., `percentile(Score, 95, P95)`)

**Collection Aggregations:**
- `collect_list/2` - Aggregate into list (preserves duplicates)
- `collect_set/2` - Aggregate into set (unique values)

**Grouped Aggregations:**
- `group_by/3` with single or multiple operations
- `HAVING` clause support
- Nested grouping

### Window Functions

**Ranking Functions:**
- `row_number/2` - Sequential row numbering within partition
- `rank/2` - Rank with gaps for ties
- `dense_rank/2` - Rank without gaps

**Value Access Functions (NEW - 2025-12-25):**
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
% Calculate day-over-day change
daily_change(Date, Price, PrevPrice, Change) :-
    stock_price(Date, Price),
    lag(Date, Price, 1, 0, PrevPrice),
    Change is Price - PrevPrice.
```

### JSON Processing

- **Input:** Parse JSONL streams with field extraction
- **Output:** Generate JSON/JSONL output
- **Schema Validation:** Type checking, min/max, regex format
- **Nested Access:** JSON path expressions for deep field access

### Database Integration (BoltDB)

- **Embedded Database:** Pure Go, ACID transactions
- **Secondary Indexes:** Manual index hints via `:- index(pred/arity, field)`
- **Predicate Pushdown:** Equality filters pushed to storage layer
- **Cost-Based Optimization:** Statistics-driven query planning

```prolog
?- compile_to_go(query/3, [db_backend(bbolt), db_file("data.db")], Code).
```

### Observability

- **Progress Reporting:** `progress(interval(N))` - Report every N records
- **Error Logging:** `error_file(Path)` - Log errors to JSON file
- **Error Threshold:** `error_threshold(count(N))` - Exit after N errors
- **Metrics Export:** `metrics_file(Path)` - Performance metrics

### Input Sources

| Source | Status |
|--------|--------|
| JSONL Stream | ✅ |
| CSV/Delimited | ✅ |
| XML | ✅ |
| BoltDB | ✅ |

## Compilation Options

```prolog
compile_to_go(Pred/Arity, Options, GoCode)
```

| Option | Description | Default |
|--------|-------------|---------|
| `json_input(Bool)` | Parse JSONL input | `false` |
| `json_output(Bool)` | Generate JSON output | `false` |
| `json_schema(Name)` | Schema for validation | `none` |
| `db_backend(bbolt)` | Use BoltDB storage | - |
| `db_file(Path)` | Database file path | - |
| `aggregation(Op)` | Aggregation operation | `none` |
| `progress(interval(N))` | Progress reporting | - |
| `error_file(Path)` | Error log file | - |
| `metrics_file(Path)` | Metrics output file | - |
| `include_main(Bool)` | Include main function | `true` |

## Generated Code Structure

```go
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
    // ... additional imports based on features
)

func main() {
    // Input processing
    scanner := bufio.NewScanner(os.Stdin)
    for scanner.Scan() {
        // Parse and process each record
        // Apply filters, joins, aggregations
        // Output results
    }
}
```

## Best Use Cases

1. **High-Performance Data Processing** - Single binary, minimal overhead
2. **Embedded Database Queries** - BoltDB integration for persistent storage
3. **Analytics Pipelines** - Statistical aggregations, window functions
4. **Microservices** - Container-friendly deployment
5. **Edge Computing** - No runtime dependencies

## Comparison with Other Targets

| Aspect | Go | Rust | Python | Bash |
|--------|-----|------|--------|------|
| Binary Size | Small | Small | N/A (interpreted) | N/A |
| Startup Time | Fast | Fast | Slow | Fast |
| Memory Efficiency | High | Very High | Medium | Medium |
| Database Support | BoltDB | - | SQLite | - |
| Window Functions | ✅ Full | ❌ | ✅ | ❌ |
| Statistical Aggs | ✅ | ✅ | ✅ | ❌ |

## See Also

- [Target Overview](overview.md)
- [Target Comparison](comparison.md)
- [Go Semantic Runtime](../proposals/go_semantic_runtime.md)
