# Enhanced Pipeline Chaining

UnifyWeaver supports **enhanced pipeline chaining** across all major targets, enabling complex data flow patterns beyond simple linear pipelines.

## Overview

Enhanced pipeline chaining adds the following stage types to the standard predicate stages:

| Stage Type | Description |
|------------|-------------|
| `fan_out(Stages)` | Broadcast each record to multiple stages (sequential execution) |
| `parallel(Stages)` | Execute stages concurrently using target-native parallelism |
| `parallel(Stages, Options)` | Execute stages concurrently with options (e.g., `ordered(true)`) |
| `merge` | Combine results from fan_out or parallel stages |
| `route_by(Pred, Routes)` | Route records to different stages based on a predicate condition |
| `filter_by(Pred)` | Filter records that satisfy a predicate |
| `batch(N)` | Collect N records into batches for bulk processing |
| `unbatch` | Flatten batches back to individual records |
| `unique(Field)` | Keep first record per unique field value (deduplicate) |
| `first(Field)` | Alias for unique - keep first occurrence |
| `last(Field)` | Keep last record per unique field value |
| `group_by(Field, Agg)` | Group by field and apply aggregations |
| `reduce(Pred, Init)` | Sequential fold across all records |
| `scan(Pred, Init)` | Like reduce but emits intermediate values |
| `order_by(Field)` | Sort records by field (ascending) |
| `order_by(Field, Dir)` | Sort records by field with direction (asc/desc) |
| `order_by(FieldSpecs)` | Sort by multiple fields with directions |
| `sort_by(ComparePred)` | Sort using custom comparator function |
| `Pred/Arity` | Standard predicate stage (unchanged) |

### Fan-out vs Parallel

The key difference between `fan_out` and `parallel`:

- **`fan_out(Stages)`**: Processes stages **sequentially**, one after another. Safe for any workload.
- **`parallel(Stages)`**: Processes stages **concurrently** using target-native mechanisms:
  - **Python**: `ThreadPoolExecutor`
  - **Go**: Goroutines with `sync.WaitGroup`
  - **C#**: `Task.WhenAll`
  - **Rust**: `std::thread` by default, or rayon with `parallel_mode(rayon)` option
  - **PowerShell**: Runspace pools
  - **Bash**: Background processes with `wait`
  - **IronPython**: .NET `Task.Factory.StartNew` with `ConcurrentBag<T>`
  - **AWK**: Sequential by default, or GNU Parallel with `parallel_mode(gnu_parallel)` option

## Supported Targets

Enhanced chaining is available across all major targets:

| Target | Entry Point | PR |
|--------|-------------|-----|
| **Python (CPython)** | `compile_enhanced_pipeline/3` | #296 |
| **Go** | `compile_go_enhanced_pipeline/3` | #297 |
| **C#** | `compile_csharp_enhanced_pipeline/3` | #297 |
| **Rust** | `compile_rust_enhanced_pipeline/3` | #297 |
| **PowerShell** | `compile_powershell_enhanced_pipeline/3` | #297 |
| **AWK** | `compile_awk_enhanced_pipeline/3` | #298 |
| **Bash** | `compile_bash_enhanced_pipeline/3` | #299 |
| **IronPython** | `compile_ironpython_enhanced_pipeline/3` | #300 |

## Basic Usage

### Python Example

```prolog
:- use_module(src/unifyweaver/targets/python_target).

% Simple enhanced pipeline
compile_enhanced_pipeline([
    extract/1,
    filter_by(is_active),
    fan_out([validate/1, enrich/1]),
    merge,
    output/1
], [pipeline_name(my_pipeline)], PythonCode).
```

### Go Example

```prolog
:- use_module(src/unifyweaver/targets/go_target).

compile_go_enhanced_pipeline([
    parse/1,
    filter_by(is_valid),
    fan_out([transform/1, audit/1]),
    merge,
    route_by(has_error, [(true, error_handler/1), (false, success/1)]),
    output/1
], [pipeline_name(dataPipeline), output_format(jsonl)], GoCode).
```

### Bash Example

```prolog
:- use_module(src/unifyweaver/targets/bash_target).

compile_bash_enhanced_pipeline([
    extract/1,
    filter_by(is_active),
    fan_out([validate/1, enrich/1, audit/1]),
    merge,
    route_by(has_error, [(true, error_log/1), (false, transform/1)]),
    output/1
], [pipeline_name(complex_pipe), record_format(jsonl)], BashCode).
```

## Stage Types in Detail

### Fan-Out (`fan_out/1`)

Broadcasts each input record to multiple stages **sequentially** and collects all results.

```prolog
fan_out([validate/1, enrich/1, audit/1])
```

**Data Flow:**
```
Input Record ─► validate/1 ─► Result 1
             ─► enrich/1   ─► Result 2  (after validate completes)
             ─► audit/1    ─► Result 3  (after enrich completes)
```

All results are collected and passed to the next stage.

**Generated Code (Python):**
```python
def fan_out_records(record, stages):
    results = []
    for stage in stages:  # Sequential iteration
        for result in stage(iter([record])):
            results.append(result)
    return results
```

### Parallel (`parallel/1`)

Executes stages **concurrently** using target-native parallelism mechanisms.

```prolog
parallel([validate/1, enrich/1, audit/1])
```

**Data Flow:**
```
Input Record ─┬─► validate/1 ─► Result 1  ─┐
              ├─► enrich/1   ─► Result 2  ─┼─► Collected
              └─► audit/1    ─► Result 3  ─┘
              (all stages execute simultaneously)
```

**Generated Code (Python):**
```python
def parallel_records(record, stages):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_stage(stage):
        return list(stage(iter([record])))

    results = []
    with ThreadPoolExecutor(max_workers=len(stages)) as executor:
        futures = {executor.submit(run_stage, stage): i for i, stage in enumerate(stages)}
        for future in as_completed(futures):
            results.extend(future.result())
    return results
```

**Generated Code (Go):**
```go
func parallelRecords(record Record, stages []func([]Record) []Record) []Record {
    var wg sync.WaitGroup
    var mu sync.Mutex
    var results []Record

    for _, stage := range stages {
        wg.Add(1)
        go func(s func([]Record) []Record) {
            defer wg.Done()
            stageResults := s([]Record{record})
            mu.Lock()
            results = append(results, stageResults...)
            mu.Unlock()
        }(stage)
    }

    wg.Wait()
    return results
}
```

#### Ordered Parallel (`parallel/2` with `ordered(true)`)

By default, `parallel/1` returns results in **completion order** (fastest stage first). Use `ordered(true)` to preserve **stage definition order**.

```prolog
% Unordered (default) - Results in completion order
parallel([slow_stage/1, fast_stage/1])

% Ordered - Results in stage definition order
parallel([slow_stage/1, fast_stage/1], [ordered(true)])
```

**Data Flow (ordered):**
```
Input Record ─┬─► slow_stage/1 ─► Result 1  ─┐
              └─► fast_stage/1 ─► Result 2  ─┘
              Completion: [fast, slow]
              Output: [slow_result, fast_result]  ← ordered by stage position
```

**When to use `ordered(true)`:**
- When downstream processing depends on result order
- When stage outputs must maintain predictable ordering
- When combining results with zip-like operations

**Generated Code (Python - ordered):**
```python
def parallel_records_ordered(record, stages):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_stage(stage):
        return list(stage(iter([record])))

    indexed_results = [None] * len(stages)
    with ThreadPoolExecutor(max_workers=len(stages)) as executor:
        futures = {executor.submit(run_stage, stage): i for i, stage in enumerate(stages)}
        for future in as_completed(futures):
            idx = futures[future]
            indexed_results[idx] = future.result()

    # Flatten results in order
    results = []
    for stage_results in indexed_results:
        if stage_results:
            results.extend(stage_results)
    return results
```

**Generated Code (Go - ordered):**
```go
func parallelRecordsOrdered(record Record, stages []func([]Record) []Record) []Record {
    var wg sync.WaitGroup
    indexedResults := make([][]Record, len(stages))

    for i, stage := range stages {
        wg.Add(1)
        go func(idx int, s func([]Record) []Record) {
            defer wg.Done()
            indexedResults[idx] = s([]Record{record})
        }(i, stage)
    }

    wg.Wait()

    // Flatten results in order
    var results []Record
    for _, stageResults := range indexedResults {
        results = append(results, stageResults...)
    }
    return results
}
```

### Merge (`merge`)

Combines results from fan_out or parallel stages into a single stream.

```prolog
fan_out([a/1, b/1, c/1]),
merge,
output/1
```

Typically used immediately after `fan_out` to flatten the results before continuing the pipeline.

### Conditional Routing (`route_by/2`)

Routes records to different stages based on a predicate condition.

```prolog
route_by(has_error, [
    (true, error_handler/1),
    (false, success_handler/1)
])
```

**Data Flow:**
```
Input Record ─► has_error(Record) ─┬─ true  ─► error_handler/1
                                   └─ false ─► success_handler/1
```

**Generated Code (Go):**
```go
func routeRecord(record Record, conditionFn func(Record) interface{},
                 routeMap map[interface{}]func([]Record) []Record,
                 defaultFn func([]Record) []Record) []Record {
    condition := conditionFn(record)
    if stageFn, ok := routeMap[condition]; ok {
        return stageFn([]Record{record})
    }
    if defaultFn != nil {
        return defaultFn([]Record{record})
    }
    return []Record{record}
}
```

### Filter (`filter_by/1`)

Filters records, only passing those that satisfy the predicate.

```prolog
filter_by(is_active)
```

Records where `is_active(Record)` returns false/fails are dropped from the stream.

**Generated Code (Bash):**
```bash
filter_record() {
    local record="$1"
    local predicate="$2"

    if "$predicate" "$record" >/dev/null 2>&1; then
        echo "$record"
    else
        echo ""
    fi
}
```

### Batch (`batch/1`)

Collects N records into batches for bulk processing. Useful for:
- Bulk database inserts
- Batch API calls with rate limits
- Efficient I/O operations

```prolog
batch(100)
```

**Data Flow:**
```
Record 1 ─┐
Record 2 ─┤
   ...    ├─► [Batch of 100] ─► Next Stage
Record 100─┘
Record 101─┐
   ...    ├─► [Batch of 100] ─► Next Stage
Record 200─┘
```

**Generated Code (Python):**
```python
def batch_records(stream, batch_size):
    '''
    Batch: Collect records into batches of specified size.
    Yields each batch as a list. Final batch may be smaller.
    '''
    batch = []
    for record in stream:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:  # Flush remaining records
        yield batch
```

### Unbatch (`unbatch`)

Flattens batches back to individual records. Typically follows batch processing.

```prolog
unbatch
```

**Data Flow:**
```
[Batch 1] ─► Record 1, Record 2, ... Record 100
[Batch 2] ─► Record 101, Record 102, ... Record 200
```

**Generated Code (Python):**
```python
def unbatch_records(stream):
    '''
    Unbatch: Flatten batches back to individual records.
    '''
    for batch in stream:
        if isinstance(batch, list):
            for record in batch:
                yield record
        else:
            yield batch
```

## Aggregation Stages

Pipeline-level aggregation stages for deduplication, grouping, and sequential processing.

### Unique (`unique/1`) / First (`first/1`)

Keep only the first record for each unique field value. Deduplicates the stream.

```prolog
unique(user_id)
% or equivalently:
first(user_id)
```

**Data Flow:**
```
{id: 1, user: "A"} ─┐
{id: 2, user: "B"} ─┼─► {id: 1, user: "A"}, {id: 2, user: "B"}
{id: 3, user: "A"} ─┘   (second "A" is filtered out)
```

### Last (`last/1`)

Keep only the last record for each unique field value.

```prolog
last(user_id)
```

### Group By (`group_by/2`)

Group records by a field and apply aggregations.

```prolog
% Single aggregation
group_by(category, count)

% Multiple aggregations
group_by(category, [count, sum(amount), avg(price)])
```

**Available Aggregations:**
| Aggregation | Description |
|-------------|-------------|
| `count` | Count records in group |
| `sum(Field)` | Sum numeric field |
| `avg(Field)` | Average numeric field |
| `min(Field)` | Minimum value |
| `max(Field)` | Maximum value |
| `first(Field)` | First value in group |
| `last(Field)` | Last value in group |
| `collect(Field)` | Collect all values into list |

**Example:**
```prolog
compile_enhanced_pipeline([
    parse/1,
    group_by(region, [count, sum(sales), avg(sales)]),
    output/1
], [pipeline_name(sales_by_region)], Code).
```

**Generated Output:**
```json
{"region": "North", "count": 150, "sum": 45000.0, "avg": 300.0}
{"region": "South", "count": 200, "sum": 60000.0, "avg": 300.0}
```

### Reduce (`reduce/2`)

Apply a reducer function sequentially across all records. Unlike `group_by`, `reduce` sees all data in order and maintains running state.

```prolog
% reduce(ReducerPred, InitialValue)
reduce(running_total, 0)
```

**Use Cases:**
- Running totals/cumulative sums
- State machines
- Sequential aggregations where order matters

### Scan (`scan/2`)

Like `reduce` but emits intermediate results after each record.

```prolog
% scan(ReducerPred, InitialValue)
scan(running_sum, 0)
```

**Data Flow:**
```
{amount: 10} ─► {result: 10}
{amount: 20} ─► {result: 30}
{amount: 5}  ─► {result: 35}
```

### Batch Processing Example

```prolog
compile_enhanced_pipeline([
    extract/1,
    filter_by(is_valid),
    batch(100),           % Collect 100 records
    bulk_insert/1,        % Process batch
    unbatch,              % Flatten back to records
    output/1
], [pipeline_name(batch_pipe)], Code).
```

## Sorting Stages

Pipeline-level sorting stages for ordering records by field values or custom comparison logic.

### Order By (`order_by/1`, `order_by/2`)

Sort records by a single field.

```prolog
% Ascending (default)
order_by(timestamp)

% With explicit direction
order_by(timestamp, asc)
order_by(timestamp, desc)
```

**Data Flow:**
```
{ts: 3, name: "C"} ─┐          {ts: 1, name: "A"}
{ts: 1, name: "A"} ─┼─► order_by(ts) ─►  {ts: 2, name: "B"}
{ts: 2, name: "B"} ─┘          {ts: 3, name: "C"}
```

### Multi-Field Order By

Sort by multiple fields with individual directions.

```prolog
% Sort by category ascending, then by price descending
order_by([(category, asc), (price, desc)])
```

**Example:**
```prolog
compile_enhanced_pipeline([
    parse/1,
    order_by([(department, asc), (salary, desc)]),
    output/1
], [pipeline_name(sorted_employees)], Code).
```

### Sort By (`sort_by/1`)

Sort using a custom comparator function. The comparator takes two records and returns:
- Negative (or `-1`) if first record comes before second
- Zero (`0`) if records are equal
- Positive (or `1`) if first record comes after second

```prolog
sort_by(compare_priority)
```

**Use Cases:**
- Complex sorting logic that can't be expressed as field ordering
- Multi-criteria sorting with custom weighting
- Domain-specific ordering rules

**Example (Python target):**
```prolog
compile_enhanced_pipeline([
    parse/1,
    sort_by(compare_priority),
    output/1
], [pipeline_name(priority_sorted)], Code).
```

The generated code expects a `compare_priority(record_a, record_b)` function that returns the comparison result.

### Order By vs Sort By

| Stage | Use When |
|-------|----------|
| `order_by(Field)` | Simple field-based sorting |
| `order_by(Field, Dir)` | Field sorting with direction |
| `order_by(FieldSpecs)` | Multiple fields with directions |
| `sort_by(Pred)` | Custom comparison logic needed |

**Key Distinction:** `order_by` is declarative (specify fields), `sort_by` is programmatic (specify comparison function).

## Complex Pipeline Example

Here's a complete example combining all stage types:

```prolog
compile_enhanced_pipeline([
    % Stage 1: Parse input
    parse_record/1,

    % Stage 2: Filter inactive records
    filter_by(is_active),

    % Stage 3: Fan-out to validation, enrichment, and audit
    fan_out([
        validate_schema/1,
        enrich_metadata/1,
        audit_log/1
    ]),

    % Stage 4: Merge parallel results
    merge,

    % Stage 5: Route based on validation result
    route_by(validation_passed, [
        (true, transform_success/1),
        (false, handle_error/1)
    ]),

    % Stage 6: Format output
    format_output/1
], [pipeline_name(data_processor)], Code).
```

**Visual Flow:**
```
Input ─► parse ─► filter(active) ─┬─► validate ──┐
                                  ├─► enrich   ──┼─► merge ─► route ─┬─► transform ─► output
                                  └─► audit    ──┘           (pass?) └─► error     ─► output
```

## Target-Specific Features

### Python/IronPython
- Uses Python generators (`yield from`) for lazy evaluation
- IronPython uses .NET `List<T>` and `Dictionary<TKey,TValue>` for CLR interop
- IronPython parallel uses .NET `Task.Factory.StartNew` with `ConcurrentBag<T>`
- Supports both CPython and IronPython runtimes

**IronPython Parallel Example:**
```prolog
compile_ironpython_enhanced_pipeline([
    extract/1,
    parallel([validate/1, enrich/1]),  % Concurrent via .NET Tasks
    merge,
    output/1
], [pipeline_name(my_pipeline)], Code).
```

**Generated IronPython Code:**
```python
def parallel_records(record, stages):
    from System.Threading.Tasks import Task, TaskFactory
    from System.Collections.Concurrent import ConcurrentBag

    results_bag = ConcurrentBag[object]()

    def run_stage(stage):
        stage_results = list(stage(iter([record])))
        for result in stage_results:
            results_bag.Add(result)

    tasks = List[Task]()
    for stage in stages:
        task = Task.Factory.StartNew(lambda s=stage: run_stage(s))
        tasks.Add(task)

    Task.WaitAll(tasks.ToArray())
    return list(results_bag)
```

### Go
- Uses slices and maps for efficient collection handling
- Generates helper functions: `fanOutRecords`, `mergeStreams`, `routeRecord`, `filterRecords`
- Fully compiled, standalone binaries

### C#
- Uses LINQ and IEnumerable for streaming
- Generates `EnhancedPipelineHelpers` class
- Integrates with .NET ecosystem

### Rust
- Uses iterators and closures with generics
- Memory-safe with ownership semantics
- Generates type-safe helper functions
- **Parallel modes:**
  - `std_thread` (default): Use `std::thread` for parallelism (no extra deps)
  - `rayon`: Use rayon crate for parallel iterators (requires `rayon = "1.8"` in Cargo.toml)

**Rust Rayon Example:**
```prolog
compile_rust_enhanced_pipeline([
    extract/1,
    parallel([validate/1, enrich/1]),
    merge,
    output/1
], [pipeline_name(my_pipe), parallel_mode(rayon)], Code).
```

This generates Rust code using `par_iter()` from rayon for efficient work-stealing parallelism.

### PowerShell
- Uses cmdlet-style helper functions
- Object pipeline integration with `PSCustomObject`
- Cross-platform (PowerShell 7+)

### AWK
- Uses indirect function calls (`@function_name`) for dynamic dispatch
- Global arrays for state management
- JSONL processing helpers
- **Parallel modes:**
  - `sequential` (default): Pure AWK, parallel stages run sequentially
  - `gnu_parallel`: Bash+AWK hybrid using GNU Parallel for true concurrency

**AWK GNU Parallel Example:**
```prolog
compile_awk_enhanced_pipeline([
    extract/1,
    parallel([validate/1, enrich/1]),
    merge,
    output/1
], [pipeline_name(my_pipe), parallel_mode(gnu_parallel)], Code).
```

This generates a bash script that uses GNU Parallel to execute parallel stages concurrently.
Requires: `parallel` (GNU Parallel) and `gawk`.

### Bash
- Uses associative arrays (Bash 4.0+)
- `FAN_OUT_RESULTS` array for fan-out collection
- `ROUTE_MAP` associative array for routing

## Pipeline Validation

Enhanced pipelines are validated at compile-time to catch errors early. Validation is enabled by default and checks for:

**Errors** (compilation fails):
- Empty pipeline
- Invalid stage types
- Empty `fan_out` (no sub-stages)
- Empty `parallel` (no sub-stages)
- Single-stage `parallel` (use regular stage instead - need 2+ for parallelism benefit)
- Empty `route_by` (no routes)
- Invalid route format (must be `(Condition, Stage)`)

**Warnings** (compilation succeeds with message):
- `fan_out` without subsequent `merge` - parallel results may be nested
- `parallel` without subsequent `merge` - parallel results may be nested
- `merge` without preceding `fan_out` or `parallel` - results may be unexpected

### Validation Options

| Option | Description | Default |
|--------|-------------|---------|
| `validate(Bool)` | Enable/disable validation | `true` |
| `strict(Bool)` | Treat warnings as errors | `false` |

### Example: Disabling Validation

```prolog
% Skip validation (not recommended)
compile_enhanced_pipeline([...], [validate(false)], Code).
```

### Example: Strict Mode

```prolog
% Treat warnings as errors
compile_enhanced_pipeline([...], [strict(true)], Code).
```

## Options

Common options across all targets:

| Option | Description | Default |
|--------|-------------|---------|
| `pipeline_name(Name)` | Name of the generated pipeline function | `enhanced_pipeline` |
| `record_format(Format)` | Input/output format (`jsonl`, `tsv`, `csv`) | `jsonl` |
| `output_format(Format)` | Output serialization format | Target-specific |
| `validate(Bool)` | Enable compile-time validation | `true` |
| `strict(Bool)` | Treat warnings as errors | `false` |

Target-specific options are documented in each target's guide.

## Testing

Each target includes comprehensive tests:

```bash
# Run pipeline validation tests
swipl -g "use_module(src/unifyweaver/core/pipeline_validation), test_pipeline_validation" -t halt
./tests/integration/test_pipeline_validation.sh

# Run all enhanced chaining tests for a specific target
swipl -g "use_module(src/unifyweaver/targets/python_target), test_enhanced_pipeline_chaining" -t halt
swipl -g "use_module(src/unifyweaver/targets/go_target), test_go_enhanced_chaining" -t halt
swipl -g "use_module(src/unifyweaver/targets/bash_target), test_bash_enhanced_chaining" -t halt

# Run E2E integration tests
./tests/integration/test_enhanced_chaining_multi_target.sh
./tests/integration/test_awk_enhanced_chaining.sh
./tests/integration/test_bash_enhanced_chaining.sh
./tests/integration/test_ironpython_enhanced_chaining.sh
```

## Best Practices

1. **Use `fan_out` for parallel processing** - When you need to apply multiple independent transformations to each record

2. **Always follow `fan_out` with `merge`** - Unless you want nested results

3. **Use `filter_by` early** - Filter records as early as possible to reduce downstream processing

4. **Use `route_by` for conditional flows** - Instead of filtering and processing separately

5. **Keep predicates pure** - Stage predicates should not have side effects for predictable behavior

## See Also

- [Python Target Guide](PYTHON_TARGET.md)
- [Go Target Guide](GO_TARGET.md)
- [PowerShell Target Guide](POWERSHELL_TARGET.md)
- [Cross-Target Glue Guide](guides/cross-target-glue.md)
