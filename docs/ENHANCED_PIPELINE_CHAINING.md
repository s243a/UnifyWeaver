# Enhanced Pipeline Chaining

UnifyWeaver supports **enhanced pipeline chaining** across all major targets, enabling complex data flow patterns beyond simple linear pipelines.

## Overview

Enhanced pipeline chaining adds five new stage types to the standard predicate stages:

| Stage Type | Description |
|------------|-------------|
| `fan_out(Stages)` | Broadcast each record to multiple stages (sequential execution) |
| `parallel(Stages)` | Execute stages concurrently using target-native parallelism |
| `merge` | Combine results from fan_out or parallel stages |
| `route_by(Pred, Routes)` | Route records to different stages based on a predicate condition |
| `filter_by(Pred)` | Filter records that satisfy a predicate |
| `Pred/Arity` | Standard predicate stage (unchanged) |

### Fan-out vs Parallel

The key difference between `fan_out` and `parallel`:

- **`fan_out(Stages)`**: Processes stages **sequentially**, one after another. Safe for any workload.
- **`parallel(Stages)`**: Processes stages **concurrently** using target-native mechanisms:
  - **Python**: `ThreadPoolExecutor`
  - **Go**: Goroutines with `sync.WaitGroup`
  - **C#**: `Task.WhenAll`
  - **Rust**: `std::thread`
  - **PowerShell**: Runspace pools
  - **Bash**: Background processes with `wait`
  - **IronPython**: .NET `Task.Factory.StartNew` with `ConcurrentBag<T>`
  - **AWK**: Sequential (single-threaded by design)

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

### PowerShell
- Uses cmdlet-style helper functions
- Object pipeline integration with `PSCustomObject`
- Cross-platform (PowerShell 7+)

### AWK
- Uses indirect function calls (`@function_name`) for dynamic dispatch
- Global arrays for state management
- JSONL processing helpers

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
