---
file_type: UnifyWeaver Example Library
spec_version: 1.0
---

# Partitioning and Parallel Execution Examples

This file contains examples of using UnifyWeaver's partitioning system for parallel data processing.

---

## Overview

UnifyWeaver's partitioning system enables:
- **Data parallelism**: Split large datasets across multiple workers
- **Strategy selection**: Choose partitioning strategy based on data characteristics
- **Plugin architecture**: Extensible with custom partitioners
- **Backend integration**: Works with Bash parallel, GNU Parallel, etc.

---

### Fixed-Size Partitioning

> [!example-record]
> id: 20251117-fixed-size-rows
> name: unifyweaver.partitioning.fixed_size_rows
> pattern: fixed_size_partitioning
> difficulty: beginner
> tags: [partitioning, parallel, rows]

Partition data into fixed-size chunks by row count.

**Use Case**: Process large files in batches of N rows.

**Example: Process 1000 items in batches of 100**

```prolog
% Initialize partitioner with fixed size (100 rows per partition)
:- use_module(unifyweaver(core/partitioner)).

process_large_dataset(InputFile, Results) :-
    % Read input data
    read_file_to_string(InputFile, DataStr, []),
    split_string(DataStr, "\n", "", Lines),

    % Initialize fixed-size partitioner
    partitioner_init(fixed_size(rows(100)), [], Handle),

    % Partition the data
    partitioner_partition(Handle, Lines, Partitions),

    % Process each partition (could be parallel)
    maplist(process_partition, Partitions, PartialResults),

    % Combine results
    append(PartialResults, Results),

    % Clean up
    partitioner_cleanup(Handle).

process_partition(Partition, Result) :-
    length(Partition, N),
    format('Processing partition with ~w items~n', [N]),
    % ... actual processing logic ...
    Result = Partition.
```

**Key Points**:
- `rows(100)`: 100 items per partition
- Last partition may be smaller if data doesn't divide evenly
- Deterministic ordering maintained

**Compilation**: When compiled to Bash, generates parallel processing scripts:
```bash
# Generated Bash uses parallel processing
cat input.txt | partition_by_rows 100 | while read partition; do
    process_partition "$partition"
done
```

---

### Hash-Based Partitioning

> [!example-record]
> id: 20251117-hash-based
> name: unifyweaver.partitioning.hash_based
> pattern: hash_partitioning
> difficulty: intermediate
> tags: [partitioning, parallel, hash, key_distribution]

Partition data by hash value for balanced distribution.

**Use Case**: Distribute data across workers based on key hash.

**Example: Partition users by ID hash**

```prolog
:- use_module(unifyweaver(core/partitioner)).

partition_users_by_id(Users, NumWorkers, Partitions) :-
    % Initialize hash-based partitioner
    partitioner_init(
        hash_based(
            field(user_id),      % Hash on user_id field
            partitions(NumWorkers) % N partitions
        ),
        [],
        Handle
    ),

    % Partition users
    partitioner_partition(Handle, Users, Partitions),

    % Cleanup
    partitioner_cleanup(Handle).

% Example usage
example_hash_partitioning :-
    Users = [
        user(id_001, 'Alice'),
        user(id_002, 'Bob'),
        user(id_003, 'Charlie'),
        user(id_004, 'Diana'),
        user(id_005, 'Eve')
    ],

    % Partition across 3 workers
    partition_users_by_id(Users, 3, Partitions),

    % Each partition gets roughly 1-2 users
    forall(
        nth1(N, Partitions, Partition),
        (   length(Partition, Count),
            format('Worker ~w: ~w users~n', [N, Count])
        )
    ).
```

**Key Points**:
- Consistent hashing: Same key always goes to same partition
- Balanced distribution: Hash function distributes evenly
- Useful for grouping related data together

**Bash Compilation**: Generates hash-based routing:
```bash
# Generated script routes by hash
while read line; do
    hash=$(echo "$line" | hash_field 1)
    worker=$((hash % 3))
    echo "$line" >> "partition_$worker.txt"
done < users.txt
```

---

### Key-Based Partitioning

> [!example-record]
> id: 20251117-key-based
> name: unifyweaver.partitioning.key_based
> pattern: key_partitioning
> difficulty: intermediate
> tags: [partitioning, parallel, key_grouping]

Group data by key value (e.g., all items with same category together).

**Use Case**: Group related items for batch processing.

**Example: Group orders by customer**

```prolog
:- use_module(unifyweaver(core/partitioner)).

partition_orders_by_customer(Orders, CustomerPartitions) :-
    % Initialize key-based partitioner
    partitioner_init(
        key_based(
            field(customer_id),    % Group by customer_id
            strategy(group_all)    % All items with same key in one partition
        ),
        [],
        Handle
    ),

    % Partition orders
    partitioner_partition(Handle, Orders, CustomerPartitions),

    % Cleanup
    partitioner_cleanup(Handle).

% Example usage
example_key_partitioning :-
    Orders = [
        order(1, customer_a, 100),
        order(2, customer_b, 50),
        order(3, customer_a, 75),
        order(4, customer_c, 200),
        order(5, customer_b, 125)
    ],

    partition_orders_by_customer(Orders, Partitions),

    % Should have 3 partitions (one per customer)
    length(Partitions, NumPartitions),
    format('Created ~w partitions (one per customer)~n', [NumPartitions]),

    % Show partition sizes
    forall(
        member(Partition, Partitions),
        (   Partition = [First|_],
            First = order(_, CustomerID, _),
            length(Partition, Count),
            format('  Customer ~w: ~w orders~n', [CustomerID, Count])
        )
    ).
```

**Key Points**:
- Preserves relationships: All items with same key together
- Variable partition sizes: Depends on key distribution
- Good for aggregation operations

**Expected Output**:
```
Created 3 partitions (one per customer)
  Customer customer_a: 2 orders
  Customer customer_b: 2 orders
  Customer customer_c: 1 orders
```

---

### Partitioning with Backends

> [!example-record]
> id: 20251117-parallel-backend
> name: unifyweaver.partitioning.parallel_backend
> pattern: partitioning_with_backend
> parent_example: unifyweaver.partitioning.fixed_size_rows
> difficulty: advanced
> tags: [partitioning, parallel, gnu_parallel, backend]

Combine partitioning with execution backends for actual parallel processing.

**Use Case**: Process large datasets with GNU Parallel or bash parallel.

**Example: Process log files in parallel**

```prolog
:- use_module(unifyweaver(core/partitioner)).
:- use_module(unifyweaver(core/backends/gnu_parallel)).

process_logs_parallel(LogFile, SummaryFile) :-
    % Read log lines
    read_file_to_string(LogFile, Content, []),
    split_string(Content, "\n", "", Lines),

    % Partition into chunks of 1000 lines
    partitioner_init(fixed_size(rows(1000)), [], PartHandle),
    partitioner_partition(PartHandle, Lines, Partitions),
    partitioner_cleanup(PartHandle),

    % Set up GNU Parallel backend
    backend_init(gnu_parallel(workers(4)), [], BackendHandle),

    % Process partitions in parallel
    backend_execute(BackendHandle, process_partition, Partitions, Results),

    % Aggregate results
    aggregate_results(Results, Summary),

    % Write summary
    open(SummaryFile, write, Out),
    write(Out, Summary),
    close(Out),

    % Cleanup backend
    backend_cleanup(BackendHandle).

process_partition(PartitionLines, PartitionSummary) :-
    % Analyze lines in this partition
    length(PartitionLines, LineCount),
    include(is_error_line, PartitionLines, ErrorLines),
    length(ErrorLines, ErrorCount),

    PartitionSummary = summary(
        total_lines(LineCount),
        error_lines(ErrorCount)
    ).

is_error_line(Line) :-
    sub_string(Line, _, _, _, "ERROR").

aggregate_results(Summaries, AggregatedSummary) :-
    maplist(arg(1), Summaries, TotalLineCounts),
    maplist(arg(2), Summaries, ErrorLineCounts),
    sum_list(TotalLineCounts, TotalLines),
    sum_list(ErrorLineCounts, TotalErrors),

    AggregatedSummary = final_summary(
        total_lines(TotalLines),
        error_lines(TotalErrors),
        error_rate(Rate)
    ),
    Rate is TotalErrors / TotalLines.
```

**Compilation to Bash**:

Generates script using GNU Parallel:
```bash
#!/bin/bash
# Generated by UnifyWeaver
# Uses GNU Parallel for parallel processing

# Partition input
cat logs.txt | partition_by_rows 1000 > partitions.list

# Process in parallel (4 workers)
cat partitions.list | parallel -j 4 process_partition

# Aggregate results
aggregate_results summary.txt
```

**Performance**:
- 4 workers: ~4x speedup (ideal case)
- I/O bound tasks: Less speedup
- CPU bound tasks: Near-linear scaling

---

### Complete Example: ETL Pipeline with Partitioning

> [!example-record]
> id: 20251117-etl-pipeline
> name: unifyweaver.workflow.etl_partitioned
> pattern: complete_etl
> child_examples: [unifyweaver.partitioning.fixed_size_rows, unifyweaver.partitioning.parallel_backend]
> difficulty: advanced
> tags: [etl, partitioning, parallel, production]

Complete ETL pipeline using partitioning for parallel processing.

**Scenario**: Process 1M customer records from CSV, transform, and load to database.

**Pipeline Stages**:
1. **Extract**: Read CSV file
2. **Partition**: Split into 100K-record chunks
3. **Transform**: Process each partition in parallel
4. **Load**: Write to database in batches

**Implementation**:

```prolog
:- use_module(unifyweaver(sources/csv)).
:- use_module(unifyweaver(core/partitioner)).
:- use_module(unifyweaver(core/backends/gnu_parallel)).

etl_pipeline(InputCSV, OutputDB) :-
    format('Starting ETL pipeline...~n'),

    % Stage 1: Extract
    format('  [1/4] Extracting data from ~w~n', [InputCSV]),
    csv_read(InputCSV, Records),
    length(Records, TotalRecords),
    format('        Read ~w records~n', [TotalRecords]),

    % Stage 2: Partition
    format('  [2/4] Partitioning data~n'),
    partitioner_init(fixed_size(rows(100000)), [], PHandle),
    partitioner_partition(PHandle, Records, Partitions),
    length(Partitions, NumPartitions),
    format('        Created ~w partitions~n', [NumPartitions]),
    partitioner_cleanup(PHandle),

    % Stage 3: Transform (parallel)
    format('  [3/4] Transforming data (parallel)~n'),
    backend_init(gnu_parallel(workers(8)), [], BHandle),
    backend_execute(BHandle, transform_partition, Partitions, TransformedPartitions),
    backend_cleanup(BHandle),
    format('        Transformation complete~n'),

    % Stage 4: Load
    format('  [4/4] Loading to database~n'),
    append(TransformedPartitions, AllTransformed),
    load_to_database(OutputDB, AllTransformed),
    length(AllTransformed, LoadedRecords),
    format('        Loaded ~w records~n', [LoadedRecords]),

    format('ETL pipeline complete!~n').

transform_partition(InputRecords, OutputRecords) :-
    maplist(transform_record, InputRecords, OutputRecords).

transform_record(
    customer(ID, Name, Email),
    transformed_customer(ID, NormalizedName, ValidatedEmail, Timestamp)
) :-
    normalize_name(Name, NormalizedName),
    validate_email(Email, ValidatedEmail),
    get_time(Timestamp).

normalize_name(Name, Normalized) :-
    upcase_atom(Name, Upper),
    Normalized = Upper.

validate_email(Email, validated(Email)) :-
    sub_atom(Email, _, _, _, '@'),
    !.
validate_email(Email, invalid(Email)).

load_to_database(DBFile, Records) :-
    % Batch insert to database
    open(DBFile, write, Stream),
    forall(
        member(Record, Records),
        (   format(Stream, '~q.~n', [Record])
        )
    ),
    close(Stream).
```

**Usage**:
```bash
# Compile the ETL pipeline
swipl -q -g "
    asserta(file_search_path(unifyweaver, 'src/unifyweaver')),
    [etl_pipeline.pl],
    use_module(unifyweaver(core/compiler_driver)),
    compile(etl_pipeline/2, [target(bash)], Scripts),
    halt"

# Run the generated script
./education/output/advanced/etl_pipeline.sh customers.csv output.db
```

**Expected Output**:
```
Starting ETL pipeline...
  [1/4] Extracting data from customers.csv
        Read 1000000 records
  [2/4] Partitioning data
        Created 10 partitions
  [3/4] Transforming data (parallel)
        Transformation complete
  [4/4] Loading to database
        Loaded 1000000 records
ETL pipeline complete!
```

**Performance Characteristics**:
- **Sequential**: ~10 minutes
- **Parallel (8 workers)**: ~2-3 minutes (3-5x speedup)
- **Bottleneck**: Usually I/O (file reading/writing)
- **Optimization**: Increase workers if CPU-bound

---

## Partitioning Strategies Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **fixed_size** | Uniform processing | Simple, predictable | May imbalance if data varies |
| **hash_based** | Distributed systems | Even distribution | Requires good hash function |
| **key_based** | Grouping/aggregation | Preserves relationships | Variable partition sizes |

## Best Practices

### Choosing Partition Size

```prolog
% Too small: Overhead dominates
partitioner_init(fixed_size(rows(10)), [], Handle),  % Bad for 1M records

% Too large: No parallelism benefit
partitioner_init(fixed_size(rows(1000000)), [], Handle),  % Bad for 8 workers

% Good: 10-100x workers
% For 1M records, 8 workers: 100K-125K per partition
partitioner_init(fixed_size(rows(125000)), [], Handle),  % Good
```

### Error Handling

```prolog
partition_with_fallback(Data, Partitions) :-
    catch(
        (   partitioner_init(fixed_size(rows(1000)), [], H),
            partitioner_partition(H, Data, Partitions),
            partitioner_cleanup(H)
        ),
        Error,
        (   format('Partitioning failed: ~w~n', [Error]),
            Partitions = [Data]  % Fallback: single partition
        )
    ).
```

### Testing Partitioning Logic

```prolog
test_partitioning :-
    % Test data
    Data = [1,2,3,4,5,6,7,8,9,10],

    % Partition
    partitioner_init(fixed_size(rows(3)), [], H),
    partitioner_partition(H, Data, Partitions),
    partitioner_cleanup(H),

    % Verify
    Partitions = [[1,2,3], [4,5,6], [7,8,9], [10]],  % Expected
    format('✓ Partitioning test passed~n').
```

## References

- **UnifyWeaver Core Documentation**: Partitioner API
- **GNU Parallel**: https://www.gnu.org/software/parallel/
- **Related Examples**:
  - [Data Source Integration](data_source_playbooks.md)
  - [Testing Patterns](testing_examples.md)
  - [Compilation Examples](compilation_examples.md)

---

**Version**: 1.0
**Last Updated**: 2025-11-17
**Example Count**: 5 (basic partitioning) + 1 (complete ETL)
