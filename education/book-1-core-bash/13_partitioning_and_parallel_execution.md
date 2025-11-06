<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 13: Partitioning and Parallel Execution

**Target Audience:** Intermediate users who understand basic UnifyWeaver concepts
**Prerequisites:** Chapters 1-5 (especially stream compilation and data sources)
**Time to Read:** 20-30 minutes

---

## Table of Contents

1. [Introduction: Why Partition Data?](#introduction)
2. [Partitioning Fundamentals](#fundamentals)
3. [Access Patterns and Storage Models](#access-patterns)
4. [Batch vs Streaming Partitioning](#batch-vs-streaming)
5. [Implemented Partitioning Strategies](#strategies)
6. [Parallel Execution Backend](#parallel-backend)
7. [Practical Examples](#examples)
8. [When to Use Which Strategy](#decision-guide)
9. [Future Directions](#future)

---

## Introduction: Why Partition Data?

Partitioning is the art of dividing data into smaller, manageable chunks. It serves two fundamental purposes:

1. **Enable Parallelism** - Process multiple chunks simultaneously across CPU cores or machines
2. **Enable Divide-and-Conquer** - Break complex problems into simpler subproblems (foundation for algorithms like quicksort)

UnifyWeaver's partitioning system provides a **pluggable architecture** where you can choose the right partitioning strategy for your use case.

### Real-World Motivations

**Corporate Environment:**
- Process 100GB log file across 16 cores
- Each core processes ~6GB partition independently
- 16× speedup for embarassingly parallel tasks

**Development:**
- Test data processing pipeline on small partition
- Validate correctness before running on full dataset
- Debug specific data ranges that caused issues

**Resource-Constrained (Termux/Android):**
- Limited memory prevents loading entire dataset
- Process data in fixed-size chunks that fit in RAM
- Stream results without materializing full output

---

## Partitioning Fundamentals

### Core Concepts

**Partition**: A subset of the input data, identified by an ID and containing a list of items.

```prolog
% Prolog representation
partition(0, [item1, item2, item3]).
partition(1, [item4, item5, item6]).
partition(2, [item7, item8, item9]).
```

**Partitioning Strategy**: An algorithm that decides how to split data.

**Plugin Interface**: All strategies implement the same lifecycle:

```prolog
% Initialize strategy with configuration
strategy_init(+Config, -State).

% Partition entire dataset (batch mode)
strategy_partition(+State, +DataStream, -Partitions).

% Assign single item to partition (streaming mode)
strategy_assign(+State, +Item, -PartitionID).

% Clean up resources
strategy_cleanup(+State).
```

### The Partitioner Registry

UnifyWeaver uses a **plugin registry pattern** for partitioning strategies:

```prolog
% Register a strategy (done once at initialization)
register_partitioner(fixed_size, fixed_size_partitioner).
register_partitioner(hash_based, hash_based_partitioner).
register_partitioner(key_based, key_based_partitioner).

% Use any registered strategy
partitioner_init(fixed_size(rows(100)), [], Handle).
partitioner_partition(Handle, Data, Partitions).
partitioner_cleanup(Handle).
```

This allows you to:
- Switch strategies without code changes
- Add custom strategies as plugins
- Compare strategies for your workload

---

## Access Patterns and Storage Models

Different systems optimize for different access patterns. Understanding these helps choose the right partitioning strategy.

### Memory-Mapped Files (nanoGPT, Machine Learning)

**Approach**: Map entire file into virtual memory, use random access pointers.

```
File: training_data.bin (10GB)
├─ Memory map entire file
├─ Partition by byte offsets:
│  ├─ Partition 0: bytes 0 - 1,073,741,824     (1GB)
│  ├─ Partition 1: bytes 1,073,741,824 - ...   (1GB)
│  └─ ...
└─ Each worker accesses its byte range directly
```

**Advantages:**
- ✅ True random access (O(1) seek to any position)
- ✅ No data copying - kernel manages paging
- ✅ Efficient for fixed-size records

**Disadvantages:**
- ❌ Requires entire file fits in virtual memory space
- ❌ Doesn't work well with variable-length records (CSV, JSON)
- ❌ Page faults if working set exceeds physical RAM

**UnifyWeaver Equivalent:**
```prolog
% Partition by byte ranges (requires seekable data source)
partitioner_init(fixed_size(bytes(1073741824)), [], Handle).
% Each partition gets 1GB slice
```

### File-Based Partitioning (Hadoop, MapReduce)

**Approach**: Split data across multiple files aligned to filesystem block boundaries.

```
HDFS Storage:
├─ part-00000 (128MB, HDFS block boundary)
├─ part-00001 (128MB, HDFS block boundary)
├─ part-00002 (128MB, HDFS block boundary)
└─ ...

Each mapper reads entire files locally (data locality).
```

**Advantages:**
- ✅ Excellent data locality (process where data lives)
- ✅ Filesystem block alignment (no partial block reads)
- ✅ Natural checkpoint/resume points
- ✅ Works with append-only storage (HDFS)

**Disadvantages:**
- ❌ Requires splitting data during ingestion
- ❌ Not suitable for small datasets (overhead of multiple files)

**UnifyWeaver Future:**
```prolog
% Not yet implemented, but conceptually:
partitioner_init(file_based(block_size(134217728)), [], Handle).
% Would create part-00000, part-00001, ... aligned to 128MB blocks
```

### Hybrid Approach (Practical Reality)

**Approach**: Multiple files, each with known size/record count metadata.

```
Dataset:
├─ users_2024_01.csv (10,000 users, 2.5MB)
├─ users_2024_02.csv (12,000 users, 3.0MB)
├─ users_2024_03.csv (11,500 users, 2.8MB)
└─ metadata.json:
    {
      "users_2024_01.csv": {"rows": 10000, "bytes": 2621440},
      "users_2024_02.csv": {"rows": 12000, "bytes": 3145728},
      ...
    }
```

**Advantages:**
- ✅ File-based partitioning (one file = one partition)
- ✅ Known boundaries for work distribution
- ✅ Can pre-compute partition sizes
- ✅ Easy to add/remove partitions (just files)

**UnifyWeaver Support:**
```prolog
% Each data source file naturally becomes a partition
source(csv, 'users_2024_01.csv', [...]),
source(csv, 'users_2024_02.csv', [...]),
source(csv, 'users_2024_03.csv', [...]).

% Or use hash partitioning to redistribute within files
partitioner_init(hash_based(key(column(1)), num_partitions(8)), [], H).
```

### Summary: Which Model When?

| Model | Best For | Example Use Cases |
|-------|----------|-------------------|
| **Memory-Mapped** | Fixed-size records, random access | ML training (nanoGPT), binary data, databases |
| **File-Based** | Distributed systems, large datasets | Hadoop/MapReduce, data lakes, log aggregation |
| **Hybrid** | Practical data pipelines | Time-series data (one file per day), partitioned tables |
| **In-Memory** | Small datasets, flexible schemas | Development, testing, ad-hoc analysis |

UnifyWeaver currently focuses on **in-memory** and **hybrid** models, with future support for **file-based** partitioning planned.

---

## Batch vs Streaming Partitioning

This is a fundamental design distinction that affects how partitioning interacts with the rest of your pipeline.

### Batch Partitioning (Current Implementation)

**Model**: Scan entire dataset first, then partition, then process.

```
Data Flow:
┌─────────────┐
│ Data Source │
└──────┬──────┘
       │ (Read all data into memory)
       ▼
┌─────────────┐
│ Partitioner │ ← Sees entire dataset
└──────┬──────┘
       │ (Split into N partitions)
       ▼
┌─────────────┐
│  Partition  │───┐
│     0       │   │
└─────────────┘   │
┌─────────────┐   │  (Process in parallel)
│  Partition  │───┤
│     1       │   │
└─────────────┘   │
┌─────────────┐   │
│  Partition  │───┘
│     2       │
└─────────────┘
       │
       ▼
┌─────────────┐
│   Results   │
└─────────────┘
```

**Advantages:**
- ✅ Simple to implement
- ✅ Can make global decisions (e.g., balance partition sizes)
- ✅ Works with any partitioning strategy
- ✅ Easy to test and debug

**Disadvantages:**
- ❌ Must materialize entire dataset in memory
- ❌ Adds latency (can't start processing until scan complete)
- ❌ Not suitable for infinite streams

**UnifyWeaver Example:**
```prolog
% Read all data
numlist(1, 1000, Data),

% Partition into 10 chunks
partitioner_init(fixed_size(rows(100)), [], Handle),
partitioner_partition(Handle, Data, Partitions),
% Partitions = [partition(0, [1..100]), partition(1, [101..200]), ...]

% Execute in parallel
backend_init(gnu_parallel(workers(4)), BHandle),
backend_execute(BHandle, Partitions, 'process.sh', Results).
```

### Streaming Partitioning (Future Design)

**Model**: Partition items as they arrive, trigger processing on-the-fly.

```
Data Flow (Event-Driven):
┌─────────────┐
│ Data Source │ (Streaming: one item at a time)
└──────┬──────┘
       │ item1, item2, item3, ...
       ▼
┌─────────────┐
│ Partitioner │ ← Assigns each item to partition
└──────┬──────┘
       │ (Triggers pipeline for each partition range)
       ▼
┌──────────────────────────────────┐
│  Parallel Backend (Event-Driven) │
│  ┌──────────┐  ┌──────────┐     │
│  │ Worker 0 │  │ Worker 1 │ ... │
│  │ (rows    │  │ (rows    │     │
│  │  0-100)  │  │  101-200)│     │
│  └──────────┘  └──────────┘     │
└───────────────┬──────────────────┘
                │
                ▼
          ┌─────────────┐
          │   Results   │
          │  (streaming)│
          └─────────────┘
```

**Advantages:**
- ✅ Constant memory usage (don't materialize full dataset)
- ✅ Low latency (start processing immediately)
- ✅ Supports infinite streams
- ✅ Can trigger upstream pipelines with range parameters

**Disadvantages:**
- ❌ More complex to implement
- ❌ Harder to balance partition sizes (don't know total size upfront)
- ❌ Some strategies require global knowledge (e.g., range partitioning)

**UnifyWeaver Future Design:**
```prolog
% Conceptual API (not yet implemented)

% Streaming partitioner assigns each item as it arrives
partitioner_init(hash_based(key(column(1)), num_partitions(4)),
                [mode(streaming)], Handle),

% Backend receives items as events, triggers processing per partition
backend_init(gnu_parallel(workers(4)), [mode(event_driven)], BHandle),

% Pipeline processes items as they flow through
process_stream(DataSourceStream, Handle, BHandle, ResultStream).
```

### Example: Range-Based Streaming

**Scenario**: Process rows 1000-2000 from a 1TB CSV file without loading the entire file.

**Batch Approach** (inefficient):
```prolog
% Read entire file (1TB!)
read_csv_all('huge.csv', AllData),

% Partition to find rows 1000-2000
partitioner_init(fixed_size(rows(1000)), [], H),
partitioner_partition(H, AllData, Partitions),
member(partition(1, Rows1000to2000), Partitions).
```

**Streaming Approach** (efficient):
```prolog
% Partitioner triggers upstream pipeline with range
partitioner_init(range_based(start(1000), end(2000)), [mode(streaming)], H),

% Backend tells CSV source: "skip to row 1000, read 1000 rows"
% CSV source uses efficient seeking (no need to read rows 0-999)
backend_execute_streaming(BHandle,
    trigger_source(csv_source, [start_row(1000), limit(1000)]),
    'process.sh',
    Results).
```

**Key Insight**: The parallel backend can call **prior pipeline stages** with specific parameters (start row, byte offset, etc.), enabling efficient random access without materializing the full dataset.

### When to Use Each

| Use Case | Approach | Reason |
|----------|----------|--------|
| Small dataset (fits in RAM) | Batch | Simpler, no downside |
| Large dataset, full scan | Batch | All data needed anyway |
| Large dataset, subset needed | Streaming | Avoid reading unnecessary data |
| Infinite stream (logs, sensors) | Streaming | No "entire dataset" concept |
| Need balanced partitions | Batch | Requires global knowledge |
| Low latency requirement | Streaming | Start processing immediately |

---

## Implemented Partitioning Strategies

UnifyWeaver v0.0.2+ includes three partitioning strategies, each optimized for different use cases.

### Strategy 1: Fixed-Size Partitioning

**Purpose**: Split data into equal-sized chunks.

**Modes:**
- `rows(N)` - N items per partition
- `bytes(B)` - Approximately B bytes per partition (estimates item sizes)

**Example:**
```prolog
% Partition by rows
numlist(1, 1000, Data),
partitioner_init(fixed_size(rows(100)), [], Handle),
partitioner_partition(Handle, Data, Partitions).
% Result: [partition(0, [1..100]), partition(1, [101..200]), ..., partition(9, [901..1000])]
```

**Use Cases:**
- Simple work distribution (each worker gets ~same number of items)
- Testing with small batches
- Resource-constrained environments (limit partition size to available RAM)

**Advantages:**
- ✅ Predictable partition sizes
- ✅ Simple to understand and debug
- ✅ Works with any data type

**Disadvantages:**
- ❌ Doesn't consider data distribution (some partitions may take longer)
- ❌ Byte mode is approximate (estimates, doesn't measure)

### Strategy 2: Hash-Based Partitioning

**Purpose**: Distribute data by hashing a key field (MapReduce-compatible).

**Configuration:**
```prolog
partitioner_init(
    hash_based(
        key(column(1)),              % Hash first column
        num_partitions(8),           % Create 8 partitions
        hash_function(term_hash)     % Use Prolog's term_hash/2
    ),
    [],
    Handle
).
```

**Hash Functions:**
- `simple_mod` - Fast, good for integers (`Key mod NumPartitions`)
- `term_hash` - General purpose, works with any Prolog term
- `atom_hash` - Optimized for atoms/strings

**Example:**
```prolog
% Data with names
Data = [
    row(1, alice, 25),
    row(2, bob, 30),
    row(3, alice, 26),   % Same key as row 1
    row(4, charlie, 35),
    row(5, bob, 31)      % Same key as row 2
],

% Partition by name (column 2)
partitioner_init(hash_based(key(column(2)), num_partitions(3)), [], Handle),
partitioner_partition(Handle, Data, Partitions).

% Result (deterministic):
% partition(0, [row(2, bob, 30), row(5, bob, 31)])      % All 'bob' rows
% partition(1, [row(1, alice, 25), row(3, alice, 26)])  % All 'alice' rows
% partition(2, [row(4, charlie, 35)])                   % All 'charlie' rows
```

**Key Property: Determinism**
- Same key → Same partition (always)
- Critical for MapReduce, distributed joins
- Enables co-location of related data

**Use Cases:**
- **MapReduce**: Shuffle phase (GROUP BY in distributed systems)
- **Distributed Joins**: Co-locate matching keys in same partition
- **Load Balancing**: Distribute work evenly (assuming uniform key distribution)

**Advantages:**
- ✅ Hadoop/MapReduce compatible
- ✅ Deterministic assignment (crucial for distributed systems)
- ✅ Works in streaming mode (can assign items one at a time)

**Disadvantages:**
- ❌ Skewed keys → unbalanced partitions (e.g., 90% of data has same key)
- ❌ Requires choosing good hash key

### Strategy 3: Key-Based Partitioning

**Purpose**: Group all items with the same key together (SQL GROUP BY semantics).

**Configuration:**
```prolog
partitioner_init(
    key_based(key(column(1))),  % Group by first column
    [],
    Handle
).
```

**Example:**
```prolog
% Log data
Logs = [
    log(error, "Connection failed"),
    log(info, "Server started"),
    log(error, "Timeout"),
    log(warning, "High memory"),
    log(info, "Request processed")
],

% Group by log level
partitioner_init(key_based(key(column(1))), [], Handle),
partitioner_partition(Handle, Logs, Partitions).

% Result (each key gets its own partition):
% partition(0, key(error), [log(error, "Connection failed"),
%                           log(error, "Timeout")])
% partition(1, key(info), [log(info, "Server started"),
%                          log(info, "Request processed")])
% partition(2, key(warning), [log(warning, "High memory")])
```

**Key Difference from Hash-Based:**
- **Hash-based**: Fixed number of partitions (N), keys distributed across them
- **Key-based**: Number of partitions = number of unique keys

**Use Cases:**
- **Aggregation**: SUM/COUNT/AVG per group
- **Grouping**: Organize logs by level, sales by region, etc.
- **Category Analysis**: Process each category independently

**Advantages:**
- ✅ Natural for GROUP BY operations
- ✅ Each partition is semantically meaningful (all items with key K)
- ✅ No hash collisions (each key gets own partition)

**Disadvantages:**
- ❌ Number of partitions unknown upfront (depends on unique key count)
- ❌ Skewed keys → unbalanced partitions (same as hash-based)

### Comparison Matrix

| Strategy | Partition Count | Balance | Deterministic | Use Case |
|----------|----------------|---------|---------------|----------|
| **Fixed-Size** | Depends on data size | Good (by construction) | No | Simple parallelism, testing |
| **Hash-Based** | Fixed (configured) | Good (if keys uniform) | Yes | MapReduce, distributed joins |
| **Key-Based** | Unique key count | Depends on data | Yes | GROUP BY, aggregation |

---

## Parallel Execution Backend

Once data is partitioned, we need to execute the processing logic in parallel. UnifyWeaver's backend system uses the same plugin pattern as the partitioner.

### Backend Interface

```prolog
% Initialize backend with configuration
backend_init(+Config, -Handle).

% Execute script on partitions in parallel
backend_execute(+Handle, +Partitions, +ScriptPath, -Results).

% Clean up backend resources
backend_cleanup(+Handle).
```

### GNU Parallel Backend

**What is GNU Parallel?**
- Battle-tested command-line tool for parallel job execution
- Handles process management, error handling, resource limits
- Available on all Unix-like systems (`apt-get install parallel`)

**How UnifyWeaver Uses It:**

```prolog
% Initialize with 4 workers
backend_init(gnu_parallel(workers(4)), BHandle).

% Execute script on 10 partitions
Partitions = [partition(0, [...]), partition(1, [...]), ...],
backend_execute(BHandle, Partitions, 'process.sh', Results).

% Results = [result(0, Output0), result(1, Output1), ...]
```

**Behind the Scenes:**

1. **Write Batch Files**: Each partition's data → temp file
   ```
   /tmp/unifyweaver_12345/
   ├── batch_0.txt  (partition 0 data)
   ├── batch_1.txt  (partition 1 data)
   └── ...
   ```

2. **Build GNU Parallel Command**:
   ```bash
   parallel --jobs 4 \
            --results /tmp/unifyweaver_12345/output_{#} \
            "bash process.sh < {}" \
            ::: batch_0.txt batch_1.txt batch_2.txt ...
   ```

3. **Execute in Parallel**: GNU Parallel manages:
   - Worker pool (max 4 concurrent jobs)
   - Load balancing (starts new job when worker finishes)
   - Output collection (each job's stdout saved separately)

4. **Collect Results**: Read output files → Prolog result terms

**Configuration:**
```prolog
% Default: 4 workers
backend_init(gnu_parallel(workers(4)), Handle).

% More workers for CPU-intensive tasks
backend_init(gnu_parallel(workers(16)), Handle).

% Fewer workers for I/O-bound tasks
backend_init(gnu_parallel(workers(2)), Handle).
```

### Integration with Partitioner

**Complete Example:**

```prolog
% 1. Load data
numlist(1, 100, Data),

% 2. Partition data (10 partitions of 10 items each)
partitioner_init(fixed_size(rows(10)), [], PHandle),
partitioner_partition(PHandle, Data, Partitions),
partitioner_cleanup(PHandle),

% 3. Compile processing script
%    (Imagine a script that doubles each number)
ScriptPath = 'double.sh',

% 4. Execute in parallel (4 workers)
backend_init(gnu_parallel(workers(4)), BHandle),
backend_execute(BHandle, Partitions, ScriptPath, Results),
backend_cleanup(BHandle),

% 5. Results contain output from each partition
% Results = [result(0, "2\n4\n6\n..."), result(1, "22\n24\n..."), ...]
```

### Future: Bash Fork Backend

**Design Goal**: Fallback when GNU Parallel is not available.

```prolog
% Would manually manage worker processes
backend_init(bash_fork(workers(4)), Handle).
```

**Implementation** (future):
- Spawn background bash processes (`bash script.sh &`)
- Track PIDs, wait for completion
- Manage job queue, handle failures
- ~300 lines of bash/Prolog

**Why Not Implemented Yet:**
- GNU Parallel does this better (4+ hours saved)
- Easy to install (`apt-get install parallel`)
- Bash fork backend is low priority (can add later if needed)

---

## Practical Examples

### Example 1: Parallel Log Processing

**Scenario**: Process 1 million log entries, count errors per hour.

```prolog
% 1. Load logs (imagine reading from file)
Logs = [
    log('2024-01-01 10:15:23', error, 'Connection failed'),
    log('2024-01-01 10:16:45', info, 'Request processed'),
    log('2024-01-01 11:02:11', error, 'Timeout'),
    ...  % 1 million entries
],

% 2. Partition by hour (key-based)
partitioner_init(
    key_based(key(hour_of_timestamp)),  % Custom key extractor
    [],
    PHandle
),
partitioner_partition(PHandle, Logs, Partitions),
% Result: partition(0, key('2024-01-01 10'), [...]),
%         partition(1, key('2024-01-01 11'), [...]), ...

% 3. Create aggregation script (counts errors)
compile_aggregation_script('count_errors.sh'),

% 4. Process each hour in parallel
backend_init(gnu_parallel(workers(8)), BHandle),
backend_execute(BHandle, Partitions, 'count_errors.sh', Results),

% 5. Results: [result(0, "15 errors"), result(1, "8 errors"), ...]
```

**Speedup**: 8× faster with 8 workers (assuming CPU-bound aggregation).

### Example 2: CSV Processing with Hash Partitioning

**Scenario**: Process large CSV, join with lookup table (simulated MapReduce).

```prolog
% 1. Load CSV data
source(csv, sales, [file('sales.csv')]),
compile_dynamic_source(sales/4, [], SalesData),
% SalesData = [sale(1, 'Alice', 'Widget', 100), ...]

% 2. Partition by customer name (hash-based)
%    This co-locates all sales for same customer
partitioner_init(
    hash_based(key(column(2)), num_partitions(4)),
    [],
    PHandle
),
partitioner_partition(PHandle, SalesData, Partitions),

% 3. Create aggregation script (SUM sales per customer)
compile_aggregation('sum_sales.sh'),

% 4. Execute in parallel (4 workers, one per partition)
backend_init(gnu_parallel(workers(4)), BHandle),
backend_execute(BHandle, Partitions, 'sum_sales.sh', Results),

% 5. Collect results: [result(0, "Alice: 500\nBob: 300"),
%                       result(1, "Charlie: 200"), ...]
```

**Why Hash Partitioning?**
- Ensures all sales for "Alice" go to same partition
- Allows per-partition aggregation (no cross-partition communication)
- Mimics MapReduce shuffle phase

### Example 3: Testing with Small Partitions

**Scenario**: Test processing logic on subset before running on full dataset.

```prolog
% 1. Load full dataset
read_csv('massive_data.csv', AllData),

% 2. Create small partition for testing (first 100 rows)
partitioner_init(fixed_size(rows(100)), [], Handle),
partitioner_partition(Handle, AllData, Partitions),
[partition(0, TestData)|_] = Partitions,

% 3. Test processing logic on small partition
test_process(TestData, TestResults),

% 4. Verify results are correct
assert_correct(TestResults),

% 5. If tests pass, process all partitions in parallel
backend_init(gnu_parallel(workers(16)), BHandle),
backend_execute(BHandle, Partitions, 'process.sh', AllResults).
```

---

## When to Use Which Strategy

### Decision Tree

```
Start: Do I need parallelism?
│
├─ NO → Don't partition, process sequentially
│
└─ YES → Continue...
    │
    ├─ Is data small (< 1GB, fits in RAM)?
    │  └─ YES → Use fixed_size for simple parallelism
    │
    ├─ Do I need to group by a field (GROUP BY)?
    │  └─ YES → Use key_based partitioning
    │
    ├─ Do I need co-location (distributed join)?
    │  └─ YES → Use hash_based partitioning
    │
    ├─ Do I need consistent assignment (same key → same partition)?
    │  └─ YES → Use hash_based partitioning
    │
    └─ Default → Use fixed_size for simplicity
```

### Strategy Selection Guide

| Your Need | Strategy | Configuration |
|-----------|----------|---------------|
| Simple parallelism | `fixed_size` | `rows(N)` where N = DataSize / NumWorkers |
| Limit memory usage | `fixed_size` | `bytes(B)` where B = Available RAM / NumWorkers |
| GROUP BY aggregation | `key_based` | `key(column(K))` where K is grouping column |
| MapReduce shuffle | `hash_based` | `key(column(K)), num_partitions(W)` |
| Distributed join | `hash_based` | Same key for both datasets |
| Testing/debugging | `fixed_size` | `rows(100)` for small test partition |

---

## Future Directions

### 1. Streaming Partitioning (Event-Driven)

**Vision**: Partition items as they arrive, trigger pipelines on-the-fly.

```prolog
% Future API (conceptual)
partitioner_init(
    hash_based(key(column(1)), num_partitions(4)),
    [mode(streaming)],
    Handle
),

% Process stream item-by-item
process_stream(InputStream, Handle, BHandle, OutputStream).
```

**Use Cases:**
- Real-time log processing
- Sensor data aggregation
- Infinite streams (network traffic, IoT)

### 2. Range-Based Partitioning

**Purpose**: Partition by value ranges (useful for ordered data).

```prolog
% Partition by timestamp ranges
partitioner_init(
    range_based([
        range(0, '2024-01-01', '2024-01-31'),     % January
        range(1, '2024-02-01', '2024-02-29'),     % February
        range(2, '2024-03-01', '2024-03-31')      % March
    ]),
    [],
    Handle
).
```

**Use Cases:**
- Time-series data (one partition per day/month)
- Sorted data (numeric ranges)
- Distributed databases (range-based sharding)

### 3. Pivot-Based Partitioning (Quicksort Connection)

**Purpose**: Partition around a pivot value (foundation for divide-and-conquer algorithms).

```prolog
% Choose pivot, partition into [<pivot], [=pivot], [>pivot]
partitioner_init(
    pivot_based(pivot(50)),
    [],
    Handle
),
partitioner_partition(Handle, Data, Partitions).
% Result: [partition(less, [...]), partition(equal, [...]), partition(greater, [...])]
```

**Future**: Foundation for implementing quicksort when bash supports tree recursion.

### 4. Histogram-Based Partitioning

**Purpose**: Analyze data distribution, create balanced partitions based on statistics.

```prolog
% Pre-analyze data, choose partition boundaries for equal sizes
partitioner_init(
    histogram_based(
        strategy(equal_size),
        num_partitions(8)
    ),
    [],
    Handle
).
```

**Use Cases:**
- Skewed data (compensate for uneven distribution)
- Optimal load balancing
- Database query optimization

### 5. File-Based Partitioning (Hadoop-Style)

**Purpose**: Partition at file granularity, align with filesystem blocks.

```prolog
% Partition data across multiple files
partitioner_init(
    file_based(
        block_size(134217728),  % 128MB HDFS blocks
        output_dir('/data/partitions')
    ),
    [],
    Handle
).
```

**Benefits:**
- Data locality (process where data lives)
- Checkpoint/resume (can restart failed partitions)
- Storage efficiency (aligned to filesystem blocks)

### 6. Custom Partitioning Strategies

**Plugin Architecture**: Add your own strategies!

```prolog
% Define custom strategy module
:- module(my_partitioner, [
    strategy_init/2,
    strategy_partition/3,
    strategy_assign/3,
    strategy_cleanup/1
]).

% Register it
register_partitioner(my_strategy, my_partitioner).

% Use it
partitioner_init(my_strategy(custom_config), [], Handle).
```

---

## Summary

**Key Takeaways:**

1. **Partitioning enables parallelism and divide-and-conquer** - Two fundamental computational patterns

2. **Access patterns matter** - Memory-mapped (nanoGPT), file-based (Hadoop), hybrid (practical)

3. **Batch vs Streaming** - Trade-off between simplicity and efficiency
   - Batch: Scan all, partition, process (current)
   - Streaming: Partition on-the-fly, trigger pipelines (future)

4. **Three strategies implemented**:
   - `fixed_size` - Simple parallelism, predictable chunks
   - `hash_based` - MapReduce-compatible, deterministic assignment
   - `key_based` - GROUP BY semantics, natural grouping

5. **Plugin architecture** - Easy to add custom strategies

6. **GNU Parallel backend** - Battle-tested, handles complexity

**Next Steps:**

- Try the examples in `examples/test_partitioners.pl`
- Experiment with different strategies on your data
- Read design docs: `docs/proposals/partitioning_strategies.md`

**Related Chapters:**
- Chapter 5: Stream Compilation (how data flows through pipelines)
- Chapter 9: Advanced Recursion (divide-and-conquer patterns)
- Data Sources Guide: Loading data for processing

---

**License**: CC BY 4.0 (same as other UnifyWeaver education materials)
**Last Updated**: 2025-10-27
**Version**: 1.0
