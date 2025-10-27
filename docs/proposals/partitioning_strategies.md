# Proposal: Pluggable Partitioning Strategies for Batcher

**Status:** Design Draft
**Version:** 1.0
**Date:** 2025-10-26
**Related:** `parallel_architecture.md` (Phase 2 - Batcher component)
**Proposed Implementation:** Phase 2a (before full parallel backend)

---

## Executive Summary

The batcher component in UnifyWeaver's parallel architecture needs flexible partitioning strategies to split data for parallel processing. This proposal introduces a **pluggable partitioning system** that supports multiple strategies, from simple size-based batching to sophisticated pivot-based partitioning (like quicksort).

**Key Insight:** Partitioning logic is reusable across:
- Parallel batch processing (Phase 2)
- Divide-and-conquer algorithms (future)
- MapReduce-style operations (Phase 3)
- Multi-target compilation (when adding new languages)

---

## Table of Contents

1. [Motivation](#motivation)
2. [Partitioning Strategies](#partitioning-strategies)
3. [Architecture](#architecture)
4. [Prolog DSL](#prolog-dsl)
5. [Implementation Plan](#implementation-plan)
6. [Examples](#examples)
7. [Integration Points](#integration-points)
8. [Future Extensions](#future-extensions)

---

## Motivation

### Current Batcher Limitations

From `parallel_architecture.md:126-151`, the current batcher design has simple partitioning:

```prolog
% Simple size-based batching
:- batch_config(batch_size(1000)).        % N rows per batch
:- batch_config(batch_bytes(10_000_000)). % Size in bytes
:- batch_config(batch_window(seconds(5))). % Time window
:- batch_config(batch_key(1)).            % Group by column
```

**Limitations:**
- No load balancing (batches may have uneven work)
- No data-aware partitioning (can't use pivot/median)
- No support for divide-and-conquer patterns
- No optimization for specific data distributions

### Use Cases Requiring Better Partitioning

1. **Skewed Data Distribution**
   - Example: Log files where 90% of entries are from one source
   - Problem: Size-based batching creates unbalanced workload
   - Solution: Pivot-based partitioning splits by data characteristics

2. **Range Queries**
   - Example: "Process users with age 20-30, 31-40, 41-50"
   - Problem: Can't express range-based batching
   - Solution: Range partitioning strategy

3. **Hash-Based Distribution (MapReduce)**
   - Example: "Group by user_id for aggregation"
   - Problem: Need consistent hashing for reduce phase
   - Solution: Hash partitioning strategy

4. **Divide-and-Conquer Algorithms**
   - Example: Quicksort, mergesort (future when adding tree support)
   - Problem: Recursive partitioning based on data properties
   - Solution: Pivot/median partitioning strategy

5. **Hadoop Compatibility**
   - Hadoop's MapReduce uses hash partitioning for reducer assignment
   - Need compatible partitioning for seamless integration

---

## Partitioning Strategies

### Strategy Taxonomy

```
Partitioning Strategies
│
├── Static (data-independent)
│   ├── Fixed Size (N rows)
│   ├── Fixed Bytes (M bytes)
│   ├── Time Window (T seconds)
│   └── Round Robin (cycling assignment)
│
├── Content-Based (data-dependent)
│   ├── Key-Based (group by column value)
│   ├── Range (split by value ranges)
│   ├── Hash (consistent hashing)
│   └── Pivot (quicksort-style partitioning)
│
└── Adaptive (learns from data)
    ├── Histogram-Based (balanced distribution)
    ├── Sample-Based (sample data to determine splits)
    └── Load-Aware (balance worker load)
```

### 1. Fixed Size Partitioning (Current)

**Description:** Split by row count or byte size

**Configuration:**
```prolog
:- partition_strategy(fixed_size(rows(1000))).
:- partition_strategy(fixed_size(bytes(10_000_000))).
```

**Characteristics:**
- ✅ Simple, predictable
- ✅ Works for uniform data
- ❌ May create imbalanced workload
- ❌ Ignores data distribution

**Use Cases:** Simple parallel processing, uniform datasets

---

### 2. Key-Based Partitioning

**Description:** Group data by column value (existing `batch_key`)

**Configuration:**
```prolog
% Partition by first column value
:- partition_strategy(key_based(column(1))).

% Partition by extracted field
:- partition_strategy(key_based(
    extract(user_id, 'awk -F: {print $2}')
)).
```

**Characteristics:**
- ✅ Groups related data together
- ✅ Good for aggregation
- ❌ May create very unbalanced partitions (skewed keys)
- ❌ Number of partitions = number of distinct keys (unbounded)

**Use Cases:** GROUP BY operations, join co-partitioning

---

### 3. Hash Partitioning (MapReduce Standard)

**Description:** Hash key value to determine partition (modulo N)

**Configuration:**
```prolog
% Hash first column, split into 8 partitions
:- partition_strategy(hash_based(
    key(column(1)),
    num_partitions(8),
    hash_function(murmur3)  % murmur3, md5, or simple_mod
)).
```

**Algorithm:**
```prolog
assign_partition(Key, NumPartitions, Partition) :-
    hash(Key, HashValue),
    Partition is HashValue mod NumPartitions.
```

**Characteristics:**
- ✅ Balanced distribution (with good hash function)
- ✅ Fixed number of partitions
- ✅ Compatible with Hadoop/Spark
- ✅ Deterministic (same key → same partition)
- ❌ Related keys may be split across partitions

**Use Cases:** MapReduce, distributed joins, parallel aggregation

**Hadoop Compatibility:**
```bash
# Hadoop uses hash partitioning by default
# UnifyWeaver can generate compatible partitioning:
awk -F: '{print $1 "\t" $0}' | \
  sort -k1,1 | \
  hadoop jar streaming.jar \
    -partitioner org.apache.hadoop.mapred.lib.HashPartitioner \
    -numReduceTasks 8 \
    ...
```

---

### 4. Range Partitioning

**Description:** Split data into ranges (e.g., age 0-20, 21-40, 41+)

**Configuration:**
```prolog
% Explicit ranges
:- partition_strategy(range_based(
    key(column(2)),  % Age column
    ranges([
        range(0, 20),
        range(21, 40),
        range(41, 60),
        range(61, infinity)
    ])
)).

% Automatic range splitting (equal-width)
:- partition_strategy(range_based(
    key(column(2)),
    num_partitions(4),
    min_value(0),
    max_value(100)
)).
% Creates: [0-25), [25-50), [50-75), [75-100]
```

**Characteristics:**
- ✅ Preserves order within partitions
- ✅ Good for range queries
- ✅ Fixed number of partitions
- ❌ Requires knowing data distribution (min/max)
- ❌ May be unbalanced if data is skewed

**Use Cases:** Time-series data, age ranges, sorted data processing

---

### 5. Pivot-Based Partitioning (Quicksort-Style)

**Description:** Choose pivot, split into less-than and greater-or-equal partitions

**Configuration:**
```prolog
% Pivot-based partitioning
:- partition_strategy(pivot_based(
    key(column(1)),           % Partition by first column
    pivot_selection(median),  % median, random, or first
    max_depth(3)              % Limit recursive depth
)).

% For divide-and-conquer algorithms
:- partition_strategy(pivot_based(
    key(column(1)),
    pivot_selection(sample(100)),  % Sample 100 elements, use median
    recursive(true),                % Enable recursive partitioning
    base_case(size(100))           % Stop when partition < 100 elements
)).
```

**Algorithm:**
```prolog
pivot_partition(Data, Key, Pivot, Less, Greater) :-
    partition(Data, is_less_than(Key, Pivot), Less, Greater).

is_less_than(Key, Pivot, Row) :-
    extract_key(Row, Key, Value),
    Value < Pivot.
```

**Characteristics:**
- ✅ Adapts to data distribution
- ✅ Can be recursively applied (divide-and-conquer)
- ✅ Naturally balanced with good pivot selection
- ❌ Requires reading data to select pivot (2-pass)
- ❌ Variable number of partitions
- ❌ More complex than fixed-size

**Use Cases:** Quicksort, load balancing, adaptive partitioning

**Pivot Selection Strategies:**

1. **First Element** (simple but risky)
   ```prolog
   select_pivot(first, [First|_], First).
   ```

2. **Random Element** (better average case)
   ```prolog
   select_pivot(random, Data, Pivot) :-
       length(Data, Len),
       random_between(0, Len, Idx),
       nth0(Idx, Data, Pivot).
   ```

3. **Median of Sample** (best balance, requires sampling)
   ```prolog
   select_pivot(median, Data, Pivot) :-
       length(Data, Len),
       SampleSize is min(100, Len),
       random_sample(Data, SampleSize, Sample),
       median(Sample, Pivot).
   ```

4. **Median-of-Three** (good compromise)
   ```prolog
   select_pivot(median_of_three, Data, Pivot) :-
       length(Data, Len),
       First is 0,
       Middle is Len // 2,
       Last is Len - 1,
       nth0(First, Data, V1),
       nth0(Middle, Data, V2),
       nth0(Last, Data, V3),
       median([V1, V2, V3], Pivot).
   ```

**Connection to Quicksort:**

Quicksort IS pivot-based partitioning applied recursively:

```prolog
% Quicksort using pivot partitioning
quicksort([], []).
quicksort([Pivot|Rest], Sorted) :-
    % This IS pivot partitioning
    partition(Rest, <(Pivot), Less, Greater),

    % Recursive partitioning
    quicksort(Less, SortedLess),
    quicksort(Greater, SortedGreater),

    % Merge results
    append(SortedLess, [Pivot|SortedGreater], Sorted).
```

**For Parallel Processing:**

The batcher can use pivot partitioning to create balanced work:

```prolog
% Batcher with pivot partitioning
batch_data(Data, Batches) :-
    select_pivot(median, Data, Pivot),
    pivot_partition(Data, Pivot, Less, Greater),

    % Create two batches (could recurse further)
    Batches = [batch(1, Less), batch(2, Greater)].
```

This creates balanced batches even with skewed data!

---

### 6. Histogram-Based Partitioning (Adaptive)

**Description:** Sample data, build histogram, create equal-height buckets

**Configuration:**
```prolog
:- partition_strategy(histogram_based(
    key(column(1)),
    num_partitions(8),
    sample_size(10000),      % Sample 10k rows
    bucket_method(equal_height)  % equal_height or equal_width
)).
```

**Algorithm:**
```
1. Sample data (e.g., 10,000 rows)
2. Build histogram of key values
3. Determine bucket boundaries for equal distribution
4. Assign data to buckets based on boundaries
```

**Characteristics:**
- ✅ Best balance for skewed data
- ✅ Adapts to data distribution
- ✅ Fixed number of partitions
- ❌ Requires 2-pass (sample + partition)
- ❌ More computational overhead

**Use Cases:** Large skewed datasets, load balancing

---

## Architecture

### Partitioner Interface

All partitioning strategies implement a common interface:

```prolog
% Initialize partitioner with configuration
partitioner_init(Strategy, Config, Handle).

% Partition a data stream
partitioner_partition(Handle, DataStream, Partitions).

% Get partition for a single item (for streaming)
partitioner_assign(Handle, Item, PartitionID).

% Cleanup
partitioner_cleanup(Handle).
```

### Partitioner Lifecycle

```
INIT → SAMPLE (optional) → PARTITION → CLEANUP
         ↓
    Build Histogram /
    Select Pivot /
    Calculate Ranges
```

### Integration with Batcher

```
┌─────────────────┐
│ Dynamic Source  │ → produces data stream
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Partitioner    │ → applies strategy, emits partitions
├─────────────────┤
│ - Strategy      │ (fixed_size, hash, pivot, etc.)
│ - Config        │ (num_partitions, key, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Batcher     │ → wraps partitions as batches
├─────────────────┤
│ - Batch ID      │
│ - Metadata      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Event Queue    │ → emits batch_ready events
└─────────────────┘
```

**Key Insight:** Partitioner is a **plugin** for the batcher, not a replacement.

---

## Prolog DSL

### Strategy Selection

```prolog
% Global default partitioning
:- set_partition_strategy(hash_based(
    key(column(1)),
    num_partitions(8)
)).

% Per-predicate partitioning
:- dynamic_source(user_data/2, external(csv, 'users.csv'), [
    partition_strategy(pivot_based(
        key(column(3)),  % Partition by age
        pivot_selection(median)
    )),
    parallel(true)
]).
```

### Partitioning Hints

```prolog
% Provide hints for better partitioning
:- partition_hints(user_data/2, [
    data_distribution(skewed),       % Hint: data is skewed
    key_cardinality(high),           % Many distinct keys
    value_range(0, 100),             % Age range 0-100
    prefer_strategy(histogram_based) % Suggest histogram
]).
```

### Composing Strategies

```prolog
% First partition by hash, then by size within each hash bucket
:- partition_strategy(composite([
    hash_based(key(column(1)), num_partitions(4)),
    fixed_size(rows(1000))
])).
```

---

## Implementation Plan

### Phase 2a: Foundation (Before Full Parallel Backend)

**Goal:** Implement partitioner infrastructure and basic strategies

**Tasks:**
1. Design partitioner interface
2. Implement fixed-size partitioning (current behavior)
3. Implement hash partitioning (MapReduce compatibility)
4. Implement key-based partitioning (GROUP BY)
5. Add partitioner configuration to Prolog DSL
6. Unit tests for each strategy

**Deliverables:**
- `src/unifyweaver/core/partitioner.pl` - Core interface
- `src/unifyweaver/core/partitioners/` - Strategy implementations
  - `fixed_size.pl`
  - `hash_based.pl`
  - `key_based.pl`
- Tests and documentation

**Timeline:** 1-2 weeks

### Phase 2b: Advanced Strategies

**Goal:** Add adaptive and data-aware partitioning

**Tasks:**
1. Implement range partitioning
2. Implement pivot-based partitioning
3. Implement histogram-based partitioning (with sampling)
4. Add pivot selection strategies (median, random, median-of-three)
5. Performance benchmarks

**Deliverables:**
- `partitioners/range_based.pl`
- `partitioners/pivot_based.pl`
- `partitioners/histogram_based.pl`
- Benchmarks showing balanced workload

**Timeline:** 2-3 weeks

### Phase 3: Integration with Parallel Backend

**Goal:** Use partitioning in actual parallel execution

**Tasks:**
1. Integrate partitioner with batcher
2. Test with GNU Parallel backend
3. Test with Hadoop backend (hash partitioning compatibility)
4. Add partitioning metrics to event system
5. Performance tuning

**Timeline:** 1-2 weeks

---

## Examples

### Example 1: Hash Partitioning for MapReduce

**Problem:** Word count on large text corpus

```prolog
% Partition words by hash for parallel counting
:- dynamic_source(document/2, external(csv, 'documents.csv'), [
    partition_strategy(hash_based(
        key(extract_first_word),
        num_partitions(8),
        hash_function(murmur3)
    )),
    parallel(true),
    backend(gnu_parallel)
]).

% Each partition processes subset of words
word_count(Word, Count) :-
    document(_, Text),
    split_string(Text, " ", "", Words),
    member(Word, Words),
    % Aggregation happens per partition
    findall(1, member(Word, Words), Ones),
    length(Ones, Count).
```

**Result:** Words with same hash go to same partition, enabling local aggregation

---

### Example 2: Pivot Partitioning for Load Balancing

**Problem:** Log analysis where 90% of logs are INFO level

**Without Pivot Partitioning:**
```prolog
% Simple size-based batching
:- partition_strategy(fixed_size(rows(10000))).

% Problem: Most batches are all INFO logs (fast processing)
%          Few batches have ERROR logs (slow processing)
% Result: Unbalanced worker load
```

**With Pivot Partitioning:**
```prolog
% Pivot on log level
:- partition_strategy(pivot_based(
    key(log_level),          % Partition by log level
    pivot_selection(value('INFO'))  % Pivot = 'INFO'
)).

% Creates two partitions:
% - Partition 1: INFO logs (90% of data)
% - Partition 2: WARN/ERROR logs (10% of data)

% Further partition large INFO partition
:- partition_strategy(composite([
    pivot_based(key(log_level), pivot_selection(value('INFO'))),
    % INFO partition gets split by time
    range_based(key(timestamp), num_partitions(8))
])).
```

**Result:** Balanced workload across workers

---

### Example 3: Range Partitioning for Time Series

**Problem:** Process web logs by time range

```prolog
:- dynamic_source(web_log/4, external(csv, 'access.log'), [
    partition_strategy(range_based(
        key(column(1)),  % Timestamp column
        ranges([
            range('2025-10-26T00:00:00', '2025-10-26T06:00:00'),
            range('2025-10-26T06:00:00', '2025-10-26T12:00:00'),
            range('2025-10-26T12:00:00', '2025-10-26T18:00:00'),
            range('2025-10-26T18:00:00', '2025-10-27T00:00:00')
        ])
    )),
    parallel(true)
]).

% Query specific time range efficiently
peak_traffic(Hour, Count) :-
    web_log(Timestamp, IP, Path, Status),
    extract_hour(Timestamp, Hour),
    Hour >= 12, Hour < 18,  % Peak hours
    findall(1, web_log(_, _, _, _), Requests),
    length(Requests, Count).
```

**Result:** Each worker processes one time range, naturally parallelized

---

## Integration Points

### With Existing Systems

1. **Dynamic Sources (v0.0.2)**
   - Partitioner reads from source output
   - Transparent to source plugins

2. **Batcher (parallel_architecture.md)**
   - Partitioner generates partitions
   - Batcher wraps as batches with IDs

3. **Event System**
   - Emit partitioning events: `partition_created`, `partition_assigned`
   - Track metrics: partition sizes, balance factor

4. **Hadoop/Spark**
   - Hash partitioning compatible with Hadoop's HashPartitioner
   - Custom partitioners can be Hadoop UDFs

### Configuration Hierarchy

```
Global Default → Firewall Policy → Dynamic Source → Per-Predicate
   (lowest)                                           (highest)
```

Example:
```prolog
% Global default
:- set_partition_strategy(fixed_size(rows(1000))).

% Override for specific source
:- dynamic_source(logs/2, external(csv, 'logs.csv'), [
    partition_strategy(pivot_based(key(log_level)))
]).
```

---

## Future Extensions

### 1. Multi-Level Partitioning (Recursive)

Enable recursive partitioning for divide-and-conquer:

```prolog
:- partition_strategy(recursive(
    pivot_based(key(column(1))),
    max_depth(3),
    base_case(size(100))
)).
```

**Use Case:** When you add tree support to bash, this enables quicksort compilation.

### 2. Adaptive Partitioning

Learn from execution metrics and adjust strategy:

```prolog
:- partition_strategy(adaptive(
    initial(hash_based(key(column(1)), num_partitions(8))),
    rebalance_threshold(imbalance(0.3)),  % >30% imbalance triggers rebalance
    rebalance_strategy(histogram_based)
)).
```

### 3. Cost-Based Partitioning

Use cost model to choose strategy:

```prolog
% System selects strategy based on:
% - Data size (small → fixed_size, large → histogram)
% - Key cardinality (low → key_based, high → hash)
% - Distribution (uniform → fixed_size, skewed → pivot)
:- partition_strategy(cost_based(
    cost_model(balance_vs_overhead),
    optimize_for(throughput)  % throughput or latency
)).
```

### 4. Integration with Tree Compilation

When bash supports trees (or when compiling to other languages):

```prolog
% Quicksort becomes a partitioning + tree recursion pattern
quicksort(List, Sorted) :-
    % Use pivot partitioning strategy
    partition_strategy(pivot_based(
        pivot_selection(median_of_three),
        recursive(true)
    )),
    % Compile as tree recursion
    compile_as(tree_recursion).
```

---

## Open Questions

### 1. Partitioning vs Sorting

**Question:** Should partitioner also sort within partitions?

**Trade-off:**
- Sorting enables merge (for mergesort)
- But adds overhead (O(n log n) per partition)
- Hadoop's shuffle phase does this

**Proposal:** Make sorting optional:
```prolog
:- partition_strategy(hash_based(...), [
    sort_within_partitions(true),
    sort_key(column(1))
]).
```

### 2. Partition Size Limits

**Question:** What if pivot partitioning creates huge imbalance?

**Options:**
1. Fallback to fixed-size if imbalance > threshold
2. Re-partition large partitions
3. Accept imbalance (user responsibility)

**Proposal:** Configurable fallback:
```prolog
:- partition_strategy(pivot_based(...), [
    max_imbalance_ratio(2.0),  % Max size = 2x average
    fallback(fixed_size(rows(1000)))
]).
```

### 3. Memory Constraints

**Question:** What if histogram/pivot selection needs too much memory?

**Solution:** Streaming algorithms:
- Use reservoir sampling (fixed memory)
- Use approximate quantiles (CountMin sketch)
- Spillover to disk if needed

---

## Summary

This partitioning system provides:

1. **Flexibility:** Multiple strategies for different use cases
2. **Extensibility:** Plugin architecture for new strategies
3. **Reusability:** Same code for parallel processing and divide-and-conquer
4. **Hadoop Compatibility:** Hash partitioning matches MapReduce
5. **Future-Proof:** Ready for tree recursion (quicksort) when available

**The key insight:** Partitioning is fundamental to both parallelism AND divide-and-conquer. By building a solid partitioning system now, you enable:
- Better parallel load balancing (Phase 2)
- MapReduce integration (Phase 3)
- Divide-and-conquer algorithms (when you add tree support)
- Multi-target compilation (PowerShell, Python, etc.)

**Next Step:** Implement Phase 2a (basic partitioner with fixed-size, hash, key-based strategies) before adding full parallel backend.

---

**End of Proposal**
