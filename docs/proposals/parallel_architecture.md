# Proposal: Event-Driven Batch Parallel Architecture

**Status:** Partially Implemented (Phase 3)
**Version:** 1.1
**Date:** 2025-12-25
**Implementation Phase:** Phase 3 In Progress (v0.0.5)

### Implementation Summary
- ✅ GNU Parallel backend - Complete
- ✅ Bash Fork backend - Complete (no external deps)
- ✅ Dask Distributed backend - Complete
- ✅ Hadoop Streaming backend - Complete
- 🚧 Spark/PySpark backend - Planned

## Executive Summary

This proposal introduces an event-driven batch parallel architecture for UnifyWeaver that enables distributed processing of large datasets using industry-standard backends (Hadoop, Spark, Dask, GNU Parallel). The design decouples batching logic from execution backends, allowing seamless switching between local parallel execution and distributed systems without changing Prolog code.

**Key Innovation:** Prolog predicates define *what* to compute and *how to batch*, while a pluggable backend layer handles *where and how* to execute in parallel.

## Table of Contents

1. [Motivation](#motivation)
2. [Design Goals](#design-goals)
3. [Architecture Overview](#architecture-overview)
4. [Component Design](#component-design)
5. [Prolog DSL](#prolog-dsl)
6. [Event System](#event-system)
7. [Backend Abstraction](#backend-abstraction)
8. [Integration with Dynamic Sources](#integration-with-dynamic-sources)
9. [Implementation Phases](#implementation-phases)
10. [Examples](#examples)
11. [Performance Considerations](#performance-considerations)
12. [Open Questions](#open-questions)

---

## Motivation

### Current Limitations

UnifyWeaver v0.0.2 supports dynamic sources (CSV, JSON, HTTP, SQLite) but processes data sequentially. For large datasets or compute-intensive operations, this becomes a bottleneck:

- **Large CSV files:** Processing millions of rows sequentially is slow
- **HTTP scraping:** Sequential requests underutilize network bandwidth
- **Data joins:** Joining large datasets requires sequential iteration
- **Aggregations:** Computing statistics over large datasets is slow

### Target Use Cases

1. **Log Analysis:** Process gigabytes of log files in parallel
2. **Web Scraping:** Fetch and process thousands of URLs concurrently
3. **Data ETL:** Transform large CSV/JSON datasets using parallel pipelines
4. **Machine Learning:** Distribute feature extraction across compute cluster
5. **Report Generation:** Generate reports from distributed data sources

### Hadoop-Style Vision

The user's vision:
> "I want to introduce parallel type architectures (e.g. hadoop). My thought for this is you want some kind of batcher, than the batches are called as events by some parallel type system."

This aligns with MapReduce-style distributed computing but with a Prolog-first API.

---

## Design Goals

1. **Declarative:** Users specify batching/parallelism in Prolog, not bash
2. **Backend-Agnostic:** Same Prolog code runs on GNU Parallel, Hadoop, Spark, or Dask
3. **Event-Driven:** Batch processing emits events for monitoring and chaining
4. **Incremental Adoption:** Works with existing dynamic sources, no rewrite needed
5. **Observability:** Track batch progress, failures, and performance metrics
6. **Graceful Degradation:** Fall back to sequential execution if parallel backend unavailable

---

## Architecture Overview

### High-Level Data Flow

```
┌─────────────────┐
│  Prolog Query   │  user_age(User, Age), Age > 30
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dynamic Source  │  CSV/JSON/HTTP/SQL reader
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Batcher      │  Split data into chunks (size, count, time)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Event Queue    │  Emit: batch_ready, batch_complete, batch_failed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Worker Pool    │  Parallel execution (local or distributed)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Aggregator    │  Merge results from workers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Final Output   │  Return to Prolog query
└─────────────────┘
```

### Key Insight: Separation of Concerns

- **Batcher:** Knows how to split data (by size, count, time window)
- **Event Queue:** Coordinates workers, handles failures/retries
- **Workers:** Execute Prolog predicates on batch chunks
- **Backend:** Provides actual parallelism (threads, processes, cluster nodes)

This separation allows:
- Swapping GNU Parallel for Hadoop without changing batcher
- Testing with single-threaded mode for debugging
- Observing progress through event listeners

---

## Component Design

### 1. Batcher

**Responsibility:** Split input stream into manageable chunks

**Configuration Options:**
```prolog
% Batch by row count
:- batch_config(batch_size(1000)).

% Batch by data size (MB)
:- batch_config(batch_bytes(10_000_000)).

% Batch by time window (for streaming sources)
:- batch_config(batch_window(seconds(5))).

% Batch by key (group by first column)
:- batch_config(batch_key(1)).
```

**Implementation:**
- Reads from dynamic source output stream
- Buffers data until batch threshold reached
- Emits `batch_ready(BatchID, Data)` event
- Handles partial final batches

**Key Consideration:** Batching happens *after* dynamic source reads data but *before* parallel execution. This means CSV files are still read sequentially, but processing happens in parallel.

### 2. Event Queue

**Responsibility:** Coordinate batch lifecycle and worker communication

**Events:**
```prolog
% Batch is ready for processing
batch_ready(BatchID, Data, Metadata).

% Worker claimed a batch
batch_claimed(BatchID, WorkerID, Timestamp).

% Batch processing started
batch_started(BatchID, WorkerID, Timestamp).

% Batch completed successfully
batch_complete(BatchID, WorkerID, Results, Stats).

% Batch failed (with retry support)
batch_failed(BatchID, WorkerID, Error, RetryCount).

% All batches complete
batches_complete(TotalBatches, SuccessCount, FailureCount, AggregatedResults).
```

**Features:**
- **Queue Persistence:** Optional disk-based queue for crash recovery
- **Retry Logic:** Configurable retry on failure (exponential backoff)
- **Timeouts:** Detect stuck workers and reassign batches
- **Priority:** Support urgent batches (e.g., recent logs first)

**Implementation Notes:**
- For local execution: in-memory queue with FIFO semantics
- For Hadoop/Spark: queue maps to task scheduler
- For streaming: integrate with message queue (RabbitMQ, Kafka)

### 3. Worker Pool

**Responsibility:** Execute Prolog predicates on batch data in parallel

**Worker Types:**

1. **Local Threads** (GNU Parallel)
   - Spawn N bash processes
   - Each process sources compiled Prolog script
   - Communicate via pipes/files

2. **Distributed Processes** (Hadoop Streaming)
   - Map tasks run Prolog scripts
   - Reduce tasks aggregate results
   - HDFS for intermediate storage

3. **Spark Executors** (PySpark)
   - Python wrapper calls Prolog via subprocess
   - RDD/DataFrame partitioning maps to batches
   - Spark handles scheduling and fault tolerance

4. **Dask Workers** (Dask Distributed)
   - Similar to Spark but Python-native
   - Good for scientific computing workloads

**Worker Lifecycle:**
```
IDLE → CLAIMED → PROCESSING → COMPLETE → IDLE
                      ↓
                   FAILED → RETRY
```

**Configuration:**
```prolog
% Local parallel execution (4 workers)
:- parallel_config(backend(gnu_parallel), workers(4)).

% Hadoop streaming
:- parallel_config(backend(hadoop_streaming),
                   hadoop_opts(['-D', 'mapred.reduce.tasks=10'])).

% Spark cluster
:- parallel_config(backend(spark),
                   master('spark://cluster:7077'),
                   executor_memory('4g')).
```

### 4. Aggregator

**Responsibility:** Merge results from parallel workers into final output

**Aggregation Strategies:**

1. **Concatenation** (default)
   - Simply append all worker outputs
   - Preserves order if needed (via batch IDs)

2. **Set Union**
   - Remove duplicates across batches
   - Useful for deduplication tasks

3. **Reduce Operation**
   - Custom reduce function (sum, max, count, etc.)
   - Defined in Prolog

4. **MapReduce Pattern**
   - Map emits key-value pairs
   - Reduce groups by key and aggregates

**Example Configurations:**
```prolog
% Simple concatenation
:- aggregation(concat).

% Sum numeric results
:- aggregation(reduce(sum)).

% Count distinct values
:- aggregation(reduce(count_distinct)).

% Custom reducer
:- aggregation(reduce(my_custom_reducer/3)).

my_custom_reducer(Batch1, Batch2, Merged) :-
    append(Batch1, Batch2, Combined),
    sort(Combined, Merged).
```

---

## Prolog DSL

### Parallel Execution Hints

Users annotate predicates to enable parallel execution:

```prolog
% Mark predicate for parallel execution
:- parallel(process_logs/3, [
    batch_size(1000),
    backend(gnu_parallel),
    workers(8)
]).

process_logs(LogFile, Pattern, Result) :-
    log_entry(LogFile, Line),
    regex_match(Pattern, Line, Match),
    process_match(Match, Result).
```

### Batch Configuration

```prolog
% Configure batching for a dynamic source
:- dynamic_source(user_data/2, external(csv, 'users.csv'), [
    batch_size(5000),
    batch_parallel(true)
]).
```

### Backend Selection

```prolog
% Global backend setting
:- set_parallel_backend(gnu_parallel).

% Or per-predicate
:- parallel(expensive_computation/2, [
    backend(spark),
    executor_memory('8g')
]).
```

### Event Listeners

Users can register event handlers for monitoring:

```prolog
% Register event listener
:- on_event(batch_complete, log_batch_stats/1).

log_batch_stats(Event) :-
    Event = batch_complete(BatchID, WorkerID, Results, Stats),
    format('Batch ~w completed by worker ~w: ~w results~n',
           [BatchID, WorkerID, Stats]).
```

---

## Event System

### Event Bus Architecture

```
┌──────────────────────────────────────────────┐
│             Event Bus (Central)              │
├──────────────────────────────────────────────┤
│  - Publish/Subscribe pattern                 │
│  - Topic-based routing                       │
│  - Async delivery (non-blocking)             │
└──────────┬───────────────────────────────────┘
           │
    ┌──────┴──────┬──────────────┬─────────────┐
    ▼             ▼              ▼             ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│ Logger  │ │ Progress │ │ Metrics  │ │ User       │
│ Handler │ │ Bar      │ │ Collector│ │ Listeners  │
└─────────┘ └──────────┘ └──────────┘ └────────────┘
```

### Built-in Event Handlers

1. **Logger:** Write events to log file
2. **Progress Bar:** Display completion percentage
3. **Metrics Collector:** Track throughput, latency, errors
4. **Failure Handler:** Implement retry logic

### Event Schema

All events follow a standard schema:

```prolog
event(
    event_type,        % batch_ready, batch_complete, etc.
    timestamp,         % ISO 8601 timestamp
    source,            % Component that emitted event
    batch_id,          % Unique batch identifier
    payload            % Event-specific data
).
```

Example:
```prolog
event(
    batch_complete,
    '2025-10-26T14:32:15Z',
    worker_3,
    batch_00042,
    [
        results_count(823),
        processing_time_ms(1245),
        memory_mb(67)
    ]
).
```

---

## Backend Abstraction

### Backend Interface

All parallel backends implement a common interface:

```prolog
% Initialize backend
backend_init(Backend, Config, Handle).

% Submit batch for processing
backend_submit(Handle, BatchID, Data, Script, Result).

% Wait for completion
backend_wait(Handle, Timeout, Status).

% Shutdown backend
backend_shutdown(Handle).
```

### Supported Backends

#### 1. GNU Parallel (Local)

**Pros:**
- Simple, no cluster setup
- Fast for CPU-bound tasks
- Works on single machine

**Cons:**
- Limited to local cores
- No fault tolerance
- Limited scalability

**Implementation:**
```bash
# Generated parallel command
parallel --jobs 8 --pipe --block 10M \
  'source compiled_script.sh && process_batch' \
  < input_data.csv
```

#### 2. Hadoop Streaming

**Pros:**
- Industry-standard distributed computing
- Fault-tolerant (HDFS replication)
- Scales to thousands of nodes

**Cons:**
- Complex setup (requires Hadoop cluster)
- High latency for small jobs
- Overkill for small datasets

**Implementation:**
```bash
hadoop jar hadoop-streaming.jar \
  -input /hdfs/input/data.csv \
  -output /hdfs/output/results \
  -mapper compiled_mapper.sh \
  -reducer compiled_reducer.sh \
  -file compiled_script.sh
```

#### 3. Apache Spark

**Pros:**
- In-memory processing (faster than Hadoop)
- Rich API (SQL, DataFrames, streaming)
- Good for iterative algorithms

**Cons:**
- Requires Spark cluster
- More memory-intensive
- Steeper learning curve

**Implementation:**
```python
# PySpark wrapper for Prolog script
from pyspark import SparkContext

sc = SparkContext(master="spark://cluster:7077")
data = sc.textFile("hdfs://input/data.csv")

def process_batch(line):
    # Call compiled Prolog script via subprocess
    result = subprocess.check_output(
        ['bash', 'compiled_script.sh', 'process', line]
    )
    return result

results = data.map(process_batch).collect()
```

#### 4. Dask Distributed

**Pros:**
- Pure Python (easy integration)
- Familiar pandas-like API
- Good for data science workflows

**Cons:**
- Less mature than Spark
- Smaller community
- Limited non-Python integration

**Implementation:**
```python
# Dask DataFrame processing
import dask.dataframe as dd

df = dd.read_csv('data.csv')
results = df.map_partitions(lambda part:
    subprocess_call_prolog(part)
)
results.compute()
```

### Backend Selection Logic

```prolog
% Automatic backend selection based on data size
select_backend(DataSize, Backend) :-
    DataSize < 100_000 -> Backend = sequential ;
    DataSize < 10_000_000 -> Backend = gnu_parallel ;
    DataSize < 1_000_000_000 -> Backend = spark ;
    Backend = hadoop_streaming.
```

---

## Integration with Dynamic Sources

### Seamless Integration

Parallel execution integrates naturally with existing dynamic sources:

```prolog
% Existing CSV source (v0.0.2)
:- dynamic_source(user/2, external(csv, 'users.csv')).

% Add parallel execution (v0.0.4+)
:- dynamic_source(user/2, external(csv, 'users.csv'), [
    batch_parallel(true),
    batch_size(10000),
    backend(gnu_parallel),
    workers(4)
]).
```

### Streaming Sources

For HTTP/SQL sources, batching happens naturally:

```prolog
% HTTP API with pagination (batches = pages)
:- dynamic_source(api_user/2, external(http,
    'https://api.example.com/users?page={{page}}&size=100'
), [
    paginated(true),
    parallel_pages(8)  % Fetch 8 pages concurrently
]).
```

### Source-Specific Optimizations

Different sources have different batching strategies:

| Source Type | Batching Strategy | Parallelism |
|------------|-------------------|-------------|
| CSV | Row count / file size | Process batches in parallel |
| JSON | Array chunks | Process chunks in parallel |
| HTTP | Pagination / URL list | Concurrent requests |
| SQL | LIMIT/OFFSET ranges | Parallel queries with sharding |
| AWK | Split input file | Parallel awk processes |

---

## Implementation Phases

### Phase 1: Foundation (v0.0.3) ✅ COMPLETE
*Prerequisites before parallel execution*

- [x] Test infrastructure improvements (POST_RELEASE_TODO.md)
- [x] Fix module import conflicts (#5a)
- [x] PowerShell sequential execution issue (#5b)
- [x] Stabilize dynamic sources

### Phase 2: Local Parallel Backend (v0.0.4) ✅ COMPLETE
*Prove the concept with GNU Parallel*

- [x] Implement batcher component (`partitioner.pl`, `partitioners/`)
- [x] Event system (in-memory queue)
- [x] GNU Parallel backend (`backends/gnu_parallel.pl`)
- [x] Bash Fork backend (`backends/bash_fork.pl`) - no external deps
- [x] Basic aggregator (concat, sum)
- [x] Prolog DSL for batch hints
- [x] Integration tests with CSV sources

**Delivered:**
- Users can parallelize CSV processing with `:- parallel(pred/arity, [workers(N)])`
- Event logging shows batch progress
- Performance benchmarks vs sequential execution

### Phase 3: Distributed Backends (v0.0.5) 🚧 IN PROGRESS
*Add Hadoop, Dask, and Spark support*

- [x] Backend abstraction interface (`parallel_backend.pl`)
- [x] Backend auto-loader (`backend_loader.pl`)
- [x] Hadoop Streaming backend (`backends/hadoop_streaming.pl`)
- [x] Dask Distributed backend (`backends/dask_distributed.pl`)
- [ ] Spark backend (PySpark wrapper) - PLANNED
- [x] HDFS integration for large files (via Hadoop Streaming)
- [ ] Fault tolerance and retry logic - PARTIAL
- [ ] Cross-backend test suite - IN PROGRESS

**Current Status (December 2025):**
- Hadoop Streaming: MapReduce jobs with configurable mapper/reducer
- Dask: Threads, processes, and distributed schedulers
- Demo: `examples/demo_distributed_backends.pl`

**Remaining:**
- Same Prolog code runs on GNU Parallel, Hadoop, Dask, or Spark
- Automatic backend selection based on data size
- Documentation for cluster setup

### Phase 4: Advanced Features (v0.1.0)
*Production-ready parallel execution*

- [x] Dask backend (MOVED TO PHASE 3)
- [ ] Spark/PySpark backend
- [ ] Custom reduce functions
- [ ] Streaming sources (Kafka integration)
- [ ] Event persistence (crash recovery)
- [ ] Metrics dashboard
- [ ] Adaptive batching (auto-tune batch size)
- [ ] Cost optimization (spot instances, auto-scaling)

**Deliverables:**
- Production-grade parallel execution
- Real-time streaming support
- Comprehensive monitoring

---

## Examples

### Example 1: Parallel Log Analysis

**Task:** Count error patterns in 10GB log file

**Sequential (v0.0.2):**
```prolog
:- dynamic_source(log_entry/1, external(awk,
    "awk '{print $0}' /var/log/app.log"
)).

count_errors(Count) :-
    findall(1, (log_entry(Line), sub_string(Line, _, _, _, "ERROR")), Errors),
    length(Errors, Count).
```

**Parallel (v0.0.4+):**
```prolog
:- dynamic_source(log_entry/1, external(awk,
    "awk '{print $0}' /var/log/app.log"
), [
    batch_size(100000),     % Process 100k lines per batch
    backend(gnu_parallel),
    workers(8)
]).

% Aggregation happens automatically
count_errors(Count) :-
    findall(1, (log_entry(Line), sub_string(Line, _, _, _, "ERROR")), Errors),
    length(Errors, Count).
```

**Expected Speedup:** 6-7x on 8-core machine

### Example 2: Web Scraping with Parallel Requests

**Task:** Scrape product prices from 10,000 URLs

**Sequential:**
```prolog
:- dynamic_source(product_url/1, external(csv, 'urls.csv')).

scrape_prices :-
    product_url(URL),
    http_get(URL, HTML),
    extract_price(HTML, Price),
    format('~w: $~w~n', [URL, Price]),
    fail.
scrape_prices.
```

**Parallel:**
```prolog
:- dynamic_source(product_url/1, external(csv, 'urls.csv'), [
    batch_size(100),           % Process 100 URLs per batch
    backend(gnu_parallel),
    workers(20)                % 20 concurrent HTTP requests
]).

:- parallel(scrape_prices/0, [
    aggregation(concat)
]).

scrape_prices :-
    product_url(URL),
    http_get(URL, HTML),
    extract_price(HTML, Price),
    format('~w: $~w~n', [URL, Price]),
    fail.
scrape_prices.
```

**Expected Speedup:** 15-20x (network-bound)

### Example 3: MapReduce-Style Word Count

**Task:** Count word frequencies in large text corpus

**Parallel with Custom Reducer:**
```prolog
:- dynamic_source(document/2, external(csv, 'documents.csv')).

% Map phase: emit (Word, 1) for each word
:- parallel(word_count_map/2, [
    batch_size(1000),
    backend(hadoop_streaming),
    aggregation(reduce(word_count_reduce/3))
]).

word_count_map(Doc, word(Word, 1)) :-
    document(Doc, Text),
    split_string(Text, " \n\t", "", Words),
    member(WordStr, Words),
    string_lower(WordStr, Word).

% Reduce phase: sum counts for each word
word_count_reduce(word(W, C1), word(W, C2), word(W, C)) :-
    C is C1 + C2.

% Query
top_words(N, TopWords) :-
    findall(count(Count, Word), word_count_map(_, word(Word, Count)), Counts),
    sort(0, @>=, Counts, Sorted),
    take(N, Sorted, TopWords).
```

**Expected Speedup:** 50-100x on Hadoop cluster

---

## Performance Considerations

### Overhead vs Benefit

Parallel execution adds overhead:
- **Batching:** CPU cycles to split data
- **IPC:** Inter-process communication (pipes, files)
- **Aggregation:** Merging results from workers
- **Coordination:** Event queue management

**Rule of Thumb:** Only parallelize if processing time >> overhead

```
Speedup = T_sequential / (T_parallel + T_overhead)

Break-even point: T_sequential >= N * T_overhead
  where N = number of workers
```

### Optimal Batch Size

Too small: overhead dominates
Too large: poor load balancing

**Heuristic:**
```prolog
optimal_batch_size(DataSize, Workers, BatchSize) :-
    MinBatches is Workers * 4,  % At least 4 batches per worker
    BatchSize is max(1000, DataSize // MinBatches).
```

### Memory Management

Large batches can exhaust memory:

```prolog
% Limit batch memory
:- batch_config(max_batch_memory(100_000_000)).  % 100 MB
```

### Network Latency

For distributed backends (Hadoop, Spark):
- Minimize data shuffling
- Use compression for intermediate results
- Co-locate data and compute when possible

---

## Open Questions

### 1. Fault Tolerance Semantics

**Question:** If a batch fails, should we:
- Retry entire batch?
- Checkpoint partial progress?
- Mark batch as failed and continue?

**Proposal:** Configurable retry policy:
```prolog
:- retry_policy(max_attempts(3), backoff(exponential)).
```

### 2. Result Ordering

**Question:** Should parallel execution preserve input order?

**Trade-off:**
- Preserving order requires buffering (memory overhead)
- Out-of-order is faster but may confuse users

**Proposal:** Make it configurable:
```prolog
:- parallel_config(preserve_order(true)).  % Default: false
```

### 3. Nested Parallelism

**Question:** Can parallel predicates call other parallel predicates?

**Challenges:**
- Resource contention (too many workers)
- Deadlock potential
- Complexity in event tracking

**Proposal:** Initially forbid nested parallelism, revisit in Phase 4

### 4. Prolog Compilation for Workers

**Question:** Do workers need full Prolog runtime or just bash scripts?

**Options:**
1. Compile to bash (current approach) - workers are lightweight
2. Use SWI-Prolog runtime - more powerful but heavier

**Proposal:** Start with bash compilation, add Prolog runtime option in Phase 4 for complex predicates

### 5. Cost Optimization

**Question:** How to minimize cloud costs (EC2 spot instances, auto-scaling)?

**Ideas:**
- Integrate with cloud provider APIs (AWS, GCP)
- Use spot instances for fault-tolerant workloads
- Auto-scale workers based on queue depth

**Proposal:** Phase 4 feature

---

## Related Work

### Similar Systems

1. **Pig Latin (Apache Pig)**
   - High-level language for Hadoop
   - Similar declarative approach
   - More limited than Prolog

2. **Hive (SQL on Hadoop)**
   - SQL dialect for big data
   - Relation to Prolog: both declarative
   - Less flexible than Prolog

3. **Datalog Engines (BigDatalog, LogicBlox)**
   - Datalog = subset of Prolog
   - Focus on recursive queries
   - UnifyWeaver advantage: full Prolog + bash interop

4. **Luigi / Airflow (Workflow Orchestration)**
   - Task dependency management
   - Could integrate with UnifyWeaver events
   - More focused on DAG scheduling than parallelism

### Differentiation

UnifyWeaver's unique position:
- **Prolog-first:** Full Prolog logic, not just Datalog
- **Bash compilation:** No runtime overhead
- **Cross-platform:** Works on WSL, PowerShell, Cygwin
- **Gradual adoption:** Start sequential, add parallelism later
- **Lightweight:** No JVM required (unlike Spark/Hadoop)

---

## Success Criteria

This proposal is successful if:

1. **Performance:** 5-10x speedup on embarrassingly parallel workloads (8-core machine)
2. **Usability:** Users can add parallelism with 1-2 line config change
3. **Portability:** Same code runs on GNU Parallel and Hadoop
4. **Reliability:** Fault tolerance handles worker failures gracefully
5. **Observability:** Event logs provide clear insight into execution

---

## Appendix A: Implementation Checklist

### Batcher Component
- [ ] Row-based batching
- [ ] Size-based batching (bytes)
- [ ] Time-window batching (streaming)
- [ ] Key-based batching (grouping)
- [ ] Adaptive batch sizing

### Event System
- [ ] Event bus (pub/sub)
- [ ] Standard event schema
- [ ] Built-in handlers (logger, progress, metrics)
- [ ] User-defined event listeners
- [ ] Event persistence (optional)

### Worker Pool
- [ ] GNU Parallel backend
- [ ] Hadoop Streaming backend
- [ ] Spark backend
- [ ] Dask backend
- [ ] Worker health monitoring
- [ ] Timeout detection

### Aggregator
- [ ] Concatenation
- [ ] Set union
- [ ] Sum/count/avg
- [ ] Custom reduce functions
- [ ] MapReduce pattern

### Prolog DSL
- [ ] `:- parallel/2` directive
- [ ] `:- batch_config/1` directive
- [ ] `:- parallel_config/1` directive
- [ ] `:- on_event/2` directive
- [ ] `:- aggregation/1` directive

### Testing
- [ ] Unit tests for each component
- [ ] Integration tests (CSV, JSON, HTTP sources)
- [ ] Performance benchmarks
- [ ] Fault injection tests
- [ ] Cross-platform tests (WSL, PowerShell, Linux)

---

## Appendix B: References

- **Hadoop Streaming Documentation:** https://hadoop.apache.org/docs/stable/hadoop-streaming/HadoopStreaming.html
- **GNU Parallel Tutorial:** https://www.gnu.org/software/parallel/parallel_tutorial.html
- **Apache Spark Documentation:** https://spark.apache.org/docs/latest/
- **Dask Documentation:** https://docs.dask.org/
- **MapReduce Paper (Dean & Ghemawat, 2004):** https://research.google/pubs/pub62/

---

## Appendix C: Glossary

- **Batch:** A chunk of data processed as a unit
- **Batcher:** Component that splits data into batches
- **Event:** Notification about batch lifecycle (ready, complete, failed)
- **Worker:** Process/thread that executes batch processing
- **Backend:** Parallel execution engine (GNU Parallel, Hadoop, Spark)
- **Aggregator:** Component that merges results from workers
- **MapReduce:** Programming model for distributed computing
- **Fault Tolerance:** Ability to handle worker failures
- **HDFS:** Hadoop Distributed File System
- **RDD:** Resilient Distributed Dataset (Spark)

---

**End of Proposal**
