# Parallel Backend Implementation Plan

**Date:** 2025-10-26
**Status:** ✅ COMPLETE (2025-12-25)
**Related:** `parallel_architecture.md`, `partitioning_strategies.md`

> **Note:** This plan has been fully implemented. All backends (GNU Parallel, Bash Fork,
> Hadoop Streaming, Hadoop Native, Spark, Dask) are now available with full JVM language
> support. See `parallel_architecture.md` for current status.

---

## Overview

This document compares two approaches for implementing parallel execution backends and provides a unified implementation plan.

---

## Backend Comparison

### Option A: Simple Bash Fork Backend

**Complexity: MEDIUM**

**What needs to be implemented:**

1. **Process Spawning** (30 lines)
   ```bash
   spawn_worker() {
       local batch_file="$1"
       local output_file="$2"
       bash compiled_script.sh < "$batch_file" > "$output_file" 2>&1 &
       echo $!  # Return PID
   }
   ```

2. **Process Tracking** (40 lines)
   ```bash
   declare -A worker_pids    # PID -> batch_id
   declare -A worker_status  # PID -> running|completed|failed

   track_worker() {
       local pid="$1"
       local batch_id="$2"
       worker_pids[$pid]="$batch_id"
       worker_status[$pid]="running"
   }
   ```

3. **Wait for Completion** (50 lines)
   ```bash
   wait_for_slot() {
       # Wait until worker count < max_workers
       while [ ${#worker_pids[@]} -ge $max_workers ]; do
           # Check for completed workers
           for pid in "${!worker_pids[@]}"; do
               if ! kill -0 $pid 2>/dev/null; then
                   # Worker completed
                   wait $pid
                   local exit_code=$?
                   handle_worker_completion $pid $exit_code
                   unset worker_pids[$pid]
               fi
           done
           sleep 0.1  # Small delay to avoid busy loop
       done
   }
   ```

4. **Error Handling** (30 lines)
   ```bash
   handle_worker_completion() {
       local pid="$1"
       local exit_code="$2"
       local batch_id="${worker_pids[$pid]}"

       if [ $exit_code -eq 0 ]; then
           worker_status[$pid]="completed"
           echo "Batch $batch_id completed"
       else
           worker_status[$pid]="failed"
           echo "Batch $batch_id failed with code $exit_code"
       fi
   }
   ```

5. **Cleanup** (20 lines)
   ```bash
   cleanup_workers() {
       # Kill remaining workers
       for pid in "${!worker_pids[@]}"; do
           kill $pid 2>/dev/null || true
       done
       wait  # Wait for all to terminate
   }
   trap cleanup_workers EXIT
   ```

**Total Bash Code:** ~170 lines
**Prolog Backend Module:** ~150 lines
**Testing Complexity:** HIGH (need to test process management, edge cases)

**Pros:**
- ✅ No dependencies
- ✅ Full control
- ✅ Educational

**Cons:**
- ❌ Need to handle all edge cases manually
- ❌ More code to debug
- ❌ Potential issues with process cleanup

---

### Option B: GNU Parallel Backend

**Complexity: LOW**

**What needs to be implemented:**

1. **Command Generation** (30 lines)
   ```prolog
   generate_parallel_command(BatchFiles, NumWorkers, ScriptPath, Command) :-
       format(string(Command),
              'parallel --jobs ~w "bash ~w < {}" ::: ~w',
              [NumWorkers, ScriptPath, BatchFilesStr]).
   ```

2. **Execution** (20 lines)
   ```prolog
   execute_parallel(Command, Output) :-
       process_create('/bin/bash', ['-c', Command],
                      [stdout(pipe(Stream))]),
       read_string(Stream, _, Output),
       close(Stream).
   ```

3. **Result Collection** (40 lines)
   ```prolog
   collect_results(OutputFiles, Results) :-
       maplist(read_file_to_string, OutputFiles, Results).
   ```

**Total Bash Code:** 0 lines (GNU Parallel handles it)
**Prolog Backend Module:** ~90 lines
**Testing Complexity:** LOW (GNU Parallel is tested)

**Pros:**
- ✅ Very simple to implement
- ✅ Battle-tested tool
- ✅ Handles all edge cases
- ✅ Good performance

**Cons:**
- ❌ External dependency
- ❌ Need to check if installed

**Installation:**
```bash
# Ubuntu/Debian
sudo apt-get install parallel

# macOS
brew install parallel

# From source
wget https://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2
tar -xjf parallel-latest.tar.bz2
cd parallel-*
./configure && make && sudo make install
```

---

## Unified Backend Interface

Both backends implement the same interface:

```prolog
% Backend interface (all backends must implement)
backend_init(+Config, -Handle).
backend_execute(+Handle, +Partitions, +ScriptPath, -Results).
backend_cleanup(+Handle).
```

### Backend State Structure

```prolog
% Bash fork backend
Handle = handle(bash_fork, state(
    max_workers(4),
    temp_dir('/tmp/unifyweaver_12345'),
    worker_pids([]),
    batch_files([])
)).

% GNU Parallel backend
Handle = handle(gnu_parallel, state(
    num_jobs(4),
    temp_dir('/tmp/unifyweaver_12345'),
    parallel_path('/usr/bin/parallel')
)).
```

---

## Recommended Implementation Strategy

### Phase 1: Backend Interface (30 minutes)
**File:** `src/unifyweaver/core/parallel_backend.pl`

Create common interface that both backends implement:

```prolog
:- module(parallel_backend, [
    backend_init/2,
    backend_execute/4,
    backend_cleanup/1,
    register_backend/2,
    list_backends/1
]).

% Similar to partitioner.pl plugin system
```

**Complexity:** EASY (copy pattern from partitioner.pl)

---

### Phase 2: Choose Initial Backend

**Analysis:**

| Aspect | Bash Fork | GNU Parallel |
|--------|-----------|--------------|
| Implementation Time | 3-4 hours | 1-2 hours |
| Code Complexity | HIGH | LOW |
| Testing Effort | HIGH | LOW |
| Dependencies | None | parallel command |
| Maintenance | HIGH | LOW |
| Learning Value | HIGH | MEDIUM |

**Recommendation: Start with GNU Parallel (Option B)**

**Reasoning:**
1. **Much faster to implement** (1-2 hours vs 3-4 hours)
2. **Less code to debug** (90 lines vs 320 lines)
3. **Proven reliability** (GNU Parallel handles edge cases)
4. **Can add bash backend later** if needed as fallback
5. **Installation is easy** (one command)

---

### Phase 2: GNU Parallel Backend (1-2 hours)

**File:** `src/unifyweaver/core/backends/gnu_parallel.pl`

**Implementation Steps:**

1. **Detect GNU Parallel** (15 min)
   ```prolog
   check_parallel_installed :-
       process_create(path(parallel), ['--version'],
                      [stdout(null), stderr(null), process(PID)]),
       process_wait(PID, exit(0)).
   ```

2. **Generate Batch Files** (20 min)
   ```prolog
   write_batch_files(Partitions, TempDir, BatchFiles) :-
       maplist(write_partition_to_file(TempDir), Partitions, BatchFiles).

   write_partition_to_file(TempDir, partition(ID, Data), FilePath) :-
       format(atom(FilePath), '~w/batch_~w.txt', [TempDir, ID]),
       open(FilePath, write, Stream),
       maplist(writeln(Stream), Data),
       close(Stream).
   ```

3. **Build Parallel Command** (15 min)
   ```prolog
   build_parallel_command(BatchFiles, NumJobs, ScriptPath, Command) :-
       atomic_list_concat(BatchFiles, ' ', BatchFilesStr),
       format(string(Command),
              'parallel --jobs ~w --results ~/output_{#} "bash ~w < {}" ::: ~w',
              [NumJobs, ScriptPath, BatchFilesStr]).
   ```

4. **Execute** (20 min)
   ```prolog
   execute_parallel_command(Command, ExitCode) :-
       process_create('/bin/bash', ['-c', Command],
                      [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
       read_string(Out, _, StdOut),
       read_string(Err, _, StdErr),
       close(Out), close(Err),
       process_wait(PID, exit(ExitCode)),
       (ExitCode =:= 0 -> true ; format('Parallel error: ~w~n', [StdErr])).
   ```

5. **Collect Results** (15 min)
   ```prolog
   collect_output_files(TempDir, NumPartitions, Results) :-
       findall(FilePath,
               (between(1, NumPartitions, N),
                format(atom(FilePath), '~w/output_~w', [TempDir, N])),
               OutputFiles),
       maplist(read_file_to_string, OutputFiles, Results).
   ```

6. **Cleanup** (10 min)
   ```prolog
   cleanup_temp_files(TempDir) :-
       format(atom(Command), 'rm -rf ~w', [TempDir]),
       shell(Command).
   ```

**Total:** ~95 minutes (1.5 hours)

---

### Phase 3: Bash Fork Backend (Optional, 3-4 hours)

**File:** `src/unifyweaver/core/backends/bash_fork.pl`

Only implement if:
- GNU Parallel not available
- Need educational example
- Want fallback option

**Implementation Steps:**

1. **Generate Worker Script** (30 min)
   - Template for worker process
   - Process single batch
   - Report status

2. **Process Manager** (90 min)
   - Spawn workers
   - Track PIDs
   - Wait for slots
   - Handle failures

3. **Coordination** (60 min)
   - Queue management
   - Load balancing
   - Status reporting

4. **Testing** (60 min)
   - Edge cases (crashes, max processes)
   - Cleanup verification
   - Performance comparison

**Total:** ~4 hours

**Priority:** LOW (defer to later if needed)

---

## Integration with Partitioner

Both backends use the same integration pattern:

```prolog
% High-level API
parallel_execute(Predicate, Data, NumWorkers, Results) :-
    % 1. Partition data
    partitioner_init(hash_based(key(column(1)), num_partitions(NumWorkers)), [], PHandle),
    partitioner_partition(PHandle, Data, Partitions),
    partitioner_cleanup(PHandle),

    % 2. Compile predicate to bash script
    compile_to_bash(Predicate, [], ScriptPath),

    % 3. Execute in parallel
    backend_init(gnu_parallel(workers(NumWorkers)), BHandle),
    backend_execute(BHandle, Partitions, ScriptPath, Results),
    backend_cleanup(BHandle).
```

---

## Testing Plan

### Test 1: Simple Parallel Execution
```prolog
test_parallel_execution :-
    % Data: numbers 1-100
    numlist(1, 100, Data),

    % Predicate: double each number
    TestPred = (double(X, Y) :- Y is X * 2),

    % Execute in parallel (4 workers)
    parallel_execute(double/2, Data, 4, Results),

    % Verify: all numbers doubled
    length(Results, 100),
    member(result(1, 2), Results),
    member(result(100, 200), Results).
```

### Test 2: Load Balancing
```prolog
test_load_balancing :-
    % Create skewed data (90% in one partition)
    append(Uniform, Skewed, Data),

    % Use hash partitioning for balance
    parallel_execute(process/2, Data, 4, Results),

    % Measure execution time per partition
    verify_balanced_execution(Results).
```

### Test 3: Error Handling
```prolog
test_error_handling :-
    % Data with some invalid items
    Data = [valid(1), invalid(foo), valid(2)],

    % Execute with error handling
    parallel_execute(validate/2, Data, 2, Results),

    % Verify: valid items processed, errors reported
    member(error(invalid(foo), _), Results).
```

---

## Installation Instructions

### GNU Parallel Installation

**Ubuntu/Debian/WSL:**
```bash
sudo apt-get update
sudo apt-get install parallel
```

**Test Installation:**
```bash
parallel --version
# Should show: GNU parallel 20XXXXXX

# Test basic functionality
echo -e "1\n2\n3\n4\n5" | parallel echo "Processing {}"
```

**If not available via apt:**
```bash
# Install from source
cd /tmp
wget https://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2
tar -xjf parallel-latest.tar.bz2
cd parallel-*
./configure --prefix=$HOME/.local
make
make install

# Add to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Decision Matrix

### Recommendation: Implement GNU Parallel First

**Time Investment:**
- GNU Parallel: 1.5-2 hours
- Bash Fork: 4+ hours

**Risk:**
- GNU Parallel: LOW (proven tool)
- Bash Fork: MEDIUM (need to handle edge cases)

**Value:**
- GNU Parallel: HIGH (production-ready immediately)
- Bash Fork: MEDIUM (educational, fallback option)

**Conclusion:**
Start with GNU Parallel to get working parallel execution quickly. Add bash fork backend later if needed as a fallback or for educational purposes.

---

## Next Steps

1. ✅ Install GNU Parallel
2. ✅ Implement backend interface (30 min)
3. ✅ Implement GNU Parallel backend (1.5 hours)
4. ✅ Create integration tests (1 hour)
5. ✅ Bash fork backend
6. ✅ Hadoop Streaming backend
7. ✅ Hadoop Native backend (Java/Scala/Kotlin/Clojure)
8. ✅ Spark backend (PySpark + Java/Scala/Kotlin/Clojure)
9. ✅ Dask Distributed backend
10. ✅ JVM Glue with all language bridges

**Total Time to Working Parallel Execution:** ~3 hours

---

## Code Structure

```
src/unifyweaver/core/
├── parallel_backend.pl          (interface, ~100 lines)
└── backends/
    ├── gnu_parallel.pl          (GNU Parallel, ~150 lines)
    └── bash_fork.pl             (optional, ~300 lines)

examples/
└── test_parallel_execution.pl   (tests, ~200 lines)
```

---

**Conclusion:** GNU Parallel is significantly easier and faster to implement. Start there, add bash fork backend later if needed.
