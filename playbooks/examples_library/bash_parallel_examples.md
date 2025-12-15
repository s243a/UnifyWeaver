---
file_type: UnifyWeaver Example Library
---
# Bash Partitioning and Parallel Execution Examples

This file contains executable records for the bash parallel execution playbook.

## `unifyweaver.execution.bash_fork_basic`

> [!example-record]
> id: unifyweaver.execution.bash_fork_basic
> name: Bash Fork Backend Basic Usage
> platform: bash

This record demonstrates parallel execution using pure bash fork (no GNU Parallel required).

```bash
#!/bin/bash
# Bash Fork Backend Demo - Basic Usage
# Demonstrates parallel execution using pure bash fork (no GNU Parallel required)

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Bash Fork Backend Demo: Basic Usage ==="

# Create test directory
mkdir -p tmp/parallel_demo

# Create test processing script
cat > tmp/parallel_demo/process_batch.sh << 'SCRIPT'
#!/bin/bash
# Simple batch processor - uppercase each line
while IFS= read -r line; do
    echo "$line" | tr '[:lower:]' '[:upper:]'
    sleep 0.1  # Simulate some work
done
SCRIPT
chmod +x tmp/parallel_demo/process_batch.sh

# Create test data
cat > tmp/parallel_demo/data.txt << 'DATA'
hello world
foo bar baz
testing parallel
execution with bash
fork backend demo
prolog unifyweaver
data processing
streaming pipeline
DATA

echo "Created test data and processing script"

# Create Prolog script that uses bash_fork backend
cat > tmp/parallel_demo/demo_fork.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/backends/bash_fork').

main :-
    format("~n=== Initializing Bash Fork Backend ===~n~n"),

    % Initialize backend with 4 workers
    Config = [backend_args([workers(4)])],
    backend_init_impl(Config, State),

    % Create partitions from test data
    Partitions = [
        partition(0, ["hello world", "foo bar baz"]),
        partition(1, ["testing parallel", "execution with bash"]),
        partition(2, ["fork backend demo", "prolog unifyweaver"]),
        partition(3, ["data processing", "streaming pipeline"])
    ],

    format("~nExecuting partitions in parallel...~n"),

    % Execute processing script on all partitions
    ScriptPath = 'tmp/parallel_demo/process_batch.sh',
    backend_execute_impl(State, Partitions, ScriptPath, Results),

    format("~n=== Results from parallel execution ===~n"),
    forall(member(result(ID, Output), Results),
           format("Partition ~w:~n~w~n", [ID, Output])),

    % Cleanup
    format("~nCleaning up...~n"),
    backend_cleanup_impl(State),

    format("~n=== Bash fork demo complete ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to execute parallel processing..."
swipl tmp/parallel_demo/demo_fork.pl 2>&1

echo ""
echo "Success: Bash fork parallel demo complete"
```

## `unifyweaver.execution.gnu_parallel`

> [!example-record]
> id: unifyweaver.execution.gnu_parallel
> name: GNU Parallel Backend Demo
> platform: bash

This record demonstrates parallel execution using GNU Parallel (if installed).

```bash
#!/bin/bash
# GNU Parallel Backend Demo
# Demonstrates parallel execution using GNU Parallel (if installed)

set -euo pipefail
cd /root/UnifyWeaver

echo "=== GNU Parallel Backend Demo ==="

# Check if GNU Parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel is not installed. Skipping this demo."
    echo "Install with: apt-get install parallel"
    echo ""
    echo "Using bash fork fallback demonstration instead..."

    mkdir -p tmp/parallel_demo

    # Demonstrate manual parallel with bash
    cat > tmp/parallel_demo/manual_parallel.sh << 'SCRIPT'
#!/bin/bash
# Manual parallel execution using bash background jobs

process_item() {
    local item="$1"
    echo "Processing: $item"
    sleep 0.5
    echo "Done: $item -> $(echo "$item" | tr '[:lower:]' '[:upper:]')"
}

items=("task1" "task2" "task3" "task4")
pids=()

echo "Starting parallel tasks..."
for item in "${items[@]}"; do
    process_item "$item" &
    pids+=($!)
done

echo "Waiting for all tasks..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks completed"
SCRIPT
    chmod +x tmp/parallel_demo/manual_parallel.sh
    bash tmp/parallel_demo/manual_parallel.sh

    exit 0
fi

mkdir -p tmp/parallel_demo

# Create processing script
cat > tmp/parallel_demo/transform.sh << 'SCRIPT'
#!/bin/bash
input="$1"
echo "Transformed: $(echo "$input" | tr '[:lower:]' '[:upper:]')"
SCRIPT
chmod +x tmp/parallel_demo/transform.sh

echo ""
echo "GNU Parallel detected. Running parallel demo..."

# Create test items
echo -e "apple\nbanana\ncherry\ndate\nelderberry" > tmp/parallel_demo/items.txt

echo ""
echo "Input items:"
cat tmp/parallel_demo/items.txt

echo ""
echo "Running with GNU Parallel (4 jobs):"
cat tmp/parallel_demo/items.txt | parallel --jobs 4 bash tmp/parallel_demo/transform.sh {}

echo ""
echo "Success: GNU Parallel demo complete"
```

## `unifyweaver.execution.partitioning_strategies`

> [!example-record]
> id: unifyweaver.execution.partitioning_strategies
> name: Data Partitioning Strategies
> platform: bash

This record demonstrates different ways to partition data for parallel processing.

```bash
#!/bin/bash
# Data Partitioning Strategies Demo
# Demonstrates different ways to partition data for parallel processing

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Data Partitioning Strategies Demo ==="

mkdir -p tmp/partition_demo

# Create sample data (100 lines)
seq 1 100 > tmp/partition_demo/numbers.txt
echo "Created test data: 100 numbers"

echo ""
echo "=== Strategy 1: Round-Robin Partitioning ==="
echo "Distributes items evenly across N partitions"

# Round-robin into 4 partitions
for i in 0 1 2 3; do
    awk -v n=4 -v i="$i" 'NR % n == i' tmp/partition_demo/numbers.txt > "tmp/partition_demo/partition_rr_$i.txt"
done

echo "Partition sizes (round-robin):"
for f in tmp/partition_demo/partition_rr_*.txt; do
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done

echo ""
echo "=== Strategy 2: Block Partitioning ==="
echo "Splits data into contiguous blocks"

# Block partition into 4 parts (25 lines each)
split -l 25 -d tmp/partition_demo/numbers.txt tmp/partition_demo/partition_block_

echo "Partition sizes (block):"
for f in tmp/partition_demo/partition_block_*; do
    echo "  $(basename "$f"): $(wc -l < "$f") lines"
done

echo ""
echo "=== Strategy 3: Hash Partitioning ==="
echo "Distributes based on hash of key field"

# Create keyed data
cat > tmp/partition_demo/keyed_data.tsv << 'TSV'
user_001	Alice	Engineering
user_002	Bob	Sales
user_003	Charlie	Engineering
user_004	Diana	Marketing
user_005	Eve	Sales
user_006	Frank	Engineering
user_007	Grace	Marketing
user_008	Henry	Sales
TSV

# Hash partition by first field into 3 buckets
for i in 0 1 2; do
    awk -F'\t' -v n=3 -v i="$i" '
    function hash(s) {
        h = 0
        for(j=1; j<=length(s); j++) h = h * 31 + ord(substr(s,j,1))
        return h % n
    }
    function ord(c) {
        return index("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_", c)
    }
    hash($1) == i { print }
    ' tmp/partition_demo/keyed_data.tsv > "tmp/partition_demo/partition_hash_$i.tsv" 2>/dev/null || true
done

echo "Partition contents (hash by user_id):"
for f in tmp/partition_demo/partition_hash_*.tsv; do
    echo "  $(basename "$f"):"
    cat "$f" | sed 's/^/    /'
done

echo ""
echo "=== Strategy 4: Range Partitioning ==="
echo "Partitions by value ranges"

# Range partition numbers: 1-25, 26-50, 51-75, 76-100
awk '$1 <= 25' tmp/partition_demo/numbers.txt > tmp/partition_demo/partition_range_0.txt
awk '$1 > 25 && $1 <= 50' tmp/partition_demo/numbers.txt > tmp/partition_demo/partition_range_1.txt
awk '$1 > 50 && $1 <= 75' tmp/partition_demo/numbers.txt > tmp/partition_demo/partition_range_2.txt
awk '$1 > 75' tmp/partition_demo/numbers.txt > tmp/partition_demo/partition_range_3.txt

echo "Partition sizes (range):"
for f in tmp/partition_demo/partition_range_*.txt; do
    echo "  $(basename "$f"): $(wc -l < "$f") lines, range: $(head -1 "$f")-$(tail -1 "$f")"
done

echo ""
echo "Success: Partitioning strategies demo complete"
```

## `unifyweaver.execution.worker_pool`

> [!example-record]
> id: unifyweaver.execution.worker_pool
> name: Worker Pool Pattern
> platform: bash

This record demonstrates a simple worker pool for parallel task processing.

```bash
#!/bin/bash
# Worker Pool Pattern Demo
# Demonstrates a simple worker pool for parallel task processing

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Worker Pool Pattern Demo ==="

mkdir -p tmp/worker_demo

# Create the worker pool script
cat > tmp/worker_demo/worker_pool.sh << 'SCRIPT'
#!/bin/bash
# Simple worker pool implementation

MAX_WORKERS=${1:-4}
TASK_QUEUE="tmp/worker_demo/task_queue"
RESULT_DIR="tmp/worker_demo/results"

mkdir -p "$RESULT_DIR"
rm -f "$TASK_QUEUE"

# Track active workers
declare -A worker_pids

# Worker function
worker() {
    local worker_id=$1
    local task=$2
    local result_file="$RESULT_DIR/result_${worker_id}_$$.txt"

    echo "[Worker $worker_id] Processing: $task"
    sleep $(echo "scale=1; $RANDOM / 32768" | bc)  # Random 0-1s delay

    # Process task (uppercase transformation)
    echo "Task: $task -> Result: $(echo "$task" | tr '[:lower:]' '[:upper:]')" > "$result_file"
    echo "[Worker $worker_id] Completed: $task"
}

# Spawn workers for tasks
process_tasks() {
    local tasks=("$@")
    local active=0
    local task_idx=0

    while [ $task_idx -lt ${#tasks[@]} ] || [ $active -gt 0 ]; do
        # Spawn new workers if slots available
        while [ $active -lt $MAX_WORKERS ] && [ $task_idx -lt ${#tasks[@]} ]; do
            worker $task_idx "${tasks[$task_idx]}" &
            worker_pids[$!]=$task_idx
            ((active++))
            ((task_idx++))
        done

        # Wait for any worker to finish
        if [ $active -gt 0 ]; then
            for pid in "${!worker_pids[@]}"; do
                if ! kill -0 $pid 2>/dev/null; then
                    wait $pid 2>/dev/null || true
                    unset worker_pids[$pid]
                    ((active--))
                fi
            done
            sleep 0.1
        fi
    done

    # Final wait
    wait
}

# Define tasks
tasks=(
    "hello world"
    "foo bar"
    "test data"
    "parallel processing"
    "worker pool"
    "bash scripting"
    "unix pipeline"
    "stream processing"
)

echo "Starting worker pool with $MAX_WORKERS workers"
echo "Processing ${#tasks[@]} tasks..."
echo ""

process_tasks "${tasks[@]}"

echo ""
echo "=== All Results ==="
for f in "$RESULT_DIR"/result_*.txt; do
    cat "$f"
done
SCRIPT
chmod +x tmp/worker_demo/worker_pool.sh

echo "Running worker pool with 4 workers..."
echo ""
bash tmp/worker_demo/worker_pool.sh 4

echo ""
echo "Success: Worker pool demo complete"
```

## `unifyweaver.execution.parallel_module_info`

> [!example-record]
> id: unifyweaver.execution.parallel_module_info
> name: Parallel Execution Module Info
> platform: bash

This record displays parallel backend capabilities and configuration options.

```bash
#!/bin/bash
# Parallel Execution Module Information
# Shows parallel backend capabilities

set -euo pipefail
cd /root/UnifyWeaver

echo "=== Parallel Execution Backend Information ==="

echo ""
echo "=== Available Backends ==="
echo ""
echo "1. bash_fork (src/unifyweaver/core/backends/bash_fork.pl)"
echo "   - Pure bash implementation using background jobs (&)"
echo "   - No external dependencies"
echo "   - Job control with PID tracking"
echo "   - Configurable worker count"
echo ""
echo "2. gnu_parallel (src/unifyweaver/core/backends/gnu_parallel.pl)"
echo "   - Uses GNU Parallel tool"
echo "   - More features: progress, retries, logging"
echo "   - Requires: apt-get install parallel"

echo ""
echo "=== Backend API ==="
echo ""
echo "backend_init_impl(+Config, -State)"
echo "  Initialize backend with configuration"
echo "  Config: [backend_args([workers(N)])]"
echo ""
echo "backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)"
echo "  Execute script on partitions in parallel"
echo "  Partitions: [partition(ID, [Items...]), ...]"
echo "  Results: [result(ID, Output), ...]"
echo ""
echo "backend_cleanup_impl(+State)"
echo "  Clean up temporary files and resources"

echo ""
echo "=== Partitioning Strategies ==="
echo ""
echo "1. Round-Robin: Even distribution across partitions"
echo "   awk 'NR % N == i'"
echo ""
echo "2. Block: Contiguous chunks"
echo "   split -l LINES file prefix"
echo ""
echo "3. Hash: By key hash"
echo "   awk 'hash(key) % N == i'"
echo ""
echo "4. Range: By value ranges"
echo "   awk 'value >= LOW && value < HIGH'"

echo ""
echo "=== Usage Example ==="
echo ""
echo "% Initialize"
echo "Config = [backend_args([workers(4)])],"
echo "backend_init_impl(Config, State),"
echo ""
echo "% Create partitions"
echo "Partitions = ["
echo "    partition(0, [\"item1\", \"item2\"]),"
echo "    partition(1, [\"item3\", \"item4\"])"
echo "],"
echo ""
echo "% Execute"
echo "backend_execute_impl(State, Partitions, 'script.sh', Results),"
echo ""
echo "% Cleanup"
echo "backend_cleanup_impl(State)."

echo ""
echo "Success: Parallel module info displayed"
```

