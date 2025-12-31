# Playbook: Bash Partitioning and Parallel Execution

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to partition data and execute processing scripts in parallel.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "bash_parallel" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use bash parallel"


## Workflow Overview
Use UnifyWeaver's parallel execution backends to:
1. Partition input data for parallel processing
2. Execute bash scripts across partitions using worker pools
3. Collect and merge results from parallel workers
4. Choose between bash_fork (no dependencies) and GNU Parallel backends

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** - `playbooks/examples_library/bash_parallel_examples.md`
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step 1: Navigate to project root
```bash
cd /root/UnifyWeaver
```

### Step 2: Extract the bash fork demo
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.bash_fork_basic" \
  playbooks/examples_library/bash_parallel_examples.md \
  > tmp/run_bash_fork.sh
```

### Step 3: Make it executable and run
```bash
chmod +x tmp/run_bash_fork.sh
bash tmp/run_bash_fork.sh
```

**Expected Output**:
```
=== Bash Fork Backend Demo: Basic Usage ===

Created test data and processing script

Running Prolog to execute parallel processing...

=== Initializing Bash Fork Backend ===

[BashFork] Initialized: workers=4, temp_dir=tmp/unifyweaver_bashfork_...

Executing partitions in parallel...
[BashFork] Executing 4 partitions with 4 workers
[BashFork] Created 4 batch files
[BashFork] Executing parallel script
[BashFork] Started worker PID=... for batch 0
[BashFork] Started worker PID=... for batch 1
...
[BashFork] All batches completed

=== Results from parallel execution ===
Partition 0:
HELLO WORLD
FOO BAR BAZ

Partition 1:
TESTING PARALLEL
EXECUTION WITH BASH
...

Success: Bash fork parallel demo complete
```

### Step 4: Test partitioning strategies (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.partitioning_strategies" \
  playbooks/examples_library/bash_parallel_examples.md \
  > tmp/run_partitioning.sh
chmod +x tmp/run_partitioning.sh
bash tmp/run_partitioning.sh
```

### Step 5: Test worker pool pattern (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.worker_pool" \
  playbooks/examples_library/bash_parallel_examples.md \
  > tmp/run_worker_pool.sh
chmod +x tmp/run_worker_pool.sh
bash tmp/run_worker_pool.sh
```

### Step 6: View module info (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.parallel_module_info" \
  playbooks/examples_library/bash_parallel_examples.md \
  > tmp/run_parallel_info.sh
chmod +x tmp/run_parallel_info.sh
bash tmp/run_parallel_info.sh
```

## What This Playbook Demonstrates

1. **bash_fork backend** (`src/unifyweaver/core/backends/bash_fork.pl`):
   - Pure bash implementation using background jobs (`&`)
   - No external dependencies
   - Job control with PID tracking
   - Configurable worker count

2. **gnu_parallel backend** (`src/unifyweaver/core/backends/gnu_parallel.pl`):
   - Uses GNU Parallel tool
   - More features: progress bars, retries, logging
   - Requires: `apt-get install parallel`

3. **Backend API**:
   - `backend_init_impl(+Config, -State)` - Initialize backend
   - `backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)` - Execute in parallel
   - `backend_cleanup_impl(+State)` - Clean up resources

4. **Partitioning strategies**:
   - Round-robin: Even distribution
   - Block: Contiguous chunks
   - Hash: By key hash
   - Range: By value ranges

## Partitioning Strategies

### Round-Robin (even distribution):
```bash
# Split into N partitions
for i in $(seq 0 $((N-1))); do
    awk -v n=$N -v i=$i 'NR % n == i' data.txt > partition_$i.txt
done
```

### Block (contiguous chunks):
```bash
# Split into files of LINES_PER_PARTITION lines
split -l $LINES_PER_PARTITION data.txt partition_
```

### Hash (by key):
```bash
# Partition by hash of first field
awk -F'\t' -v n=$N 'BEGIN{srand()} {
    h = 0
    for(i=1;i<=length($1);i++) h = (h*31 + ord(substr($1,i,1))) % n
    print > "partition_" h ".txt"
}'
```

### Range (by value):
```bash
# Partition by value ranges
awk '$1 <= 25' data.txt > partition_0.txt
awk '$1 > 25 && $1 <= 50' data.txt > partition_1.txt
```

## Backend Configuration

### Bash Fork:
```prolog
Config = [backend_args([workers(4)])],
backend_init_impl(Config, State),

Partitions = [
    partition(0, ["item1", "item2"]),
    partition(1, ["item3", "item4"])
],

backend_execute_impl(State, Partitions, 'process.sh', Results),
backend_cleanup_impl(State).
```

### GNU Parallel:
```prolog
% Same API, different backend module
Config = [backend_args([workers(8)])],
gnu_parallel_backend:backend_init_impl(Config, State),
% ... execute and cleanup
```

## Common Mistakes to Avoid

- **DO NOT** run extracted scripts with `swipl` - they are bash scripts
- **DO** ensure partitions are roughly equal size for best performance
- **DO** handle worker failures gracefully
- **DO** clean up temporary files after processing

## Expected Outcome
- Parallel execution of bash scripts across partitions
- Configurable worker pool size
- Results collected from all partitions
- No external dependencies required (bash_fork)

## Citations
[1] playbooks/examples_library/bash_parallel_examples.md
[2] src/unifyweaver/core/backends/bash_fork.pl
[3] src/unifyweaver/core/backends/gnu_parallel.pl
[4] skills/skill_unifyweaver_environment.md

