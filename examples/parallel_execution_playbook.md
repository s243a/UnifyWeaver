# Playbook: Parallel Execution Pipeline (MapReduce Style)

## Goal

Demonstrate UnifyWeaver's MapReduce-style parallel execution capabilities by partitioning a dataset, processing the partitions in parallel, and aggregating the results.

## Context

This playbook tests the project's advanced parallel execution features. UnifyWeaver provides modules for declaratively partitioning data and executing worker scripts across these partitions using different parallel backends. This is a powerful pattern for high-performance data processing.

> NOTE: All temporary artifacts live under `${TMP_FOLDER:-tmp}`. Run the playbook from the repo root and follow the TMP guidance in `docs/development/ai-skills/workflow_environment.md`.

This playbook is based on **Demo 1** from `examples/demo_partition_parallel.pl`.

## Strategy

1.  **Setup:** Create a temporary Prolog file and ensure all necessary modules (`partitioner`, `parallel_backend`, etc.) are loaded.
2.  **Define Worker Logic:** Create a simple bash "worker" script that can process one partition of data (e.g., sum a list of numbers).
3.  **Orchestrate Pipeline:** Write a main Prolog predicate that:
    a.  Generates a dataset.
    b.  Uses the `partitioner` module to split the data into chunks.
    c.  Uses the `parallel_backend` module to execute the worker script on each chunk in parallel.
    d.  Uses an `aggregator` predicate to sum the results from the parallel workers.
    e.  Verifies the final result.
4.  **Execute and Verify:** Run the main Prolog script and confirm the final output is correct.

## Detailed Execution Steps (for Agent)

### Step 1: Create the Worker Script

Create a simple bash script that reads numbers from standard input and prints their sum.

**Save to file:**
```bash
TMP_FOLDER="${TMP_FOLDER:-tmp}"
mkdir -p "$TMP_FOLDER"
cat > "$TMP_FOLDER/sum_worker.sh" <<'EOF'
#!/bin/bash
# Sum all numbers from stdin
sum=0
while IFS= read -r line; do
    if [[ "$line" =~ ^[0-9]+$ ]]; then
        sum=$((sum + line))
    fi
done
echo "$sum"
EOF
chmod +x /tmp/sum_worker.sh
```

### Step 2: Create the Prolog Orchestration Script

Create a Prolog script that defines and runs the entire pipeline.

**Save to file:**
```bash
cat > "$TMP_FOLDER/parallel_pipeline.pl" <<'EOF'
:- use_module('src/unifyweaver/core/partitioner').
:- use_module('src/unifyweaver/core/partitioners/fixed_size').
:- use_module('src/unifyweaver/core/parallel_backend').
:- use_module('src/unifyweaver/core/backends/bash_fork').

% aggregate_sums(+Results, -TotalSum)
% Sums the integer results from the parallel workers.
aggregate_sums(Results, TotalSum) :-
    findall(Sum,
            (   member(result(_, Output), Results),
                atom_string(Output, OutputStr),
                split_string(OutputStr, "\n", " \t\r", [SumStr|_]),
                number_string(Sum, SumStr)
            ),
            Sums),
    sum_list(Sums, TotalSum).

% setup_system/0
% Registers the required backends and partitioners.
setup_system :-
    register_backend(bash_fork, bash_fork_backend),
    register_partitioner(fixed_size, fixed_size_partitioner).

% run_pipeline/0
% The main predicate that orchestrates the parallel execution.
run_pipeline :-
    % 0. Register components
    setup_system,

    % 1. Generate Data
    numlist(1, 1000, Numbers),

    % 2. Partition Data
    partitioner_init(fixed_size(rows(100)), [], PHandle),
    partitioner_partition(PHandle, Numbers, Partitions),
    partitioner_cleanup(PHandle),

    % 3. Execute in Parallel
    backend_init(bash_fork(workers(4)), BHandle),
    backend_execute(BHandle, Partitions, 'tmp/sum_worker.sh', Results),
    backend_cleanup(BHandle),

    % 4. Aggregate and Verify
    aggregate_sums(Results, TotalSum),
    ExpectedSum is 1000 * 1001 / 2,
    (   TotalSum =:= ExpectedSum ->
        format('SUCCESS: Final sum is ~w~n', [TotalSum])
    ;   format('FAILURE: Expected ~w but got ~w~n', [ExpectedSum, TotalSum]),
        halt(1)
    ).
EOF
```

### Step 3: Execute the Pipeline

Run the Prolog orchestration script.

```bash
swipl -g "consult('${TMP_FOLDER:-tmp}/parallel_pipeline.pl'), run_pipeline, halt"
```

## Verification

**Expected output:**
The command should execute without errors and print the following line to standard output:
```
SUCCESS: Final sum is 500500
```

The script should exit with status code 0.
