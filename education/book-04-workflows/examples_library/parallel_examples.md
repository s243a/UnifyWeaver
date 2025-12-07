# Parallel Execution Examples

This file contains examples of how to run parallel execution pipelines.

---

### Example: Parallel Sum (MapReduce Style)

This example demonstrates a full MapReduce-style pipeline to calculate a sum in parallel.

> [!example-record]
> id: 2025-11-13-parallel-sum
> name: unifyweaver.execution.parallel_sum_pipeline
> description: A full pipeline that partitions a list of numbers, sums each partition in parallel, and aggregates the results.

```bash
#!/bin/bash
#
# This script demonstrates a full MapReduce-style pipeline.
# It is designed to be extracted and run directly.

# --- Config ---
TMP_FOLDER="${TMP_FOLDER:-tmp}"
WORKER_SCRIPT="$TMP_FOLDER/sum_worker.sh"
PIPELINE_SCRIPT="$TMP_FOLDER/parallel_pipeline.pl"

# --- Step 1: Create Worker Script (The "Map" function) ---
echo "Creating worker script..."
mkdir -p "$TMP_FOLDER"
cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
sum=0
while IFS= read -r line; do
    if [[ "$line" =~ ^[0-9]+$ ]]; then
        sum=$((sum + line))
    fi
done
echo "$sum"
EOF
chmod +x "$WORKER_SCRIPT"

# --- Step 2: Create Prolog Orchestrator ---
echo "Creating Prolog orchestrator..."
cat > "$PIPELINE_SCRIPT" <<'EOF'
:- use_module('src/unifyweaver/core/partitioner').
:- use_module('src/unifyweaver/core/partitioners/fixed_size').
:- use_module('src/unifyweaver/core/parallel_backend').
:- use_module('src/unifyweaver/core/backends/bash_fork').

aggregate_sums(Results, TotalSum) :-
    findall(Sum,
            (   member(result(_, Output), Results),
                atom_string(Output, OutputStr),
                split_string(OutputStr, "\n", " \t\r", [SumStr|_]),
                number_string(Sum, SumStr)
            ),
            Sums),
    sum_list(Sums, TotalSum).

setup_system :-
    register_backend(bash_fork, bash_fork_backend),
    register_partitioner(fixed_size, fixed_size_partitioner).

run_pipeline :-
    setup_system,
    numlist(1, 1000, Numbers),
    partitioner_init(fixed_size(rows(100)), [], PHandle),
    partitioner_partition(PHandle, Numbers, Partitions),
    partitioner_cleanup(PHandle),
    backend_init(bash_fork(workers(4)), BHandle),
    backend_execute(BHandle, Partitions, 'tmp/sum_worker.sh', Results),
    backend_cleanup(BHandle),
    aggregate_sums(Results, TotalSum),
    ExpectedSum is 1000 * 1001 / 2,
    (   TotalSum =:= ExpectedSum ->
        format('SUCCESS: Final sum is ~w~n', [TotalSum])
    ;   format('FAILURE: Expected ~w but got ~w~n', [ExpectedSum, TotalSum]),
        halt(1)
    ).
EOF

# --- Step 3: Execute the Pipeline ---
echo "Executing pipeline..."
swipl -g "consult('$PIPELINE_SCRIPT'), run_pipeline, halt"

```