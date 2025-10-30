:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% bash_fork.pl - Pure Bash fork-based parallel backend
% Uses bash background processes (&) and job control for parallel execution
% No external dependencies (no GNU Parallel required)

:- module(bash_fork_backend, [
    backend_init_impl/2,
    backend_execute_impl/4,
    backend_cleanup_impl/1
]).

:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module(library(lists)).

%% ============================================
%% BACKEND IMPLEMENTATION
%% ============================================

%% backend_init_impl(+Config, -State)
%  Initialize bash fork backend
%
%  Configuration options:
%  - backend_args([workers(N)]) - Number of parallel workers
%
%  @example Initialize with 4 workers
%    ?- backend_init_impl([backend_args([workers(4)])], State).
backend_init_impl(Config, State) :-
    % Extract configuration
    (   member(backend_args(Args), Config)
    ->  true
    ;   Args = []
    ),

    % Parse number of workers
    (   member(workers(NumWorkers), Args)
    ->  true
    ;   NumWorkers = 4  % Default: 4 workers
    ),

    % Validate
    (   integer(NumWorkers), NumWorkers > 0
    ->  true
    ;   throw(error(domain_error(positive_integer, NumWorkers),
                    context(backend_init_impl/2, 'workers must be positive integer')))
    ),

    % Create temporary directory for batch files and worker script
    create_temp_directory(TempDir),

    % Initialize state
    State = state(
        num_workers(NumWorkers),
        temp_dir(TempDir)
    ),

    format('[BashFork] Initialized: workers=~w, temp_dir=~w~n',
           [NumWorkers, TempDir]).

%% backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  Execute script on partitions in parallel using bash background jobs
%
%  @arg State Backend state from backend_init_impl/2
%  @arg Partitions List of partition(ID, Data) terms
%  @arg ScriptPath Path to bash script to execute
%  @arg Results List of result(PartitionID, Output) terms
backend_execute_impl(State, Partitions, ScriptPath, Results) :-
    State = state(num_workers(NumWorkers), temp_dir(TempDir)),

    length(Partitions, NumPartitions),
    format('[BashFork] Executing ~w partitions with ~w workers~n',
           [NumPartitions, NumWorkers]),

    % Write partition data to batch files
    write_batch_files(Partitions, TempDir, BatchFiles),
    format('[BashFork] Created ~w batch files~n', [NumPartitions]),

    % Generate parallel execution script
    generate_parallel_script(BatchFiles, NumWorkers, ScriptPath, TempDir, ParallelScript),
    format(atom(ParallelScriptPath), '~w/parallel_executor.sh', [TempDir]),
    write_file(ParallelScriptPath, ParallelScript),

    % Make script executable
    process_create('/bin/chmod', ['+x', ParallelScriptPath], []),

    % Execute parallel script
    format('[BashFork] Executing parallel script~n', []),
    execute_parallel_script(ParallelScriptPath, ExitCode),

    (   ExitCode =:= 0
    ->  format('[BashFork] Execution completed successfully~n', [])
    ;   format('[BashFork] Warning: Exit code ~w~n', [ExitCode])
    ),

    % Collect results from output files
    collect_results(Partitions, TempDir, Results),
    format('[BashFork] Collected ~w results~n', [NumPartitions]).

%% backend_cleanup_impl(+State)
%  Clean up bash fork backend resources
backend_cleanup_impl(state(_, temp_dir(TempDir))) :-
    format('[BashFork] Cleaning up temp directory: ~w~n', [TempDir]),
    (   exists_directory(TempDir)
    ->  delete_directory_and_contents(TempDir),
        format('[BashFork] Cleanup complete~n', [])
    ;   format('[BashFork] Temp directory already removed~n', [])
    ).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% create_temp_directory(-TempDir)
%  Create temporary directory for batch files
create_temp_directory(TempDir) :-
    % Generate unique directory name
    get_time(Timestamp),
    format(atom(TempDir), '/tmp/unifyweaver_bashfork_~w', [Timestamp]),
    make_directory(TempDir).

%% write_batch_files(+Partitions, +TempDir, -BatchFiles)
%  Write partition data to batch files
write_batch_files(Partitions, TempDir, BatchFiles) :-
    maplist(write_partition_file(TempDir), Partitions, BatchFiles).

write_partition_file(TempDir, partition(ID, Data), FilePath) :-
    format(atom(FilePath), '~w/batch_~w.txt', [TempDir, ID]),
    open(FilePath, write, Stream, [encoding(utf8)]),
    maplist(write_item(Stream), Data),
    close(Stream).

write_item(Stream, Item) :-
    (   atom(Item)
    ->  writeln(Stream, Item)
    ;   string(Item)
    ->  writeln(Stream, Item)
    ;   write_term(Stream, Item, [quoted(true), nl(true)])
    ).

%% generate_parallel_script(+BatchFiles, +NumWorkers, +ScriptPath, +TempDir, -Script)
%  Generate bash script for parallel execution using fork and job control
generate_parallel_script(BatchFiles, NumWorkers, ScriptPath, TempDir, Script) :-
    % Build batch file list
    maplist(format_batch_entry, BatchFiles, BatchEntries),
    atomic_list_concat(BatchEntries, '\n', BatchFilesList),

    % Generate script
    format(string(Script),
'#!/bin/bash
# Auto-generated parallel execution script using pure bash fork
# No external dependencies (no GNU Parallel required)

# Note: Don''''t use set -e here - we handle errors explicitly
set -o pipefail

# Configuration
MAX_WORKERS=~w
SCRIPT_PATH="~w"
OUTPUT_DIR="~w"

# Batch files (one per line)
BATCH_FILES=(
~w
)

# Worker tracking
declare -A worker_pids     # PID -> batch_index
declare -A worker_status   # PID -> running|completed|failed
active_workers=0

# Cleanup handler
cleanup() {
    echo "[BashFork] Cleaning up workers..." >&2
    # Kill all running workers
    for pid in "${!worker_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Spawn worker for batch file
spawn_worker() {
    local batch_file="$1"
    local batch_id="$2"
    local output_file="${OUTPUT_DIR}/output_${batch_id}.txt"

    # Execute in background
    bash "$SCRIPT_PATH" < "$batch_file" > "$output_file" 2>&1 &
    local pid=$!

    # Track worker
    worker_pids[$pid]=$batch_id
    worker_status[$pid]="running"
    ((active_workers++))

    echo "[BashFork] Started worker PID=$pid for batch $batch_id" >&2
    return 0
}

# Wait for a worker slot to become available
wait_for_slot() {
    while [ $active_workers -ge $MAX_WORKERS ]; do
        # Check for completed workers
        for pid in "${!worker_pids[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # Worker completed
                wait $pid
                exit_code=$?
                handle_worker_completion $pid $exit_code
            fi
        done
        sleep 0.1  # Small delay to avoid busy loop
    done
}

# Handle worker completion
handle_worker_completion() {
    local pid="$1"
    local exit_code="$2"
    local batch_id="${worker_pids[$pid]}"

    if [ "$exit_code" -eq 0 ] 2>/dev/null; then
        worker_status[$pid]="completed"
        echo "[BashFork] Batch $batch_id completed (PID=$pid)" >&2
    else
        worker_status[$pid]="failed"
        echo "[BashFork] Batch $batch_id failed with code $exit_code (PID=$pid)" >&2
    fi

    # Remove from tracking
    unset worker_pids[$pid]
    unset worker_status[$pid]
    ((active_workers--))
}

# Main execution loop
echo "[BashFork] Starting parallel execution with $MAX_WORKERS workers" >&2
echo "[BashFork] Total batches: ${#BATCH_FILES[@]}" >&2

batch_index=0
for batch_file in "${BATCH_FILES[@]}"; do
    # Wait for available slot
    wait_for_slot

    # Spawn worker for this batch
    spawn_worker "$batch_file" "$batch_index"
    ((batch_index++))
done

# Wait for all remaining workers to complete
echo "[BashFork] Waiting for remaining workers to complete..." >&2
while [ $active_workers -gt 0 ]; do
    for pid in "${!worker_pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            wait $pid
            exit_code=$?
            handle_worker_completion $pid $exit_code
        fi
    done
    sleep 0.1
done

echo "[BashFork] All batches completed" >&2
exit 0
', [NumWorkers, ScriptPath, TempDir, BatchFilesList]).

format_batch_entry(FilePath, Entry) :-
    format(atom(Entry), '"~w"', [FilePath]).

%% write_file(+FilePath, +Content)
%  Write content to file
write_file(FilePath, Content) :-
    open(FilePath, write, Stream, [encoding(utf8)]),
    write(Stream, Content),
    close(Stream).

%% execute_parallel_script(+ScriptPath, -ExitCode)
%  Execute the parallel bash script
execute_parallel_script(ScriptPath, ExitCode) :-
    process_create('/bin/bash', [ScriptPath],
                   [stdout(pipe(OutStream)),
                    stderr(pipe(ErrStream)),
                    process(PID)]),

    % Read output streams
    read_string(OutStream, _, StdOut),
    read_string(ErrStream, _, StdErr),
    close(OutStream),
    close(ErrStream),

    % Wait for completion
    process_wait(PID, exit(ExitCode)),

    % Log output if non-empty
    (   StdOut \= ""
    ->  format('[BashFork] stdout: ~w~n', [StdOut])
    ;   true
    ),
    (   StdErr \= ""
    ->  format('[BashFork] stderr: ~w~n', [StdErr])
    ;   true
    ).

%% collect_results(+Partitions, +TempDir, -Results)
%  Collect results from output files
collect_results(Partitions, TempDir, Results) :-
    maplist(collect_partition_result(TempDir), Partitions, Results).

collect_partition_result(TempDir, partition(ID, _Data), result(ID, Output)) :-
    format(atom(OutputFile), '~w/output_~w.txt', [TempDir, ID]),

    (   exists_file(OutputFile)
    ->  read_file_to_string(OutputFile, Output, [])
    ;   % File doesn't exist, return empty
        Output = ""
    ).
