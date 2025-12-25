:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% dask_distributed.pl - Dask distributed backend for parallel execution
% Uses Python Dask for scalable parallel batch processing
% Supports both local threading and distributed cluster modes

:- module(dask_distributed_backend, [
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
%  Initialize Dask distributed backend
%
%  Configuration options:
%  - backend_args([workers(N)]) - Number of parallel workers
%  - backend_args([scheduler(S)]) - Scheduler type: synchronous | threads | processes | distributed
%  - backend_args([cluster_address(A)]) - Address for distributed scheduler (e.g., 'tcp://scheduler:8786')
%  - backend_args([memory_limit(M)]) - Memory limit per worker (e.g., '4GB')
%
%  @example Initialize with 4 workers using threads
%    ?- backend_init_impl([backend_args([workers(4), scheduler(threads)])], State).
%
%  @example Initialize with distributed cluster
%    ?- backend_init_impl([backend_args([scheduler(distributed), cluster_address('tcp://localhost:8786')])], State).
backend_init_impl(Config, State) :-
    % Check Python and Dask are available
    check_dask_installed,

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

    % Parse scheduler type
    (   member(scheduler(Scheduler), Args)
    ->  true
    ;   Scheduler = threads  % Default: threaded scheduler
    ),

    % Parse cluster address (for distributed mode)
    (   member(cluster_address(ClusterAddress), Args)
    ->  true
    ;   ClusterAddress = none
    ),

    % Parse memory limit per worker
    (   member(memory_limit(MemoryLimit), Args)
    ->  true
    ;   MemoryLimit = 'auto'
    ),

    % Validate workers
    (   integer(NumWorkers), NumWorkers > 0
    ->  true
    ;   throw(error(domain_error(positive_integer, NumWorkers),
                    context(backend_init_impl/2, 'workers must be positive integer')))
    ),

    % Validate scheduler
    valid_scheduler(Scheduler),

    % Create temporary directory for batch files and Python script
    create_temp_directory(TempDir),

    % Initialize state
    State = state(
        num_workers(NumWorkers),
        scheduler(Scheduler),
        cluster_address(ClusterAddress),
        memory_limit(MemoryLimit),
        temp_dir(TempDir)
    ),

    format('[Dask] Initialized: workers=~w, scheduler=~w, temp_dir=~w~n',
           [NumWorkers, Scheduler, TempDir]).

%% valid_scheduler(+Scheduler)
%  Validate scheduler type
valid_scheduler(synchronous) :- !.
valid_scheduler(threads) :- !.
valid_scheduler(processes) :- !.
valid_scheduler(distributed) :- !.
valid_scheduler(Scheduler) :-
    throw(error(domain_error(dask_scheduler, Scheduler),
                context(valid_scheduler/1,
                       'Scheduler must be one of: synchronous, threads, processes, distributed'))).

%% backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  Execute script on partitions in parallel using Dask
%
%  @arg State Backend state from backend_init_impl/2
%  @arg Partitions List of partition(ID, Data) terms
%  @arg ScriptPath Path to bash script to execute (will be wrapped)
%  @arg Results List of result(PartitionID, Output) terms
backend_execute_impl(State, Partitions, ScriptPath, Results) :-
    State = state(
        num_workers(NumWorkers),
        scheduler(Scheduler),
        cluster_address(ClusterAddress),
        memory_limit(MemoryLimit),
        temp_dir(TempDir)
    ),

    length(Partitions, NumPartitions),
    format('[Dask] Executing ~w partitions with ~w workers (scheduler: ~w)~n',
           [NumPartitions, NumWorkers, Scheduler]),

    % Write partition data to batch files
    write_batch_files(Partitions, TempDir, BatchFiles),
    format('[Dask] Created ~w batch files~n', [NumPartitions]),

    % Generate Dask Python script
    generate_dask_script(BatchFiles, NumWorkers, Scheduler, ClusterAddress,
                         MemoryLimit, ScriptPath, TempDir, PythonScript),
    format(atom(PythonScriptPath), '~w/dask_executor.py', [TempDir]),
    write_file(PythonScriptPath, PythonScript),

    % Execute Python script
    format('[Dask] Executing Dask script~n', []),
    execute_python_script(PythonScriptPath, ExitCode),

    (   ExitCode =:= 0
    ->  format('[Dask] Execution completed successfully~n', [])
    ;   format('[Dask] Warning: Exit code ~w~n', [ExitCode])
    ),

    % Collect results from output files
    collect_results(Partitions, TempDir, Results),
    format('[Dask] Collected ~w results~n', [NumPartitions]).

%% backend_cleanup_impl(+State)
%  Clean up Dask backend resources
backend_cleanup_impl(state(_, _, _, _, temp_dir(TempDir))) :-
    format('[Dask] Cleaning up temp directory: ~w~n', [TempDir]),
    (   exists_directory(TempDir)
    ->  delete_directory_and_contents(TempDir),
        format('[Dask] Cleanup complete~n', [])
    ;   format('[Dask] Temp directory already removed~n', [])
    ).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% check_dask_installed
%  Verify Python and Dask are installed and accessible
check_dask_installed :-
    catch(
        (   process_create(path(python3), ['-c', 'import dask; print(dask.__version__)'],
                          [stdout(pipe(Stream)), stderr(null), process(PID)]),
            read_string(Stream, _, VersionOutput),
            close(Stream),
            process_wait(PID, exit(ExitCode)),
            (   ExitCode =:= 0
            ->  normalize_space(string(Version), VersionOutput),
                format('[Dask] Detected Dask version: ~w~n', [Version])
            ;   throw(error(system_error, 'dask import failed'))
            )
        ),
        Error,
        (   format('[Dask] ERROR: Python or Dask not found~n', []),
            format('[Dask] Error: ~w~n', [Error]),
            format('[Dask] Install with: pip install dask distributed~n', []),
            throw(error(existence_error(python_module, dask),
                       context(check_dask_installed/0,
                              'Dask is not installed. Run: pip install dask distributed')))
        )
    ).

%% create_temp_directory(-TempDir)
%  Create temporary directory for batch files
create_temp_directory(TempDir) :-
    get_time(Timestamp),
    format(atom(TempDir), '/tmp/unifyweaver_dask_~w', [Timestamp]),
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

%% generate_dask_script(+BatchFiles, +NumWorkers, +Scheduler, +ClusterAddress,
%%                      +MemoryLimit, +ScriptPath, +TempDir, -Script)
%  Generate Python script for Dask parallel execution
generate_dask_script(BatchFiles, NumWorkers, Scheduler, ClusterAddress,
                     MemoryLimit, ScriptPath, TempDir, Script) :-
    % Build batch files Python list
    maplist(format_python_string, BatchFiles, BatchStrings),
    atomic_list_concat(BatchStrings, ', ', BatchFilesList),

    % Generate scheduler configuration
    generate_scheduler_config(Scheduler, ClusterAddress, NumWorkers, MemoryLimit, SchedulerConfig),

    % Generate script
    format(string(Script), '#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated Dask parallel execution script
UnifyWeaver distributed backend
"""

import subprocess
import sys
import os
from pathlib import Path

# Dask imports
try:
    import dask
    from dask import delayed, compute
    from dask.distributed import Client, LocalCluster
except ImportError:
    print("[Dask] ERROR: Dask not installed. Run: pip install dask distributed", file=sys.stderr)
    sys.exit(1)

# Configuration
BATCH_FILES = [~w]
SCRIPT_PATH = "~w"
OUTPUT_DIR = "~w"
NUM_WORKERS = ~w

~w

def process_batch(batch_file: str, batch_id: int) -> str:
    """Process a single batch file using the bash script."""
    output_file = f"{OUTPUT_DIR}/output_{batch_id}.txt"

    try:
        # Read input from batch file
        with open(batch_file, "r", encoding="utf-8") as f:
            input_data = f.read()

        # Execute bash script with input
        result = subprocess.run(
            ["bash", SCRIPT_PATH],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per batch
        )

        output = result.stdout
        if result.returncode != 0:
            output += f"\\n[ERROR] Exit code: {result.returncode}\\n{result.stderr}"

        # Write output to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)

        return f"Batch {batch_id}: OK ({len(output)} bytes)"

    except subprocess.TimeoutExpired:
        error_msg = f"[ERROR] Batch {batch_id} timed out"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(error_msg)
        return error_msg

    except Exception as e:
        error_msg = f"[ERROR] Batch {batch_id}: {str(e)}"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(error_msg)
        return error_msg


def main():
    """Main execution with Dask parallelization."""
    print(f"[Dask] Starting execution with {NUM_WORKERS} workers", file=sys.stderr)
    print(f"[Dask] Processing {len(BATCH_FILES)} batches", file=sys.stderr)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create delayed tasks for each batch
    tasks = []
    for batch_id, batch_file in enumerate(BATCH_FILES):
        task = delayed(process_batch)(batch_file, batch_id)
        tasks.append(task)

    # Execute all tasks in parallel
    print(f"[Dask] Scheduler: {dask.config.get(\"scheduler\", \"default\")}", file=sys.stderr)

    try:
        results = compute(*tasks)

        # Report results
        for result in results:
            print(f"[Dask] {result}", file=sys.stderr)

        print(f"[Dask] All {len(results)} batches completed", file=sys.stderr)
        return 0

    except Exception as e:
        print(f"[Dask] ERROR: {str(e)}", file=sys.stderr)
        return 1

    finally:
        # Cleanup distributed client if used
        if "client" in dir():
            try:
                client.close()
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())
', [BatchFilesList, ScriptPath, TempDir, NumWorkers, SchedulerConfig]).

%% generate_scheduler_config(+Scheduler, +ClusterAddress, +NumWorkers, +MemoryLimit, -Config)
%  Generate Python code for scheduler configuration
generate_scheduler_config(synchronous, _, _, _, Config) :-
    Config = 'dask.config.set(scheduler="synchronous")\nprint("[Dask] Using synchronous scheduler (single-threaded)", file=sys.stderr)'.

generate_scheduler_config(threads, _, NumWorkers, _, Config) :-
    format(string(Config), 'dask.config.set(scheduler="threads", num_workers=~w)
print("[Dask] Using threaded scheduler with ~w threads", file=sys.stderr)', [NumWorkers, NumWorkers]).

generate_scheduler_config(processes, _, NumWorkers, _, Config) :-
    format(string(Config), 'dask.config.set(scheduler="processes", num_workers=~w)
print("[Dask] Using multiprocessing scheduler with ~w processes", file=sys.stderr)', [NumWorkers, NumWorkers]).

generate_scheduler_config(distributed, ClusterAddress, NumWorkers, MemoryLimit, Config) :-
    (   ClusterAddress = none
    ->  % Start local cluster
        format(string(Config), '# Start local Dask cluster
cluster = LocalCluster(n_workers=~w, threads_per_worker=1, memory_limit="~w")
client = Client(cluster)
print(f"[Dask] Started local cluster: {client.dashboard_link}", file=sys.stderr)',
               [NumWorkers, MemoryLimit])
    ;   % Connect to existing cluster
        format(string(Config), '# Connect to existing Dask cluster
client = Client("~w")
print(f"[Dask] Connected to cluster: {client.dashboard_link}", file=sys.stderr)',
               [ClusterAddress])
    ).

%% format_python_string(+Atom, -PythonString)
%  Format atom as Python string literal
format_python_string(Atom, PythonString) :-
    format(atom(PythonString), '"~w"', [Atom]).

%% write_file(+FilePath, +Content)
%  Write content to file
write_file(FilePath, Content) :-
    open(FilePath, write, Stream, [encoding(utf8)]),
    write(Stream, Content),
    close(Stream).

%% execute_python_script(+ScriptPath, -ExitCode)
%  Execute the Dask Python script
execute_python_script(ScriptPath, ExitCode) :-
    process_create(path(python3), [ScriptPath],
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
    ->  format('[Dask] stdout: ~w~n', [StdOut])
    ;   true
    ),
    (   StdErr \= ""
    ->  format('[Dask] stderr: ~w~n', [StdErr])
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
