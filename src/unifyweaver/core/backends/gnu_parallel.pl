:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% gnu_parallel.pl - GNU Parallel backend for parallel execution
% Uses GNU Parallel for batch processing across partitions

:- module(gnu_parallel_backend, [
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
%  Initialize GNU Parallel backend
%
%  Configuration options:
%  - backend_args([workers(N)]) - Number of parallel workers
%
%  @example Initialize with 4 workers
%    ?- backend_init_impl([backend_args([workers(4)])], State).
backend_init_impl(Config, State) :-
    % Check GNU Parallel is installed
    check_parallel_installed,

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

    % Create temporary directory for batch files
    create_temp_directory(TempDir),

    % Initialize state
    State = state(
        num_workers(NumWorkers),
        temp_dir(TempDir),
        parallel_path('/usr/bin/parallel')
    ),

    format('[GNUParallel] Initialized: workers=~w, temp_dir=~w~n',
           [NumWorkers, TempDir]).

%% backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  Execute script on partitions in parallel using GNU Parallel
%
%  @arg State Backend state from backend_init_impl/2
%  @arg Partitions List of partition(ID, Data) terms
%  @arg ScriptPath Path to bash script to execute
%  @arg Results List of result(PartitionID, Output) terms
backend_execute_impl(State, Partitions, ScriptPath, Results) :-
    State = state(num_workers(NumWorkers), temp_dir(TempDir), _),

    length(Partitions, NumPartitions),
    format('[GNUParallel] Executing ~w partitions with ~w workers~n',
           [NumPartitions, NumWorkers]),

    % Write partition data to batch files
    write_batch_files(Partitions, TempDir, BatchFiles),
    format('[GNUParallel] Created ~w batch files~n', [NumPartitions]),

    % Build GNU Parallel command
    build_parallel_command(BatchFiles, NumWorkers, ScriptPath, TempDir, Command),

    % Execute parallel command
    format('[GNUParallel] Executing: ~w~n', [Command]),
    execute_parallel_command(Command, ExitCode),

    (   ExitCode =:= 0
    ->  format('[GNUParallel] Execution completed successfully~n', [])
    ;   format('[GNUParallel] Warning: Exit code ~w~n', [ExitCode])
    ),

    % Collect results from output files
    collect_results(Partitions, TempDir, Results),
    format('[GNUParallel] Collected ~w results~n', [NumPartitions]).

%% backend_cleanup_impl(+State)
%  Clean up GNU Parallel backend resources
backend_cleanup_impl(state(_, temp_dir(TempDir), _)) :-
    format('[GNUParallel] Cleaning up temp directory: ~w~n', [TempDir]),
    (   exists_directory(TempDir)
    ->  delete_directory_and_contents(TempDir),
        format('[GNUParallel] Cleanup complete~n', [])
    ;   format('[GNUParallel] Temp directory already removed~n', [])
    ).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% check_parallel_installed
%  Verify GNU Parallel is installed and accessible
check_parallel_installed :-
    % Try to run 'parallel --version'
    catch(
        (   process_create(path(parallel), ['--version'],
                          [stdout(pipe(Stream)), stderr(null), process(PID)]),
            read_string(Stream, _, VersionOutput),
            close(Stream),
            process_wait(PID, exit(ExitCode)),
            (   ExitCode =:= 0
            ->  % Extract version from output
                sub_string(VersionOutput, _, _, _, "GNU parallel"),
                format('[GNUParallel] Detected: ~w', [VersionOutput])
            ;   throw(error(system_error, 'parallel command failed'))
            )
        ),
        Error,
        (   format('[GNUParallel] ERROR: GNU Parallel not found or not working~n', []),
            format('[GNUParallel] Error: ~w~n', [Error]),
            format('[GNUParallel] Install with: sudo apt-get install parallel~n', []),
            throw(error(existence_error(system_command, parallel),
                       context(check_parallel_installed/0,
                              'GNU Parallel is not installed or not in PATH')))
        )
    ).

%% create_temp_directory(-TempDir)
%  Create temporary directory for batch files
create_temp_directory(TempDir) :-
    % Generate unique directory name
    get_time(Timestamp),
    format(atom(TempDir), '/tmp/unifyweaver_parallel_~w', [Timestamp]),
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

%% build_parallel_command(+BatchFiles, +NumWorkers, +ScriptPath, +TempDir, -Command)
%  Build GNU Parallel command string
build_parallel_command(BatchFiles, NumWorkers, ScriptPath, TempDir, Command) :-
    % Build list of batch file paths
    atomic_list_concat(BatchFiles, ' ', BatchFilesStr),

    % Build command:
    % parallel --jobs N --results DIR/output_{#} 'bash SCRIPT < {}' ::: batch_files
    format(string(Command),
           'parallel --jobs ~w --results ~w/output_{#} "bash ~w < {}" ::: ~w',
           [NumWorkers, TempDir, ScriptPath, BatchFilesStr]).

%% execute_parallel_command(+Command, -ExitCode)
%  Execute GNU Parallel command via bash
execute_parallel_command(Command, ExitCode) :-
    process_create('/bin/bash', ['-c', Command],
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
    ->  format('[GNUParallel] stdout: ~w~n', [StdOut])
    ;   true
    ),
    (   StdErr \= ""
    ->  format('[GNUParallel] stderr: ~w~n', [StdErr])
    ;   true
    ).

%% collect_results(+Partitions, +TempDir, -Results)
%  Collect results from GNU Parallel output files
collect_results(Partitions, TempDir, Results) :-
    maplist(collect_partition_result(TempDir), Partitions, Results).

collect_partition_result(TempDir, partition(ID, _Data), result(ID, Output)) :-
    % GNU Parallel creates output_N files (1-indexed) for each job
    % Partition ID is 0-indexed, so add 1
    JobNumber is ID + 1,
    format(atom(OutputFile), '~w/output_~w', [TempDir, JobNumber]),

    (   exists_file(OutputFile)
    ->  read_file_to_string(OutputFile, Output, [])
    ;   % File doesn't exist, return empty
        Output = ""
    ).
