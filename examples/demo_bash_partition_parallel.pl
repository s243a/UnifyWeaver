:- encoding(utf8).
% demo_bash_partition_parallel.pl - Demonstrate Bash Partitioning + Parallel Processing
%
% Shows the complete no-dependency pipeline:
% 1. Generate pure bash partitioning script (no Prolog dependency)
% 2. Use it to partition data
% 3. Process partitions in parallel using bash fork backend
%
% This demonstrates UnifyWeaver's ability to create fully self-contained
% bash-based data processing pipelines.

:- use_module('../src/unifyweaver/targets/bash_partitioning_target').
:- use_module('../src/unifyweaver/core/parallel_backend').
:- use_module('../src/unifyweaver/core/backends/bash_fork').
:- use_module('../src/unifyweaver/core/partitioner').
:- use_module('../src/unifyweaver/core/partitioners/fixed_size').

%% ============================================
%% MAIN DEMONSTRATIONS
%% ============================================

main :-
    format('~n╔═══════════════════════════════════════════════════════════╗~n', []),
    format('║  Bash Partitioning + Parallel Processing Demo           ║~n', []),
    format('╚═══════════════════════════════════════════════════════════╝~n~n', []),

    % Setup
    setup_system,

    % Run demonstrations
    demo_1_generate_bash_partitioner,
    demo_2_partition_with_bash_script,
    demo_3_parallel_process_partitions,

    format('~n╔═══════════════════════════════════════════════════════════╗~n', []),
    format('║  All Demonstrations Complete ✓                           ║~n', []),
    format('╚═══════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Demonstrations failed~n', []),
    halt(1).

%% ============================================
%% SETUP
%% ============================================

setup_system :-
    register_backend(bash_fork, bash_fork_backend),
    register_partitioner(fixed_size, fixed_size_partitioner),
    format('[System] Registered backends and partitioners~n', []).

%% ============================================
%% DEMO 1: GENERATE PURE BASH PARTITIONER
%% ============================================

demo_1_generate_bash_partitioner :-
    format('~n[Demo 1] Generate Pure Bash Partitioning Script~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Generate bash partitioner (no Prolog dependency at runtime!)
    generate_bash_partitioner(
        fixed_size(rows(100)),
        [function_name(partition_data)],
        BashCode
    ),

    % Write to file
    ScriptPath = '/tmp/bash_partitioner.sh',
    write_bash_partitioner(BashCode, ScriptPath),

    % Verify script exists
    (   exists_file(ScriptPath)
    ->  format('  ✓ Generated bash partitioner: ~w~n', [ScriptPath])
    ;   format('  ✗ FAIL: Script not created~n', []),
        fail
    ),

    % Show script characteristics
    size_file(ScriptPath, Size),
    format('  ✓ Script size: ~w bytes~n', [Size]),
    format('  ✓ Strategy: Fixed-size (100 rows per partition)~n', []),
    format('  ✓ Dependencies: None (pure bash + split command)~n', []),

    format('[✓] Demo 1 Complete~n', []),
    !.

%% ============================================
%% DEMO 2: USE BASH SCRIPT TO PARTITION DATA
%% ============================================

demo_2_partition_with_bash_script :-
    format('~n[Demo 2] Partition Data Using Generated Bash Script~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Create test data (1000 numbers)
    TestDataPath = '/tmp/test_numbers.txt',
    open(TestDataPath, write, Stream),
    forall(between(1, 1000, N), writeln(Stream, N)),
    close(Stream),
    format('  ✓ Created test data: 1000 numbers~n', []),

    % Execute bash partitioner
    ScriptPath = '/tmp/bash_partitioner.sh',
    PartitionDir = '/tmp/bash_partitions',
    format(atom(Cmd), 'PARTITION_DIR=~w ~w ~w', [PartitionDir, ScriptPath, TestDataPath]),

    format('  Executing: ~w~n', [Cmd]),
    shell(Cmd, Status),

    % Verify execution
    (   Status = 0
    ->  format('  ✓ Partitioning completed successfully~n', [])
    ;   format('  ✗ FAIL: Partitioning failed with status ~w~n', [Status]),
        fail
    ),

    % Check partitions created
    format(atom(PartitionPattern), '~w/partition_*.txt', [PartitionDir]),
    expand_file_name(PartitionPattern, PartitionFiles),
    length(PartitionFiles, NumPartitions),
    format('  ✓ Created ~w partitions~n', [NumPartitions]),

    % Verify partition count (1000/100 = 10 partitions)
    (   NumPartitions =:= 10
    ->  format('  ✓ Correct number of partitions (expected 10)~n', [])
    ;   format('  ⚠ Got ~w partitions (expected 10)~n', [NumPartitions])
    ),

    % Check metadata
    format(atom(MetadataPath), '~w/metadata.json', [PartitionDir]),
    (   exists_file(MetadataPath)
    ->  format('  ✓ Metadata file created~n', [])
    ;   format('  ⚠ Metadata file not found~n', [])
    ),

    format('[✓] Demo 2 Complete~n', []),
    !.

%% ============================================
%% DEMO 3: PARALLEL PROCESSING OF PARTITIONS
%% ============================================

demo_3_parallel_process_partitions :-
    format('~n[Demo 3] Process Partitions in Parallel~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Read partitions created by bash script
    PartitionDir = '/tmp/bash_partitions',
    format(atom(PartitionPattern), '~w/partition_*.txt', [PartitionDir]),
    expand_file_name(PartitionPattern, PartitionFiles),
    sort(PartitionFiles, SortedFiles),  % Sort to ensure consistent ordering

    % Convert partition files to partition terms
    convert_files_to_partitions(SortedFiles, Partitions),
    length(Partitions, NumPartitions),
    format('  ✓ Loaded ~w partitions for processing~n', [NumPartitions]),

    % Create processing script (sum numbers)
    create_sum_script('/tmp/sum_partition.sh'),
    format('  ✓ Created processing script~n', []),

    % Execute in parallel using bash fork backend
    backend_init(bash_fork(workers(4)), BHandle),
    format('  Executing parallel processing (4 workers)...~n', []),
    get_time(Start),
    backend_execute(BHandle, Partitions, '/tmp/sum_partition.sh', Results),
    get_time(End),
    backend_cleanup(BHandle),

    Time is End - Start,
    format('  ✓ Parallel processing completed in ~3f seconds~n', [Time]),

    % Aggregate results
    aggregate_sums(Results, TotalSum),
    ExpectedSum is 1000 * 1001 / 2,  % Gauss formula: n(n+1)/2 for 1..1000
    format('  ✓ Total sum: ~w (expected: ~w)~n', [TotalSum, ExpectedSum]),

    % Verify correctness
    (   TotalSum =:= ExpectedSum
    ->  format('  ✓ Result verified correct!~n', [])
    ;   format('  ✗ FAIL: Incorrect result~n', []),
        fail
    ),

    % Cleanup
    delete_file('/tmp/sum_partition.sh'),
    delete_file('/tmp/test_numbers.txt'),
    delete_file('/tmp/bash_partitioner.sh'),
    format(atom(CleanupCmd), 'rm -rf ~w', [PartitionDir]),
    shell(CleanupCmd),

    format('[✓] Demo 3 Complete~n', []),
    !.

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% convert_files_to_partitions(+Files, -Partitions)
%  Convert list of partition files to partition(ID, Data) terms
convert_files_to_partitions(Files, Partitions) :-
    convert_files_to_partitions_impl(Files, 0, Partitions).

convert_files_to_partitions_impl([], _, []).
convert_files_to_partitions_impl([File|Files], ID, [partition(ID, Lines)|Rest]) :-
    % Read file content
    read_file_to_codes(File, Codes, []),
    atom_codes(Atom, Codes),
    split_string(Atom, "\n", " \t\r", LineStrs),

    % Filter empty lines and convert to atoms
    findall(Line, (
        member(LineStr, LineStrs),
        LineStr \= "",
        atom_string(Line, LineStr)
    ), Lines),

    NextID is ID + 1,
    convert_files_to_partitions_impl(Files, NextID, Rest).

%% create_sum_script(+Path)
%  Create bash script that sums numbers from stdin
create_sum_script(Path) :-
    Script = '#!/bin/bash\n# Sum all numbers from stdin\nsum=0\nwhile IFS= read -r line; do\n    if [[ \"$line\" =~ ^[0-9]+$ ]]; then\n        sum=$((sum + line))\n    fi\ndone\necho \"$sum\"\n',
    open(Path, write, Stream),
    write(Stream, Script),
    close(Stream),
    process_create('/bin/chmod', ['+x', Path], []).

%% aggregate_sums(+Results, -TotalSum)
%  Sum all partition results
aggregate_sums(Results, TotalSum) :-
    findall(Sum,
            (   member(result(_, Output), Results),
                atom_string(Output, OutputStr),
                split_string(OutputStr, "\n", " \t\r", [SumStr|_]),
                number_string(Sum, SumStr)
            ),
            Sums),
    sum_list(Sums, TotalSum).

:- initialization(main, main).
