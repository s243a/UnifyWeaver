:- encoding(utf8).
% test_bash_partitioning_target.pl - Test Pure Bash Partitioning Code Generator
%
% Tests the bash partitioning target that generates pure bash scripts
% for partitioning data without any Prolog dependency

:- use_module('../src/unifyweaver/targets/bash_partitioning_target').

%% ============================================
%% TESTS
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Pure Bash Partitioning Target                 ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_generate_row_partitioner,
    test_generate_byte_partitioner,
    test_execute_row_partitioner,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: GENERATE ROW PARTITIONER
%% ============================================

test_generate_row_partitioner :-
    format('~n[Test 1] Generate bash script for row partitioning~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate bash code
    generate_bash_partitioner(
        fixed_size(rows(100)),
        [function_name(partition_by_rows)],
        BashCode
    ),

    % Verify contains bash shebang
    (   sub_atom(BashCode, _, _, _, '#!/bin/bash')
    ->  format('  ✓ Contains bash shebang~n', [])
    ;   format('  ✗ FAIL: Missing bash shebang~n', []),
        fail
    ),

    % Verify contains split command
    (   sub_atom(BashCode, _, _, _, 'split -l 100')
    ->  format('  ✓ Contains split command with correct row count~n', [])
    ;   format('  ✗ FAIL: Missing or incorrect split command~n', []),
        fail
    ),

    % Verify contains function definition
    (   sub_atom(BashCode, _, _, _, 'partition_by_rows()')
    ->  format('  ✓ Contains function definition~n', [])
    ;   format('  ✗ FAIL: Missing function definition~n', []),
        fail
    ),

    % Verify contains metadata generation
    (   sub_atom(BashCode, _, _, _, 'metadata.json')
    ->  format('  ✓ Contains metadata generation~n', [])
    ;   format('  ✗ FAIL: Missing metadata generation~n', []),
        fail
    ),

    % Write to file
    ScriptPath = '/tmp/partition_by_rows.sh',
    write_bash_partitioner(BashCode, ScriptPath),
    format('  ✓ Generated bash script: ~w~n', [ScriptPath]),

    format('[✓] Test 1 Passed~n', []),
    !.

%% ============================================
%% TEST 2: GENERATE BYTE PARTITIONER
%% ============================================

test_generate_byte_partitioner :-
    format('~n[Test 2] Generate bash script for byte partitioning~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate bash code
    generate_bash_partitioner(
        fixed_size(bytes(1024)),
        [function_name(partition_by_bytes), include_header(true)],
        BashCode
    ),

    % Verify contains split command with byte option
    (   sub_atom(BashCode, _, _, _, 'split -b 1024')
    ->  format('  ✓ Contains split command with correct byte count~n', [])
    ;   format('  ✗ FAIL: Missing or incorrect split command~n', []),
        fail
    ),

    % Verify strategy description
    (   sub_atom(BashCode, _, _, _, 'bytes per partition')
    ->  format('  ✓ Contains strategy description~n', [])
    ;   format('  ✗ FAIL: Missing strategy description~n', []),
        fail
    ),

    % Write to file
    ScriptPath = '/tmp/partition_by_bytes.sh',
    write_bash_partitioner(BashCode, ScriptPath),
    format('  ✓ Generated bash script: ~w~n', [ScriptPath]),

    format('[✓] Test 2 Passed~n', []),
    !.

%% ============================================
%% TEST 3: EXECUTE ROW PARTITIONER
%% ============================================

test_execute_row_partitioner :-
    format('~n[Test 3] Execute bash row partitioner with test data~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data file
    TestDataPath = '/tmp/test_data.txt',
    open(TestDataPath, write, Stream),
    forall(between(1, 250, N), (
        format(Stream, 'Line ~w: Test data for partitioning~n', [N])
    )),
    close(Stream),
    format('  ✓ Created test data file with 250 lines~n', []),

    % Execute partition script
    ScriptPath = '/tmp/partition_by_rows.sh',
    PartitionDir = '/tmp/test_partitions',
    format(atom(Cmd), 'PARTITION_DIR=~w bash ~w ~w', [PartitionDir, ScriptPath, TestDataPath]),
    shell(Cmd, Status),

    % Verify execution succeeded
    (   Status = 0
    ->  format('  ✓ Script executed successfully~n', [])
    ;   format('  ✗ FAIL: Script failed with status ~w~n', [Status]),
        fail
    ),

    % Verify partitions were created
    format(atom(PartitionPattern), '~w/partition_*.txt', [PartitionDir]),
    expand_file_name(PartitionPattern, PartitionFiles),
    length(PartitionFiles, PartitionCount),
    (   PartitionCount > 0
    ->  format('  ✓ Created ~w partitions in ~w~n', [PartitionCount, PartitionDir])
    ;   format('  ✗ FAIL: No partitions found~n', []),
        fail
    ),

    % Verify metadata file exists
    format(atom(MetadataPath), '~w/metadata.json', [PartitionDir]),
    (   exists_file(MetadataPath)
    ->  format('  ✓ Metadata file created~n', [])
    ;   format('  ✗ FAIL: Metadata file not found~n', []),
        fail
    ),

    % Show sample partition
    format('~n  Sample partition content:~n', []),
    format(atom(SampleCmd), 'head -3 ~w/partition_00.txt 2>/dev/null', [PartitionDir]),
    shell(SampleCmd),

    % Cleanup
    format(atom(CleanupCmd), 'rm -rf ~w', [PartitionDir]),
    shell(CleanupCmd),
    delete_file(TestDataPath),
    format('~n  ✓ Cleanup complete~n', []),

    format('[✓] Test 3 Passed~n', []),
    !.

:- initialization(main, main).
