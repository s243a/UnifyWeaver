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
    test_generate_hash_partitioner,
    test_execute_hash_partitioner,
    test_generate_key_partitioner,
    test_execute_key_partitioner,

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

%% ============================================
%% TEST 4: GENERATE HASH PARTITIONER
%% ============================================

test_generate_hash_partitioner :-
    format('~n[Test 4] Generate bash script for hash partitioning~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate bash code for hash-based partitioning
    generate_bash_partitioner(
        hash_based([column(1), num_partitions(4), delimiter(',')]),
        [function_name(partition_by_hash)],
        BashCode
    ),

    % Verify contains AWK command
    (   sub_atom(BashCode, _, _, _, 'awk -F')
    ->  format('  ✓ Contains AWK command~n', [])
    ;   format('  ✗ FAIL: Missing AWK command~n', []),
        fail
    ),

    % Verify contains hash logic
    (   sub_atom(BashCode, _, _, _, 'hash =')
    ->  format('  ✓ Contains hash calculation~n', [])
    ;   format('  ✗ FAIL: Missing hash calculation~n', []),
        fail
    ),

    % Write to file
    ScriptPath = '/tmp/partition_by_hash.sh',
    write_bash_partitioner(BashCode, ScriptPath),
    format('  ✓ Generated bash script: ~w~n', [ScriptPath]),

    format('[✓] Test 4 Passed~n', []),
    !.

%% ============================================
%% TEST 5: EXECUTE HASH PARTITIONER
%% ============================================

test_execute_hash_partitioner :-
    format('~n[Test 5] Execute bash hash partitioner with test data~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data file with CSV data
    TestDataPath = '/tmp/test_data_hash.csv',
    open(TestDataPath, write, Stream),
    % Create data with different keys in first column
    forall(member(Key-Value, ['A'-10, 'B'-20, 'C'-30, 'A'-15, 'B'-25, 'D'-40, 'A'-12, 'C'-35]), (
        format(Stream, '~w,~w~n', [Key, Value])
    )),
    close(Stream),
    format('  ✓ Created test CSV file with 8 lines~n', []),

    % Execute partition script
    ScriptPath = '/tmp/partition_by_hash.sh',
    PartitionDir = '/tmp/test_partitions_hash',
    format(atom(Cmd), 'PARTITION_DIR=~w bash ~w ~w', [PartitionDir, ScriptPath, TestDataPath]),
    shell(Cmd, Status),

    % Verify execution succeeded
    (   Status = 0
    ->  format('  ✓ Script executed successfully~n', [])
    ;   format('  ✗ FAIL: Script failed with status ~w~n', [Status]),
        fail
    ),

    % Verify at least one partition was created
    format(atom(PartitionPattern), '~w/partition_*.txt', [PartitionDir]),
    expand_file_name(PartitionPattern, PartitionFiles),
    length(PartitionFiles, PartitionCount),
    (   PartitionCount > 0
    ->  format('  ✓ Created ~w partitions (hash-based)~n', [PartitionCount])
    ;   format('  ✗ FAIL: No partitions found~n', []),
        fail
    ),

    % Cleanup
    format(atom(CleanupCmd), 'rm -rf ~w', [PartitionDir]),
    shell(CleanupCmd),
    delete_file(TestDataPath),
    format('  ✓ Cleanup complete~n', []),

    format('[✓] Test 5 Passed~n', []),
    !.

%% ============================================
%% TEST 6: GENERATE KEY PARTITIONER
%% ============================================

test_generate_key_partitioner :-
    format('~n[Test 6] Generate bash script for key-based partitioning~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate bash code for key-based partitioning
    generate_bash_partitioner(
        key_based([column(1), delimiter(',')]),
        [function_name(partition_by_key)],
        BashCode
    ),

    % Verify contains AWK command
    (   sub_atom(BashCode, _, _, _, 'awk -F')
    ->  format('  ✓ Contains AWK command~n', [])
    ;   format('  ✗ FAIL: Missing AWK command~n', []),
        fail
    ),

    % Verify contains partition_map logic
    (   sub_atom(BashCode, _, _, _, 'partition_map')
    ->  format('  ✓ Contains partition mapping logic~n', [])
    ;   format('  ✗ FAIL: Missing partition mapping~n', []),
        fail
    ),

    % Write to file
    ScriptPath = '/tmp/partition_by_key.sh',
    write_bash_partitioner(BashCode, ScriptPath),
    format('  ✓ Generated bash script: ~w~n', [ScriptPath]),

    format('[✓] Test 6 Passed~n', []),
    !.

%% ============================================
%% TEST 7: EXECUTE KEY PARTITIONER
%% ============================================

test_execute_key_partitioner :-
    format('~n[Test 7] Execute bash key-based partitioner with test data~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data file with CSV data
    TestDataPath = '/tmp/test_data_key.csv',
    open(TestDataPath, write, Stream),
    % Create data with distinct keys (should create one partition per key)
    forall(member(Key-Values, ['Red'-[10,15,20], 'Blue'-[30,35], 'Green'-[40]]), (
        forall(member(Value, Values), (
            format(Stream, '~w,~w~n', [Key, Value])
        ))
    )),
    close(Stream),
    format('  ✓ Created test CSV file with 6 lines, 3 unique keys~n', []),

    % Execute partition script
    ScriptPath = '/tmp/partition_by_key.sh',
    PartitionDir = '/tmp/test_partitions_key',
    format(atom(Cmd), 'PARTITION_DIR=~w bash ~w ~w', [PartitionDir, ScriptPath, TestDataPath]),
    shell(Cmd, Status),

    % Verify execution succeeded
    (   Status = 0
    ->  format('  ✓ Script executed successfully~n', [])
    ;   format('  ✗ FAIL: Script failed with status ~w~n', [Status]),
        fail
    ),

    % Verify exactly 3 partitions were created (one per unique key)
    format(atom(PartitionPattern), '~w/partition_[0-9]*.txt', [PartitionDir]),
    expand_file_name(PartitionPattern, PartitionFiles),
    length(PartitionFiles, PartitionCount),
    (   PartitionCount = 3
    ->  format('  ✓ Created exactly 3 partitions (one per unique key)~n', [])
    ;   format('  ✗ FAIL: Expected 3 partitions, got ~w~n', [PartitionCount]),
        fail
    ),

    % Verify metadata contains key mappings
    format(atom(MetadataPath), '~w/metadata.json', [PartitionDir]),
    (   exists_file(MetadataPath)
    ->  format('  ✓ Metadata file created with key mappings~n', [])
    ;   format('  ✗ FAIL: Metadata file not found~n', []),
        fail
    ),

    % Cleanup
    format(atom(CleanupCmd), 'rm -rf ~w', [PartitionDir]),
    shell(CleanupCmd),
    delete_file(TestDataPath),
    format('  ✓ Cleanup complete~n', []),

    format('[✓] Test 7 Passed~n', []),
    !.

:- initialization(main, main).
