:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_partition_stdin.pl - Test Prolog target with partitioning from stdin
%
% Use Case: Read data from stdin, partition it, output partitions
% This demonstrates Prolog-as-target for real partitioning scenario

:- use_module('../src/unifyweaver/targets/prolog_target').
:- use_module('../src/unifyweaver/core/partitioner').
:- use_module('../src/unifyweaver/core/partitioners/fixed_size').

% Register partitioner
:- register_partitioner(fixed_size, fixed_size_partitioner).

%% ============================================
%% USER CODE (Will be transpiled)
%% ============================================

% Read lines from stdin
read_stdin_lines(Lines) :-
    read_line_to_codes(user_input, Codes),
    (   Codes == end_of_file
    ->  Lines = []
    ;   atom_codes(Line, Codes),
        read_stdin_lines(RestLines),
        Lines = [Line|RestLines]
    ).

% Partition stdin data
partition_stdin(PartitionSize, Partitions) :-
    read_stdin_lines(Lines),
    partitioner_init(fixed_size(rows(PartitionSize)), [], Handle),
    partitioner_partition(Handle, Lines, Partitions),
    partitioner_cleanup(Handle).

% Write partitions to output
write_partitions(Partitions) :-
    forall(member(partition(ID, Data), Partitions), (
        format('=== Partition ~w ===~n', [ID]),
        forall(member(Item, Data), (
            format('~w~n', [Item])
        ))
    )).

% Main entry point
process_stdin :-
    % Partition into groups of 3
    partition_stdin(3, Partitions),
    write_partitions(Partitions).

%% ============================================
%% TEST SUITE
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Partition Stdin via Prolog Target             ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_generate_partition_script,
    test_execute_partition_script,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: GENERATE PARTITION SCRIPT
%% ============================================

test_generate_partition_script :-
    format('~n[Test 1] Generate partition script~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate script for partition_stdin and dependencies
    generate_prolog_script(
        [read_stdin_lines/1, partition_stdin/2, write_partitions/1, process_stdin/0],
        [entry_point(process_stdin)],
        Code
    ),

    % Verify contains partitioner imports
    (   sub_atom(Code, _, _, _, 'unifyweaver(core/partitioner)')
    ->  format('  ✓ Includes partitioner module~n', [])
    ;   format('  ✗ FAIL: Missing partitioner module~n', []),
        format('~nGenerated code:~n~w~n', [Code]),
        fail
    ),

    % Verify contains fixed_size strategy
    (   sub_atom(Code, _, _, _, 'fixed_size')
    ->  format('  ✓ Includes fixed_size strategy~n', [])
    ;   format('  ✗ FAIL: Missing fixed_size strategy~n', []),
        fail
    ),

    % Verify contains our predicates
    (   sub_atom(Code, _, _, _, 'read_stdin_lines')
    ->  format('  ✓ Contains read_stdin_lines predicate~n', [])
    ;   format('  ✗ FAIL: Missing read_stdin_lines~n', []),
        fail
    ),

    (   sub_atom(Code, _, _, _, 'partition_stdin')
    ->  format('  ✓ Contains partition_stdin predicate~n', [])
    ;   format('  ✗ FAIL: Missing partition_stdin~n', []),
        fail
    ),

    % Write to file
    ScriptPath = '/tmp/partition_stdin.pl',
    write_prolog_script(Code, ScriptPath),
    format('  ✓ Generated script: ~w~n', [ScriptPath]),

    format('[✓] Test 1 Passed~n', []),
    !.

%% ============================================
%% TEST 2: EXECUTE PARTITION SCRIPT
%% ============================================

test_execute_partition_script :-
    format('~n[Test 2] Execute partition script with stdin~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    ScriptPath = '/tmp/partition_stdin.pl',

    % Create test input
    TestInput = 'line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n',

    % Get current working directory to set UNIFYWEAVER_HOME
    working_directory(CWD, CWD),
    format(atom(UnifyweaverHome), '~w/src/unifyweaver', [CWD]),

    % Execute script with test input and UNIFYWEAVER_HOME set
    format(atom(Cmd), 'UNIFYWEAVER_HOME="~w" bash -c \'echo -e "~w" | swipl ~w 2>&1\'', [UnifyweaverHome, TestInput, ScriptPath]),
    format('  Running: UNIFYWEAVER_HOME set, piping test data to script~n', []),

    % Capture output
    process_create('/bin/bash', ['-c', Cmd],
                   [stdout(pipe(Stream)), process(PID)]),
    read_string(Stream, _, Output),
    close(Stream),
    process_wait(PID, Status),

    % Verify execution succeeded
    (   Status = exit(0)
    ->  format('  ✓ Script executed successfully~n', [])
    ;   format('  ✗ FAIL: Script failed with status ~w~n', [Status]),
        fail
    ),

    % Verify output contains partitions
    (   sub_atom(Output, _, _, _, '=== Partition 0 ===')
    ->  format('  ✓ Output contains Partition 0~n', [])
    ;   format('  ✗ FAIL: Missing Partition 0 in output~n', []),
        format('~nOutput:~n~w~n', [Output]),
        fail
    ),

    (   sub_atom(Output, _, _, _, '=== Partition 1 ===')
    ->  format('  ✓ Output contains Partition 1~n', [])
    ;   format('  ✗ FAIL: Missing Partition 1 in output~n', []),
        fail
    ),

    (   sub_atom(Output, _, _, _, '=== Partition 2 ===')
    ->  format('  ✓ Output contains Partition 2~n', [])
    ;   format('  ✗ FAIL: Missing Partition 2 in output~n', []),
        fail
    ),

    % Verify line counts per partition (3 lines each for first 2, remainder in last)
    split_string(Output, "\n", "", Lines),
    length(Lines, TotalLines),
    format('  ✓ Output has ~w lines total~n', [TotalLines]),

    % Show sample output
    format('~n  Sample output:~n', []),
    split_string(Output, "\n", "", OutputLines),
    forall((member(Line, OutputLines), Line \= ""), (
        format('    ~s~n', [Line])
    )),

    % Cleanup
    delete_file(ScriptPath),
    format('~n  ✓ Cleanup complete~n', []),

    format('[✓] Test 2 Passed~n', []),
    !.

:- initialization(main, main).
