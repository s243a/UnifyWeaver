:- encoding(utf8).
% test_prolog_service_target.pl - Test Prolog-as-Service bash script generation
%
% Tests the Prolog-as-Service target transpiler

:- use_module('../src/unifyweaver/targets/prolog_target').
:- use_module('../src/unifyweaver/targets/prolog_service_target').
:- use_module('../src/unifyweaver/core/partitioner').
:- use_module('../src/unifyweaver/core/partitioners/fixed_size').

% Register partitioner
:- register_partitioner(fixed_size, fixed_size_partitioner).

%% ============================================
%% USER CODE (Will be transpiled to bash)
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
    partition_stdin(3, Partitions),
    write_partitions(Partitions).

%% ============================================
%% TESTS
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Prolog-as-Service Target                      ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_generate_bash_script,
    test_execute_bash_script,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: GENERATE BASH SCRIPT
%% ============================================

test_generate_bash_script :-
    format('~n[Test 1] Generate bash script with Prolog service~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate bash script
    generate_bash_with_prolog_service(
        [read_stdin_lines/1, partition_stdin/2, write_partitions/1, process_stdin/0],
        [entry_point(process_stdin), service_name(partition_service)],
        BashCode
    ),

    % Verify contains bash shebang
    (   sub_atom(BashCode, _, _, _, '#!/bin/bash')
    ->  format('  ✓ Contains bash shebang~n', [])
    ;   format('  ✗ FAIL: Missing bash shebang~n', []),
        fail
    ),

    % Verify contains environment setup
    (   sub_atom(BashCode, _, _, _, 'setup_unifyweaver_env')
    ->  format('  ✓ Contains environment setup~n', [])
    ;   format('  ✗ FAIL: Missing environment setup~n', []),
        fail
    ),

    % Verify contains heredoc
    (   sub_atom(BashCode, _, _, _, '<< \'PROLOG\'')
    ->  format('  ✓ Contains heredoc pattern~n', [])
    ;   format('  ✗ FAIL: Missing heredoc~n', []),
        fail
    ),

    % Verify contains partitioner imports
    (   sub_atom(BashCode, _, _, _, 'unifyweaver(core/partitioner)')
    ->  format('  ✓ Includes partitioner module~n', [])
    ;   format('  ✗ FAIL: Missing partitioner module~n', []),
        fail
    ),

    % Verify contains user predicates
    (   sub_atom(BashCode, _, _, _, 'read_stdin_lines')
    ->  format('  ✓ Contains user predicates~n', [])
    ;   format('  ✗ FAIL: Missing user predicates~n', []),
        fail
    ),

    % Write to file
    ScriptPath = '/tmp/partition_service.sh',
    write_bash_script(BashCode, ScriptPath),
    format('  ✓ Generated bash script: ~w~n', [ScriptPath]),

    format('[✓] Test 1 Passed~n', []),
    !.

%% ============================================
%% TEST 2: EXECUTE BASH SCRIPT
%% ============================================

test_execute_bash_script :-
    format('~n[Test 2] Execute bash script with stdin~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    ScriptPath = '/tmp/partition_service.sh',

    % Create test input
    TestInput = 'apple\\nbanana\\ncherry\\ndate\\neggplant\\nfig\\ngrape\\n',

    % Get current working directory to set UNIFYWEAVER_HOME
    working_directory(CWD, CWD),
    format(atom(UnifyweaverHome), '~w/src/unifyweaver', [CWD]),

    % Execute bash script with test input and UNIFYWEAVER_HOME set
    format(atom(Cmd), 'UNIFYWEAVER_HOME="~w" bash -c \'echo -e "~w" | ~w 2>&1\'',
           [UnifyweaverHome, TestInput, ScriptPath]),
    format('  Running: UNIFYWEAVER_HOME set, piping test data to bash script~n', []),

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
        format('~nOutput:~n~w~n', [Output]),
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
