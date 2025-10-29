:- encoding(utf8).
% test_parallel_backends.pl - Test parallel execution backends
%
% Tests both GNU Parallel and Pure Bash Fork backends

:- use_module('../src/unifyweaver/core/parallel_backend').
:- use_module('../src/unifyweaver/core/backends/gnu_parallel').
:- use_module('../src/unifyweaver/core/backends/bash_fork').
:- use_module('../src/unifyweaver/core/partitioner').
:- use_module('../src/unifyweaver/core/partitioners/fixed_size').

%% ============================================
%% MAIN TEST SUITE
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Parallel Execution Backends                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Register backends
    register_backend(gnu_parallel, gnu_parallel_backend),
    register_backend(bash_fork, bash_fork_backend),

    % Register partitioner
    register_partitioner(fixed_size, fixed_size_partitioner),

    % Run tests for each backend
    test_bash_fork_backend,
    test_gnu_parallel_backend,
    test_backend_comparison,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: BASH FORK BACKEND
%% ============================================

test_bash_fork_backend :-
    format('~n[Test 1] Bash Fork Backend - Pure Bash Parallelization~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test script that doubles numbers
    create_doubler_script('/tmp/doubler.sh'),
    format('  ✓ Created test script~n', []),

    % Create test data
    numlist(1, 20, Numbers),
    maplist(atom_number_format, Numbers, Data),

    % Partition data
    partitioner_init(fixed_size(rows(5)), [], PHandle),
    partitioner_partition(PHandle, Data, Partitions),
    partitioner_cleanup(PHandle),
    length(Partitions, NumPartitions),
    format('  ✓ Created ~w partitions~n', [NumPartitions]),

    % Execute in parallel with bash fork
    backend_init(bash_fork(workers(4)), Handle),
    backend_execute(Handle, Partitions, '/tmp/doubler.sh', Results),
    backend_cleanup(Handle),

    % Verify results
    length(Results, NumResults),
    (   NumResults =:= NumPartitions
    ->  format('  ✓ Got ~w results (expected ~w)~n', [NumResults, NumPartitions])
    ;   format('  ✗ FAIL: Got ~w results, expected ~w~n', [NumResults, NumPartitions]),
        fail
    ),

    % Verify output contains doubled numbers
    member(result(0, Output), Results),
    (   sub_string(Output, _, _, _, "2")  % First partition should have 1*2=2
    ->  format('  ✓ Output contains expected values~n', [])
    ;   format('  ✗ FAIL: Output missing expected values~n', []),
        fail
    ),

    % Cleanup
    delete_file('/tmp/doubler.sh'),
    format('  ✓ Cleanup complete~n', []),

    format('[✓] Test 1 Passed~n', []),
    !.

%% ============================================
%% TEST 2: GNU PARALLEL BACKEND
%% ============================================

test_gnu_parallel_backend :-
    format('~n[Test 2] GNU Parallel Backend - Comparison Test~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Check if GNU Parallel is installed
    (   check_gnu_parallel_available
    ->  format('  ✓ GNU Parallel is available~n', []),
        run_gnu_parallel_test
    ;   format('  ⊘ GNU Parallel not installed, skipping test~n', []),
        format('[⊘] Test 2 Skipped (GNU Parallel not available)~n', [])
    ),
    !.

run_gnu_parallel_test :-
    % Create test script
    create_doubler_script('/tmp/doubler.sh'),

    % Create test data
    numlist(1, 20, Numbers),
    maplist(atom_number_format, Numbers, Data),

    % Partition data
    partitioner_init(fixed_size(rows(5)), [], PHandle),
    partitioner_partition(PHandle, Data, Partitions),
    partitioner_cleanup(PHandle),

    % Execute with GNU Parallel
    backend_init(gnu_parallel(workers(4)), Handle),
    backend_execute(Handle, Partitions, '/tmp/doubler.sh', Results),
    backend_cleanup(Handle),

    % Verify results
    length(Results, NumResults),
    length(Partitions, NumPartitions),
    (   NumResults =:= NumPartitions
    ->  format('  ✓ Got ~w results~n', [NumResults])
    ;   format('  ✗ FAIL: Got ~w results, expected ~w~n', [NumResults, NumPartitions]),
        fail
    ),

    % Cleanup
    delete_file('/tmp/doubler.sh'),
    format('  ✓ Cleanup complete~n', []),

    format('[✓] Test 2 Passed~n', []).

%% ============================================
%% TEST 3: BACKEND COMPARISON
%% ============================================

test_backend_comparison :-
    format('~n[Test 3] Compare Bash Fork vs GNU Parallel Performance~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create larger test dataset
    numlist(1, 100, Numbers),
    maplist(atom_number_format, Numbers, Data),

    % Partition data
    partitioner_init(fixed_size(rows(10)), [], PHandle),
    partitioner_partition(PHandle, Data, Partitions),
    partitioner_cleanup(PHandle),
    length(Partitions, NumPartitions),
    format('  ✓ Created ~w partitions with 100 items~n', [NumPartitions]),

    % Create test script
    create_doubler_script('/tmp/doubler.sh'),

    % Test Bash Fork
    format('~n  Testing Bash Fork backend:~n', []),
    get_time(Start1),
    backend_init(bash_fork(workers(4)), Handle1),
    backend_execute(Handle1, Partitions, '/tmp/doubler.sh', Results1),
    backend_cleanup(Handle1),
    get_time(End1),
    BashForkTime is End1 - Start1,
    length(Results1, NumResults1),
    format('    Time: ~3f seconds, Results: ~w~n', [BashForkTime, NumResults1]),

    % Test GNU Parallel (if available)
    (   check_gnu_parallel_available
    ->  format('~n  Testing GNU Parallel backend:~n', []),
        get_time(Start2),
        backend_init(gnu_parallel(workers(4)), Handle2),
        backend_execute(Handle2, Partitions, '/tmp/doubler.sh', Results2),
        backend_cleanup(Handle2),
        get_time(End2),
        GnuTime is End2 - Start2,
        length(Results2, NumResults2),
        format('    Time: ~3f seconds, Results: ~w~n', [GnuTime, NumResults2]),

        % Compare
        (   BashForkTime < GnuTime * 2  % Bash Fork should be within 2x of GNU Parallel
        ->  format('~n  ✓ Bash Fork performance is reasonable~n', [])
        ;   format('~n  ⚠ Bash Fork is slower (acceptable for pure bash)~n', [])
        )
    ;   format('~n  ⊘ GNU Parallel not available, skipping comparison~n', [])
    ),

    % Cleanup
    delete_file('/tmp/doubler.sh'),
    format('  ✓ Cleanup complete~n', []),

    format('[✓] Test 3 Passed~n', []),
    !.

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% create_doubler_script(+Path)
%  Create a bash script that reads numbers from stdin and doubles them
create_doubler_script(Path) :-
    Script = '#!/bin/bash\n# Read numbers from stdin and double them\nwhile IFS= read -r line; do\n    if [[ "$line" =~ ^[0-9]+$ ]]; then\n        doubled=$((line * 2))\n        echo "$doubled"\n    fi\ndone\n',
    open(Path, write, Stream),
    write(Stream, Script),
    close(Stream),
    % Make executable
    process_create('/bin/chmod', ['+x', Path], []).

%% atom_number_format(+Number, -Atom)
%  Format number as atom
atom_number_format(Number, Atom) :-
    atom_number(Atom, Number).

%% check_gnu_parallel_available
%  Check if GNU Parallel is installed and accessible
check_gnu_parallel_available :-
    catch(
        (   process_create(path(parallel), ['--version'],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(ExitCode)),
            ExitCode =:= 0
        ),
        _,
        fail
    ).

:- initialization(main, main).
