:- encoding(utf8).
% test_parallel_backend.pl - Tests for parallel execution backend
%
% Tests the GNU Parallel backend with partitioned data

% Load core modules
:- use_module('../src/unifyweaver/core/parallel_backend').
:- use_module('../src/unifyweaver/core/partitioner').

% Ensure backend and strategy modules are loaded (but not imported)
:- ensure_loaded('../src/unifyweaver/core/backends/gnu_parallel').
:- ensure_loaded('../src/unifyweaver/core/partitioners/fixed_size').
:- ensure_loaded('../src/unifyweaver/core/partitioners/hash_based').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Parallel Backend Tests                               ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Register backend and partitioners
    register_backend(gnu_parallel, gnu_parallel_backend),
    register_partitioner(fixed_size, fixed_size_partitioner),
    register_partitioner(hash_based, hash_based_partitioner),

    % Run tests
    test_simple_parallel_execution,
    test_parallel_with_partitioner,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Parallel Backend Tests Passed ✓                  ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Parallel backend tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: SIMPLE PARALLEL EXECUTION
%% ============================================

test_simple_parallel_execution :-
    format('~n[Test 1] Simple parallel execution~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create simple test script that doubles input numbers
    create_test_script('/tmp/test_double.sh'),

    % Create test partitions
    Partitions = [
        partition(0, [1, 2, 3]),
        partition(1, [4, 5, 6]),
        partition(2, [7, 8, 9])
    ],

    % Initialize backend
    backend_init(gnu_parallel(workers(2)), Handle),
    format('  ✓ Backend initialized~n', []),

    % Execute in parallel
    backend_execute(Handle, Partitions, '/tmp/test_double.sh', Results),
    format('  ✓ Parallel execution completed~n', []),

    % Verify results
    length(Results, NumResults),
    (   NumResults =:= 3
    ->  format('  ✓ Got 3 results~n', [])
    ;   format('  ✗ FAIL: Expected 3 results, got ~w~n', [NumResults]),
        fail
    ),

    % Verify each result has output
    (   maplist(has_output, Results)
    ->  format('  ✓ All results have output~n', [])
    ;   format('  ✗ FAIL: Some results missing output~n', []),
        fail
    ),

    % Cleanup
    backend_cleanup(Handle),
    delete_file('/tmp/test_double.sh'),
    format('[✓] Test 1 Passed~n', []),
    !.

has_output(result(_, Output)) :-
    Output \= "".

%% ============================================
%% TEST 2: PARALLEL WITH PARTITIONER
%% ============================================

test_parallel_with_partitioner :-
    format('~n[Test 2] Parallel execution with partitioner~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data
    numlist(1, 20, Data),

    % Partition data using fixed_size strategy
    partitioner_init(fixed_size(rows(5)), [], PHandle),
    partitioner_partition(PHandle, Data, Partitions),
    partitioner_cleanup(PHandle),

    length(Partitions, NumPartitions),
    format('  ✓ Partitioned into ~w partitions~n', [NumPartitions]),

    % Create test script
    create_test_script('/tmp/test_process.sh'),

    % Initialize backend and execute
    backend_init(gnu_parallel(workers(3)), BHandle),
    backend_execute(BHandle, Partitions, '/tmp/test_process.sh', Results),

    % Verify results
    length(Results, NumResults),
    (   NumResults =:= NumPartitions
    ->  format('  ✓ Got ~w results matching ~w partitions~n', [NumResults, NumPartitions])
    ;   format('  ✗ FAIL: Result count mismatch~n', []),
        fail
    ),

    % Cleanup
    backend_cleanup(BHandle),
    delete_file('/tmp/test_process.sh'),
    format('[✓] Test 2 Passed~n', []),
    !.

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% create_test_script(+ScriptPath)
%  Create a simple bash script for testing
create_test_script(ScriptPath) :-
    open(ScriptPath, write, Stream),
    writeln(Stream, '#!/bin/bash'),
    writeln(Stream, '# Simple test script: double each input number'),
    writeln(Stream, 'while read -r line; do'),
    writeln(Stream, '    if [[ "$line" =~ ^[0-9]+$ ]]; then'),
    writeln(Stream, '        echo $((line * 2))'),
    writeln(Stream, '    else'),
    writeln(Stream, '        echo "$line"'),
    writeln(Stream, '    fi'),
    writeln(Stream, 'done'),
    close(Stream),
    % Make executable
    format(atom(ChmodCmd), 'chmod +x ~w', [ScriptPath]),
    shell(ChmodCmd).

:- initialization(main, main).
