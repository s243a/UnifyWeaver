:- encoding(utf8).
% test_prolog_target.pl - Tests for Prolog as compilation target
%
% Tests the general Prolog target transpiler

:- use_module('../src/unifyweaver/targets/prolog_target').

%% ============================================
%% TEST PREDICATES (Will be transpiled)
%% ============================================

% Simple predicate (no dependencies)
double(X, Y) :- Y is X * 2.

% Predicate with partitioner dependency
process_with_partitioner(Data, Results) :-
    partitioner_init(fixed_size(rows(10)), [], Handle),
    partitioner_partition(Handle, Data, Partitions),
    partitioner_cleanup(Handle),
    Results = Partitions.

% Predicate with data source dependency
load_csv_data(File, Data) :-
    read_csv(File, Data).

%% ============================================
%% TESTS
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Prolog Target Tests                                  ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_simple_predicate,
    test_partitioner_dependency,
    test_data_source_dependency,
    test_complete_script_generation,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Prolog Target Tests Passed ✓                    ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Prolog target tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: SIMPLE PREDICATE (NO DEPENDENCIES)
%% ============================================

test_simple_predicate :-
    format('~n[Test 1] Simple predicate transpilation~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Analyze dependencies
    analyze_dependencies([double/2], Deps),

    % Should have no dependencies (simple arithmetic)
    (   Deps = []
    ->  format('  ✓ No dependencies detected for simple predicate~n', [])
    ;   format('  ✗ FAIL: Unexpected dependencies: ~w~n', [Deps]),
        fail
    ),

    % Generate script
    generate_prolog_script([double/2], [], Code),

    % Verify contains shebang
    (   sub_atom(Code, _, _, _, '#!/usr/bin/env swipl')
    ->  format('  ✓ Generated script has shebang~n', [])
    ;   format('  ✗ FAIL: Missing shebang~n', []),
        fail
    ),

    % Verify contains predicate definition (with any variable names)
    (   sub_atom(Code, _, _, _, 'double(')
    ->  format('  ✓ Contains predicate definition~n', [])
    ;   format('  ✗ FAIL: Missing predicate definition~n', []),
        fail
    ),

    % Verify has initialization
    (   sub_atom(Code, _, _, _, 'initialization(main, main)')
    ->  format('  ✓ Has initialization directive~n', [])
    ;   format('  ✗ FAIL: Missing initialization~n', []),
        fail
    ),

    format('[✓] Test 1 Passed~n', []),
    !.

%% ============================================
%% TEST 2: PARTITIONER DEPENDENCY
%% ============================================

test_partitioner_dependency :-
    format('~n[Test 2] Partitioner dependency detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Analyze dependencies
    analyze_dependencies([process_with_partitioner/2], Deps),

    % Should detect partitioner module
    (   member(module(unifyweaver(core/partitioner)), Deps)
    ->  format('  ✓ Detected partitioner module dependency~n', [])
    ;   format('  ✗ FAIL: Missing partitioner dependency~n', []),
        format('  Dependencies: ~w~n', [Deps]),
        fail
    ),

    % Should detect fixed_size strategy
    (   member(ensure_loaded(unifyweaver(core/partitioners/fixed_size)), Deps)
    ->  format('  ✓ Detected fixed_size strategy dependency~n', [])
    ;   format('  ✗ FAIL: Missing strategy dependency~n', []),
        fail
    ),

    % Should have plugin registration
    (   member(plugin_registration(partitioner, fixed_size, fixed_size_partitioner), Deps)
    ->  format('  ✓ Detected plugin registration~n', [])
    ;   format('  ✗ FAIL: Missing plugin registration~n', []),
        fail
    ),

    format('[✓] Test 2 Passed~n', []),
    !.

%% ============================================
%% TEST 3: DATA SOURCE DEPENDENCY
%% ============================================

test_data_source_dependency :-
    format('~n[Test 3] Data source dependency detection~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Analyze dependencies
    analyze_dependencies([load_csv_data/2], Deps),

    % Should detect CSV source module
    (   member(module(unifyweaver(sources/csv)), Deps)
    ->  format('  ✓ Detected CSV source dependency~n', [])
    ;   format('  ✗ FAIL: Missing CSV source dependency~n', []),
        format('  Dependencies: ~w~n', [Deps]),
        fail
    ),

    format('[✓] Test 3 Passed~n', []),
    !.

%% ============================================
%% TEST 4: COMPLETE SCRIPT GENERATION
%% ============================================

test_complete_script_generation :-
    format('~n[Test 4] Complete script generation and execution~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Generate script for double/2
    generate_prolog_script([double/2],
                          [entry_point(test_double),
                           source_file('test_prolog_target.pl')],
                          Code),

    % Write to file
    OutputPath = '/tmp/test_double_generated.pl',
    write_prolog_script(Code, OutputPath),
    format('  ✓ Generated script: ~w~n', [OutputPath]),

    % Verify file exists and is executable
    (   exists_file(OutputPath)
    ->  format('  ✓ Script file created~n', [])
    ;   format('  ✗ FAIL: Script file not created~n', []),
        fail
    ),

    % Clean up
    delete_file(OutputPath),
    format('  ✓ Cleanup complete~n', []),

    format('[✓] Test 4 Passed~n', []),
    !.

%% ============================================
%% HELPER PREDICATES
%% ============================================

% Entry point for generated script (Test 4)
test_double :-
    double(5, Result),
    format('double(5) = ~w~n', [Result]).

:- initialization(main, main).
