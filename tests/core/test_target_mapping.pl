/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Unit tests for target_mapping module
 */

:- use_module('../../src/unifyweaver/core/target_registry').
:- use_module('../../src/unifyweaver/core/target_mapping').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Target Mapping Tests ===~n~n'),
    cleanup,
    run_all_tests,
    cleanup,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_declare_target,
    test_declare_target_options,
    test_declare_location,
    test_declare_connection,
    test_resolve_location,
    test_resolve_transport,
    test_listing,
    test_validation.

cleanup :-
    % Clean up any test declarations
    retractall(target_mapping:user_target(_, _, _)),
    retractall(target_mapping:user_location(_, _)),
    retractall(target_mapping:user_connection(_, _, _)).

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED~n', [TestName]),
        fail
    ).

assert_false(Goal, TestName) :-
    (   \+ call(Goal)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED (expected false)~n', [TestName]),
        fail
    ).

assert_equal(Got, Expected, TestName) :-
    (   Got == Expected
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED: got ~w, expected ~w~n', [TestName, Got, Expected]),
        fail
    ).

%% ============================================
%% Test: Declare Target
%% ============================================

test_declare_target :-
    format('Test: Declare target~n'),

    % Declare a target
    declare_target(filter/2, awk),
    assert_true(predicate_target(filter/2, awk), 'filter/2 maps to awk'),

    % Declare another
    declare_target(analyze/3, python),
    assert_true(predicate_target(analyze/3, python), 'analyze/3 maps to python'),

    % Undeclare
    undeclare_target(filter/2),
    assert_false(predicate_target(filter/2, _), 'filter/2 undeclared'),

    % Cleanup
    undeclare_target(analyze/3),
    format('~n').

%% ============================================
%% Test: Declare Target with Options
%% ============================================

test_declare_target_options :-
    format('Test: Declare target with options~n'),

    % Declare with options
    declare_target(process/2, go, [optimize(speed), streaming(true)]),

    predicate_target_options(process/2, Target, Options),
    assert_equal(Target, go, 'process/2 maps to go'),
    assert_true(member(optimize(speed), Options), 'has optimize option'),
    assert_true(member(streaming(true), Options), 'has streaming option'),

    % Re-declare overwrites
    declare_target(process/2, rust, [memory_safe(true)]),
    predicate_target_options(process/2, NewTarget, NewOptions),
    assert_equal(NewTarget, rust, 'process/2 now maps to rust'),
    assert_true(member(memory_safe(true), NewOptions), 'has new option'),
    assert_false(member(optimize(speed), NewOptions), 'old option gone'),

    undeclare_target(process/2),
    format('~n').

%% ============================================
%% Test: Declare Location
%% ============================================

test_declare_location :-
    format('Test: Declare location~n'),

    % Local separate process
    declare_location(worker/1, [process(separate)]),
    predicate_location(worker/1, WorkerLoc),
    assert_true(member(process(separate), WorkerLoc), 'worker has separate process'),

    % Remote
    declare_location(ml_model/2, [host('ml-server.local'), port(8080)]),
    predicate_location(ml_model/2, MlLoc),
    assert_true(member(host('ml-server.local'), MlLoc), 'ml_model has host'),
    assert_true(member(port(8080), MlLoc), 'ml_model has port'),

    undeclare_location(worker/1),
    undeclare_location(ml_model/2),
    format('~n').

%% ============================================
%% Test: Declare Connection
%% ============================================

test_declare_connection :-
    format('Test: Declare connection~n'),

    % Declare connection
    declare_connection(producer/2, consumer/2, [transport(pipe), format(json)]),

    connection_transport(producer/2, consumer/2, Transport),
    assert_equal(Transport, pipe, 'connection uses pipe'),

    connection_options(producer/2, consumer/2, Options),
    assert_true(member(format(json), Options), 'connection uses json format'),

    % Order shouldn't matter for lookup
    connection_transport(consumer/2, producer/2, Transport2),
    assert_equal(Transport2, pipe, 'reverse lookup works'),

    undeclare_connection(producer/2, consumer/2),
    assert_false(connection_transport(producer/2, consumer/2, _), 'connection removed'),
    format('~n').

%% ============================================
%% Test: Resolve Location
%% ============================================

test_resolve_location :-
    format('Test: Resolve location~n'),

    % With explicit location
    declare_target(remote_pred/1, python),
    declare_location(remote_pred/1, [host('worker.local')]),
    resolve_location(remote_pred/1, Loc1),
    assert_equal(Loc1, remote('worker.local'), 'explicit remote location resolved'),

    % With default (based on target family)
    declare_target(dotnet_pred/1, csharp),
    resolve_location(dotnet_pred/1, Loc2),
    assert_equal(Loc2, in_process, 'csharp defaults to in_process'),

    declare_target(shell_pred/1, bash),
    resolve_location(shell_pred/1, Loc3),
    assert_equal(Loc3, local_process, 'bash defaults to local_process'),

    % Unknown predicate falls back to local_process
    resolve_location(unknown/0, Loc4),
    assert_equal(Loc4, local_process, 'unknown defaults to local_process'),

    undeclare_target(remote_pred/1),
    undeclare_location(remote_pred/1),
    undeclare_target(dotnet_pred/1),
    undeclare_target(shell_pred/1),
    format('~n').

%% ============================================
%% Test: Resolve Transport
%% ============================================

test_resolve_transport :-
    format('Test: Resolve transport~n'),

    % With explicit connection
    declare_connection(pred_a/1, pred_b/1, [transport(socket)]),
    resolve_transport(pred_a/1, pred_b/1, T1),
    assert_equal(T1, socket, 'explicit transport resolved'),

    % Without explicit - uses location defaults
    declare_target(local_a/1, bash),
    declare_target(local_b/1, python),
    resolve_transport(local_a/1, local_b/1, T2),
    assert_equal(T2, pipe, 'local_process to local_process defaults to pipe'),

    % In-process
    declare_target(cs_a/1, csharp),
    declare_target(cs_b/1, powershell),
    resolve_transport(cs_a/1, cs_b/1, T3),
    assert_equal(T3, direct, 'in_process to in_process defaults to direct'),

    undeclare_connection(pred_a/1, pred_b/1),
    undeclare_target(local_a/1),
    undeclare_target(local_b/1),
    undeclare_target(cs_a/1),
    undeclare_target(cs_b/1),
    format('~n').

%% ============================================
%% Test: Listing
%% ============================================

test_listing :-
    format('Test: Listing~n'),

    declare_target(list_test_a/1, awk),
    declare_target(list_test_b/2, python),
    declare_connection(list_test_a/1, list_test_b/2, [transport(pipe)]),

    list_mappings(Mappings),
    assert_true(member(mapping(list_test_a/1, awk, _), Mappings), 'mappings includes list_test_a'),
    assert_true(member(mapping(list_test_b/2, python, _), Mappings), 'mappings includes list_test_b'),

    list_connections(Connections),
    length(Connections, N),
    assert_true(N >= 1, 'at least one connection'),

    undeclare_target(list_test_a/1),
    undeclare_target(list_test_b/2),
    undeclare_connection(list_test_a/1, list_test_b/2),
    format('~n').

%% ============================================
%% Test: Validation
%% ============================================

test_validation :-
    format('Test: Validation~n'),

    % Valid mapping
    declare_target(valid_pred/1, python),
    validate_mapping(valid_pred/1, Errors1),
    assert_equal(Errors1, [], 'valid mapping has no errors'),

    % Missing target
    validate_mapping(unmapped_pred/1, Errors2),
    assert_true(member(error(no_target, unmapped_pred/1), Errors2), 'missing target detected'),

    % Unknown target (register a fake one first)
    retractall(target_mapping:user_target(fake_target_pred/1, _, _)),
    assertz(target_mapping:user_target(fake_target_pred/1, nonexistent_target, [])),
    validate_mapping(fake_target_pred/1, Errors3),
    assert_true(member(error(unknown_target, nonexistent_target), Errors3), 'unknown target detected'),

    undeclare_target(valid_pred/1),
    retractall(target_mapping:user_target(fake_target_pred/1, _, _)),
    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
