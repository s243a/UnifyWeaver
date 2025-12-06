/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Unit tests for target_registry module
 */

:- use_module('../../src/unifyweaver/core/target_registry').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Target Registry Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_builtin_targets,
    test_target_family,
    test_same_family,
    test_capabilities,
    test_custom_target,
    test_default_location,
    test_default_transport,
    test_listing.

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
%% Test: Built-in Targets
%% ============================================

test_builtin_targets :-
    format('Test: Built-in targets~n'),
    assert_true(target_exists(bash), 'bash exists'),
    assert_true(target_exists(awk), 'awk exists'),
    assert_true(target_exists(python), 'python exists'),
    assert_true(target_exists(go), 'go exists'),
    assert_true(target_exists(rust), 'rust exists'),
    assert_true(target_exists(csharp), 'csharp exists'),
    assert_true(target_exists(sql), 'sql exists'),
    assert_false(target_exists(nonexistent), 'nonexistent does not exist'),
    format('~n').

%% ============================================
%% Test: Target Family
%% ============================================

test_target_family :-
    format('Test: Target families~n'),
    target_family(bash, BashFamily),
    assert_equal(BashFamily, shell, 'bash is shell family'),

    target_family(awk, AwkFamily),
    assert_equal(AwkFamily, shell, 'awk is shell family'),

    target_family(csharp, CsharpFamily),
    assert_equal(CsharpFamily, dotnet, 'csharp is dotnet family'),

    target_family(go, GoFamily),
    assert_equal(GoFamily, native, 'go is native family'),

    target_family(python, PythonFamily),
    assert_equal(PythonFamily, python, 'python is python family'),
    format('~n').

%% ============================================
%% Test: Same Family
%% ============================================

test_same_family :-
    format('Test: Same family detection~n'),
    assert_true(targets_same_family(bash, awk), 'bash and awk same family'),
    assert_true(targets_same_family(csharp, powershell), 'csharp and powershell same family'),
    assert_true(targets_same_family(go, rust), 'go and rust same family'),
    assert_false(targets_same_family(bash, python), 'bash and python different families'),
    assert_false(targets_same_family(csharp, go), 'csharp and go different families'),
    format('~n').

%% ============================================
%% Test: Capabilities
%% ============================================

test_capabilities :-
    format('Test: Target capabilities~n'),
    target_capabilities(awk, AwkCaps),
    assert_true(member(streaming, AwkCaps), 'awk has streaming'),
    assert_true(member(regex, AwkCaps), 'awk has regex'),
    assert_true(member(aggregation, AwkCaps), 'awk has aggregation'),

    target_capabilities(python, PyCaps),
    assert_true(member(libraries, PyCaps), 'python has libraries'),
    assert_true(member(ml, PyCaps), 'python has ml'),
    format('~n').

%% ============================================
%% Test: Custom Target Registration
%% ============================================

test_custom_target :-
    format('Test: Custom target registration~n'),

    % Register custom target
    register_target(my_custom, custom_family, [feature1, feature2]),
    assert_true(target_exists(my_custom), 'custom target exists after registration'),

    target_family(my_custom, CustomFamily),
    assert_equal(CustomFamily, custom_family, 'custom target has correct family'),

    target_capabilities(my_custom, CustomCaps),
    assert_true(member(feature1, CustomCaps), 'custom target has feature1'),

    % Unregister
    unregister_target(my_custom),
    assert_false(target_exists(my_custom), 'custom target gone after unregister'),
    format('~n').

%% ============================================
%% Test: Default Location
%% ============================================

test_default_location :-
    format('Test: Default location~n'),

    % .NET targets default to in_process
    default_location(csharp, CsharpLoc),
    assert_equal(CsharpLoc, in_process, 'csharp defaults to in_process'),

    default_location(powershell, PsLoc),
    assert_equal(PsLoc, in_process, 'powershell defaults to in_process'),

    % Shell targets default to local_process
    default_location(bash, BashLoc),
    assert_equal(BashLoc, local_process, 'bash defaults to local_process'),

    default_location(awk, AwkLoc),
    assert_equal(AwkLoc, local_process, 'awk defaults to local_process'),
    format('~n').

%% ============================================
%% Test: Default Transport
%% ============================================

test_default_transport :-
    format('Test: Default transport~n'),

    default_transport(in_process, in_process, DirectTransport),
    assert_equal(DirectTransport, direct, 'in_process to in_process uses direct'),

    default_transport(local_process, local_process, PipeTransport),
    assert_equal(PipeTransport, pipe, 'local_process to local_process uses pipe'),

    default_transport(local_process, remote(somehost), HttpTransport1),
    assert_equal(HttpTransport1, http, 'local to remote uses http'),

    default_transport(remote(host1), remote(host2), HttpTransport2),
    assert_equal(HttpTransport2, http, 'remote to remote uses http'),
    format('~n').

%% ============================================
%% Test: Listing
%% ============================================

test_listing :-
    format('Test: Listing~n'),

    list_targets(Targets),
    assert_true(member(bash, Targets), 'targets list includes bash'),
    assert_true(member(python, Targets), 'targets list includes python'),
    length(Targets, N),
    assert_true(N > 10, 'more than 10 targets registered'),

    list_families(Families),
    assert_true(member(shell, Families), 'families list includes shell'),
    assert_true(member(dotnet, Families), 'families list includes dotnet'),
    assert_true(member(native, Families), 'families list includes native'),
    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
