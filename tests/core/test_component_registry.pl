% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Unit tests for component_registry module

:- encoding(utf8).

:- use_module('../../src/unifyweaver/core/component_registry').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Component Registry Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_category_definition,
    test_type_registration,
    test_component_query,
    test_builtin_test.

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  ~c ~w~n', [10003, TestName])  % checkmark
    ;   format('  X ~w FAILED~n', [TestName]),
        fail
    ).

assert_false(Goal, TestName) :-
    (   \+ call(Goal)
    ->  format('  ~c ~w~n', [10003, TestName])  % checkmark
    ;   format('  X ~w FAILED (expected false)~n', [TestName]),
        fail
    ).

assert_equal(Got, Expected, TestName) :-
    (   Got == Expected
    ->  format('  ~c ~w~n', [10003, TestName])  % checkmark
    ;   format('  X ~w FAILED: got ~w, expected ~w~n', [TestName, Got, Expected]),
        fail
    ).

%% ============================================
%% Test: Category Definition
%% ============================================

test_category_definition :-
    format('Test: Category definition~n'),

    % Runtime category should already exist (from module init)
    assert_true(category(runtime, _, _), 'runtime category exists'),

    % Define a new test category
    define_category(test_cat, "Test category", [requires_compilation(false)]),
    assert_true(category(test_cat, "Test category", _), 'test category defined'),

    % List categories
    list_categories(Cats),
    assert_true(member(runtime, Cats), 'runtime in category list'),
    assert_true(member(test_cat, Cats), 'test_cat in category list'),

    % Cleanup
    retractall(component_registry:stored_category(test_cat, _, _)),

    format('~n').

%% ============================================
%% Test: Type Registration
%% ============================================

test_type_registration :-
    format('Test: Type registration~n'),

    % Define test category first
    define_category(test_cat2, "Test category 2", []),

    % Register a mock type (module doesn't need to exist for registration)
    register_component_type(test_cat2, mock_type, mock_module, [
        description("Mock type for testing")
    ]),

    assert_true(component_type(test_cat2, mock_type, mock_module, _), 'mock type registered'),

    % List types
    list_types(test_cat2, Types),
    assert_true(member(mock_type, Types), 'mock_type in types list'),

    % Cleanup
    retractall(component_registry:stored_type(test_cat2, _, _, _)),
    retractall(component_registry:stored_category(test_cat2, _, _)),

    format('~n').

%% ============================================
%% Test: Component Query API
%% ============================================

test_component_query :-
    format('Test: Component query API~n'),

    % list_components for runtime (may be empty initially)
    list_components(runtime, Names),
    assert_true(is_list(Names), 'list_components returns list'),

    % list_types for runtime
    list_types(runtime, RTypes),
    assert_true(is_list(RTypes), 'list_types returns list'),

    format('~n').

%% ============================================
%% Test: Built-in Test
%% ============================================

test_builtin_test :-
    format('Test: Built-in registry test~n'),

    % Run the component_registry's own test
    (   catch(test_component_registry, E, (format('  Built-in test error: ~w~n', [E]), fail))
    ->  format('  ~c Built-in test completed~n', [10003])  % checkmark
    ;   format('  X Built-in test failed~n'),
        fail
    ),

    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
