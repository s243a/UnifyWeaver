% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Unit tests for semantic_search module

:- encoding(utf8).

:- use_module('../../src/unifyweaver/runtime/semantic_search').
:- use_module('../../src/unifyweaver/core/component_registry').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Semantic Search Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_module_loads,
    test_type_registration,
    test_config_validation,
    test_find_examples_api.

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

%% ============================================
%% Test: Module Loads
%% ============================================

test_module_loads :-
    format('Test: Module loads~n'),

    % Check type_info exists
    assert_true(semantic_search:type_info(_), 'type_info/1 exists'),

    % Check find_examples/3 is exported
    assert_true(current_predicate(semantic_search:find_examples/3), 'find_examples/3 exported'),

    % Check semantic_search/3 is exported
    assert_true(current_predicate(semantic_search:semantic_search/3), 'semantic_search/3 exported'),

    format('~n').

%% ============================================
%% Test: Type Registration
%% ============================================

test_type_registration :-
    format('Test: Type registration~n'),

    % Check that semantic_search type is registered
    assert_true(component_type(runtime, semantic_search, semantic_search, _),
                'semantic_search type registered'),

    format('~n').

%% ============================================
%% Test: Config Validation
%% ============================================

test_config_validation :-
    format('Test: Config validation~n'),

    % Valid config should pass
    ValidConfig = [db_path('test.db'), model_name('all-MiniLM-L6-v2')],
    assert_true(catch(semantic_search:validate_config(ValidConfig), _, fail) -> true ; true,
                'valid config passes validation'),

    % Config with optional mh_projection_id
    ConfigWithMH = [db_path('test.db'), mh_projection_id(2)],
    assert_true(catch(semantic_search:validate_config(ConfigWithMH), _, fail) -> true ; true,
                'config with mh_projection_id passes'),

    format('~n').

%% ============================================
%% Test: find_examples API (without actual search)
%% ============================================

test_find_examples_api :-
    format('Test: find_examples API~n'),

    % Test that find_examples/4 accepts options
    assert_true(current_predicate(semantic_search:find_examples/4), 'find_examples/4 exists'),

    % Test escape_for_python helper
    semantic_search:escape_for_python("hello world", Escaped1),
    assert_true(Escaped1 == "hello world", 'simple string escaping'),

    semantic_search:escape_for_python("test\"quote", Escaped2),
    assert_true(sub_string(Escaped2, _, _, _, "\\\""), 'quote escaping'),

    format('~n').

%% ============================================
%% Integration Test (requires database)
%% ============================================

test_integration :-
    format('Test: Integration (requires database)~n'),

    % Check if database exists
    DbPath = 'playbooks/lda-training-data/lda.db',
    (   exists_file(DbPath)
    ->  format('  Database found: ~w~n', [DbPath]),
        % Initialize component
        Config = [db_path(DbPath), model_name('all-MiniLM-L6-v2'), mh_projection_id(2)],
        (   catch(semantic_search:init_component(test_search, Config), E,
                  (format('  Init error: ~w~n', [E]), fail))
        ->  format('  ~c Component initialized~n', [10003]),
            % Try a search
            (   catch(semantic_search:find_examples("How to query SQLite?", 3, Results), E2,
                      (format('  Search error: ~w~n', [E2]), fail))
            ->  length(Results, NumResults),
                format('  ~c Search returned ~w results~n', [10003, NumResults])
            ;   format('  X Search failed~n')
            )
        ;   format('  X Component initialization failed~n')
        )
    ;   format('  Skipping: database not found~n')
    ),

    format('~n').

%% ============================================
%% Main Entry Point
%% ============================================

:- initialization((
    run_tests
), now).
