%% test_ui_patterns.pl - plunit tests for UI patterns module
%%
%% Tests generalizable UI patterns for code generation.
%%
%% Run with: swipl -g "run_tests" -t halt test_ui_patterns.pl

:- module(test_ui_patterns, []).

:- use_module(library(plunit)).
:- use_module(ui_patterns).

%% ============================================================================
%% Tests: Navigation Patterns
%% ============================================================================

:- begin_tests(navigation_patterns).

test(stack_navigator_pattern_creation) :-
    navigation_pattern(stack, [
        screen(home, 'HomeScreen', []),
        screen(detail, 'DetailScreen', [])
    ], Pattern),
    Pattern = navigation(stack, _, _).

test(tab_navigator_pattern_creation) :-
    navigation_pattern(tab, [
        screen(feed, 'FeedScreen', []),
        screen(profile, 'ProfileScreen', [])
    ], Pattern),
    Pattern = navigation(tab, _, _).

test(drawer_navigator_pattern_creation) :-
    navigation_pattern(drawer, [
        screen(home, 'HomeScreen', []),
        screen(settings, 'SettingsScreen', [])
    ], Pattern),
    Pattern = navigation(drawer, _, _).

test(stack_navigator_compiles_to_react_native) :-
    navigation_pattern(stack, [
        screen(home, 'HomeScreen', [title('Home')])
    ], Pattern),
    define_pattern(test_stack, Pattern, []),
    compile_pattern(test_stack, react_native, [], Code),
    sub_string(Code, _, _, _, "createStackNavigator"),
    sub_string(Code, _, _, _, "HomeScreen").

test(tab_navigator_compiles_to_react_native) :-
    navigation_pattern(tab, [
        screen(feed, 'FeedScreen', [])
    ], Pattern),
    define_pattern(test_tab, Pattern, []),
    compile_pattern(test_tab, react_native, [], Code),
    sub_string(Code, _, _, _, "createBottomTabNavigator").

test(drawer_navigator_compiles_to_react_native) :-
    navigation_pattern(drawer, [
        screen(menu, 'MenuScreen', [])
    ], Pattern),
    define_pattern(test_drawer, Pattern, []),
    compile_pattern(test_drawer, react_native, [], Code),
    sub_string(Code, _, _, _, "createDrawerNavigator").

test(navigator_has_navigation_container) :-
    navigation_pattern(stack, [screen(home, 'HomeScreen', [])], Pattern),
    define_pattern(test_nav_container, Pattern, []),
    compile_pattern(test_nav_container, react_native, [], Code),
    sub_string(Code, _, _, _, "NavigationContainer").

test(navigator_imports_screens) :-
    navigation_pattern(stack, [
        screen(home, 'HomeScreen', []),
        screen(detail, 'DetailScreen', [])
    ], Pattern),
    define_pattern(test_imports, Pattern, []),
    compile_pattern(test_imports, react_native, [], Code),
    sub_string(Code, _, _, _, "import { HomeScreen }"),
    sub_string(Code, _, _, _, "import { DetailScreen }").

:- end_tests(navigation_patterns).

%% ============================================================================
%% Tests: State Patterns
%% ============================================================================

:- begin_tests(state_patterns).

test(local_state_pattern_creation) :-
    state_pattern(local, [field(count, 0)], Pattern),
    Pattern = state(local, _, _).

test(global_state_pattern_creation) :-
    state_pattern(global, [store(app), slices([])], Pattern),
    Pattern = state(global, _, _).

test(derived_state_pattern_creation) :-
    state_pattern(derived, [deps([a, b]), derive("a + b")], Pattern),
    Pattern = state(derived, _, _).

test(local_state_compiles_to_use_state) :-
    state_pattern(local, [field(value, 0)], Pattern),
    define_pattern(test_local, Pattern, []),
    compile_pattern(test_local, react_native, [], Code),
    sub_string(Code, _, _, _, "useState").

test(global_state_compiles_to_zustand, [nondet]) :-
    global_state(testStore, [
        slice(counter, [field(count, number)], [
            action(increment, "(set) => set((state) => ({ count: state.count + 1 }))")
        ])
    ], _Pattern),
    compile_pattern(testStore, react_native, [], Code),
    sub_string(Code, _, _, _, "create").

test(derived_state_compiles_to_use_memo) :-
    state_pattern(derived, [deps([x, y]), derive("x * y")], Pattern),
    define_pattern(test_derived, Pattern, []),
    compile_pattern(test_derived, react_native, [], Code),
    sub_string(Code, _, _, _, "useMemo").

:- end_tests(state_patterns).

%% ============================================================================
%% Tests: Data Patterns
%% ============================================================================

:- begin_tests(data_patterns).

test(query_pattern_creation) :-
    data_pattern(query, [name(fetch_data), endpoint('/api/data')], Pattern),
    Pattern = data(query, _).

test(mutation_pattern_creation) :-
    data_pattern(mutation, [name(save_data), endpoint('/api/data')], Pattern),
    Pattern = data(mutation, _).

test(infinite_pattern_creation) :-
    data_pattern(infinite, [name(load_more), endpoint('/api/items')], Pattern),
    Pattern = data(infinite, _).

test(query_pattern_compiles_to_use_query, [nondet]) :-
    query_pattern(test_query, '/api/test', [], _Pattern),
    compile_pattern(test_query, react_native, [], Code),
    sub_string(Code, _, _, _, "useQuery"),
    sub_string(Code, _, _, _, "/api/test").

test(mutation_pattern_compiles_to_use_mutation, [nondet]) :-
    mutation_pattern(test_mutation, '/api/mutate', [], _Pattern),
    compile_pattern(test_mutation, react_native, [], Code),
    sub_string(Code, _, _, _, "useMutation").

test(paginated_pattern_compiles_to_infinite_query, [nondet]) :-
    paginated_pattern(test_paginated, '/api/paginated', [], _Pattern),
    compile_pattern(test_paginated, react_native, [], Code),
    sub_string(Code, _, _, _, "useInfiniteQuery").

test(query_has_stale_time_option, [nondet]) :-
    query_pattern(test_stale, '/api/test', [stale_time(60000)], _Pattern),
    compile_pattern(test_stale, react_native, [], Code),
    sub_string(Code, _, _, _, "staleTime: 60000").

test(mutation_has_invalidate_queries, [nondet]) :-
    mutation_pattern(test_invalidate, '/api/test', [], _Pattern),
    compile_pattern(test_invalidate, react_native, [], Code),
    sub_string(Code, _, _, _, "invalidateQueries").

:- end_tests(data_patterns).

%% ============================================================================
%% Tests: Persistence Patterns
%% ============================================================================

:- begin_tests(persistence_patterns).

test(local_storage_pattern_creation) :-
    persistence_pattern(local, [key(user_data), schema(string)], Pattern),
    Pattern = persistence(local, _).

test(mmkv_storage_pattern_creation) :-
    persistence_pattern(mmkv, [key(fast_data), schema(object)], Pattern),
    Pattern = persistence(mmkv, _).

test(async_storage_compiles_correctly, [nondet]) :-
    local_storage(test_storage, '{ name: string }', _Pattern),
    compile_pattern(test_storage, react_native, [], Code),
    sub_string(Code, _, _, _, "AsyncStorage"),
    sub_string(Code, _, _, _, "getItem"),
    sub_string(Code, _, _, _, "setItem").

test(async_storage_has_loading_state, [nondet]) :-
    local_storage(test_loading, 'string', _Pattern),
    compile_pattern(test_loading, react_native, [], Code),
    sub_string(Code, _, _, _, "loading"),
    sub_string(Code, _, _, _, "setLoading").

test(async_storage_has_error_handling, [nondet]) :-
    local_storage(test_error, 'string', _Pattern),
    compile_pattern(test_error, react_native, [], Code),
    sub_string(Code, _, _, _, "error"),
    sub_string(Code, _, _, _, "setError").

test(mmkv_compiles_correctly) :-
    persistence_pattern(mmkv, [key(mmkv_test), schema('string')], Pattern),
    define_pattern(test_mmkv, Pattern, []),
    compile_pattern(test_mmkv, react_native, [], Code),
    sub_string(Code, _, _, _, "MMKV"),
    sub_string(Code, _, _, _, "getString").

:- end_tests(persistence_patterns).

%% ============================================================================
%% Tests: Pattern Composition
%% ============================================================================

:- begin_tests(pattern_composition).

test(patterns_can_be_composed) :-
    navigation_pattern(stack, [screen(home, 'HomeScreen', [])], NavPattern),
    state_pattern(local, [field(count, 0)], StatePattern),
    define_pattern(nav_part, NavPattern, []),
    define_pattern(state_part, StatePattern, []),
    compose_patterns([nav_part, state_part], [], ComposedSpec),
    ComposedSpec = composite(_, _).

test(pattern_compatibility_check) :-
    navigation_pattern(stack, [], P1),
    state_pattern(local, [], P2),
    define_pattern(compat_nav, P1, [composable_with([compat_state])]),
    define_pattern(compat_state, P2, [composable_with([compat_nav])]),
    pattern_compatible(compat_nav, compat_state).

test(pattern_requirements_extracted) :-
    navigation_pattern(stack, [], P),
    define_pattern(req_test, P, [requires([navigation, screens])]),
    pattern_requires(req_test, Caps),
    member(navigation, Caps),
    member(screens, Caps).

:- end_tests(pattern_composition).

%% ============================================================================
%% Tests: Pattern Definition
%% ============================================================================

:- begin_tests(pattern_definition).

test(pattern_can_be_defined) :-
    define_pattern(test_def, navigation(stack, [], []), []),
    pattern(test_def, _, _).

test(pattern_can_be_redefined) :-
    define_pattern(redef_test, state(local, [], []), []),
    define_pattern(redef_test, state(global, [], []), []),
    pattern(redef_test, Spec, _),
    Spec = state(global, _, _).

test(pattern_options_stored) :-
    define_pattern(opts_test, state(local, [], []), [requires([hooks]), singleton(true)]),
    pattern(opts_test, _, Opts),
    member(requires([hooks]), Opts),
    member(singleton(true), Opts).

:- end_tests(pattern_definition).

%% ============================================================================
%% Tests: Generated Code Quality
%% ============================================================================

:- begin_tests(code_quality).

test(generated_code_has_react_import) :-
    navigation_pattern(stack, [screen(home, 'HomeScreen', [])], Pattern),
    define_pattern(code_test, Pattern, []),
    compile_pattern(code_test, react_native, [], Code),
    sub_string(Code, _, _, _, "import React from 'react'").

test(generated_code_has_export) :-
    navigation_pattern(stack, [screen(home, 'HomeScreen', [])], Pattern),
    define_pattern(export_test, Pattern, []),
    compile_pattern(export_test, react_native, [], Code),
    sub_string(Code, _, _, _, "export").

test(generated_hook_has_typescript_types, [nondet]) :-
    query_pattern(typed_query, '/api/typed', [], _),
    compile_pattern(typed_query, react_native, [], Code),
    sub_string(Code, _, _, _, "interface"),
    sub_string(Code, _, _, _, "UseQueryResult").

test(generated_storage_hook_has_return_type, [nondet]) :-
    local_storage(typed_storage, 'string', _),
    compile_pattern(typed_storage, react_native, [], Code),
    sub_string(Code, _, _, _, "interface Use"),
    sub_string(Code, _, _, _, "Result").

:- end_tests(code_quality).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization(run_tests, main).
