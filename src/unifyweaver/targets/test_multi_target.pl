% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_multi_target.pl - Multi-Framework Compilation Tests
%
% Tests that the same UI patterns compile correctly to all supported targets:
%   - React Native
%   - Vue
%   - Flutter
%   - SwiftUI
%
% Run with: swipl -g "run_tests" -t halt test_multi_target.pl

:- module(test_multi_target, []).

:- use_module(library(plunit)).

% Load all targets (use except to avoid import conflicts)
:- use_module('../patterns/ui_patterns').
:- use_module('vue_target', []).
:- use_module('flutter_target', []).
:- use_module('swiftui_target', []).

% ============================================================================
% Test Data - Shared Patterns
% ============================================================================

test_screens([
    screen(home, 'HomeScreen', [title('Home')]),
    screen(search, 'SearchScreen', [title('Search')]),
    screen(profile, 'ProfileScreen', [title('Profile')])
]).

test_global_state([
    store(appStore),
    slices([
        slice(ui, [
            field(theme, "'light' | 'dark'"),
            field(sidebarOpen, "boolean")
        ], [
            action(toggleTheme, _),
            action(toggleSidebar, _)
        ])
    ])
]).

test_query_config([
    name(fetchUsers),
    endpoint('/api/users'),
    stale_time(60000)
]).

test_mutation_config([
    name(createUser),
    endpoint('/api/users'),
    method('POST')
]).

test_persistence_config([
    key(userPrefs),
    schema('{ theme: string }')
]).

% ============================================================================
% Tests: Navigation Patterns
% ============================================================================

:- begin_tests(navigation_multi_target).

test(react_native_tab_nav) :-
    test_screens(Screens),
    ui_patterns:compile_navigation_pattern(tab, Screens, [], react_native, [], Code),
    sub_string(Code, _, _, _, "createBottomTabNavigator"),
    sub_string(Code, _, _, _, "Tab.Screen").

test(vue_tab_nav) :-
    test_screens(Screens),
    vue_target:compile_navigation_pattern(tab, Screens, [], vue, [], Code),
    sub_string(Code, _, _, _, "router-link"),
    sub_string(Code, _, _, _, "tab-bar").

test(flutter_tab_nav) :-
    test_screens(Screens),
    flutter_target:compile_navigation_pattern(tab, Screens, [], flutter, [], Code),
    sub_string(Code, _, _, _, "BottomNavigationBar"),
    sub_string(Code, _, _, _, "StatefulWidget").

test(swiftui_tab_nav) :-
    test_screens(Screens),
    swiftui_target:compile_navigation_pattern(tab, Screens, [], swiftui, [], Code),
    sub_string(Code, _, _, _, "TabView"),
    sub_string(Code, _, _, _, "tabItem").

test(react_native_stack_nav) :-
    test_screens(Screens),
    ui_patterns:compile_navigation_pattern(stack, Screens, [], react_native, [], Code),
    sub_string(Code, _, _, _, "createStackNavigator"),
    sub_string(Code, _, _, _, "Stack.Navigator").

test(vue_stack_nav) :-
    test_screens(Screens),
    vue_target:compile_navigation_pattern(stack, Screens, [], vue, [], Code),
    sub_string(Code, _, _, _, "createRouter"),
    sub_string(Code, _, _, _, "RouteRecordRaw").

test(flutter_stack_nav) :-
    test_screens(Screens),
    flutter_target:compile_navigation_pattern(stack, Screens, [], flutter, [], Code),
    sub_string(Code, _, _, _, "GoRouter"),
    sub_string(Code, _, _, _, "GoRoute").

test(swiftui_stack_nav) :-
    test_screens(Screens),
    swiftui_target:compile_navigation_pattern(stack, Screens, [], swiftui, [], Code),
    sub_string(Code, _, _, _, "NavigationStack"),
    sub_string(Code, _, _, _, "navigationDestination").

:- end_tests(navigation_multi_target).

% ============================================================================
% Tests: State Patterns
% ============================================================================

:- begin_tests(state_multi_target).

test(react_native_global_state) :-
    test_global_state(Shape),
    ui_patterns:compile_state_pattern(global, Shape, [], react_native, [], Code),
    sub_string(Code, _, _, _, "zustand"),
    sub_string(Code, _, _, _, "create").

test(vue_global_state) :-
    test_global_state(Shape),
    vue_target:compile_state_pattern(global, Shape, [], vue, [], Code),
    sub_string(Code, _, _, _, "defineStore"),
    sub_string(Code, _, _, _, "pinia").

test(flutter_global_state) :-
    test_global_state(Shape),
    flutter_target:compile_state_pattern(global, Shape, [], flutter, [], Code),
    sub_string(Code, _, _, _, "StateNotifier"),
    sub_string(Code, _, _, _, "StateNotifierProvider").

test(swiftui_global_state) :-
    test_global_state(Shape),
    swiftui_target:compile_state_pattern(global, Shape, [], swiftui, [], Code),
    sub_string(Code, _, _, _, "ObservableObject"),
    sub_string(Code, _, _, _, "@Published").

test(react_native_local_state) :-
    ui_patterns:compile_state_pattern(local, [field(count, 0)], [], react_native, [], Code),
    sub_string(Code, _, _, _, "useState").

test(vue_local_state) :-
    vue_target:compile_state_pattern(local, [field(count, 0)], [], vue, [], Code),
    sub_string(Code, _, _, _, "ref").

test(flutter_local_state) :-
    flutter_target:compile_state_pattern(local, [field(count, 0)], [], flutter, [], Code),
    sub_string(Code, _, _, _, "StatefulWidget"),
    sub_string(Code, _, _, _, "setState").

test(swiftui_local_state) :-
    swiftui_target:compile_state_pattern(local, [field(count, 0)], [], swiftui, [], Code),
    sub_string(Code, _, _, _, "@State").

:- end_tests(state_multi_target).

% ============================================================================
% Tests: Data Patterns
% ============================================================================

:- begin_tests(data_multi_target).

test(react_native_query) :-
    test_query_config(Config),
    ui_patterns:compile_data_pattern(query, Config, react_native, [], Code),
    sub_string(Code, _, _, _, "useQuery"),
    sub_string(Code, _, _, _, "@tanstack/react-query").

test(vue_query) :-
    test_query_config(Config),
    vue_target:compile_data_pattern(query, Config, vue, [], Code),
    sub_string(Code, _, _, _, "useQuery"),
    sub_string(Code, _, _, _, "@tanstack/vue-query").

test(flutter_query) :-
    test_query_config(Config),
    flutter_target:compile_data_pattern(query, Config, flutter, [], Code),
    sub_string(Code, _, _, _, "FutureProvider"),
    sub_string(Code, _, _, _, "http.get").

test(swiftui_query) :-
    test_query_config(Config),
    swiftui_target:compile_data_pattern(query, Config, swiftui, [], Code),
    sub_string(Code, _, _, _, "URLSession"),
    sub_string(Code, _, _, _, "async").

test(react_native_mutation) :-
    test_mutation_config(Config),
    ui_patterns:compile_data_pattern(mutation, Config, react_native, [], Code),
    sub_string(Code, _, _, _, "useMutation").

test(vue_mutation) :-
    test_mutation_config(Config),
    vue_target:compile_data_pattern(mutation, Config, vue, [], Code),
    sub_string(Code, _, _, _, "useMutation").

test(flutter_mutation) :-
    test_mutation_config(Config),
    flutter_target:compile_data_pattern(mutation, Config, flutter, [], Code),
    sub_string(Code, _, _, _, "mutate"),
    sub_string(Code, _, _, _, "StateNotifier").

test(swiftui_mutation) :-
    test_mutation_config(Config),
    swiftui_target:compile_data_pattern(mutation, Config, swiftui, [], Code),
    sub_string(Code, _, _, _, "httpMethod"),
    sub_string(Code, _, _, _, "POST").

:- end_tests(data_multi_target).

% ============================================================================
% Tests: Persistence Patterns
% ============================================================================

:- begin_tests(persistence_multi_target).

test(react_native_persistence) :-
    test_persistence_config(Config),
    ui_patterns:compile_persistence_pattern(local, Config, react_native, [], Code),
    sub_string(Code, _, _, _, "AsyncStorage").

test(vue_persistence) :-
    test_persistence_config(Config),
    vue_target:compile_persistence_pattern(local, Config, vue, [], Code),
    sub_string(Code, _, _, _, "localStorage").

test(flutter_persistence) :-
    test_persistence_config(Config),
    flutter_target:compile_persistence_pattern(local, Config, flutter, [], Code),
    sub_string(Code, _, _, _, "SharedPreferences").

test(swiftui_persistence) :-
    test_persistence_config(Config),
    swiftui_target:compile_persistence_pattern(local, Config, swiftui, [], Code),
    sub_string(Code, _, _, _, "UserDefaults").

:- end_tests(persistence_multi_target).

% ============================================================================
% Tests: Target Capabilities
% ============================================================================

:- begin_tests(target_capabilities).

test(vue_has_capabilities) :-
    vue_target:target_capabilities(Caps),
    member(supports(ui_components), Caps).

test(flutter_has_capabilities) :-
    flutter_target:target_capabilities(Caps),
    member(supports(material_design), Caps).

test(swiftui_has_capabilities) :-
    swiftui_target:target_capabilities(Caps),
    member(supports(declarative_ui), Caps).

:- end_tests(target_capabilities).

% ============================================================================
% Tests: Cross-Framework Consistency
% ============================================================================

:- begin_tests(cross_framework_consistency).

% Verify all targets generate non-empty code for the same pattern
test(all_targets_generate_tab_nav) :-
    test_screens(Screens),
    ui_patterns:compile_navigation_pattern(tab, Screens, [], react_native, [], RN),
    vue_target:compile_navigation_pattern(tab, Screens, [], vue, [], Vue),
    flutter_target:compile_navigation_pattern(tab, Screens, [], flutter, [], Flutter),
    swiftui_target:compile_navigation_pattern(tab, Screens, [], swiftui, [], SwiftUI),
    RN \= "",
    Vue \= "",
    Flutter \= "",
    SwiftUI \= "".

test(all_targets_generate_global_state) :-
    test_global_state(Shape),
    ui_patterns:compile_state_pattern(global, Shape, [], react_native, [], RN),
    vue_target:compile_state_pattern(global, Shape, [], vue, [], Vue),
    flutter_target:compile_state_pattern(global, Shape, [], flutter, [], Flutter),
    swiftui_target:compile_state_pattern(global, Shape, [], swiftui, [], SwiftUI),
    RN \= "",
    Vue \= "",
    Flutter \= "",
    SwiftUI \= "".

test(all_targets_generate_query) :-
    test_query_config(Config),
    ui_patterns:compile_data_pattern(query, Config, react_native, [], RN),
    vue_target:compile_data_pattern(query, Config, vue, [], Vue),
    flutter_target:compile_data_pattern(query, Config, flutter, [], Flutter),
    swiftui_target:compile_data_pattern(query, Config, swiftui, [], SwiftUI),
    RN \= "",
    Vue \= "",
    Flutter \= "",
    SwiftUI \= "".

:- end_tests(cross_framework_consistency).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
