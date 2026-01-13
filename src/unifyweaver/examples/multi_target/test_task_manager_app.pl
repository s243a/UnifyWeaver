% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% test_task_manager_app.pl - plunit tests for Task Manager example
%
% Tests the multi-target compilation of a complete app specification.
%
% Run with: swipl -g "run_tests" -t halt test_task_manager_app.pl

:- module(test_task_manager_app, []).

:- use_module(library(plunit)).
:- use_module(task_manager_app).
:- use_module('../../patterns/ui_patterns').

% ============================================================================
% Tests: Pattern Definition
% ============================================================================

:- begin_tests(pattern_definition).

test(defines_all_patterns) :-
    define_task_manager_app,
    task_manager_app:app_patterns(Patterns),
    forall(member(P, Patterns),
           ui_patterns:stored_pattern(P, _, _)).

test(navigation_pattern_has_screens) :-
    define_task_manager_app,
    ui_patterns:stored_pattern(app_navigation, Spec, _),
    Spec = navigation(tab, Screens, _),
    length(Screens, 2).

test(task_store_has_actions) :-
    define_task_manager_app,
    ui_patterns:stored_pattern(task_store, Spec, _),
    Spec = state(global, Shape, _),
    member(slices(Slices), Shape),
    member(slice(tasks, _, Actions), Slices),
    length(Actions, L),
    L >= 4.

test(data_patterns_have_endpoints) :-
    define_task_manager_app,
    ui_patterns:stored_pattern(fetch_tasks, Spec, _),
    Spec = data(query, Config),
    member(endpoint('/api/tasks'), Config).

test(persistence_patterns_have_keys) :-
    define_task_manager_app,
    ui_patterns:stored_pattern(user_prefs, Spec, _),
    Spec = persistence(local, Config),
    member(key(userPrefs), Config).

:- end_tests(pattern_definition).

% ============================================================================
% Tests: React Native Compilation
% ============================================================================

:- begin_tests(react_native_compilation).

test(generates_react_native_code) :-
    compile_task_manager_app(react_native, Code),
    Code \= "".

test(includes_navigation_container) :-
    compile_task_manager_app(react_native, Code),
    sub_string(Code, _, _, _, "NavigationContainer").

test(includes_query_client) :-
    compile_task_manager_app(react_native, Code),
    sub_string(Code, _, _, _, "QueryClient").

test(includes_react_import) :-
    compile_task_manager_app(react_native, Code),
    sub_string(Code, _, _, _, "import React").

test(has_app_export) :-
    compile_task_manager_app(react_native, Code),
    sub_string(Code, _, _, _, "export default function App").

:- end_tests(react_native_compilation).

% ============================================================================
% Tests: Vue Compilation
% ============================================================================

:- begin_tests(vue_compilation).

test(generates_vue_code) :-
    compile_task_manager_app(vue, Code),
    Code \= "".

test(includes_vue_imports) :-
    compile_task_manager_app(vue, Code),
    sub_string(Code, _, _, _, "createApp").

test(includes_pinia) :-
    compile_task_manager_app(vue, Code),
    sub_string(Code, _, _, _, "createPinia").

test(includes_vue_router) :-
    compile_task_manager_app(vue, Code),
    sub_string(Code, _, _, _, "createRouter").

test(has_script_setup) :-
    compile_task_manager_app(vue, Code),
    sub_string(Code, _, _, _, "<script setup").

:- end_tests(vue_compilation).

% ============================================================================
% Tests: Flutter Compilation
% ============================================================================

:- begin_tests(flutter_compilation).

test(generates_flutter_code) :-
    compile_task_manager_app(flutter, Code),
    Code \= "".

test(includes_material_import) :-
    compile_task_manager_app(flutter, Code),
    sub_string(Code, _, _, _, "package:flutter/material.dart").

test(includes_riverpod) :-
    compile_task_manager_app(flutter, Code),
    sub_string(Code, _, _, _, "flutter_riverpod").

test(includes_go_router) :-
    compile_task_manager_app(flutter, Code),
    sub_string(Code, _, _, _, "go_router").

test(has_main_function) :-
    compile_task_manager_app(flutter, Code),
    sub_string(Code, _, _, _, "void main()").

test(has_provider_scope) :-
    compile_task_manager_app(flutter, Code),
    sub_string(Code, _, _, _, "ProviderScope").

:- end_tests(flutter_compilation).

% ============================================================================
% Tests: SwiftUI Compilation
% ============================================================================

:- begin_tests(swiftui_compilation).

test(generates_swiftui_code) :-
    compile_task_manager_app(swiftui, Code),
    Code \= "".

test(includes_swiftui_import) :-
    compile_task_manager_app(swiftui, Code),
    sub_string(Code, _, _, _, "import SwiftUI").

test(has_app_struct) :-
    compile_task_manager_app(swiftui, Code),
    sub_string(Code, _, _, _, "struct TaskManagerApp: App").

test(has_state_object) :-
    compile_task_manager_app(swiftui, Code),
    sub_string(Code, _, _, _, "@StateObject").

test(has_window_group) :-
    compile_task_manager_app(swiftui, Code),
    sub_string(Code, _, _, _, "WindowGroup").

:- end_tests(swiftui_compilation).

% ============================================================================
% Tests: All Targets
% ============================================================================

:- begin_tests(all_targets).

test(compile_all_returns_four_results) :-
    compile_all_targets(Results),
    length(Results, 4).

test(all_targets_have_code) :-
    compile_all_targets(Results),
    forall(member(result(_, Code), Results),
           Code \= "// Compilation failed").

test(targets_are_distinct) :-
    compile_all_targets(Results),
    findall(T, member(result(T, _), Results), Targets),
    sort(Targets, Sorted),
    length(Sorted, 4).

test(each_target_code_is_different) :-
    compile_task_manager_app(react_native, RN),
    compile_task_manager_app(vue, Vue),
    compile_task_manager_app(flutter, Flutter),
    compile_task_manager_app(swiftui, Swift),
    RN \= Vue,
    Vue \= Flutter,
    Flutter \= Swift.

:- end_tests(all_targets).

% ============================================================================
% Tests: Code Quality
% ============================================================================

:- begin_tests(code_quality).

test(react_native_code_not_empty) :-
    compile_task_manager_app(react_native, Code),
    string_length(Code, Len),
    Len > 500.

test(vue_code_not_empty) :-
    compile_task_manager_app(vue, Code),
    string_length(Code, Len),
    Len > 300.

test(flutter_code_not_empty) :-
    compile_task_manager_app(flutter, Code),
    string_length(Code, Len),
    Len > 300.

test(swiftui_code_not_empty) :-
    compile_task_manager_app(swiftui, Code),
    string_length(Code, Len),
    Len > 200.

:- end_tests(code_quality).

% ============================================================================
% Run tests when loaded directly
% ============================================================================

:- initialization(run_tests, main).
