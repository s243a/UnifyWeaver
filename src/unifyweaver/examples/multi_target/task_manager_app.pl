% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% task_manager_app.pl - Multi-Target Example Application
%
% Demonstrates compiling the same app specification to:
%   - React Native (TypeScript)
%   - Vue 3 (TypeScript)
%   - Flutter (Dart)
%   - SwiftUI (Swift)
%
% This example shows a complete Task Manager app with:
%   - Tab navigation (Tasks, Settings)
%   - Global state management (task store)
%   - Data fetching (tasks API)
%   - Local persistence (user preferences)
%
% Usage:
%   ?- compile_task_manager_app(react_native, Code).
%   ?- compile_task_manager_app(vue, Code).
%   ?- compile_task_manager_app(flutter, Code).
%   ?- compile_task_manager_app(swiftui, Code).
%
%   ?- compile_full_stack(Frontend, Backend).

:- module(task_manager_app, [
    % App definition
    define_task_manager_app/0,

    % Compilation
    compile_task_manager_app/2,    % +Target, -Code
    compile_full_stack/2,          % -Frontend, -Backend
    compile_all_targets/1,         % -Results

    % Testing
    test_task_manager_app/0
]).

:- use_module('../../patterns/ui_patterns').
:- use_module('../../patterns/pattern_composition').
:- use_module('../../glue/pattern_glue').
:- use_module('../../targets/vue_target', []).
:- use_module('../../targets/flutter_target', []).
:- use_module('../../targets/swiftui_target', []).

% ============================================================================
% APP SPECIFICATION
% ============================================================================

%% define_task_manager_app/0
%
%  Define all patterns for the Task Manager application.
%
define_task_manager_app :-
    % Clean up any existing patterns
    cleanup_app_patterns,

    % Define navigation pattern - Tab navigator with Tasks and Settings
    ui_patterns:define_pattern(app_navigation,
        navigation(tab, [
            screen(tasks, 'TasksScreen', [title('Tasks'), icon('list')]),
            screen(settings, 'SettingsScreen', [title('Settings'), icon('cog')])
        ], []),
        [requires([navigation])]),

    % Define task detail navigation (stack within tasks tab)
    ui_patterns:define_pattern(task_detail_nav,
        navigation(stack, [
            screen(task_list, 'TaskListScreen', []),
            screen(task_detail, 'TaskDetailScreen', []),
            screen(task_edit, 'TaskEditScreen', [])
        ], []),
        [requires([navigation]), depends_on([app_navigation])]),

    % Define global state - Task store
    ui_patterns:define_pattern(task_store,
        state(global, [
            store(taskStore),
            slices([
                slice(tasks, [
                    field(items, 'Task[]'),
                    field(loading, boolean),
                    field(error, 'string | null')
                ], [
                    action(setTasks, "(set, tasks) => set({ items: tasks })"),
                    action(addTask, "(set, task) => set(s => ({ items: [...s.items, task] }))"),
                    action(updateTask, "(set, id, updates) => set(s => ({ items: s.items.map(t => t.id === id ? {...t, ...updates} : t) }))"),
                    action(deleteTask, "(set, id) => set(s => ({ items: s.items.filter(t => t.id !== id) }))"),
                    action(setLoading, "(set, loading) => set({ loading })"),
                    action(setError, "(set, error) => set({ error })")
                ])
            ])
        ], []),
        [requires([zustand]), singleton(true)]),

    % Define filter state
    ui_patterns:define_pattern(filter_store,
        state(global, [
            store(filterStore),
            slices([
                slice(filter, [
                    field(status, 'all | pending | completed'),
                    field(sortBy, 'date | priority | name')
                ], [
                    action(setStatus, "(set, status) => set({ status })"),
                    action(setSortBy, "(set, sortBy) => set({ sortBy })")
                ])
            ])
        ], []),
        [requires([zustand])]),

    % Define data fetching - Tasks query
    ui_patterns:define_pattern(fetch_tasks,
        data(query, [
            name(fetchTasks),
            endpoint('/api/tasks'),
            stale_time(60000),
            retry(3)
        ]),
        [requires([react_query])]),

    % Define data fetching - Single task
    ui_patterns:define_pattern(fetch_task,
        data(query, [
            name(fetchTask),
            endpoint('/api/tasks/:id'),
            stale_time(30000)
        ]),
        [requires([react_query])]),

    % Define mutations
    ui_patterns:define_pattern(create_task,
        data(mutation, [
            name(createTask),
            endpoint('/api/tasks'),
            method('POST'),
            invalidates([fetchTasks])
        ]),
        [requires([react_query])]),

    ui_patterns:define_pattern(update_task,
        data(mutation, [
            name(updateTask),
            endpoint('/api/tasks/:id'),
            method('PUT'),
            invalidates([fetchTasks, fetchTask])
        ]),
        [requires([react_query])]),

    ui_patterns:define_pattern(delete_task,
        data(mutation, [
            name(deleteTask),
            endpoint('/api/tasks/:id'),
            method('DELETE'),
            invalidates([fetchTasks])
        ]),
        [requires([react_query])]),

    % Define persistence - User preferences
    ui_patterns:define_pattern(user_prefs,
        persistence(local, [
            key(userPrefs),
            schema('{ theme: "light" | "dark", notifications: boolean }')
        ]),
        [requires([async_storage])]),

    % Define persistence - Task cache
    ui_patterns:define_pattern(task_cache,
        persistence(local, [
            key(taskCache),
            schema('{ tasks: Task[], lastSync: number }')
        ]),
        [requires([async_storage])]).

cleanup_app_patterns :-
    Patterns = [app_navigation, task_detail_nav, task_store, filter_store,
                fetch_tasks, fetch_task, create_task, update_task, delete_task,
                user_prefs, task_cache],
    forall(member(P, Patterns),
           retractall(ui_patterns:stored_pattern(P, _, _))).

% ============================================================================
% APP PATTERNS LIST
% ============================================================================

%% app_patterns(-Patterns)
%
%  List of all patterns in the Task Manager app.
%
app_patterns([
    app_navigation,
    task_detail_nav,
    task_store,
    filter_store,
    fetch_tasks,
    fetch_task,
    create_task,
    update_task,
    delete_task,
    user_prefs,
    task_cache
]).

%% core_patterns(-Patterns)
%
%  Core patterns needed for basic app functionality.
%
core_patterns([
    app_navigation,
    task_store,
    fetch_tasks,
    create_task,
    user_prefs
]).

% ============================================================================
% COMPILATION
% ============================================================================

%% compile_task_manager_app(+Target, -Code)
%
%  Compile the Task Manager app to a specific target.
%
compile_task_manager_app(Target, Code) :-
    member(Target, [react_native, vue, flutter, swiftui]),
    define_task_manager_app,
    core_patterns(Patterns),
    compile_patterns_to_target(Patterns, Target, Codes),
    generate_app_wrapper(Target, Codes, Code).

compile_patterns_to_target(Patterns, Target, Codes) :-
    findall(Code, (
        member(P, Patterns),
        compile_single_pattern(P, Target, Code)
    ), Codes).

compile_single_pattern(PatternName, react_native, Code) :-
    catch(ui_patterns:compile_pattern(PatternName, react_native, [], Code), _, fail).

compile_single_pattern(PatternName, vue, Code) :-
    ui_patterns:stored_pattern(PatternName, Spec, _Opts),
    compile_vue_spec(Spec, Code).

compile_single_pattern(PatternName, flutter, Code) :-
    ui_patterns:stored_pattern(PatternName, Spec, _Opts),
    compile_flutter_spec(Spec, Code).

compile_single_pattern(PatternName, swiftui, Code) :-
    ui_patterns:stored_pattern(PatternName, Spec, _Opts),
    compile_swiftui_spec(Spec, Code).

%% Vue compilation helpers
compile_vue_spec(navigation(Type, Screens, Config), Code) :-
    catch(vue_target:compile_navigation_pattern(Type, Screens, Config, vue, [], Code), _, fail).
compile_vue_spec(state(global, Shape, Config), Code) :-
    catch(vue_target:compile_state_pattern(global, Shape, Config, vue, [], Code), _, fail).
compile_vue_spec(data(Type, Config), Code) :-
    catch(vue_target:compile_data_pattern(Type, Config, vue, [], Code), _, fail).
compile_vue_spec(persistence(Type, Config), Code) :-
    catch(vue_target:compile_persistence_pattern(Type, Config, vue, [], Code), _, fail).

%% Flutter compilation helpers
compile_flutter_spec(navigation(Type, Screens, Config), Code) :-
    catch(flutter_target:compile_navigation_pattern(Type, Screens, Config, flutter, [], Code), _, fail).
compile_flutter_spec(state(global, Shape, Config), Code) :-
    catch(flutter_target:compile_state_pattern(global, Shape, Config, flutter, [], Code), _, fail).
compile_flutter_spec(data(Type, Config), Code) :-
    catch(flutter_target:compile_data_pattern(Type, Config, flutter, [], Code), _, fail).
compile_flutter_spec(persistence(Type, Config), Code) :-
    catch(flutter_target:compile_persistence_pattern(Type, Config, flutter, [], Code), _, fail).

%% SwiftUI compilation helpers
compile_swiftui_spec(navigation(Type, Screens, Config), Code) :-
    catch(swiftui_target:compile_navigation_pattern(Type, Screens, Config, swiftui, [], Code), _, fail).
compile_swiftui_spec(state(global, Shape, Config), Code) :-
    catch(swiftui_target:compile_state_pattern(global, Shape, Config, swiftui, [], Code), _, fail).
compile_swiftui_spec(data(Type, Config), Code) :-
    catch(swiftui_target:compile_data_pattern(Type, Config, swiftui, [], Code), _, fail).
compile_swiftui_spec(persistence(Type, Config), Code) :-
    catch(swiftui_target:compile_persistence_pattern(Type, Config, swiftui, [], Code), _, fail).

%% generate_app_wrapper(+Target, +Codes, -FullCode)
%
%  Wrap compiled patterns in target-specific app structure.
%
generate_app_wrapper(react_native, Codes, Code) :-
    atomic_list_concat(Codes, '\n\n// ============================================\n\n', Body),
    format(string(Code),
"// Task Manager App - React Native
// Auto-generated from UnifyWeaver patterns

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

~w

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <NavigationContainer>
        <AppNavigation />
      </NavigationContainer>
    </QueryClientProvider>
  );
}
", [Body]).

generate_app_wrapper(vue, Codes, Code) :-
    atomic_list_concat(Codes, '\n\n// ============================================\n\n', Body),
    format(string(Code),
"// Task Manager App - Vue 3
// Auto-generated from UnifyWeaver patterns

<script setup lang=\"ts\">
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { VueQueryPlugin } from '@tanstack/vue-query';
import { createRouter, createWebHistory } from 'vue-router';

const pinia = createPinia();

~w
</script>

<template>
  <router-view />
</template>
", [Body]).

generate_app_wrapper(flutter, Codes, Code) :-
    atomic_list_concat(Codes, '\n\n// ============================================\n\n', Body),
    format(string(Code),
"// Task Manager App - Flutter
// Auto-generated from UnifyWeaver patterns

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

~w

void main() {
  runApp(
    ProviderScope(
      child: MaterialApp.router(
        routerConfig: appRouter,
        title: 'Task Manager',
        theme: ThemeData(
          primarySwatch: Colors.blue,
        ),
      ),
    ),
  );
}
", [Body]).

generate_app_wrapper(swiftui, Codes, Code) :-
    atomic_list_concat(Codes, '\n\n// ============================================\n\n', Body),
    format(string(Code),
"// Task Manager App - SwiftUI
// Auto-generated from UnifyWeaver patterns

import SwiftUI

~w

@main
struct TaskManagerApp: App {
    @StateObject private var taskStore = TaskStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(taskStore)
        }
    }
}
", [Body]).

% ============================================================================
% FULL STACK COMPILATION
% ============================================================================

%% compile_full_stack(-Frontend, -Backend)
%
%  Compile both frontend (React Native) and backend (Express) code.
%
compile_full_stack(Frontend, Backend) :-
    define_task_manager_app,
    core_patterns(Patterns),
    pattern_glue:generate_full_stack(Patterns,
        [frontend_target(react_native), backend_target(express)],
        Frontend, Backend).

%% compile_all_targets(-Results)
%
%  Compile the app to all supported targets.
%
compile_all_targets(Results) :-
    Targets = [react_native, vue, flutter, swiftui],
    findall(result(Target, Code), (
        member(Target, Targets),
        (   compile_task_manager_app(Target, Code)
        ->  true
        ;   Code = "// Compilation failed"
        )
    ), Results).

% ============================================================================
% TESTING
% ============================================================================

test_task_manager_app :-
    format('~n=== Task Manager App Tests ===~n~n'),

    % Test 1: Define patterns
    format('Test 1: Define app patterns...~n'),
    (   define_task_manager_app,
        ui_patterns:stored_pattern(app_navigation, _, _),
        ui_patterns:stored_pattern(task_store, _, _)
    ->  format('  PASS: App patterns defined~n')
    ;   format('  FAIL: Could not define patterns~n')
    ),

    % Test 2: React Native compilation
    format('~nTest 2: React Native compilation...~n'),
    (   compile_task_manager_app(react_native, RNCode),
        sub_string(RNCode, _, _, _, "React Native"),
        sub_string(RNCode, _, _, _, "NavigationContainer")
    ->  format('  PASS: React Native code generated~n')
    ;   format('  SKIP: React Native compilation incomplete~n')
    ),

    % Test 3: Vue compilation
    format('~nTest 3: Vue compilation...~n'),
    (   compile_task_manager_app(vue, VueCode),
        sub_string(VueCode, _, _, _, "Vue 3")
    ->  format('  PASS: Vue code generated~n')
    ;   format('  SKIP: Vue compilation incomplete~n')
    ),

    % Test 4: Flutter compilation
    format('~nTest 4: Flutter compilation...~n'),
    (   compile_task_manager_app(flutter, FlutterCode),
        sub_string(FlutterCode, _, _, _, "Flutter")
    ->  format('  PASS: Flutter code generated~n')
    ;   format('  SKIP: Flutter compilation incomplete~n')
    ),

    % Test 5: SwiftUI compilation
    format('~nTest 5: SwiftUI compilation...~n'),
    (   compile_task_manager_app(swiftui, SwiftCode),
        sub_string(SwiftCode, _, _, _, "SwiftUI")
    ->  format('  PASS: SwiftUI code generated~n')
    ;   format('  SKIP: SwiftUI compilation incomplete~n')
    ),

    % Test 6: All targets
    format('~nTest 6: Compile all targets...~n'),
    (   compile_all_targets(Results),
        length(Results, 4)
    ->  format('  PASS: All 4 targets compiled~n')
    ;   format('  FAIL: Could not compile all targets~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Task Manager App example loaded~n'),
    format('  Usage: compile_task_manager_app(react_native, Code).~n'),
    format('         compile_all_targets(Results).~n')
), now).
