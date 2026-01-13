% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% target_interface.pl - Common Interface for All Code Generation Targets
%
% Defines the contract that each target (React Native, Vue, Flutter, SwiftUI)
% must implement to compile UI patterns. This enables framework-agnostic
% pattern definitions that compile to multiple targets.
%
% Each target must implement:
%   1. compile_navigation_pattern/6 - Stack, Tab, Drawer navigation
%   2. compile_state_pattern/6      - Local, Global, Derived state
%   3. compile_data_pattern/5       - Query, Mutation, Infinite
%   4. compile_persistence_pattern/5 - Local, Secure storage
%   5. target_capabilities/2        - What the target supports
%
% Pattern types are defined in ui_patterns.pl and are target-agnostic.

:- module(target_interface, [
    % Target registration
    register_target/2,              % +TargetName, +Module
    registered_target/2,            % ?TargetName, ?Module

    % Capability checking
    target_supports/2,              % +Target, +Capability
    target_library/2,               % +Target, +Library
    target_limitation/2,            % +Target, +Limitation

    % Framework mappings - common concepts across frameworks
    state_hook_equivalent/2,        % +Target, -Equivalent
    query_hook_equivalent/2,        % +Target, -Equivalent
    navigation_equivalent/2,        % +Target, -Equivalent
    persistence_equivalent/2,       % +Target, -Equivalent

    % Testing
    test_target_interface/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic registered_target_/2.    % registered_target_(Name, Module)

% ============================================================================
% TARGET REGISTRATION
% ============================================================================

%% register_target(+TargetName, +Module)
%
%  Register a target module for compilation.
%
register_target(Name, Module) :-
    atom(Name),
    atom(Module),
    (   registered_target_(Name, _)
    ->  retract(registered_target_(Name, _))
    ;   true
    ),
    assertz(registered_target_(Name, Module)).

%% registered_target(?Name, ?Module)
%
%  Query registered targets.
%
registered_target(Name, Module) :-
    registered_target_(Name, Module).

% ============================================================================
% FRAMEWORK EQUIVALENCE MAPPINGS
% ============================================================================
%
% These mappings document how common UI patterns translate across frameworks.
% Each target implements these differently, but the concepts are equivalent.

%% state_hook_equivalent(+Target, -Equivalent)
%
%  What each framework uses for local component state.
%
state_hook_equivalent(react_native, 'useState').
state_hook_equivalent(vue, 'ref/reactive').
state_hook_equivalent(flutter, 'StatefulWidget/setState').
state_hook_equivalent(swiftui, '@State').

%% query_hook_equivalent(+Target, -Equivalent)
%
%  What each framework uses for data fetching with caching.
%
query_hook_equivalent(react_native, '@tanstack/react-query').
query_hook_equivalent(vue, '@tanstack/vue-query').
query_hook_equivalent(flutter, 'FutureBuilder/riverpod').
query_hook_equivalent(swiftui, 'async/await + @State').

%% navigation_equivalent(+Target, -Equivalent)
%
%  What each framework uses for navigation.
%
navigation_equivalent(react_native, '@react-navigation').
navigation_equivalent(vue, 'vue-router').
navigation_equivalent(flutter, 'Navigator/GoRouter').
navigation_equivalent(swiftui, 'NavigationStack').

%% persistence_equivalent(+Target, -Equivalent)
%
%  What each framework uses for local persistence.
%
persistence_equivalent(react_native, '@react-native-async-storage').
persistence_equivalent(vue, 'localStorage/pinia-persist').
persistence_equivalent(flutter, 'shared_preferences/hive').
persistence_equivalent(swiftui, 'UserDefaults/@AppStorage').

% ============================================================================
% CAPABILITY CHECKING
% ============================================================================

%% target_supports(+Target, +Capability)
%
%  Check if a target supports a capability.
%
target_supports(Target, Capability) :-
    registered_target(Target, Module),
    Goal =.. [target_capabilities, Caps],
    call(Module:Goal),
    member(supports(Capability), Caps).

%% target_library(+Target, +Library)
%
%  Get libraries used by a target.
%
target_library(Target, Library) :-
    registered_target(Target, Module),
    Goal =.. [target_capabilities, Caps],
    call(Module:Goal),
    member(library(Library), Caps).

%% target_limitation(+Target, +Limitation)
%
%  Get limitations of a target.
%
target_limitation(Target, Limitation) :-
    registered_target(Target, Module),
    Goal =.. [target_capabilities, Caps],
    call(Module:Goal),
    member(limitation(Limitation), Caps).

% ============================================================================
% REQUIRED PREDICATES DOCUMENTATION
% ============================================================================
%
% Each target module MUST export these predicates:
%
% target_capabilities(-Capabilities)
%   Returns list of: supports(X), library(X), limitation(X), glue_required(X)
%
% compile_navigation_pattern(+Type, +Screens, +Config, +Target, +Options, -Code)
%   Type: stack | tab | drawer
%   Screens: [screen(Name, Component, Opts), ...]
%   Generates: Navigation component/widget for the target framework
%
% compile_state_pattern(+Type, +Shape, +Config, +Target, +Options, -Code)
%   Type: local | global | derived
%   Shape: Pattern-specific state shape
%   Generates: State management code for the target framework
%
% compile_data_pattern(+Type, +Config, +Target, +Options, -Code)
%   Type: query | mutation | infinite
%   Config: [name(N), endpoint(E), ...]
%   Generates: Data fetching hooks/widgets for the target framework
%
% compile_persistence_pattern(+Type, +Config, +Target, +Options, -Code)
%   Type: local | secure
%   Config: [key(K), schema(S), ...]
%   Generates: Persistence hooks/widgets for the target framework

% ============================================================================
% TESTING
% ============================================================================

test_target_interface :-
    format('~n=== Target Interface Tests ===~n~n'),

    % Test 1: Framework equivalence mappings
    format('Test 1: Framework equivalence mappings...~n'),
    (   state_hook_equivalent(react_native, RNState),
        state_hook_equivalent(vue, VueState),
        state_hook_equivalent(flutter, FlutterState),
        state_hook_equivalent(swiftui, SwiftState),
        RNState \= VueState,
        FlutterState \= SwiftState
    ->  format('  PASS: All frameworks have state equivalents~n')
    ;   format('  FAIL: Missing state equivalents~n')
    ),

    % Test 2: Query equivalence mappings
    format('~nTest 2: Query equivalence mappings...~n'),
    (   query_hook_equivalent(react_native, _),
        query_hook_equivalent(vue, _),
        query_hook_equivalent(flutter, _),
        query_hook_equivalent(swiftui, _)
    ->  format('  PASS: All frameworks have query equivalents~n')
    ;   format('  FAIL: Missing query equivalents~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Target interface module loaded~n', [])
), now).
