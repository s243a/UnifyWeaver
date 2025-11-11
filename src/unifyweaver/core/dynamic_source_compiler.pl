:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% dynamic_source_compiler.pl - Core infrastructure for dynamic data sources
% Provides plugin registration and compilation dispatch

:- module(dynamic_source_compiler, [
    register_source_type/2,         % +Type, +Module
    compile_dynamic_source/3,       % +Pred/Arity, +Options, -BashCode
    is_dynamic_source/1,            % +Pred/Arity
    list_source_types/1,            % -Types
    register_dynamic_source/3,      % +Pred/Arity, +SourceSpec, +Options
    dynamic_source_def/3,           % +Pred/Arity, -Type, -Config (multifile)
    test_dynamic_source_compiler/0
]).

:- use_module(library(lists)).
:- use_module(firewall).

%% ============================================
%% DYNAMIC PREDICATES
%% ============================================

:- dynamic source_type_registry/2.  % source_type_registry(Type, Module)
:- dynamic dynamic_source_def/3.     % dynamic_source_def(Pred/Arity, Type, Config)
:- multifile dynamic_source_def/3.   % Allow plugins to add facts

%% ============================================
%% SOURCE TYPE REGISTRATION
%% ============================================

%% register_source_type(+Type, +Module)
%  Register a source plugin module
%  Called by plugins during initialization
register_source_type(Type, Module) :-
    (   source_type_registry(Type, ExistingModule)
    ->  (   ExistingModule = Module
        ->  true  % Already registered by same module
        ;   format('Warning: Source type ~w already registered by ~w, overriding with ~w~n',
                   [Type, ExistingModule, Module]),
            retract(source_type_registry(Type, ExistingModule)),
            assertz(source_type_registry(Type, Module))
        )
    ;   assertz(source_type_registry(Type, Module)),
        format('Registered source type: ~w -> ~w~n', [Type, Module])
    ).

%% list_source_types(-Types)
%  Get list of all registered source types
list_source_types(Types) :-
    findall(Type, source_type_registry(Type, _), Types).

%% get_source_module(+Type, -Module)
%  Get module for a registered source type
get_source_module(Type, Module) :-
    source_type_registry(Type, Module),
    !.
get_source_module(Type, _) :-
    format('Error: Source type ~w not registered~n', [Type]),
    fail.

%% ============================================
%% DYNAMIC SOURCE DEFINITION
%% ============================================

%% register_dynamic_source(+Pred/Arity, +SourceSpec, +Options)
%  Register a predicate as using a dynamic source
%  SourceSpec is either:
%    - Type (atom) - e.g., awk
%    - type(Type, Config) - e.g., type(awk, [command='awk ...'])
register_dynamic_source(Pred/Arity, SourceSpec, Options) :-
    % Parse SourceSpec
    (   SourceSpec = type(Type, Config)
    ->  true
    ;   Type = SourceSpec,
        Config = []
    ),

    % Merge config with options
    append(Config, Options, MergedConfig),

    % Enforce firewall policy at declaration time
    enforce_firewall_policy(Pred/Arity, MergedConfig),

    % Store definition
    retractall(dynamic_source_def(Pred/Arity, _, _)),
    assertz(dynamic_source_def(Pred/Arity, Type, MergedConfig)),

    format('Registered dynamic source: ~w/~w using ~w~n', [Pred, Arity, Type]).

%% is_dynamic_source(+Pred/Arity)
%  Check if predicate is registered as dynamic source
is_dynamic_source(Pred/Arity) :-
    dynamic_source_def(Pred/Arity, _, _).

%% ============================================
%% COMPILATION
%% ============================================

%% compile_dynamic_source(+Pred/Arity, +Options, -BashCode)
%  Compile a dynamic source predicate to bash code
compile_dynamic_source(Pred/Arity, RuntimeOptions, BashCode) :-
    % Get source definition
    dynamic_source_def(Pred/Arity, Type, Config),

    % Get source plugin module
    get_source_module(Type, Module),

    % Merge runtime options with stored config
    append(RuntimeOptions, Config, AllOptions),

    % Re-validate against firewall in case runtime options add new capabilities
    enforce_firewall_policy(Pred/Arity, AllOptions),

    format('Compiling dynamic source: ~w/~w using ~w~n', [Pred, Arity, Type]),

    % Call plugin's compile predicate
    % Plugin interface: compile_source(+Pred/Arity, +Config, +Options, -BashCode)
    Module:compile_source(Pred/Arity, Config, AllOptions, BashCode).

%% ============================================
%% FIREWALL ENFORCEMENT
%% ============================================

enforce_firewall_policy(PredIndicator, Options) :-
    firewall:get_firewall_policy(PredIndicator, Firewall),
    resolve_target(Options, Target),
    firewall:enforce_firewall(PredIndicator, Target, Options, Firewall).

resolve_target(Options, Target) :-
    (   member(target(TargetOption), Options)
    ->  Target = TargetOption
    ;   Target = bash
    ).

%% ============================================
%% TESTS
%% ============================================

test_dynamic_source_compiler :-
    writeln('=== Testing Dynamic Source Compiler ==='),

    % Test 1: Register source type
    write('Test 1 - Register source type: '),
    register_source_type(test_source, test_module),
    (   source_type_registry(test_source, test_module)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Test 2: List source types
    write('Test 2 - List source types: '),
    list_source_types(Types),
    (   member(test_source, Types)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Test 3: Register dynamic source
    write('Test 3 - Register dynamic source: '),
    register_dynamic_source(my_data/2, test_source, [option1=value1]),
    (   is_dynamic_source(my_data/2)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Test 4: Check source definition
    write('Test 4 - Check source definition: '),
    (   dynamic_source_def(my_data/2, test_source, Config),
        member(option1=value1, Config)
    ->  writeln('PASS')
    ;   writeln('FAIL')
    ),

    % Clean up
    retractall(source_type_registry(test_source, _)),
    retractall(dynamic_source_def(my_data/2, _, _)),

    writeln('=== Dynamic Source Compiler Tests Complete ===').
