% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
% binding_registry.pl - Foreign Function Binding Registry
%
% This module manages bindings that map Prolog predicates to target language
% functions. Bindings preserve semantic information about effects, types,
% and execution characteristics needed for correct code generation.
%
% See: docs/proposals/BINDING_PREDICATE_PROPOSAL.md

:- module(binding_registry, [
    % Core binding API
    binding/6,                      % binding(Target, Pred, TargetName, Inputs, Outputs, Options)
    declare_binding/6,              % declare_binding(Target, Pred, TargetName, Inputs, Outputs, Options)
    clear_binding/2,                % clear_binding(Target, Pred)
    clear_all_bindings/0,           % clear_all_bindings
    clear_all_bindings/1,           % clear_all_bindings(Target)

    % Effect annotations
    effect/2,                       % effect(Pred, Effects)
    declare_effect/2,               % declare_effect(Pred, Effects)
    clear_effect/1,                 % clear_effect(Pred)

    % Design patterns
    pattern/2,                      % pattern(PatternName, Description)
    declare_pattern/2,              % declare_pattern(PatternName, Description)

    % Bidirectional predicates
    bidirectional/2,                % bidirectional(Pred, Modes)
    declare_bidirectional/2,        % declare_bidirectional(Pred, Modes)
    binding_mode/5,                 % binding_mode(Target, Pred, Mode, TargetName, Options)
    declare_binding_mode/5,         % declare_binding_mode(Target, Pred, Mode, TargetName, Options)

    % Query API
    bindings_for_target/2,          % bindings_for_target(Target, Bindings)
    bindings_for_predicate/2,       % bindings_for_predicate(Pred, Bindings)
    resolve_binding/4,              % resolve_binding(Preferences, Pred, Target, Binding)
    binding_imports/2,              % binding_imports(Target, Imports)
    binding_has_effect/3,           % binding_has_effect(Target, Pred, Effect)

    % Utility
    is_pure_binding/2,              % is_pure_binding(Target, Pred)
    is_total_binding/2,             % is_total_binding(Target, Pred)
    is_deterministic_binding/2,     % is_deterministic_binding(Target, Pred)

    % Testing
    test_binding_registry/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_binding/6.        % stored_binding(Target, Pred, TargetName, Inputs, Outputs, Options)
:- dynamic stored_effect/2.         % stored_effect(Pred, Effects)
:- dynamic stored_pattern/2.        % stored_pattern(PatternName, Description)
:- dynamic stored_bidirectional/2.  % stored_bidirectional(Pred, Modes)
:- dynamic stored_binding_mode/5.   % stored_binding_mode(Target, Pred, Mode, TargetName, Options)

% ============================================================================
% CORE BINDING API
% ============================================================================

%% binding(?Target, ?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query bindings from the registry. All arguments can be unbound for queries.
%
%  @param Target      atom - target language (go, bash, python, powershell, etc.)
%  @param Pred        Name/Arity - the Prolog predicate being mapped
%  @param TargetName  string/atom - the target language function/command
%  @param Inputs      list - input argument specifications
%  @param Outputs     list - output argument specifications
%  @param Options     list - options/effects/annotations
%
binding(Target, Pred, TargetName, Inputs, Outputs, Options) :-
    stored_binding(Target, Pred, TargetName, Inputs, Outputs, Options).

%% declare_binding(+Target, +Pred, +TargetName, +Inputs, +Outputs, +Options)
%
%  Declare a new binding. Replaces any existing binding for the same Target/Pred.
%
declare_binding(Target, Pred, TargetName, Inputs, Outputs, Options) :-
    atom(Target),
    validate_pred_indicator(Pred),
    is_list(Inputs),
    is_list(Outputs),
    is_list(Options),
    retractall(stored_binding(Target, Pred, _, _, _, _)),
    assertz(stored_binding(Target, Pred, TargetName, Inputs, Outputs, Options)).

%% clear_binding(+Target, +Pred)
%
%  Remove a specific binding.
%
clear_binding(Target, Pred) :-
    retractall(stored_binding(Target, Pred, _, _, _, _)).

%% clear_all_bindings
%
%  Remove all bindings from the registry.
%
clear_all_bindings :-
    retractall(stored_binding(_, _, _, _, _, _)).

%% clear_all_bindings(+Target)
%
%  Remove all bindings for a specific target.
%
clear_all_bindings(Target) :-
    retractall(stored_binding(Target, _, _, _, _, _)).

% ============================================================================
% EFFECT ANNOTATIONS
% ============================================================================

%% effect(?Pred, ?Effects)
%
%  Query effect annotations for a predicate.
%
effect(Pred, Effects) :-
    stored_effect(Pred, Effects).

%% declare_effect(+Pred, +Effects)
%
%  Declare effects for a predicate (target-independent).
%
declare_effect(Pred, Effects) :-
    validate_pred_indicator(Pred),
    is_list(Effects),
    retractall(stored_effect(Pred, _)),
    assertz(stored_effect(Pred, Effects)).

%% clear_effect(+Pred)
%
%  Remove effect annotations for a predicate.
%
clear_effect(Pred) :-
    retractall(stored_effect(Pred, _)).

% ============================================================================
% DESIGN PATTERNS
% ============================================================================

%% pattern(?PatternName, ?Description)
%
%  Query or list design patterns for non-standard function usage.
%
pattern(PatternName, Description) :-
    stored_pattern(PatternName, Description).

%% declare_pattern(+PatternName, +Description)
%
%  Declare a new design pattern.
%
declare_pattern(PatternName, Description) :-
    atom(PatternName),
    retractall(stored_pattern(PatternName, _)),
    assertz(stored_pattern(PatternName, Description)).

% Initialize standard patterns
:- declare_pattern(stdout_return, "Return value via stdout (shell idiom)").
:- declare_pattern(variable_return, "Return value via variable assignment").
:- declare_pattern(exit_code_bool, "Boolean result via exit code").
:- declare_pattern(pipe_transform, "Transform data through pipe").
:- declare_pattern(accumulator, "Build result via accumulator parameter").
:- declare_pattern(object_pipeline, "PowerShell object pipeline pattern").
:- declare_pattern(cmdlet_output, "PowerShell cmdlet structured output").

% ============================================================================
% BIDIRECTIONAL PREDICATES
% ============================================================================

%% bidirectional(?Pred, ?Modes)
%
%  Query bidirectional predicate declarations.
%
bidirectional(Pred, Modes) :-
    stored_bidirectional(Pred, Modes).

%% declare_bidirectional(+Pred, +Modes)
%
%  Declare that a predicate supports multiple argument modes.
%
declare_bidirectional(Pred, Modes) :-
    validate_pred_indicator(Pred),
    is_list(Modes),
    retractall(stored_bidirectional(Pred, _)),
    assertz(stored_bidirectional(Pred, Modes)).

%% binding_mode(?Target, ?Pred, ?Mode, ?TargetName, ?Options)
%
%  Query mode-specific bindings for bidirectional predicates.
%
binding_mode(Target, Pred, Mode, TargetName, Options) :-
    stored_binding_mode(Target, Pred, Mode, TargetName, Options).

%% declare_binding_mode(+Target, +Pred, +Mode, +TargetName, +Options)
%
%  Declare a binding for a specific mode of a bidirectional predicate.
%
declare_binding_mode(Target, Pred, Mode, TargetName, Options) :-
    atom(Target),
    validate_pred_indicator(Pred),
    is_list(Options),
    retractall(stored_binding_mode(Target, Pred, Mode, _, _)),
    assertz(stored_binding_mode(Target, Pred, Mode, TargetName, Options)).

% ============================================================================
% QUERY API
% ============================================================================

%% bindings_for_target(+Target, -Bindings)
%
%  Get all bindings for a specific target.
%
bindings_for_target(Target, Bindings) :-
    findall(
        binding(Target, P, N, I, O, Opts),
        stored_binding(Target, P, N, I, O, Opts),
        Bindings
    ).

%% bindings_for_predicate(+Pred, -Bindings)
%
%  Get all bindings (across targets) for a specific predicate.
%
bindings_for_predicate(Pred, Bindings) :-
    findall(
        binding(T, Pred, N, I, O, Opts),
        stored_binding(T, Pred, N, I, O, Opts),
        Bindings
    ).

%% resolve_binding(+Preferences, +Pred, -Target, -Binding)
%
%  Resolve a binding given a list of preferred targets.
%  Falls back through the list until a binding is found.
%
resolve_binding([Target|_], Pred, Target, Binding) :-
    stored_binding(Target, Pred, Name, In, Out, Opts),
    !,
    Binding = binding(Target, Pred, Name, In, Out, Opts).
resolve_binding([_|Rest], Pred, Target, Binding) :-
    resolve_binding(Rest, Pred, Target, Binding).

%% binding_imports(+Target, -Imports)
%
%  Get all imports required by bindings for a target.
%
binding_imports(Target, Imports) :-
    findall(
        Module,
        (stored_binding(Target, _, _, _, _, Opts),
         member(import(Module), Opts)),
        ImportsRaw
    ),
    sort(ImportsRaw, Imports).

%% binding_has_effect(+Target, +Pred, +Effect)
%
%  Check if a binding has a specific effect.
%
binding_has_effect(Target, Pred, Effect) :-
    stored_binding(Target, Pred, _, _, _, Opts),
    (   member(effect(Effect), Opts)
    ;   member(Effect, Opts)
    ).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% is_pure_binding(+Target, +Pred)
%
%  True if the binding is declared as pure (no side effects).
%
is_pure_binding(Target, Pred) :-
    stored_binding(Target, Pred, _, _, _, Opts),
    member(pure, Opts).

%% is_total_binding(+Target, +Pred)
%
%  True if the binding is declared as total (always succeeds).
%
is_total_binding(Target, Pred) :-
    stored_binding(Target, Pred, _, _, _, Opts),
    member(total, Opts).

%% is_deterministic_binding(+Target, +Pred)
%
%  True if the binding is declared as deterministic (single result).
%
is_deterministic_binding(Target, Pred) :-
    stored_binding(Target, Pred, _, _, _, Opts),
    member(deterministic, Opts).

% ============================================================================
% VALIDATION HELPERS
% ============================================================================

validate_pred_indicator(Name/Arity) :-
    atom(Name),
    integer(Arity),
    Arity >= 0,
    !.
validate_pred_indicator(Pred) :-
    throw(error(invalid_predicate_indicator(Pred), context(binding_registry, _))).

% ============================================================================
% TESTING
% ============================================================================

test_binding_registry :-
    format('~n=== Testing Binding Registry ===~n~n'),

    % Clean up
    clear_all_bindings,

    % Test 1: Declare and query bindings
    format('[Test 1] Declare and query bindings~n'),
    declare_binding(powershell, length/2, 'Measure-Object -Character',
                    [string], [int], [pure, deterministic]),
    declare_binding(powershell, file_exists/1, 'Test-Path',
                    [path], [], [effect(io), pattern(cmdlet_output)]),
    (   binding(powershell, length/2, _, _, _, _)
    ->  format('  [PASS] length/2 binding found~n')
    ;   format('  [FAIL] length/2 binding not found~n')
    ),

    % Test 2: Query by target
    format('[Test 2] Query bindings by target~n'),
    bindings_for_target(powershell, PSBindings),
    length(PSBindings, NumBindings),
    format('  Found ~w PowerShell bindings~n', [NumBindings]),
    (   NumBindings >= 2
    ->  format('  [PASS] Expected bindings found~n')
    ;   format('  [FAIL] Expected at least 2 bindings~n')
    ),

    % Test 3: Check pure binding
    format('[Test 3] Check pure binding~n'),
    (   is_pure_binding(powershell, length/2)
    ->  format('  [PASS] length/2 is pure~n')
    ;   format('  [FAIL] length/2 should be pure~n')
    ),
    (   \+ is_pure_binding(powershell, file_exists/1)
    ->  format('  [PASS] file_exists/1 is not pure~n')
    ;   format('  [FAIL] file_exists/1 should not be pure~n')
    ),

    % Test 4: Effects
    format('[Test 4] Effect annotations~n'),
    declare_effect(my_io_pred/2, [effect(io), nondeterministic]),
    (   effect(my_io_pred/2, Effects), member(effect(io), Effects)
    ->  format('  [PASS] Effect annotation stored~n')
    ;   format('  [FAIL] Effect annotation not found~n')
    ),

    % Test 5: Patterns
    format('[Test 5] Design patterns~n'),
    (   pattern(stdout_return, _)
    ->  format('  [PASS] stdout_return pattern exists~n')
    ;   format('  [FAIL] stdout_return pattern not found~n')
    ),
    (   pattern(cmdlet_output, _)
    ->  format('  [PASS] cmdlet_output pattern exists~n')
    ;   format('  [FAIL] cmdlet_output pattern not found~n')
    ),

    % Test 6: Resolve binding with fallback
    format('[Test 6] Resolve binding with fallback~n'),
    declare_binding(python, length/2, 'len', [list], [int], [pure]),
    (   resolve_binding([rust, powershell, python], length/2, Target, _)
    ->  format('  Resolved to: ~w~n', [Target]),
        (   Target == powershell
        ->  format('  [PASS] Correctly resolved to powershell~n')
        ;   format('  [FAIL] Should resolve to powershell first~n')
        )
    ;   format('  [FAIL] Resolution failed~n')
    ),

    % Cleanup
    clear_all_bindings,
    format('~n=== All Binding Registry Tests Complete ===~n').
