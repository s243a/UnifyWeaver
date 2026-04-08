:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% semantic_compiler.pl - Generic Semantic Search Compilation Interface
%
% This module provides a unified way to define and compile semantic search
% predicates across multiple targets with support for fallback providers.

:- module(semantic_compiler, [
    declare_semantic_provider/2,     % +Predicate, +Options
    get_semantic_provider/3,         % +Target, +Predicate, -ProviderInfo
    is_semantic_predicate/1,         % +Goal
    compile_semantic_call/4,         % +Target, +Goal, +VarMap, -Code

    % Fuzzy logic compilation
    is_fuzzy_predicate/1,            % +Goal
    compile_fuzzy_call/3             % +Target, +Goal, -Code
]).

:- use_module(library(lists)).
:- use_module(library(option)).

% ============================================================================
% DYNAMIC STORAGE
% ============================================================================

:- dynamic stored_semantic_provider/2. % stored_semantic_provider(Predicate, Options)
:- dynamic user_providers_loaded/0.

% ============================================================================
% CORE API
% ============================================================================

%% load_user_providers
%
%  Discover and register semantic providers declared in the user module
%  via semantic_provider/2. Runs at most once per session.
%
load_user_providers :-
    user_providers_loaded, !.
load_user_providers :-
    (   current_predicate(user:semantic_provider/2)
    ->  forall(user:semantic_provider(Pred/Arity, Options),
               declare_semantic_provider(Pred/Arity, Options))
    ;   true
    ),
    assertz(user_providers_loaded).

%% declare_semantic_provider(+Predicate, +Options)
%
%  Register a semantic search provider configuration.
%
%  Options:
%    - targets(TargetList) where TargetList is a list of:
%        target(Name, [provider(P), model(M), device(D), ...])
%    - fallback(ProviderName)
%
declare_semantic_provider(Pred/Arity, Options) :-
    atom(Pred), integer(Arity),
    retractall(stored_semantic_provider(Pred/Arity, _)),
    assertz(stored_semantic_provider(Pred/Arity, Options)).

%% get_semantic_provider(+Target, +Predicate, -ProviderInfo)
%
%  Find the best provider for a target.
%
get_semantic_provider(Target, Pred/Arity, ProviderInfo) :-
    load_user_providers,
    stored_semantic_provider(Pred/Arity, Options),
    member(targets(Targets), Options),
    member(target(Target, ProviderInfo), Targets),
    !.
get_semantic_provider(_Target, Pred/Arity, ProviderInfo) :-
    stored_semantic_provider(Pred/Arity, Options),
    member(fallback(ProviderInfo), Options),
    !.

%% is_semantic_predicate(+Goal)
%
%  True if the goal is registered as a semantic predicate.
%
is_semantic_predicate(Goal) :-
    load_user_providers,
    functor(Goal, Pred, Arity),
    stored_semantic_provider(Pred/Arity, _).

%% compile_semantic_call(+Target, +Goal, +VarMap, -Code)
%
%  Dispatch compilation to target-specific semantic handler.
%
compile_semantic_call(Target, Goal, VarMap, Code) :-
    functor(Goal, Pred, Arity),
    get_semantic_provider(Target, Pred/Arity, ProviderInfo),
    semantic_dispatch(Target, Goal, ProviderInfo, VarMap, Code).

% ============================================================================
% TARGET DISPATCH (Multifile hooks)
% ============================================================================

:- multifile semantic_dispatch/5.
% semantic_dispatch(+Target, +Goal, +ProviderInfo, +VarMap, -Code)

% ============================================================================
% FUZZY LOGIC COMPILATION
% ============================================================================

%% is_fuzzy_predicate(+Goal)
%
%  True if the goal is a fuzzy logic operation that can be compiled
%  via fuzzy_dispatch/4.
%
is_fuzzy_predicate(Goal) :-
    functor(Goal, Name, _),
    fuzzy_op(Name).

fuzzy_op(f_and).
fuzzy_op(f_or).
fuzzy_op(f_dist_or).
fuzzy_op(f_union).
fuzzy_op(f_not).
fuzzy_op(eval_fuzzy_expr).
fuzzy_op(blend_scores).
fuzzy_op(multiply_scores).
fuzzy_op(top_k).
fuzzy_op(apply_filter).
fuzzy_op(apply_boost).

%% compile_fuzzy_call(+Target, +Goal, -Code)
%
%  Dispatch compilation of a fuzzy logic goal to the target-specific handler.
%
compile_fuzzy_call(Target, Goal, Code) :-
    fuzzy_dispatch(Target, Goal, Code).

:- multifile fuzzy_dispatch/3.
% fuzzy_dispatch(+Target, +Goal, -Code)
