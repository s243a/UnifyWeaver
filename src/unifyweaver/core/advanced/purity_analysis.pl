:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% purity_analysis.pl - Static purity analysis for Prolog goals
% Determines whether goals are free of side effects, enabling
% automatic linear-to-tail recursion transformation.

:- module(purity_analysis, [
    is_pure_goal/1,           % +Goal
    is_pure_body/1,           % +Body (conjunction)
    pure_builtin/1,           % +Functor/Arity
    is_associative_op/1       % +Op
]).

%% pure_builtin(?Functor/Arity)
%  Whitelisted pure built-in predicates.
%  These are known to have no side effects.

% Arithmetic
pure_builtin(is/2).
pure_builtin(succ/2).
pure_builtin(plus/3).

% Comparison
pure_builtin((>)/2).
pure_builtin((<)/2).
pure_builtin((>=)/2).
pure_builtin((=<)/2).
pure_builtin((=:=)/2).
pure_builtin((=\=)/2).

% Unification
pure_builtin((=)/2).
pure_builtin((\=)/2).

% Type checks
pure_builtin(number/1).
pure_builtin(integer/1).
pure_builtin(float/1).
pure_builtin(atom/1).
pure_builtin(compound/1).
pure_builtin(is_list/1).
pure_builtin(var/1).
pure_builtin(nonvar/1).
pure_builtin(ground/1).

% Term manipulation
pure_builtin(functor/3).
pure_builtin(arg/3).
pure_builtin((=..)/2).
pure_builtin(copy_term/2).

% List operations (pure, though member/2 is nondeterministic)
pure_builtin(length/2).
pure_builtin(append/3).
pure_builtin(msort/2).
pure_builtin(sort/2).
pure_builtin(nth0/3).
pure_builtin(nth1/3).
pure_builtin(last/2).
pure_builtin(member/2).
pure_builtin(between/3).
pure_builtin(reverse/2).

% Control (pure forms)
pure_builtin(true/0).

%% is_pure_goal(+Goal)
%  Succeeds if Goal is known to be free of side effects.
is_pure_goal(true).
is_pure_goal(Goal) :-
    callable(Goal),
    functor(Goal, F, A),
    pure_builtin(F/A).

%% is_pure_body(+Body)
%  Succeeds if all goals in a conjunction are pure.
is_pure_body(true).
is_pure_body((A, B)) :-
    !,
    is_pure_body(A),
    is_pure_body(B).
is_pure_body(Goal) :-
    is_pure_goal(Goal).

%% is_associative_op(+Op)
%  Arithmetic operators that are associative and admit accumulator transformation.
%  a op (b op c) = (a op b) op c
is_associative_op(+).
is_associative_op(*).
