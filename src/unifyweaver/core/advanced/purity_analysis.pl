:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% purity_analysis.pl — Static purity analysis for Prolog goals
%
% Historical role: determined whether goals are free of side effects,
% to enable automatic linear-to-tail recursion transformation.
%
% Phase P2 of the purity certificate plan: this module is now a thin
% wrapper over purity_certificate. The pure_builtin/1 catalogue moved
% to the certificate module (single source of truth). is_pure_goal/1
% and is_pure_body/1 delegate to purity_certificate:is_whitelist_pure_goal/1
% so tail-recursion transformations see exactly the same strict
% verdicts they always did.
%
% is_associative_op/1 stays local — it's orthogonal to purity (it's
% about arithmetic operator associativity for accumulator transform).

:- module(purity_analysis, [
    is_pure_goal/1,           % +Goal
    is_pure_body/1,           % +Body (conjunction)
    pure_builtin/1,           % +Functor/Arity
    is_associative_op/1       % +Op
]).

% Explicit import list — avoids the pure_builtin/1 name clash. We
% re-export pure_builtin/1 ourselves as a thin delegator below, so we
% don't want the symbol imported directly.
:- use_module('../purity_certificate',
              [is_whitelist_pure_goal/1]).

%% pure_builtin(?Functor/Arity)
%  Delegates to purity_certificate:pure_builtin/1. Kept as a re-export
%  for back-compat with any external caller that module-qualifies
%  purity_analysis:pure_builtin/1.
pure_builtin(FA) :- purity_certificate:pure_builtin(FA).

%% is_pure_goal(+Goal)
%  Succeeds if Goal is known to be free of side effects by the STRICT
%  whitelist. Thin wrapper — identical semantics to the pre-P2
%  standalone check.
is_pure_goal(Goal) :- purity_certificate:is_whitelist_pure_goal(Goal).

%% is_pure_body(+Body)
%  Succeeds if all goals in a conjunction are whitelist-pure.
is_pure_body(true).
is_pure_body((A, B)) :- !,
    is_pure_body(A),
    is_pure_body(B).
is_pure_body(Goal) :-
    is_pure_goal(Goal).

%% is_associative_op(+Op)
%  Arithmetic operators that are associative and admit accumulator
%  transformation. a op (b op c) = (a op b) op c.
%  Orthogonal to purity — kept local to this module.
is_associative_op(+).
is_associative_op(*).
