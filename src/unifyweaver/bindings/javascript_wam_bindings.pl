:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% javascript_wam_bindings.pl - Builtin catalogue for the JS WAM runtime.
%
% The interpreter in templates/targets/javascript_wam/runtime.js.mustache
% dispatches these names from BuiltinCall (and as a Call/Execute fallback
% when no user label exists). Status:
%   implemented - full for the conformance + probe suite
%   partial     - works for ground/simple goals; ISO free-var grouping
%                 (bagof/setof witnesses) is not implemented

:- module(javascript_wam_bindings, [
    javascript_wam_builtin/3,          % Name, Arity, Status
    javascript_wam_builtins/1,         % -List of Name/Arity
    init_javascript_wam_bindings/0
]).

:- use_module('../core/binding_registry', []).

init_javascript_wam_bindings.

%% javascript_wam_builtin(?Name, ?Arity, ?Status)
javascript_wam_builtin(true, 0, implemented).
javascript_wam_builtin(fail, 0, implemented).
javascript_wam_builtin(false, 0, implemented).
javascript_wam_builtin('!', 0, implemented).
javascript_wam_builtin((=), 2, implemented).
javascript_wam_builtin((==), 2, implemented).
javascript_wam_builtin((\==), 2, implemented).
javascript_wam_builtin(is, 2, implemented).
javascript_wam_builtin((=:=), 2, implemented).
javascript_wam_builtin((=\=), 2, implemented).
javascript_wam_builtin((>), 2, implemented).
javascript_wam_builtin((<), 2, implemented).
javascript_wam_builtin((>=), 2, implemented).
javascript_wam_builtin((=<), 2, implemented).
javascript_wam_builtin(member, 2, implemented).
javascript_wam_builtin(length, 2, implemented).
javascript_wam_builtin(between, 3, implemented).
javascript_wam_builtin(write, 1, implemented).
javascript_wam_builtin(nl, 0, implemented).
javascript_wam_builtin(functor, 3, implemented).
javascript_wam_builtin(arg, 3, implemented).
javascript_wam_builtin((=..), 2, implemented).
javascript_wam_builtin(copy_term, 2, implemented).
javascript_wam_builtin((\+), 1, implemented).
javascript_wam_builtin(call, 1, implemented).
javascript_wam_builtin(findall, 3, implemented).
javascript_wam_builtin(bagof, 3, partial).
javascript_wam_builtin(setof, 3, partial).
javascript_wam_builtin(aggregate_all, 3, implemented).
javascript_wam_builtin(atom, 1, implemented).
javascript_wam_builtin(integer, 1, implemented).
javascript_wam_builtin(float, 1, implemented).
javascript_wam_builtin(number, 1, implemented).
javascript_wam_builtin(compound, 1, implemented).
javascript_wam_builtin(var, 1, implemented).
javascript_wam_builtin(nonvar, 1, implemented).
javascript_wam_builtin(is_list, 1, implemented).
javascript_wam_builtin(ground, 1, implemented).

javascript_wam_builtins(List) :-
    findall(Name/Arity, javascript_wam_builtin(Name, Arity, _), List).
