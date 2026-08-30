:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% javascript_wam_bindings.pl - Builtin catalogue for the JS WAM runtime.
%
% The interpreter in templates/targets/javascript_wam/runtime.js.mustache
% dispatches these names from BuiltinCall (and as a Call/Execute fallback
% when no user label exists). Status:
%   implemented - full for the conformance + probe suite, including
%                 ISO bagof/3 / setof/3 witness grouping and ^/2

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
javascript_wam_builtin(bagof, 3, implemented).
javascript_wam_builtin(setof, 3, implemented).
javascript_wam_builtin('^', 2, implemented).
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
javascript_wam_builtin(deterministic, 0, implemented).
javascript_wam_builtin(writeln, 1, implemented).
javascript_wam_builtin(compare, 3, implemented).
javascript_wam_builtin(sort, 2, implemented).
javascript_wam_builtin(sort, 4, implemented).
javascript_wam_builtin(msort, 2, implemented).
javascript_wam_builtin(keysort, 2, implemented).
javascript_wam_builtin(predsort, 3, implemented).
javascript_wam_builtin(append, 3, implemented).
javascript_wam_builtin(reverse, 2, implemented).
javascript_wam_builtin(nth0, 3, implemented).
javascript_wam_builtin(nth1, 3, implemented).
javascript_wam_builtin(last, 2, implemented).
javascript_wam_builtin(sum_list, 2, implemented).
javascript_wam_builtin(sumlist, 2, implemented).
javascript_wam_builtin(max_list, 2, implemented).
javascript_wam_builtin(min_list, 2, implemented).
javascript_wam_builtin(list_to_set, 2, implemented).
javascript_wam_builtin(select, 3, implemented).
javascript_wam_builtin(include, 3, implemented).
javascript_wam_builtin(exclude, 3, implemented).
javascript_wam_builtin(atom_concat, 3, implemented).
javascript_wam_builtin(atom_length, 2, implemented).
javascript_wam_builtin(atom_chars, 2, implemented).
javascript_wam_builtin(atom_codes, 2, implemented).
javascript_wam_builtin(char_code, 2, implemented).
javascript_wam_builtin(sub_atom, 5, implemented).
javascript_wam_builtin(atom_string, 2, implemented).
javascript_wam_builtin(number_codes, 2, implemented).
javascript_wam_builtin(number_string, 2, implemented).
javascript_wam_builtin(split_string, 4, implemented).
javascript_wam_builtin(string_concat, 3, implemented).
javascript_wam_builtin(string_chars, 2, implemented).
javascript_wam_builtin(upcase_atom, 2, implemented).
javascript_wam_builtin(downcase_atom, 2, implemented).
javascript_wam_builtin(format, 2, implemented).
javascript_wam_builtin(format, 3, implemented).
javascript_wam_builtin(tab, 1, implemented).
javascript_wam_builtin(empty_assoc, 1, implemented).
javascript_wam_builtin(list_to_assoc, 2, implemented).
javascript_wam_builtin(get_assoc, 3, implemented).
javascript_wam_builtin(put_assoc, 4, implemented).
javascript_wam_builtin(assoc_to_list, 2, implemented).
javascript_wam_builtin(assoc_to_keys, 2, implemented).

javascript_wam_builtins(List) :-
    findall(Name/Arity, javascript_wam_builtin(Name, Arity, _), List).
