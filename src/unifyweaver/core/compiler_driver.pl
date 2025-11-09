% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- module(compiler_driver, [
    compile/2,
    compile/3
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(dependency_analyzer).
:- use_module(recursive_compiler).
:- use_module(stream_compiler).

:- dynamic compiled/1.

%% compile(+Predicate, -GeneratedScripts)
compile(Predicate, GeneratedScripts) :-
    compile(Predicate, [], GeneratedScripts).

%% compile(+Predicate, +Options, -GeneratedScripts)
%  Recursively compiles a predicate and its dependencies.
compile(Predicate, Options, GeneratedScripts) :-
    retractall(compiled(_)),
    compile_entry(Predicate, Options, GeneratedScriptsUnsorted),
    list_to_set(GeneratedScriptsUnsorted, GeneratedScripts).

compile_entry(Predicate, Options, GeneratedScripts) :-
    (   compiled(Predicate) ->
        GeneratedScripts = []
    ;   assertz(compiled(Predicate)),
        find_dependencies(Predicate, AllDependencies),
        % Filter out built-in predicates so we don't try to compile them
        exclude(is_builtin, AllDependencies, UserDependencies),
        compile_dependencies(UserDependencies, Options, DepScripts),
        compile_current(Predicate, Options, CurrentScript),
        append(DepScripts, [CurrentScript], GeneratedScripts)
    ).

is_builtin(Functor/Arity) :-
    functor(Head, Functor, Arity),
    predicate_property(Head, built_in).

compile_dependencies([], _, []).
compile_dependencies([Dep|Rest], Options, GeneratedScripts) :-
    compile_entry(Dep, Options, DepScripts),
    compile_dependencies(Rest, Options, RestScripts),
    append(DepScripts, RestScripts, GeneratedScripts).

compile_current(Predicate, Options, GeneratedScript) :-
    Predicate = Functor/_Arity,
    classify_predicate(Predicate, Classification),
    (   Classification = non_recursive ->
        stream_compiler:compile_predicate(Predicate, Options, BashCode)
    ;   recursive_compiler:compile_recursive(Predicate, Options, BashCode)
    ),
    option(output_dir(OutputDir), Options, 'education/output/advanced'),
    atomic_list_concat([OutputDir, '/', Functor, '.sh'], GeneratedScript),
    open(GeneratedScript, write, Stream),
    write(Stream, BashCode),
    close(Stream).

% --- Predicate Classification ---
% (Copied from recursive_compiler.pl to avoid modifying it)

classify_predicate(Pred/Arity, Classification) :-
    functor(Head, Pred, Arity),
    findall(Body, clause(Head, Body), Bodies),
    (   contains_recursive_call(Pred, Bodies) ->
        Classification = recursive % Simplified for the driver
    ;   Classification = non_recursive
    ).

contains_recursive_call(Pred, Bodies) :-
    member(Body, Bodies),
    contains_goal(Body, Goal),
    functor(Goal, Pred, _).

contains_goal(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_).
contains_goal((A, _), Goal) :-
    contains_goal(A, Goal).
contains_goal((_, B), Goal) :-
    contains_goal(B, Goal).
contains_goal((A; _), Goal) :-
    contains_goal(A, Goal).
contains_goal((_;B), Goal) :-
    contains_goal(B, Goal).
