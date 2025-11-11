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
:- use_module(dynamic_source_compiler). % Support for dynamic data sources
:- use_module('advanced/advanced_recursive_compiler'). % Added for mutual recursion classification
:- use_module('advanced/call_graph'). % Added for mutual recursion classification

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

    % Check if this is a dynamic source FIRST (before classification)
    (   dynamic_source_compiler:is_dynamic_source(Predicate) ->
        % Compile as dynamic source via appropriate plugin
        dynamic_source_compiler:compile_dynamic_source(Predicate, Options, BashCode),
        !  % Cut to prevent backtracking into static predicate path
    ;   % Original logic: classify and compile as static predicate
        classify_predicate(Predicate, Classification),
        (   Classification = non_recursive ->
            stream_compiler:compile_predicate(Predicate, Options, BashCode),
            !  % Cut after successful compilation
        ;   recursive_compiler:compile_recursive(Predicate, Options, BashCode),
            !  % Cut after successful compilation
        )
    ),

    % Write generated code to file
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

    % Check for mutual recursion FIRST (before self-recursion check)
    % BUT: exclude dynamic sources from the group - they're not part of mutual recursion
    (   call_graph:predicates_in_group(Pred/Arity, Group),
        % Filter out dynamic sources from group
        exclude(dynamic_source_compiler:is_dynamic_source, Group, StaticGroup),
        length(StaticGroup, GroupSize),
        GroupSize > 1 ->
        Classification = mutual_recursion
    ;   % Check if self-recursive
        contains_recursive_call(Pred, Bodies) ->
        analyze_recursion_pattern(Pred, Arity, Bodies, Classification)
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

%% Analyze recursion pattern
analyze_recursion_pattern(Pred, Arity, Bodies, Pattern) :-
    % Separate base cases from recursive cases
    partition(is_recursive_clause(Pred), Bodies, RecClauses, BaseClauses),
    
    % Check for mutual recursion first
    (   call_graph:predicates_in_group(Pred/Arity, Group),
        length(Group, GroupSize),
        GroupSize > 1 ->
        Pattern = mutual_recursion
    ;   % Check for transitive closure pattern
        is_transitive_closure(Pred, Arity, BaseClauses, RecClauses, BasePred) ->
        Pattern = transitive_closure(BasePred)
    ;   is_tail_recursive(Pred, RecClauses) ->
        Pattern = tail_recursion
    ;   is_linear_recursive(Pred, RecClauses) ->
        Pattern = linear_recursion
    ;   Pattern = unknown_recursion
    ).

is_recursive_clause(Pred, Body) :-
    contains_goal(Body, Goal),
    functor(Goal, Pred, _).

%% Check for transitive closure pattern
% Two patterns supported:
% 1. Forward: pred(X,Z) :- base(X,Y), pred(Y,Z).  [e.g., ancestor]
% 2. Reverse: pred(X,Z) :- base(Y,X), pred(Y,Z).  [e.g., descendant]
is_transitive_closure(Pred, 2, BaseClauses, RecClauses, BasePred) :-
    % Check base case is a single predicate call
    member(BaseBody, BaseClauses),
    BaseBody \= true,
    functor(BaseBody, BasePred, 2),
    BasePred \= Pred,

    % Check recursive case matches pattern
    member(RecBody, RecClauses),
    RecBody = (BaseCall, RecCall),
    functor(BaseCall, BasePred, 2),
    functor(RecCall, Pred, 2),

    % Try both forward and reverse patterns
    (   % Pattern 1: Forward transitive closure
        % base(X,Y), recursive(Y,Z) - Y flows from base to recursive
        BaseCall =.. [BasePred, _X, Y],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ;   % Pattern 2: Reverse transitive closure
        % base(Y,X), recursive(Y,Z) - Y flows from base to recursive (reversed args)
        BaseCall =.. [BasePred, Y, _X],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ).

is_transitive_closure(_, _, _, _, _) :- fail.

%% Check for tail recursion
is_tail_recursive(Pred, RecClauses) :-
    member(Body, RecClauses),
    last_goal(Body, Goal),
    functor(Goal, Pred, _).

last_goal(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_).
last_goal((_, B), Goal) :-
    last_goal(B, Goal).

%% Check for linear recursion
is_linear_recursive(Pred, RecClauses) :-
    member(Body, RecClauses),
    findall(G, contains_goal(Body, G), Goals),
    findall(G, (member(G, Goals), functor(G, Pred, _)), RecGoals),
    length(RecGoals, 1).  % Exactly one recursive call