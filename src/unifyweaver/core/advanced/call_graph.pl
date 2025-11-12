:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% call_graph.pl - Build and analyze predicate call graphs
% Used for detecting mutual recursion and dependency ordering

:- module(call_graph, [
    build_call_graph/2,      % +Predicates, -Graph
    get_dependencies/2,      % +Pred/Arity, -Dependencies
    is_self_recursive/1,     % +Pred/Arity
    predicates_in_group/2,   % +RootPred, -AllPredicates
    test_call_graph/0        % Test predicate
]).

:- use_module(library(lists)).

%% build_call_graph(+Predicates, -Graph)
%  Build a directed graph of predicate calls
%  Graph is a list of edges: [pred1/arity1 -> pred2/arity2, ...]
build_call_graph(Predicates, Graph) :-
    findall(Edge,
        (   member(Pred/Arity, Predicates),
            find_predicate_calls(Pred/Arity, CalledPred),
            Edge = (Pred/Arity -> CalledPred)
        ),
        Graph).

%% find_predicate_calls(+Pred/Arity, -CalledPred)
%  Find all predicates called by this predicate
find_predicate_calls(Pred/Arity, CalledPred/CalledArity) :-
    functor(Head, Pred, Arity),
    % Use user:clause to access predicates from any module (including test predicates)
    user:clause(Head, Body),
    extract_calls_from_body(Body, CalledPred/CalledArity),
    % Only include user-defined predicates (skip built-ins)
    \+ is_builtin(CalledPred/CalledArity).

%% extract_calls_from_body(+Body, -Pred/Arity)
%  Extract all predicate calls from a clause body
extract_calls_from_body(true, _) :- !, fail.
extract_calls_from_body((A, B), Pred/Arity) :-
    !,
    (   extract_calls_from_body(A, Pred/Arity)
    ;   extract_calls_from_body(B, Pred/Arity)
    ).
extract_calls_from_body((A; B), Pred/Arity) :-
    !,
    (   extract_calls_from_body(A, Pred/Arity)
    ;   extract_calls_from_body(B, Pred/Arity)
    ).
extract_calls_from_body((A -> B), Pred/Arity) :-
    !,
    (   extract_calls_from_body(A, Pred/Arity)
    ;   extract_calls_from_body(B, Pred/Arity)
    ).
extract_calls_from_body(\+ A, Pred/Arity) :-
    !,
    extract_calls_from_body(A, Pred/Arity).
extract_calls_from_body(Call, Pred/Arity) :-
    compound(Call),
    functor(Call, Pred, Arity),
    Pred \= ',',
    Pred \= ';',
    Pred \= '->'.

%% is_builtin(+Pred/Arity)
%  Check if predicate is a Prolog built-in
is_builtin(Pred/_) :-
    member(Pred, [
        is, =, \=, ==, \==, @<, @>, @=<, @>=,
        <, >, =<, >=, =:=, =\=,
        !, true, fail, call, catch, throw,
        atom, number, compound, var, nonvar,
        functor, arg, =.., copy_term,
        assert, asserta, assertz, retract, retractall, abolish,
        findall, bagof, setof,
        member, append, length, reverse, sort,
        write, writeln, format, nl, put, get,
        atom_string, atom_chars, string_chars, atom_concat,
        atomic_list_concat, split_string
    ]).

%% get_dependencies(+Pred/Arity, -Dependencies)
%  Get all predicates that this predicate depends on
get_dependencies(Pred/Arity, Dependencies) :-
    findall(Dep,
        find_predicate_calls(Pred/Arity, Dep),
        DepsWithDups),
    sort(DepsWithDups, Dependencies).

%% is_self_recursive(+Pred/Arity)
%  Check if a predicate is self-recursive (calls itself)
is_self_recursive(Pred/Arity) :-
    get_dependencies(Pred/Arity, Deps),
    member(Pred/Arity, Deps).

%% predicates_in_group(+RootPred, -SCC)
%  Finds the Strongly Connected Component (SCC) containing the RootPred.
%  This correctly identifies mutually recursive predicates.
predicates_in_group(Root, SCC) :-
    % 1. Find all predicates reachable from the Root.
    find_reachable(Root, Reachable),
    % 2. From that reachable set, include only those that can in turn reach the Root.
    include(can_reach(Root), Reachable, SCC).

%% can_reach(+Target, +Start)
%  Succeeds if Target is reachable from Start in the call graph.
can_reach(Target, Start) :-
    find_reachable(Start, ReachableFromStart),
    member(Target, ReachableFromStart).

%% find_reachable(+Root, -Reachable)
%  Helper to find all predicates reachable from a root predicate (transitive closure).
find_reachable(Root, Reachable) :-
    find_reachable_acc([Root], [Root], Reachable).

find_reachable_acc([], Acc, Acc).
find_reachable_acc([Pred|Queue], Visited, Reachable) :-
    get_dependencies(Pred, Deps),
    subtract(Deps, Visited, NewDeps),
    append(Queue, NewDeps, NewQueue),
    append(Visited, NewDeps, NewVisited),
    find_reachable_acc(NewQueue, NewVisited, Reachable).

%% ============================================
%% TESTS
%% ============================================

test_call_graph :-
    writeln('=== CALL GRAPH TESTS ==='),

    % Clear any existing test predicates
    catch(abolish(test_fact/1), _, true),
    catch(abolish(test_linear/2), _, true),
    catch(abolish(test_even/1), _, true),
    catch(abolish(test_odd/1), _, true),

    % Define test predicates in user module so user:clause can find them
    % 1. Non-recursive fact
    assertz(user:test_fact(a)),
    assertz(user:test_fact(b)),

    % 2. Self-recursive (linear)
    assertz(user:(test_linear([], 0))),
    assertz(user:(test_linear([_|T], N) :- test_linear(T, N1), N is N1 + 1)),

    % 3. Mutually recursive (even/odd)
    assertz(user:test_even(0)),
    assertz(user:(test_even(N) :- N > 0, N1 is N - 1, test_odd(N1))),
    assertz(user:test_odd(1)),
    assertz(user:(test_odd(N) :- N > 1, N1 is N - 1, test_even(N1))),

    % Test 1: Non-recursive has no dependencies
    writeln('Test 1: Non-recursive predicate'),
    get_dependencies(test_fact/1, Deps1),
    format('  test_fact/1 dependencies: ~w~n', [Deps1]),
    (   Deps1 = [] -> writeln('  ✓ PASS') ; writeln('  ✗ FAIL')),

    % Test 2: Self-recursive detected
    writeln('Test 2: Self-recursive predicate'),
    (   is_self_recursive(test_linear/2) ->
        writeln('  ✓ PASS - test_linear/2 is self-recursive')
    ;   writeln('  ✗ FAIL - should be self-recursive')
    ),

    % Test 3: Mutual recursion dependencies
    writeln('Test 3: Mutual recursion'),
    get_dependencies(test_even/1, EvenDeps),
    get_dependencies(test_odd/1, OddDeps),
    format('  test_even/1 calls: ~w~n', [EvenDeps]),
    format('  test_odd/1 calls: ~w~n', [OddDeps]),
    (   member(test_odd/1, EvenDeps), member(test_even/1, OddDeps) ->
        writeln('  ✓ PASS - mutual recursion detected')
    ;   writeln('  ✗ FAIL - should detect mutual recursion')
    ),

    % Test 4: Build call graph
    writeln('Test 4: Build call graph'),
    build_call_graph([test_even/1, test_odd/1], Graph),
    format('  Graph edges: ~w~n', [Graph]),
    (   length(Graph, Len), Len >= 2 ->
        writeln('  ✓ PASS - graph built')
    ;   writeln('  ✗ FAIL - graph should have edges')
    ),

    % Test 5: Predicate group
    writeln('Test 5: Find predicate group'),
    predicates_in_group(test_even/1, Group),
    format('  Group from test_even/1: ~w~n', [Group]),
    (   member(test_odd/1, Group) ->
        writeln('  ✓ PASS - found mutually recursive partner')
    ;   writeln('  ✗ FAIL - should find test_odd/1 in group')
    ),

    writeln('=== CALL GRAPH TESTS COMPLETE ===').
