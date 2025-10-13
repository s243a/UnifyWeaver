:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% fold_helper_generator.pl - Automatic generation of fold helper patterns
% Generates _graph and fold_ predicates for tree recursion with fold pattern

:- module(fold_helper_generator, [
    generate_fold_helpers/2,           % +Pred/Arity, -Clauses
    generate_graph_builder/3,          % +Pred/Arity, +OrigClauses, -GraphClauses
    generate_fold_computer/3,          % +Pred/Arity, +OrigClauses, -FoldClauses
    generate_wrapper/2,                % +Pred/Arity, -WrapperClause
    install_fold_helpers/1,            % +Pred/Arity - Install generated helpers
    test_fold_helper_generator/0       % Test predicate
]).

:- use_module(library(lists)).
:- use_module(pattern_matchers).

%% ============================================
%% FOLD HELPER GENERATION
%% ============================================
%
% This module automatically generates fold helper predicates for tree recursion.
%
% Given a predicate like:
%   fib(0, 0).
%   fib(1, 1).
%   fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.
%
% It generates:
%   fib_graph(0, leaf(0)).
%   fib_graph(1, leaf(1)).
%   fib_graph(N, node(N, [L, R])) :- N > 1, N1 is N-1, N2 is N-2, fib_graph(N1,L), fib_graph(N2,R).
%
%   fold_fib(leaf(V), V).
%   fold_fib(node(_, [L, R]), V) :- fold_fib(L, VL), fold_fib(R, VR), V is VL+VR.
%
%   fib_fold(N, F) :- fib_graph(N, Graph), fold_fib(Graph, F).

%% generate_fold_helpers(+Pred/Arity, -Clauses)
%  Generate all fold helper clauses for a predicate
%
%  Returns: List of clauses for _graph, fold_, and _fold predicates
%
generate_fold_helpers(Pred/Arity, AllClauses) :-
    % Get original clauses
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), OrigClauses),

    % Generate graph builder clauses
    generate_graph_builder(Pred/Arity, OrigClauses, GraphClauses),

    % Generate fold computer clauses
    generate_fold_computer(Pred/Arity, OrigClauses, FoldClauses),

    % Generate wrapper clause
    generate_wrapper(Pred/Arity, WrapperClause),

    % Combine all clauses
    append([GraphClauses, FoldClauses, [WrapperClause]], AllClauses).

%% generate_graph_builder(+Pred/Arity, +OrigClauses, -GraphClauses)
%  Generate _graph/2 predicate that builds dependency tree structure
%
%  Strategy:
%  1. Base cases: pred(X, Value) => pred_graph(X, leaf(Value))
%  2. Recursive cases: pred(...) :- ..., pred(A1, R1), pred(A2, R2), ...
%                   => pred_graph(...) :- ..., pred_graph(A1, G1), pred_graph(A2, G2), ...
%                      with result node(Input, [G1, G2, ...])
%
generate_graph_builder(Pred/Arity, OrigClauses, GraphClauses) :-
    atom_concat(Pred, '_graph', GraphPred),

    % Process each clause
    findall(GraphClause, (
        member(clause(Head, Body), OrigClauses),
        transform_to_graph_clause(Pred, GraphPred, Arity, Head, Body, GraphClause)
    ), GraphClauses).

%% transform_to_graph_clause(+Pred, +GraphPred, +Arity, +Head, +Body, -GraphClause)
%  Transform original clause to graph-building clause
%
transform_to_graph_clause(Pred, GraphPred, Arity, Head, Body, GraphClause) :-
    Head =.. [Pred|Args],

    % Check if this is a base case (no recursive calls)
    ( \+ contains_call_to(Body, Pred) ->
        % Base case: pred(Input, Output) => pred_graph(Input, leaf(Output))
        Args = [Input, Output],
        GraphHead =.. [GraphPred, Input, leaf(Output)],
        GraphClause = clause(GraphHead, Body)
    ;
        % Recursive case: transform recursive calls to graph calls
        transform_recursive_to_graph(Pred, GraphPred, Arity, Args, Body, GraphHead, GraphBody),
        GraphClause = clause(GraphHead, GraphBody)
    ).

%% transform_recursive_to_graph(+Pred, +GraphPred, +Arity, +Args, +Body, -GraphHead, -GraphBody)
%  Transform recursive clause body to graph-building body
%
transform_recursive_to_graph(Pred, GraphPred, Arity, Args, Body, GraphHead, GraphBody) :-
    % Extract input argument (first arg)
    Args = [Input, _Output],

    % Find all recursive calls in body
    findall(RecCall, (
        extract_goal(Body, RecCall),
        functor(RecCall, Pred, Arity)
    ), RecCalls),

    % Create graph variables for each recursive call
    length(RecCalls, NumCalls),
    length(GraphVars, NumCalls),

    % Replace recursive calls with graph calls in body
    replace_recursive_calls_with_graph(Body, Pred, GraphPred, RecCalls, GraphVars, GraphBody),

    % Create graph head: pred_graph(Input, node(Input, [G1, G2, ...]))
    GraphHead =.. [GraphPred, Input, node(Input, GraphVars)].

%% replace_recursive_calls_with_graph(+Body, +Pred, +GraphPred, +RecCalls, +GraphVars, -GraphBody)
%  Replace all recursive calls in body with graph-building calls
%
replace_recursive_calls_with_graph(Body, Pred, GraphPred, RecCalls, GraphVars, GraphBody) :-
    replace_calls_in_body(Body, Pred, GraphPred, RecCalls, GraphVars, 0, GraphBody).

%% replace_calls_in_body(+Body, +Pred, +GraphPred, +RecCalls, +GraphVars, +Index, -NewBody)
%  Recursively replace calls in body structure
%
replace_calls_in_body(Goal, Pred, GraphPred, RecCalls, GraphVars, Index, NewGoal) :-
    % Check if this goal is a recursive call
    ( functor(Goal, Pred, _) ->
        % Find which recursive call this is
        nth0(Index, RecCalls, Goal), !,
        nth0(Index, GraphVars, GraphVar),
        % Replace: pred(A, R) => pred_graph(A, GraphVar)
        Goal =.. [Pred, InputArg, _ResultVar],
        NewGoal =.. [GraphPred, InputArg, GraphVar]
    ; Goal = (A, B) ->
        % Conjunction
        replace_calls_in_body(A, Pred, GraphPred, RecCalls, GraphVars, Index, NewA),
        count_rec_in_goal(A, Pred, CountA),
        Index2 is Index + CountA,
        replace_calls_in_body(B, Pred, GraphPred, RecCalls, GraphVars, Index2, NewB),
        NewGoal = (NewA, NewB)
    ; Goal = (A; B) ->
        % Disjunction
        replace_calls_in_body(A, Pred, GraphPred, RecCalls, GraphVars, Index, NewA),
        replace_calls_in_body(B, Pred, GraphPred, RecCalls, GraphVars, Index, NewB),
        NewGoal = (NewA; NewB)
    ;
        % Other goal - keep as is
        NewGoal = Goal
    ).

%% count_rec_in_goal(+Goal, +Pred, -Count)
%  Count recursive calls in a single goal or conjunction
%
count_rec_in_goal(Goal, Pred, Count) :-
    findall(1, (
        extract_goal(Goal, G),
        functor(G, Pred, _)
    ), Ones),
    length(Ones, Count).

%% generate_fold_computer(+Pred/Arity, +OrigClauses, -FoldClauses)
%  Generate fold_pred/2 predicate that computes values from structure
%
%  Strategy:
%  1. Leaf case: fold_pred(leaf(V), V).
%  2. Node case: fold_pred(node(_, Children), Result) :-
%                   maplist(fold_pred, Children, Values),
%                   combine_values(Values, Result).
%
generate_fold_computer(Pred/Arity, OrigClauses, FoldClauses) :-
    atom_concat('fold_', Pred, FoldPred),

    % Generate leaf clause: fold_pred(leaf(V), V).
    LeafClause = clause(FoldHead, true),
    FoldHead =.. [FoldPred, leaf(LeafVar), LeafVar],

    % Generate node clause based on original recursive case
    find_recursive_clause(OrigClauses, Pred, RecClause),
    generate_node_fold_clause(Pred, FoldPred, RecClause, NodeClause),

    FoldClauses = [LeafClause, NodeClause].

%% find_recursive_clause(+Clauses, +Pred, -RecClause)
%  Find first recursive clause from original predicate
%
find_recursive_clause(Clauses, Pred, RecClause) :-
    member(RecClause, Clauses),
    RecClause = clause(_Head, Body),
    contains_call_to(Body, Pred), !.

%% generate_node_fold_clause(+Pred, +FoldPred, +RecClause, -NodeClause)
%  Generate fold clause for node case
%
%  Extracts the combination operation from original recursive clause
%
generate_node_fold_clause(Pred, FoldPred, RecClause, NodeClause) :-
    RecClause = clause(Head, Body),
    Head =.. [Pred, _Input, Output],

    % Find all recursive calls
    findall(RecCall, (
        extract_goal(Body, RecCall),
        functor(RecCall, Pred, _)
    ), RecCalls),

    % Extract result variables from recursive calls
    findall(ResultVar, (
        member(Call, RecCalls),
        Call =.. [Pred, _, ResultVar]
    ), ResultVars),

    % Find the combination operation (arithmetic after recursive calls)
    extract_combination_op(Body, ResultVars, Output, CombineOp),

    % Generate fold variables
    length(ResultVars, NumVars),
    length(FoldVars, NumVars),

    % Create fold calls for children
    create_fold_calls(FoldPred, [left, right], FoldVars, FoldCalls),

    % Create combination with folded values
    replace_vars_in_op(CombineOp, ResultVars, FoldVars, FinalOp),

    % Build node clause
    NodeHead =.. [FoldPred, node(_, [left, right]), Output],
    NodeBody = (FoldCalls, FinalOp),
    NodeClause = clause(NodeHead, NodeBody).

%% extract_combination_op(+Body, +ResultVars, +Output, -CombineOp)
%  Extract the operation that combines recursive results
%
extract_combination_op(Body, ResultVars, Output, CombineOp) :-
    % Find 'is' expression that uses result variables
    extract_goal(Body, Goal),
    Goal =.. [is, Output, Expr],
    % Check that expression uses at least one result variable
    term_variables(Expr, ExprVars),
    intersection(ExprVars, ResultVars, Common),
    Common \= [], !,
    CombineOp = (Output is Expr).

%% create_fold_calls(+FoldPred, +Children, +FoldVars, -FoldCalls)
%  Create conjunction of fold calls for each child
%
create_fold_calls(FoldPred, [Child], [FoldVar], FoldCall) :-
    !, FoldCall =.. [FoldPred, Child, FoldVar].
create_fold_calls(FoldPred, [Child|Children], [FoldVar|FoldVars], (FoldCall, RestCalls)) :-
    FoldCall =.. [FoldPred, Child, FoldVar],
    create_fold_calls(FoldPred, Children, FoldVars, RestCalls).

%% replace_vars_in_op(+Op, +OldVars, +NewVars, -NewOp)
%  Replace variables in operation
%
replace_vars_in_op(Op, OldVars, NewVars, NewOp) :-
    copy_term(Op, NewOp),
    replace_vars_list(OldVars, NewVars, NewOp).

replace_vars_list([], [], _).
replace_vars_list([Old|Olds], [New|News], Term) :-
    ( var(Term), Term == Old ->
        Term = New
    ; true ),
    replace_vars_list(Olds, News, Term).

%% generate_wrapper(+Pred/Arity, -WrapperClause)
%  Generate wrapper predicate that combines graph building and folding
%
%  pred_fold(Input, Result) :- pred_graph(Input, Graph), fold_pred(Graph, Result).
%
generate_wrapper(Pred/Arity, WrapperClause) :-
    Arity =:= 2,
    atom_concat(Pred, '_fold', WrapperPred),
    atom_concat(Pred, '_graph', GraphPred),
    atom_concat('fold_', Pred, FoldPred),

    WrapperHead =.. [WrapperPred, Input, Result],
    GraphCall =.. [GraphPred, Input, Graph],
    FoldCall =.. [FoldPred, Graph, Result],
    WrapperBody = (GraphCall, FoldCall),

    WrapperClause = clause(WrapperHead, WrapperBody).

%% install_fold_helpers(+Pred/Arity)
%  Generate and install fold helper predicates into user module
%
install_fold_helpers(Pred/Arity) :-
    generate_fold_helpers(Pred/Arity, Clauses),

    % Install each clause into user module
    forall(member(clause(Head, Body), Clauses), (
        assertz(user:(Head :- Body))
    )).

%% ============================================
%% TESTS
%% ============================================

test_fold_helper_generator :-
    writeln('=== FOLD HELPER GENERATOR TESTS ==='),

    % Setup test fibonacci predicate
    writeln('Test 1: Generate fold helpers for fibonacci'),
    catch(abolish(user:test_fib2/2), _, true),
    catch(abolish(user:test_fib2_graph/2), _, true),
    catch(abolish(user:fold_test_fib2/2), _, true),
    catch(abolish(user:test_fib2_fold/2), _, true),

    assertz(user:(test_fib2(0, 0))),
    assertz(user:(test_fib2(1, 1))),
    assertz(user:(test_fib2(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib2(N1, F1), test_fib2(N2, F2), F is F1 + F2)),

    % Generate helpers
    ( generate_fold_helpers(test_fib2/2, Clauses) ->
        length(Clauses, NumClauses),
        format('  ✓ PASS - generated ~w clauses~n', [NumClauses]),
        % Show generated clauses
        writeln('  Generated clauses:'),
        forall(member(clause(H, B), Clauses), (
            format('    ~w :- ~w~n', [H, B])
        ))
    ;   writeln('  ✗ FAIL - could not generate helpers')
    ),

    writeln('=== FOLD HELPER GENERATOR TESTS COMPLETE ===').
