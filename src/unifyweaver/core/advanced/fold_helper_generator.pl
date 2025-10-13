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
% IMPLEMENTATION: Template-based approach (v0.2)
% See docs/FOLD_GENERATOR_DESIGN.md for design rationale and future work.
%
% This implementation uses templates for common patterns rather than general
% term transformation. This is simpler, more maintainable, and produces
% cleaner code for 80% of use cases.
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
%  Generate all fold helper clauses for a predicate using templates
%
%  Returns: List of clauses for _graph, fold_, and _fold predicates
%
generate_fold_helpers(Pred/Arity, AllClauses) :-
    % Get original clauses
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), OrigClauses),

    % Extract template parameters from original predicate
    extract_template_params(Pred/Arity, OrigClauses, Params),

    % Generate clauses using binary tree template
    generate_from_template(binary_tree, Pred/Arity, Params, AllClauses).

%% extract_template_params(+Pred/Arity, +OrigClauses, -Params)
%  Extract parameters needed for template instantiation
%
%  Params = params(base_cases, operator, rec_clause)
%
extract_template_params(Pred/Arity, OrigClauses, Params) :-
    % Separate base and recursive clauses
    partition_clauses(Pred, OrigClauses, BaseClauses, RecClauses),

    % Get first recursive clause
    RecClauses = [RecClause|_],
    RecClause = clause(RecHead, RecBody),

    % Extract combination operator (can be done on original clause)
    extract_operator(RecBody, _, Operator),

    Params = params(
        base_clauses(BaseClauses),
        operator(Operator),
        rec_clause(RecClause)
    ).

%% partition_clauses(+Pred, +Clauses, -BaseClauses, -RecClauses)
partition_clauses(Pred, Clauses, BaseClauses, RecClauses) :-
    partition(is_recursive_clause(Pred), Clauses, RecClauses, BaseClauses).

is_recursive_clause(Pred, clause(_Head, Body)) :-
    contains_call_to(Body, Pred).

%% extract_guards(+Body, +Pred, +OrigHead, +NewArgs, -Guards)
%  Extract guard conditions and rename variables to match new clause
%
%  Args:
%    Body: Original clause body
%    Pred: Predicate name (to filter out recursive calls)
%    OrigHead: Original clause head (for variable mapping)
%    NewArgs: New arguments to use in generated clause
%    Guards: List of guard goals with renamed variables
%
extract_guards(Body, Pred, OrigHead, NewArgs, Guards) :-
    % Build variable mapping from original to new variables
    build_var_mapping(OrigHead, NewArgs, VarMap),

    % Extract and rename guards
    findall(RenamedGoal, (
        extract_goal(Body, Goal),
        \+ functor(Goal, is, 2),
        \+ functor(Goal, Pred, _),
        is_before_computations(Goal, Body),
        rename_goal_vars(Goal, VarMap, RenamedGoal)
    ), Guards).

is_before_computations(Goal, Body) :-
    % Simplified: assume guards come first
    % More sophisticated: check position in conjunction
    true.

%% extract_computations(+Body, +Pred, +OrigHead, +NewArgs, -Computations, -RecArgs)
%  Extract arithmetic computations and rename variables to match new clause
%
%  Args:
%    Body: Original clause body
%    Pred: Predicate name
%    OrigHead: Original clause head (for variable mapping)
%    NewArgs: New arguments to use in generated clause
%    Computations: List of computation goals with renamed variables
%    RecArgs: Arguments used in recursive calls (with renamed variables)
%
extract_computations(RecBody, Pred, OrigHead, NewArgs, Computations, RecArgs) :-
    % Build variable mapping
    build_var_mapping(OrigHead, NewArgs, VarMap),

    % Find all 'is' expressions that compute inputs for recursive calls
    % (not the final result computation)
    findall(RenamedComp, (
        extract_goal(RecBody, Comp),
        Comp =.. [is, Var, _],
        % Check if this variable is used in a recursive call as input
        extract_goal(RecBody, RecCall),
        functor(RecCall, Pred, _),
        RecCall =.. [Pred, Arg, _],
        Arg == Var,  % Variable is used as input to recursive call
        rename_goal_vars(Comp, VarMap, RenamedComp)
    ), ComputationsWithDups),
    sort(ComputationsWithDups, Computations),

    % Extract arguments used in recursive calls (renamed)
    findall(RenamedArg, (
        extract_goal(RecBody, RecCall),
        functor(RecCall, Pred, _),
        RecCall =.. [Pred, Arg, _],
        rename_goal_vars(Arg, VarMap, RenamedArg)
    ), RecArgs).

%% extract_operator(+Body, +OutputVar, -Operator)
%  Extract the binary operator used to combine results
%
extract_operator(Body, OutputVar, Operator) :-
    % Find the 'is' expression that computes the output
    extract_goal(Body, Goal),
    Goal =.. [is, OutputVar, Expr],
    % Extract operator from expression (e.g., +(F1, F2) -> '+')
    (   Expr =.. [Operator, _, _],
        member(Operator, [+, -, *, /, max, min])
    ->  true
    ;   Operator = unknown
    ).

%% ============================================
%% VARIABLE MAPPING HELPERS
%% ============================================
% These helpers solve the variable binding problem when extracting goals
% from original clauses. Variables in extracted goals are bound to the
% original clause's scope. We need to rename them to match the new clause.

%% build_var_mapping(+OrigHead, +NewArgs, -VarMap)
%  Create a mapping from original clause variables to new clause variables
%
%  Args:
%    OrigHead: Original clause head (e.g., fib(N, F))
%    NewArgs: List of new variables to use (e.g., [_InputVar, _OutputVar])
%    VarMap: List of pairs mapping old vars to new vars (e.g., [N-_InputVar, F-_OutputVar])
%
build_var_mapping(OrigHead, NewArgs, VarMap) :-
    OrigHead =.. [_Pred|OrigArgs],
    pairs_keys_values(VarMap, OrigArgs, NewArgs).

%% rename_goal_vars(+Goal, +VarMap, -RenamedGoal)
%  Rename variables in a goal using the variable mapping
%
%  Args:
%    Goal: Original goal with old variables (e.g., N > 1)
%    VarMap: Mapping from old to new variables (e.g., [N-_InputVar, ...])
%    RenamedGoal: Goal with renamed variables (e.g., _InputVar > 1)
%
rename_goal_vars(Goal, VarMap, RenamedGoal) :-
    copy_term(Goal, RenamedGoal),
    apply_var_mapping(RenamedGoal, VarMap).

%% apply_var_mapping(+Term, +VarMap)
%  Recursively apply variable mapping to a term (modifies term in place via unification)
%
%  Args:
%    Term: The term to modify
%    VarMap: List of OldVar-NewVar pairs
%
apply_var_mapping(Term, VarMap) :-
    (   var(Term) ->
        % If Term is a variable, try to find it in the mapping
        (   member(OldVar-NewVar, VarMap),
            OldVar == Term  % Use == to check if same variable
        ->  Term = NewVar   % Unify with new variable
        ;   true            % Not in mapping, leave as is
        )
    ;   compound(Term) ->
        % Recursively process compound terms
        Term =.. [_Functor|Args],
        maplist(apply_var_mapping_to_arg(VarMap), Args)
    ;   % Atomic term (number, atom), leave as is
        true
    ).

%% apply_var_mapping_to_arg(+VarMap, +Arg)
%  Helper to apply mapping to a single argument
%
apply_var_mapping_to_arg(VarMap, Arg) :-
    apply_var_mapping(Arg, VarMap).

%% rename_goal_vars_with_map(+VarMap, +Goal, -RenamedGoal)
%  Helper for maplist to rename goal variables
%  (maplist expects the predicate argument to come first)
%
rename_goal_vars_with_map(VarMap, Goal, RenamedGoal) :-
    rename_goal_vars(Goal, VarMap, RenamedGoal).

%% transform_body_for_graph(+Body, +Pred, +GraphPred, +Arity, -TransformedBody, -GraphVars)
%  Transform clause body for graph generation:
%  - Replace recursive calls: pred(Args..., Res) => pred_graph(Args..., GraphVar)
%  - Remove final result computation (the 'is' that computes output)
%  - Keep guards and argument computations
%  - Collect GraphVars from transformed recursive calls
%
transform_body_for_graph(Body, Pred, GraphPred, Arity, TransformedBody, GraphVars) :-
    transform_body_goals(Body, Pred, GraphPred, Arity, [], TransformedBody, GraphVars).

%% transform_body_goals(+Body, +Pred, +GraphPred, +Arity, +Seen, -Transformed, -GraphVars)
%  Recursively transform body goals and collect graph variables
%  Seen contains seen goals in reverse order (most recent first)
%
transform_body_goals((A, B), Pred, GraphPred, Arity, Seen, Transformed, GraphVars) :- !,
    transform_body_goals(A, Pred, GraphPred, Arity, Seen, TransA, GVarsA),
    transform_body_goals(B, Pred, GraphPred, Arity, [A|Seen], TransB, GVarsB),
    append(GVarsA, GVarsB, GraphVars),
    (   TransA = true ->
        Transformed = TransB
    ;   TransB = true ->
        Transformed = TransA
    ;   Transformed = (TransA, TransB)
    ).
transform_body_goals(Goal, Pred, GraphPred, Arity, _Seen, TransformedGoal, [GraphVar]) :-
    % Check if this is a recursive call
    functor(Goal, Pred, Arity), !,
    % Transform: pred(Args..., _Res) => pred_graph(Args..., GraphVar)
    % Extract all args, split into inputs and output
    Goal =.. [Pred|Args],
    append(InputArgs, [_Res], Args),
    % Build new call with inputs + graph var
    append(InputArgs, [GraphVar], GraphCallArgs),
    TransformedGoal =.. [GraphPred|GraphCallArgs].
transform_body_goals(Goal, Pred, _GraphPred, _Arity, Seen, true, []) :-
    % Check if this is the final result computation (should be removed)
    % It's an 'is' goal that comes AFTER recursive calls
    Goal =.. [is, _Var, _Expr],
    has_seen_recursive_call(Seen, Pred), !.
transform_body_goals(Goal, _Pred, _GraphPred, _Arity, _Seen, Goal, []).

%% has_seen_recursive_call(+Seen, +Pred)
%  Check if we've seen a recursive call in the Seen list
%
has_seen_recursive_call([Goal|_], Pred) :-
    functor(Goal, Pred, _), !.
has_seen_recursive_call([_|Rest], Pred) :-
    has_seen_recursive_call(Rest, Pred).

%% member_goal_with_var(+Goals, +Var)
%  Check if Var appears in any of the goals
%
member_goal_with_var([Goal|_], Var) :-
    term_variables(Goal, Vars),
    member(V, Vars),
    V == Var, !.
member_goal_with_var([_|Rest], Var) :-
    member_goal_with_var(Rest, Var).

%% generate_from_template(+TemplateType, +Pred/Arity, +Params, -AllClauses)
%  Generate all clauses from template
%
generate_from_template(binary_tree, Pred/Arity, Params, AllClauses) :-
    atom_concat(Pred, '_graph', GraphPred),
    atom_concat('fold_', Pred, FoldPred),
    atom_concat(Pred, '_fold', WrapperPred),

    % Extract parameters
    Params = params(
        base_clauses(BaseClauses),
        operator(Operator),
        rec_clause(RecClause)
    ),

    % Generate graph builder base clauses
    findall(GraphBaseClause, (
        member(clause(BaseHead, BaseBody), BaseClauses),
        BaseHead =.. [Pred|BaseArgs],
        % Split: all args but last are inputs, last is output
        append(BaseInputs, [BaseOutput], BaseArgs),
        % Graph head: pred_graph(Inputs..., leaf(Output))
        append(BaseInputs, [leaf(BaseOutput)], GraphHeadArgs),
        GraphBaseHead =.. [GraphPred|GraphHeadArgs],
        GraphBaseClause = clause(GraphBaseHead, BaseBody)
    ), GraphBaseClauses),

    % Generate graph builder recursive clause
    % Copy the entire recursive clause to get fresh variables with correct sharing
    RecClause = clause(RecHead, RecBody),
    copy_term(clause(RecHead, RecBody), clause(NewRecHead, NewRecBody)),

    % Extract input args and output arg from the copied head
    NewRecHead =.. [Pred|NewArgs],
    append(InputVars, [_OutputVar], NewArgs),

    % Transform the body and collect graph variables
    transform_body_for_graph(NewRecBody, Pred, GraphPred, Arity, TransformedBody, GraphVars),

    % Build graph head: pred_graph(Inputs..., node(_, GraphVars))
    % Keep the node with 2 args for compatibility with fold template
    append(InputVars, [node(_, GraphVars)], GraphRecHeadArgs),
    GraphRecHead =.. [GraphPred|GraphRecHeadArgs],
    GraphRecClause = clause(GraphRecHead, TransformedBody),

    % Generate fold computer leaf clause
    % Use fresh variable for leaf value
    FoldLeafHead =.. [FoldPred, leaf(_V), _V],
    FoldLeafClause = clause(FoldLeafHead, true),

    % Generate fold computer node clause
    % Use fresh variables for children and their values
    FoldRecCall1 =.. [FoldPred, _FL, _VL],
    FoldRecCall2 =.. [FoldPred, _FR, _VR],
    CombineExpr =.. [Operator, _VL, _VR],
    FoldCombine = (_OutputVar is CombineExpr),
    FoldRecBody = (FoldRecCall1, FoldRecCall2, FoldCombine),
    FoldRecHead =.. [FoldPred, node(_, [_FL, _FR]), _OutputVar],
    FoldRecClause = clause(FoldRecHead, FoldRecBody),

    % Generate wrapper clause
    % Wrapper takes all input args plus output arg
    length(InputVars, NumInputs),
    length(WrapperInputVars, NumInputs),
    % Build wrapper head: pred_fold(Inputs..., Output)
    append(WrapperInputVars, [_WOutputVar], WrapperHeadArgs),
    WrapperHead =.. [WrapperPred|WrapperHeadArgs],
    % Build graph call: pred_graph(Inputs..., Graph)
    append(WrapperInputVars, [_Graph], GraphCallArgs),
    WrapperGraphCall =.. [GraphPred|GraphCallArgs],
    % Build fold call: fold_pred(Graph, Output)
    WrapperFoldCall =.. [FoldPred, _Graph, _WOutputVar],
    WrapperBody = (WrapperGraphCall, WrapperFoldCall),
    WrapperClause = clause(WrapperHead, WrapperBody),

    % Combine all clauses
    append([GraphBaseClauses, [GraphRecClause], [FoldLeafClause, FoldRecClause], [WrapperClause]], AllClauses).

%% build_conjunction(+Goals, -Conjunction)
%  Build conjunction from list of goals
%
build_conjunction([], true) :- !.
build_conjunction([Goal], Goal) :- !.
build_conjunction([Goal|Goals], (Goal, Rest)) :-
    build_conjunction(Goals, Rest).

%% ============================================
%% COMPATIBILITY STUBS
%% ============================================
% These predicates are exported for backwards compatibility
% They delegate to the new template-based implementation

%% generate_graph_builder(+Pred/Arity, +OrigClauses, -GraphClauses)
%  Extract only graph builder clauses from full generation
%
generate_graph_builder(Pred/Arity, _OrigClauses, GraphClauses) :-
    generate_fold_helpers(Pred/Arity, AllClauses),
    atom_concat(Pred, '_graph', GraphPred),
    include(is_clause_for_pred(GraphPred), AllClauses, GraphClauses).

%% generate_fold_computer(+Pred/Arity, +OrigClauses, -FoldClauses)
%  Extract only fold computer clauses from full generation
%
generate_fold_computer(Pred/Arity, _OrigClauses, FoldClauses) :-
    generate_fold_helpers(Pred/Arity, AllClauses),
    atom_concat('fold_', Pred, FoldPred),
    include(is_clause_for_pred(FoldPred), AllClauses, FoldClauses).

%% generate_wrapper(+Pred/Arity, -WrapperClause)
%  Extract only wrapper clause from full generation
%
generate_wrapper(Pred/Arity, WrapperClause) :-
    generate_fold_helpers(Pred/Arity, AllClauses),
    atom_concat(Pred, '_fold', WrapperPred),
    include(is_clause_for_pred(WrapperPred), AllClauses, [WrapperClause]).

is_clause_for_pred(Pred, clause(Head, _Body)) :-
    functor(Head, Pred, _).

%% ============================================
%% OLD IMPLEMENTATION (KEPT FOR REFERENCE)
%% ============================================
% The following predicates implement the general term transformation approach.
% See docs/FOLD_GENERATOR_DESIGN.md for details.
% This code is kept for future reference but not currently used.

%% generate_graph_builder_old(+Pred/Arity, +OrigClauses, -GraphClauses)
%  [OLD IMPLEMENTATION - NOT USED]
%  Generate _graph/2 predicate that builds dependency tree structure
%
generate_graph_builder_old(Pred/Arity, OrigClauses, GraphClauses) :-
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
