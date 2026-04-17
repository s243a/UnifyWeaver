:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% clause_body_analysis.pl — Target-independent clause body analysis
%
% Provides shared infrastructure for native clause body lowering across
% all targets. Extracted from typr_target.pl (PRs #826–#898).
%
% This module handles the ANALYSIS side (structure detection, variable
% tracking) while targets provide the CODE GENERATION side (syntax,
% formatting, indentation).

:- module(clause_body_analysis, [
    % Goal normalization
    normalize_goals/2,              % +Body, -GoalList

    % Control flow pattern matching
    if_then_else_goal/4,            % +Goal, -If, -Then, -Else
    if_then_goal/3,                 % +Goal, -If, -Then
    disjunction_alternatives/2,     % +Goal, -Alternatives

    % Goal classification
    classify_goal/3,                % +Goal, +VarMap, -Classification
    is_guard_goal/2,                % +Goal, +VarMap
    is_output_goal/2,               % +Goal, +VarMap
    goal_output_var/2,              % +Goal, -OutputVar
    goal_output_vars/2,             % +Goal, -OutputVars

    % Shared output variable detection
    shared_output_var/3,            % +Alternatives, +VarMap, -SharedVar
    shared_output_vars/3,           % +Alternatives, +VarMap, -SharedVars
    if_then_else_shared_output_var/4,   % +Then, +Else, +VarMap, -SharedVar
    if_then_else_shared_output_vars/4,  % +Then, +Else, +VarMap, -SharedVars
    if_then_output_var/3,           % +Then, +VarMap, -OutputVar
    if_then_output_vars/3,          % +Then, +VarMap, -OutputVars

    % VarMap management
    build_head_varmap/3,            % +HeadArgs, +StartIndex, -VarMap
    lookup_var/3,                   % +Var, +VarMap, -Name
    ensure_var/5,                   % +VarMap, +Var, -Name, -VarMapOut, -IntroKind
    ensure_var/4,                   % +VarMap, +Var, -Name, -VarMapOut
    ensure_vars/4,                  % +Vars, +VarMap, -Pairs, -VarMapOut
    varmap_contains_var/2,          % +VarMap, +Var
    exclude_varmap_vars/3,          % +VarMap, +Vars, -Filtered
    reserve_internal_var/5,         % +VarMap, -Token, -Name, -VarMapOut, -IntroKind
    remove_var_mapping/3,           % +Var, +VarMap, -VarMapOut
    next_var_index/2,               % +VarMap, -NextIndex

    % Variable identity utilities
    var_member_by_identity/2,       % +Var, +List
    unique_vars_by_identity/2,      % +Vars, -Unique
    intersect_output_vars/3,        % +Alternative, +Candidates, -Intersected

    % Head condition analysis
    head_conditions/3,              % +HeadArgs, +StartIndex, -Conditions
    head_arg_is_var/1,              % +Arg — true if arg is unbound variable

    % Expression analysis
    expr_op/2,                      % +PrologOp, -StandardOp
    translate_expr/3,               % +Expr, +VarMap, -TranslatedParts
    literal_value/2,                % +PrologValue, -LiteralParts

    % Clause structure analysis
    analyze_clauses/2,              % +Clauses, -Analysis
    clause_guard_output_split/4,    % +Goals, +VarMap, -Guards, -Outputs

    % Advanced goal sequence analysis (inspired by TypR native lowering)
    classify_goal_sequence/3,       % +Goals, +VarMap, -ClassifiedGoals
    is_output_control_flow/3,       % +Goal, +VarMap, -OutputInfo
    if_then_else_output_info/4,     % +If, +Then, +Else, -OutputInfo
    disjunction_output_info/2,      % +Alternatives, -OutputInfo
    if_then_output_info/3,          % +If, +Then, -OutputInfo

    % Multi-clause compilation helper
    compile_multi_clause/4,         % +Clauses, +VarMap, +Strategy, -Analysis

    % Recursive compile_expression framework
    compile_expression/6,           % +Target, +Goal, +VarMap, -Code, -OutputVars, -VarMapOut
    compile_branch/6,               % +Target, +Branch, +VarMap, -Lines, -OutputVars, -VarMapOut
    compile_classified_sequence/5,  % +Target, +ClassifiedGoals, +VarMap, -Lines, -VarMapOut

    % Parallelism analysis
    classify_parallelism/3,         % +PredIndicator, +Clauses, -Strategy
    is_order_independent/2,         % +PredIndicator, -Reason
    order_independent/1,            % Directive
    parallel_safe/1                 % Directive
]).

% Directives
:- dynamic order_independent/1.
:- dynamic parallel_safe/1.

%% Multifile hooks for target-specific rendering
:- multifile render_output_goal/6.      % +Target, +Goal, +VarMap, -Line, -VarName, -VarMapOut
:- multifile render_guard_condition/4.  % +Target, +Goal, +VarMap, -CondStr
:- multifile render_branch_value/4.     % +Target, +Branch, +VarMap, -ExprStr
:- multifile render_assignment/5.       % +Target, +VarName, +ExprStr, +Indent, -Line
:- multifile render_return/4.           % +Target, +VarNames, +Indent, -Line
:- multifile render_ite_block/7.        % +Target, +Cond, +ThenLines, +ElseLines, +Indent, +ReturnVars, -Lines

:- use_module(library(lists)).
:- use_module(purity_certificate).

% ============================================================================
% GOAL NORMALIZATION
% ============================================================================

%% normalize_goals(+Body, -GoalList)
%  Flatten a Prolog clause body (conjunction tree) into a list of goals.
%  Strips module qualifications and handles true (empty body).
normalize_goals(true, []) :- !.
normalize_goals((Left, Right), Goals) :-
    !,
    normalize_goals(Left, LeftGoals),
    normalize_goals(Right, RightGoals),
    append(LeftGoals, RightGoals, Goals).
normalize_goals(_Module:Goal, Goals) :-
    !,
    normalize_goals(Goal, Goals).
normalize_goals(Goal, [Goal]).

% ============================================================================
% CONTROL FLOW PATTERN MATCHING
% ============================================================================

%% if_then_else_goal(+Goal, -IfGoal, -ThenGoal, -ElseGoal)
%  Match Prolog's (If -> Then ; Else) pattern.
%  SWI-Prolog may represent this as ;(->(If,Then), Else).
if_then_else_goal((IfGoal -> ThenGoal ; ElseGoal), IfGoal, ThenGoal, ElseGoal) :- !.
if_then_else_goal(;(->(IfGoal, ThenGoal), ElseGoal), IfGoal, ThenGoal, ElseGoal) :- !.

%% if_then_goal(+Goal, -IfGoal, -ThenGoal)
%  Match Prolog's (If -> Then) pattern (no else branch).
if_then_goal((IfGoal -> ThenGoal), IfGoal, ThenGoal) :- !.
if_then_goal(->(IfGoal, ThenGoal), IfGoal, ThenGoal) :- !.

%% disjunction_alternatives(+Goal, -Alternatives)
%  Flatten a Prolog disjunction (A ; B ; C) into a list [A, B, C].
disjunction_alternatives((Left ; Right), Alternatives) :-
    !,
    disjunction_alternatives(Left, LeftAlts),
    disjunction_alternatives(Right, RightAlts),
    append(LeftAlts, RightAlts, Alternatives).
disjunction_alternatives(Goal, [Goal]).

% ============================================================================
% GOAL CLASSIFICATION
% ============================================================================

%% classify_goal(+Goal, +VarMap, -Classification)
%  Classify a goal as guard, output, multi_result_output, or control_flow.
classify_goal(Goal, VarMap, control_flow) :-
    (if_then_else_goal(Goal, _, _, _) ; if_then_goal(Goal, _, _)),
    !,
    (   is_output_goal(Goal, VarMap)
    ->  true   % it's control_flow that produces output
    ;   true   % it's control_flow that's a guard
    ).
classify_goal(Goal, _VarMap, output) :-
    goal_output_var(Goal, _),
    !.
classify_goal(Goal, VarMap, guard) :-
    is_guard_goal(Goal, VarMap),
    !.
classify_goal(_, _, unknown).

%% is_guard_goal(+Goal, +VarMap)
%  A goal is a guard if it's a comparison or type check that produces no
%  new variable bindings (only tests existing values).
is_guard_goal(true, _) :- !.
is_guard_goal(fail, _) :- !.
is_guard_goal(false, _) :- !.
%% Negation-as-failure: \+ Goal is a guard (boolean test, no bindings)
is_guard_goal(\+(Goal), VarMap) :- !,
    (is_guard_goal(Goal, VarMap) -> true ; true).
is_guard_goal(not(Goal), VarMap) :- !,
    (is_guard_goal(Goal, VarMap) -> true ; true).
is_guard_goal(_Module:Goal, VarMap) :-
    !,
    is_guard_goal(Goal, VarMap).
%% If-then-else used as guard (no output variables)
is_guard_goal(Goal, VarMap) :-
    if_then_else_goal(Goal, _, _, _),
    \+ is_output_goal(Goal, VarMap),
    !.
is_guard_goal(Goal, _VarMap) :-
    compound(Goal),
    Goal =.. [Op, _, _],
    comparison_op(Op),
    !.
is_guard_goal(Goal, _VarMap) :-
    compound(Goal),
    Goal =.. [TypeCheck, _],
    type_check_pred(TypeCheck),
    !.

%% is_output_goal(+Goal, +VarMap)
%  A goal is an output if it binds a variable not already in VarMap.
is_output_goal(Goal, VarMap) :-
    goal_output_var(Goal, Var),
    var(Var),
    \+ varmap_contains_var(VarMap, Var).

%% comparison_op(+Op)
%  Prolog comparison operators that act as guards.
comparison_op(>).
comparison_op(<).
comparison_op(>=).
comparison_op(=<).
comparison_op(=:=).
comparison_op(=\=).
comparison_op(==).
comparison_op(\==).
comparison_op(\=).

%% type_check_pred(+Pred)
%  Prolog type check predicates that act as guards.
type_check_pred(integer).
type_check_pred(float).
type_check_pred(number).
type_check_pred(atom).
type_check_pred(is_list).
type_check_pred(compound).
type_check_pred(var).
type_check_pred(nonvar).

% ============================================================================
% OUTPUT VARIABLE DETECTION
% ============================================================================

%% goal_output_var(+Goal, -OutputVar)
%  Extract the primary output variable from a goal.
goal_output_var(_Module:Goal, OutputVar) :-
    !,
    goal_output_var(Goal, OutputVar).
goal_output_var(Goal, OutputVar) :-
    goal_output_vars(Goal, [OutputVar]),
    !.
goal_output_var(Goal, OutputVar) :-
    goal_output_var_simple(Goal, OutputVar).

%% goal_output_vars(+Goal, -OutputVars)
%  Extract all output variables from a goal (handles nested control flow).
goal_output_vars(_Module:Goal, OutputVars) :-
    !,
    goal_output_vars(Goal, OutputVars).
goal_output_vars(Goal, OutputVars) :-
    if_then_else_goal(Goal, _IfGoal, ThenGoal, ElseGoal),
    if_then_else_goal_output_vars(ThenGoal, ElseGoal, OutputVars),
    !.
goal_output_vars(Goal, OutputVars) :-
    if_then_goal(Goal, _IfGoal, ThenGoal),
    if_then_goal_output_vars(ThenGoal, OutputVars),
    !.
goal_output_vars(Goal, OutputVars) :-
    disjunction_alternatives(Goal, Alternatives),
    Alternatives = [_|[_|_]],
    disjunction_goal_output_vars(Alternatives, OutputVars),
    !.
goal_output_vars(Goal, [OutputVar]) :-
    goal_output_var_simple(Goal, OutputVar).

%% goal_output_var_simple(+Goal, -OutputVar)
%  Extract output var from simple goals (Var is Expr, Var = Value).
goal_output_var_simple(_Module:Goal, OutputVar) :-
    !,
    goal_output_var_simple(Goal, OutputVar).
goal_output_var_simple(Var is _, Var) :- var(Var), !.
goal_output_var_simple(Var = _, Var) :- var(Var), !.
goal_output_var_simple(_ = Var, Var) :- var(Var), !.
goal_output_var_simple(Goal, OutputVar) :-
    compound(Goal),
    Goal =.. [Pred|Args],
    \+ comparison_op(Pred),
    \+ type_check_pred(Pred),
    last(Args, OutputVar),
    var(OutputVar),
    !.

%% alternative_output_var(+Alternative, -OutputVar)
%  Find the last output variable in an alternative branch.
alternative_output_var(Alternative, OutputVar) :-
    normalize_goals(Alternative, Goals),
    reverse(Goals, ReversedGoals),
    member(Goal, ReversedGoals),
    goal_output_var(Goal, OutputVar),
    !.

%% alternative_output_vars(+Alternative, -OutputVars)
%  Collect all output variables from an alternative branch.
alternative_output_vars(Alternative, OutputVars) :-
    normalize_goals(Alternative, Goals),
    collect_goal_output_vars(Goals, OutputVars0),
    unique_vars_by_identity(OutputVars0, OutputVars).

collect_goal_output_vars([], []).
collect_goal_output_vars([Goal|Rest], OutputVars) :-
    (   goal_output_vars(Goal, GoalOutputVars)
    ->  append(GoalOutputVars, RestOutputs, OutputVars)
    ;   OutputVars = RestOutputs
    ),
    collect_goal_output_vars(Rest, RestOutputs).

%% Nested output var helpers
if_then_else_goal_output_vars(ThenGoal, ElseGoal, OutputVars) :-
    alternative_output_vars(ThenGoal, ThenOutputVars),
    intersect_output_vars(ElseGoal, ThenOutputVars, OutputVars),
    OutputVars \= [].

if_then_goal_output_vars(ThenGoal, OutputVars) :-
    alternative_output_vars(ThenGoal, OutputVars),
    OutputVars \= [].

disjunction_goal_output_vars([Alternative|Rest], OutputVars) :-
    alternative_output_vars(Alternative, FirstOutputVars),
    foldl(intersect_output_vars, Rest, FirstOutputVars, OutputVars),
    OutputVars \= [].

% ============================================================================
% SHARED OUTPUT VARIABLE DETECTION
% ============================================================================

%% shared_output_var(+Alternatives, +VarMap, -SharedVar)
%  Find a single variable that all alternatives assign to.
shared_output_var([Alternative|Rest], VarMap, SharedVar) :-
    alternative_output_var(Alternative, SharedVar),
    var(SharedVar),
    output_var_allowed(VarMap, SharedVar),
    maplist(alternative_output_var_matches(SharedVar), Rest).

%% shared_output_vars(+Alternatives, +VarMap, -SharedVars)
%  Find all variables that all alternatives assign to.
shared_output_vars([Alternative|Rest], VarMap, SharedVars) :-
    alternative_output_vars(Alternative, FirstOutputVars0),
    include(output_var_allowed(VarMap), FirstOutputVars0, FirstOutputVars),
    foldl(intersect_output_vars, Rest, FirstOutputVars, SharedVars),
    SharedVars \= [].

%% if_then_else_shared_output_var(+Then, +Else, +VarMap, -SharedVar)
if_then_else_shared_output_var(ThenGoal, ElseGoal, VarMap, SharedVar) :-
    alternative_output_var(ThenGoal, SharedVar),
    var(SharedVar),
    output_var_allowed(VarMap, SharedVar),
    alternative_output_var(ElseGoal, ElseVar),
    ElseVar == SharedVar.

%% if_then_else_shared_output_vars(+Then, +Else, +VarMap, -SharedVars)
%  Find variables bound by both branches. Allows head arg variables
%  (arg1, arg2...) since these are output positions in the clause.
if_then_else_shared_output_vars(ThenGoal, ElseGoal, VarMap, SharedVars) :-
    alternative_output_vars(ThenGoal, ThenOutputVars0),
    include(output_var_allowed(VarMap), ThenOutputVars0, ThenOutputVars),
    intersect_output_vars(ElseGoal, ThenOutputVars, SharedVars),
    SharedVars \= [].

%% if_then_output_var(+Then, +VarMap, -OutputVar)
if_then_output_var(ThenGoal, VarMap, OutputVar) :-
    alternative_output_var(ThenGoal, OutputVar),
    var(OutputVar),
    output_var_allowed(VarMap, OutputVar).

%% if_then_output_vars(+Then, +VarMap, -OutputVars)
if_then_output_vars(ThenGoal, VarMap, OutputVars) :-
    alternative_output_vars(ThenGoal, OutputVars0),
    include(output_var_allowed(VarMap), OutputVars0, OutputVars),
    OutputVars \= [].

%% output_var_allowed(+VarMap, +Var)
%  A variable can be an output if it's not in VarMap, or if it's an arg.
output_var_allowed(VarMap, Var) :-
    \+ varmap_contains_var(VarMap, Var),
    !.
output_var_allowed(VarMap, Var) :-
    lookup_var(Var, VarMap, ArgName),
    sub_string(ArgName, 0, 3, _, "arg").

alternative_output_var_matches(ExpectedVar, Alternative) :-
    alternative_output_var(Alternative, ActualVar),
    ActualVar == ExpectedVar.

% ============================================================================
% VARMAP MANAGEMENT
% ============================================================================

%% build_head_varmap(+HeadArgs, +StartIndex, -VarMap)
%  Create initial VarMap from predicate head arguments.
%  Variables become arg1, arg2, ...; non-variables are skipped.
build_head_varmap([], _, []).
build_head_varmap([Arg|Args], Index, [Arg-ArgName|Rest]) :-
    var(Arg),
    !,
    format(string(ArgName), 'arg~w', [Index]),
    NextIndex is Index + 1,
    build_head_varmap(Args, NextIndex, Rest).
build_head_varmap([_|Args], Index, Rest) :-
    NextIndex is Index + 1,
    build_head_varmap(Args, NextIndex, Rest).

%% lookup_var(+Var, +VarMap, -Name)
%  Find a variable's target name in VarMap using == identity.
lookup_var(Var, [StoredVar-Name|_], Name) :-
    Var == StoredVar,
    !.
lookup_var(Var, [_|Rest], Name) :-
    lookup_var(Var, Rest, Name).

%% ensure_var(+VarMap, +Var, -Name, -VarMapOut, -IntroKind)
%  Look up or allocate a variable. IntroKind is 'existing' or 'new'.
ensure_var(VarMap, Var, Name, VarMap, existing) :-
    lookup_var(Var, VarMap, Name),
    !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap], new) :-
    next_var_index(VarMap, NextIndex),
    format(string(Name), 'v~w', [NextIndex]).

%% ensure_var/4 — without IntroKind
ensure_var(VarMap, Var, Name, VarMapOut) :-
    ensure_var(VarMap, Var, Name, VarMapOut, _).

%% ensure_vars(+Vars, +VarMap, -Pairs, -VarMapOut)
%  Allocate multiple variables, returning Var-Name pairs.
ensure_vars([], VarMap, [], VarMap).
ensure_vars([Var|Rest], VarMap0, [Var-Name|RestPairs], VarMapOut) :-
    ensure_var(VarMap0, Var, Name, VarMap1, _),
    ensure_vars(Rest, VarMap1, RestPairs, VarMapOut).

%% varmap_contains_var(+VarMap, +Var)
varmap_contains_var([StoredVar-_|_], Var) :-
    Var == StoredVar, !.
varmap_contains_var([_|Rest], Var) :-
    varmap_contains_var(Rest, Var).

%% exclude_varmap_vars(+VarMap, +Vars, -Filtered)
%  Remove vars that are already in VarMap.
exclude_varmap_vars(_VarMap, [], []).
exclude_varmap_vars(VarMap, [Var|Rest], Filtered) :-
    (   varmap_contains_var(VarMap, Var)
    ->  exclude_varmap_vars(VarMap, Rest, Filtered)
    ;   Filtered = [Var|FilteredRest],
        exclude_varmap_vars(VarMap, Rest, FilteredRest)
    ).

%% reserve_internal_var(+VarMap, -Token, -Name, -VarMapOut, -IntroKind)
%  Allocate a temporary internal variable (for containers, etc.).
reserve_internal_var(VarMap0, Token, Name, VarMap, IntroKind) :-
    ensure_var(VarMap0, Token, Name, VarMap, IntroKind).

%% remove_var_mapping(+Var, +VarMap, -VarMapOut)
%  Remove a variable from VarMap (used after container extraction).
remove_var_mapping(_Var, [], []).
remove_var_mapping(Var, [StoredVar-Name|Rest], Filtered) :-
    (   Var == StoredVar
    ->  Filtered = FilteredRest
    ;   Filtered = [StoredVar-Name|FilteredRest]
    ),
    remove_var_mapping(Var, Rest, FilteredRest).

%% next_var_index(+VarMap, -NextIndex)
%  Get the next available variable index (v1, v2, ...).
next_var_index([], 1).
next_var_index(VarMap, NextIndex) :-
    findall(Index, (
        member(_-Name, VarMap),
        name_index(Name, Index)
    ), Indices),
    (   Indices = []
    ->  NextIndex = 1
    ;   max_list(Indices, MaxIndex),
        NextIndex is MaxIndex + 1
    ).

%% name_index(+Name, -Index)
%  Extract numeric index from a variable name like "arg3" or "v5".
name_index(Name, Index) :-
    string(Name),
    (   sub_string(Name, 0, 3, _, "arg")
    ->  sub_string(Name, 3, _, 0, Digits)
    ;   sub_string(Name, 0, 1, _, "v"),
        sub_string(Name, 1, _, 0, Digits)
    ),
    number_string(Index, Digits).

% ============================================================================
% VARIABLE IDENTITY UTILITIES
% ============================================================================

%% var_member_by_identity(+Var, +List)
%  Check if Var is in List using == (not unification).
var_member_by_identity(Var, [Candidate|_]) :-
    Var == Candidate, !.
var_member_by_identity(Var, [_|Rest]) :-
    var_member_by_identity(Var, Rest).

%% unique_vars_by_identity(+Vars, -Unique)
%  Remove duplicates using == identity.
unique_vars_by_identity([], []).
unique_vars_by_identity([Var|Rest], Unique) :-
    unique_vars_by_identity(Rest, RestUnique),
    (   var_member_by_identity(Var, RestUnique)
    ->  Unique = RestUnique
    ;   Unique = [Var|RestUnique]
    ).

%% intersect_output_vars(+Alternative, +Candidates, -Intersected)
%  Keep only candidates that also appear as outputs of Alternative.
intersect_output_vars(Alternative, Candidates0, Candidates) :-
    alternative_output_vars(Alternative, OutputVars),
    include_vars_by_identity(Candidates0, OutputVars, Candidates).

include_vars_by_identity([], _Allowed, []).
include_vars_by_identity([Var|Rest], Allowed, Included) :-
    (   var_member_by_identity(Var, Allowed)
    ->  Included = [Var|IncludedRest]
    ;   Included = IncludedRest
    ),
    include_vars_by_identity(Rest, Allowed, IncludedRest).

% ============================================================================
% HEAD CONDITION ANALYSIS
% ============================================================================

%% head_conditions(+HeadArgs, +StartIndex, -Conditions)
%  Extract conditions from predicate head arguments.
%  Variables produce no condition; constants produce equality checks.
%  Returns a list of condition(ArgIndex, Value) terms.
head_conditions([], _, []).
head_conditions([HeadArg|Rest], Index, Conditions) :-
    (   var(HeadArg)
    ->  Conditions = RestConditions
    ;   Conditions = [condition(Index, HeadArg)|RestConditions]
    ),
    NextIndex is Index + 1,
    head_conditions(Rest, NextIndex, RestConditions).

%% head_arg_is_var(+Arg) — true if arg is unbound variable
head_arg_is_var(Arg) :- var(Arg).

% ============================================================================
% EXPRESSION ANALYSIS
% ============================================================================

%% expr_op(+PrologOp, -StandardOp)
%  Map Prolog operators to standard operator names.
%  Targets translate StandardOp to their own syntax.
expr_op(>, '>').
expr_op(<, '<').
expr_op(>=, '>=').
expr_op(=<, '<=').
expr_op(=:=, '==').
expr_op(=\=, '!=').
expr_op(==, '==').
expr_op(\=, '!=').
expr_op(\==, '!=').
expr_op(+, '+').
expr_op(-, '-').
expr_op(*, '*').
expr_op(/, '/').
expr_op(//, 'floor_div').
expr_op(mod, '%').
expr_op(and, '&&').
expr_op(or, '||').

%% translate_expr(+Expr, +VarMap, -Parts)
%  Decompose a Prolog expression into parts for target translation.
%  Returns structured terms: var(Name), literal(Value), op(Op, Left, Right),
%  call(Fn, Args), negation(Inner).
translate_expr(Var, VarMap, var(Name)) :-
    var(Var), !,
    lookup_var(Var, VarMap, Name).
translate_expr(Atom, _VarMap, literal(Atom)) :-
    atom(Atom), !.
translate_expr(Text, _VarMap, literal(Text)) :-
    string(Text), !.
translate_expr(Number, _VarMap, literal(Number)) :-
    number(Number), !.
translate_expr(-Expr, VarMap, negation(Inner)) :-
    !,
    translate_expr(Expr, VarMap, Inner).
translate_expr(Expr, VarMap, op(Op, Left, Right)) :-
    compound(Expr),
    Expr =.. [PrologOp, LeftExpr, RightExpr],
    expr_op(PrologOp, Op),
    !,
    translate_expr(LeftExpr, VarMap, Left),
    translate_expr(RightExpr, VarMap, Right).
translate_expr(Expr, VarMap, call(Fn, TranslatedArgs)) :-
    compound(Expr),
    Expr =.. [Fn|Args],
    Args \= [],
    maplist(translate_expr_arg(VarMap), Args, TranslatedArgs).

translate_expr_arg(VarMap, Arg, Translated) :-
    translate_expr(Arg, VarMap, Translated).

%% literal_value(+PrologValue, -LiteralParts)
%  Classify a Prolog value for target-specific literal rendering.
%  Returns: null, bool(true/false), string(Text), number(N), atom(A), term(T).
literal_value(Value, null) :- var(Value), !.
literal_value(true, bool(true)) :- !.
literal_value(false, bool(false)) :- !.
literal_value(Value, string(Value)) :- string(Value), !.
literal_value(Value, number(Value)) :- number(Value), !.
literal_value(Value, atom(Value)) :- atom(Value), !.
literal_value(Value, term(Value)).

% ============================================================================
% CLAUSE STRUCTURE ANALYSIS
% ============================================================================

%% analyze_clauses(+Clauses, -Analysis)
%  Analyze a list of Head-Body clauses and return structured analysis.
%  Analysis is a list of clause_info(Head, HeadArgs, HeadConditions, Goals).
analyze_clauses([], []).
analyze_clauses([Head-Body|Rest], [clause_info(Head, HeadArgs, HeadConditions, Goals)|RestAnalysis]) :-
    Head =.. [_Pred|HeadArgs],
    head_conditions(HeadArgs, 1, HeadConditions),
    normalize_goals(Body, Goals),
    analyze_clauses(Rest, RestAnalysis).

%% clause_guard_output_split(+Goals, +VarMap, -Guards, -Outputs)
%  Split a list of goals into leading guards and trailing outputs.
%  Guards are comparison/type-check goals before any assignment.
%  Once an assignment is seen, remaining goals are all outputs.
clause_guard_output_split([], _VarMap, [], []).
clause_guard_output_split([Goal|Rest], VarMap, [Goal|Guards], Outputs) :-
    is_guard_goal(Goal, VarMap),
    !,
    clause_guard_output_split(Rest, VarMap, Guards, Outputs).
clause_guard_output_split(Outputs, _VarMap, [], Outputs).

% ============================================================================
% ADVANCED GOAL SEQUENCE ANALYSIS
% ============================================================================
%
% Inspired by TypR's native_typr_prefix_goals, but target-agnostic.
% Returns a classified list of goals rather than generating code.
%
% Classifications:
%   guard(Goal, Expr)           — boolean test, no new bindings
%   output(Goal, Var, Expr)     — binds a new variable
%   output_ite(If, Then, Else, SharedVars) — if-then-else that binds shared var(s)
%   output_disj(Alternatives, SharedVars)  — disjunction binding shared var(s)
%   output_if_then(If, Then, OutputVars)   — if-then binding var(s) (no else)
%   passthrough(Goal)           — unclassified goal (target must handle)

%% classify_goal_sequence(+Goals, +VarMap, -ClassifiedGoals)
%  Walk a list of goals and classify each one.
%  VarMap is updated as outputs are encountered.
classify_goal_sequence([], _VarMap, []).
classify_goal_sequence([Goal|Rest], VarMap, [Classified|RestClassified]) :-
    classify_single_goal(Goal, VarMap, Classified, VarMap1),
    classify_goal_sequence(Rest, VarMap1, RestClassified).

%% classify_single_goal(+Goal, +VarMap, -Classified, -VarMapOut)
classify_single_goal(Goal, VarMap, Classified, VarMapOut) :-
    %% Strip module qualification
    (Goal = _Module:InnerGoal -> G = InnerGoal ; G = Goal),
    classify_single_goal_(G, VarMap, Classified, VarMapOut).

classify_single_goal_(Goal, VarMap, output_ite(If, Then, Else, SharedVars), VarMapOut) :-
    if_then_else_goal(Goal, If, Then, Else),
    if_then_else_shared_output_vars(Then, Else, VarMap, SharedVars),
    SharedVars \= [],
    !,
    ensure_vars(SharedVars, VarMap, _, VarMapOut).

classify_single_goal_(Goal, VarMap, output_disj(Alternatives, SharedVars), VarMapOut) :-
    disjunction_alternatives(Goal, Alternatives),
    Alternatives = [_,_|_],  %% at least 2 alternatives
    shared_output_vars(Alternatives, VarMap, SharedVars),
    SharedVars \= [],
    !,
    ensure_vars(SharedVars, VarMap, _, VarMapOut).

classify_single_goal_(Goal, VarMap, output_if_then(If, Then, OutputVars), VarMapOut) :-
    if_then_goal(Goal, If, Then),
    if_then_output_vars(Then, VarMap, OutputVars),
    OutputVars \= [],
    !,
    ensure_vars(OutputVars, VarMap, _, VarMapOut).

classify_single_goal_(Goal, VarMap, guard(Goal, Expr), VarMap) :-
    is_guard_goal(Goal, VarMap),
    !,
    translate_expr(Goal, VarMap, Expr).

classify_single_goal_(Goal, VarMap, output(Goal, Var, Expr), VarMapOut) :-
    goal_output_var(Goal, Var),
    var(Var),
    output_var_allowed(VarMap, Var),
    !,
    ensure_var(VarMap, Var, _, VarMapOut),
    translate_output_expr(Goal, VarMap, Expr).

classify_single_goal_(Goal, VarMap, passthrough(Goal), VarMap).

%% translate_output_expr(+Goal, +VarMap, -Expr)
%  Extract the expression part of an output goal.
translate_output_expr(Var is ArithExpr, VarMap, Expr) :-
    var(Var), !,
    translate_expr(ArithExpr, VarMap, Expr).
translate_output_expr(Var = Value, VarMap, Expr) :-
    var(Var), !,
    translate_expr(Value, VarMap, Expr).
translate_output_expr(_ = Var, VarMap, Expr) :-
    var(Var), !,
    translate_expr(Var, VarMap, Expr).
translate_output_expr(Goal, VarMap, call(Fn, TranslatedArgs)) :-
    compound(Goal),
    Goal =.. [Fn|Args],
    maplist(translate_expr_arg(VarMap), Args, TranslatedArgs).
translate_output_expr(Goal, _, literal(Goal)).

% ============================================================================
% OUTPUT CONTROL FLOW DETECTION
% ============================================================================

%% is_output_control_flow(+Goal, +VarMap, -OutputInfo)
%  Check if a control flow goal (if-then-else, disjunction) produces output.
is_output_control_flow(Goal, VarMap, OutputInfo) :-
    if_then_else_goal(Goal, If, Then, Else),
    if_then_else_output_info(If, Then, Else, OutputInfo0),
    OutputInfo0 = ite_output(_, _, _, SharedVars),
    exclude_varmap_vars(VarMap, SharedVars, NewVars),
    NewVars \= [],
    OutputInfo = ite_output(If, Then, Else, NewVars).
is_output_control_flow(Goal, VarMap, OutputInfo) :-
    disjunction_alternatives(Goal, Alternatives),
    Alternatives = [_,_|_],
    disjunction_output_info(Alternatives, OutputInfo0),
    OutputInfo0 = disj_output(_, SharedVars),
    exclude_varmap_vars(VarMap, SharedVars, NewVars),
    NewVars \= [],
    OutputInfo = disj_output(Alternatives, NewVars).
is_output_control_flow(Goal, VarMap, OutputInfo) :-
    if_then_goal(Goal, If, Then),
    if_then_output_info(If, Then, OutputInfo0),
    OutputInfo0 = if_then_output(_, _, OutputVars),
    exclude_varmap_vars(VarMap, OutputVars, NewVars),
    NewVars \= [],
    OutputInfo = if_then_output(If, Then, NewVars).

%% if_then_else_output_info(+If, +Then, +Else, -OutputInfo)
if_then_else_output_info(If, Then, Else, ite_output(If, Then, Else, SharedVars)) :-
    alternative_output_vars(Then, ThenVars),
    intersect_output_vars(Else, ThenVars, SharedVars),
    SharedVars \= [].

%% disjunction_output_info(+Alternatives, -OutputInfo)
disjunction_output_info(Alternatives, disj_output(Alternatives, SharedVars)) :-
    Alternatives = [First|Rest],
    alternative_output_vars(First, FirstVars),
    foldl(intersect_output_vars, Rest, FirstVars, SharedVars),
    SharedVars \= [].

%% if_then_output_info(+If, +Then, -OutputInfo)
if_then_output_info(If, Then, if_then_output(If, Then, OutputVars)) :-
    alternative_output_vars(Then, OutputVars),
    OutputVars \= [].

% ============================================================================
% MULTI-CLAUSE COMPILATION HELPER
% ============================================================================

%% compile_multi_clause(+Clauses, +VarMap, +Strategy, -Analysis)
%  Analyze multiple clauses for compilation.
%  Strategy is: if_else_chain | match_arms | pattern_heads | switch_cases
%  Returns a list of clause_branch(HeadConditions, GuardExprs, ClassifiedGoals).
compile_multi_clause([], _VarMap, _Strategy, []).
compile_multi_clause([Head-Body|Rest], VarMap, Strategy, [Branch|RestBranches]) :-
    Head =.. [_Pred|HeadArgs],
    head_conditions(HeadArgs, 1, HeadConditions),
    build_head_varmap(HeadArgs, 1, ClauseVarMap),
    normalize_goals(Body, Goals),
    classify_goal_sequence(Goals, ClauseVarMap, ClassifiedGoals),
    Branch = clause_branch(HeadConditions, ClassifiedGoals, Strategy),
    compile_multi_clause(Rest, VarMap, Strategy, RestBranches).

% ============================================================================
% RECURSIVE COMPILE_EXPRESSION FRAMEWORK
% ============================================================================
%
% The shared entry point for recursive template-then-lower compilation.
% Targets register multifile hooks for rendering; the shared module
% provides the classification and dispatch logic.
%
% This enables templates and native lowering to compose at every
% nesting level — branch bodies, sub-expressions, and nested control
% flow all go through the same pipeline.

%% compile_expression(+Target, +Goal, +VarMap, -Code, -OutputVars, -VarMapOut)
%  The recursive entry point. Classifies a single goal and dispatches
%  to target-specific rendering via multifile hooks.
compile_expression(Target, Goal, VarMap, Code, OutputVars, VarMapOut) :-
    %% Strip module qualification
    (Goal = _Module:InnerGoal -> G = InnerGoal ; G = Goal),
    compile_expression_(Target, G, VarMap, Code, OutputVars, VarMapOut).

%% Try output_ite template
compile_expression_(Target, Goal, VarMap, Code, OutputVars, VarMapOut) :-
    if_then_else_goal(Goal, If, Then, Else),
    if_then_else_shared_output_vars(Then, Else, VarMap, SharedVars),
    SharedVars \= [],
    !,
    render_guard_condition(Target, If, VarMap, Cond),
    compile_branch(Target, Then, VarMap, ThenLines, _ThenVars, _),
    compile_branch(Target, Else, VarMap, ElseLines, _ElseVars, _),
    ensure_vars(SharedVars, VarMap, _, VarMapOut),
    maplist(ensure_var_name_from(VarMapOut), SharedVars, OutputVars),
    render_ite_block(Target, Cond, ThenLines, ElseLines, "    ", OutputVars, CodeLines),
    atomic_list_concat(CodeLines, '\n', Code).

%% Try output_disj template
compile_expression_(Target, Goal, VarMap, Code, OutputVars, VarMapOut) :-
    disjunction_alternatives(Goal, Alternatives),
    Alternatives = [_,_|_],
    shared_output_vars(Alternatives, VarMap, SharedVars),
    SharedVars \= [],
    !,
    ensure_vars(SharedVars, VarMap, _, VarMapOut),
    maplist(ensure_var_name_from(VarMapOut), SharedVars, OutputVars),
    compile_disj_branches(Target, Alternatives, VarMap, OutputVars, CodeLines),
    atomic_list_concat(CodeLines, '\n', Code).

%% Try output goal (assignment)
compile_expression_(Target, Goal, VarMap, Code, [VarName], VarMapOut) :-
    render_output_goal(Target, Goal, VarMap, Code, VarName, VarMapOut),
    !.

%% Try guard (produces condition, no code lines)
compile_expression_(Target, Goal, VarMap, Code, [], VarMap) :-
    is_guard_goal(Goal, VarMap),
    render_guard_condition(Target, Goal, VarMap, Code),
    !.

%% Passthrough: target-specific fallback
compile_expression_(Target, Goal, VarMap, Code, OutputVars, VarMapOut) :-
    render_output_goal(Target, Goal, VarMap, Code, VarName, VarMapOut),
    !,
    OutputVars = [VarName].
compile_expression_(_Target, _Goal, VarMap, "", [], VarMap).

%% compile_branch(+Target, +Branch, +VarMap, -Lines, -OutputVars, -VarMapOut)
%  Compile a branch body (Then or Else) through the full pipeline.
compile_branch(Target, Branch, VarMap, Lines, OutputVars, VarMapOut) :-
    normalize_goals(Branch, Goals),
    Goals \= [],
    classify_goal_sequence(Goals, VarMap, ClassifiedGoals),
    ClassifiedGoals \= [],
    !,
    compile_classified_sequence(Target, ClassifiedGoals, VarMap, Lines, VarMapOut),
    collect_classified_output_vars(ClassifiedGoals, VarMapOut, OutputVars).
compile_branch(Target, Branch, VarMap, [ExprStr], [ExprStr], VarMap) :-
    render_branch_value(Target, Branch, VarMap, ExprStr),
    !.
compile_branch(_Target, _Branch, VarMap, [], [], VarMap).

%% compile_classified_sequence(+Target, +ClassifiedGoals, +VarMap, -Lines, -VarMapOut)
%  Render a sequence of classified goals via target hooks.
compile_classified_sequence(_Target, [], VarMap, [], VarMap).
compile_classified_sequence(Target, [Classified|Rest], VarMap, Lines, VarMapOut) :-
    compile_classified_goal(Target, Classified, VarMap, MidLines, VarMap1),
    compile_classified_sequence(Target, Rest, VarMap1, RestLines, VarMapOut),
    append(MidLines, RestLines, Lines).

%% compile_classified_goal(+Target, +Classified, +VarMap, -Lines, -VarMapOut)
compile_classified_goal(_Target, guard(_, _), VarMap, [], VarMap) :- !.

compile_classified_goal(Target, output(Goal, _Var, _Expr), VarMap0, [Line], VarMapOut) :-
    !,
    render_output_goal(Target, Goal, VarMap0, Line, _, VarMapOut).

compile_classified_goal(Target, output_ite(If, Then, Else, SharedVars), VarMap0, Lines, VarMapOut) :-
    !,
    render_guard_condition(Target, If, VarMap0, Cond),
    compile_branch(Target, Then, VarMap0, ThenLines, _ThenVars, _),
    compile_branch(Target, Else, VarMap0, ElseLines, _ElseVars, _),
    ensure_vars(SharedVars, VarMap0, _, VarMapOut),
    maplist(ensure_var_name_from(VarMapOut), SharedVars, VarNames),
    render_ite_block(Target, Cond, ThenLines, ElseLines, "    ", VarNames, Lines).

compile_classified_goal(Target, passthrough(Goal), VarMap0, Lines, VarMapOut) :-
    !,
    (   render_output_goal(Target, Goal, VarMap0, Line, _, VarMapOut)
    ->  Lines = [Line]
    ;   Lines = [], VarMapOut = VarMap0
    ).

compile_classified_goal(_Target, _, VarMap, [], VarMap).

%% collect_classified_output_vars(+ClassifiedGoals, +VarMap, -OutputVarNames)
collect_classified_output_vars([], _, []).
collect_classified_output_vars([output(Goal, _, _)|Rest], VarMap, [Name|Names]) :-
    goal_output_var(Goal, Var),
    lookup_var(Var, VarMap, Name),
    !,
    collect_classified_output_vars(Rest, VarMap, Names).
collect_classified_output_vars([output_ite(_, _, _, SharedVars)|Rest], VarMap, Names) :-
    !,
    maplist(ensure_var_name_from(VarMap), SharedVars, SVNames),
    collect_classified_output_vars(Rest, VarMap, RestNames),
    append(SVNames, RestNames, Names).
collect_classified_output_vars([_|Rest], VarMap, Names) :-
    collect_classified_output_vars(Rest, VarMap, Names).

%% compile_disj_branches(+Target, +Alternatives, +VarMap, +OutputVars, -Lines)
%  Compile disjunction alternatives into target-specific if/elif/else.
compile_disj_branches(_Target, [], _VarMap, _OutputVars, []).
compile_disj_branches(Target, [Alt], VarMap, _OutputVars, Lines) :-
    !,
    compile_branch(Target, Alt, VarMap, BranchLines, _BranchVars, _),
    Lines = BranchLines.
compile_disj_branches(Target, [Alt|Rest], VarMap, OutputVars, Lines) :-
    normalize_goals(Alt, Goals),
    clause_guard_output_split(Goals, VarMap, Guards, _Outputs),
    (   Guards \= []
    ->  maplist(render_guard_condition(Target), Guards, [VarMap], CondStrs),
        atomic_list_concat(CondStrs, ' and ', CondExpr)
    ;   CondExpr = "True"
    ),
    compile_branch(Target, Alt, VarMap, BranchLines, _BranchVars, _),
    render_ite_block(Target, CondExpr, BranchLines, [], "    ", OutputVars, AltLines),
    compile_disj_branches(Target, Rest, VarMap, OutputVars, RestLines),
    append(AltLines, RestLines, Lines).

%% ensure_var_name_from(+VarMap, +Var, -Name)
ensure_var_name_from(VarMap, Var, Name) :-
    lookup_var(Var, VarMap, Name), !.
ensure_var_name_from(_, _, "_").

% ============================================================================
% PARALLELISM ANALYSIS
% ============================================================================

%% classify_parallelism(+PredIndicator, +Clauses, -Strategy)
%  Determine the parallelism strategy for a predicate.
classify_parallelism(_PredIndicator, [Head-Body], goal_parallel(Head, ParallelGoals, ResultGoals)) :-
    % Only single-clause predicates for goal parallelism for now
    normalize_goals(Body, Goals),
    term_variables(Head, HeadVars),
    partition_parallel_goals(Goals, HeadVars, [], ParallelGoals, ResultGoals),
    ParallelGoals = [_,_|_],
    goals_are_independent(ParallelGoals),
    !.
classify_parallelism(PredIndicator, _Clauses, clause_parallel) :-
    is_order_independent(PredIndicator, _),
    !.
classify_parallelism(_, _, sequential).

%% partition_parallel_goals(+Goals, +HeadVars, +SeenVars, -Parallel, -Result)
%  Collect goals into Parallel list until a data dependency or impurity is found.
%  Invariant: A goal G is independent if its introduced variables (IntroVars = GVars - HeadVars)
%  do not intersect with any variables seen in previous goals (SeenVars). 
%  This catches both "output-after-input" and "input-after-output" dependencies because 
%  SeenVars accumulates all variables from previous parallel goals.
partition_parallel_goals([], _, _, [], []).
partition_parallel_goals([G|Rest], HeadVars, SeenVars, Parallel, Result) :-
    is_pure_goal(G),
    term_variables(G, GVars),
    % Variables introduced in this goal list (excluding head inputs)
    exclude_vars_by_identity(GVars, HeadVars, IntroVars),
    % Check for dependencies: IntroVars must not intersect with SeenVars
    (   intersection_by_identity(IntroVars, SeenVars, [])
    ->  Parallel = [G|RestParallel],
        append(IntroVars, SeenVars, NewSeenVars),
        partition_parallel_goals(Rest, HeadVars, NewSeenVars, RestParallel, Result)
    ;   Parallel = [],
        Result = [G|Rest]
    ).

%% is_order_independent(+PredIndicator, -Reason)
%  True if the predicate's clauses can be evaluated in any order.
%  Thin wrapper over purity_certificate:analyze_predicate_purity/2.
%  Preserves the legacy return shape: `declared` for user-annotated
%  predicates, `proven([pure_goals])` for statically-verified ones.
is_order_independent(PredIndicator, Reason) :-
    purity_certificate:analyze_predicate_purity(PredIndicator, Cert),
    Cert = purity_cert(pure, Proof, _, _),
    ( Proof = declared
    -> Reason = declared
    ;  Reason = proven([pure_goals])
    ).

%% goals_are_independent(+Goals)
%  True if a list of goals can be executed in parallel.
goals_are_independent(Goals) :-
    forall(member(G, Goals), is_pure_goal(G)),
    goals_have_disjoint_bindings(Goals).

%% is_pure_goal(+Goal)
%  True if the goal has no side effects.
%  Thin wrapper over purity_certificate:analyze_goal_purity/2. The
%  impurity catalogue now lives in purity_certificate:impurity_class/2
%  so both modules share a single source of truth.
is_pure_goal(Goal) :-
    purity_certificate:analyze_goal_purity(Goal,
                                           purity_cert(pure, _, _, _)).

%% goals_have_disjoint_bindings(+Goals)
%  Conservative check for disjoint bindings.
%  We allow shared inputs, but outputs must be disjoint from all other goals' variables.
goals_have_disjoint_bindings(Goals) :-
    maplist(get_goal_io_vars, Goals, GoalIOVars),
    all_goal_outputs_disjoint(GoalIOVars).

%% get_goal_io_vars(+Goal, -IOVars)
get_goal_io_vars(Goal, io(Inputs, Outputs)) :-
    term_variables(Goal, AllVars),
    (   goal_output_vars(Goal, Outputs) -> true ; Outputs = [] ),
    exclude_vars_by_identity(AllVars, Outputs, Inputs).

%% all_goal_outputs_disjoint(+GoalIOVars)
all_goal_outputs_disjoint([]).
all_goal_outputs_disjoint([io(In1, Out1)|Rest]) :-
    forall(member(io(In2, Out2), Rest), (
        intersection_by_identity(Out1, In2, []),
        intersection_by_identity(Out1, Out2, []),
        intersection_by_identity(Out2, In1, [])
    )),
    all_goal_outputs_disjoint(Rest).

%% exclude_vars_by_identity(+AllVars, +ToExclude, -Result)
exclude_vars_by_identity([], _, []).
exclude_vars_by_identity([X|Rest], List, Result) :-
    (   var_member_by_identity(X, List)
    ->  Result = RestResult
    ;   Result = [X|RestResult]
    ),
    exclude_vars_by_identity(Rest, List, RestResult).

%% intersection_by_identity(+List1, +List2, -Intersection)
intersection_by_identity([], _, []).
intersection_by_identity([X|Rest], List, Result) :-
    (   var_member_by_identity(X, List)
    ->  Result = [X|RestResult]
    ;   Result = RestResult
    ),
    intersection_by_identity(Rest, List, RestResult).
