/**
 * Fuzzy Logic DSL - Boolean Metadata Operations
 *
 * Provides crisp boolean operations for metadata filtering:
 * - b_and: Boolean AND (all conditions must match)
 * - b_or: Boolean OR (any condition matches)
 * - b_not: Boolean NOT (negation)
 *
 * These return 1.0 (true) or 0.0 (false), making them compatible
 * with fuzzy operations via multiplication.
 *
 * Note: Named b_and, b_or, b_not to avoid conflict with Prolog builtins.
 */

:- module(fuzzy_boolean, [
    % Boolean operations
    b_and/1,
    b_or/1,
    b_not/1,

    % Evaluation forms
    b_and/2,
    b_or/2,
    b_not/2,

    % Condition evaluation
    eval_condition/2
]).

:- use_module(predicates).

% =============================================================================
% Symbolic Forms
% =============================================================================

%% b_and(+Conditions)
%  Symbolic boolean AND. All conditions must be satisfied.
b_and(Conditions) :-
    is_list(Conditions).

%% b_or(+Conditions)
%  Symbolic boolean OR. At least one condition must be satisfied.
b_or(Conditions) :-
    is_list(Conditions).

%% b_not(+Condition)
%  Symbolic boolean NOT.
b_not(_Condition).

% =============================================================================
% Evaluation Forms
% =============================================================================

%% b_and(+Conditions, -Result)
%  Evaluate boolean AND: 1.0 if all conditions true, 0.0 otherwise.
b_and([], 1.0).
b_and([Cond|Rest], Result) :-
    eval_condition(Cond, Score),
    (   Score > 0.0
    ->  b_and(Rest, Result)
    ;   Result = 0.0
    ).

%% b_or(+Conditions, -Result)
%  Evaluate boolean OR: 1.0 if any condition true, 0.0 otherwise.
b_or([], 0.0).
b_or([Cond|Rest], Result) :-
    eval_condition(Cond, Score),
    (   Score > 0.0
    ->  Result = 1.0
    ;   b_or(Rest, Result)
    ).

%% b_not(+Condition, -Result)
%  Evaluate boolean NOT: 1.0 if condition false, 0.0 if true.
b_not(Cond, Result) :-
    eval_condition(Cond, Score),
    (   Score > 0.0
    ->  Result = 0.0
    ;   Result = 1.0
    ).

% =============================================================================
% Condition Evaluation
% =============================================================================

%% eval_condition(+Condition, -Score)
%  Evaluate a condition and return its score (0.0 or 1.0 for boolean,
%  or [0.0, 1.0] for fuzzy conditions).

% Nested boolean operations
eval_condition(b_and(Conds), Score) :-
    b_and(Conds, Score).
eval_condition(b_or(Conds), Score) :-
    b_or(Conds, Score).
eval_condition(b_not(Cond), Score) :-
    b_not(Cond, Score).

% Metadata predicates (delegate to predicates module)
eval_condition(is_type(Type), Score) :-
    (   is_type(Type) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(has_account(Account), Score) :-
    (   has_account(Account) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(has_parent(Parent), Score) :-
    (   has_parent(Parent) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(in_subtree(Subtree), Score) :-
    (   in_subtree(Subtree) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(has_tag(Tag), Score) :-
    (   has_tag(Tag) -> Score = 1.0 ; Score = 0.0 ).

% Hierarchical filters
eval_condition(child_of(Node), Score) :-
    (   child_of(Node) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(descendant_of(Node), Score) :-
    (   descendant_of(Node) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(parent_of(Node), Score) :-
    (   parent_of(Node) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(ancestor_of(Node), Score) :-
    (   ancestor_of(Node) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(sibling_of(Node), Score) :-
    (   sibling_of(Node) -> Score = 1.0 ; Score = 0.0 ).
eval_condition(has_depth(N), Score) :-
    (   has_depth(N) -> Score = 1.0 ; Score = 0.0 ).

% Fuzzy distance (returns score in [0,1])
eval_condition(near(Node, Decay), Score) :-
    near(Node, Decay, Score).

% Custom filter (arity 1, returns boolean)
eval_condition(Filter, Score) :-
    callable(Filter),
    (   call(Filter) -> Score = 1.0 ; Score = 0.0 ).
