/**
 * Fuzzy Logic DSL - Convenience Operators
 *
 * Optional module providing syntactic sugar for fuzzy expressions:
 *
 * - Term:Weight     -> w(Term, Weight)
 * - A & B           -> f_and([A, B])
 * - A \/ B          -> f_or([A, B])
 * - A |/ B          -> f_dist_or (requires base score context)
 * - ~A              -> f_not(A)
 *
 * Import this module to enable operator syntax:
 *   :- use_module(fuzzy/operators).
 *
 * Without this module, use the verbose core forms to avoid
 * potential conflicts with Prolog's : (module qualification).
 */

:- module(fuzzy_operators, [
    % Expansion predicates
    expand_fuzzy/2,
    expand_weighted/2,
    expand_weighted_list/2,

    % Collect operators into list
    collect_and/2,
    collect_or/2,

    % Convenience macros
    fuzzy_and/2,
    fuzzy_or/2
]).

:- use_module(core).

% =============================================================================
% Operator Definitions
% =============================================================================

% Operators are defined via op/3 directives.
% The precedences are chosen to allow natural expression building:
%   bash:0.9 & shell:0.5 \/ scripting:0.3
% parses as expected.

:- op(400, xfy, &).      % Fuzzy AND
:- op(400, xfy, v).      % Fuzzy OR (v for disjunction, avoids | conflict)
:- op(200, fy, ~).       % Fuzzy NOT

% Note: We don't redefine : operator as it conflicts with module qualification.
% Use w(Term, Weight) or the colon in list context: [bash:0.9] works fine.
% We use 'v' for OR instead of \/ or | which conflict with Prolog builtins.

% =============================================================================
% Term Expansion
% =============================================================================

%% expand_weighted(+TermOrWeighted, -Expanded)
%  Expand a term to w(Term, Weight) form.
expand_weighted(Term:Weight, w(Term, Weight)) :- !.
expand_weighted(w(Term, Weight), w(Term, Weight)) :- !.
expand_weighted(Term, w(Term, 1.0)) :-
    atom(Term), !.
expand_weighted(Term, Term).  % Pass through complex terms

%% expand_weighted_list(+List, -Expanded)
%  Expand a list of terms to w/2 form.
expand_weighted_list([], []).
expand_weighted_list([H|T], [HExp|TExp]) :-
    expand_weighted(H, HExp),
    expand_weighted_list(T, TExp).

%% collect_and(+Expr, -List)
%  Collect terms from A & B & C into a flat list.
collect_and(A & B, List) :- !,
    collect_and(A, ListA),
    collect_and(B, ListB),
    append(ListA, ListB, List).
collect_and(Term, [Expanded]) :-
    expand_weighted(Term, Expanded).

%% collect_or(+Expr, -List)
%  Collect terms from A v B v C into a flat list.
collect_or(A v B, List) :- !,
    collect_or(A, ListA),
    collect_or(B, ListB),
    append(ListA, ListB, List).
collect_or(Term, [Expanded]) :-
    expand_weighted(Term, Expanded).

%% expand_fuzzy(+Expr, -Expanded)
%  Expand operator syntax to functor form.

% Fuzzy AND: A & B -> f_and([A, B, ...])
expand_fuzzy(Expr, f_and(Terms)) :-
    Expr = (_ & _), !,
    collect_and(Expr, Terms).

% Fuzzy OR: A v B -> f_or([A, B, ...])
expand_fuzzy(Expr, f_or(Terms)) :-
    Expr = (_ v _), !,
    collect_or(Expr, Terms).

% Fuzzy NOT: ~A -> f_not(A)
expand_fuzzy(~A, f_not(AExp)) :- !,
    expand_fuzzy(A, AExp).

% Weighted term
expand_fuzzy(Term:Weight, w(Term, Weight)) :- !.

% Already expanded or plain term
expand_fuzzy(f_and(T), f_and(TExp)) :- !,
    expand_weighted_list(T, TExp).
expand_fuzzy(f_or(T), f_or(TExp)) :- !,
    expand_weighted_list(T, TExp).
expand_fuzzy(f_dist_or(B, T), f_dist_or(B, TExp)) :- !,
    expand_weighted_list(T, TExp).
expand_fuzzy(f_union(B, T), f_union(B, TExp)) :- !,
    expand_weighted_list(T, TExp).
expand_fuzzy(f_not(E), f_not(EExp)) :- !,
    expand_fuzzy(E, EExp).
expand_fuzzy(w(T, W), w(T, W)) :- !.
expand_fuzzy(Term, w(Term, 1.0)) :-
    atom(Term), !.
expand_fuzzy(Term, Term).

% =============================================================================
% Goal Expansion (compile-time transformation)
% =============================================================================

%% goal_expansion for fuzzy functors with operator syntax in lists
user:goal_expansion(f_and(List, Result), f_and(Expanded, Result)) :-
    is_list(List),
    expand_weighted_list(List, Expanded),
    List \== Expanded.

user:goal_expansion(f_or(List, Result), f_or(Expanded, Result)) :-
    is_list(List),
    expand_weighted_list(List, Expanded),
    List \== Expanded.

user:goal_expansion(f_dist_or(Base, List, Result), f_dist_or(Base, Expanded, Result)) :-
    is_list(List),
    expand_weighted_list(List, Expanded),
    List \== Expanded.

user:goal_expansion(f_union(Base, List, Result), f_union(Base, Expanded, Result)) :-
    is_list(List),
    expand_weighted_list(List, Expanded),
    List \== Expanded.

% =============================================================================
% Term Expansion (for top-level expressions)
% =============================================================================

%% term_expansion for fuzzy expressions
user:term_expansion(fuzzy_expr(Expr), fuzzy_expr(Expanded)) :-
    expand_fuzzy(Expr, Expanded),
    Expr \== Expanded.

% =============================================================================
% Convenience macros
% =============================================================================

%% fuzzy_and(+Expr, -Result)
%  Evaluate an AND expression with operator syntax.
%  Example: fuzzy_and(bash:0.9 & shell:0.5, R)
fuzzy_and(Expr, Result) :-
    expand_fuzzy(Expr, f_and(Terms)),
    f_and(Terms, Result).

%% fuzzy_or(+Expr, -Result)
%  Evaluate an OR expression with operator syntax.
%  Example: fuzzy_or(bash:0.9 \/ shell:0.5, R)
fuzzy_or(Expr, Result) :-
    expand_fuzzy(Expr, f_or(Terms)),
    f_or(Terms, Result).
