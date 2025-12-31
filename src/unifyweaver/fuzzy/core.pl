/**
 * Fuzzy Logic DSL - Core Operations
 *
 * Provides fuzzy logic functors for semantic search and scoring:
 * - f_and: Fuzzy AND (product t-norm)
 * - f_or: Fuzzy OR (probabilistic sum)
 * - f_dist_or: Distributed OR (base score into each term)
 * - f_union: Non-distributed OR (base * OR result)
 * - f_not: Fuzzy NOT (complement)
 *
 * Each functor has two forms:
 * - Symbolic (no Result): for building expressions
 * - Evaluation (with Result): for computing scores
 */

:- module(fuzzy_core, [
    % Symbolic forms (for expression building)
    f_and/1,
    f_or/1,
    f_dist_or/2,
    f_union/2,
    f_not/1,

    % Evaluation forms (for computing)
    f_and/2,
    f_or/2,
    f_dist_or/3,
    f_union/3,
    f_not/2,

    % Evaluation predicate
    eval_fuzzy/3,

    % Term scoring
    get_term_score/2,
    term_score/2,

    % Helpers
    expand_term/2,
    expand_terms/2
]).

%% w(+Term, +Weight)
%  Weighted term constructor. Weight should be in [0.0, 1.0].
%  w(bash, 0.9) means "term 'bash' with weight 0.9"

%% expand_term(+TermOrWeighted, -Expanded)
%  Expand a term to w(Term, Weight) form.
%  Bare atoms get weight 1.0.
expand_term(w(Term, Weight), w(Term, Weight)) :- !.
expand_term(Term:Weight, w(Term, Weight)) :- !.  % Colon syntax support
expand_term(Term, w(Term, 1.0)) :- atom(Term), !.
expand_term(Term, Term).  % Pass through other structures

%% expand_terms(+Terms, -Expanded)
%  Expand a list of terms to w/2 form.
expand_terms([], []).
expand_terms([H|T], [HExp|TExp]) :-
    expand_term(H, HExp),
    expand_terms(T, TExp).

% =============================================================================
% Symbolic Forms (for expression building, used by operators)
% =============================================================================

%% f_and(+Terms)
%  Symbolic fuzzy AND. Terms is a list of w(Term, Weight) or bare atoms.
f_and(Terms) :-
    is_list(Terms),
    expand_terms(Terms, _).  % Validate structure

%% f_or(+Terms)
%  Symbolic fuzzy OR.
f_or(Terms) :-
    is_list(Terms),
    expand_terms(Terms, _).

%% f_dist_or(+BaseScore, +Terms)
%  Symbolic distributed OR. BaseScore distributed into each term before OR.
f_dist_or(BaseScore, Terms) :-
    number(BaseScore),
    is_list(Terms),
    expand_terms(Terms, _).

%% f_union(+BaseScore, +Terms)
%  Symbolic union (non-distributed OR). BaseScore * f_or(Terms).
f_union(BaseScore, Terms) :-
    number(BaseScore),
    is_list(Terms),
    expand_terms(Terms, _).

%% f_not(+Expr)
%  Symbolic fuzzy NOT.
f_not(_Expr).

% =============================================================================
% Evaluation Forms (for computing actual scores)
% =============================================================================

%% f_and(+Terms, -Result)
%  Evaluate fuzzy AND: product of weighted term scores.
%  Formula: w1*t1 * w2*t2 * ...
f_and(Terms, Result) :-
    expand_terms(Terms, Expanded),
    f_and_eval(Expanded, 1.0, Result).

f_and_eval([], Acc, Acc).
f_and_eval([w(Term, Weight)|Rest], Acc, Result) :-
    get_term_score(Term, Score),
    NewAcc is Acc * Weight * Score,
    f_and_eval(Rest, NewAcc, Result).

%% f_or(+Terms, -Result)
%  Evaluate fuzzy OR: probabilistic sum.
%  Formula: 1 - (1-w1*t1) * (1-w2*t2) * ...
f_or(Terms, Result) :-
    expand_terms(Terms, Expanded),
    f_or_eval(Expanded, 1.0, Complement),
    Result is 1 - Complement.

f_or_eval([], Acc, Acc).
f_or_eval([w(Term, Weight)|Rest], Acc, Result) :-
    get_term_score(Term, Score),
    NewAcc is Acc * (1 - Weight * Score),
    f_or_eval(Rest, NewAcc, Result).

%% f_dist_or(+BaseScore, +Terms, -Result)
%  Evaluate distributed OR: base score distributed into each term.
%  Formula: 1 - (1-Base*w1*t1) * (1-Base*w2*t2) * ...
%  Note: f_dist_or(1.0, Terms, R) is equivalent to f_or(Terms, R)
f_dist_or(BaseScore, Terms, Result) :-
    expand_terms(Terms, Expanded),
    f_dist_or_eval(BaseScore, Expanded, 1.0, Complement),
    Result is 1 - Complement.

f_dist_or_eval(_, [], Acc, Acc).
f_dist_or_eval(Base, [w(Term, Weight)|Rest], Acc, Result) :-
    get_term_score(Term, Score),
    NewAcc is Acc * (1 - Base * Weight * Score),
    f_dist_or_eval(Base, Rest, NewAcc, Result).

%% f_union(+BaseScore, +Terms, -Result)
%  Evaluate union (non-distributed OR): base * OR result.
%  Formula: Base * (1 - (1-w1*t1)(1-w2*t2)...)
%  Differs from f_dist_or in interaction term: Sab vs S²ab
f_union(BaseScore, Terms, Result) :-
    f_or(Terms, OrResult),
    Result is BaseScore * OrResult.

%% f_not(+Score, -Result)
%  Evaluate fuzzy NOT: complement.
%  Formula: 1 - Score
f_not(Score, Result) :-
    Result is 1 - Score.

% =============================================================================
% Expression Evaluation
% =============================================================================

%% eval_fuzzy(+Expr, +Context, -Result)
%  Evaluate a symbolic fuzzy expression in a given context.
%  Context provides bindings for term scores.

eval_fuzzy(f_and(Terms), Ctx, Result) :-
    with_context(Ctx, f_and(Terms, Result)).

eval_fuzzy(f_or(Terms), Ctx, Result) :-
    with_context(Ctx, f_or(Terms, Result)).

eval_fuzzy(f_dist_or(Base, Terms), Ctx, Result) :-
    with_context(Ctx, f_dist_or(Base, Terms, Result)).

eval_fuzzy(f_union(Base, Terms), Ctx, Result) :-
    with_context(Ctx, f_union(Base, Terms, Result)).

eval_fuzzy(f_not(Expr), Ctx, Result) :-
    eval_fuzzy(Expr, Ctx, InnerResult),
    Result is 1 - InnerResult.

% Evaluate nested expressions
eval_fuzzy(w(Term, Weight), Ctx, Result) :-
    with_context(Ctx, get_term_score(Term, Score)),
    Result is Weight * Score.

% Pass through numbers
eval_fuzzy(N, _, N) :- number(N).

% =============================================================================
% Context and Term Scoring (to be extended)
% =============================================================================

%% term_score(+Term, -Score)
%  Get the score for a term. This is a hook to be defined by the user
%  or overridden with context-specific scoring.
%  Scores are asserted dynamically via with_context/2.
:- dynamic term_score/2.

%% get_term_score(+Term, -Score)
%  Get score for a term with fallback to 0.5 (neutral score).
%  Uses if-then-else to ensure asserted clauses take precedence.
get_term_score(Term, Score) :-
    (   term_score(Term, S)
    ->  Score = S
    ;   Score = 0.5
    ).

%% with_context(+Context, +Goal)
%  Execute Goal with Context providing term_score bindings.
%  Context is a list of term-score pairs or a scoring predicate.
with_context([], Goal) :-
    call(Goal).
with_context([Term-Score|Rest], Goal) :-
    asserta(term_score(Term, Score), Ref),
    with_context(Rest, Goal),
    erase(Ref).
with_context(Pred, Goal) :-
    callable(Pred),
    \+ is_list(Pred),
    % Use Pred as the scoring function
    asserta((term_score(T, S) :- call(Pred, T, S)), Ref),
    call(Goal),
    erase(Ref).
