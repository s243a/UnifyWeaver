/**
 * Fuzzy Logic DSL - Evaluation Engine
 *
 * Provides the main evaluation interface for fuzzy expressions:
 *
 * - eval_fuzzy_expr/3: Evaluate expression with term scores
 * - eval_fuzzy_query/4: Evaluate query over items
 * - score_items/3: Score a list of items with an expression
 *
 * This module ties together core operations, boolean filters,
 * and predicates for complete expression evaluation.
 */

:- module(fuzzy_eval, [
    % Main evaluation
    eval_fuzzy_expr/3,
    eval_fuzzy_query/4,

    % Batch scoring
    score_items/3,
    score_item/3,

    % Context management
    with_term_scores/2,
    with_item_context/2,

    % Score combination
    multiply_scores/3,
    blend_scores/4
]).

:- use_module(core).
:- use_module(boolean).
:- use_module(predicates).

% =============================================================================
% Main Evaluation Interface
% =============================================================================

%% eval_fuzzy_expr(+Expr, +TermScores, -Result)
%  Evaluate a fuzzy expression with given term scores.
%  TermScores is a list of Term-Score pairs or a scoring predicate.
%
%  Example:
%    eval_fuzzy_expr(
%        f_and([w(bash, 0.9), w(shell, 0.5)]),
%        [bash-0.8, shell-0.6],
%        Result
%    )
eval_fuzzy_expr(Expr, TermScores, Result) :-
    with_term_scores(TermScores, eval_expr(Expr, Result)).

%% eval_expr(+Expr, -Result)
%  Internal expression evaluator (assumes term_score/2 is set up).
eval_expr(f_and(Terms), Result) :-
    f_and(Terms, Result).
eval_expr(f_or(Terms), Result) :-
    f_or(Terms, Result).
eval_expr(f_dist_or(Base, Terms), Result) :-
    f_dist_or(Base, Terms, Result).
eval_expr(f_union(Base, Terms), Result) :-
    f_union(Base, Terms, Result).
eval_expr(f_not(Inner), Result) :-
    eval_expr(Inner, InnerResult),
    Result is 1 - InnerResult.
eval_expr(w(Term, Weight), Result) :-
    term_score(Term, Score),
    Result is Weight * Score.
eval_expr(N, N) :-
    number(N).

% Boolean expressions
eval_expr(b_and(Conds), Result) :-
    b_and(Conds, Result).
eval_expr(b_or(Conds), Result) :-
    b_or(Conds, Result).
eval_expr(b_not(Cond), Result) :-
    b_not(Cond, Result).

%% eval_fuzzy_query(+Expr, +Items, +ScoreFn, -ScoredItems)
%  Evaluate a fuzzy expression over a list of items.
%  ScoreFn is a predicate that computes term scores for an item.
%  Returns list of Item-Score pairs, sorted by score descending.
%
%  Example:
%    eval_fuzzy_query(
%        f_and([w(bash, 0.9)]),
%        Items,
%        compute_item_scores,
%        ScoredItems
%    )
eval_fuzzy_query(Expr, Items, ScoreFn, ScoredItems) :-
    findall(
        Item-Score,
        (   member(Item, Items),
            call(ScoreFn, Item, TermScores),
            eval_fuzzy_expr(Expr, TermScores, Score)
        ),
        UnsortedScores
    ),
    sort(2, @>=, UnsortedScores, ScoredItems).

% =============================================================================
% Batch Scoring
% =============================================================================

%% score_items(+Expr, +Items, -ScoredItems)
%  Score items using expression with default term scoring.
%  Items should have embeddings or term scores accessible.
score_items(Expr, Items, ScoredItems) :-
    findall(
        Item-Score,
        (   member(Item, Items),
            score_item(Expr, Item, Score)
        ),
        UnsortedScores
    ),
    sort(2, @>=, UnsortedScores, ScoredItems).

%% score_item(+Expr, +Item, -Score)
%  Score a single item with an expression.
score_item(Expr, Item, Score) :-
    with_item_context(Item, eval_expr(Expr, Score)).

% =============================================================================
% Context Management
% =============================================================================

%% with_term_scores(+TermScores, +Goal)
%  Execute Goal with term_score/2 bound to TermScores.
%  TermScores can be:
%  - List of Term-Score pairs: [bash-0.8, shell-0.6]
%  - Scoring predicate: my_scorer (called as my_scorer(Term, Score))

with_term_scores([], Goal) :- !,
    call(Goal).

with_term_scores([Term-Score|Rest], Goal) :- !,
    setup_call_cleanup(
        assertz(fuzzy_core:term_score(Term, Score), Ref),
        with_term_scores(Rest, Goal),
        erase(Ref)
    ).

with_term_scores(Pred, Goal) :-
    callable(Pred),
    setup_call_cleanup(
        assertz((fuzzy_core:term_score(T, S) :- call(Pred, T, S)), Ref),
        call(Goal),
        erase(Ref)
    ).

%% with_item_context(+Item, +Goal)
%  Execute Goal with current item set for predicate evaluation.
with_item_context(Item, Goal) :-
    setup_call_cleanup(
        set_current_item(Item),
        call(Goal),
        retractall(current_item_data(_))
    ).

% =============================================================================
% Score Combination
% =============================================================================

%% multiply_scores(+Scores1, +Scores2, -Result)
%  Element-wise multiplication of score lists.
%  Scores are lists of Item-Score pairs.
multiply_scores([], [], []).
multiply_scores([Item-S1|Rest1], [Item-S2|Rest2], [Item-S|RestR]) :-
    S is S1 * S2,
    multiply_scores(Rest1, Rest2, RestR).

%% blend_scores(+Alpha, +Scores1, +Scores2, -Result)
%  Blend two score lists: Alpha*Scores1 + (1-Alpha)*Scores2
blend_scores(Alpha, Scores1, Scores2, Result) :-
    Beta is 1 - Alpha,
    findall(
        Item-S,
        (   member(Item-S1, Scores1),
            member(Item-S2, Scores2),
            S is Alpha * S1 + Beta * S2
        ),
        Result
    ).

% =============================================================================
% Top-K Selection
% =============================================================================

%% top_k(+ScoredItems, +K, -TopK)
%  Get top K items by score.
top_k(ScoredItems, K, TopK) :-
    length(TopK, K),
    append(TopK, _, ScoredItems), !.
top_k(ScoredItems, _, ScoredItems).  % Fewer than K items

% =============================================================================
% Pipeline Helpers
% =============================================================================

%% apply_filter(+ScoredItems, +Filter, -Filtered)
%  Apply a boolean filter to scored items.
%  Items with filter score 0.0 are removed.
apply_filter(ScoredItems, Filter, Filtered) :-
    findall(
        Item-Score,
        (   member(Item-Score, ScoredItems),
            with_item_context(Item, eval_condition(Filter, FilterScore)),
            FilterScore > 0.0
        ),
        Filtered
    ).

%% apply_boost(+ScoredItems, +Expr, -Boosted)
%  Apply a fuzzy boost expression to scored items.
%  Multiplies existing scores by expression result.
apply_boost(ScoredItems, Expr, Boosted) :-
    findall(
        Item-NewScore,
        (   member(Item-OldScore, ScoredItems),
            with_item_context(Item, eval_expr(Expr, BoostScore)),
            NewScore is OldScore * BoostScore
        ),
        Boosted
    ).
