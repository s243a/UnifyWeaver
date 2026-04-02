% Category Influence Propagation
%
% Computes how much "influence" each top-level category has over articles
% in the knowledge graph, using the spectral distance weighting from
% the effective distance benchmark.
%
% influence(Root, n) = Sum over (article, path) of hops^(-n)
%
% The dimensionality parameter n controls how much short paths dominate:
%   n=1: harmonic weighting (long paths still contribute)
%   n=2: moderate weighting
%   n=5: spectral weighting (matches d_eff — short paths dominate)
%
% This exercises grouped transitive closure — each root category
% independently accumulates influence through the hierarchy.
%
% Usage:
%   ?- category_influence(Root, Score).
%   Root = 'Nature', Score = 42.5 ;
%   Root = 'Society', Score = 31.2 ;
%   ...

:- discontiguous category_parent/2.
:- discontiguous article_category/2.
:- discontiguous root_category/1.

max_depth(10).

% Dimensionality parameter for spectral weighting
% n=5 matches effective distance (short paths dominate)
influence_dimension(5).

% category_ancestor(Source, Ancestor, Hops)
% Counted transitive closure written in the direct recursive form the
% current native lowerings compile across targets.
category_ancestor(Cat, Parent, 1) :-
    category_parent(Cat, Parent).
category_ancestor(Cat, Ancestor, Hops) :-
    max_depth(MaxD),
    category_parent(Cat, Mid),
    category_ancestor(Mid, Ancestor, H1),
    Hops is H1 + 1,
    Hops =< MaxD.

% root_category_set/1 — collect all root categories (parents with no parents)
root_category_set(Roots) :-
    findall(R, root_category(R), Roots).

% article_root_weight(Article, Root, Weight)
% For each article, compute spectral weight hops^(-n) for each path to root
article_root_weight(Article, Root, Weight) :-
    article_category(Article, Cat),
    root_category(Root),
    Cat = Root,
    Weight is 1.0.    % direct membership = distance 1, weight 1^(-n) = 1
article_root_weight(Article, Root, Weight) :-
    influence_dimension(N),
    article_category(Article, Cat),
    root_category(Root),
    Cat \= Root,
    category_ancestor(Cat, Root, Hops),
    Distance is Hops + 1,  % +1 because article→category is 1 hop
    Weight is Distance ** (-N).

% category_influence(Root, Score)
% Total influence of a root category across all articles
category_influence(Root, Score) :-
    root_category(Root),
    aggregate_all(sum(W),
        article_root_weight(_, Root, W),
        Score),
    Score > 0.

% Top-level query: all root categories ranked by influence
run :-
    findall(Score-Root, category_influence(Root, Score), Pairs),
    sort(1, @>=, Pairs, Sorted),
    format("root_category\tinfluence_score\tarticle_reach~n"),
    forall(
        member(Score-Root, Sorted),
        (   aggregate_all(count, article_root_weight(_, Root, _), Count),
            format("~w\t~6f\t~w~n", [Root, Score, Count])
        )
    ).

:- initialization(run, main).
