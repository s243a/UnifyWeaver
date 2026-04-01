% Shortest Path to Root
%
% For each article, find the minimum hop count to the root category
% through the category hierarchy. Unlike effective distance (which
% aggregates all paths via power mean), this returns only the single
% shortest path length.
%
% This exercises a different aggregation over the same transitive
% closure: min(hops) instead of (Σ hops^(-n))^(-1/n).
%
% The distinction matters for the query engine: d_eff needs per-path
% enumeration (path multiplicity), while shortest path only needs
% the global minimum per (article, root) pair.
%
% Usage:
%   ?- shortest_path(Article, Root, MinHops).

:- discontiguous category_parent/2.
:- discontiguous article_category/2.
:- discontiguous root_category/1.

max_depth(10).

% category_ancestor/3 — counted transitive closure
category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).
category_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth), Depth < MaxD, !,
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

category_ancestor(Cat, Ancestor, Hops) :-
    category_ancestor(Cat, Ancestor, Hops, [Cat]).

% shortest_path(Article, Root, MinHops)
% Minimum hop distance from article to root category
shortest_path(Article, Root, MinHops) :-
    root_category(Root),
    article_category(Article, _),  % ensure Article exists
    aggregate_all(min(Dist),
        article_root_distance(Article, Root, Dist),
        MinHops),
    MinHops \= inf.

% article_root_distance(Article, Root, Distance)
% All distances from article to root (via any category assignment)
article_root_distance(Article, Root, 1) :-
    article_category(Article, Root).
article_root_distance(Article, Root, Distance) :-
    article_category(Article, Cat),
    Cat \= Root,
    category_ancestor(Cat, Root, Hops),
    Distance is Hops + 1.

% Run query: all articles sorted by shortest path to root
run :-
    root_category(Root),
    findall(MinHops-Article,
        shortest_path(Article, Root, MinHops),
        Pairs),
    sort(Pairs, Sorted),
    format("article\troot_category\tshortest_path~n"),
    forall(
        member(H-Art, Sorted),
        format("~w\t~w\t~w~n", [Art, Root, H])
    ).

:- initialization(run, main).
