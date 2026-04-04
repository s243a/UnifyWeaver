% Weighted shortest path to root
%
% For each article, find the minimum positive weighted distance to the
% root category through the category hierarchy.
%
% The step weight is attached to the source category via
% category_weight/2. The generated Prolog min helper can then retain
% only the best known accumulated weight per seeded query while still
% preserving the original recursive predicate.

:- discontiguous category_parent/2.
:- discontiguous article_category/2.
:- discontiguous root_category/1.
:- discontiguous category_weight/2.

max_depth(10).

category_weighted_ancestor(Cat, Parent, Weight, Visited) :-
    category_parent(Cat, Parent),
    category_weight(Cat, Step),
    Step > 0,
    Weight is Step,
    \+ member(Parent, Visited).

category_weighted_ancestor(Cat, Ancestor, Weight, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth), Depth < MaxD, !,
    category_parent(Cat, Mid),
    category_weight(Cat, Step),
    Step > 0,
    \+ member(Mid, Visited),
    category_weighted_ancestor(Mid, Ancestor, PrevWeight, [Mid|Visited]),
    Weight is PrevWeight + Step.

category_weighted_ancestor(Cat, Ancestor, Weight) :-
    category_weighted_ancestor(Cat, Ancestor, Weight, [Cat]).
