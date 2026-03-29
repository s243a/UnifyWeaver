%% ==========================================================================
%% Cross-Target Effective Distance Benchmark — C# Query Engine version
%%
%% This variant drops the explicit Visited list from category_ancestor/4
%% because the C# Query Engine's FixpointNode handles cycle detection
%% automatically via semi-naive evaluation with HashSet deduplication.
%%
%% The predicates are otherwise identical to effective_distance.pl.
%% ==========================================================================

:- discontiguous article_category/2.
:- discontiguous category_parent/2.
:- discontiguous root_category/1.

%% category_ancestor(+Cat, -Ancestor, -Hops)
%% Transitive closure over category_parent/2.
%% No Visited list needed — FixpointNode converges automatically.
category_ancestor(Cat, Parent, 1) :-
    category_parent(Cat, Parent).

category_ancestor(Cat, Ancestor, Hops) :-
    category_parent(Cat, Mid),
    category_ancestor(Mid, Ancestor, H1),
    Hops is H1 + 1.

%% path_to_root(+Article, -Root, -Hops)
path_to_root(Article, Root, 1) :-
    article_category(Article, Cat),
    root_category(Cat),
    Root = Cat.

path_to_root(Article, Root, Hops) :-
    article_category(Article, Cat),
    category_ancestor(Cat, AncestorCat, CatHops),
    root_category(AncestorCat),
    Root = AncestorCat,
    Hops is CatHops + 1.

%% effective_distance(+Article, -Root, -Deff)
effective_distance(Article, Root, Deff) :-
    setof(A, C^article_category(A, C), Articles),
    member(Article, Articles),
    root_category(Root),
    aggregate_all(sum(W),
        (path_to_root(Article, Root, Hops),
         W is Hops ** (-5)),
        WeightSum),
    WeightSum > 0,
    Deff is WeightSum ** (-0.2).
