%% ==========================================================================
%% Cross-Target Effective Distance Benchmark
%%
%% Computes effective distance from Wikipedia articles to root categories
%% via the category hierarchy, using the formula:
%%
%%     d_eff = (Σ dᵢ^(-n))^(-1/n)   where n = 5
%%
%% This program is designed to be transpiled to multiple targets:
%%   - C# Query Engine (semi-naive fixpoint)
%%   - Go (compiled fixpoint)
%%   - AWK (transitive closure + aggregation)
%%   - Python (memoized recursion)
%%
%% Usage (SWI-Prolog):
%%   swipl -l examples/benchmark/effective_distance.pl \
%%         -l data/benchmark/dev/facts.pl \
%%         -g run_benchmark -t halt
%%
%% See: docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_SPEC.md
%% ==========================================================================

%% --------------------------------------------------------------------------
%% Fact declarations — these are loaded from a separate facts file.
%% Declared as dynamic so they can be asserted or loaded from any source.
%% --------------------------------------------------------------------------

:- discontiguous article_category/2.
:- discontiguous category_parent/2.
:- discontiguous root_category/1.

%% --------------------------------------------------------------------------
%% Dimensionality parameter
%% --------------------------------------------------------------------------

dimension_n(5).

%% Max DFS depth — paths beyond this are cut. With n=5, d^(-5) at
%% depth 10 is 0.00001, contributing negligibly to d_eff. This prevents
%% combinatorial explosion on large graphs while preserving accuracy.
%% Can be overridden by asserting max_depth/1 before loading this file.
:- dynamic max_depth/1.
max_depth(10).

%% --------------------------------------------------------------------------
%% category_ancestor(+Cat, -Ancestor, -Hops, +Visited)
%%
%% Transitive closure over category_parent/2 with cycle detection.
%% Visited is a list of already-seen categories to prevent infinite loops
%% in Wikipedia's cyclic category graph. Depth is bounded by max_depth/1
%% to prevent combinatorial explosion on large graphs.
%% --------------------------------------------------------------------------

category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD),
    length(Visited, Depth),
    Depth < MaxD, !,
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

%% --------------------------------------------------------------------------
%% path_to_root(+Article, -Root, -Hops)
%%
%% Find all paths from an article to a root category.
%% The natural tree structure is: root → categories → pages.
%% An article's distance = 1 hop (article→category) + hops to root.
%% --------------------------------------------------------------------------

path_to_root(Article, Root, 1) :-
    article_category(Article, Cat),
    root_category(Cat),
    Root = Cat.

path_to_root(Article, Root, Hops) :-
    article_category(Article, Cat),
    category_ancestor(Cat, AncestorCat, CatHops, [Cat]),
    root_category(AncestorCat),
    Root = AncestorCat,
    Hops is CatHops + 1.

%% --------------------------------------------------------------------------
%% effective_distance(+Article, -Root, -Deff)
%%
%% d_eff = (Σ d^(-N))^(-1/N)
%%
%% Aggregates over all paths from Article to Root.
%% The decomposition:
%%   1. path_to_root/3  → recursive path finding (all paths)
%%   2. W is Hops^(-N)  → per-path arithmetic
%%   3. sum(W)          → aggregation
%%   4. Deff            → post-aggregation arithmetic
%% --------------------------------------------------------------------------

effective_distance(Article, Root, Deff) :-
    % Enumerate distinct (Article, Root) pairs
    setof(A, C^article_category(A, C), Articles),
    member(Article, Articles),
    root_category(Root),
    % Then aggregate all paths for this specific pair
    dimension_n(N),
    NegN is -N,
    aggregate_all(sum(W),
        (path_to_root(Article, Root, Hops),
         W is Hops ** NegN),
        WeightSum),
    WeightSum > 0,
    InvN is -1 / N,
    Deff is WeightSum ** InvN.

%% --------------------------------------------------------------------------
%% ranked_articles(+Root, -Article, -Distance)
%%
%% All articles ranked by effective distance to a root category.
%% --------------------------------------------------------------------------

ranked_articles(Root, Article, Distance) :-
    root_category(Root),
    effective_distance(Article, Root, Distance).

%% --------------------------------------------------------------------------
%% Auxiliary queries
%% --------------------------------------------------------------------------

%% depth_histogram(-Depth, -Count)
%% Distribution of effective distances rounded to integer.
%% Useful for tree layering analysis.
depth_histogram(Depth, Count) :-
    findall(D,
        (effective_distance(_, _, Deff), D is round(Deff)),
        AllDepths),
    msort(AllDepths, Sorted),
    clumped(Sorted, Pairs),
    member(Depth-Count, Pairs).

%% root_article_count(-Root, -Count)
%% How many articles connect to each root category.
root_article_count(Root, Count) :-
    root_category(Root),
    aggregate_all(count,
        effective_distance(_, Root, _),
        Count).

%% --------------------------------------------------------------------------
%% Benchmark runner — outputs TSV to stdout
%% --------------------------------------------------------------------------

run_benchmark :-
    format("article\troot_category\teffective_distance~n"),
    findall(Distance-Article-Root,
        ranked_articles(Root, Article, Distance),
        Results),
    msort(Results, Sorted),
    forall(
        member(Dist-Art-Rt, Sorted),
        format("~w\t~w\t~6f~n", [Art, Rt, Dist])
    ).
