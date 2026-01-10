%% pearltrees/queries.pl - Aggregate queries for Pearltrees data
%%
%% Educational example showing UnifyWeaver aggregate queries that can
%% be compiled to multiple targets. Each target generates idiomatic
%% grouping/aggregation code.

:- module(pearltrees_queries, [
    % Core queries
    tree_with_children/3,
    tree_child_count/2,
    incomplete_tree/2,
    trees_by_cluster/2,
    pagepearl_urls/2,

    % Query-based filters
    filter_trees/2,
    filter_children/3,

    % Domain filters
    has_domain_links/2,
    trees_with_domain/2,
    has_wikipedia_links/1,

    % Type filters
    has_child_type/2,
    trees_with_type/2,
    children_of_type/3,

    % Title/keyword filters
    title_contains/2,
    trees_matching/2,
    children_matching/3,

    % Count filters
    trees_with_min_children/2,
    trees_with_max_children/2,
    trees_in_size_range/3,

    % Combined filters
    apply_filters/3,

    % Statistics
    tree_size_distribution/2,

    % Hierarchy statistics (Phase 6)
    depth_distribution/1,
    orphan_count/1,
    leaf_count/1
]).

:- use_module(sources).
:- use_module(hierarchy).

%% --------------------------------------------------------------------
%% Aggregate: Group children by parent tree
%%
%% Generated code per target:
%%   - Python: itertools.groupby or dict comprehension
%%   - C#: LINQ GroupBy + ToList
%%   - Go: map[string][]Child with append
%%   - SQL: GROUP BY with JSON_AGG or similar
%% --------------------------------------------------------------------

%% tree_with_children(?TreeId, ?Title, ?Children) is nondet.
%%   Get a tree with all its children as a list.
%%   Children are child(Type, Title, Url, Order) terms.
tree_with_children(TreeId, Title, Children) :-
    pearl_trees(tree, TreeId, Title, _, _),
    aggregate_all(
        bag(child(Type, ChildTitle, Url, Order)),
        pearl_children(TreeId, Type, ChildTitle, Order, Url, _),
        TreeId,
        Children
    ).

%% --------------------------------------------------------------------
%% Aggregate: Count children per tree
%%
%% Generated code per target:
%%   - Python: len(list) or Counter
%%   - C#: LINQ Count()
%%   - Go: len(slice)
%%   - SQL: COUNT(*)
%% --------------------------------------------------------------------

%% tree_child_count(?TreeId, ?Count) is nondet.
%%   Count children for each tree.
tree_child_count(TreeId, Count) :-
    pearl_trees(tree, TreeId, _, _, _),
    aggregate_all(
        count,
        pearl_children(TreeId, _, _, _, _, _),
        TreeId,
        Count
    ).

%% --------------------------------------------------------------------
%% Derived: Find incomplete trees
%%
%% Trees with only a root pearl (count <= 1) need repair via API.
%% --------------------------------------------------------------------

%% incomplete_tree(?TreeId, ?Title) is nondet.
%%   Find trees that need repair (only root, no children).
incomplete_tree(TreeId, Title) :-
    tree_child_count(TreeId, Count),
    Count =< 1,
    pearl_trees(tree, TreeId, Title, _, _).

%% --------------------------------------------------------------------
%% Aggregate: Group trees by cluster
%%
%% Useful for organizing mindmaps into folders.
%% --------------------------------------------------------------------

%% trees_by_cluster(?ClusterId, ?Trees) is nondet.
%%   Group trees by their cluster/parent.
trees_by_cluster(ClusterId, Trees) :-
    pearl_trees(tree, _, _, _, ClusterId),
    aggregate_all(
        bag(tree(TreeId, Title, Uri)),
        pearl_trees(tree, TreeId, Title, Uri, ClusterId),
        ClusterId,
        Trees
    ).

%% --------------------------------------------------------------------
%% Aggregate: Collect URLs from pagepearls
%%
%% Extract all bookmark URLs from a tree for analysis.
%% --------------------------------------------------------------------

%% pagepearl_urls(?TreeId, ?Urls) is nondet.
%%   Get all pagepearl URLs for a tree.
pagepearl_urls(TreeId, Urls) :-
    pearl_trees(tree, TreeId, _, _, _),
    aggregate_all(
        bag(Url),
        (   pearl_children(TreeId, pagepearl, _, _, Url, _),
            Url \= null
        ),
        TreeId,
        Urls
    ).

%% --------------------------------------------------------------------
%% Example: Statistics query
%%
%% Aggregate of aggregates - count trees by child count ranges.
%% --------------------------------------------------------------------

%% tree_size_distribution(?SizeRange, ?Count) is nondet.
%%   Count trees by size category.
tree_size_distribution(empty, Count) :-
    aggregate_all(count, (tree_child_count(_, C), C == 0), Count).
tree_size_distribution(small, Count) :-
    aggregate_all(count, (tree_child_count(_, C), C > 0, C =< 10), Count).
tree_size_distribution(medium, Count) :-
    aggregate_all(count, (tree_child_count(_, C), C > 10, C =< 50), Count).
tree_size_distribution(large, Count) :-
    aggregate_all(count, (tree_child_count(_, C), C > 50), Count).

%% ====================================================================
%% Query-Based Filtering Predicates
%% ====================================================================
%%
%% Composable filters for selecting trees and children based on
%% various criteria. These can be compiled to WHERE clauses in SQL,
%% LINQ predicates in C#, or filter() calls in Python/Go.

%% --------------------------------------------------------------------
%% Generic Filter Framework
%% --------------------------------------------------------------------

%% filter_trees(+Filter, -TreeId) is nondet.
%%   Apply a filter to find matching trees.
%%   Filter is a term describing the filter criteria.
filter_trees(domain(Domain), TreeId) :-
    has_domain_links(TreeId, Domain).
filter_trees(type(Type), TreeId) :-
    has_child_type(TreeId, Type).
filter_trees(title_match(Pattern), TreeId) :-
    title_contains(TreeId, Pattern).
filter_trees(min_children(N), TreeId) :-
    trees_with_min_children(N, TreeId).
filter_trees(max_children(N), TreeId) :-
    trees_with_max_children(N, TreeId).
filter_trees(incomplete, TreeId) :-
    incomplete_tree(TreeId, _).
filter_trees(cluster(ClusterId), TreeId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId).

%% filter_children(+TreeId, +Filter, -Child) is nondet.
%%   Apply a filter to find matching children within a tree.
filter_children(TreeId, type(Type), child(Type, Title, Url, Order)) :-
    pearl_children(TreeId, Type, Title, Order, Url, _).
filter_children(TreeId, title_match(Pattern), child(Type, Title, Url, Order)) :-
    pearl_children(TreeId, Type, Title, Order, Url, _),
    title_matches(Title, Pattern).
filter_children(TreeId, has_url, child(Type, Title, Url, Order)) :-
    pearl_children(TreeId, Type, Title, Order, Url, _),
    Url \= null,
    Url \= ''.
filter_children(TreeId, domain(Domain), child(Type, Title, Url, Order)) :-
    pearl_children(TreeId, Type, Title, Order, Url, _),
    Url \= null,
    sub_atom(Url, _, _, _, Domain).

%% --------------------------------------------------------------------
%% Domain Filters
%% --------------------------------------------------------------------

%% has_domain_links(+TreeId, +Domain) is semidet.
%%   True if tree contains links to the specified domain.
has_domain_links(TreeId, Domain) :-
    pearl_trees(tree, TreeId, _, _, _),
    pearl_children(TreeId, pagepearl, _, _, Url, _),
    Url \= null,
    sub_atom(Url, _, _, _, Domain),
    !.  % Cut after first match

%% trees_with_domain(+Domain, -TreeId) is nondet.
%%   Find all trees containing links to a domain.
trees_with_domain(Domain, TreeId) :-
    pearl_trees(tree, TreeId, _, _, _),
    has_domain_links(TreeId, Domain).

%% has_wikipedia_links(+TreeId) is semidet.
%%   True if tree contains any Wikipedia links.
%%   (Kept for backward compatibility)
has_wikipedia_links(TreeId) :-
    has_domain_links(TreeId, 'wikipedia.org').

%% --------------------------------------------------------------------
%% Type Filters
%% --------------------------------------------------------------------

%% has_child_type(+TreeId, +Type) is semidet.
%%   True if tree has at least one child of the given type.
has_child_type(TreeId, Type) :-
    pearl_trees(tree, TreeId, _, _, _),
    pearl_children(TreeId, Type, _, _, _, _),
    !.  % Cut after first match

%% trees_with_type(+Type, -TreeId) is nondet.
%%   Find all trees containing children of a specific type.
trees_with_type(Type, TreeId) :-
    pearl_trees(tree, TreeId, _, _, _),
    has_child_type(TreeId, Type).

%% children_of_type(+TreeId, +Type, -Children) is det.
%%   Get all children of a specific type from a tree.
children_of_type(TreeId, Type, Children) :-
    aggregate_all(
        bag(child(Type, Title, Url, Order)),
        pearl_children(TreeId, Type, Title, Order, Url, _),
        Children
    ).

%% --------------------------------------------------------------------
%% Title/Keyword Filters
%% --------------------------------------------------------------------

%% title_contains(+TreeId, +Pattern) is semidet.
%%   True if tree title or any child title contains pattern.
title_contains(TreeId, Pattern) :-
    pearl_trees(tree, TreeId, Title, _, _),
    (   title_matches(Title, Pattern)
    ->  true
    ;   pearl_children(TreeId, _, ChildTitle, _, _, _),
        title_matches(ChildTitle, Pattern)
    ),
    !.

%% title_matches(+Title, +Pattern) is semidet.
%%   Case-insensitive substring match.
title_matches(Title, Pattern) :-
    downcase_atom(Title, LowerTitle),
    downcase_atom(Pattern, LowerPattern),
    sub_atom(LowerTitle, _, _, _, LowerPattern).

%% trees_matching(+Pattern, -TreeId) is nondet.
%%   Find trees with titles matching a pattern.
trees_matching(Pattern, TreeId) :-
    pearl_trees(tree, TreeId, Title, _, _),
    title_matches(Title, Pattern).

%% children_matching(+TreeId, +Pattern, -Children) is det.
%%   Get children with titles matching a pattern.
children_matching(TreeId, Pattern, Children) :-
    aggregate_all(
        bag(child(Type, Title, Url, Order)),
        (   pearl_children(TreeId, Type, Title, Order, Url, _),
            title_matches(Title, Pattern)
        ),
        Children
    ).

%% --------------------------------------------------------------------
%% Count Filters
%% --------------------------------------------------------------------

%% trees_with_min_children(+MinCount, -TreeId) is nondet.
%%   Find trees with at least N children.
trees_with_min_children(MinCount, TreeId) :-
    tree_child_count(TreeId, Count),
    Count >= MinCount.

%% trees_with_max_children(+MaxCount, -TreeId) is nondet.
%%   Find trees with at most N children.
trees_with_max_children(MaxCount, TreeId) :-
    tree_child_count(TreeId, Count),
    Count =< MaxCount.

%% trees_in_size_range(+Min, +Max, -TreeId) is nondet.
%%   Find trees with children count in range [Min, Max].
trees_in_size_range(Min, Max, TreeId) :-
    tree_child_count(TreeId, Count),
    Count >= Min,
    Count =< Max.

%% --------------------------------------------------------------------
%% Combined Filters
%% --------------------------------------------------------------------

%% apply_filters(+Filters, -TreeId, -TreeInfo) is nondet.
%%   Apply multiple filters and return matching trees with info.
%%   Filters is a list of filter terms.
%%   TreeInfo = tree_info(TreeId, Title, ChildCount).
apply_filters(Filters, TreeId, tree_info(TreeId, Title, ChildCount)) :-
    pearl_trees(tree, TreeId, Title, _, _),
    tree_child_count(TreeId, ChildCount),
    all_filters_match(Filters, TreeId).

all_filters_match([], _).
all_filters_match([Filter|Rest], TreeId) :-
    filter_matches(Filter, TreeId),
    all_filters_match(Rest, TreeId).

filter_matches(domain(Domain), TreeId) :-
    has_domain_links(TreeId, Domain).
filter_matches(type(Type), TreeId) :-
    has_child_type(TreeId, Type).
filter_matches(title_match(Pattern), TreeId) :-
    title_contains(TreeId, Pattern).
filter_matches(min_children(N), TreeId) :-
    tree_child_count(TreeId, Count),
    Count >= N.
filter_matches(max_children(N), TreeId) :-
    tree_child_count(TreeId, Count),
    Count =< N.
filter_matches(incomplete, TreeId) :-
    tree_child_count(TreeId, Count),
    Count =< 1.
filter_matches(complete, TreeId) :-
    tree_child_count(TreeId, Count),
    Count > 1.
filter_matches(cluster(ClusterId), TreeId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId).
filter_matches(not(Filter), TreeId) :-
    \+ filter_matches(Filter, TreeId).

%% Hierarchy-based filters (Phase 6 integration)
filter_matches(is_root, TreeId) :-
    root_tree(TreeId).
filter_matches(is_leaf, TreeId) :-
    leaf_tree(TreeId).
filter_matches(is_orphan, TreeId) :-
    orphan_tree(TreeId).
filter_matches(at_depth(Depth), TreeId) :-
    tree_depth(TreeId, Depth).
filter_matches(max_depth(MaxDepth), TreeId) :-
    tree_depth(TreeId, Depth),
    Depth =< MaxDepth.
filter_matches(min_depth(MinDepth), TreeId) :-
    tree_depth(TreeId, Depth),
    Depth >= MinDepth.
filter_matches(under(AncestorId), TreeId) :-
    subtree_tree(AncestorId, TreeId),
    TreeId \= AncestorId.
filter_matches(has_descendant(DescendantId), TreeId) :-
    subtree_tree(TreeId, DescendantId),
    DescendantId \= TreeId.
filter_matches(sibling_of(SiblingId), TreeId) :-
    tree_siblings(SiblingId, Siblings),
    member(TreeId, Siblings).

%% --------------------------------------------------------------------
%% Hierarchy Statistics (Phase 6 integration)
%% --------------------------------------------------------------------

%% depth_distribution(-Distribution) is det.
%%   Get count of trees at each depth level.
%%   Distribution is a list of depth-count pairs.
depth_distribution(Distribution) :-
    findall(Depth-TreeId,
            (pearl_trees(tree, TreeId, _, _, _),
             tree_depth(TreeId, Depth)),
            Pairs),
    keysort(Pairs, Sorted),
    group_count(Sorted, Distribution).

%% group_count(+SortedPairs, -CountPairs) is det.
group_count([], []).
group_count([K-_|Rest], [K-Count|Counts]) :-
    count_same_key(K, Rest, 1, Count, Remaining),
    group_count(Remaining, Counts).

count_same_key(K, [K-_|Rest], Acc, Count, Remaining) :-
    !,
    Acc1 is Acc + 1,
    count_same_key(K, Rest, Acc1, Count, Remaining).
count_same_key(_, Rest, Count, Count, Rest).

%% orphan_count(-Count) is det.
%%   Count trees with missing parents.
orphan_count(Count) :-
    findall(T, orphan_tree(T), Orphans),
    length(Orphans, Count).

%% leaf_count(-Count) is det.
%%   Count leaf trees (no children).
leaf_count(Count) :-
    findall(T, leaf_tree(T), Leaves),
    length(Leaves, Count).
