%% pearltrees/queries.pl - Aggregate queries for Pearltrees data
%%
%% Educational example showing UnifyWeaver aggregate queries that can
%% be compiled to multiple targets. Each target generates idiomatic
%% grouping/aggregation code.

:- module(pearltrees_queries, [
    tree_with_children/3,
    tree_child_count/2,
    incomplete_tree/2,
    trees_by_cluster/2,
    pagepearl_urls/2
]).

:- use_module(sources).

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
%% Example: Filter trees by URL domain
%%
%% Shows how predicates can compose for filtering.
%% --------------------------------------------------------------------

%% has_wikipedia_links(?TreeId) is nondet.
%%   True if tree contains any Wikipedia links.
has_wikipedia_links(TreeId) :-
    pagepearl_urls(TreeId, Urls),
    member(Url, Urls),
    sub_atom(Url, _, _, _, 'wikipedia.org'),
    !.  % Cut after first match

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
