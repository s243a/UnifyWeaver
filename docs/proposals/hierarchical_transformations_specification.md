# Specification: Hierarchical Tree Transformations

## Overview

This document specifies the Prolog predicates for hierarchical tree transformations in the Pearltrees UnifyWeaver example. These predicates build on the existing `queries.pl` foundation.

## Module Structure

```prolog
:- module(pearltrees_hierarchy, [
    % Navigation predicates
    tree_parent/2,
    tree_ancestors/2,
    tree_descendants/2,
    tree_siblings/2,
    tree_depth/2,
    tree_path/2,

    % Structural queries
    root_tree/1,
    leaf_trees/1,
    orphan_trees/1,
    subtree_trees/2,

    % Transformation predicates
    flatten_tree/3,
    prune_tree/3,
    reroot_tree/3,
    merge_trees/3,

    % Path operations
    path_depth/2,
    truncate_path/3,
    path_prefix/3,
    common_ancestor/3,

    % Grouping predicates
    trees_at_depth/2,
    trees_by_parent/2,
    group_by_ancestor/3
]).
```

## Data Model

### Tree Relationships

Trees are related through `cluster_id` (parent reference):

```prolog
%% pearl_trees(Type, TreeId, Title, Uri, ClusterId)
%%   ClusterId is the parent tree's URI, or root marker for top-level trees.

%% Example hierarchy:
%%   Root (cluster_id = 'root')
%%     ├── Science (cluster_id = Root.uri)
%%     │   ├── Physics (cluster_id = Science.uri)
%%     │   └── Chemistry (cluster_id = Science.uri)
%%     └── Arts (cluster_id = Root.uri)
%%         └── Music (cluster_id = Arts.uri)
```

### Path Representation

Paths are lists from root to node:

```prolog
%% path(['Root', 'Science', 'Physics'])
%% path([RootId, ScienceId, PhysicsId])  % TreeId version
```

## Predicate Specifications

### Navigation Predicates

#### tree_parent/2

```prolog
%% tree_parent(?TreeId, ?ParentId) is nondet.
%%   True if ParentId is the immediate parent of TreeId.
%%
%%   Implementation notes:
%%   - Derives parent from cluster_id relationship
%%   - Fails for root trees (no parent)
%%
%% Example:
%%   ?- tree_parent('physics_123', Parent).
%%   Parent = 'science_456'.

tree_parent(TreeId, ParentId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    cluster_to_tree_id(ClusterId, ParentId).
```

#### tree_ancestors/2

```prolog
%% tree_ancestors(?TreeId, ?Ancestors) is det.
%%   Ancestors is the list of tree IDs from TreeId to root (exclusive).
%%   Returns [] for root trees.
%%
%% Example:
%%   ?- tree_ancestors('physics_123', Ancestors).
%%   Ancestors = ['science_456', 'root_789'].

tree_ancestors(TreeId, Ancestors) :-
    tree_ancestors_(TreeId, [], Ancestors).

tree_ancestors_(TreeId, Acc, Ancestors) :-
    (   tree_parent(TreeId, ParentId)
    ->  tree_ancestors_(ParentId, [ParentId|Acc], Ancestors)
    ;   reverse(Acc, Ancestors)
    ).
```

#### tree_descendants/2

```prolog
%% tree_descendants(?TreeId, ?Descendants) is det.
%%   Descendants is the list of all tree IDs under TreeId (recursive).
%%
%% Example:
%%   ?- tree_descendants('science_456', Descendants).
%%   Descendants = ['physics_123', 'chemistry_124', 'quantum_125'].

tree_descendants(TreeId, Descendants) :-
    findall(ChildId, tree_parent(ChildId, TreeId), DirectChildren),
    maplist(tree_descendants, DirectChildren, NestedDescendants),
    append([DirectChildren|NestedDescendants], Descendants).
```

#### tree_siblings/2

```prolog
%% tree_siblings(?TreeId, ?Siblings) is det.
%%   Siblings is the list of trees sharing the same parent (excluding TreeId).
%%
%% Example:
%%   ?- tree_siblings('physics_123', Siblings).
%%   Siblings = ['chemistry_124'].

tree_siblings(TreeId, Siblings) :-
    tree_parent(TreeId, ParentId),
    findall(SibId,
            (tree_parent(SibId, ParentId), SibId \= TreeId),
            Siblings).
```

#### tree_depth/2

```prolog
%% tree_depth(?TreeId, ?Depth) is det.
%%   Depth is the number of edges from root to TreeId.
%%   Root has depth 0.
%%
%% Example:
%%   ?- tree_depth('physics_123', Depth).
%%   Depth = 2.

tree_depth(TreeId, Depth) :-
    tree_ancestors(TreeId, Ancestors),
    length(Ancestors, Depth).
```

#### tree_path/2

```prolog
%% tree_path(?TreeId, ?Path) is det.
%%   Path is the list of TreeIds from root to TreeId (inclusive).
%%
%% Example:
%%   ?- tree_path('physics_123', Path).
%%   Path = ['root_789', 'science_456', 'physics_123'].

tree_path(TreeId, Path) :-
    tree_ancestors(TreeId, Ancestors),
    append(Ancestors, [TreeId], Path).
```

### Structural Queries

#### root_tree/1

```prolog
%% root_tree(?TreeId) is nondet.
%%   True if TreeId has no parent (is a root).
%%
%% Example:
%%   ?- root_tree(RootId).
%%   RootId = 'root_789'.

root_tree(TreeId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId),
    (ClusterId = root ; ClusterId = '').
```

#### leaf_trees/1

```prolog
%% leaf_trees(?TreeId) is nondet.
%%   True if TreeId has no child trees.
%%
%% Example:
%%   ?- leaf_trees(LeafId).
%%   LeafId = 'physics_123' ;
%%   LeafId = 'chemistry_124'.

leaf_trees(TreeId) :-
    pearl_trees(tree, TreeId, _, _, _),
    \+ tree_parent(_, TreeId).
```

#### orphan_trees/1

```prolog
%% orphan_trees(?TreeId) is nondet.
%%   True if TreeId's parent doesn't exist in the dataset.
%%   These are trees disconnected from the main hierarchy.
%%
%% Example:
%%   ?- orphan_trees(OrphanId).
%%   OrphanId = 'lost_999'.

orphan_trees(TreeId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    ClusterId \= '',
    \+ cluster_to_tree_id(ClusterId, _).
```

#### subtree_trees/2

```prolog
%% subtree_trees(?RootId, ?TreeId) is nondet.
%%   True if TreeId is RootId or a descendant of RootId.
%%
%% Example:
%%   ?- subtree_trees('science_456', TreeId).
%%   TreeId = 'science_456' ;
%%   TreeId = 'physics_123' ;
%%   TreeId = 'chemistry_124'.

subtree_trees(RootId, RootId) :-
    pearl_trees(tree, RootId, _, _, _).
subtree_trees(RootId, TreeId) :-
    tree_descendants(RootId, Descendants),
    member(TreeId, Descendants).
```

### Transformation Predicates

#### flatten_tree/3

```prolog
%% flatten_tree(?TreeId, +MaxDepth, ?FlatChildren) is det.
%%   Collect all descendants up to MaxDepth into a flat list.
%%   Children beyond MaxDepth are "promoted" to MaxDepth level.
%%
%% Example:
%%   ?- flatten_tree('root_789', 1, FlatChildren).
%%   % Returns Science, Arts, Physics, Chemistry, Music all at depth 1
%%   FlatChildren = [tree_info('science_456', 'Science', 1), ...].

flatten_tree(TreeId, MaxDepth, FlatChildren) :-
    findall(
        tree_info(DescId, Title, min(Depth, MaxDepth)),
        (   subtree_trees(TreeId, DescId),
            DescId \= TreeId,
            pearl_trees(tree, DescId, Title, _, _),
            tree_depth_relative(TreeId, DescId, Depth)
        ),
        FlatChildren
    ).

%% tree_depth_relative(+AncestorId, +DescendantId, -RelativeDepth)
%%   Depth of Descendant relative to Ancestor.
tree_depth_relative(AncestorId, DescendantId, RelativeDepth) :-
    tree_depth(AncestorId, AncestorDepth),
    tree_depth(DescendantId, DescendantDepth),
    RelativeDepth is DescendantDepth - AncestorDepth.
```

#### prune_tree/3

```prolog
%% prune_tree(?TreeId, +Filter, ?PrunedDescendants) is det.
%%   Return descendants that match Filter, excluding pruned branches.
%%   If a node fails Filter, its entire subtree is excluded.
%%
%% Filter terms:
%%   - max_depth(N): Exclude nodes deeper than N
%%   - has_type(Type): Include only nodes with children of Type
%%   - title_match(Pattern): Include only matching titles
%%
%% Example:
%%   ?- prune_tree('root_789', max_depth(2), Pruned).
%%   Pruned = ['science_456', 'physics_123', 'arts_457'].

prune_tree(TreeId, Filter, PrunedDescendants) :-
    findall(
        DescId,
        (   subtree_trees(TreeId, DescId),
            DescId \= TreeId,
            passes_prune_filter(DescId, TreeId, Filter)
        ),
        PrunedDescendants
    ).

passes_prune_filter(TreeId, RootId, max_depth(N)) :-
    tree_depth_relative(RootId, TreeId, Depth),
    Depth =< N.
passes_prune_filter(TreeId, _, has_type(Type)) :-
    has_child_type(TreeId, Type).
passes_prune_filter(TreeId, _, title_match(Pattern)) :-
    pearl_trees(tree, TreeId, Title, _, _),
    title_matches(Title, Pattern).
```

#### reroot_tree/3

```prolog
%% reroot_tree(+OldRoot, +NewRoot, -Mapping) is det.
%%   Compute new paths as if NewRoot were the root.
%%   Mapping is a list of reroot_info(TreeId, OldPath, NewPath) terms.
%%
%%   Trees not descended from NewRoot get paths prefixed with '..'.
%%
%% Example:
%%   ?- reroot_tree('root_789', 'science_456', Mapping).
%%   Mapping = [
%%     reroot_info('physics_123', [...], ['science_456', 'physics_123']),
%%     reroot_info('arts_457', [...], ['..', 'arts_457'])
%%   ].

reroot_tree(OldRoot, NewRoot, Mapping) :-
    findall(
        reroot_info(TreeId, OldPath, NewPath),
        (   subtree_trees(OldRoot, TreeId),
            tree_path(TreeId, OldPath),
            compute_rerooted_path(TreeId, NewRoot, NewPath)
        ),
        Mapping
    ).

compute_rerooted_path(TreeId, NewRoot, NewPath) :-
    (   subtree_trees(NewRoot, TreeId)
    ->  % TreeId is under NewRoot: compute relative path
        tree_path(TreeId, FullPath),
        tree_path(NewRoot, NewRootPath),
        length(NewRootPath, PrefixLen),
        length(FullPath, FullLen),
        DropLen is PrefixLen - 1,
        (DropLen > 0 -> length(Prefix, DropLen), append(Prefix, NewPath, FullPath) ; NewPath = FullPath)
    ;   % TreeId is outside NewRoot: prefix with '..'
        tree_path(TreeId, FullPath),
        NewPath = ['..', TreeId]
    ).
```

#### merge_trees/3

```prolog
%% merge_trees(+TreeIds, +MergeOptions, -MergedChildren) is det.
%%   Merge children from multiple trees into a single list.
%%   Handles duplicates based on MergeOptions.
%%
%% MergeOptions:
%%   - dedup(url): Remove duplicates by URL
%%   - dedup(title): Remove duplicates by title
%%   - keep_all: Keep all children
%%   - sort_by(order): Sort by original order
%%   - sort_by(title): Sort alphabetically
%%
%% Example:
%%   ?- merge_trees(['physics_123', 'chemistry_124'], [dedup(url)], Merged).

merge_trees(TreeIds, MergeOptions, MergedChildren) :-
    findall(
        Child,
        (   member(TreeId, TreeIds),
            tree_with_children(TreeId, _, Children),
            member(Child, Children)
        ),
        AllChildren
    ),
    apply_merge_options(AllChildren, MergeOptions, MergedChildren).

apply_merge_options(Children, Options, Result) :-
    (   member(dedup(url), Options)
    ->  dedup_by_url(Children, Deduped)
    ;   member(dedup(title), Options)
    ->  dedup_by_title(Children, Deduped)
    ;   Deduped = Children
    ),
    (   member(sort_by(title), Options)
    ->  sort_by_title(Deduped, Result)
    ;   Result = Deduped
    ).
```

### Path Operations

#### path_depth/2

```prolog
%% path_depth(+Path, -Depth) is det.
%%   Depth is the number of elements in Path minus 1.
%%
%% Example:
%%   ?- path_depth(['root', 'science', 'physics'], Depth).
%%   Depth = 2.

path_depth(Path, Depth) :-
    length(Path, Len),
    Depth is Len - 1.
```

#### truncate_path/3

```prolog
%% truncate_path(+Path, +MaxDepth, -TruncatedPath) is det.
%%   Truncate Path to at most MaxDepth+1 elements (root + MaxDepth levels).
%%
%% Example:
%%   ?- truncate_path(['root', 'a', 'b', 'c', 'd'], 2, Truncated).
%%   Truncated = ['root', 'a', 'b'].

truncate_path(Path, MaxDepth, TruncatedPath) :-
    MaxLen is MaxDepth + 1,
    length(Path, Len),
    (   Len =< MaxLen
    ->  TruncatedPath = Path
    ;   length(TruncatedPath, MaxLen),
        append(TruncatedPath, _, Path)
    ).
```

#### common_ancestor/3

```prolog
%% common_ancestor(+TreeId1, +TreeId2, -AncestorId) is det.
%%   Find the nearest common ancestor of two trees.
%%
%% Example:
%%   ?- common_ancestor('physics_123', 'chemistry_124', Ancestor).
%%   Ancestor = 'science_456'.

common_ancestor(TreeId1, TreeId2, AncestorId) :-
    tree_path(TreeId1, Path1),
    tree_path(TreeId2, Path2),
    common_prefix(Path1, Path2, CommonPath),
    last(CommonPath, AncestorId).

common_prefix([H|T1], [H|T2], [H|Common]) :-
    !,
    common_prefix(T1, T2, Common).
common_prefix(_, _, []).
```

### Grouping Predicates

#### trees_at_depth/2

```prolog
%% trees_at_depth(+Depth, -TreeIds) is det.
%%   Find all trees at exactly Depth levels from root.
%%
%% Example:
%%   ?- trees_at_depth(2, Trees).
%%   Trees = ['physics_123', 'chemistry_124', 'music_125'].

trees_at_depth(Depth, TreeIds) :-
    findall(TreeId,
            (pearl_trees(tree, TreeId, _, _, _), tree_depth(TreeId, Depth)),
            TreeIds).
```

#### trees_by_parent/2

```prolog
%% trees_by_parent(?ParentId, ?Children) is nondet.
%%   Group trees by their parent.
%%
%% Example:
%%   ?- trees_by_parent('science_456', Children).
%%   Children = ['physics_123', 'chemistry_124'].

trees_by_parent(ParentId, Children) :-
    pearl_trees(tree, ParentId, _, ParentUri, _),
    findall(ChildId,
            (pearl_trees(tree, ChildId, _, _, ClusterId),
             ClusterId = ParentUri),
            Children),
    Children \= [].
```

#### group_by_ancestor/3

```prolog
%% group_by_ancestor(+Depth, -Groups) is det.
%%   Group all trees by their ancestor at Depth.
%%   Groups is a list of group(AncestorId, TreeIds) terms.
%%
%% Example:
%%   ?- group_by_ancestor(1, Groups).
%%   Groups = [group('science_456', ['physics_123', 'chemistry_124']),
%%             group('arts_457', ['music_125'])].

group_by_ancestor(Depth, Groups) :-
    trees_at_depth(Depth, Ancestors),
    findall(
        group(AncestorId, Descendants),
        (   member(AncestorId, Ancestors),
            findall(DescId, subtree_trees(AncestorId, DescId), AllDesc),
            exclude(=(AncestorId), AllDesc, Descendants)
        ),
        Groups
    ).
```

## Helper Predicates

```prolog
%% cluster_to_tree_id(+ClusterId, -TreeId) is semidet.
%%   Convert a cluster URI to its tree ID.
cluster_to_tree_id(ClusterId, TreeId) :-
    pearl_trees(tree, TreeId, _, ClusterId, _).

%% title_matches/2 - from queries.pl
%% has_child_type/2 - from queries.pl
%% tree_with_children/3 - from queries.pl
```

## Integration with Existing Predicates

These predicates build on `queries.pl`:

| queries.pl | hierarchy.pl |
|------------|--------------|
| `tree_with_children/3` | Used by `merge_trees/3` |
| `tree_child_count/2` | Used for leaf detection |
| `has_child_type/2` | Used in `prune_tree/3` filters |
| `title_matches/2` | Used in `prune_tree/3` filters |
| `apply_filters/3` | Can compose with hierarchy predicates |

## Test Requirements

Each predicate requires tests with mock data covering:

1. **Base cases**: Empty trees, single nodes
2. **Recursion**: Multi-level hierarchies
3. **Edge cases**: Orphans, cycles (if applicable), maximum depth
4. **Composition**: Combining with filter predicates

Target: 30+ tests for hierarchy predicates.
