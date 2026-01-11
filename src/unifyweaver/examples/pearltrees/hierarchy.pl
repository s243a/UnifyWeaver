%% pearltrees/hierarchy.pl - Hierarchical tree transformations
%%
%% Predicates for navigating, querying, and transforming Pearltrees hierarchies.
%% Builds on queries.pl and supports cross-account traversal via AliasPearls.
%%
%% See docs/proposals/hierarchical_transformations_specification.md for full spec.

:- module(pearltrees_hierarchy, [
    % Navigation predicates (Phase 1)
    tree_parent/2,
    tree_parent/3,
    tree_ancestors/2,
    tree_ancestors/3,
    tree_descendants/2,
    tree_descendants/3,
    tree_siblings/2,
    tree_depth/2,
    tree_path/2,
    tree_path/3,
    tree_title/2,

    % Structural queries (Phase 2)
    root_tree/1,
    leaf_tree/1,
    orphan_tree/1,
    subtree_tree/2,
    subtree_tree/3,

    % Path operations (Phase 3)
    path_depth/2,
    truncate_path/3,
    common_ancestor/3,
    materialized_path/2,
    hierarchical_title_path/2,

    % Embedding support
    structural_embedding_input/3,
    format_id_path/2,
    format_title_hierarchy/2,

    % Basic transformations (Phase 4)
    flatten_tree/3,
    prune_tree/3,
    trees_at_depth/2,
    trees_by_parent/2,

    % Advanced transformations (Phase 5)
    reroot_tree/3,
    merge_trees/3,
    group_by_ancestor/3,

    % Alias handling
    alias_target/2,
    alias_children/2,
    tree_aliases/2,

    % Cycle detection
    has_cycle/1,
    cycle_free_path/2
]).

:- use_module(sources).

%% ============================================================================
%% Phase 1: Navigation Predicates
%% ============================================================================

%% tree_parent(?TreeId, ?ParentId) is nondet.
%% tree_parent(?TreeId, ?ParentId, +Options) is nondet.
%%   True if ParentId is the immediate parent of TreeId.
%%   Options: follow_aliases(true/false) - whether to follow alias links
tree_parent(TreeId, ParentId) :-
    tree_parent(TreeId, ParentId, []).

tree_parent(TreeId, ParentId, _Options) :-
    pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    ClusterId \= '',
    cluster_to_tree_id(ClusterId, ParentId).

%% tree_ancestors(?TreeId, ?Ancestors) is det.
%% tree_ancestors(?TreeId, ?Ancestors, +Options) is det.
%%   Ancestors is the list of tree IDs from root to TreeId (exclusive of TreeId).
%%   Returns [] for root trees.
tree_ancestors(TreeId, Ancestors) :-
    tree_ancestors(TreeId, Ancestors, []).

tree_ancestors(TreeId, Ancestors, Options) :-
    tree_ancestors_safe(TreeId, Options, [TreeId], [], Ancestors).

%% tree_descendants(?TreeId, ?Descendants) is det.
%% tree_descendants(?TreeId, ?Descendants, +Options) is det.
%%   Descendants is the list of all tree IDs under TreeId (recursive).
%%   Options: follow_aliases(true/false) - whether to follow alias links
%%   Uses cycle detection to prevent infinite loops.
tree_descendants(TreeId, Descendants) :-
    tree_descendants(TreeId, Descendants, []).

tree_descendants(TreeId, Descendants, Options) :-
    findall(Desc, tree_descendant_safe(TreeId, Desc, Options, [TreeId]), Descendants).

%% Helper: tree_descendant_of(+TreeId, -Descendant, +Options)
%%   Legacy interface - uses cycle-safe implementation internally.
tree_descendant_of(TreeId, Descendant, Options) :-
    tree_descendant_safe(TreeId, Descendant, Options, [TreeId]).

%% tree_siblings(?TreeId, ?Siblings) is det.
%%   Siblings is the list of trees sharing the same parent (excluding TreeId).
tree_siblings(TreeId, Siblings) :-
    (   tree_parent(TreeId, ParentId)
    ->  findall(SibId,
                (tree_parent(SibId, ParentId), SibId \= TreeId),
                Siblings)
    ;   Siblings = []
    ).

%% tree_depth(?TreeId, ?Depth) is det.
%%   Depth is the number of edges from root to TreeId.
%%   Root has depth 0.
tree_depth(TreeId, Depth) :-
    tree_ancestors(TreeId, Ancestors),
    length(Ancestors, Depth).

%% tree_path(?TreeId, ?Path) is det.
%% tree_path(?TreeId, ?Path, +Options) is det.
%%   Path is the list of TreeIds from root to TreeId (inclusive).
tree_path(TreeId, Path) :-
    tree_path(TreeId, Path, []).

tree_path(TreeId, Path, Options) :-
    tree_ancestors(TreeId, Ancestors, Options),
    append(Ancestors, [TreeId], Path).

%% tree_title(+TreeId, -Title) is det.
%%   Get the title of a tree.
tree_title(TreeId, Title) :-
    pearl_trees(tree, TreeId, Title, _, _).

%% ============================================================================
%% Phase 2: Structural Queries
%% ============================================================================

%% root_tree(?TreeId) is nondet.
%%   True if TreeId has no parent (is a root).
root_tree(TreeId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId),
    (ClusterId = root ; ClusterId = '').

%% leaf_tree(?TreeId) is nondet.
%%   True if TreeId has no child trees.
leaf_tree(TreeId) :-
    pearl_trees(tree, TreeId, _, _, _),
    \+ tree_parent(_, TreeId).

%% orphan_tree(?TreeId) is nondet.
%%   True if TreeId's parent doesn't exist in the dataset.
%%   These are trees disconnected from the main hierarchy.
orphan_tree(TreeId) :-
    pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    ClusterId \= '',
    \+ cluster_to_tree_id(ClusterId, _).

%% subtree_tree(?RootId, ?TreeId) is nondet.
%% subtree_tree(?RootId, ?TreeId, +Options) is nondet.
%%   True if TreeId is RootId or a descendant of RootId.
subtree_tree(RootId, TreeId) :-
    subtree_tree(RootId, TreeId, []).

subtree_tree(RootId, RootId, _Options) :-
    pearl_trees(tree, RootId, _, _, _).
subtree_tree(RootId, TreeId, Options) :-
    tree_descendant_of(RootId, TreeId, Options).

%% ============================================================================
%% Phase 3: Path Operations
%% ============================================================================

%% path_depth(+Path, -Depth) is det.
%%   Depth is the number of elements in Path minus 1.
path_depth(Path, Depth) :-
    length(Path, Len),
    Depth is Len - 1.

%% truncate_path(+Path, +MaxDepth, -TruncatedPath) is det.
%%   Truncate Path to at most MaxDepth+1 elements (root + MaxDepth levels).
truncate_path(Path, MaxDepth, TruncatedPath) :-
    MaxLen is MaxDepth + 1,
    length(Path, Len),
    (   Len =< MaxLen
    ->  TruncatedPath = Path
    ;   length(TruncatedPath, MaxLen),
        append(TruncatedPath, _, Path)
    ).

%% common_ancestor(+TreeId1, +TreeId2, -AncestorId) is det.
%%   Find the nearest common ancestor of two trees.
common_ancestor(TreeId1, TreeId2, AncestorId) :-
    tree_path(TreeId1, Path1),
    tree_path(TreeId2, Path2),
    common_prefix(Path1, Path2, CommonPath),
    (   CommonPath = []
    ->  fail  % No common ancestor
    ;   last(CommonPath, AncestorId)
    ).

common_prefix([H|T1], [H|T2], [H|Common]) :-
    !,
    common_prefix(T1, T2, Common).
common_prefix(_, _, []).

%% materialized_path(+TreeId, -PathIds) is det.
%%   Get the materialized path of IDs from root to TreeId.
materialized_path(TreeId, PathIds) :-
    tree_path(TreeId, PathIds).

%% hierarchical_title_path(+TreeId, -TitlePath) is det.
%%   Get the path as a list of titles from root to TreeId.
hierarchical_title_path(TreeId, TitlePath) :-
    tree_path(TreeId, PathIds),
    maplist(tree_title, PathIds, TitlePath).

%% ============================================================================
%% Embedding Support
%% ============================================================================

%% structural_embedding_input(+TreeId, +ChildTitle, -EmbeddingText) is det.
%%   Generate the exact format used for output embeddings.
%%   Format matches pearltrees_target_generator.py:
%%     Line 1: /id1/id2/id3
%%     Lines 2+: Indented title hierarchy
structural_embedding_input(TreeId, ChildTitle, EmbeddingText) :-
    % Get ID path
    tree_path(TreeId, PathIds),
    format_id_path(PathIds, IdPathLine),

    % Get title path and add child
    hierarchical_title_path(TreeId, PathTitles),
    append(PathTitles, [ChildTitle], FullTitles),
    format_title_hierarchy(FullTitles, TitleLines),

    % Combine: ID path line + title hierarchy
    atomic_list_concat([IdPathLine|TitleLines], '\n', EmbeddingText).

%% format_id_path(+PathIds, -IdPathLine) is det.
%%   Format IDs as slash-separated path: /id1/id2/id3
format_id_path(PathIds, IdPathLine) :-
    atomic_list_concat(PathIds, '/', IdsJoined),
    atom_concat('/', IdsJoined, IdPathLine).

%% format_title_hierarchy(+Titles, -Lines) is det.
%%   Format titles as indented markdown list (2 spaces per level).
format_title_hierarchy(Titles, Lines) :-
    format_title_hierarchy_(Titles, 0, Lines).

format_title_hierarchy_([], _, []).
format_title_hierarchy_([Title|Rest], Depth, [Line|Lines]) :-
    % 2 spaces per indent level
    IndentCount is Depth * 2,
    length(SpaceList, IndentCount),
    maplist(=(0' ), SpaceList),
    atom_codes(Indent, SpaceList),
    format(atom(Line), '~w- ~w', [Indent, Title]),
    NextDepth is Depth + 1,
    format_title_hierarchy_(Rest, NextDepth, Lines).

%% ============================================================================
%% Helper Predicates
%% ============================================================================

%% cluster_to_tree_id(+ClusterId, -TreeId) is semidet.
%%   Convert a cluster URI to its tree ID.
cluster_to_tree_id(ClusterId, TreeId) :-
    pearl_trees(tree, TreeId, _, ClusterId, _).

%% ============================================================================
%% Phase 4: Basic Transformations
%% ============================================================================

%% flatten_tree(+TreeId, +MaxDepth, -FlattenedTrees) is det.
%%   Collapse a tree hierarchy to MaxDepth levels.
%%   Returns all trees in the subtree with their paths truncated.
%%   Result is a list of TreeId-TruncatedPath pairs.
flatten_tree(TreeId, MaxDepth, FlattenedTrees) :-
    findall(DescId-TruncPath,
            (subtree_tree(TreeId, DescId),
             tree_path(DescId, FullPath),
             truncate_path(FullPath, MaxDepth, TruncPath)),
            FlattenedTrees).

%% prune_tree(+TreeId, +Criteria, -PrunedTrees) is det.
%%   Remove branches from a tree based on Criteria.
%%   Criteria can be:
%%     - max_depth(N): Remove trees deeper than N
%%     - has_type(Type): Keep only trees with children of Type
%%     - exclude_orphans: Remove orphan trees
%%   Returns list of TreeIds that pass the criteria.
prune_tree(TreeId, Criteria, PrunedTrees) :-
    findall(DescId,
            (subtree_tree(TreeId, DescId),
             satisfies_criteria(DescId, Criteria)),
            PrunedTrees).

%% satisfies_criteria(+TreeId, +Criteria) is semidet.
%%   Check if a tree satisfies the pruning criteria.
satisfies_criteria(TreeId, max_depth(MaxDepth)) :-
    tree_depth(TreeId, Depth),
    Depth =< MaxDepth.
satisfies_criteria(TreeId, has_children) :-
    tree_parent(_, TreeId),
    !.
satisfies_criteria(TreeId, is_leaf) :-
    leaf_tree(TreeId).
satisfies_criteria(TreeId, exclude_orphans) :-
    \+ orphan_tree(TreeId).

%% trees_at_depth(+Depth, -Trees) is det.
%%   Find all trees at exactly the specified depth.
trees_at_depth(Depth, Trees) :-
    findall(TreeId,
            (pearl_trees(tree, TreeId, _, _, _),
             tree_depth(TreeId, Depth)),
            Trees).

%% trees_by_parent(-GroupedTrees) is det.
%%   Group all trees by their parent.
%%   Returns a list of ParentId-ChildrenList pairs.
%%   Root trees are grouped under the atom 'root'.
trees_by_parent(GroupedTrees) :-
    findall(ParentId-TreeId,
            (pearl_trees(tree, TreeId, _, _, _),
             (   tree_parent(TreeId, ParentId)
             ->  true
             ;   ParentId = root
             )),
            Pairs),
    group_pairs_by_key(Pairs, GroupedTrees).

%% group_pairs_by_key(+Pairs, -Grouped) is det.
%%   Group a list of Key-Value pairs by key.
%%   Uses library(pairs) style grouping.
group_pairs_by_key(Pairs, Grouped) :-
    keysort(Pairs, Sorted),
    group_pairs_by_key_(Sorted, Grouped).

group_pairs_by_key_([], []).
group_pairs_by_key_([K-V|Rest], [K-[V|Vs]|Groups]) :-
    same_key(K, Rest, Vs, Remaining),
    group_pairs_by_key_(Remaining, Groups).

same_key(K, [K-V|Rest], [V|Vs], Remaining) :-
    !,
    same_key(K, Rest, Vs, Remaining).
same_key(_, Rest, [], Rest).

%% ============================================================================
%% Phase 5: Advanced Transformations
%% ============================================================================

%% reroot_tree(+NewRootId, +TreeIds, -RerootedPaths) is det.
%%   Change the root of a set of trees to NewRootId.
%%   Returns TreeId-NewPath pairs where NewPath starts from NewRootId.
%%   Trees not under NewRootId get their full original path.
reroot_tree(NewRootId, TreeIds, RerootedPaths) :-
    tree_path(NewRootId, NewRootPath),
    length(NewRootPath, NewRootDepth),
    findall(TreeId-RerootedPath,
            (member(TreeId, TreeIds),
             tree_path(TreeId, OrigPath),
             reroot_path(OrigPath, NewRootPath, NewRootDepth, RerootedPath)),
            RerootedPaths).

%% reroot_path(+OrigPath, +NewRootPath, +NewRootDepth, -RerootedPath) is det.
%%   Helper to reroot a single path.
reroot_path(OrigPath, NewRootPath, NewRootDepth, RerootedPath) :-
    (   append(NewRootPath, Suffix, OrigPath)
    ->  % Tree is under new root - strip new root's ancestors
        append([NewRootId], Suffix, RerootedPath),
        last(NewRootPath, NewRootId)
    ;   % Tree is not under new root - keep original path
        RerootedPath = OrigPath
    ).

%% merge_trees(+TreeIdLists, +Options, -MergedTrees) is det.
%%   Merge multiple tree lists into one, optionally deduplicating.
%%   Options:
%%     - dedup(true/false): Remove duplicate TreeIds (default: true)
%%     - preserve_order(true/false): Maintain original order (default: true)
merge_trees(TreeIdLists, Options, MergedTrees) :-
    append(TreeIdLists, AllTrees),
    (   option(dedup(false), Options)
    ->  MergedTrees = AllTrees
    ;   list_to_set(AllTrees, MergedTrees)
    ).

%% option(+Opt, +Options) is semidet.
%%   Check if option is present in list.
option(Opt, Options) :-
    member(Opt, Options), !.
option(Opt, Options) :-
    Opt =.. [Name, _],
    Default =.. [Name, _],
    \+ member(Default, Options).

%% group_by_ancestor(+TreeIds, +Depth, -GroupedTrees) is det.
%%   Group trees by their ancestor at the specified depth.
%%   Returns AncestorId-TreeIds pairs.
%%   Trees with depth < Depth are grouped under 'shallow'.
group_by_ancestor(TreeIds, Depth, GroupedTrees) :-
    findall(AncestorId-TreeId,
            (member(TreeId, TreeIds),
             ancestor_at_depth(TreeId, Depth, AncestorId)),
            Pairs),
    group_pairs_by_key(Pairs, GroupedTrees).

%% ancestor_at_depth(+TreeId, +Depth, -AncestorId) is det.
%%   Get the ancestor of TreeId at the specified depth.
%%   Returns 'shallow' if tree's depth is less than requested depth.
ancestor_at_depth(TreeId, Depth, AncestorId) :-
    tree_path(TreeId, Path),
    length(Path, PathLen),
    TargetIdx is Depth + 1,  % 1-based index in path
    (   PathLen >= TargetIdx
    ->  nth1(TargetIdx, Path, AncestorId)
    ;   AncestorId = shallow
    ).

%% ============================================================================
%% Alias Handling
%% ============================================================================
%%
%% AliasPearls link trees across accounts. When follow_aliases(true) is set,
%% traversal predicates will follow these links to include cross-account trees.
%%
%% pearl_children(TreeId, alias, Title, Order, null, SeeAlsoUri)
%%   - SeeAlsoUri contains the target tree's URI

%% alias_target(+AliasUri, -TargetTreeId) is semidet.
%%   Resolve an alias URI to its target tree ID.
%%   The alias URI typically looks like: uri:tree_id or similar format.
alias_target(AliasUri, TargetTreeId) :-
    nonvar(AliasUri),
    AliasUri \= null,
    AliasUri \= '',
    pearl_trees(tree, TargetTreeId, _, AliasUri, _).

%% alias_children(+TreeId, -AliasTargets) is det.
%%   Get all alias children of a tree and resolve them to target tree IDs.
%%   Returns list of target tree IDs that this tree aliases to.
alias_children(TreeId, AliasTargets) :-
    findall(TargetId,
            (pearl_children(TreeId, alias, _, _, _, SeeAlsoUri),
             alias_target(SeeAlsoUri, TargetId)),
            AliasTargets).

%% tree_aliases(+TreeId, -AliasInfo) is det.
%%   Get detailed alias information for a tree.
%%   Returns list of alias(Title, Order, TargetId) terms.
tree_aliases(TreeId, AliasInfo) :-
    findall(alias(Title, Order, TargetId),
            (pearl_children(TreeId, alias, Title, Order, _, SeeAlsoUri),
             alias_target(SeeAlsoUri, TargetId)),
            AliasInfo).

%% ============================================================================
%% Cycle Detection
%% ============================================================================
%%
%% Cycles can occur when:
%% 1. A tree's cluster_id points to a descendant (structural cycle)
%% 2. Aliases create logical cycles (A -> B -> A)
%%
%% All traversal predicates use cycle detection internally.

%% has_cycle(+TreeId) is semidet.
%%   True if following ancestors from TreeId leads back to TreeId.
%%   Detects structural cycles in the hierarchy.
has_cycle(TreeId) :-
    has_cycle_(TreeId, [TreeId]).

has_cycle_(TreeId, Visited) :-
    tree_parent(TreeId, ParentId, []),  % Don't follow aliases for cycle check
    (   member(ParentId, Visited)
    ->  true  % Cycle detected
    ;   has_cycle_(ParentId, [ParentId|Visited])
    ).

%% cycle_free_path(+TreeId, -Path) is det.
%%   Get the path from root to TreeId, stopping if a cycle is detected.
%%   Returns the path up to (but not including) the repeated node.
cycle_free_path(TreeId, Path) :-
    cycle_free_path_(TreeId, [], [], Path).

cycle_free_path_(TreeId, Visited, Acc, Path) :-
    (   member(TreeId, Visited)
    ->  % Cycle detected - return accumulated path
        reverse(Acc, Path)
    ;   (   tree_parent(TreeId, ParentId, [])
        ->  cycle_free_path_(ParentId, [TreeId|Visited], [TreeId|Acc], Path)
        ;   % Reached root
            reverse([TreeId|Acc], Path)
        )
    ).

%% ============================================================================
%% Internal: Cycle-Safe Traversal Helpers
%% ============================================================================

%% tree_ancestors_safe(+TreeId, +Options, +Visited, +Acc, -Ancestors) is det.
%%   Cycle-safe version of tree_ancestors_.
%%   Stops traversal if a cycle is detected.
tree_ancestors_safe(TreeId, Options, Visited, Acc, Ancestors) :-
    (   member(TreeId, Visited)
    ->  % Cycle detected - stop here
        Ancestors = Acc
    ;   (   tree_parent_extended(TreeId, ParentId, Options)
        ->  tree_ancestors_safe(ParentId, Options, [TreeId|Visited], [ParentId|Acc], Ancestors)
        ;   Ancestors = Acc
        )
    ).

%% tree_parent_extended(+TreeId, -ParentId, +Options) is nondet.
%%   Extended parent lookup that respects follow_aliases option.
%%   When follow_aliases(true), also returns parent via alias resolution.
tree_parent_extended(TreeId, ParentId, Options) :-
    % Direct parent relationship
    tree_parent(TreeId, ParentId, []).
tree_parent_extended(TreeId, ParentId, Options) :-
    % Parent via alias (if follow_aliases is true)
    option(follow_aliases(true), Options, false),
    pearl_children(ParentTreeId, alias, _, _, _, SeeAlsoUri),
    alias_target(SeeAlsoUri, TreeId),
    ParentId = ParentTreeId.

%% tree_descendant_safe(+TreeId, -Descendant, +Options, +Visited) is nondet.
%%   Cycle-safe descendant enumeration.
tree_descendant_safe(TreeId, Descendant, Options, Visited) :-
    \+ member(TreeId, Visited),
    (   % Direct child
        tree_parent(ChildId, TreeId, Options),
        \+ member(ChildId, Visited),
        (   Descendant = ChildId
        ;   tree_descendant_safe(ChildId, Descendant, Options, [TreeId|Visited])
        )
    ;   % Child via alias (if follow_aliases is true)
        option(follow_aliases(true), Options, false),
        alias_children(TreeId, AliasTargets),
        member(AliasTarget, AliasTargets),
        \+ member(AliasTarget, Visited),
        (   Descendant = AliasTarget
        ;   tree_descendant_safe(AliasTarget, Descendant, Options, [TreeId|Visited])
        )
    ).
