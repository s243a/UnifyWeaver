%% pearltrees/test_hierarchy.pl - Unit tests for hierarchy predicates
%%
%% Tests for navigation, structural queries, and path operations.
%% Run with: swipl -g "run_tests" -t halt test_hierarchy.pl

:- module(test_pearltrees_hierarchy, []).

:- use_module(library(plunit)).

%% ============================================================================
%% Mock Data
%%
%% Hierarchy structure:
%%
%%   root_1 (depth 0)
%%   ├── science_2 (depth 1)
%%   │   ├── physics_3 (depth 2)
%%   │   │   └── quantum_6 (depth 3)
%%   │   └── chemistry_4 (depth 2)
%%   ├── arts_5 (depth 1)
%%   │   └── music_7 (depth 2)
%%   └── orphan_99 (disconnected - parent doesn't exist)
%%
%% ============================================================================

:- dynamic mock_pearl_trees/5.

setup_hierarchy_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)),

    % Root tree
    assertz(mock_pearl_trees(tree, 'root_1', 'Root', 'uri:root_1', root)),

    % Level 1: Science and Arts under Root
    assertz(mock_pearl_trees(tree, 'science_2', 'Science', 'uri:science_2', 'uri:root_1')),
    assertz(mock_pearl_trees(tree, 'arts_5', 'Arts', 'uri:arts_5', 'uri:root_1')),

    % Level 2: Physics and Chemistry under Science
    assertz(mock_pearl_trees(tree, 'physics_3', 'Physics', 'uri:physics_3', 'uri:science_2')),
    assertz(mock_pearl_trees(tree, 'chemistry_4', 'Chemistry', 'uri:chemistry_4', 'uri:science_2')),

    % Level 2: Music under Arts
    assertz(mock_pearl_trees(tree, 'music_7', 'Music', 'uri:music_7', 'uri:arts_5')),

    % Level 3: Quantum under Physics
    assertz(mock_pearl_trees(tree, 'quantum_6', 'Quantum Mechanics', 'uri:quantum_6', 'uri:physics_3')),

    % Orphan: parent doesn't exist in dataset
    assertz(mock_pearl_trees(tree, 'orphan_99', 'Lost Tree', 'uri:orphan_99', 'uri:nonexistent')).

cleanup_hierarchy_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)).

%% ============================================================================
%% Mock versions of predicates using mock data
%% ============================================================================

%% Mock pearl_trees - redirects to mock data
mock_cluster_to_tree_id(ClusterId, TreeId) :-
    mock_pearl_trees(tree, TreeId, _, ClusterId, _).

%% Mock tree_parent
mock_tree_parent(TreeId, ParentId) :-
    mock_pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    ClusterId \= '',
    mock_cluster_to_tree_id(ClusterId, ParentId).

%% Mock tree_ancestors
mock_tree_ancestors(TreeId, Ancestors) :-
    mock_tree_ancestors_(TreeId, [], Ancestors).

mock_tree_ancestors_(TreeId, Acc, Ancestors) :-
    (   mock_tree_parent(TreeId, ParentId)
    ->  mock_tree_ancestors_(ParentId, [ParentId|Acc], Ancestors)
    ;   Ancestors = Acc
    ).

%% Mock tree_descendants
mock_tree_descendants(TreeId, Descendants) :-
    findall(Desc, mock_tree_descendant_of(TreeId, Desc), Descendants).

mock_tree_descendant_of(TreeId, Descendant) :-
    mock_tree_parent(ChildId, TreeId),
    (Descendant = ChildId ; mock_tree_descendant_of(ChildId, Descendant)).

%% Mock tree_siblings
mock_tree_siblings(TreeId, Siblings) :-
    (   mock_tree_parent(TreeId, ParentId)
    ->  findall(SibId,
                (mock_tree_parent(SibId, ParentId), SibId \= TreeId),
                Siblings)
    ;   Siblings = []
    ).

%% Mock tree_depth
mock_tree_depth(TreeId, Depth) :-
    mock_tree_ancestors(TreeId, Ancestors),
    length(Ancestors, Depth).

%% Mock tree_path
mock_tree_path(TreeId, Path) :-
    mock_tree_ancestors(TreeId, Ancestors),
    append(Ancestors, [TreeId], Path).

%% Mock tree_title
mock_tree_title(TreeId, Title) :-
    mock_pearl_trees(tree, TreeId, Title, _, _).

%% Mock root_tree
mock_root_tree(TreeId) :-
    mock_pearl_trees(tree, TreeId, _, _, ClusterId),
    (ClusterId = root ; ClusterId = '').

%% Mock leaf_tree
mock_leaf_tree(TreeId) :-
    mock_pearl_trees(tree, TreeId, _, _, _),
    \+ mock_tree_parent(_, TreeId).

%% Mock orphan_tree
mock_orphan_tree(TreeId) :-
    mock_pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    ClusterId \= '',
    \+ mock_cluster_to_tree_id(ClusterId, _).

%% Mock subtree_tree
mock_subtree_tree(RootId, TreeId) :-
    mock_pearl_trees(tree, RootId, _, _, _),
    (   TreeId = RootId
    ;   mock_tree_descendant_of(RootId, TreeId)
    ).

%% Mock path operations
mock_path_depth(Path, Depth) :-
    length(Path, Len),
    Depth is Len - 1.

mock_truncate_path(Path, MaxDepth, TruncatedPath) :-
    MaxLen is MaxDepth + 1,
    length(Path, Len),
    (   Len =< MaxLen
    ->  TruncatedPath = Path
    ;   length(TruncatedPath, MaxLen),
        append(TruncatedPath, _, Path)
    ).

mock_common_ancestor(TreeId1, TreeId2, AncestorId) :-
    mock_tree_path(TreeId1, Path1),
    mock_tree_path(TreeId2, Path2),
    mock_common_prefix(Path1, Path2, CommonPath),
    CommonPath \= [],
    last(CommonPath, AncestorId).

mock_common_prefix([H|T1], [H|T2], [H|Common]) :-
    !,
    mock_common_prefix(T1, T2, Common).
mock_common_prefix(_, _, []).

%% Mock hierarchical_title_path
mock_hierarchical_title_path(TreeId, TitlePath) :-
    mock_tree_path(TreeId, PathIds),
    maplist(mock_tree_title, PathIds, TitlePath).

%% Mock format_id_path
mock_format_id_path(PathIds, IdPathLine) :-
    atomic_list_concat(PathIds, '/', IdsJoined),
    atom_concat('/', IdsJoined, IdPathLine).

%% Mock format_title_hierarchy
mock_format_title_hierarchy(Titles, Lines) :-
    mock_format_title_hierarchy_(Titles, 0, Lines).

mock_format_title_hierarchy_([], _, []).
mock_format_title_hierarchy_([Title|Rest], Depth, [Line|Lines]) :-
    IndentCount is Depth * 2,
    length(SpaceList, IndentCount),
    maplist(=(0' ), SpaceList),
    atom_codes(Indent, SpaceList),
    format(atom(Line), '~w- ~w', [Indent, Title]),
    NextDepth is Depth + 1,
    mock_format_title_hierarchy_(Rest, NextDepth, Lines).

%% Mock structural_embedding_input
mock_structural_embedding_input(TreeId, ChildTitle, EmbeddingText) :-
    mock_tree_path(TreeId, PathIds),
    mock_format_id_path(PathIds, IdPathLine),
    mock_hierarchical_title_path(TreeId, PathTitles),
    append(PathTitles, [ChildTitle], FullTitles),
    mock_format_title_hierarchy(FullTitles, TitleLines),
    atomic_list_concat([IdPathLine|TitleLines], '\n', EmbeddingText).

%% Mock Phase 4: Basic Transformations

%% Mock flatten_tree
mock_flatten_tree(TreeId, MaxDepth, FlattenedTrees) :-
    findall(DescId-TruncPath,
            (mock_subtree_tree(TreeId, DescId),
             mock_tree_path(DescId, FullPath),
             mock_truncate_path(FullPath, MaxDepth, TruncPath)),
            FlattenedTrees).

%% Mock prune_tree
mock_prune_tree(TreeId, Criteria, PrunedTrees) :-
    findall(DescId,
            (mock_subtree_tree(TreeId, DescId),
             mock_satisfies_criteria(DescId, Criteria)),
            PrunedTrees).

%% Mock satisfies_criteria
mock_satisfies_criteria(TreeId, max_depth(MaxDepth)) :-
    mock_tree_depth(TreeId, Depth),
    Depth =< MaxDepth.
mock_satisfies_criteria(TreeId, has_children) :-
    mock_tree_parent(_, TreeId),
    !.
mock_satisfies_criteria(TreeId, is_leaf) :-
    mock_leaf_tree(TreeId).
mock_satisfies_criteria(TreeId, exclude_orphans) :-
    \+ mock_orphan_tree(TreeId).

%% Mock trees_at_depth
mock_trees_at_depth(Depth, Trees) :-
    findall(TreeId,
            (mock_pearl_trees(tree, TreeId, _, _, _),
             mock_tree_depth(TreeId, Depth)),
            Trees).

%% Mock trees_by_parent
mock_trees_by_parent(GroupedTrees) :-
    findall(ParentId-TreeId,
            (mock_pearl_trees(tree, TreeId, _, _, _),
             (   mock_tree_parent(TreeId, ParentId)
             ->  true
             ;   ParentId = root
             )),
            Pairs),
    mock_group_pairs_by_key(Pairs, GroupedTrees).

mock_group_pairs_by_key(Pairs, Grouped) :-
    keysort(Pairs, Sorted),
    mock_group_pairs_by_key_(Sorted, Grouped).

mock_group_pairs_by_key_([], []).
mock_group_pairs_by_key_([K-V|Rest], [K-[V|Vs]|Groups]) :-
    mock_same_key(K, Rest, Vs, Remaining),
    mock_group_pairs_by_key_(Remaining, Groups).

mock_same_key(K, [K-V|Rest], [V|Vs], Remaining) :-
    !,
    mock_same_key(K, Rest, Vs, Remaining).
mock_same_key(_, Rest, [], Rest).

%% Mock Phase 5: Advanced Transformations

%% Mock reroot_tree
mock_reroot_tree(NewRootId, TreeIds, RerootedPaths) :-
    mock_tree_path(NewRootId, NewRootPath),
    findall(TreeId-RerootedPath,
            (member(TreeId, TreeIds),
             mock_tree_path(TreeId, OrigPath),
             mock_reroot_path(OrigPath, NewRootPath, RerootedPath)),
            RerootedPaths).

mock_reroot_path(OrigPath, NewRootPath, RerootedPath) :-
    (   append(NewRootPath, Suffix, OrigPath)
    ->  last(NewRootPath, NewRootId),
        append([NewRootId], Suffix, RerootedPath)
    ;   RerootedPath = OrigPath
    ).

%% Mock merge_trees
mock_merge_trees(TreeIdLists, Options, MergedTrees) :-
    append(TreeIdLists, AllTrees),
    (   member(dedup(false), Options)
    ->  MergedTrees = AllTrees
    ;   list_to_set(AllTrees, MergedTrees)
    ).

%% Mock group_by_ancestor
mock_group_by_ancestor(TreeIds, Depth, GroupedTrees) :-
    findall(AncestorId-TreeId,
            (member(TreeId, TreeIds),
             mock_ancestor_at_depth(TreeId, Depth, AncestorId)),
            Pairs),
    mock_group_pairs_by_key(Pairs, GroupedTrees).

mock_ancestor_at_depth(TreeId, Depth, AncestorId) :-
    mock_tree_path(TreeId, Path),
    length(Path, PathLen),
    TargetIdx is Depth + 1,
    (   PathLen >= TargetIdx
    ->  nth1(TargetIdx, Path, AncestorId)
    ;   AncestorId = shallow
    ).

%% ============================================================================
%% Tests: Phase 1 - Navigation Predicates
%% ============================================================================

:- begin_tests(tree_parent, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(root_has_no_parent, [fail]) :-
    mock_tree_parent('root_1', _).

test(science_parent_is_root) :-
    mock_tree_parent('science_2', Parent),
    Parent == 'root_1'.

test(physics_parent_is_science) :-
    mock_tree_parent('physics_3', Parent),
    Parent == 'science_2'.

test(quantum_parent_is_physics) :-
    mock_tree_parent('quantum_6', Parent),
    Parent == 'physics_3'.

test(orphan_has_no_valid_parent, [fail]) :-
    mock_tree_parent('orphan_99', _).

:- end_tests(tree_parent).

:- begin_tests(tree_ancestors, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(root_has_no_ancestors) :-
    mock_tree_ancestors('root_1', Ancestors),
    Ancestors == [].

test(science_ancestors) :-
    mock_tree_ancestors('science_2', Ancestors),
    Ancestors == ['root_1'].

test(physics_ancestors) :-
    mock_tree_ancestors('physics_3', Ancestors),
    Ancestors == ['root_1', 'science_2'].

test(quantum_ancestors) :-
    mock_tree_ancestors('quantum_6', Ancestors),
    Ancestors == ['root_1', 'science_2', 'physics_3'].

test(orphan_has_no_ancestors) :-
    mock_tree_ancestors('orphan_99', Ancestors),
    Ancestors == [].

:- end_tests(tree_ancestors).

:- begin_tests(tree_descendants, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(leaf_has_no_descendants) :-
    mock_tree_descendants('quantum_6', Descendants),
    Descendants == [].

test(physics_descendants) :-
    mock_tree_descendants('physics_3', Descendants),
    Descendants == ['quantum_6'].

test(science_descendants) :-
    mock_tree_descendants('science_2', Descendants),
    msort(Descendants, Sorted),
    Sorted == ['chemistry_4', 'physics_3', 'quantum_6'].

test(root_descendants) :-
    mock_tree_descendants('root_1', Descendants),
    length(Descendants, Count),
    Count == 6.  % science, arts, physics, chemistry, music, quantum

:- end_tests(tree_descendants).

:- begin_tests(tree_siblings, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(root_has_no_siblings) :-
    mock_tree_siblings('root_1', Siblings),
    Siblings == [].

test(science_siblings) :-
    mock_tree_siblings('science_2', Siblings),
    Siblings == ['arts_5'].

test(physics_siblings) :-
    mock_tree_siblings('physics_3', Siblings),
    Siblings == ['chemistry_4'].

test(quantum_has_no_siblings) :-
    mock_tree_siblings('quantum_6', Siblings),
    Siblings == [].

:- end_tests(tree_siblings).

:- begin_tests(tree_depth, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(root_depth_is_zero) :-
    mock_tree_depth('root_1', Depth),
    Depth == 0.

test(science_depth_is_one) :-
    mock_tree_depth('science_2', Depth),
    Depth == 1.

test(physics_depth_is_two) :-
    mock_tree_depth('physics_3', Depth),
    Depth == 2.

test(quantum_depth_is_three) :-
    mock_tree_depth('quantum_6', Depth),
    Depth == 3.

:- end_tests(tree_depth).

:- begin_tests(tree_path, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(root_path) :-
    mock_tree_path('root_1', Path),
    Path == ['root_1'].

test(science_path) :-
    mock_tree_path('science_2', Path),
    Path == ['root_1', 'science_2'].

test(quantum_path) :-
    mock_tree_path('quantum_6', Path),
    Path == ['root_1', 'science_2', 'physics_3', 'quantum_6'].

:- end_tests(tree_path).

%% ============================================================================
%% Tests: Phase 2 - Structural Queries
%% ============================================================================

:- begin_tests(root_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(finds_root) :-
    findall(TreeId, mock_root_tree(TreeId), Roots),
    Roots == ['root_1'].

test(science_is_not_root, [fail]) :-
    mock_root_tree('science_2').

:- end_tests(root_tree).

:- begin_tests(leaf_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(quantum_is_leaf) :-
    mock_leaf_tree('quantum_6').

test(chemistry_is_leaf) :-
    mock_leaf_tree('chemistry_4').

test(music_is_leaf) :-
    mock_leaf_tree('music_7').

test(orphan_is_leaf) :-
    mock_leaf_tree('orphan_99').

test(science_is_not_leaf, [fail]) :-
    mock_leaf_tree('science_2').

test(root_is_not_leaf, [fail]) :-
    mock_leaf_tree('root_1').

:- end_tests(leaf_tree).

:- begin_tests(orphan_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(finds_orphan) :-
    findall(TreeId, mock_orphan_tree(TreeId), Orphans),
    Orphans == ['orphan_99'].

test(root_is_not_orphan, [fail]) :-
    mock_orphan_tree('root_1').

test(science_is_not_orphan, [fail]) :-
    mock_orphan_tree('science_2').

:- end_tests(orphan_tree).

:- begin_tests(subtree_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(tree_is_in_own_subtree, [nondet]) :-
    mock_subtree_tree('science_2', 'science_2').

test(physics_in_science_subtree, [nondet]) :-
    mock_subtree_tree('science_2', 'physics_3').

test(quantum_in_science_subtree, [nondet]) :-
    mock_subtree_tree('science_2', 'quantum_6').

test(music_not_in_science_subtree, [fail]) :-
    mock_subtree_tree('science_2', 'music_7').

test(science_subtree_count) :-
    findall(T, mock_subtree_tree('science_2', T), Trees),
    length(Trees, Count),
    Count == 4.  % science, physics, chemistry, quantum

:- end_tests(subtree_tree).

%% ============================================================================
%% Tests: Phase 3 - Path Operations
%% ============================================================================

:- begin_tests(path_depth, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(single_element_path) :-
    mock_path_depth(['root_1'], Depth),
    Depth == 0.

test(two_element_path) :-
    mock_path_depth(['root_1', 'science_2'], Depth),
    Depth == 1.

test(four_element_path) :-
    mock_path_depth(['root_1', 'science_2', 'physics_3', 'quantum_6'], Depth),
    Depth == 3.

:- end_tests(path_depth).

:- begin_tests(truncate_path, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(truncate_long_path) :-
    mock_truncate_path(['root_1', 'science_2', 'physics_3', 'quantum_6'], 2, Truncated),
    Truncated == ['root_1', 'science_2', 'physics_3'].

test(truncate_short_path_unchanged) :-
    mock_truncate_path(['root_1', 'science_2'], 5, Truncated),
    Truncated == ['root_1', 'science_2'].

test(truncate_to_root_only) :-
    mock_truncate_path(['root_1', 'science_2', 'physics_3'], 0, Truncated),
    Truncated == ['root_1'].

:- end_tests(truncate_path).

:- begin_tests(common_ancestor, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(siblings_common_ancestor) :-
    mock_common_ancestor('physics_3', 'chemistry_4', Ancestor),
    Ancestor == 'science_2'.

test(cousins_common_ancestor) :-
    mock_common_ancestor('quantum_6', 'music_7', Ancestor),
    Ancestor == 'root_1'.

test(parent_child_common_ancestor) :-
    mock_common_ancestor('physics_3', 'quantum_6', Ancestor),
    Ancestor == 'physics_3'.

test(same_tree_common_ancestor) :-
    mock_common_ancestor('physics_3', 'physics_3', Ancestor),
    Ancestor == 'physics_3'.

:- end_tests(common_ancestor).

%% ============================================================================
%% Tests: Embedding Support
%% ============================================================================

:- begin_tests(hierarchical_title_path, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(root_title_path) :-
    mock_hierarchical_title_path('root_1', TitlePath),
    TitlePath == ['Root'].

test(quantum_title_path) :-
    mock_hierarchical_title_path('quantum_6', TitlePath),
    TitlePath == ['Root', 'Science', 'Physics', 'Quantum Mechanics'].

:- end_tests(hierarchical_title_path).

:- begin_tests(format_id_path, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(single_id_path) :-
    mock_format_id_path(['root_1'], IdPath),
    IdPath == '/root_1'.

test(multi_id_path) :-
    mock_format_id_path(['root_1', 'science_2', 'physics_3'], IdPath),
    IdPath == '/root_1/science_2/physics_3'.

:- end_tests(format_id_path).

:- begin_tests(format_title_hierarchy, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(single_title) :-
    mock_format_title_hierarchy(['Root'], Lines),
    Lines == ['- Root'].

test(nested_titles) :-
    mock_format_title_hierarchy(['Root', 'Science', 'Physics'], Lines),
    Lines == ['- Root', '  - Science', '    - Physics'].

:- end_tests(format_title_hierarchy).

:- begin_tests(structural_embedding_input, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(embedding_format) :-
    mock_structural_embedding_input('physics_3', 'Quantum Mechanics', Text),
    Text == '/root_1/science_2/physics_3\n- Root\n  - Science\n    - Physics\n      - Quantum Mechanics'.

test(root_embedding_format) :-
    mock_structural_embedding_input('root_1', 'Top Level Item', Text),
    Text == '/root_1\n- Root\n  - Top Level Item'.

:- end_tests(structural_embedding_input).

%% ============================================================================
%% Tests: Phase 4 - Basic Transformations
%% ============================================================================

:- begin_tests(flatten_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(flatten_root_to_depth_1, [nondet]) :-
    mock_flatten_tree('root_1', 1, Flattened),
    % At depth 1: root_1, science_2, arts_5 are in the result
    member('root_1'-['root_1'], Flattened),
    member('science_2'-['root_1', 'science_2'], Flattened),
    member('arts_5'-['root_1', 'arts_5'], Flattened),
    % Deeper trees get truncated paths
    member('physics_3'-['root_1', 'science_2'], Flattened),
    member('quantum_6'-['root_1', 'science_2'], Flattened).

test(flatten_science_to_depth_2, [nondet]) :-
    mock_flatten_tree('science_2', 2, Flattened),
    length(Flattened, 4),  % science, physics, chemistry, quantum
    member('science_2'-['root_1', 'science_2'], Flattened),
    member('physics_3'-['root_1', 'science_2', 'physics_3'], Flattened),
    member('chemistry_4'-['root_1', 'science_2', 'chemistry_4'], Flattened),
    member('quantum_6'-['root_1', 'science_2', 'physics_3'], Flattened).

test(flatten_leaf_tree) :-
    mock_flatten_tree('quantum_6', 0, Flattened),
    Flattened = ['quantum_6'-['root_1']].

:- end_tests(flatten_tree).

:- begin_tests(prune_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(prune_by_max_depth, [nondet]) :-
    mock_prune_tree('root_1', max_depth(1), Pruned),
    % Only root and depth-1 trees
    member('root_1', Pruned),
    member('science_2', Pruned),
    member('arts_5', Pruned),
    \+ member('physics_3', Pruned),
    \+ member('quantum_6', Pruned).

test(prune_by_has_children, [nondet]) :-
    mock_prune_tree('root_1', has_children, Pruned),
    % Only trees with children: root_1, science_2, arts_5, physics_3
    member('root_1', Pruned),
    member('science_2', Pruned),
    member('arts_5', Pruned),
    member('physics_3', Pruned),
    \+ member('chemistry_4', Pruned),  % leaf
    \+ member('quantum_6', Pruned).    % leaf

test(prune_by_is_leaf, [nondet]) :-
    mock_prune_tree('science_2', is_leaf, Pruned),
    % Only leaves in science subtree: chemistry, quantum
    \+ member('science_2', Pruned),
    \+ member('physics_3', Pruned),
    member('chemistry_4', Pruned),
    member('quantum_6', Pruned).

test(prune_excludes_orphans) :-
    % First check that orphan exists
    mock_orphan_tree('orphan_99'),
    % But prune should exclude it
    mock_prune_tree('root_1', exclude_orphans, Pruned),
    \+ member('orphan_99', Pruned).

:- end_tests(prune_tree).

:- begin_tests(trees_at_depth, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(trees_at_depth_0, [nondet]) :-
    mock_trees_at_depth(0, Trees),
    % root_1 and orphan_99 (orphan has no parent so depth 0)
    length(Trees, 2),
    member('root_1', Trees),
    member('orphan_99', Trees).

test(trees_at_depth_1, [nondet]) :-
    mock_trees_at_depth(1, Trees),
    length(Trees, 2),
    member('science_2', Trees),
    member('arts_5', Trees).

test(trees_at_depth_2, [nondet]) :-
    mock_trees_at_depth(2, Trees),
    length(Trees, 3),
    member('physics_3', Trees),
    member('chemistry_4', Trees),
    member('music_7', Trees).

test(trees_at_depth_3) :-
    mock_trees_at_depth(3, Trees),
    Trees = ['quantum_6'].

test(trees_at_depth_4_empty) :-
    mock_trees_at_depth(4, Trees),
    Trees = [].

:- end_tests(trees_at_depth).

:- begin_tests(trees_by_parent, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(trees_grouped_by_parent, [nondet]) :-
    mock_trees_by_parent(Grouped),
    % Check root group (includes orphan_99 too since its parent doesn't exist)
    member(root-RootChildren, Grouped),
    member('root_1', RootChildren),
    member('orphan_99', RootChildren),
    % Check science group
    member('root_1'-ScienceChildren, Grouped),
    member('science_2', ScienceChildren),
    member('arts_5', ScienceChildren),
    % Check physics group
    member('science_2'-PhysicsChildren, Grouped),
    member('physics_3', PhysicsChildren),
    member('chemistry_4', PhysicsChildren).

test(group_count) :-
    mock_trees_by_parent(Grouped),
    length(Grouped, Count),
    % Groups: root (root_1, orphan_99), root_1, science_2, arts_5, physics_3
    Count == 5.

:- end_tests(trees_by_parent).

%% ============================================================================
%% Tests: Phase 5 - Advanced Transformations
%% ============================================================================

:- begin_tests(reroot_tree, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(reroot_to_science, [nondet]) :-
    % Reroot to science_2, so science subtree paths should start from science
    mock_reroot_tree('science_2',
                     ['science_2', 'physics_3', 'quantum_6', 'arts_5'],
                     Rerooted),
    % science_2 path becomes just [science_2]
    member('science_2'-['science_2'], Rerooted),
    % physics_3 path becomes [science_2, physics_3]
    member('physics_3'-['science_2', 'physics_3'], Rerooted),
    % quantum_6 path becomes [science_2, physics_3, quantum_6]
    member('quantum_6'-['science_2', 'physics_3', 'quantum_6'], Rerooted),
    % arts_5 is not under science_2, keeps original path
    member('arts_5'-['root_1', 'arts_5'], Rerooted).

test(reroot_preserves_non_descendants, [nondet]) :-
    mock_reroot_tree('physics_3', ['root_1', 'arts_5'], Rerooted),
    % Neither is under physics_3, both keep original paths
    member('root_1'-['root_1'], Rerooted),
    member('arts_5'-['root_1', 'arts_5'], Rerooted).

:- end_tests(reroot_tree).

:- begin_tests(merge_trees, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(merge_with_dedup, [nondet]) :-
    mock_merge_trees([['a', 'b'], ['b', 'c'], ['a', 'd']], [], Merged),
    length(Merged, 4),
    member('a', Merged),
    member('b', Merged),
    member('c', Merged),
    member('d', Merged).

test(merge_without_dedup) :-
    mock_merge_trees([['a', 'b'], ['b', 'c']], [dedup(false)], Merged),
    length(Merged, 4),
    Merged = ['a', 'b', 'b', 'c'].

test(merge_empty_lists) :-
    mock_merge_trees([[], []], [], Merged),
    Merged = [].

:- end_tests(merge_trees).

:- begin_tests(group_by_ancestor, [setup(setup_hierarchy_mock_data), cleanup(cleanup_hierarchy_mock_data)]).

test(group_by_depth_1, [nondet]) :-
    % Group all trees by their ancestor at depth 1
    mock_group_by_ancestor(['science_2', 'physics_3', 'chemistry_4', 'quantum_6',
                            'arts_5', 'music_7', 'root_1'],
                           1, Grouped),
    % Trees under science_2 (at depth 1)
    member('science_2'-ScienceGroup, Grouped),
    member('science_2', ScienceGroup),
    member('physics_3', ScienceGroup),
    member('chemistry_4', ScienceGroup),
    member('quantum_6', ScienceGroup),
    % Trees under arts_5 (at depth 1)
    member('arts_5'-ArtsGroup, Grouped),
    member('arts_5', ArtsGroup),
    member('music_7', ArtsGroup),
    % root_1 is at depth 0, so grouped as 'shallow'
    member(shallow-ShallowGroup, Grouped),
    member('root_1', ShallowGroup).

test(group_by_depth_0, [nondet]) :-
    % All trees should be grouped by root
    mock_group_by_ancestor(['root_1', 'science_2', 'physics_3'], 0, Grouped),
    member('root_1'-RootGroup, Grouped),
    member('root_1', RootGroup),
    member('science_2', RootGroup),
    member('physics_3', RootGroup).

test(group_by_depth_2, [nondet]) :-
    mock_group_by_ancestor(['quantum_6', 'chemistry_4', 'music_7', 'science_2'], 2, Grouped),
    % quantum_6 and chemistry_4 are grouped by physics_3 and chemistry_4 respectively
    member('physics_3'-PhysicsGroup, Grouped),
    member('quantum_6', PhysicsGroup),
    member('chemistry_4'-ChemGroup, Grouped),
    member('chemistry_4', ChemGroup),
    member('music_7'-MusicGroup, Grouped),
    member('music_7', MusicGroup),
    % science_2 is at depth 1, so shallow
    member(shallow-ShallowGroup, Grouped),
    member('science_2', ShallowGroup).

:- end_tests(group_by_ancestor).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization((
    setup_hierarchy_mock_data,
    run_tests,
    cleanup_hierarchy_mock_data
), main).
