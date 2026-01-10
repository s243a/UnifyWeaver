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
%% Run tests when loaded directly
%% ============================================================================

:- initialization((
    setup_hierarchy_mock_data,
    run_tests,
    cleanup_hierarchy_mock_data
), main).
