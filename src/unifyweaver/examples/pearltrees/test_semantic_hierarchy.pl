%% pearltrees/test_semantic_hierarchy.pl - Unit tests for semantic hierarchy predicates
%%
%% Tests for Phases 7-9: embedding, clustering, and semantic hierarchy.
%% Run with: swipl -g "run_tests" -t halt test_semantic_hierarchy.pl

:- module(test_pearltrees_semantic_hierarchy, []).

:- use_module(library(plunit)).

%% ============================================================================
%% Mock Data
%%
%% Uses same hierarchy as test_hierarchy.pl:
%%
%%   root_1 (depth 0)
%%   ├── science_2 (depth 1)
%%   │   ├── physics_3 (depth 2)
%%   │   │   └── quantum_6 (depth 3)
%%   │   └── chemistry_4 (depth 2)
%%   ├── arts_5 (depth 1)
%%   │   └── music_7 (depth 2)
%%   └── orphan_99 (disconnected)
%%
%% ============================================================================

:- dynamic mock_pearl_trees/5.

setup_semantic_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)),

    % Root tree
    assertz(mock_pearl_trees(tree, 'root_1', 'Root', 'uri:root_1', root)),

    % Level 1: Science and Arts under Root
    assertz(mock_pearl_trees(tree, 'science_2', 'Science Topics', 'uri:science_2', 'uri:root_1')),
    assertz(mock_pearl_trees(tree, 'arts_5', 'Arts and Culture', 'uri:arts_5', 'uri:root_1')),

    % Level 2: Physics and Chemistry under Science
    assertz(mock_pearl_trees(tree, 'physics_3', 'Physics Research', 'uri:physics_3', 'uri:science_2')),
    assertz(mock_pearl_trees(tree, 'chemistry_4', 'Chemistry Notes', 'uri:chemistry_4', 'uri:science_2')),

    % Level 2: Music under Arts
    assertz(mock_pearl_trees(tree, 'music_7', 'Classical Music', 'uri:music_7', 'uri:arts_5')),

    % Level 3: Quantum under Physics
    assertz(mock_pearl_trees(tree, 'quantum_6', 'Quantum Mechanics', 'uri:quantum_6', 'uri:physics_3')),

    % Orphan
    assertz(mock_pearl_trees(tree, 'orphan_99', 'Lost Tree', 'uri:orphan_99', 'uri:nonexistent')).

cleanup_semantic_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)).

%% ============================================================================
%% Mock versions of predicates
%% ============================================================================

%% Mock navigation predicates (from hierarchy.pl)
mock_cluster_to_tree_id(ClusterId, TreeId) :-
    mock_pearl_trees(tree, TreeId, _, ClusterId, _).

mock_tree_parent(TreeId, ParentId) :-
    mock_pearl_trees(tree, TreeId, _, _, ClusterId),
    ClusterId \= root,
    ClusterId \= '',
    mock_cluster_to_tree_id(ClusterId, ParentId).

mock_tree_ancestors(TreeId, Ancestors) :-
    mock_tree_ancestors_(TreeId, [], Ancestors).

mock_tree_ancestors_(TreeId, Acc, Ancestors) :-
    (   mock_tree_parent(TreeId, ParentId)
    ->  mock_tree_ancestors_(ParentId, [ParentId|Acc], Ancestors)
    ;   Ancestors = Acc
    ).

mock_tree_path(TreeId, Path) :-
    mock_tree_ancestors(TreeId, Ancestors),
    append(Ancestors, [TreeId], Path).

mock_tree_title(TreeId, Title) :-
    mock_pearl_trees(tree, TreeId, Title, _, _).

mock_subtree_tree(RootId, TreeId) :-
    mock_pearl_trees(tree, RootId, _, _, _),
    (   TreeId = RootId
    ;   mock_tree_descendant_of(RootId, TreeId)
    ).

mock_tree_descendant_of(TreeId, Descendant) :-
    mock_tree_parent(ChildId, TreeId),
    (Descendant = ChildId ; mock_tree_descendant_of(ChildId, Descendant)).

%% Mock embedding predicates
mock_embedding_input_text(TreeId, Text) :-
    mock_tree_title(TreeId, Title),
    mock_tree_path(TreeId, PathIds),
    mock_format_id_path(PathIds, IdPath),
    mock_hierarchical_title_path(TreeId, TitlePath),
    append(TitlePath, [Title], FullTitles),
    mock_format_title_hierarchy(FullTitles, TitleLines),
    atomic_list_concat([IdPath|TitleLines], '\n', Text).

mock_format_id_path(PathIds, IdPathLine) :-
    atomic_list_concat(PathIds, '/', IdsJoined),
    atom_concat('/', IdsJoined, IdPathLine).

mock_hierarchical_title_path(TreeId, TitlePath) :-
    mock_tree_path(TreeId, PathIds),
    maplist(mock_tree_title, PathIds, TitlePath).

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

%% Mock placeholder embedding (deterministic based on text)
mock_placeholder_embedding(Text, Embedding) :-
    atom_codes(Text, Codes),
    sum_list(Codes, Sum),
    Seed is Sum mod 1000,
    mock_generate_embedding(Seed, 10, Embedding).  % Use 10 dims for testing

mock_generate_embedding(_, 0, []) :- !.
mock_generate_embedding(Seed, N, [Val|Rest]) :-
    N > 0,
    NextSeed is (Seed * 1103515245 + 12345) mod (2^31),
    Val is (NextSeed mod 2000 - 1000) / 1000.0,
    N1 is N - 1,
    mock_generate_embedding(NextSeed, N1, Rest).

mock_tree_embedding(TreeId, Embedding) :-
    mock_embedding_input_text(TreeId, Text),
    mock_placeholder_embedding(Text, Embedding).

%% Mock similarity
mock_cosine_similarity(Vec1, Vec2, Score) :-
    mock_dot_product(Vec1, Vec2, Dot),
    mock_magnitude(Vec1, Mag1),
    mock_magnitude(Vec2, Mag2),
    (   Mag1 > 0, Mag2 > 0
    ->  Score is Dot / (Mag1 * Mag2)
    ;   Score = 0.0
    ).

mock_dot_product([], [], 0.0).
mock_dot_product([A|As], [B|Bs], Result) :-
    mock_dot_product(As, Bs, Rest),
    Result is A * B + Rest.

mock_magnitude(Vec, Mag) :-
    maplist(mock_square, Vec, Squared),
    sum_list(Squared, SumSq),
    Mag is sqrt(SumSq).

mock_square(X, Y) :- Y is X * X.

mock_tree_similarity(TreeId1, TreeId2, Score) :-
    mock_tree_embedding(TreeId1, Emb1),
    mock_tree_embedding(TreeId2, Emb2),
    mock_cosine_similarity(Emb1, Emb2, Score).

%% Mock centroid
mock_compute_centroid([Single], Single) :- !.
mock_compute_centroid(Embeddings, Centroid) :-
    Embeddings = [First|_],
    length(First, Dim),
    length(Zeros, Dim),
    maplist(=(0.0), Zeros),
    mock_sum_embeddings(Embeddings, Zeros, Sums),
    length(Embeddings, Count),
    maplist(mock_divide_by(Count), Sums, Centroid).

mock_sum_embeddings([], Acc, Acc).
mock_sum_embeddings([Emb|Rest], Acc, Result) :-
    maplist(mock_add, Emb, Acc, NewAcc),
    mock_sum_embeddings(Rest, NewAcc, Result).

mock_add(A, B, C) :- C is A + B.
mock_divide_by(N, X, Y) :- Y is X / N.

%% ============================================================================
%% Tests: Phase 7 - Embedding Predicates
%% ============================================================================

:- begin_tests(embedding_input_text, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(root_embedding_text, [nondet]) :-
    mock_embedding_input_text('root_1', Text),
    % Should contain ID path
    sub_atom(Text, _, _, _, '/root_1'),
    % Should contain title
    sub_atom(Text, _, _, _, 'Root').

test(physics_embedding_text, [nondet]) :-
    mock_embedding_input_text('physics_3', Text),
    % Should contain full ID path
    sub_atom(Text, _, _, _, '/root_1/science_2/physics_3'),
    % Should contain title hierarchy
    sub_atom(Text, _, _, _, 'Science Topics'),
    sub_atom(Text, _, _, _, 'Physics Research').

:- end_tests(embedding_input_text).

:- begin_tests(placeholder_embedding, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(embedding_has_correct_dimensions) :-
    mock_placeholder_embedding('test text', Embedding),
    length(Embedding, 10).

test(embedding_is_deterministic) :-
    mock_placeholder_embedding('same text', Emb1),
    mock_placeholder_embedding('same text', Emb2),
    Emb1 == Emb2.

test(different_text_different_embedding) :-
    mock_placeholder_embedding('text one', Emb1),
    mock_placeholder_embedding('text two', Emb2),
    Emb1 \== Emb2.

test(embedding_values_in_range) :-
    mock_placeholder_embedding('test', Embedding),
    forall(member(V, Embedding), (V >= -1.0, V =< 1.0)).

:- end_tests(placeholder_embedding).

:- begin_tests(tree_embedding, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(tree_has_embedding) :-
    mock_tree_embedding('science_2', Embedding),
    length(Embedding, 10).

test(different_trees_different_embeddings) :-
    mock_tree_embedding('science_2', Emb1),
    mock_tree_embedding('arts_5', Emb2),
    Emb1 \== Emb2.

:- end_tests(tree_embedding).

%% ============================================================================
%% Tests: Phase 8 - Clustering Predicates
%% ============================================================================

:- begin_tests(cosine_similarity, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(identical_vectors_similarity_1) :-
    Vec = [1.0, 0.0, 0.0],
    mock_cosine_similarity(Vec, Vec, Score),
    abs(Score - 1.0) < 0.001.

test(orthogonal_vectors_similarity_0) :-
    Vec1 = [1.0, 0.0, 0.0],
    Vec2 = [0.0, 1.0, 0.0],
    mock_cosine_similarity(Vec1, Vec2, Score),
    abs(Score) < 0.001.

test(opposite_vectors_similarity_minus1) :-
    Vec1 = [1.0, 0.0, 0.0],
    Vec2 = [-1.0, 0.0, 0.0],
    mock_cosine_similarity(Vec1, Vec2, Score),
    abs(Score - (-1.0)) < 0.001.

:- end_tests(cosine_similarity).

:- begin_tests(tree_similarity, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(self_similarity_is_1) :-
    mock_tree_similarity('science_2', 'science_2', Score),
    abs(Score - 1.0) < 0.001.

test(similarity_is_symmetric) :-
    mock_tree_similarity('science_2', 'arts_5', Score1),
    mock_tree_similarity('arts_5', 'science_2', Score2),
    abs(Score1 - Score2) < 0.001.

test(similarity_in_valid_range) :-
    mock_tree_similarity('physics_3', 'chemistry_4', Score),
    Score >= -1.0,
    Score =< 1.0.

:- end_tests(tree_similarity).

:- begin_tests(centroid, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(single_embedding_centroid) :-
    Emb = [1.0, 2.0, 3.0],
    mock_compute_centroid([Emb], Centroid),
    Centroid == Emb.

test(two_embedding_centroid) :-
    Emb1 = [0.0, 2.0, 4.0],
    Emb2 = [2.0, 4.0, 6.0],
    mock_compute_centroid([Emb1, Emb2], Centroid),
    Centroid = [C1, C2, C3],
    abs(C1 - 1.0) < 0.001,
    abs(C2 - 3.0) < 0.001,
    abs(C3 - 5.0) < 0.001.

test(centroid_dimensions_preserved) :-
    Emb1 = [1.0, 2.0, 3.0, 4.0, 5.0],
    Emb2 = [5.0, 4.0, 3.0, 2.0, 1.0],
    mock_compute_centroid([Emb1, Emb2], Centroid),
    length(Centroid, 5).

:- end_tests(centroid).

%% ============================================================================
%% Tests: Phase 9 - Semantic Hierarchy (basic tests without full clustering)
%% ============================================================================

:- begin_tests(semantic_structure, [setup(setup_semantic_mock_data), cleanup(cleanup_semantic_mock_data)]).

test(mock_data_has_trees) :-
    findall(T, mock_pearl_trees(tree, T, _, _, _), Trees),
    length(Trees, 8).

test(subtree_of_science, [nondet]) :-
    findall(T, mock_subtree_tree('science_2', T), Subtree),
    member('science_2', Subtree),
    member('physics_3', Subtree),
    member('chemistry_4', Subtree),
    member('quantum_6', Subtree).

:- end_tests(semantic_structure).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization((
    setup_semantic_mock_data,
    run_tests,
    cleanup_semantic_mock_data
), main).
