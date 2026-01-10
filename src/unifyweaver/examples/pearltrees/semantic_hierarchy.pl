%% pearltrees/semantic_hierarchy.pl - Semantic hierarchy transformations
%%
%% Phases 7-9: Semantic integration for Pearltrees hierarchies.
%% Builds on hierarchy.pl structural predicates and integrates with
%% UnifyWeaver's component registry, bindings, and cross-target glue.
%%
%% See docs/proposals/hierarchical_transformations_specification.md for spec.

:- module(pearltrees_semantic_hierarchy, [
    %% Phase 7: Embedding Predicates
    tree_embedding/2,           % tree_embedding(+TreeId, -Embedding)
    tree_embedding/3,           % tree_embedding(+TreeId, +Options, -Embedding)
    child_embedding/2,          % child_embedding(+ChildInfo, -Embedding)
    tree_centroid/2,            % tree_centroid(+TreeId, -Centroid)
    tree_centroid/3,            % tree_centroid(+TreeId, +Options, -Centroid)
    embedding_input_text/2,     % embedding_input_text(+TreeId, -Text)

    %% Phase 8: Clustering Predicates
    tree_similarity/3,          % tree_similarity(+TreeId1, +TreeId2, -Score)
    most_similar_trees/3,       % most_similar_trees(+TreeId, +K, -SimilarTrees)
    cluster_trees/3,            % cluster_trees(+TreeIds, +K, -Clusters)
    cluster_trees/4,            % cluster_trees(+TreeIds, +K, +Options, -Clusters)

    %% Phase 9: Semantic Hierarchy
    semantic_group/3,           % semantic_group(+TreeId, +Options, -GroupId)
    build_semantic_hierarchy/3, % build_semantic_hierarchy(+TreeIds, +Options, -Hierarchy)
    curated_folder_structure/3  % curated_folder_structure(+TreeIds, +Options, -Folders)
]).

:- use_module(hierarchy).
:- use_module(sources).

%% For component registration
:- use_module('../../core/component_registry', [
    declare_component/4,
    invoke_component/4,
    define_category/3,
    register_component_type/4
]).

%% For target mapping
:- use_module('../../core/target_mapping', [
    declare_target/2,
    declare_target/3
]).

%% ============================================================================
%% Component Registration
%% ============================================================================

%% Define the embedding category if not already defined
:- initialization((
    (   component_registry:category(embedding, _, _)
    ->  true
    ;   component_registry:define_category(embedding,
            "Embedding providers for semantic search",
            [requires_compilation(false)])
    ),
    (   component_registry:category(clustering, _, _)
    ->  true
    ;   component_registry:define_category(clustering,
            "Clustering algorithms for tree organization",
            [requires_compilation(false)])
    )
), now).

%% ============================================================================
%% Phase 7: Embedding Predicates
%% ============================================================================

%% tree_embedding(+TreeId, -Embedding) is det.
%% tree_embedding(+TreeId, +Options, -Embedding) is det.
%%   Get the embedding vector for a tree's content.
%%   Uses the structural embedding format from hierarchy.pl.
%%
%%   Options:
%%     - backend(Backend): python_onnx, go_service, rust_candle (default: python_onnx)
%%     - model(Model): Embedding model name (default: 'all-MiniLM-L6-v2')
%%     - include_children(Bool): Include child content (default: true)
%%
tree_embedding(TreeId, Embedding) :-
    tree_embedding(TreeId, [], Embedding).

tree_embedding(TreeId, Options, Embedding) :-
    % Get text representation for embedding
    embedding_input_text(TreeId, Text),
    % Get backend from options or default
    option(backend(Backend), Options, python_onnx),
    option(model(Model), Options, 'all-MiniLM-L6-v2'),
    % Compute embedding via registered component or binding
    compute_embedding(Backend, Model, Text, Embedding).

%% embedding_input_text(+TreeId, -Text) is det.
%%   Generate the text representation used for embedding.
%%   Uses the structural embedding format: ID path + title hierarchy.
%%
embedding_input_text(TreeId, Text) :-
    tree_title(TreeId, Title),
    structural_embedding_input(TreeId, Title, Text).

%% child_embedding(+ChildInfo, -Embedding) is det.
%%   Get embedding for a child item.
%%   ChildInfo = child(Type, Title, Url, Order)
%%
child_embedding(child(_Type, Title, Url, _Order), Embedding) :-
    % Combine title and URL for embedding
    (   Url \= null, Url \= ''
    ->  format(atom(Text), '~w ~w', [Title, Url])
    ;   Text = Title
    ),
    compute_embedding(python_onnx, 'all-MiniLM-L6-v2', Text, Embedding).

%% tree_centroid(+TreeId, -Centroid) is det.
%% tree_centroid(+TreeId, +Options, -Centroid) is det.
%%   Compute the centroid embedding from a tree and its descendants.
%%
%%   Options:
%%     - include_descendants(Bool): Include all descendants (default: true)
%%     - weight_by_depth(Bool): Weight by inverse depth (default: false)
%%
tree_centroid(TreeId, Centroid) :-
    tree_centroid(TreeId, [], Centroid).

tree_centroid(TreeId, Options, Centroid) :-
    option(include_descendants(IncludeDesc), Options, true),
    % Get embeddings for tree and optionally descendants
    (   IncludeDesc == true
    ->  findall(Emb,
                (subtree_tree(TreeId, SubTreeId),
                 tree_embedding(SubTreeId, Options, Emb)),
                Embeddings)
    ;   tree_embedding(TreeId, Options, Emb),
        Embeddings = [Emb]
    ),
    % Compute mean (centroid)
    (   Embeddings = []
    ->  Centroid = []
    ;   compute_centroid(Embeddings, Centroid)
    ).

%% compute_embedding(+Backend, +Model, +Text, -Embedding) is det.
%%   Backend dispatch for embedding computation.
%%   This is the integration point with UnifyWeaver's component/target system.
%%
compute_embedding(python_onnx, Model, Text, Embedding) :-
    !,
    % Try component registry first
    (   component_registry:component(embedding, onnx_embedder, _, _)
    ->  component_registry:invoke_component(embedding, onnx_embedder,
            embed_request{text: Text, model: Model}, Embedding)
    ;   % Fallback: placeholder for direct binding call
        % In production, this would call the Python ONNX backend
        placeholder_embedding(Text, Embedding)
    ).

compute_embedding(go_service, Model, Text, Embedding) :-
    !,
    (   component_registry:component(embedding, go_embedder, _, _)
    ->  component_registry:invoke_component(embedding, go_embedder,
            embed_request{text: Text, model: Model}, Embedding)
    ;   placeholder_embedding(Text, Embedding)
    ).

compute_embedding(rust_candle, Model, Text, Embedding) :-
    !,
    (   component_registry:component(embedding, rust_embedder, _, _)
    ->  component_registry:invoke_component(embedding, rust_embedder,
            embed_request{text: Text, model: Model}, Embedding)
    ;   placeholder_embedding(Text, Embedding)
    ).

compute_embedding(_, _, Text, Embedding) :-
    % Default fallback
    placeholder_embedding(Text, Embedding).

%% placeholder_embedding(+Text, -Embedding) is det.
%%   Placeholder that generates a deterministic pseudo-embedding.
%%   Used for testing when no real backend is available.
%%   Returns a 384-dimensional vector based on text hash.
%%
placeholder_embedding(Text, Embedding) :-
    atom_codes(Text, Codes),
    sum_list(Codes, Sum),
    % Generate deterministic pseudo-random embedding
    Seed is Sum mod 1000,
    length(Embedding, 384),
    generate_pseudo_embedding(Seed, 384, Embedding).

generate_pseudo_embedding(_, 0, []) :- !.
generate_pseudo_embedding(Seed, N, [Val|Rest]) :-
    N > 0,
    % Simple linear congruential generator
    NextSeed is (Seed * 1103515245 + 12345) mod (2^31),
    Val is (NextSeed mod 2000 - 1000) / 1000.0,  % Range [-1, 1]
    N1 is N - 1,
    generate_pseudo_embedding(NextSeed, N1, Rest).

%% compute_centroid(+Embeddings, -Centroid) is det.
%%   Compute the mean of a list of embeddings.
%%
compute_centroid([Single], Single) :- !.
compute_centroid(Embeddings, Centroid) :-
    Embeddings = [First|_],
    length(First, Dim),
    length(Zeros, Dim),
    maplist(=(0.0), Zeros),
    sum_embeddings(Embeddings, Zeros, Sums),
    length(Embeddings, Count),
    maplist(divide_by(Count), Sums, Centroid).

sum_embeddings([], Acc, Acc).
sum_embeddings([Emb|Rest], Acc, Result) :-
    maplist(add, Emb, Acc, NewAcc),
    sum_embeddings(Rest, NewAcc, Result).

add(A, B, C) :- C is A + B.
divide_by(N, X, Y) :- Y is X / N.

%% ============================================================================
%% Phase 8: Clustering Predicates
%% ============================================================================

%% tree_similarity(+TreeId1, +TreeId2, -Score) is det.
%%   Compute cosine similarity between two trees.
%%   Score is in range [-1, 1], where 1 means identical.
%%
tree_similarity(TreeId1, TreeId2, Score) :-
    tree_embedding(TreeId1, Emb1),
    tree_embedding(TreeId2, Emb2),
    cosine_similarity(Emb1, Emb2, Score).

%% cosine_similarity(+Vec1, +Vec2, -Score) is det.
%%   Compute cosine similarity between two vectors.
%%
cosine_similarity(Vec1, Vec2, Score) :-
    dot_product(Vec1, Vec2, Dot),
    magnitude(Vec1, Mag1),
    magnitude(Vec2, Mag2),
    (   Mag1 > 0, Mag2 > 0
    ->  Score is Dot / (Mag1 * Mag2)
    ;   Score = 0.0
    ).

dot_product([], [], 0.0).
dot_product([A|As], [B|Bs], Result) :-
    dot_product(As, Bs, Rest),
    Result is A * B + Rest.

magnitude(Vec, Mag) :-
    maplist(square, Vec, Squared),
    sum_list(Squared, SumSq),
    Mag is sqrt(SumSq).

square(X, Y) :- Y is X * X.

%% most_similar_trees(+TreeId, +K, -SimilarTrees) is det.
%%   Find K most similar trees to TreeId.
%%   Returns list of similar(TreeId, Score) terms, sorted by descending score.
%%
most_similar_trees(TreeId, K, SimilarTrees) :-
    % Get all other trees
    findall(OtherId,
            (pearl_trees(tree, OtherId, _, _, _), OtherId \= TreeId),
            OtherIds),
    % Compute similarities
    findall(similar(OtherId, Score),
            (member(OtherId, OtherIds),
             tree_similarity(TreeId, OtherId, Score)),
            AllSimilar),
    % Sort by descending score
    sort(2, @>=, AllSimilar, Sorted),
    % Take top K
    take(K, Sorted, SimilarTrees).

take(_, [], []) :- !.
take(0, _, []) :- !.
take(K, [H|T], [H|Rest]) :-
    K > 0,
    K1 is K - 1,
    take(K1, T, Rest).

%% cluster_trees(+TreeIds, +K, -Clusters) is det.
%% cluster_trees(+TreeIds, +K, +Options, -Clusters) is det.
%%   Cluster trees into K groups using k-means.
%%   Returns list of cluster(ClusterId, CentroidTreeId, MemberTreeIds) terms.
%%
%%   Options:
%%     - max_iterations(N): Maximum k-means iterations (default: 100)
%%     - convergence_threshold(T): Stop when centroids move less than T (default: 0.001)
%%
cluster_trees(TreeIds, K, Clusters) :-
    cluster_trees(TreeIds, K, [], Clusters).

cluster_trees(TreeIds, K, Options, Clusters) :-
    option(max_iterations(MaxIter), Options, 100),
    option(convergence_threshold(Threshold), Options, 0.001),
    % Get embeddings for all trees
    findall(TreeId-Embedding,
            (member(TreeId, TreeIds), tree_embedding(TreeId, Embedding)),
            TreeEmbeddings),
    % Initialize centroids (take first K trees)
    take(K, TreeEmbeddings, InitialCentroids),
    maplist(arg(2), InitialCentroids, InitCentroidVecs),
    % Run k-means
    kmeans_iterate(TreeEmbeddings, InitCentroidVecs, MaxIter, Threshold, FinalCentroids, Assignments),
    % Format output
    format_clusters(Assignments, FinalCentroids, Clusters).

%% kmeans_iterate(+TreeEmbeddings, +Centroids, +MaxIter, +Threshold, -FinalCentroids, -Assignments)
kmeans_iterate(_, Centroids, 0, _, Centroids, []) :- !.
kmeans_iterate(TreeEmbeddings, Centroids, Iter, Threshold, FinalCentroids, Assignments) :-
    Iter > 0,
    % Assign each tree to nearest centroid
    assign_to_clusters(TreeEmbeddings, Centroids, NewAssignments),
    % Recompute centroids
    recompute_centroids(NewAssignments, Centroids, NewCentroids),
    % Check convergence
    centroid_movement(Centroids, NewCentroids, Movement),
    (   Movement < Threshold
    ->  FinalCentroids = NewCentroids,
        Assignments = NewAssignments
    ;   Iter1 is Iter - 1,
        kmeans_iterate(TreeEmbeddings, NewCentroids, Iter1, Threshold, FinalCentroids, Assignments)
    ).

assign_to_clusters([], _, []).
assign_to_clusters([TreeId-Embedding|Rest], Centroids, [assignment(TreeId, ClusterIdx)|RestAssign]) :-
    find_nearest_centroid(Embedding, Centroids, 0, -1, 999999, ClusterIdx),
    assign_to_clusters(Rest, Centroids, RestAssign).

find_nearest_centroid(_, [], _, BestIdx, _, BestIdx).
find_nearest_centroid(Embedding, [Centroid|Rest], Idx, BestIdx, BestDist, ResultIdx) :-
    euclidean_distance(Embedding, Centroid, Dist),
    (   Dist < BestDist
    ->  NewBestIdx = Idx, NewBestDist = Dist
    ;   NewBestIdx = BestIdx, NewBestDist = BestDist
    ),
    NextIdx is Idx + 1,
    find_nearest_centroid(Embedding, Rest, NextIdx, NewBestIdx, NewBestDist, ResultIdx).

euclidean_distance(Vec1, Vec2, Dist) :-
    maplist(diff_squared, Vec1, Vec2, Diffs),
    sum_list(Diffs, SumSq),
    Dist is sqrt(SumSq).

diff_squared(A, B, D) :- D is (A - B) * (A - B).

recompute_centroids(Assignments, OldCentroids, NewCentroids) :-
    length(OldCentroids, K),
    findall(NewCentroid,
            (between(0, K1, Idx), K1 is K - 1,
             cluster_members(Idx, Assignments, Members),
             (   Members = []
             ->  nth0(Idx, OldCentroids, NewCentroid)
             ;   compute_centroid(Members, NewCentroid)
             )),
            NewCentroids).

cluster_members(Idx, Assignments, Embeddings) :-
    findall(Emb,
            (member(assignment(TreeId, Idx), Assignments),
             tree_embedding(TreeId, Emb)),
            Embeddings).

centroid_movement([], [], 0.0).
centroid_movement([C1|Rest1], [C2|Rest2], TotalMovement) :-
    euclidean_distance(C1, C2, Dist),
    centroid_movement(Rest1, Rest2, RestMovement),
    TotalMovement is Dist + RestMovement.

format_clusters(Assignments, Centroids, Clusters) :-
    length(Centroids, K),
    findall(cluster(Idx, Members),
            (between(0, K1, Idx), K1 is K - 1,
             findall(TreeId,
                     member(assignment(TreeId, Idx), Assignments),
                     Members)),
            Clusters).

%% ============================================================================
%% Phase 9: Semantic Hierarchy
%% ============================================================================

%% semantic_group(+TreeId, +Options, -GroupId) is det.
%%   Assign a tree to a semantic group based on its embedding.
%%
%%   Options:
%%     - num_groups(N): Number of groups (default: 10)
%%     - method(Method): Grouping method - kmeans, hierarchical (default: kmeans)
%%
semantic_group(TreeId, Options, GroupId) :-
    option(num_groups(NumGroups), Options, 10),
    % Get all trees
    findall(TId, pearl_trees(tree, TId, _, _, _), AllTreeIds),
    % Cluster all trees
    cluster_trees(AllTreeIds, NumGroups, Clusters),
    % Find which cluster this tree belongs to
    member(cluster(GroupId, Members), Clusters),
    member(TreeId, Members),
    !.

%% build_semantic_hierarchy(+TreeIds, +Options, -Hierarchy) is det.
%%   Build a semantic hierarchy using embeddings and clustering.
%%   This is the UnifyWeaver equivalent of the Python curated folders algorithm.
%%
%%   Options:
%%     - num_groups(N): Number of top-level groups
%%     - max_depth(D): Maximum hierarchy depth
%%     - min_cluster_size(S): Minimum members per cluster
%%
%%   Hierarchy = hierarchy(Groups) where Groups is list of:
%%     group(GroupId, CentroidTreeId, SubGroups | Members)
%%
build_semantic_hierarchy(TreeIds, Options, Hierarchy) :-
    option(num_groups(NumGroups), Options, 5),
    option(max_depth(MaxDepth), Options, 2),
    option(min_cluster_size(MinSize), Options, 3),
    % First level clustering
    cluster_trees(TreeIds, NumGroups, Options, TopClusters),
    % Recursively build sub-hierarchies
    build_hierarchy_levels(TopClusters, MaxDepth, MinSize, Options, Groups),
    Hierarchy = hierarchy(Groups).

build_hierarchy_levels([], _, _, _, []).
build_hierarchy_levels([cluster(Idx, Members)|Rest], Depth, MinSize, Options, [Group|RestGroups]) :-
    length(Members, Count),
    (   Depth > 1, Count >= MinSize * 2
    ->  % Can subdivide
        SubK is max(2, Count // MinSize),
        cluster_trees(Members, SubK, Options, SubClusters),
        NewDepth is Depth - 1,
        build_hierarchy_levels(SubClusters, NewDepth, MinSize, Options, SubGroups),
        Group = group(Idx, SubGroups)
    ;   % Leaf cluster
        Group = group(Idx, Members)
    ),
    build_hierarchy_levels(Rest, Depth, MinSize, Options, RestGroups).

%% curated_folder_structure(+TreeIds, +Options, -Folders) is det.
%%   Generate a curated folder structure for the given trees.
%%   This is the main entry point for the full curated folders algorithm.
%%
%%   Options:
%%     - max_folder_depth(D): Maximum folder path depth (default: 3)
%%     - num_top_folders(N): Number of top-level folders (default: 10)
%%
%%   Folders is a list of folder_assignment(TreeId, FolderPath) terms.
%%
curated_folder_structure(TreeIds, Options, Folders) :-
    option(max_folder_depth(MaxDepth), Options, 3),
    option(num_top_folders(NumFolders), Options, 10),
    % Build semantic hierarchy
    build_semantic_hierarchy(TreeIds,
        [num_groups(NumFolders), max_depth(MaxDepth)],
        hierarchy(Groups)),
    % Convert hierarchy to folder assignments
    hierarchy_to_folders(Groups, [], Folders).

hierarchy_to_folders([], _, []).
hierarchy_to_folders([group(Idx, Content)|Rest], PathPrefix, AllFolders) :-
    atom_concat('folder_', Idx, FolderName),
    append(PathPrefix, [FolderName], CurrentPath),
    (   is_list(Content), Content = [group(_,_)|_]
    ->  % Has sub-groups
        hierarchy_to_folders(Content, CurrentPath, SubFolders)
    ;   is_list(Content)
    ->  % Leaf with member list
        findall(folder_assignment(TreeId, CurrentPath),
                member(TreeId, Content),
                SubFolders)
    ;   SubFolders = []
    ),
    hierarchy_to_folders(Rest, PathPrefix, RestFolders),
    append(SubFolders, RestFolders, AllFolders).

%% ============================================================================
%% Target Declarations
%% ============================================================================

%% Declare which targets handle which predicates
:- initialization((
    % Phase 7: Embeddings can use any backend
    declare_target(compute_embedding/4, python, [file('embed.py')]),

    % Phase 8: Clustering uses Go for performance
    declare_target(cluster_trees/4, go, [file('cluster.go')]),

    % Phase 9: Hierarchy building uses Prolog (pure logic)
    declare_target(build_semantic_hierarchy/3, prolog, [])
), now).
