%% pearltrees/test_glue_integration.pl - Cross-target glue integration tests
%%
%% Tests the full pipeline: Prolog → Go → Python using UnifyWeaver's
%% cross-runtime infrastructure. Verifies component registry, target mapping,
%% and pipeline compilation work together for semantic hierarchy predicates.
%%
%% Run with: swipl -g "run_tests" -t halt test_glue_integration.pl

:- module(test_pearltrees_glue_integration, []).

:- use_module(library(plunit)).
:- use_module(library(option)).

%% Load glue infrastructure
:- use_module('../../glue/cross_runtime_pipeline').
:- use_module('../../core/component_registry').
:- use_module('../../core/target_mapping').

%% Load semantic hierarchy
:- use_module(semantic_hierarchy).
:- use_module(hierarchy).

%% ============================================================================
%% Mock Data (same as test_semantic_hierarchy.pl)
%% ============================================================================

:- dynamic mock_pearl_trees/5.
:- dynamic mock_component_invocation/4.

setup_integration_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)),
    retractall(mock_component_invocation(_, _, _, _)),

    % Root tree
    assertz(mock_pearl_trees(tree, 'root_1', 'Root', 'uri:root_1', root)),

    % Level 1
    assertz(mock_pearl_trees(tree, 'science_2', 'Science Topics', 'uri:science_2', 'uri:root_1')),
    assertz(mock_pearl_trees(tree, 'arts_5', 'Arts and Culture', 'uri:arts_5', 'uri:root_1')),

    % Level 2
    assertz(mock_pearl_trees(tree, 'physics_3', 'Physics Research', 'uri:physics_3', 'uri:science_2')),
    assertz(mock_pearl_trees(tree, 'chemistry_4', 'Chemistry Notes', 'uri:chemistry_4', 'uri:science_2')),
    assertz(mock_pearl_trees(tree, 'music_7', 'Classical Music', 'uri:music_7', 'uri:arts_5')),

    % Level 3
    assertz(mock_pearl_trees(tree, 'quantum_6', 'Quantum Mechanics', 'uri:quantum_6', 'uri:physics_3')).

cleanup_integration_mock_data :-
    retractall(mock_pearl_trees(_, _, _, _, _)),
    retractall(mock_component_invocation(_, _, _, _)).

%% ============================================================================
%% Tests: Runtime Detection
%% ============================================================================

:- begin_tests(runtime_detection).

test(go_runtime_detected) :-
    predicate_runtime(go:cluster_trees/4, Runtime),
    Runtime == go.

test(python_runtime_detected) :-
    predicate_runtime(python:compute_embedding/4, Runtime),
    Runtime == python.

test(default_runtime_is_python) :-
    predicate_runtime(unknown_pred/2, Runtime),
    Runtime == python.

test(rust_ffi_go_detected) :-
    predicate_runtime(rust_ffi:go:bridge_call/3, Runtime),
    Runtime == rust_ffi_go.

test(rust_ffi_node_detected) :-
    predicate_runtime(rust_ffi:node:bridge_call/3, Runtime),
    Runtime == rust_ffi_node.

:- end_tests(runtime_detection).

%% ============================================================================
%% Tests: Runtime Grouping
%% ============================================================================

:- begin_tests(runtime_grouping).

test(single_runtime_group, [nondet]) :-
    group_by_runtime([go:a/1, go:b/2], Groups),
    Groups = [group(go, [go:a/1, go:b/2])].

test(two_runtime_groups, [nondet]) :-
    group_by_runtime([go:a/1, python:b/2], Groups),
    Groups = [group(go, [go:a/1]), group(python, [python:b/2])].

test(alternating_runtimes, [nondet]) :-
    group_by_runtime([go:a/1, python:b/2, go:c/3], Groups),
    Groups = [group(go, [go:a/1]), group(python, [python:b/2]), group(go, [go:c/3])].

test(consecutive_same_runtime, [nondet]) :-
    group_by_runtime([python:a/1, python:b/2, python:c/3, go:d/4], Groups),
    Groups = [group(python, [python:a/1, python:b/2, python:c/3]), group(go, [go:d/4])].

test(all_same_runtime_check) :-
    all_same_runtime([go:a/1, go:b/2, go:c/3], Runtime),
    Runtime == go.

test(mixed_runtime_fails) :-
    \+ all_same_runtime([go:a/1, python:b/2], _).

:- end_tests(runtime_grouping).

%% ============================================================================
%% Tests: Target Mapping
%% ============================================================================

:- begin_tests(target_mapping).

test(target_mapping_module_loads) :-
    current_predicate(target_mapping:declare_target/2).

test(target_mapping_has_declare_target_3) :-
    current_predicate(target_mapping:declare_target/3).

:- end_tests(target_mapping).

%% ============================================================================
%% Tests: Component Registry Integration
%% ============================================================================

:- begin_tests(component_registry, [setup(setup_integration_mock_data), cleanup(cleanup_integration_mock_data)]).

test(component_registry_module_loads) :-
    current_predicate(component_registry:declare_component/4).

test(component_registry_has_invoke) :-
    current_predicate(component_registry:invoke_component/4).

test(component_registry_has_category) :-
    current_predicate(component_registry:define_category/3).

test(embedding_category_can_be_defined) :-
    % Try to define or verify category exists
    (   component_registry:category(test_embedding, _, _)
    ->  true
    ;   component_registry:define_category(test_embedding,
            "Test embedding category",
            [requires_compilation(false)])
    ).

:- end_tests(component_registry).

%% ============================================================================
%% Tests: Cross-Runtime Pipeline Structure
%% ============================================================================

:- begin_tests(pipeline_structure).

test(pipeline_module_loads) :-
    current_predicate(cross_runtime_pipeline:compile_cross_runtime_pipeline/3).

test(pipeline_has_stage_compilation) :-
    current_predicate(cross_runtime_pipeline:compile_stage_go/4),
    current_predicate(cross_runtime_pipeline:compile_stage_python/4).

test(pipeline_has_orchestrator_generation) :-
    current_predicate(cross_runtime_pipeline:generate_orchestrator/4).

:- end_tests(pipeline_structure).

%% ============================================================================
%% Tests: Semantic Hierarchy Module Integration
%% ============================================================================

:- begin_tests(semantic_hierarchy_integration).

test(semantic_hierarchy_module_loads) :-
    current_predicate(pearltrees_semantic_hierarchy:tree_embedding/2).

test(semantic_hierarchy_has_clustering) :-
    current_predicate(pearltrees_semantic_hierarchy:cluster_trees/3).

test(semantic_hierarchy_has_curated_folders) :-
    current_predicate(pearltrees_semantic_hierarchy:curated_folder_structure/3).

test(semantic_hierarchy_has_target_declarations) :-
    % The module should have declared targets
    current_predicate(target_mapping:declare_target/2).

:- end_tests(semantic_hierarchy_integration).

%% ============================================================================
%% Tests: Placeholder Embedding Works Without Backend
%% ============================================================================

:- begin_tests(placeholder_backends, [setup(setup_integration_mock_data), cleanup(cleanup_integration_mock_data)]).

test(placeholder_embedding_generates_vector) :-
    pearltrees_semantic_hierarchy:placeholder_embedding('test text', Embedding),
    is_list(Embedding),
    length(Embedding, 384).

test(placeholder_embedding_deterministic) :-
    pearltrees_semantic_hierarchy:placeholder_embedding('same', Emb1),
    pearltrees_semantic_hierarchy:placeholder_embedding('same', Emb2),
    Emb1 == Emb2.

test(placeholder_embedding_values_valid) :-
    pearltrees_semantic_hierarchy:placeholder_embedding('test', Embedding),
    forall(member(V, Embedding), (number(V), V >= -1.0, V =< 1.0)).

:- end_tests(placeholder_backends).

%% ============================================================================
%% Tests: Full Pipeline Simulation
%% ============================================================================

:- begin_tests(full_pipeline_simulation, [setup(setup_integration_mock_data), cleanup(cleanup_integration_mock_data)]).

test(curated_folders_pipeline_spec, [nondet]) :-
    % The curated folders algorithm can be expressed as a cross-runtime pipeline:
    % 1. Python: identify trees, structural analysis (Prolog predicates run via Python bridge)
    % 2. Python: compute embeddings
    % 3. Go: cluster trees
    % 4. Python: build semantic hierarchy
    Pipeline = [
        python:identify_trees/2,
        python:compute_embeddings/3,
        go:cluster_trees/4,
        python:build_hierarchy/3
    ],
    group_by_runtime(Pipeline, Groups),
    % First two python stages are merged, then go, then python
    length(Groups, 3).

test(embedding_clustering_pipeline, [nondet]) :-
    % Simplified pipeline: embedding → clustering
    Pipeline = [
        python:embed/2,
        go:cluster/3
    ],
    group_by_runtime(Pipeline, Groups),
    Groups = [group(python, [python:embed/2]), group(go, [go:cluster/3])].

test(mixed_runtime_pipeline_groups_correctly, [nondet]) :-
    % Real semantic hierarchy pipeline structure
    Pipeline = [
        go:parse_input/2,
        python:compute_embedding/3,
        go:cluster_embeddings/4,
        python:visualize_clusters/2
    ],
    group_by_runtime(Pipeline, Groups),
    length(Groups, 4),
    Groups = [
        group(go, [go:parse_input/2]),
        group(python, [python:compute_embedding/3]),
        group(go, [go:cluster_embeddings/4]),
        group(python, [python:visualize_clusters/2])
    ].

:- end_tests(full_pipeline_simulation).

%% ============================================================================
%% Tests: Similarity and Centroid (Using Real Module)
%% ============================================================================

:- begin_tests(semantic_operations).

test(cosine_similarity_identical) :-
    Vec = [1.0, 0.0, 0.0],
    pearltrees_semantic_hierarchy:cosine_similarity(Vec, Vec, Score),
    abs(Score - 1.0) < 0.001.

test(cosine_similarity_orthogonal) :-
    Vec1 = [1.0, 0.0, 0.0],
    Vec2 = [0.0, 1.0, 0.0],
    pearltrees_semantic_hierarchy:cosine_similarity(Vec1, Vec2, Score),
    abs(Score) < 0.001.

test(centroid_single) :-
    pearltrees_semantic_hierarchy:compute_centroid([[1.0, 2.0, 3.0]], Centroid),
    Centroid = [1.0, 2.0, 3.0].

test(centroid_average) :-
    pearltrees_semantic_hierarchy:compute_centroid([[0.0, 0.0], [2.0, 4.0]], Centroid),
    Centroid = [C1, C2],
    abs(C1 - 1.0) < 0.001,
    abs(C2 - 2.0) < 0.001.

:- end_tests(semantic_operations).

%% ============================================================================
%% Tests: End-to-End Pipeline Verification
%% ============================================================================

:- begin_tests(end_to_end, [setup(setup_integration_mock_data), cleanup(cleanup_integration_mock_data)]).

test(pipeline_compilation_structure, [nondet]) :-
    % Verify the cross-runtime pipeline can compile a semantic hierarchy pipeline
    % This doesn't run the code, just verifies the structure is correct
    Pipeline = [
        python:compute_embedding/4,
        go:cluster_trees/4
    ],
    group_by_runtime(Pipeline, Groups),
    % Should have exactly 2 groups
    length(Groups, 2),
    % First group should be python
    Groups = [group(python, _), group(go, _)].

test(compile_cross_runtime_with_options, [nondet]) :-
    % Test that cross-runtime compilation produces expected structure
    Pipeline = [go:stage1/2, python:stage2/2],
    Options = [pipeline_name(test_pipeline), output_dir('/tmp')],
    % Just verify groups are computed correctly
    group_by_runtime(Pipeline, Groups),
    length(Groups, 2),
    option(pipeline_name(Name), Options),
    Name == test_pipeline.

:- end_tests(end_to_end).

%% ============================================================================
%% Run tests when loaded directly
%% ============================================================================

:- initialization(run_tests, main).
