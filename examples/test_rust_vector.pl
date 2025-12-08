:- module(test_rust_vector, [main/0, main_gpu/0, main_cpu/0]).
:- use_module('../src/unifyweaver/targets/rust_target').

% This predicate tests the full vector search capability
% Device can be: auto, cpu, cuda, gpu, metal
test_vector_search(Query, Device) :-
    % 1. Initialize with specified device
    init_embeddings(Device),

    % 2. Index data (with embeddings)
    crawler_run(["context/PT/pearltrees_export.rdf"], 1),

    % 3. Search
    semantic_search(Query, 5, Results).

% Default: conservative auto-detection (prefers CPU unless explicitly GPU-capable)
main :-
    compile_predicate_to_rust(test_rust_vector:test_vector_search/2, [device(auto)], Code),
    write_rust_project(Code, 'output/rust_vector_test').

% Force GPU mode
main_gpu :-
    compile_predicate_to_rust(test_rust_vector:test_vector_search/2, [device(cuda)], Code),
    write_rust_project(Code, 'output/rust_vector_gpu').

% Force CPU mode (for proot/termux)
main_cpu :-
    compile_predicate_to_rust(test_rust_vector:test_vector_search/2, [device(cpu)], Code),
    write_rust_project(Code, 'output/rust_vector_cpu').
