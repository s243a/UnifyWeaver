:- module(test_rust_vector, [main/0]).
:- use_module('../src/unifyweaver/targets/rust_target').

% This predicate tests the full vector search capability
test_vector_search(Query) :-
    % 1. Index data (with embeddings)
    crawler_run(["context/PT/pearltrees_export.rdf"], 1),
    
    % 2. Search
    semantic_search(Query, 5, Results).

main :-
    compile_predicate_to_rust(test_rust_vector:test_vector_search/1, [], Code),
    write_rust_project(Code, 'output/rust_vector_test').
