:- module(rust_semantic_playbook, [main/0]).
:- use_module('../src/unifyweaver/targets/rust_target').

main :-
    % Assert to user module so compile_predicate_to_rust (which uses user:clause) can find it
    Term = (index_data(_) :- crawler_run(["data.xml"], 1)),
    assertz(user:Term),

    % Compile the semantic predicate
    compile_predicate_to_rust(user:index_data/1, [], Code),
    write_rust_project(Code, 'output/rust_semantic_test').
