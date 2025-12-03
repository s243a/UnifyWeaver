:- module(rust_semantic_playbook, [main/0]).
:- use_module('../src/unifyweaver/targets/rust_target').

main :-
    Code = 'fn main() { println!("Runtime Test"); }',
    write_rust_project(Code, 'output/rust_semantic_test').
