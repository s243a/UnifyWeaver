:- encoding(utf8).
% Test suite for Rust Target
% Usage: swipl -g run_tests -t halt tests/test_rust_target.pl

:- use_module('../src/unifyweaver/targets/rust_target').

% Schema for JSON tests
:- json_schema(user_schema, [
    field(name, string),
    field(age, integer)
]).

run_tests :-
    test_facts_compilation,
    test_rule_compilation,
    test_aggregation_compilation,
    test_regex_compilation,
    test_json_input_compilation,
    test_json_output_compilation,
    halt.

test_facts_compilation :-
    format('~n--- Testing Facts Compilation ---~n'),
    assertz(test_fact(a, 1)),
    assertz(test_fact(b, 2)),
    
    compile_predicate_to_rust(test_fact/2, [], Code),
    write_rust_program(Code, 'test_fact.rs'),
    
    retractall(test_fact(_, _)),
    format('Generated test_fact.rs~n').

test_rule_compilation :-
    format('~n--- Testing Rule Compilation ---~n'),
    assertz((test_child(C, P) :- test_parent(P, C))),
    
    compile_predicate_to_rust(test_child/2, [field_delimiter(colon)], Code),
    write_rust_program(Code, 'test_child.rs'),
    
    retractall(test_child(_, _)),
    format('Generated test_child.rs~n').

test_aggregation_compilation :-
    format('~n--- Testing Aggregation Compilation ---~n'),
    assertz((test_sum(S) :- aggregation(sum), val(S))),
    
    compile_predicate_to_rust(test_sum/1, [aggregation(sum)], Code),
    write_rust_program(Code, 'test_sum.rs'),
    
    retractall(test_sum(_)),
    format('Generated test_sum.rs~n').

test_regex_compilation :-
    format('~n--- Testing Regex Compilation ---~n'),
    assertz((test_regex(Line) :- test_input(Line), match(Line, "^ERROR"))),
    
    compile_predicate_to_rust(test_regex/1, [field_delimiter(colon)], Code),
    write_rust_project(Code, 'output/test_regex'),
    
    exists_file('output/test_regex/Cargo.toml'),
    retractall(test_regex(_)),
    format('Generated output/test_regex~n').

test_json_input_compilation :-
    format('~n--- Testing JSON Input Compilation ---~n'),
    assertz((user_info(Name, Age) :- json_record([name-Name, age-Age]))),
    
    compile_predicate_to_rust(user_info/2, [
        json_input(true),
        json_schema(user_schema),
        field_delimiter(colon)
    ], Code),
    
    write_rust_project(Code, 'output/test_json_input'),
    
    retractall(user_info(_, _)),
    format('Generated output/test_json_input~n').

test_json_output_compilation :-
    format('~n--- Testing JSON Output Compilation ---~n'),
    assertz((output_user(Name, Age) :- input_data(Name, Age))),
    
    compile_predicate_to_rust(output_user/2, [
        json_output(true),
        field_delimiter(colon)
    ], Code),
    
    write_rust_project(Code, 'output/test_json_output'),
    
    retractall(output_user(_, _)),
    format('Generated output/test_json_output~n').