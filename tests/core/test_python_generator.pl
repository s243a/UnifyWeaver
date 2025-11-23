:- begin_tests(python_generator).

:- use_module('../../src/unifyweaver/targets/python_target').

test(simple_facts_generator) :-
    % Test that facts compile in generator mode
    assertz(edge(a, b)),
    assertz(edge(b, c)),
    
    compile_predicate_to_python(edge/2, [mode(generator)], Code),
    
    % Should generate FrozenDict class
    sub_string(Code, _, _, _, "class FrozenDict"),
    % Should generate fixpoint loop
    sub_string(Code, _, _, _, "process_stream_generator"),
    % Should generate rule functions
    sub_string(Code, _, _, _, "_apply_rule_"),
    
    retractall(edge(_,_)).

test(transitive_closure_generator) :-
    % Classic transitive closure test
    assertz(edge(a, b)),
    assertz(edge(b, c)),
    assertz(edge(c, d)),
    assertz((path(X, Y) :- edge(X, Y))),
    assertz((path(X, Z) :- edge(X, Y), path(Y, Z))),
    
    compile_predicate_to_python(path/2, [mode(generator)], Code),
    
    % Should generate both rules
    sub_string(Code, _, _, _, "_apply_rule_1"),
    sub_string(Code, _, _, _, "_apply_rule_2"),
    % Should have fixpoint iteration
    sub_string(Code, _, _, _, "while delta:"),
    
    retractall(edge(_,_)),
    retractall(path(_,_)).

:- end_tests(python_generator).
