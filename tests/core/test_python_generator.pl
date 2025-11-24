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
    
    retractall(edge(_,_)),
    !.  % Prevent choicepoints

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
    retractall(path(_,_)),
    !.  % Prevent choicepoints

test(generator_execution, [condition(python_available)]) :-
    % End-to-end test: compile, run, verify output
    % NOTE: This test requires Python 3.7+ in PATH
    % May be skipped in some environments
    
    assertz(edge(a, b)),
    assertz(edge(b, c)),
    assertz((path(X, Y) :- edge(X, Y))),
    assertz((path(X, Z) :- edge(X, Y), path(Y, Z))),
    
    % Compile to Python
    compile_predicate_to_python(path/2, [mode(generator)], Code),
    
    % For now, just verify code generation (full execution test pending)
    % TODO: Add full execution when Python environment is standardized
    sub_string(Code, _, _, _, "FrozenDict"),
    sub_string(Code, _, _, _, "process_stream_generator"),
    
    % Cleanup
    retractall(edge(_,_)),
    retractall(path(_,_)),
    !.
test(disjunction_generator) :-
    % Test disjunction (OR) in generator mode
    assertz(young(alice)),
    assertz(senior(bob)),
    assertz((valid(X) :- (young(X) ; senior(X)))),
    
    compile_predicate_to_python(valid/1, [mode(generator)], Code),
    
    % Should generate disjunctive rule
    sub_string(Code, _, _, _, "Disjunctive rule"),
    % Should have fixpoint loop
    sub_string(Code, _, _, _, "process_stream_generator"),
    % Should generate rule function
    sub_string(Code, _, _, _, "_apply_rule_"),
    
    retractall(young(_)),
    retractall(senior(_)),
    retractall(valid(_)),
    !.

test(complex_disjunction_generator) :-
    % Test disjunction with N-way joins
    % path(X,Z) :- (edge(X,Z) ; (edge(X,A), edge(A,B), edge(B,Z))).
    assertz(edge(1,2)),
    assertz(edge(2,3)),
    assertz(edge(3,4)),
    assertz((path(X,Z) :- (edge(X,Z) ; (edge(X,A), edge(A,B), edge(B,Z))))),
    
    compile_predicate_to_python(path/2, [mode(generator)], Code),
    
    % Should contain nested loops for the 3-way join part
    sub_string(Code, _, _, _, "for join_1 in total"),
    sub_string(Code, _, _, _, "for join_2 in total"),
    
    retractall(edge(_,_)),
    retractall(path(_,_)),
    !.

python_available :-
    catch(process_create(path(python), ['--version'], [stderr(null)]), _, fail).


test(negation_generator) :-
    assertz(p(a)),
    assertz(q(a)),
    assertz(s(b)),
    assertz((r(X) :- p(X), \+ q(X))),
    assertz((t(X) :- s(X), \+ q(X))),
    
    compile_predicate_to_python(r/1, [mode(generator)], CodeR),
    compile_predicate_to_python(t/1, [mode(generator)], CodeT),
    
    % Verify CodeR checks for q
    sub_string(CodeR, _, _, _, "FrozenDict.from_dict"),
    sub_string(CodeR, _, _, _, "'relation': 'q'"),
    sub_string(CodeR, _, _, _, "not in total"),
    
    % Verify CodeT checks for q
    sub_string(CodeT, _, _, _, "'relation': 'q'"),
    sub_string(CodeT, _, _, _, "not in total").

:- end_tests(python_generator).
