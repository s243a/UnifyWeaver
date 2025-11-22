:- module(test_python_target, [test_python_target/0]).
:- use_module(library(plunit)).
% We will create this module next
:- use_module(unifyweaver(targets/python_target)).

test_python_target :-
    run_tests([python_target]).

:- begin_tests(python_target).

test(module_exports) :-
    current_predicate(python_target:compile_predicate_to_python/3).

test(generates_python_structure) :-
    % Mock compilation (since we haven't implemented the full transpiler yet)
    % We expect the output to contain the main loop and helpers
    assertz(dummy(_)),
    compile_predicate_to_python(dummy/1, [], Code),
    retract(dummy(_)),
    sub_string(Code, _, _, _, "import sys"),
    sub_string(Code, _, _, _, "import json"),
    sub_string(Code, _, _, _, "def main():"),
    sub_string(Code, _, _, _, "read_jsonl(sys.stdin)").

test(compile_filter_logic) :-
    % Define a filter predicate: filter_high(R) :- R.value > 100.
    % Using get_dict/3 which is the standard way to access dicts in SWI-Prolog
    assertz((filter_high(R) :- get_dict(value, R, V), V > 100)),
    
    compile_predicate_to_python(filter_high/1, [], Code),
    
    % Check for translated logic
    % We expect something like: if record['value'] > 100: yield record
    sub_string(Code, _, _, _, "if not (v_"), % Check for the condition structure
    sub_string(Code, _, _, _, "yield v_0"),
    
    sub_string(Code, _, _, _, "yield v_0"),
    
    retract((filter_high(R) :- get_dict(value, R, V), V > 100)).

test(compile_projection) :-
    % project_fields(R, Out) :- get_dict(name, R, N), Out = _{name: N}.
    assertz((project_fields(R, Out) :- 
        get_dict(name, R, N), 
        Out = _{name: N}
    )),
    
    compile_predicate_to_python(project_fields/2, [], Code),
    
    % Check for dict creation
    % We expect something like: v_Out = {'name': v_N}
    sub_string(Code, _, _, _, " = {'name': v_"),
    
    retract((project_fields(R, Out) :- get_dict(name, R, N), Out = _{name: N})).

:- end_tests(python_target).
