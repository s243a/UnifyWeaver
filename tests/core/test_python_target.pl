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

test(compile_multiple_clauses) :-
    % parent(p1, c1).
    % parent(p1, c2).
    % This is a fact, which is a clause with body 'true'.
    assertz(parent(p1, c1)),
    assertz(parent(p1, c2)),
    
    compile_predicate_to_python(parent/2, [], Code),
    
    % We expect code that yields both c1 and c2 for input p1.
    % Currently, the compiler might only pick one.
    % Let's check if it generates code for both.
    % Since we don't know the exact python representation of facts yet,
    % we just check if it handles multiple clauses.
    % For facts: parent(X, Y) :- X=p1, Y=c1.
    % The compiler should translate unification.
    
    % Check for structure implying multiple paths
    % e.g. multiple 'yield' statements or a loop over clauses?
    % For now, just print the code to see what happens.
    format(user_error, "Generated Multi-Clause Code:\n~s\n", [Code]),
    
    retractall(parent(_, _)).

test(recursive_factorial) :-
    % factorial(0, 1).
    % factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
    assertz((factorial(0, 1))),
    assertz((factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),
    
    compile_predicate_to_python(factorial/2, [], Code),
    
    % Check for worker function with memoization
    sub_string(Code, _, _, _, "@functools.cache"),
    sub_string(Code, _, _, _, "def _factorial_worker"),
    
    % Check for base case
    sub_string(Code, _, _, _, "if "),
    
    retractall(factorial(_,_)).

test(tail_recursive_sum) :-
    % sum(0, Acc, Acc).
    % sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S).
    assertz((sum(0, Acc, Acc))),
    assertz((sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S))),
    
    compile_predicate_to_python(sum/3, [], Code),
    
    % Should detect tail recursion and generate while loop
    sub_string(Code, _, _, _, "while"),
    sub_string(Code, _, _, _, "# Tail recursion (arity 3)"),
    \+ sub_string(Code, _, _, _, "ERROR"),
    
    retractall(sum(_,_,_)).

test(mutual_even_odd) :-
    % Classic mutual recursion: is_even/is_odd
    % NOTE: Full mutual recursion requires call_graph module integration
    % For now, this compiles each predicate independently
    assertz((is_even(0))),
    assertz((is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))),
    assertz((is_odd(1))),
    assertz((is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))),
    
    % Should compile successfully (even if not as mutual recursion yet)
    compile_predicate_to_python(is_even/1, [], Code),
    atom_string(Code, CodeStr),
    atom_length(CodeStr, Len),
    assertion(Len > 100),  % Should generate some code
    
    retractall(is_even(_)),
    retractall(is_odd(_)).

:- end_tests(python_target).
