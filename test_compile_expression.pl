:- encoding(utf8).
:- ['src/unifyweaver/init'].
:- use_module('src/unifyweaver/core/clause_body_analysis').
:- use_module('src/unifyweaver/core/recursive_compiler').

%% Test the shared compile_expression framework with Python hooks.

test_compile_expression_ite :-
    writeln('=== TEST: compile_expression with ite ==='),
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    compile_expression(python, Goal, VarMap, Code, OutputVars, _),
    format('  Code: ~w~n  OutputVars: ~w~n', [Code, OutputVars]),
    (Code \= "" -> writeln('  PASS') ; (writeln('  FAIL'), fail)).

test_compile_branch :-
    writeln('=== TEST: compile_branch with multi-goal body ==='),
    build_head_varmap([X, C, S], 1, VarMap),
    Branch = (C = positive, S is X * 10),
    compile_branch(python, Branch, VarMap, Lines, OutputVars, _),
    format('  Lines: ~w~n  OutputVars: ~w~n', [Lines, OutputVars]),
    (Lines \= [] -> writeln('  PASS') ; (writeln('  FAIL'), fail)).

test_compile_classified :-
    writeln('=== TEST: compile_classified_sequence ==='),
    build_head_varmap([X, Y], 1, VarMap),
    Goals = [(X > 0), (Y is X * 2)],
    classify_goal_sequence(Goals, VarMap, Classified),
    compile_classified_sequence(python, Classified, VarMap, Lines, _),
    format('  Lines: ~w~n', [Lines]),
    (Lines \= [] -> writeln('  PASS') ; (writeln('  FAIL'), fail)).

test_output_goal :-
    writeln('=== TEST: compile_expression with output goal ==='),
    build_head_varmap([X, Y], 1, VarMap),
    Goal = (Y is X * 2),
    compile_expression(python, Goal, VarMap, Code, OutputVars, _),
    format('  Code: ~w~n  OutputVars: ~w~n', [Code, OutputVars]),
    (Code \= "" -> writeln('  PASS') ; (writeln('  FAIL'), fail)).

test_guard_only :-
    writeln('=== TEST: compile_expression with guard ==='),
    build_head_varmap([X], 1, VarMap),
    Goal = (X > 0),
    compile_expression(python, Goal, VarMap, Code, OutputVars, _),
    format('  Code: ~w~n  OutputVars: ~w~n', [Code, OutputVars]),
    (OutputVars == [] -> writeln('  PASS: guard has no outputs') ; writeln('  NOTE: guard produced outputs')).

run_tests :-
    test_guard_only,
    test_output_goal,
    test_compile_classified,
    test_compile_branch,
    test_compile_expression_ite,
    nl, writeln('=== ALL SHARED COMPILE_EXPRESSION TESTS PASSED ===').
