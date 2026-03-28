:- encoding(utf8).
:- use_module('src/unifyweaver/core/clause_body_analysis').

%% Test: classify_goal_sequence with guards and outputs
test_guard_output_sequence :-
    writeln('=== TEST: guard + output sequence ==='),
    %% classify(X, small) :- X > 0, X < 10, Y is X * 2.
    build_head_varmap([X, _Result], 1, VarMap),
    Goals = [(X > 0), (X < 10), (Y is X * 2)],
    classify_goal_sequence(Goals, VarMap, Classified),
    Classified = [guard(_, _), guard(_, _), output(_, Y, _)],
    writeln('  PASS: 2 guards + 1 output').

%% Test: if-then-else as output expression
test_ite_output :-
    writeln('=== TEST: if-then-else output ==='),
    %% label(X, L) :- (X > 0 -> L = positive ; L = negative).
    build_head_varmap([X, L], 1, VarMap),
    Goal = (X > 0 -> L = positive ; L = negative),
    classify_goal_sequence([Goal], VarMap, Classified),
    Classified = [output_ite(_, _, _, SharedVars)],
    SharedVars = [SV], SV == L,
    writeln('  PASS: detected if-then-else output binding L').

%% Test: disjunction output
test_disjunction_output :-
    writeln('=== TEST: disjunction output ==='),
    %% category(X, C) :- (X < 0, C = negative ; X =:= 0, C = zero ; X > 0, C = positive).
    build_head_varmap([_X, C], 1, VarMap),
    Goal = ((_X < 0, C = negative) ; (_X =:= 0, C = zero) ; (_X > 0, C = positive)),
    classify_goal_sequence([Goal], VarMap, Classified),
    Classified = [output_disj(Alts, SharedVars)],
    length(Alts, 3),
    SharedVars = [SV], SV == C,
    writeln('  PASS: detected 3-way disjunction output binding C').

%% Test: if-then (no else) output
test_if_then_output :-
    writeln('=== TEST: if-then output ==='),
    build_head_varmap([X, Y], 1, VarMap),
    Goal = (X > 0 -> Y is X * 2),
    classify_goal_sequence([Goal], VarMap, Classified),
    Classified = [output_if_then(_, _, OutputVars)],
    OutputVars = [OV], OV == Y,
    writeln('  PASS: detected if-then output binding Y').

%% Test: mixed guard + ite output
test_mixed_sequence :-
    writeln('=== TEST: mixed guard + if-then-else output ==='),
    build_head_varmap([X, R], 1, VarMap),
    Goals = [(X > 0), (X > 10 -> R = big ; R = small)],
    classify_goal_sequence(Goals, VarMap, Classified),
    Classified = [guard(_, _), output_ite(_, _, _, _)],
    writeln('  PASS: guard followed by ite output').

%% Test: compile_multi_clause
test_multi_clause :-
    writeln('=== TEST: compile_multi_clause ==='),
    %% Two clauses: fact(0, zero). fact(X, positive) :- X > 0.
    Head1 =.. [fact, 0, zero],
    Body1 = true,
    Head2 =.. [fact, X, Label],
    Body2 = (X > 0, Label = positive),
    compile_multi_clause([Head1-Body1, Head2-Body2], [], if_else_chain, Branches),
    length(Branches, 2),
    writeln('  PASS: 2 branches analyzed').

%% Test: backward compat — clause_guard_output_split still works
test_backward_compat :-
    writeln('=== TEST: clause_guard_output_split (backward compat) ==='),
    build_head_varmap([X, _], 1, VarMap),
    Goals = [(X > 0), (X < 10), (_Y is X * 2)],
    clause_guard_output_split(Goals, VarMap, Guards, Outputs),
    length(Guards, 2),
    length(Outputs, 1),
    writeln('  PASS: 2 guards, 1 output').

%% Test: analyze_clauses
test_analyze_clauses :-
    writeln('=== TEST: analyze_clauses ==='),
    Head =.. [classify, X, _Label],
    Body = (X > 0, _Label = positive),
    analyze_clauses([Head-Body], [Analysis]),
    Analysis = clause_info(_, _, _, _),
    writeln('  PASS: clause analyzed').

run_tests :-
    test_guard_output_sequence,
    test_ite_output,
    test_disjunction_output,
    test_if_then_output,
    test_mixed_sequence,
    test_multi_clause,
    test_backward_compat,
    test_analyze_clauses,
    nl, writeln('=== ALL 8 CLAUSE BODY ANALYSIS TESTS PASSED ===').
