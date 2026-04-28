:- encoding(utf8).
%% Test suite for binding_state_analysis.pl
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_binding_state_analysis.pl

:- use_module('../../src/unifyweaver/core/binding_state_analysis').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("Binding State Analysis Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

pass(Name) :- format("[PASS] ~w~n", [Name]).
fail_test(Name, Reason) :- format("[FAIL] ~w: ~w~n", [Name, Reason]), fail.

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_initial_env_no_mode).
test(test_initial_env_input_mode).
test(test_initial_env_output_mode).
test(test_initial_env_any_mode).
test(test_initial_env_structured_head).
test(test_get_default_unknown).
test(test_propagate_unify_var_term).
test(test_propagate_unify_two_vars_one_bound).
test(test_propagate_is).
test(test_propagate_nonvar_guard).
test(test_propagate_var_guard).
test(test_propagate_atom_guard).
test(test_propagate_functor_compose).
test(test_propagate_functor_decompose).
test(test_propagate_univ_compose).
test(test_propagate_univ_decompose).
test(test_propagate_univ_unbound_unbound).
test(test_propagate_negation_noop).
test(test_propagate_cut_noop).
test(test_propagate_arg_unknown).
test(test_propagate_copy_term_bound).
test(test_ite_meet_disagree).
test(test_ite_meet_agree).
test(test_disjunction_meet).
test(test_findall_result_bound).
test(test_call_unknown_pred_opacity).
test(test_call_any_mode_preserves_bound).
test(test_walk_body_indices).
test(test_binding_state_at_lookup).
test(test_binding_state_at_var_negative).
test(test_meet_unbound_unbound).
test(test_meet_bound_unbound).

%% ========================================================================
%% Section 1 — initial_binding_env
%% ========================================================================

test_initial_env_no_mode :-
    Test = test_initial_env_no_mode,
    initial_binding_env(foo(X, Y), none, Env),
    get_binding_state(Env, X, SX),
    get_binding_state(Env, Y, SY),
    (   SX == unknown, SY == unknown
    ->  pass(Test)
    ;   fail_test(Test, sx_or_sy_not_unknown(SX, SY))
    ).

test_initial_env_input_mode :-
    Test = test_initial_env_input_mode,
    initial_binding_env(foo(X), [input], Env),
    get_binding_state(Env, X, SX),
    (   SX == bound
    ->  pass(Test)
    ;   fail_test(Test, expected_bound_got(SX))
    ).

test_initial_env_output_mode :-
    Test = test_initial_env_output_mode,
    initial_binding_env(foo(X), [output], Env),
    get_binding_state(Env, X, SX),
    (   SX == unbound
    ->  pass(Test)
    ;   fail_test(Test, expected_unbound_got(SX))
    ).

test_initial_env_any_mode :-
    Test = test_initial_env_any_mode,
    initial_binding_env(foo(X), [any], Env),
    get_binding_state(Env, X, SX),
    (   SX == unknown
    ->  pass(Test)
    ;   fail_test(Test, expected_unknown_got(SX))
    ).

test_initial_env_structured_head :-
    Test = test_initial_env_structured_head,
    %% Head foo(f(X)) — X is bound by head unification.
    initial_binding_env(foo(f(X)), [any], Env),
    get_binding_state(Env, X, SX),
    (   SX == bound
    ->  pass(Test)
    ;   fail_test(Test, expected_bound_got(SX))
    ).

test_get_default_unknown :-
    Test = test_get_default_unknown,
    empty_binding_env(Env),
    get_binding_state(Env, _Z, SZ),
    (   SZ == unknown
    ->  pass(Test)
    ;   fail_test(Test, expected_unknown_got(SZ))
    ).

%% ========================================================================
%% Section 2 — per-goal propagation
%% ========================================================================

test_propagate_unify_var_term :-
    Test = test_propagate_unify_var_term,
    empty_binding_env(Env0),
    propagate_goal(X = foo(a), Env0, Env1),
    get_binding_state(Env1, X, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_unify_two_vars_one_bound :-
    Test = test_propagate_unify_two_vars_one_bound,
    empty_binding_env(Env0),
    set_binding_state(Env0, X, bound, Env1),
    propagate_goal(X = Y, Env1, Env2),
    get_binding_state(Env2, X, SX),
    get_binding_state(Env2, Y, SY),
    (   SX == bound, SY == bound
    ->  pass(Test)
    ;   fail_test(Test, sx_sy(SX, SY))
    ).

test_propagate_is :-
    Test = test_propagate_is,
    empty_binding_env(Env0),
    set_binding_state(Env0, X, bound, Env1),
    propagate_goal(Y is X + 1, Env1, Env2),
    get_binding_state(Env2, Y, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_nonvar_guard :-
    Test = test_propagate_nonvar_guard,
    empty_binding_env(Env0),
    propagate_goal(nonvar(X), Env0, Env1),
    get_binding_state(Env1, X, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_var_guard :-
    Test = test_propagate_var_guard,
    empty_binding_env(Env0),
    propagate_goal(var(X), Env0, Env1),
    get_binding_state(Env1, X, S),
    (   S == unbound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_atom_guard :-
    Test = test_propagate_atom_guard,
    empty_binding_env(Env0),
    propagate_goal(atom(X), Env0, Env1),
    get_binding_state(Env1, X, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_functor_compose :-
    %% functor(T, foo, 2) with T unbound — analysis can prove T bound after.
    Test = test_propagate_functor_compose,
    empty_binding_env(Env0),
    set_binding_state(Env0, T, unbound, Env1),
    propagate_goal(functor(T, foo, 2), Env1, Env2),
    get_binding_state(Env2, T, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_functor_decompose :-
    %% functor(T, N, A) with T bound — N and A become bound.
    Test = test_propagate_functor_decompose,
    empty_binding_env(Env0),
    set_binding_state(Env0, T, bound, Env1),
    propagate_goal(functor(T, N, A), Env1, Env2),
    get_binding_state(Env2, N, SN),
    get_binding_state(Env2, A, SA),
    (   SN == bound, SA == bound
    ->  pass(Test)
    ;   fail_test(Test, sn_sa(SN, SA))
    ).

test_propagate_univ_compose :-
    %% T =.. [foo, X, Y] — list literally bound (atom head, ground tail).
    Test = test_propagate_univ_compose,
    empty_binding_env(Env0),
    set_binding_state(Env0, T, unbound, Env1),
    propagate_goal(T =.. [foo, _, _], Env1, Env2),
    get_binding_state(Env2, T, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_univ_decompose :-
    Test = test_propagate_univ_decompose,
    empty_binding_env(Env0),
    set_binding_state(Env0, T, bound, Env1),
    propagate_goal(T =.. L, Env1, Env2),
    get_binding_state(Env2, L, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_univ_unbound_unbound :-
    %% Both sides unknown ⇒ both stay unknown.
    Test = test_propagate_univ_unbound_unbound,
    empty_binding_env(Env0),
    propagate_goal(T =.. L, Env0, Env1),
    get_binding_state(Env1, T, ST),
    get_binding_state(Env1, L, SL),
    (   ST == unknown, SL == unknown
    ->  pass(Test)
    ;   fail_test(Test, st_sl(ST, SL))
    ).

test_propagate_negation_noop :-
    Test = test_propagate_negation_noop,
    empty_binding_env(Env0),
    set_binding_state(Env0, X, unbound, Env1),
    propagate_goal(\+(some_pred(X)), Env1, Env2),
    get_binding_state(Env2, X, S),
    (   S == unbound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_cut_noop :-
    Test = test_propagate_cut_noop,
    empty_binding_env(Env0),
    set_binding_state(Env0, X, bound, Env1),
    propagate_goal(!, Env1, Env2),
    get_binding_state(Env2, X, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_propagate_arg_unknown :-
    Test = test_propagate_arg_unknown,
    empty_binding_env(Env0),
    set_binding_state(Env0, T, bound, Env1),
    propagate_goal(arg(1, T, A), Env1, Env2),
    get_binding_state(Env2, A, S),
    (   S == unknown -> pass(Test) ; fail_test(Test, S) ).

test_propagate_copy_term_bound :-
    Test = test_propagate_copy_term_bound,
    empty_binding_env(Env0),
    set_binding_state(Env0, T, bound, Env1),
    propagate_goal(copy_term(T, C), Env1, Env2),
    get_binding_state(Env2, C, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

%% ========================================================================
%% Section 3 — control constructs and meet
%% ========================================================================

test_ite_meet_disagree :-
    %% Then sets X bound, Else does nothing — meet ⇒ unknown.
    Test = test_ite_meet_disagree,
    empty_binding_env(Env0),
    propagate_goal(
        (true -> X = foo ; true),
        Env0, Env),
    get_binding_state(Env, X, S),
    (   S == unknown -> pass(Test) ; fail_test(Test, S) ).

test_ite_meet_agree :-
    %% Both branches set X bound — meet ⇒ bound.
    Test = test_ite_meet_agree,
    empty_binding_env(Env0),
    propagate_goal(
        (true -> X = foo ; X = bar),
        Env0, Env),
    get_binding_state(Env, X, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_disjunction_meet :-
    Test = test_disjunction_meet,
    empty_binding_env(Env0),
    propagate_goal(
        (X = a ; X = b),
        Env0, Env),
    get_binding_state(Env, X, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

test_findall_result_bound :-
    Test = test_findall_result_bound,
    empty_binding_env(Env0),
    propagate_goal(findall(_, member(_, [1,2,3]), R), Env0, Env),
    get_binding_state(Env, R, S),
    (   S == bound -> pass(Test) ; fail_test(Test, S) ).

%% ========================================================================
%% Section 4 — user calls
%% ========================================================================

test_call_unknown_pred_opacity :-
    %% Calling some_user_pred(X) without mode declaration:
    %% X transitions to unknown (was unset, default unknown — still unknown).
    Test = test_call_unknown_pred_opacity,
    empty_binding_env(Env0),
    set_binding_state(Env0, X, bound, Env1),
    propagate_goal(some_user_pred(X), Env1, Env2),
    get_binding_state(Env2, X, S),
    (   S == unknown -> pass(Test) ; fail_test(Test, S) ).

test_call_any_mode_preserves_bound :-
    %% Per WAM_HASKELL_MODE_ANALYSIS_SPEC.md §2.3.7: `?` mode leaves
    %% the argument at its pre-call state. A bound arg passed to a
    %% predicate declared with `?` mode must STAY bound after the
    %% call. This is the gating condition for the \+ member lowering
    %% to fire across opaque fact-predicate calls in real workloads
    %% (e.g. category_ancestor calling category_parent).
    Test = test_call_any_mode_preserves_bound,
    %% Setup: declare some_poly_pred as mode (?, ?).
    retractall(user:mode(some_poly_pred(_, _))),
    assertz(user:mode(some_poly_pred(?, ?))),
    empty_binding_env(Env0),
    set_binding_state(Env0, X, bound, Env1),
    set_binding_state(Env1, Y, bound, Env2),
    propagate_goal(some_poly_pred(X, Y), Env2, Env3),
    get_binding_state(Env3, X, SX),
    get_binding_state(Env3, Y, SY),
    retractall(user:mode(some_poly_pred(_, _))),
    (   SX == bound, SY == bound
    ->  pass(Test)
    ;   fail_test(Test, sx_sy(SX, SY))
    ).

%% ========================================================================
%% Section 5 — full clause walks
%% ========================================================================

test_walk_body_indices :-
    %% A 3-goal body produces 3 records with indices 1,2,3.
    Test = test_walk_body_indices,
    Head = p(_X, _Y),
    Body = (_A = a, _B = b, _C = c),
    analyse_clause_bindings(Head, Body, Bindings),
    length(Bindings, 3),
    Bindings = [goal_binding(1, _, _), goal_binding(2, _, _), goal_binding(3, _, _)],
    pass(Test).

test_binding_state_at_lookup :-
    Test = test_binding_state_at_lookup,
    Head = p(X),
    Body = (Y = foo, X = Y),
    analyse_clause_bindings(Head, Body, Bindings),
    %% At goal 2 (`X = Y`), Y should already be bound from goal 1.
    binding_state_at(2, Y, Bindings, SY),
    (   SY == bound -> pass(Test) ; fail_test(Test, SY) ).

test_binding_state_at_var_negative :-
    %% binding_state_at_var with `unknown` expected fails (it never matches).
    Test = test_binding_state_at_var_negative,
    empty_binding_env(Env),
    (   binding_state_at_var(Env, _Z, unknown)
    ->  fail_test(Test, unknown_should_not_match)
    ;   pass(Test)
    ).

test_meet_unbound_unbound :-
    Test = test_meet_unbound_unbound,
    empty_binding_env(E),
    set_binding_state(E, X, unbound, EA),
    set_binding_state(E, X, unbound, EB),
    meet_env(EA, EB, EM),
    get_binding_state(EM, X, S),
    (   S == unbound -> pass(Test) ; fail_test(Test, S) ).

test_meet_bound_unbound :-
    Test = test_meet_bound_unbound,
    empty_binding_env(E),
    set_binding_state(E, X, bound, EA),
    set_binding_state(E, X, unbound, EB),
    meet_env(EA, EB, EM),
    get_binding_state(EM, X, S),
    (   S == unknown -> pass(Test) ; fail_test(Test, S) ).

:- initialization(run_tests, main).
