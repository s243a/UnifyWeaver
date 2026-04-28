:- encoding(utf8).
%% Test suite for the IntSet visited Layer 2 codegen integration.
%%
%% Verifies the WAM compiler emits set-typed instructions
%% (build_empty_set / set_insert / not_member_set) when a predicate
%% argument is declared as a visited-set via
%%
%%     :- visited_set(Pred/Arity, ArgN).
%%
%% Three patterns matter:
%%   * \+ member(X, V) where V is the head's visited-set var
%%       => not_member_set XReg, VReg
%%   * recursive call ...(..., [X|V_visited], ...)
%%       => set_insert XReg, V_visited_Reg, FreshReg + put_value FreshReg
%%   * bootstrap call ...(..., [X], ...)
%%       => build_empty_set R + set_insert XReg, R, R + put_value R
%%
%% Without the directive declared, the compiler must keep the existing
%% Phase G behaviour (NotMemberList / put_list / etc).
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_visited_set_lowering.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(lists)).

run_tests :-
    format("~n========================================~n"),
    format("WAM IntSet visited-set codegen tests~n"),
    format("========================================~n~n"),
    findall(T, test(T), Tests),
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

run_all([], P, P).
run_all([T|Rest], Acc, P) :-
    (   catch(call(T), E, (format("[FAIL] ~w: ~w~n", [T, E]), fail))
    ->  Acc1 is Acc + 1, run_all(Rest, Acc1, P)
    ;   run_all(Rest, Acc, P)
    ).

pass(N) :- format("[PASS] ~w~n", [N]).
fail_test(N, R) :- format("[FAIL] ~w: ~w~n", [N, R]), fail.

test(test_not_member_set_when_visited_declared).
test(test_no_visited_directive_keeps_not_member_list).
test(test_visited_set_var_propagation_across_clauses).
test(test_unrelated_call_with_list_arg_unchanged).

%% Cleanup helper to keep state hygiene between tests.
:- dynamic user:visited_set/2.
:- dynamic user:mode/1.

reset_directives :-
    retractall(user:visited_set(_, _)),
    retractall(user:mode(_)).

%% ========================================================================
%% Tests
%% ========================================================================

%% Tagged predicates for each test, distinct names so user-module
%% retractall between tests doesn't cross-contaminate.

test_not_member_set_when_visited_declared :-
    Test = test_not_member_set_when_visited_declared,
    reset_directives,
    retractall(user:vis_check_a(_, _, _)),
    assertz(user:visited_set(vis_check_a/3, 3)),
    assertz(user:mode(vis_check_a(?, +, +))),
    assertz(user:(vis_check_a(_, X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(vis_check_a/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives,
        retractall(user:vis_check_a(_, _, _)),
        (   sub_string(S, _, _, _, "not_member_set"),
            \+ sub_string(S, _, _, _, "not_member_list"),
            \+ sub_string(S, _, _, _, "builtin_call \\+/1")
        ->  pass(Test)
        ;   fail_test(Test, expected_not_member_set(S))
        )
    ;   reset_directives,
        retractall(user:vis_check_a(_, _, _)),
        fail_test(Test, compile_failed)
    ).

test_no_visited_directive_keeps_not_member_list :-
    Test = test_no_visited_directive_keeps_not_member_list,
    reset_directives,
    retractall(user:vis_check_b(_, _, _)),
    assertz(user:mode(vis_check_b(?, +, +))),
    assertz(user:(vis_check_b(_, X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(vis_check_b/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives,
        retractall(user:vis_check_b(_, _, _)),
        (   sub_string(S, _, _, _, "not_member_list"),
            \+ sub_string(S, _, _, _, "not_member_set")
        ->  pass(Test)
        ;   fail_test(Test, expected_not_member_list(S))
        )
    ;   reset_directives,
        retractall(user:vis_check_b(_, _, _)),
        fail_test(Test, compile_failed)
    ).

test_visited_set_var_propagation_across_clauses :-
    %% Two clauses of the same predicate. The visited var V appears in
    %% both head positions. Each clause should independently lower
    %% \+ member(X, V) to not_member_set.
    Test = test_visited_set_var_propagation_across_clauses,
    reset_directives,
    retractall(user:vis_two/2),
    assertz(user:visited_set(vis_two/2, 2)),
    assertz(user:mode(vis_two(?, +))),
    assertz(user:(vis_two(X, V) :- \+ member(X, V))),
    assertz(user:(vis_two(X, V) :- \+ member(X, V), some_other_goal_placeholder)),
    (   catch(
            wam_target:compile_predicate_to_wam(vis_two/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives,
        retractall(user:vis_two(_, _)),
        %% Both clauses should emit not_member_set — count occurrences.
        count_substr(S, "not_member_set", N),
        (   N >= 2
        ->  pass(Test)
        ;   fail_test(Test, only_one_lowering(N, S))
        )
    ;   reset_directives,
        retractall(user:vis_two(_, _)),
        fail_test(Test, compile_failed)
    ).

test_unrelated_call_with_list_arg_unchanged :-
    %% A call to a predicate with NO visited_set declaration should
    %% NOT have its list args rewritten — the rewrite is opt-in via
    %% the directive.
    Test = test_unrelated_call_with_list_arg_unchanged,
    reset_directives,
    retractall(user:vis_passthrough(_, _)),
    %% Note: NO visited_set directive on append/3.
    assertz(user:(vis_passthrough(X, Y) :- append([X], [], Y))),
    (   catch(
            wam_target:compile_predicate_to_wam(vis_passthrough/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives,
        retractall(user:vis_passthrough(_, _)),
        (   \+ sub_string(S, _, _, _, "build_empty_set"),
            \+ sub_string(S, _, _, _, "set_insert")
        ->  pass(Test)
        ;   fail_test(Test, unexpected_set_emission(S))
        )
    ;   reset_directives,
        retractall(user:vis_passthrough(_, _)),
        fail_test(Test, compile_failed)
    ).

%% ========================================================================
%% Helpers
%% ========================================================================

count_substr(Hay, Needle, N) :-
    string_length(Needle, NLen),
    findall(P, sub_string(Hay, P, NLen, _, Needle), Ps),
    length(Ps, N).

:- initialization(run_tests, main).
