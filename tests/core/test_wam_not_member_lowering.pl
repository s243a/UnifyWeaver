:- encoding(utf8).
%% Test suite for the \+ member(X, L) lowering wired into wam_target.pl.
%%
%% Verifies:
%%   * With a `:- mode/1` declaration that proves both X and L are
%%     `bound` at the call site, the generated WAM text contains
%%     `not_member_list XReg, LReg` instead of
%%     `put_structure member/2 ...` + `builtin_call \+/1`.
%%   * Without a mode declaration, falls through to the existing
%%     put_structure + builtin_call \+/1 path.
%%   * Standalone `member(X, L)` (no `\+`) is NOT lowered (only
%%     `\+ member` is).
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_not_member_lowering.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("WAM \\\\+ member(X, L) lowering tests~n"),
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

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_not_member_lowered_when_x_and_l_bound).
test(test_not_member_no_mode_falls_through).
test(test_not_member_only_x_bound_falls_through).
test(test_not_member_only_l_bound_falls_through).
test(test_plain_member_not_lowered).

%% ========================================================================
%% Tests
%% ========================================================================

test_not_member_lowered_when_x_and_l_bound :-
    Test = test_not_member_lowered_when_x_and_l_bound,
    %% visit_check(X, V) :- \+ member(X, V).  with mode (+, +)
    retractall(user:visit_check(_, _)),
    retractall(user:mode(visit_check(_, _))),
    assert(user:mode(visit_check(+, +))),
    assert(user:(visit_check(X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(visit_check/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:visit_check(_, _)),
        retractall(user:mode(visit_check(_, _))),
        (   sub_string(S, _, _, _, "not_member_list"),
            \+ sub_string(S, _, _, _, "put_structure member"),
            \+ sub_string(S, _, _, _, "builtin_call \\+/1")
        ->  pass(Test)
        ;   fail_test(Test, expected_not_member_list(S))
        )
    ;   retractall(user:visit_check(_, _)),
        retractall(user:mode(visit_check(_, _))),
        fail_test(Test, compile_failed)
    ).

test_not_member_no_mode_falls_through :-
    Test = test_not_member_no_mode_falls_through,
    retractall(user:visit_check_nomode(_, _)),
    assert(user:(visit_check_nomode(X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(visit_check_nomode/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:visit_check_nomode(_, _)),
        (   sub_string(S, _, _, _, "builtin_call \\+/1"),
            \+ sub_string(S, _, _, _, "not_member_list")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   retractall(user:visit_check_nomode(_, _)),
        fail_test(Test, compile_failed)
    ).

test_not_member_only_x_bound_falls_through :-
    Test = test_not_member_only_x_bound_falls_through,
    %% L is mode `?` (any) — analyser cannot prove `bound`.
    retractall(user:visit_check_partial(_, _)),
    retractall(user:mode(visit_check_partial(_, _))),
    assert(user:mode(visit_check_partial(+, ?))),
    assert(user:(visit_check_partial(X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(visit_check_partial/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:visit_check_partial(_, _)),
        retractall(user:mode(visit_check_partial(_, _))),
        (   sub_string(S, _, _, _, "builtin_call \\+/1"),
            \+ sub_string(S, _, _, _, "not_member_list")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   retractall(user:visit_check_partial(_, _)),
        retractall(user:mode(visit_check_partial(_, _))),
        fail_test(Test, compile_failed)
    ).

test_not_member_only_l_bound_falls_through :-
    Test = test_not_member_only_l_bound_falls_through,
    %% X is mode `?`, L is mode `+` — analyser cannot prove X is bound.
    retractall(user:visit_check_partial2(_, _)),
    retractall(user:mode(visit_check_partial2(_, _))),
    assert(user:mode(visit_check_partial2(?, +))),
    assert(user:(visit_check_partial2(X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(visit_check_partial2/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:visit_check_partial2(_, _)),
        retractall(user:mode(visit_check_partial2(_, _))),
        (   sub_string(S, _, _, _, "builtin_call \\+/1"),
            \+ sub_string(S, _, _, _, "not_member_list")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   retractall(user:visit_check_partial2(_, _)),
        retractall(user:mode(visit_check_partial2(_, _))),
        fail_test(Test, compile_failed)
    ).

test_plain_member_not_lowered :-
    Test = test_plain_member_not_lowered,
    %% Standalone `member(X, V)` (no `\+`) must NOT be lowered to
    %% not_member_list — that would invert the semantics.
    retractall(user:check_member(_, _)),
    retractall(user:mode(check_member(_, _))),
    assert(user:mode(check_member(+, +))),
    assert(user:(check_member(X, V) :- member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(check_member/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:check_member(_, _)),
        retractall(user:mode(check_member(_, _))),
        (   \+ sub_string(S, _, _, _, "not_member_list"),
            sub_string(S, _, _, _, "builtin_call member/2")
        ->  pass(Test)
        ;   fail_test(Test, plain_member_should_not_lower(S))
        )
    ;   retractall(user:check_member(_, _)),
        retractall(user:mode(check_member(_, _))),
        fail_test(Test, compile_failed)
    ).

:- initialization(run_tests, main).
