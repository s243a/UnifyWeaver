:- encoding(utf8).
%% Test suite for the arg/3 lowering wired into wam_target.pl.
%%
%% Verifies:
%%   * With a `:- mode/1` declaration that proves T is `bound` at the
%%     program point of arg(N, T, A), and N is a literal positive
%%     integer, the generated WAM text contains `arg N TReg AReg`
%%     instead of `builtin_call arg/3`.
%%   * Without a mode declaration (T is `unknown`), the generator
%%     falls through to `builtin_call arg/3`.
%%   * A non-integer or zero N skips the lowering even with mode info.
%%   * The literal N appears in the emitted WAM text.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_arg3_lowering.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("WAM arg/3 lowering tests~n"),
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

test(test_arg3_lowered_when_t_bound_and_n_literal).
test(test_arg3_no_mode_falls_through_to_builtin).
test(test_arg3_dynamic_n_falls_through).
test(test_arg3_zero_or_negative_n_falls_through).
test(test_arg3_literal_n_appears_in_wam_text).

%% ========================================================================
%% Tests
%% ========================================================================

test_arg3_lowered_when_t_bound_and_n_literal :-
    Test = test_arg3_lowered_when_t_bound_and_n_literal,
    %% extract_first(T, X) :- arg(1, T, X).  with mode (+, -)
    retractall(user:extract_first(_, _)),
    retractall(user:mode(extract_first(_, _))),
    assert(user:mode(extract_first(+, -))),
    assert(user:(extract_first(T, X) :- arg(1, T, X))),
    (   catch(
            wam_target:compile_predicate_to_wam(extract_first/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:extract_first(_, _)),
        retractall(user:mode(extract_first(_, _))),
        (   sub_string(S, _, _, _, "arg 1,"),
            \+ sub_string(S, _, _, _, "builtin_call arg/3")
        ->  pass(Test)
        ;   fail_test(Test, expected_arg_instruction(S))
        )
    ;   retractall(user:extract_first(_, _)),
        retractall(user:mode(extract_first(_, _))),
        fail_test(Test, compile_failed)
    ).

test_arg3_no_mode_falls_through_to_builtin :-
    Test = test_arg3_no_mode_falls_through_to_builtin,
    retractall(user:extract_first_nomode(_, _)),
    assert(user:(extract_first_nomode(T, X) :- arg(1, T, X))),
    (   catch(
            wam_target:compile_predicate_to_wam(extract_first_nomode/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:extract_first_nomode(_, _)),
        (   sub_string(S, _, _, _, "builtin_call arg/3"),
            \+ sub_string(S, _, _, _, "arg 1,")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   retractall(user:extract_first_nomode(_, _)),
        fail_test(Test, compile_failed)
    ).

test_arg3_dynamic_n_falls_through :-
    Test = test_arg3_dynamic_n_falls_through,
    %% N is a runtime variable, not a literal — must NOT lower.
    retractall(user:extract_dyn(_, _, _)),
    retractall(user:mode(extract_dyn(_, _, _))),
    assert(user:mode(extract_dyn(+, +, -))),
    assert(user:(extract_dyn(N, T, X) :- arg(N, T, X))),
    (   catch(
            wam_target:compile_predicate_to_wam(extract_dyn/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:extract_dyn(_, _, _)),
        retractall(user:mode(extract_dyn(_, _, _))),
        (   sub_string(S, _, _, _, "builtin_call arg/3")
        ->  pass(Test)
        ;   fail_test(Test, dynamic_n_must_not_lower(S))
        )
    ;   retractall(user:extract_dyn(_, _, _)),
        retractall(user:mode(extract_dyn(_, _, _))),
        fail_test(Test, compile_failed)
    ).

test_arg3_zero_or_negative_n_falls_through :-
    Test = test_arg3_zero_or_negative_n_falls_through,
    retractall(user:extract_zero(_, _)),
    retractall(user:mode(extract_zero(_, _))),
    assert(user:mode(extract_zero(+, -))),
    %% N=0 is invalid for arg/3 — must NOT lower (preserves the
    %% builtin's failure semantics).
    assert(user:(extract_zero(T, X) :- arg(0, T, X))),
    (   catch(
            wam_target:compile_predicate_to_wam(extract_zero/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:extract_zero(_, _)),
        retractall(user:mode(extract_zero(_, _))),
        (   sub_string(S, _, _, _, "builtin_call arg/3")
        ->  pass(Test)
        ;   fail_test(Test, zero_n_must_not_lower(S))
        )
    ;   retractall(user:extract_zero(_, _)),
        retractall(user:mode(extract_zero(_, _))),
        fail_test(Test, compile_failed)
    ).

test_arg3_literal_n_appears_in_wam_text :-
    Test = test_arg3_literal_n_appears_in_wam_text,
    %% N=3 should appear verbatim in the emitted WAM text.
    retractall(user:extract_third(_, _)),
    retractall(user:mode(extract_third(_, _))),
    assert(user:mode(extract_third(+, -))),
    assert(user:(extract_third(T, X) :- arg(3, T, X))),
    (   catch(
            wam_target:compile_predicate_to_wam(extract_third/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:extract_third(_, _)),
        retractall(user:mode(extract_third(_, _))),
        (   sub_string(S, _, _, _, "arg 3,")
        ->  pass(Test)
        ;   fail_test(Test, missing_literal_n(S))
        )
    ;   retractall(user:extract_third(_, _)),
        retractall(user:mode(extract_third(_, _))),
        fail_test(Test, compile_failed)
    ).

:- initialization(run_tests, main).
