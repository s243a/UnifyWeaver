:- encoding(utf8).
%% Test suite for the =../2 compose-mode lowering wired into wam_target.pl.
%%
%% Verifies the M6 lowering decision:
%%   * With a `:- mode/1` declaration that proves T unbound and Name
%%     bound at the program point of T =.. [Name | Args], the generated
%%     WAM text contains `put_structure_dyn`.
%%   * Without a mode declaration (T's binding state is unknown), the
%%     generator falls through to `builtin_call =../2`.
%%   * With a mode declaration that proves T bound (decompose), the
%%     generator falls through to `builtin_call =../2`.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_univ_lowering.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("WAM =../2 compose-mode lowering tests~n"),
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

test(test_compose_mode_emits_put_structure_dyn).
test(test_no_mode_falls_through_to_builtin).
test(test_decompose_mode_falls_through_to_builtin).
test(test_compose_mode_arity_matches_list_length).
test(test_compose_mode_with_zero_arity).

%% ========================================================================
%% Fixtures
%% ========================================================================

%% A predicate where the LAST argument is the term to be built and the
%% functor name is given as an input. With mode (+, ?, -), the analyser
%% should prove Name bound and T unbound at the =../2 site, triggering
%% the lowering.
setup_compose_fixture :-
    retractall(user:build_term(_, _, _)),
    retractall(user:mode(build_term(_, _, _))),
    assert(user:mode(build_term(+, ?, -))),
    assert(user:(build_term(Name, Arg, T) :-
        T =.. [Name, Arg])).

teardown_compose_fixture :-
    retractall(user:build_term(_, _, _)),
    retractall(user:mode(build_term(_, _, _))).

%% Same predicate body but with no mode declaration — analysis cannot
%% prove the preconditions, so the lowering must NOT fire.
setup_no_mode_fixture :-
    retractall(user:build_term_nomode(_, _, _)),
    retractall(user:mode(build_term_nomode(_, _, _))),
    assert(user:(build_term_nomode(Name, Arg, T) :-
        T =.. [Name, Arg])).

teardown_no_mode_fixture :-
    retractall(user:build_term_nomode(_, _, _)).

%% Decompose: term in, name+arg out. With T bound (input mode +) the
%% analyser proves T bound and the lowering does NOT fire.
setup_decompose_fixture :-
    retractall(user:split_term(_, _, _)),
    retractall(user:mode(split_term(_, _, _))),
    assert(user:mode(split_term(+, -, -))),
    assert(user:(split_term(T, Name, Arg) :-
        T =.. [Name, Arg])).

teardown_decompose_fixture :-
    retractall(user:split_term(_, _, _)),
    retractall(user:mode(split_term(_, _, _))).

%% ========================================================================
%% Tests
%% ========================================================================

test_compose_mode_emits_put_structure_dyn :-
    Test = test_compose_mode_emits_put_structure_dyn,
    setup_compose_fixture,
    (   catch(
            wam_target:compile_predicate_to_wam(build_term/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        teardown_compose_fixture,
        (   sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, no_put_structure_dyn(S))
        )
    ;   teardown_compose_fixture,
        fail_test(Test, compile_failed)
    ).

test_no_mode_falls_through_to_builtin :-
    Test = test_no_mode_falls_through_to_builtin,
    setup_no_mode_fixture,
    (   catch(
            wam_target:compile_predicate_to_wam(build_term_nomode/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        teardown_no_mode_fixture,
        (   sub_string(S, _, _, _, "builtin_call =../2"),
            \+ sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   teardown_no_mode_fixture,
        fail_test(Test, compile_failed)
    ).

test_decompose_mode_falls_through_to_builtin :-
    Test = test_decompose_mode_falls_through_to_builtin,
    setup_decompose_fixture,
    (   catch(
            wam_target:compile_predicate_to_wam(split_term/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        teardown_decompose_fixture,
        (   sub_string(S, _, _, _, "builtin_call =../2"),
            \+ sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   teardown_decompose_fixture,
        fail_test(Test, compile_failed)
    ).

test_compose_mode_arity_matches_list_length :-
    Test = test_compose_mode_arity_matches_list_length,
    %% Use a 4-arity predicate: build_pair(Name, A, B, T) :- T =.. [Name, A, B].
    %% With mode (+, ?, ?, -) the lowering should fire and put_constant 2
    %% should appear (the literal arity).
    retractall(user:build_pair(_, _, _, _)),
    retractall(user:mode(build_pair(_, _, _, _))),
    assert(user:mode(build_pair(+, ?, ?, -))),
    assert(user:(build_pair(Name, A, B, T) :-
        T =.. [Name, A, B])),
    (   catch(
            wam_target:compile_predicate_to_wam(build_pair/4, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:build_pair(_, _, _, _)),
        retractall(user:mode(build_pair(_, _, _, _))),
        (   sub_string(S, _, _, _, "put_structure_dyn"),
            sub_string(S, _, _, _, "put_constant 2, A2")
        ->  pass(Test)
        ;   fail_test(Test, missing_arity_or_dyn(S))
        )
    ;   retractall(user:build_pair(_, _, _, _)),
        retractall(user:mode(build_pair(_, _, _, _))),
        fail_test(Test, compile_failed)
    ).

test_compose_mode_with_zero_arity :-
    Test = test_compose_mode_with_zero_arity,
    %% Build a zero-arity term (atom-as-functor): t(Name, T) :- T =.. [Name].
    %% With mode (+, -) the lowering fires and arity is 0.
    retractall(user:build_atom(_, _)),
    retractall(user:mode(build_atom(_, _))),
    assert(user:mode(build_atom(+, -))),
    assert(user:(build_atom(Name, T) :-
        T =.. [Name])),
    (   catch(
            wam_target:compile_predicate_to_wam(build_atom/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:build_atom(_, _)),
        retractall(user:mode(build_atom(_, _))),
        (   sub_string(S, _, _, _, "put_structure_dyn"),
            sub_string(S, _, _, _, "put_constant 0, A2")
        ->  pass(Test)
        ;   fail_test(Test, missing_zero_arity(S))
        )
    ;   retractall(user:build_atom(_, _)),
        retractall(user:mode(build_atom(_, _))),
        fail_test(Test, compile_failed)
    ).

:- initialization(run_tests, main).
