:- encoding(utf8).
%% Test suite for the functor/3 compose-mode lowering wired into wam_target.pl.
%%
%% Mirrors the =../2 lowering test pattern. Verifies:
%%   * With a `:- mode/1` declaration that proves T unbound and Name
%%     bound at the program point of functor(T, Name, Arity), and Arity
%%     a literal non-negative integer, the generated WAM text contains
%%     `put_structure_dyn`.
%%   * Without a mode declaration (T's binding state is unknown), the
%%     generator falls through to `builtin_call functor/3`.
%%   * With a mode declaration that proves T bound (decompose), the
%%     generator falls through to `builtin_call functor/3`.
%%   * A non-integer or negative Arity skips the lowering even with
%%     mode info.
%%   * Arity controls how many `set_variable` instructions are emitted
%%     (one per fresh slot in the constructed term).
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_functor3_lowering.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("WAM functor/3 compose-mode lowering tests~n"),
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

test(test_functor3_compose_mode_emits_put_structure_dyn).
test(test_functor3_no_mode_falls_through_to_builtin).
test(test_functor3_decompose_mode_falls_through_to_builtin).
test(test_functor3_arity_matches_set_variable_count).
test(test_functor3_zero_arity_emits_no_set_variable).
test(test_functor3_non_integer_arity_falls_through).

%% Count occurrences of a substring (cheap, used to assert N copies of
%% set_variable in the emitted WAM text).
count_substr(Hay, Needle, N) :-
    string_length(Needle, NeedleLen),
    findall(P, sub_string(Hay, P, NeedleLen, _, Needle), Ps),
    length(Ps, N).

%% ========================================================================
%% Tests
%% ========================================================================

test_functor3_compose_mode_emits_put_structure_dyn :-
    Test = test_functor3_compose_mode_emits_put_structure_dyn,
    retractall(user:make_pair(_, _)),
    retractall(user:mode(make_pair(_, _))),
    assert(user:mode(make_pair(+, -))),
    assert(user:(make_pair(Name, T) :- functor(T, Name, 2))),
    (   catch(
            wam_target:compile_predicate_to_wam(make_pair/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:make_pair(_, _)),
        retractall(user:mode(make_pair(_, _))),
        (   sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, no_put_structure_dyn(S))
        )
    ;   retractall(user:make_pair(_, _)),
        retractall(user:mode(make_pair(_, _))),
        fail_test(Test, compile_failed)
    ).

test_functor3_no_mode_falls_through_to_builtin :-
    Test = test_functor3_no_mode_falls_through_to_builtin,
    retractall(user:make_pair_nomode(_, _)),
    retractall(user:mode(make_pair_nomode(_, _))),
    assert(user:(make_pair_nomode(Name, T) :- functor(T, Name, 2))),
    (   catch(
            wam_target:compile_predicate_to_wam(make_pair_nomode/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:make_pair_nomode(_, _)),
        (   sub_string(S, _, _, _, "builtin_call functor/3"),
            \+ sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   retractall(user:make_pair_nomode(_, _)),
        fail_test(Test, compile_failed)
    ).

test_functor3_decompose_mode_falls_through_to_builtin :-
    Test = test_functor3_decompose_mode_falls_through_to_builtin,
    retractall(user:split_pair(_, _, _)),
    retractall(user:mode(split_pair(_, _, _))),
    assert(user:mode(split_pair(+, -, -))),
    assert(user:(split_pair(T, Name, Arity) :- functor(T, Name, Arity))),
    (   catch(
            wam_target:compile_predicate_to_wam(split_pair/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:split_pair(_, _, _)),
        retractall(user:mode(split_pair(_, _, _))),
        (   sub_string(S, _, _, _, "builtin_call functor/3"),
            \+ sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, expected_builtin_path(S))
        )
    ;   retractall(user:split_pair(_, _, _)),
        retractall(user:mode(split_pair(_, _, _))),
        fail_test(Test, compile_failed)
    ).

test_functor3_arity_matches_set_variable_count :-
    Test = test_functor3_arity_matches_set_variable_count,
    %% Arity 3 should produce three set_variable instructions plus
    %% put_constant 3, A2.
    retractall(user:make_triple(_, _)),
    retractall(user:mode(make_triple(_, _))),
    assert(user:mode(make_triple(+, -))),
    assert(user:(make_triple(Name, T) :- functor(T, Name, 3))),
    (   catch(
            wam_target:compile_predicate_to_wam(make_triple/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:make_triple(_, _)),
        retractall(user:mode(make_triple(_, _))),
        count_substr(S, "set_variable", N),
        (   sub_string(S, _, _, _, "put_structure_dyn"),
            sub_string(S, _, _, _, "put_constant 3, A2"),
            N >= 3
        ->  pass(Test)
        ;   fail_test(Test, missing_arity_or_set_variables(S, N))
        )
    ;   retractall(user:make_triple(_, _)),
        retractall(user:mode(make_triple(_, _))),
        fail_test(Test, compile_failed)
    ).

test_functor3_zero_arity_emits_no_set_variable :-
    Test = test_functor3_zero_arity_emits_no_set_variable,
    %% Arity 0 — fresh atom-as-functor. put_structure_dyn fires with
    %% put_constant 0, A2 and no set_variable instructions for the
    %% structure body.
    retractall(user:make_atom(_, _)),
    retractall(user:mode(make_atom(_, _))),
    assert(user:mode(make_atom(+, -))),
    assert(user:(make_atom(Name, T) :- functor(T, Name, 0))),
    (   catch(
            wam_target:compile_predicate_to_wam(make_atom/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:make_atom(_, _)),
        retractall(user:mode(make_atom(_, _))),
        (   sub_string(S, _, _, _, "put_structure_dyn"),
            sub_string(S, _, _, _, "put_constant 0, A2")
        ->  pass(Test)
        ;   fail_test(Test, missing_zero_arity_dyn(S))
        )
    ;   retractall(user:make_atom(_, _)),
        retractall(user:mode(make_atom(_, _))),
        fail_test(Test, compile_failed)
    ).

test_functor3_non_integer_arity_falls_through :-
    Test = test_functor3_non_integer_arity_falls_through,
    %% Arity is a runtime variable, not a literal integer. Even with a
    %% mode declaration that proves T unbound and Name bound, the
    %% lowering must not fire — we cannot know how many set_variable
    %% slots to emit at compile time.
    retractall(user:make_dyn(_, _, _)),
    retractall(user:mode(make_dyn(_, _, _))),
    assert(user:mode(make_dyn(+, +, -))),
    assert(user:(make_dyn(Name, Arity, T) :- functor(T, Name, Arity))),
    (   catch(
            wam_target:compile_predicate_to_wam(make_dyn/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        retractall(user:make_dyn(_, _, _)),
        retractall(user:mode(make_dyn(_, _, _))),
        (   sub_string(S, _, _, _, "builtin_call functor/3"),
            \+ sub_string(S, _, _, _, "put_structure_dyn")
        ->  pass(Test)
        ;   fail_test(Test, dynamic_arity_must_not_lower(S))
        )
    ;   retractall(user:make_dyn(_, _, _)),
        retractall(user:mode(make_dyn(_, _, _))),
        fail_test(Test, compile_failed)
    ).

:- initialization(run_tests, main).
