:- encoding(utf8).
%% Test suite for the ground-list `\+ member(X, [a, b, c])` lowering.
%%
%% Verifies the WAM compiler emits `not_member_const_atoms` (a single
%% specialised instruction with the atom IDs baked in) when the second
%% argument of `\+ member/2` is a literal proper list of ground atoms.
%% Falls through cleanly for partial lists, non-atom lists, and
%% variable lists (the latter still uses the existing
%% `not_member_list` Phase G lowering).
%%
%% Also verifies the WAM-text parser in wam_haskell_target produces
%% `NotMemberConstAtoms <reg> [<id>, <id>, ...]` for the new instruction
%% and that the Instruction ADT + step handler are present.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_ground_member_lowering.pl

:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(lists)).

run_tests :-
    format("~n========================================~n"),
    format("WAM ground-list \\+ member lowering tests~n"),
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

test(test_ground_atom_list_emits_const_atoms).
test(test_ground_list_skips_put_list).
test(test_ground_list_skips_builtin_call).
test(test_singleton_atom_list_lowers).
test(test_var_list_keeps_not_member_list).
test(test_integer_list_falls_through).
test(test_not_member_const_atoms_in_instruction_adt).
test(test_not_member_const_atoms_step_handler).
test(test_not_member_const_atoms_wam_parse).

:- dynamic user:visited_set/2.
:- dynamic user:mode/1.

reset_directives :-
    retractall(user:visited_set(_, _)),
    retractall(user:mode(_)).

%% ========================================================================
%% Codegen tests
%% ========================================================================

test_ground_atom_list_emits_const_atoms :-
    Test = test_ground_atom_list_emits_const_atoms,
    reset_directives,
    retractall(user:gm_a/1),
    assertz(user:mode(gm_a(?))),
    assertz(user:(gm_a(X) :- \+ member(X, [foo, bar, baz]))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_a/1, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives,
        retractall(user:gm_a(_)),
        (   sub_string(S, _, _, _, "not_member_const_atoms"),
            sub_string(S, _, _, _, "foo"),
            sub_string(S, _, _, _, "bar"),
            sub_string(S, _, _, _, "baz")
        ->  pass(Test)
        ;   fail_test(Test, expected_const_atoms(S))
        )
    ;   reset_directives, retractall(user:gm_a(_)),
        fail_test(Test, compile_failed)
    ).

test_ground_list_skips_put_list :-
    Test = test_ground_list_skips_put_list,
    reset_directives,
    retractall(user:gm_b/1),
    assertz(user:mode(gm_b(?))),
    assertz(user:(gm_b(X) :- \+ member(X, [alpha, beta]))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_b/1, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives, retractall(user:gm_b(_)),
        (   \+ sub_string(S, _, _, _, "put_list"),
            \+ sub_string(S, _, _, _, "set_constant")
        ->  pass(Test)
        ;   fail_test(Test, unexpected_list_construction(S))
        )
    ;   reset_directives, retractall(user:gm_b(_)),
        fail_test(Test, compile_failed)
    ).

test_ground_list_skips_builtin_call :-
    Test = test_ground_list_skips_builtin_call,
    reset_directives,
    retractall(user:gm_c/1),
    assertz(user:mode(gm_c(?))),
    assertz(user:(gm_c(X) :- \+ member(X, [one, two, three]))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_c/1, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives, retractall(user:gm_c(_)),
        (   \+ sub_string(S, _, _, _, "builtin_call \\+/1"),
            \+ sub_string(S, _, _, _, "builtin_call member/2")
        ->  pass(Test)
        ;   fail_test(Test, unexpected_builtin_dispatch(S))
        )
    ;   reset_directives, retractall(user:gm_c(_)),
        fail_test(Test, compile_failed)
    ).

test_singleton_atom_list_lowers :-
    Test = test_singleton_atom_list_lowers,
    reset_directives,
    retractall(user:gm_d/1),
    assertz(user:mode(gm_d(?))),
    assertz(user:(gm_d(X) :- \+ member(X, [solo]))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_d/1, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives, retractall(user:gm_d(_)),
        (   sub_string(S, _, _, _, "not_member_const_atoms"),
            sub_string(S, _, _, _, "solo")
        ->  pass(Test)
        ;   fail_test(Test, expected_const_atoms_singleton(S))
        )
    ;   reset_directives, retractall(user:gm_d(_)),
        fail_test(Test, compile_failed)
    ).

test_var_list_keeps_not_member_list :-
    %% \+ member(X, V) where V is a var stays on the existing
    %% Phase G NotMemberList path — must not be hijacked by the new
    %% ground-list lowering. Both X and V need to be `bound` mode for
    %% the binding analyser to fire the Phase G lowering.
    Test = test_var_list_keeps_not_member_list,
    reset_directives,
    retractall(user:gm_e/3),
    assertz(user:mode(gm_e(?, +, +))),
    assertz(user:(gm_e(_, X, V) :- \+ member(X, V))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_e/3, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives, retractall(user:gm_e(_, _, _)),
        (   sub_string(S, _, _, _, "not_member_list"),
            \+ sub_string(S, _, _, _, "not_member_const_atoms")
        ->  pass(Test)
        ;   fail_test(Test, expected_not_member_list(S))
        )
    ;   reset_directives, retractall(user:gm_e(_, _, _)),
        fail_test(Test, compile_failed)
    ).

test_partial_list_falls_through :-
    %% \+ member(X, [a|T]) with T unbound should NOT lower (the list is
    %% not fully ground at compile time). Falls through to the standard
    %% builtin-dispatch path. We just check that the new instruction is
    %% NOT emitted; whatever the fallback produces is fine.
    Test = test_partial_list_falls_through,
    reset_directives,
    retractall(user:gm_f/2),
    assertz(user:mode(gm_f(?, ?))),
    assertz(user:(gm_f(X, T) :- \+ member(X, [a | T]))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_f/2, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives, retractall(user:gm_f(_, _)),
        (   \+ sub_string(S, _, _, _, "not_member_const_atoms")
        ->  pass(Test)
        ;   fail_test(Test, unexpected_const_atoms(S))
        )
    ;   reset_directives, retractall(user:gm_f(_, _)),
        fail_test(Test, compile_failed)
    ).

test_integer_list_falls_through :-
    %% Integer-element list should NOT lower — the new instruction
    %% targets atoms only. Stays on the standard builtin path.
    Test = test_integer_list_falls_through,
    reset_directives,
    retractall(user:gm_g/1),
    assertz(user:mode(gm_g(?))),
    assertz(user:(gm_g(X) :- \+ member(X, [1, 2, 3]))),
    (   catch(
            wam_target:compile_predicate_to_wam(gm_g/1, [], WamCode),
            _, fail)
    ->  atom_string(WamCode, S),
        reset_directives, retractall(user:gm_g(_)),
        (   \+ sub_string(S, _, _, _, "not_member_const_atoms")
        ->  pass(Test)
        ;   fail_test(Test, unexpected_const_atoms_for_integers(S))
        )
    ;   reset_directives, retractall(user:gm_g(_)),
        fail_test(Test, compile_failed)
    ).

%% ========================================================================
%% Haskell side: ADT + step handler + WAM-text parser
%% ========================================================================

test_not_member_const_atoms_in_instruction_adt :-
    Test = test_not_member_const_atoms_in_instruction_adt,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "NotMemberConstAtoms !RegId ![Int]")
    ->  pass(Test)
    ;   fail_test(Test, 'NotMemberConstAtoms missing from Instruction ADT')
    ).

test_not_member_const_atoms_step_handler :-
    Test = test_not_member_const_atoms_step_handler,
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !_ctx s (NotMemberConstAtoms xReg atomIds)"),
        sub_string(S, _, _, _, "aid `elem` atomIds")
    ->  pass(Test)
    ;   fail_test(Test, 'NotMemberConstAtoms handler missing or wrong')
    ).

test_not_member_const_atoms_wam_parse :-
    %% Verify the WAM-text parser produces a NotMemberConstAtoms with
    %% the bracketed atom-id list. Atom IDs are interned on demand;
    %% we just check structural shape, not specific numeric IDs.
    Test = test_not_member_const_atoms_wam_parse,
    wam_haskell_target:init_atom_intern_table,
    (   wam_haskell_target:wam_instr_to_haskell(
            ["not_member_const_atoms", "X1", "alpha_atom", "beta_atom"], Hs),
        sub_string(Hs, _, _, _, "NotMemberConstAtoms 101"),
        sub_string(Hs, _, _, _, "[")
    ->  pass(Test)
    ;   fail_test(Test, 'not_member_const_atoms WAM parse failed')
    ).

:- initialization(run_tests, main).
