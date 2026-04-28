:- encoding(utf8).
%% Test suite for the IntSet visited runtime: VSet Value variant +
%% BuildEmptySet, SetInsert, NotMemberSet step handlers.
%%
%% These are codegen tests against generated Haskell text — they
%% verify the templates emit the right ADT entries and step-handler
%% bodies. Runtime correctness on the generated artifact is exercised
%% by tests/core/test_wam_visited_set_lowering.pl (the codegen wiring
%% test) and the macro benchmark.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_wam_intset_runtime.pl

:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module(library(lists)).

run_tests :-
    format("~n========================================~n"),
    format("WAM IntSet visited runtime tests~n"),
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

test(test_vset_in_value_type).
test(test_data_intset_imported).
test(test_buildemptyset_in_instruction_adt).
test(test_setinsert_in_instruction_adt).
test(test_notmemberset_in_instruction_adt).
test(test_buildemptyset_step_handler).
test(test_setinsert_step_handler).
test(test_notmemberset_step_handler).
test(test_vset_nfdata_instance).
test(test_build_empty_set_wam_parse).
test(test_set_insert_wam_parse).
test(test_not_member_set_wam_parse).

%% ========================================================================
%% WamTypes.hs: Value variant + IntSet import + NFData
%% ========================================================================

test_vset_in_value_type :-
    Test = test_vset_in_value_type,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "VSet !IS.IntSet")
    ->  pass(Test)
    ;   fail_test(Test, 'VSet variant missing from Value type')
    ).

test_data_intset_imported :-
    Test = test_data_intset_imported,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "import qualified Data.IntSet as IS")
    ->  pass(Test)
    ;   fail_test(Test, 'Data.IntSet not imported in WamTypes')
    ).

test_vset_nfdata_instance :-
    Test = test_vset_nfdata_instance,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "rnf (VSet")
    ->  pass(Test)
    ;   fail_test(Test, 'NFData VSet branch missing')
    ).

%% ========================================================================
%% Instruction ADT: three new constructors
%% ========================================================================

test_buildemptyset_in_instruction_adt :-
    Test = test_buildemptyset_in_instruction_adt,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "BuildEmptySet !RegId")
    ->  pass(Test)
    ;   fail_test(Test, 'BuildEmptySet missing from Instruction ADT')
    ).

test_setinsert_in_instruction_adt :-
    Test = test_setinsert_in_instruction_adt,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "SetInsert !RegId !RegId !RegId")
    ->  pass(Test)
    ;   fail_test(Test, 'SetInsert missing from Instruction ADT')
    ).

test_notmemberset_in_instruction_adt :-
    Test = test_notmemberset_in_instruction_adt,
    (   wam_haskell_target:generate_wam_types_hs(Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "NotMemberSet !RegId !RegId")
    ->  pass(Test)
    ;   fail_test(Test, 'NotMemberSet missing from Instruction ADT')
    ).

%% ========================================================================
%% WamRuntime.hs: step handlers
%% ========================================================================

test_buildemptyset_step_handler :-
    Test = test_buildemptyset_step_handler,
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !_ctx s (BuildEmptySet r)"),
        sub_string(S, _, _, _, "VSet IS.empty")
    ->  pass(Test)
    ;   fail_test(Test, 'BuildEmptySet handler missing or wrong')
    ).

test_setinsert_step_handler :-
    Test = test_setinsert_step_handler,
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !_ctx s (SetInsert eReg inReg outReg)"),
        sub_string(S, _, _, _, "IS.insert aid s0")
    ->  pass(Test)
    ;   fail_test(Test, 'SetInsert handler missing or wrong')
    ).

test_notmemberset_step_handler :-
    Test = test_notmemberset_step_handler,
    (   compile_wam_runtime_to_haskell([], [], Code),
        atom_string(Code, S),
        sub_string(S, _, _, _, "step !_ctx s (NotMemberSet eReg setReg)"),
        sub_string(S, _, _, _, "IS.member aid s0")
    ->  pass(Test)
    ;   fail_test(Test, 'NotMemberSet handler missing or wrong')
    ).

%% ========================================================================
%% WAM text parser entries
%% ========================================================================

test_build_empty_set_wam_parse :-
    Test = test_build_empty_set_wam_parse,
    (   wam_haskell_target:wam_instr_to_haskell(["build_empty_set", "X4"], Hs),
        sub_string(Hs, _, _, _, "BuildEmptySet 104")
    ->  pass(Test)
    ;   fail_test(Test, 'build_empty_set WAM parse failed')
    ).

test_set_insert_wam_parse :-
    Test = test_set_insert_wam_parse,
    (   wam_haskell_target:wam_instr_to_haskell(["set_insert", "X1,", "X2,", "X3"], Hs),
        sub_string(Hs, _, _, _, "SetInsert 101 102 103")
    ->  pass(Test)
    ;   fail_test(Test, 'set_insert WAM parse failed')
    ).

test_not_member_set_wam_parse :-
    Test = test_not_member_set_wam_parse,
    (   wam_haskell_target:wam_instr_to_haskell(["not_member_set", "X5,", "X6"], Hs),
        sub_string(Hs, _, _, _, "NotMemberSet 105 106")
    ->  pass(Test)
    ;   fail_test(Test, 'not_member_set WAM parse failed')
    ).

:- initialization(run_tests, main).
