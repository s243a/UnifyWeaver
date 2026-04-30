:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Phase 3 emitter tests for the WAM-lowered Go path.
%
% Phase 3 exercises the lowered emitter directly with synthesized
% instruction lists so the tests don't depend on the full WAM
% compilation pipeline. The two behaviors covered:
%
%   1. Atom interning is applied to atom literals in lowered output —
%      every atom resolves to a wamAtom_* identifier, never a raw
%      &Atom{Name: "..."} struct literal in get_constant / put_constant
%      emission paths.
%
%   2. The if-then-else WAM pattern (try_me_else / cut_ite / jump /
%      trust_me) is detected by has_internal_ite_pattern/1 and lowered
%      to native Go if/else with a closure-wrapped condition and a
%      trail-mark-based unwind on failure.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl

:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module('../src/unifyweaver/targets/wam_go_lowered_emitter').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Helper: lower a synthetic predicate body and return the emitted Go code.
emit_to_string(PI, Instrs, Code) :-
    init_atom_intern_table_go,
    lower_predicate_to_go(PI, Instrs, [], GoLines),
    atomic_list_concat(GoLines, '\n', Code).

%% ---------------------------------------------------------------------
%% Atom interning at the emitter level
%% ---------------------------------------------------------------------

test_get_constant_uses_interned_var :-
    Test = 'WAM-Go-Lowered Phase 3: get_constant emits an interned wamAtom_* reference',
    Instrs = [get_constant("foo", "A1"), proceed],
    emit_to_string(p/1, Instrs, Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "wamAtom_foo_"),
        % And the raw struct literal must NOT appear (would be a regression).
        \+ sub_string(S, _, _, _, "&Atom{Name: \"foo\"}")
    ->  pass(Test)
    ;   fail_test(Test, ['expected wamAtom_foo_* reference; got: ', S])
    ).

test_put_constant_uses_interned_var :-
    Test = 'WAM-Go-Lowered Phase 3: put_constant emits an interned wamAtom_* reference',
    Instrs = [put_constant("hello", "A1"), proceed],
    emit_to_string(p/1, Instrs, Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "wamAtom_hello_"),
        \+ sub_string(S, _, _, _, "&Atom{Name: \"hello\"}")
    ->  pass(Test)
    ;   fail_test(Test, ['expected wamAtom_hello_* reference; got: ', S])
    ).

test_repeated_atoms_dedup_in_emission :-
    Test = 'WAM-Go-Lowered Phase 3: identical atom literals share one Go var',
    Instrs = [
        get_constant("alpha", "A1"),
        get_constant("alpha", "A2"),
        proceed
    ],
    emit_to_string(p/2, Instrs, Code),
    atom_string(Code, S),
    % Both uses must collapse to the same wamAtom_alpha_<seq> name.
    % If dedup were broken we'd see two distinct seqs (0 and 1).
    (   sub_string(S, _, _, _, "wamAtom_alpha_0"),
        \+ sub_string(S, _, _, _, "wamAtom_alpha_1")
    ->  pass(Test)
    ;   fail_test(Test, ['dedup broken; saw alpha_0 + alpha_1 in: ', S])
    ).

test_distinct_atoms_get_distinct_vars :-
    Test = 'WAM-Go-Lowered Phase 3: distinct atom literals get distinct Go vars',
    Instrs = [
        get_constant("first",  "A1"),
        get_constant("second", "A2"),
        proceed
    ],
    emit_to_string(p/2, Instrs, Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "wamAtom_first_"),
        sub_string(S, _, _, _, "wamAtom_second_")
    ->  pass(Test)
    ;   fail_test(Test, ['expected both first and second wamAtom_* vars in: ', S])
    ).

test_get_nil_uses_interned_nil :-
    Test = 'WAM-Go-Lowered Phase 3: get_nil emits an interned [] reference',
    Instrs = [get_nil("A1"), proceed],
    emit_to_string(p/1, Instrs, Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "wamAtom_"),
        \+ sub_string(S, _, _, _, "&Atom{Name: \"[]\"}")
    ->  pass(Test)
    ;   fail_test(Test, ['expected interned [] var; got: ', S])
    ).

%% ---------------------------------------------------------------------
%% If-then-else pattern detection and lowering
%% ---------------------------------------------------------------------

test_has_internal_ite_pattern_detects :-
    Test = 'WAM-Go-Lowered Phase 3: has_internal_ite_pattern recognizes a complete ITE',
    Instrs = [
        try_me_else("L_else"),
        get_constant("c", "A1"),
        cut_ite,
        put_constant("then_val", "A2"),
        jump("L_cont"),
        trust_me,
        put_constant("else_val", "A2"),
        proceed
    ],
    (   has_internal_ite_pattern(Instrs)
    ->  pass(Test)
    ;   fail_test(Test, 'pattern not recognized')
    ).

test_has_internal_ite_pattern_rejects_partial :-
    Test = 'WAM-Go-Lowered Phase 3: has_internal_ite_pattern rejects an incomplete ITE',
    % Missing trust_me — must NOT match.
    Instrs = [
        try_me_else("L_else"),
        get_constant("c", "A1"),
        cut_ite,
        put_constant("then_val", "A2"),
        jump("L_cont"),
        proceed
    ],
    (   \+ has_internal_ite_pattern(Instrs)
    ->  pass(Test)
    ;   fail_test(Test, 'incomplete ITE was accepted')
    ).

test_ite_emits_native_if_else :-
    Test = 'WAM-Go-Lowered Phase 3: ITE pattern emits native Go if/else with trail unwind',
    Instrs = [
        try_me_else("L_else"),
        get_constant("c", "A1"),
        cut_ite,
        put_constant("then_val", "A2"),
        jump("L_cont"),
        trust_me,
        put_constant("else_val", "A2"),
        proceed
    ],
    emit_to_string(p/2, Instrs, Code),
    atom_string(Code, S),
    (   sub_string(S, _, _, _, "if-then-else"),
        sub_string(S, _, _, _, "_trailMark := vm.TrailLen"),
        sub_string(S, _, _, _, "_condOk := func() bool {"),
        sub_string(S, _, _, _, "if _condOk {"),
        sub_string(S, _, _, _, "} else {"),
        sub_string(S, _, _, _, "vm.unwindTrailTo(_trailMark)"),
        % And the cond/then/else atoms all appear via interning.
        sub_string(S, _, _, _, "wamAtom_c_"),
        sub_string(S, _, _, _, "wamAtom_then_val_"),
        sub_string(S, _, _, _, "wamAtom_else_val_")
    ->  pass(Test)
    ;   fail_test(Test, ['ITE lowering missing expected pieces: ', S])
    ).

test_ite_does_not_silently_drop_choice_instrs :-
    Test = 'WAM-Go-Lowered Phase 3: ITE pattern absorbs cut_ite/jump/trust_me into the branching block',
    Instrs = [
        try_me_else("L_else"),
        get_constant("guard", "A1"),
        cut_ite,
        put_constant("then_val", "A2"),
        jump("L_cont"),
        trust_me,
        put_constant("else_val", "A2"),
        proceed
    ],
    emit_to_string(p/2, Instrs, Code),
    atom_string(Code, S),
    % The bare "// cut_ite", "// jump", "// trust_me" comments from the
    % silent-fallback emit_one clauses must NOT appear when the ITE
    % pattern fires — those would indicate the choice-point instructions
    % leaked through to the linear emitter.
    (   \+ sub_string(S, _, _, _, "// cut_ite"),
        \+ sub_string(S, _, _, _, "// jump"),
        \+ sub_string(S, _, _, _, "// trust_me")
    ->  pass(Test)
    ;   fail_test(Test, ['choice-point instr comments leaked through ITE: ', S])
    ).

%% ---------------------------------------------------------------------
%% Runner
%% ---------------------------------------------------------------------

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Go-Lowered Phase 3 tests ===~n', []),
    test_get_constant_uses_interned_var,
    test_put_constant_uses_interned_var,
    test_repeated_atoms_dedup_in_emission,
    test_distinct_atoms_get_distinct_vars,
    test_get_nil_uses_interned_nil,
    test_has_internal_ite_pattern_detects,
    test_has_internal_ite_pattern_rejects_partial,
    test_ite_emits_native_if_else,
    test_ite_does_not_silently_drop_choice_instrs,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All Phase 3 tests passed ===~n', [])
    ).
